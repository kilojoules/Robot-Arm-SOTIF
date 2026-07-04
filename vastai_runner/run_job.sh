#!/bin/bash
# Full frozen-ResNet LOO on the already-set-up box.
# Bakes in the sdpa fix (InternVLA-M1 hardcodes flash_attention_2, but flash-attn
# isn't installed; sdpa is exact + needs no extra package). Drops DONE/FAILED.
exec > /root/run_frozen.log 2>&1
set -x
export PATH=/opt/miniconda/bin:$PATH
export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:$PYTHONPATH

# sdpa fix (idempotent) + clear stale bytecode so the patched source is used
sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/' \
    /root/InternVLA-M1/InternVLA/model/modules/vlm/QWen2_5.py
find /root/InternVLA-M1 -name '*.pyc' -delete 2>/dev/null || true

# policy server
cd /root/InternVLA-M1
PYTHONPATH=/root/InternVLA-M1 nohup python deployment/model_server/server_policy_M1.py \
    --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
    --port 10093 --use_bf16 > /root/server.log 2>&1 &
ready=0
for i in $(seq 1 120); do
    python -c "import socket;socket.create_connection(('localhost',10093),2)" 2>/dev/null && { ready=1; break; }
    sleep 10
done
[ "$ready" = 1 ] || { echo SERVER_NOT_READY; tail -60 /root/server.log; touch /root/FAILED; exit 1; }
echo SERVER_READY

# full frozen LOO
cd /root/project
python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml --loo \
    --loo-types fingerprint glare rain gaussian_noise jpeg motion_blur defocus_blur dust_camera low_light \
    --eval-episodes 10 --episodes-per-condition 10 --budget-levels 0.1 0.3 0.5 0.7 0.9 \
    --frame-stride 10 --epochs 50 --device cuda \
    --backbone resnet18 --freeze-backbone \
    --output-dir results/loo_analysis_frozen
rc=$?
tar czf /root/loo_results.tar.gz -C /root/project results/loo_analysis_frozen 2>/dev/null || true
[ $rc -eq 0 ] && touch /root/DONE || touch /root/FAILED
