#!/bin/bash
# Frozen + fine-tune ResNet LOO on the slim (working) env.
# Frozen collects the shared loo_data; fine-tune reuses it (--skip-collect),
# so both monitors train on the SAME episodes (matched comparison) and we
# avoid a second ~20h collection. Drops /root/DONE | /root/FAILED.
exec > /root/run_both.log 2>&1
set -x
export PATH=/opt/miniconda/bin:$PATH
export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:$PYTHONPATH
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
    if python -c "import socket;socket.create_connection(('localhost',10093),2)" 2>/dev/null; then ready=1; break; fi
    sleep 10
done
if [ "$ready" != 1 ]; then echo SERVER_NOT_READY; tail -60 /root/server.log; touch /root/FAILED; exit 1; fi
echo SERVER_READY

cd /root/project
COMMON="--config configs/safety_predictor.yaml --loo --loo-types fingerprint glare rain gaussian_noise jpeg motion_blur defocus_blur dust_camera low_light --eval-episodes 10 --episodes-per-condition 10 --budget-levels 0.1 0.3 0.5 0.7 0.9 --frame-stride 10 --epochs 50 --device cuda --backbone resnet18"

# --- FROZEN (collects shared loo_data) ---
python -u adversarial_dust/run_safety_predictor.py $COMMON --freeze-backbone --output-dir results/loo_analysis_frozen
rcf=$?
tar czf /root/loo_frozen.tar.gz -C /root/project results/loo_analysis_frozen 2>/dev/null || true
if [ $rcf -ne 0 ]; then echo "FROZEN_FAILED rc=$rcf"; touch /root/FAILED; exit 1; fi
echo "FROZEN_DONE"

# --- FINE-TUNE (reuse collected loo_data) ---
mkdir -p results/loo_analysis_finetune/loo_data
cp -f results/loo_analysis_frozen/loo_data/*.npz results/loo_analysis_finetune/loo_data/ 2>/dev/null || true
python -u adversarial_dust/run_safety_predictor.py $COMMON --no-freeze-backbone --skip-collect --output-dir results/loo_analysis_finetune
rcn=$?
tar czf /root/loo_results.tar.gz -C /root/project results/loo_analysis_frozen results/loo_analysis_finetune 2>/dev/null || true
[ $rcn -eq 0 ] && echo "BOTH_DONE" || echo "FINETUNE_FAILED rc=$rcn (frozen results preserved)"
touch /root/DONE
