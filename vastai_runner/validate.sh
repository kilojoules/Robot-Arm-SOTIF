#!/bin/bash
# Fast end-to-end validation on the diag box (setup already done):
# start InternVLA-M1 server -> tiny 2-type/1-episode LOO smoke (incl. rain,
# which exercises camera_occlusion + headless Vulkan rendering + policy server).
exec > /root/validate.log 2>&1
set -x
export PATH=/opt/miniconda/bin:$PATH
export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:$PYTHONPATH

cd /root/InternVLA-M1
PYTHONPATH=/root/InternVLA-M1 nohup python deployment/model_server/server_policy_M1.py \
    --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
    --port 10093 --use_bf16 > /root/server.log 2>&1 &

for i in $(seq 1 90); do
    python -c "import socket;socket.create_connection(('localhost',10093),2)" 2>/dev/null && break
    sleep 10
done
python -c "import socket;socket.create_connection(('localhost',10093),2)" 2>/dev/null \
    || { echo "SERVER_FAIL"; tail -50 /root/server.log; touch /root/VALIDATE_FAIL; exit 1; }
echo "SERVER_OK"

cd /root/project
timeout 1500 python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml --loo \
    --loo-types rain fingerprint \
    --eval-episodes 1 --episodes-per-condition 1 --budget-levels 0.5 \
    --frame-stride 10 --epochs 2 --device cuda \
    --backbone resnet18 --freeze-backbone \
    --output-dir results/validate_smoke
rc=$?
if [ $rc -eq 0 ]; then echo "SMOKE_OK"; touch /root/VALIDATE_OK; else echo "SMOKE_FAIL rc=$rc"; touch /root/VALIDATE_FAIL; fi
