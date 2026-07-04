#!/bin/bash
# Remote driver + self-destruct for the frozen-ResNet LOO run.
# Guarantees no idle billing WITHOUT depending on the laptop:
#   - 30h hard walltime cap (covers setup/job hangs)
#   - self-destruct 45 min after the job ends (grace for local result pull)
# Uses the instance-scoped key in /root/.instance_key (controls only this box).
exec > /root/run_frozen.log 2>&1
set -x
CID=$(cat /root/.instance_id 2>/dev/null)
KEYF=/root/.instance_key

# --- bootstrap miniconda + vastai EARLY so the walltime guard works from t0 ---
if [ ! -d /opt/miniconda ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p /opt/miniconda && rm /tmp/mc.sh
fi
export PATH="/opt/miniconda/bin:$PATH"
pip install -q vastai 2>&1 | tail -1
VASTAI="/opt/miniconda/bin/vastai"

selfdestruct() { for k in 1 2 3; do "$VASTAI" destroy instance "$CID" --api-key "$(cat $KEYF)" && return 0; sleep 30; done; }

# 30h hard cap — laptop-independent runaway guard (covers every hang path)
( sleep 108000; echo "WALLTIME_CAP_HIT"; selfdestruct ) &

finish() { # $1 = sentinel file
    tar czf /root/loo_results.tar.gz -C /root/project results/loo_analysis_frozen 2>/dev/null || true
    touch "$1"
    echo "SENTINEL=$1 ; sleeping 45m for local result pull, then self-destruct"
    sleep 2700
    selfdestruct
}

# 1) verified environment setup (clones repos incl. project, downloads ckpt)
bash /root/setup_nvvulkan.sh || { echo SETUP_FAILED; finish /root/FAILED; exit 1; }

export VK_ICD_FILENAMES=$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:$PYTHONPATH

# 2) InternVLA-M1 policy server (background)
cd /root/InternVLA-M1
PYTHONPATH=/root/InternVLA-M1 nohup python deployment/model_server/server_policy_M1.py \
    --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
    --port 10093 --use_bf16 > /root/server.log 2>&1 &

# 3) wait for server port (up to 15 min)
ready=0
for i in $(seq 1 90); do
    python -c "import socket;socket.create_connection(('localhost',10093),2)" 2>/dev/null && { ready=1; break; }
    sleep 10
done
[ "$ready" = 1 ] || { echo SERVER_NOT_READY; tail -50 /root/server.log; finish /root/FAILED; exit 1; }
echo "SERVER_READY"

# 4) frozen-ResNet full LOO (9 folds)
cd /root/project
python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml --loo \
    --loo-types fingerprint glare rain gaussian_noise jpeg motion_blur defocus_blur dust_camera low_light \
    --eval-episodes 10 --episodes-per-condition 10 --budget-levels 0.1 0.3 0.5 0.7 0.9 \
    --frame-stride 10 --epochs 50 --device cuda \
    --backbone resnet18 --freeze-backbone \
    --output-dir results/loo_analysis_frozen
rc=$?

# 5) archive + sentinel + self-destruct
[ $rc -eq 0 ] && finish /root/DONE || finish /root/FAILED
