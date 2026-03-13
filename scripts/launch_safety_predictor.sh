#!/bin/bash
# Launch the CNN safety predictor pipeline on a running Vast.ai instance.
#
# Prerequisites:
#   1. Vast.ai instance with nvidia/vulkan:1.3-470 image, running
#   2. setup_nvvulkan.sh already applied
#   3. InternVLA-M1 server started on port 10093
#
# Usage:
#   bash scripts/launch_safety_predictor.sh <SSH_PORT> <SSH_HOST>

set -e

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_HOST="${2:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_CMD="ssh -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no"

echo "=== Syncing project files to remote ==="
rsync -az --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    /Users/julianquick/TRI_pet_project/adversarial_dust/ \
    root@$SSH_HOST:/root/project/adversarial_dust/

rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    /Users/julianquick/TRI_pet_project/configs/safety_predictor.yaml \
    root@$SSH_HOST:/root/project/configs/

# Upload prior sweep results for adversarial params
echo "=== Uploading prior sweep results ==="
rsync -az --include='envelope_results.json' --exclude='*' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    /Users/julianquick/TRI_pet_project/results/coke_can_fingerprint/ \
    root@$SSH_HOST:/root/project/results/coke_can_fingerprint/

echo "=== Checking InternVLA-M1 server status ==="
$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 && echo 'Server running' || echo 'Server NOT running - starting...'"

$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 || \
    (export PATH=/opt/miniconda/bin:\$PATH && \
     cd /root/InternVLA-M1 && \
     PYTHONPATH=/root/InternVLA-M1:\$PYTHONPATH \
     nohup python deployment/model_server/server_policy_M1.py \
       --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
       --port 10093 --use_bf16 > /tmp/internvla_server.log 2>&1 &)"

echo "=== Waiting for InternVLA-M1 server to initialize ==="
sleep 45

echo "=== Launching safety predictor pipeline ==="
$SSH_CMD "cat > /root/run_predictor.sh << 'SCRIPT'
#!/bin/bash
set -e
export PATH=/opt/miniconda/bin:\$PATH
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:\$PYTHONPATH
cd /root/project

python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml \
    --output-dir results/safety_predictor \
    --episodes-per-condition 5 \
    --adversarial-episodes 3 \
    --frame-stride 10 \
    --epochs 50 \
    --eval-episodes 10 \
    --device cuda \
    --adv-results fingerprint:results/coke_can_fingerprint

echo 'Pipeline complete!'
tar czf /tmp/safety_predictor_results.tar.gz -C /root/project results/safety_predictor/
echo 'Results archived at /tmp/safety_predictor_results.tar.gz'
SCRIPT
chmod +x /root/run_predictor.sh"

$SSH_CMD "tmux new-session -d -s predictor 'bash /root/run_predictor.sh 2>&1 | tee /root/predictor.log'"

echo ""
echo "=== Pipeline launched in tmux session 'predictor' ==="
echo ""
echo "Monitor:"
echo "  $SSH_CMD 'tail -f /root/predictor.log'"
echo ""
echo "Download results:"
echo "  scp -P $SSH_PORT root@$SSH_HOST:/tmp/safety_predictor_results.tar.gz ."
