#!/bin/bash
# Launch the CNN safety predictor pipeline on a running Vast.ai instance.
#
# Prerequisites:
#   1. Vast.ai instance with nvidia/vulkan:1.3-470 image, running
#   2. setup_nvvulkan2.sh already applied (with all post-setup fixes)
#   3. InternVLA-M1 server started on port 10093
#
# Usage:
#   bash scripts/launch_safety_predictor.sh <SSH_PORT> <SSH_HOST>
#
# Example:
#   bash scripts/launch_safety_predictor.sh 12345 ssh5.vast.ai

set -e

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_HOST="${2:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_CMD="ssh -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no"

echo "=== Syncing project files to remote ==="
rsync -az --exclude='__pycache__' --exclude='results' --exclude='.git' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    /Users/julianquick/TRI_pet_project/adversarial_dust/ \
    root@$SSH_HOST:/root/project/adversarial_dust/

rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    /Users/julianquick/TRI_pet_project/configs/safety_predictor.yaml \
    root@$SSH_HOST:/root/project/configs/

echo "=== Checking InternVLA-M1 server status ==="
$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 && echo 'Server running' || echo 'Server NOT running - starting...'"

# Start server if not running
$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 || \
    (cd /root/InternVLA-M1 && \
     PYTHONPATH=/root/InternVLA-M1:\$PYTHONPATH \
     nohup python deployment/model_server/server_policy_M1.py \
       --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
       --port 10093 --use_bf16 > /tmp/internvla_server.log 2>&1 &)"

echo "=== Waiting for InternVLA-M1 server to initialize ==="
sleep 30

echo "=== Launching safety predictor pipeline ==="
# Create the run script on remote
$SSH_CMD "cat > /root/run_predictor.sh << 'SCRIPT'
#!/bin/bash
set -e
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:\$PYTHONPATH
cd /root/project

echo 'Starting safety predictor pipeline...'
python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml \
    --output-dir results/safety_predictor \
    --episodes-per-condition 5 \
    --frame-stride 10 \
    --epochs 30 \
    --rain-episodes 10 \
    --device cuda

echo 'Pipeline complete!'
echo 'Archiving results...'
tar czf /tmp/safety_predictor_results.tar.gz -C /root/project results/safety_predictor/
echo 'Results archived at /tmp/safety_predictor_results.tar.gz'
SCRIPT
chmod +x /root/run_predictor.sh"

# Launch in tmux so we can detach
$SSH_CMD "tmux new-session -d -s predictor 'bash /root/run_predictor.sh 2>&1 | tee /root/predictor.log'"

echo ""
echo "=== Pipeline launched in tmux session 'predictor' ==="
echo ""
echo "Monitor progress:"
echo "  $SSH_CMD 'tmux attach -t predictor'"
echo "  $SSH_CMD 'tail -f /root/predictor.log'"
echo ""
echo "Download results when done:"
echo "  scp -P $SSH_PORT root@$SSH_HOST:/tmp/safety_predictor_results.tar.gz ."
