#!/bin/bash
# Launch the coke-can safe-operating-envelope run on a Vast.ai instance.
#
# Mirrors scripts/launch_loo.sh. Runs the envelope study on the same task
# used for the LOO safety-monitor experiment, so envelope + LOO are reported
# on a single task in the paper.
#
# Usage:
#   bash scripts/launch_envelope_cokecan.sh <SSH_PORT> <SSH_HOST>

set -e

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_HOST="${2:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_CMD="ssh -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no"
PROJECT_LOCAL="/Users/julianquick/portfolio_copy/TRI_pet_project"

echo "=== Syncing project files ==="
rsync -az --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    --exclude='results' --exclude='*.tar.gz' --exclude='docs/figures/*.gif' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$PROJECT_LOCAL/adversarial_dust/" \
    root@$SSH_HOST:/root/project/adversarial_dust/

rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$PROJECT_LOCAL/configs/" \
    root@$SSH_HOST:/root/project/configs/

echo "=== Checking InternVLA-M1 server ==="
$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 || \
    (export PATH=/opt/miniconda/bin:\$PATH && \
     cd /root/InternVLA-M1 && \
     PYTHONPATH=/root/InternVLA-M1:\$PYTHONPATH \
     nohup python deployment/model_server/server_policy_M1.py \
       --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
       --port 10093 --use_bf16 > /tmp/internvla_server.log 2>&1 &)"
sleep 45

echo "=== Deploying envelope run script ==="
$SSH_CMD "cat > /root/run_envelope_cokecan.sh << 'SCRIPT'
#!/bin/bash
set -e
export PATH=/opt/miniconda/bin:\$PATH
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:\$PYTHONPATH
cd /root/project

python -u -m adversarial_dust.run_envelope \
    --config configs/envelope_coke_can_internvla.yaml

tar czf /tmp/envelope_cokecan.tar.gz -C /root/project results/envelope_coke_can_internvla/
echo 'Archived: /tmp/envelope_cokecan.tar.gz'
ls -lh /tmp/envelope_cokecan.tar.gz
SCRIPT
chmod +x /root/run_envelope_cokecan.sh"

$SSH_CMD "tmux new-session -d -s envelope 'bash /root/run_envelope_cokecan.sh 2>&1 | tee /root/envelope_cokecan.log'"

echo ""
echo "Launched in tmux session 'envelope'."
echo "Monitor: $SSH_CMD 'tail -f /root/envelope_cokecan.log'"
echo "Download: scp -P $SSH_PORT root@$SSH_HOST:/tmp/envelope_cokecan.tar.gz ."
echo "Estimated runtime: ~10-15 hours for 9 corruption types x 5 budgets."
