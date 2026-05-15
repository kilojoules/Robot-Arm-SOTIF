#!/bin/bash
# Corruption diversity curve launcher.
#
# Assumes scripts/launch_loo.sh has already been run and the shared LOO
# dataset exists on the remote at /root/project/results/loo_analysis/loo_data/.
# Runs --diversity-curve with 5 random subsets per size in {2,4,6,8}.
#
# Usage:
#   bash scripts/launch_diversity_curve.sh <SSH_PORT> <SSH_HOST>

set -e

SSH_PORT="${1:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_HOST="${2:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_CMD="ssh -p $SSH_PORT root@$SSH_HOST -o StrictHostKeyChecking=no"
PROJECT_LOCAL="/Users/julianquick/portfolio_copy/TRI_pet_project"

echo "=== Syncing updated code ==="
rsync -az --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    --exclude='results' --exclude='*.tar.gz' \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$PROJECT_LOCAL/adversarial_dust/" \
    root@$SSH_HOST:/root/project/adversarial_dust/
rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$PROJECT_LOCAL/configs/" \
    root@$SSH_HOST:/root/project/configs/

echo "=== Verifying LOO data present on remote ==="
$SSH_CMD "ls /root/project/results/loo_analysis/loo_data/*.npz 2>/dev/null | wc -l" || {
    echo "ERROR: loo_data not found on remote. Run launch_loo.sh first.";
    exit 1;
}

$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 || \
    (export PATH=/opt/miniconda/bin:\$PATH && \
     cd /root/InternVLA-M1 && \
     PYTHONPATH=/root/InternVLA-M1:\$PYTHONPATH \
     nohup python deployment/model_server/server_policy_M1.py \
       --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
       --port 10093 --use_bf16 > /tmp/internvla_server.log 2>&1 &)"
sleep 45

$SSH_CMD "cat > /root/run_diversity.sh << 'SCRIPT'
#!/bin/bash
set -e
export PATH=/opt/miniconda/bin:\$PATH
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:\$PYTHONPATH
cd /root/project

python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml \
    --diversity-curve \
    --loo-types fingerprint glare rain gaussian_noise jpeg \
               motion_blur defocus_blur dust_camera low_light \
    --eval-episodes 10 \
    --episodes-per-condition 10 \
    --budget-levels 0.1 0.3 0.5 0.7 0.9 \
    --frame-stride 10 \
    --epochs 50 \
    --diversity-subsets 5 \
    --device cuda \
    --output-dir results/loo_analysis

tar czf /tmp/diversity_curve.tar.gz -C /root/project \
    results/loo_analysis/diversity_curve/
echo 'Archived: /tmp/diversity_curve.tar.gz'
ls -lh /tmp/diversity_curve.tar.gz
SCRIPT
chmod +x /root/run_diversity.sh"

$SSH_CMD "tmux new-session -d -s diversity 'bash /root/run_diversity.sh 2>&1 | tee /root/diversity.log'"
echo ""
echo "Launched in tmux session 'diversity'."
echo "Monitor: $SSH_CMD 'tail -f /root/diversity.log'"
echo "Download: scp -P $SSH_PORT root@$SSH_HOST:/tmp/diversity_curve.tar.gz ."
echo "Estimated runtime: ~4-6 hours (4 sizes x 5 subsets x CNN training)."
