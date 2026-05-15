#!/bin/bash
# Baseline comparison launcher.
#
# Reruns the LOO evaluation loop with --run-baselines so BRISQUE, NIQE, and
# PixelCoverage scores are computed on the same episode frames as the CNN.
# Assumes launch_loo.sh has already been run (trained models + loo_data on
# the remote). Training and data collection are skipped.
#
# Usage:
#   bash scripts/launch_baselines.sh <SSH_PORT> <SSH_HOST>

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
rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$PROJECT_LOCAL/scripts/" \
    root@$SSH_HOST:/root/project/scripts/

echo "=== Verifying trained LOO models on remote ==="
$SSH_CMD "ls /root/project/results/loo_analysis/loo_fold_*/model/best_model.pt 2>/dev/null | wc -l" || {
    echo "ERROR: no trained LOO models found. Run launch_loo.sh first.";
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

$SSH_CMD "cat > /root/run_baselines.sh << 'SCRIPT'
#!/bin/bash
set -e
export PATH=/opt/miniconda/bin:\$PATH
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:\$PYTHONPATH
cd /root/project

python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml \
    --loo --skip-collect --skip-train \
    --run-baselines \
    --loo-types fingerprint glare rain gaussian_noise jpeg \
               motion_blur defocus_blur dust_camera low_light \
    --eval-episodes 10 \
    --episodes-per-condition 10 \
    --budget-levels 0.1 0.3 0.5 0.7 0.9 \
    --frame-stride 10 \
    --device cuda \
    --output-dir results/loo_analysis

tar czf /tmp/baselines.tar.gz -C /root/project \
    results/loo_analysis/loo_fold_*/fold_results.json \
    results/loo_analysis/loo_summary.json
echo 'Archived: /tmp/baselines.tar.gz'
ls -lh /tmp/baselines.tar.gz
SCRIPT
chmod +x /root/run_baselines.sh"

$SSH_CMD "tmux new-session -d -s baselines 'bash /root/run_baselines.sh 2>&1 | tee /root/baselines.log'"
echo ""
echo "Launched in tmux session 'baselines'."
echo "Monitor: $SSH_CMD 'tail -f /root/baselines.log'"
echo "Download: scp -P $SSH_PORT root@$SSH_HOST:/tmp/baselines.tar.gz ."
echo "Estimated runtime: ~1-2 hours (eval rollouts on 9 folds x 5 budgets x 10 eps)."
echo ""
echo "After download, compute baseline-vs-CNN Spearman/AUROC with:"
echo "  python scripts/summarize_baselines.py --loo-dir results/loo_analysis"
