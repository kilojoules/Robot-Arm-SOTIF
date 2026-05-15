#!/bin/bash
# Ablation sweep launcher.
#
# Sweeps backbone/freeze/data-fraction combinations, each re-running the
# 9-fold LOO evaluation on the held-out type. Reuses the already-collected
# LOO dataset (loo_data/*.npz); only classifier training + held-out eval
# rollouts repeat per variant.
#
# Runtime is dominated by eval rollouts (~25 min/fold x 9 folds x N variants).
# Edit VARIANTS and LOO_TYPES below to shorten if compute is limited.
#
# Usage:
#   bash scripts/launch_ablations.sh <SSH_PORT> <SSH_HOST>

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

$SSH_CMD "cat > /root/run_ablations.sh << 'SCRIPT'
#!/bin/bash
set -e
export PATH=/opt/miniconda/bin:\$PATH
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:\$PYTHONPATH
cd /root/project

# Narrow the fold list to 3 representative types (easy/hard/medium) to keep
# the ablation budget sane. Remove \"--loo-types\" below to restore full 9-fold.
LOO_TYPES=\"rain fingerprint motion_blur\"

run_variant() {
  local TAG=\$1
  shift
  local OUTDIR=\"results/ablations/\$TAG\"
  echo \"\"
  echo \"=================================================\"
  echo \"VARIANT: \$TAG\"
  echo \"=================================================\"
  python -u adversarial_dust/run_safety_predictor.py \\
      --config configs/safety_predictor.yaml \\
      --loo --skip-collect \\
      --loo-types \$LOO_TYPES \\
      --eval-episodes 10 \\
      --episodes-per-condition 10 \\
      --budget-levels 0.1 0.3 0.5 0.7 0.9 \\
      --frame-stride 10 \\
      --epochs 50 \\
      --device cuda \\
      --output-dir \"\$OUTDIR\" \\
      \"\$@\"
  # Symlink the shared loo_data into the ablation output dir so the
  # pipeline finds preprocessed episode .npz files.
  mkdir -p \"\$OUTDIR/loo_data\"
  for f in results/loo_analysis/loo_data/*.npz; do
      ln -sf \"\$(pwd)/\$f\" \"\$OUTDIR/loo_data/\$(basename \$f)\"
  done
}

# Variant matrix:
#   freeze/finetune/scratch ablation                 (resnet18)
#   backbone ablation at frozen setting              (resnet18/50, vit_b_16)
#   learning-curve (data-fraction) ablation          (resnet18 frozen)
run_variant resnet18_frozen         --backbone resnet18 --freeze-backbone
run_variant resnet18_finetune       --backbone resnet18 --no-freeze-backbone
run_variant resnet18_scratch        --backbone resnet18 --from-scratch --no-freeze-backbone
run_variant resnet18_frozen_half    --backbone resnet18 --freeze-backbone --data-fraction 0.5
run_variant resnet18_frozen_quarter --backbone resnet18 --freeze-backbone --data-fraction 0.25
run_variant resnet50_frozen         --backbone resnet50 --freeze-backbone
run_variant vit_b16_frozen          --backbone vit_b_16 --freeze-backbone

tar czf /tmp/ablations.tar.gz -C /root/project results/ablations/
echo 'Archived: /tmp/ablations.tar.gz'
ls -lh /tmp/ablations.tar.gz
SCRIPT
chmod +x /root/run_ablations.sh"

$SSH_CMD "tmux new-session -d -s ablations 'bash /root/run_ablations.sh 2>&1 | tee /root/ablations.log'"
echo ""
echo "Launched in tmux session 'ablations'."
echo "Monitor: $SSH_CMD 'tail -f /root/ablations.log'"
echo "Download: scp -P $SSH_PORT root@$SSH_HOST:/tmp/ablations.tar.gz ."
echo "Estimated runtime: ~7-10 hours on 3 folds x 7 variants."
