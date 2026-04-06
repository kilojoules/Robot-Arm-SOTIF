#!/bin/bash
# Launch the LOO cross-corruption analysis on a running Vast.ai instance.
#
# Prerequisites:
#   1. Vast.ai instance with nvidia/vulkan:1.3-470 image, running
#   2. setup_nvvulkan.sh already applied
#
# Usage:
#   bash scripts/launch_loo.sh <SSH_PORT> <SSH_HOST>

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

rsync -az \
    -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
    "$PROJECT_LOCAL/scripts/" \
    root@$SSH_HOST:/root/project/scripts/

# Upload prior sweep results for adversarial params (if available)
if [ -d "$PROJECT_LOCAL/results/coke_can_fingerprint" ]; then
    echo "=== Uploading prior sweep results ==="
    rsync -az --include='envelope_results.json' --exclude='*' \
        -e "ssh -p $SSH_PORT -o StrictHostKeyChecking=no" \
        "$PROJECT_LOCAL/results/coke_can_fingerprint/" \
        root@$SSH_HOST:/root/project/results/coke_can_fingerprint/ 2>/dev/null || true
fi

echo "=== Checking InternVLA-M1 server ==="
$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 && echo 'Server running' || echo 'Server NOT running - starting...'"

$SSH_CMD "pgrep -f server_policy_M1 > /dev/null 2>&1 || \
    (export PATH=/opt/miniconda/bin:\$PATH && \
     cd /root/InternVLA-M1 && \
     PYTHONPATH=/root/InternVLA-M1:\$PYTHONPATH \
     nohup python deployment/model_server/server_policy_M1.py \
       --ckpt_path /root/internvla_m1_ckpt/checkpoints/steps_50000_pytorch_model.pt \
       --port 10093 --use_bf16 > /tmp/internvla_server.log 2>&1 &)"

echo "=== Waiting for InternVLA-M1 server (45s) ==="
sleep 45

echo "=== Deploying LOO run script ==="
$SSH_CMD "cat > /root/run_loo.sh << 'SCRIPT'
#!/bin/bash
set -e
export PATH=/opt/miniconda/bin:\$PATH
export VK_ICD_FILENAMES=\$(find /usr/share/vulkan /etc/vulkan -name 'nvidia_icd*.json' 2>/dev/null | head -1)
export PYTHONPATH=/root/InternVLA-M1:/root/project:/root/camera_occlusion:\$PYTHONPATH
cd /root/project

echo '============================================'
echo 'STEP 1: Feature similarity analysis (predictions)'
echo '============================================'
python -m adversarial_dust.feature_similarity \
    --mode synthetic --device cuda \
    --output-dir results/loo_analysis/feature_similarity

echo ''
echo '============================================'
echo 'STEP 2: LOO cross-corruption analysis'
echo '============================================'
python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml \
    --loo \
    --loo-types fingerprint glare rain gaussian_noise jpeg \
               motion_blur defocus_blur fog low_light \
    --eval-episodes 10 \
    --episodes-per-condition 10 \
    --budget-levels 0.1 0.3 0.5 0.7 0.9 \
    --frame-stride 10 \
    --epochs 50 \
    --device cuda \
    --output-dir results/loo_analysis

echo ''
echo '============================================'
echo 'STEP 3: Corruption diversity curve'
echo '============================================'
python -u adversarial_dust/run_safety_predictor.py \
    --config configs/safety_predictor.yaml \
    --diversity-curve \
    --loo-types fingerprint glare rain gaussian_noise jpeg \
               motion_blur defocus_blur fog low_light \
    --eval-episodes 10 \
    --episodes-per-condition 10 \
    --budget-levels 0.1 0.3 0.5 0.7 0.9 \
    --frame-stride 10 \
    --epochs 50 \
    --device cuda \
    --diversity-subsets 3 \
    --output-dir results/loo_analysis

echo ''
echo '============================================'
echo 'STEP 4: Generate paper figures'
echo '============================================'
python scripts/generate_paper_figures.py \
    --results-dir results/loo_analysis \
    --output-dir results/loo_analysis/figures || true

echo ''
echo '============================================'
echo 'ALL DONE! Archiving results...'
echo '============================================'
tar czf /tmp/loo_results.tar.gz -C /root/project results/loo_analysis/
echo 'Results archived at /tmp/loo_results.tar.gz'
ls -lh /tmp/loo_results.tar.gz
SCRIPT
chmod +x /root/run_loo.sh"

$SSH_CMD "tmux new-session -d -s loo 'bash /root/run_loo.sh 2>&1 | tee /root/loo.log'"

echo ""
echo "==========================================="
echo "LOO analysis launched in tmux session 'loo'"
echo "==========================================="
echo ""
echo "Monitor:"
echo "  $SSH_CMD 'tail -f /root/loo.log'"
echo ""
echo "Download results when done:"
echo "  scp -P $SSH_PORT root@$SSH_HOST:/tmp/loo_results.tar.gz ."
echo ""
echo "Estimated runtime: ~20-25 hours (data collection) + ~30 min (training + diversity curve)"
