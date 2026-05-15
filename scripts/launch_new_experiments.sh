#!/bin/bash
# Master launcher for the RA-L paper's remaining experiments.
#
# Each block delegates to a sibling launch script (tmux-based, non-blocking).
# Toggle blocks by commenting/uncommenting. Blocks do NOT depend on each other
# *unless noted* — the dependency is only that they share the LOO dataset
# produced by launch_loo.sh.
#
# Dependency graph:
#   launch_loo.sh          (one-time, ~20-25 h)  ──┬─> launch_diversity_curve.sh
#                                                   ├─> launch_ablations.sh
#                                                   └─> launch_baselines.sh
#   launch_envelope_cokecan.sh   (independent, ~10-15 h)
#
# Offline (no GPU needed), run locally:
#   python scripts/compute_ece.py --loo-dir results/loo_analysis_v3
#   python scripts/benchmark_inference.py --device cuda    # on GPU host
#   python scripts/summarize_baselines.py --loo-dir results/loo_analysis
#
# Usage:
#   bash scripts/launch_new_experiments.sh <SSH_PORT> <SSH_HOST>

set -e
SSH_PORT="${1:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SSH_HOST="${2:?Usage: $0 <SSH_PORT> <SSH_HOST>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Block 1: Coke-can envelope (standalone, does not depend on LOO data).
#   Purpose: unify paper story on one task (currently LOO is coke-can,
#            envelope is eggplant).
# ---------------------------------------------------------------------------
# bash "$SCRIPT_DIR/launch_envelope_cokecan.sh" "$SSH_PORT" "$SSH_HOST"

# ---------------------------------------------------------------------------
# Block 2: Corruption diversity curve (requires LOO data collected).
#   Purpose: fulfills the \todo for Fig/plot of mean held-out rho vs
#            |O_train| in {2,4,6,8}.
# ---------------------------------------------------------------------------
# bash "$SCRIPT_DIR/launch_diversity_curve.sh" "$SSH_PORT" "$SSH_HOST"

# ---------------------------------------------------------------------------
# Block 3: Ablation sweep (requires LOO data collected).
#   Purpose: frozen vs finetune vs scratch; resnet18/50/vit; data fractions.
#   Narrowed to 3 representative LOO folds (rain/fingerprint/motion_blur)
#   inside the launch script. Edit there to widen.
# ---------------------------------------------------------------------------
# bash "$SCRIPT_DIR/launch_ablations.sh" "$SSH_PORT" "$SSH_HOST"

# ---------------------------------------------------------------------------
# Block 4: Baselines (requires trained LOO models).
#   Purpose: BRISQUE / NIQE / PixelCoverage alongside CNN, same protocol.
# ---------------------------------------------------------------------------
# bash "$SCRIPT_DIR/launch_baselines.sh" "$SSH_PORT" "$SSH_HOST"

echo "No block uncommented. Edit this file and enable the blocks you want,"
echo "then rerun. Each launches into its own tmux session so they can run"
echo "concurrently on the same instance if there is enough GPU memory."
