"""Generate figures for the RA-L paper from experiment results.

Usage:
    python scripts/generate_paper_figures.py \
        --results-dir results/safety_predictor \
        --output-dir paper/figures

Generates:
  1. Cross-corruption generalization bar chart (Spearman rho per corruption)
  2. Reliability diagram (calibration)
  3. Per-budget P(fail) vs actual SR plot
  4. Ablation results table (if multi-seed summary exists)
  5. GradCAM montage (requires model + sample frames)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(calibration_data: dict, output_path: str):
    """Plot reliability diagram from calibration results."""
    fig, axes = plt.subplots(1, len(calibration_data), figsize=(5 * len(calibration_data), 4.5),
                              squeeze=False)

    for idx, (occ_type, cal) in enumerate(calibration_data.items()):
        ax = axes[0, idx]
        bins = cal["reliability_bins"]
        predicted = [b["mean_predicted"] for b in bins if b["count"] > 0]
        actual = [b["actual_fraction"] for b in bins if b["count"] > 0]
        counts = [b["count"] for b in bins if b["count"] > 0]

        ax.bar(predicted, actual, width=0.08, alpha=0.7, label="Observed")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Mean predicted P(fail)")
        ax.set_ylabel("Actual failure fraction")
        ax.set_title(f"{occ_type.capitalize()} (ECE={cal['ece']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability diagram to {output_path}")


def plot_budget_vs_prediction(eval_results: dict, output_path: str):
    """Plot actual SR and predicted P(fail) vs budget for each corruption."""
    occ_types = [k for k in eval_results if not k.endswith("_spearman")
                 and k != "calibration"]

    fig, axes = plt.subplots(1, len(occ_types),
                              figsize=(5 * len(occ_types), 4),
                              squeeze=False)

    for idx, occ_type in enumerate(occ_types):
        ax = axes[0, idx]
        budgets_data = eval_results[occ_type]
        budgets = sorted(float(b) for b in budgets_data.keys())

        actual_srs = [budgets_data[str(b)]["actual_success_rate"] for b in budgets]
        pred_pf = [budgets_data[str(b)]["mean_predicted_p_failure"] for b in budgets]

        ax.plot(budgets, actual_srs, "bo-", label="Actual SR", linewidth=2)
        ax.plot(budgets, pred_pf, "r^--", label="Predicted P(fail)", linewidth=2)
        ax.set_xlabel("Budget level")
        ax.set_ylabel("Rate")
        ax.set_title(occ_type.capitalize())
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved budget vs prediction plot to {output_path}")


def plot_spearman_comparison(eval_results: dict, output_path: str):
    """Bar chart of Spearman rho per corruption type."""
    rho_data = {}
    for key, val in eval_results.items():
        if key.endswith("_spearman") and isinstance(val, dict):
            occ_type = key.replace("_spearman", "")
            rho_data[occ_type] = val

    if not rho_data:
        print("No Spearman data found, skipping bar chart")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(rho_data.keys())
    rhos = [rho_data[n]["rho"] for n in names]
    ci_lo = [rho_data[n].get("ci_lower", rho_data[n]["rho"]) for n in names]
    ci_hi = [rho_data[n].get("ci_upper", rho_data[n]["rho"]) for n in names]
    errors = [[r - lo for r, lo in zip(rhos, ci_lo)],
              [hi - r for r, hi in zip(rhos, ci_hi)]]

    colors = ["tab:orange" if n in ("fingerprint", "glare") else "tab:blue"
              for n in names]
    bars = ax.bar(names, rhos, yerr=errors, color=colors, alpha=0.8,
                  capsize=5, edgecolor="black", linewidth=0.5)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Spearman rho")
    ax.set_title("Cross-Corruption Generalization")
    ax.set_ylim(-1, 1)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="tab:orange", label="Training"),
              Patch(color="tab:blue", label="Held-out")]
    ax.legend(handles=legend, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Spearman comparison to {output_path}")


def plot_multi_seed_summary(summary_path: str, output_path: str):
    """Plot multi-seed aggregated results."""
    with open(summary_path) as f:
        summary = json.load(f)

    occ_types = [k for k in summary if k not in ("n_seeds", "seeds")]
    if not occ_types:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    names = []
    means = []
    stds = []

    for occ_type in occ_types:
        data = summary[occ_type]
        names.append(occ_type)
        means.append(data["mean_rho"])
        stds.append(data["std_rho"])

    ax.bar(names, means, yerr=stds, alpha=0.8, capsize=5,
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Spearman rho (mean +/- std)")
    ax.set_title(f"Multi-Seed Results (n={summary['n_seeds']})")
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved multi-seed summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str,
                        default="results/safety_predictor")
    parser.add_argument("--output-dir", type=str, default="paper/figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation results
    eval_path = results_dir / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path) as f:
            eval_results = json.load(f)

        # Budget vs prediction plot
        plot_budget_vs_prediction(
            eval_results, str(output_dir / "budget_vs_prediction.pdf"))

        # Spearman comparison bar chart
        plot_spearman_comparison(
            eval_results, str(output_dir / "spearman_comparison.pdf"))

        # Reliability diagram
        if "calibration" in eval_results:
            plot_reliability_diagram(
                eval_results["calibration"],
                str(output_dir / "reliability_diagram.pdf"))
    else:
        print(f"No evaluation results found at {eval_path}")

    # Multi-seed summary
    multi_seed_path = results_dir / "multi_seed_summary.json"
    if multi_seed_path.exists():
        plot_multi_seed_summary(
            str(multi_seed_path),
            str(output_dir / "multi_seed_summary.pdf"))

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
