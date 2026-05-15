"""Offline Expected Calibration Error (ECE) for LOO safety-monitor results.

Reads per-fold evaluation JSON from a LOO run (e.g. results/loo_analysis_v3/)
and computes ECE plus a reliability diagram. No GPU, no rollouts — pure
post-hoc analysis of probabilities already logged during evaluation.

Usage:
    python scripts/compute_ece.py \
        --loo-dir results/loo_analysis_v3 \
        --output-dir results/loo_analysis_v3/calibration \
        --n-bins 10
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_fold_episodes(fold_path: Path):
    """Return (predictions, labels) arrays pooled over all budgets in one fold."""
    with open(fold_path) as f:
        data = json.load(f)
    preds, labels = [], []
    for _budget, budget_result in data["eval_results"].items():
        for ep in budget_result["episodes"]:
            preds.append(ep["predicted_p_failure_mean"])
            labels.append(ep["actual_failure"])
    return np.array(preds, dtype=np.float64), np.array(labels, dtype=np.int64)


def compute_ece(preds: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Expected Calibration Error with equal-width binning.

    Returns dict with ece, maximum calibration error, and per-bin stats.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(preds, bin_edges, right=True) - 1, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    bins = []
    total = len(preds)
    for b in range(n_bins):
        mask = bin_indices == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"bin": b, "n": 0, "mean_confidence": None,
                         "empirical_failure_rate": None, "gap": None})
            continue
        conf = float(preds[mask].mean())
        acc = float(labels[mask].mean())
        gap = abs(conf - acc)
        ece += (n / total) * gap
        mce = max(mce, gap)
        bins.append({
            "bin": b,
            "bin_lower": float(bin_edges[b]),
            "bin_upper": float(bin_edges[b + 1]),
            "n": n,
            "mean_confidence": conf,
            "empirical_failure_rate": acc,
            "gap": gap,
        })
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins,
            "n_samples": total, "bins": bins}


def plot_reliability(bins, ece, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys, ns = [], [], []
    for bin_info in bins:
        if bin_info["n"] == 0:
            continue
        xs.append(bin_info["mean_confidence"])
        ys.append(bin_info["empirical_failure_rate"])
        ns.append(bin_info["n"])

    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect calibration")
    if xs:
        sizes = np.array(ns) * 20
        ax.scatter(xs, ys, s=sizes, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Predicted P(failure)")
    ax.set_ylabel("Empirical failure rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{title}\nECE = {ece:.3f}")
    proxy_bin = Line2D([0], [0], marker="o", linestyle="None",
                       markerfacecolor="C0", markeredgecolor="black",
                       markersize=8, alpha=0.7,
                       label="bin (area $\\propto$ count)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [proxy_bin], labels + [proxy_bin.get_label()],
              loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo-dir", type=Path, required=True,
                        help="Directory containing fold_*.json (e.g. "
                             "results/loo_analysis_v3)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write calibration/ outputs "
                             "(default: <loo-dir>/calibration)")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.output_dir or (args.loo_dir / "calibration")
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_paths = sorted(args.loo_dir.glob("fold_*.json"))
    if not fold_paths:
        raise SystemExit(f"No fold_*.json in {args.loo_dir}")

    per_fold = []
    pooled_preds, pooled_labels = [], []
    for fold_path in fold_paths:
        held_out = fold_path.stem.replace("fold_", "")
        preds, labels = load_fold_episodes(fold_path)
        if len(set(labels.tolist())) < 2:
            per_fold.append({"held_out": held_out, "n_samples": len(labels),
                             "note": "degenerate (all success or all fail)"})
            continue
        res = compute_ece(preds, labels, n_bins=args.n_bins)
        res["held_out"] = held_out
        per_fold.append(res)
        pooled_preds.append(preds)
        pooled_labels.append(labels)
        plot_reliability(
            res["bins"], res["ece"],
            title=f"Reliability — held out: {held_out}",
            out_path=out_dir / f"reliability_{held_out}.png",
        )
        print(f"{held_out:15s}  n={res['n_samples']:4d}  "
              f"ECE={res['ece']:.3f}  MCE={res['mce']:.3f}")

    if pooled_preds:
        pooled = compute_ece(np.concatenate(pooled_preds),
                             np.concatenate(pooled_labels),
                             n_bins=args.n_bins)
        plot_reliability(pooled["bins"], pooled["ece"],
                         title="Reliability — pooled across all folds",
                         out_path=out_dir / "reliability_pooled.png")
        print(f"{'POOLED':15s}  n={pooled['n_samples']:4d}  "
              f"ECE={pooled['ece']:.3f}  MCE={pooled['mce']:.3f}")
    else:
        pooled = None

    summary = {"per_fold": per_fold, "pooled": pooled,
               "n_bins": args.n_bins, "source_dir": str(args.loo_dir)}
    with open(out_dir / "ece_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_dir}/ece_summary.json and reliability_*.png")


if __name__ == "__main__":
    main()
