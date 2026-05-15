"""Test whether feature-delta similarity to training set predicts LOO performance.

Loads:
  results/loo_analysis_v2/feature_similarity.json — pairwise corruption-type
    similarity matrices in two senses: raw (cosine sim of corrupted features)
    and diff (cosine sim of the [corrupted - clean] delta vectors).
  results/loo_analysis_v3/loo_summary.json — per-held-out-fold Spearman rho
    and AUROC.

For each held-out type, we compute its max similarity to any other type
(treating "other types" as the training set, since LOO trains on the other 8)
and plot vs. its measured OOD performance. A positive correlation means the
similarity matrix is predictive of generalization; useful both as a
sanity check on the "frozen-features encode generic visual degradation"
hypothesis and as a deployment-time guide.

Outputs:
  results/loo_analysis_v3/ood_predictability.json
  results/loo_analysis_v3/ood_predictability.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity",
                        default="results/loo_analysis_v2/feature_similarity.json")
    parser.add_argument("--loo",
                        default="results/loo_analysis_v3/loo_summary.json")
    parser.add_argument("--out-prefix",
                        default="results/loo_analysis_v3/ood_predictability")
    args = parser.parse_args()

    fs = json.loads(Path(args.similarity).read_text())
    loo = json.loads(Path(args.loo).read_text())

    types = fs["corruption_types"]
    M_raw = np.array(fs["raw_similarity_matrix"])
    M_diff = np.array(fs["diff_similarity_matrix"])
    loo_results = {r["held_out"]: r for r in loo["per_fold"]}

    rows = []
    for i, t in enumerate(types):
        r = loo_results.get(t, {})
        rho = r.get("spearman_rho")
        auroc = r.get("auroc")
        others = [j for j in range(len(types)) if j != i]
        rows.append({
            "type": t,
            "max_raw_sim": float(M_raw[i, others].max()),
            "max_diff_sim": float(M_diff[i, others].max()),
            "mean_diff_sim": float(M_diff[i, others].mean()),
            "spearman_rho": rho if rho is not None and rho == rho else None,
            "auroc": auroc if auroc is not None and auroc == auroc else None,
        })

    valid = [r for r in rows if r["spearman_rho"] is not None]
    md = np.array([r["max_diff_sim"] for r in valid])
    mnd = np.array([r["mean_diff_sim"] for r in valid])
    mr = np.array([r["max_raw_sim"] for r in valid])
    rhos = np.array([r["spearman_rho"] for r in valid])
    arcs = np.array([r["auroc"] for r in valid])

    correlations = {
        "max_diff_vs_rho":    {"pearson": pearsonr(md, rhos)[0],  "spearman": spearmanr(md, rhos)[0]},
        "mean_diff_vs_rho":   {"pearson": pearsonr(mnd, rhos)[0], "spearman": spearmanr(mnd, rhos)[0]},
        "max_diff_vs_auroc":  {"pearson": pearsonr(md, arcs)[0],  "spearman": spearmanr(md, arcs)[0]},
        "mean_diff_vs_auroc": {"pearson": pearsonr(mnd, arcs)[0], "spearman": spearmanr(mnd, arcs)[0]},
        "max_raw_vs_rho":     {"pearson": pearsonr(mr, rhos)[0],  "spearman": spearmanr(mr, rhos)[0]},
        "max_raw_vs_auroc":   {"pearson": pearsonr(mr, arcs)[0],  "spearman": spearmanr(mr, arcs)[0]},
    }

    summary = {"per_type": rows, "n_valid": len(valid),
               "correlations": correlations}
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(md, arcs, s=80, c="C0", edgecolor="black", alpha=0.85, zorder=3)
    for r, x, y in zip(valid, md, arcs):
        ax.annotate(r["type"].replace("_", " "), (x, y),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    # Linear fit for visual reference
    coefs = np.polyfit(md, arcs, 1)
    xx = np.linspace(md.min() * 0.95, md.max() * 1.05, 50)
    ax.plot(xx, coefs[0] * xx + coefs[1], "k--", alpha=0.4, zorder=2,
            label=f"linear fit (slope {coefs[0]:+.2f})")
    ax.set_xlabel("Max diff-feature similarity to any training-set corruption")
    ax.set_ylabel("Held-out AUROC")
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0.55, 1.05)
    ax.set_title(
        f"Feature-similarity predicts OOD detection\n"
        f"Pearson r = {correlations['max_diff_vs_auroc']['pearson']:+.2f}, "
        f"Spearman = {correlations['max_diff_vs_auroc']['spearman']:+.2f} (n={len(valid)})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_prefix + ".png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"n valid folds = {len(valid)}")
    for k, v in correlations.items():
        print(f"  {k:<25} Pearson {v['pearson']:+.3f}  Spearman {v['spearman']:+.3f}")
    print(f"Wrote {args.out_prefix}.json and {args.out_prefix}.png")


if __name__ == "__main__":
    main()
