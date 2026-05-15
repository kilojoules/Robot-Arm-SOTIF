"""Offline: summarize baseline predictor performance on LOO folds.

Reads fold_results.json written by run_safety_predictor.py --run-baselines
and reports Spearman rho (severity ranking) and AUROC (episode-level
failure prediction) for each baseline alongside the CNN, per fold and in
aggregate.

Usage:
    python scripts/summarize_baselines.py \
        --loo-dir results/loo_analysis \
        --output results/loo_analysis/baselines_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


BASELINE_KEY_PREFIX = "baseline_"


def collect_metrics(fold_json: dict):
    """Return dict mapping predictor name -> (rho, auroc, n)."""
    eval_results = fold_json["eval_results"]
    budgets = sorted(float(b) for b in eval_results)

    # Identify baseline names by scanning one episode.
    first_ep = next(iter(eval_results.values()))["episodes"][0]
    baseline_names = [k.replace(BASELINE_KEY_PREFIX, "")
                      for k in first_ep
                      if k.startswith(BASELINE_KEY_PREFIX)]

    metrics = {}

    def _predictor_metrics(score_fn_mean):
        # Spearman: per-budget mean score vs (1 - SR)
        actuals, preds = [], []
        for b in budgets:
            r = eval_results[str(b)]
            actuals.append(1 - r["actual_success_rate"])
            preds.append(score_fn_mean(r))
        if len(set(actuals)) < 2:
            rho = float("nan")
        else:
            rho, _ = spearmanr(actuals, preds)

        # AUROC: per-episode scores vs binary failure
        ep_scores, ep_labels = [], []
        for b in budgets:
            for ep in eval_results[str(b)]["episodes"]:
                ep_labels.append(ep["actual_failure"])
                ep_scores.append(score_fn_mean({"episodes": [ep]}, episode=ep))
        if len(set(ep_labels)) < 2:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(ep_labels, ep_scores))
        return {"rho": float(rho) if not np.isnan(rho) else None,
                "auroc": auroc if not np.isnan(auroc) else None,
                "n_episodes": len(ep_labels)}

    # CNN
    def cnn_mean(r, episode=None):
        if episode is not None:
            return episode["predicted_p_failure_mean"]
        return r["mean_predicted_p_failure"]
    metrics["CNN"] = _predictor_metrics(cnn_mean)

    # Each baseline
    for name in baseline_names:
        key = BASELINE_KEY_PREFIX + name
        mean_key = "mean_" + key

        def bl_mean(r, episode=None, _key=key, _mean_key=mean_key):
            if episode is not None:
                return episode[_key]
            return r.get(_mean_key, float("nan"))

        metrics[name] = _predictor_metrics(bl_mean)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    fold_paths = sorted((args.loo_dir).glob("loo_fold_*/fold_results.json"))
    if not fold_paths:
        fold_paths = sorted((args.loo_dir).glob("fold_*.json"))
    if not fold_paths:
        raise SystemExit(f"No fold_results in {args.loo_dir}")

    per_fold = {}
    for fp in fold_paths:
        with open(fp) as f:
            fold_json = json.load(f)
        held_out = fold_json.get("held_out") or fp.parent.name.replace("loo_fold_", "")
        per_fold[held_out] = collect_metrics(fold_json)

    # Aggregate per predictor
    predictor_names = sorted({name for pf in per_fold.values() for name in pf})
    summary = {"per_fold": per_fold, "aggregate": {}}
    for pred in predictor_names:
        rhos, aurocs = [], []
        for pf in per_fold.values():
            m = pf.get(pred, {})
            if m.get("rho") is not None:
                rhos.append(m["rho"])
            if m.get("auroc") is not None:
                aurocs.append(m["auroc"])
        summary["aggregate"][pred] = {
            "mean_rho": float(np.mean(rhos)) if rhos else None,
            "std_rho": float(np.std(rhos)) if rhos else None,
            "mean_auroc": float(np.mean(aurocs)) if aurocs else None,
            "std_auroc": float(np.std(aurocs)) if aurocs else None,
            "n_folds_with_rho": len(rhos),
            "n_folds_with_auroc": len(aurocs),
        }

    out = args.output or (args.loo_dir / "baselines_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Print compact table
    print(f"{'predictor':<16} {'mean rho':>10} {'std':>6} "
          f"{'mean AUROC':>12} {'std':>6} {'folds':>6}")
    print("-" * 60)
    for pred, agg in summary["aggregate"].items():
        def fmt(x):
            return f"{x:.3f}" if x is not None else "  -  "
        print(f"{pred:<16} {fmt(agg['mean_rho']):>10} "
              f"{fmt(agg['std_rho']):>6} {fmt(agg['mean_auroc']):>12} "
              f"{fmt(agg['std_auroc']):>6} {agg['n_folds_with_rho']:>6}")

    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
