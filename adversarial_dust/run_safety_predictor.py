"""Full pipeline: collect data, train CNN safety predictor, evaluate on rain.

Orchestrates the three stages on a GPU machine:
  1. Collect training data (fingerprint + glare episodes, random + adversarial)
  2. Train SafetyPredictorCNN on collected data
  3. Evaluate on held-out rain occlusion (generalization test)
  4. Evaluate on fingerprint holdout (sanity check on training distribution)

Usage (on GPU machine):
    python -u adversarial_dust/run_safety_predictor.py \
        --config configs/safety_predictor.yaml \
        --output-dir results/safety_predictor

Multi-seed mode (for statistical rigor):
    python -u adversarial_dust/run_safety_predictor.py \
        --config configs/safety_predictor.yaml \
        --seeds 42,123,456,789,1337
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def bootstrap_spearman_ci(
    x: list, y: list, n_boot: int = 1000, ci: float = 0.95, seed: int = 42,
) -> dict:
    """Compute bootstrapped confidence interval for Spearman correlation.

    Returns dict with rho, p_value, ci_lower, ci_upper, n_boot.
    """
    from scipy.stats import spearmanr

    rho, pval = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    rhos = []
    x_arr, y_arr = np.array(x), np.array(y)
    for _ in range(n_boot):
        idx = rng.choice(len(x_arr), size=len(x_arr), replace=True)
        r, _ = spearmanr(x_arr[idx], y_arr[idx])
        if not np.isnan(r):
            rhos.append(r)

    alpha = 1 - ci
    lower = np.percentile(rhos, 100 * alpha / 2) if rhos else float("nan")
    upper = np.percentile(rhos, 100 * (1 - alpha / 2)) if rhos else float("nan")

    return {
        "rho": float(rho),
        "p_value": float(pval),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_boot": n_boot,
    }


def stage_collect(config, policy, image_shape, output_dir, args):
    """Stage 1: Collect training data from fingerprint + glare episodes."""
    from adversarial_dust.collect_training_data import collect_episodes

    data_path = output_dir / "training_data.npz"
    if data_path.exists() and not args.force_recollect:
        print(f"Training data exists at {data_path}, skipping collection. "
              f"Use --force-recollect to redo.", flush=True)
        return

    print("=" * 60, flush=True)
    print("STAGE 1: Collecting training data", flush=True)
    print("=" * 60, flush=True)

    # Parse adversarial results dirs
    adv_dirs = {}
    if args.adv_results:
        for pair in args.adv_results:
            occ_type, results_dir = pair.split(":", 1)
            adv_dirs[occ_type] = results_dir

    images, labels, metadata = collect_episodes(
        config=config,
        policy=policy,
        image_shape=image_shape,
        occlusion_types=args.occlusion_types,
        budget_levels=args.budget_levels,
        episodes_per_condition=args.episodes_per_condition,
        frame_stride=args.frame_stride,
        adversarial_results_dirs=adv_dirs,
        adversarial_episodes=args.adversarial_episodes,
    )

    np.savez_compressed(str(data_path), images=images, labels=labels)

    meta_path = data_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "n_samples": len(labels),
            "n_failures": int(labels.sum()),
            "failure_rate": float(labels.mean()),
            "adversarial_results_dirs": adv_dirs,
            "samples": metadata,
        }, f, indent=2)

    print(f"Saved {len(labels)} samples to {data_path}", flush=True)


def stage_train(output_dir, args):
    """Stage 2: Train the CNN safety predictor."""
    from adversarial_dust.safety_predictor import train_safety_predictor

    data_path = output_dir / "training_data.npz"
    model_dir = output_dir / "model"

    print("\n" + "=" * 60, flush=True)
    print("STAGE 2: Training CNN safety predictor", flush=True)
    print("=" * 60, flush=True)

    predictor = train_safety_predictor(
        dataset_path=str(data_path),
        output_dir=str(model_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        frames_per_episode=100 // args.frame_stride,
        backbone=args.backbone,
        freeze_backbone=args.freeze_backbone,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        backbone_lr=args.backbone_lr,
        seed=args.seed,
        data_fraction=args.data_fraction,
    )

    return predictor


def _evaluate_occlusion(config, policy, image_shape, predictor, output_dir,
                         args, occlusion_type, model_factory, budgets,
                         episodes_per_budget, label):
    """Run evaluation episodes with a given occlusion type and score with CNN.

    Returns dict mapping budget_str -> results.
    """
    from adversarial_dust.simpler_env_evaluator import SimplerEnvEvaluator

    # Load baselines if requested
    baselines = []
    if getattr(args, "run_baselines", False):
        from adversarial_dust.baselines import get_all_baselines
        baselines = get_all_baselines()

    results = {}

    for budget in budgets:
        print(f"\n{label} budget={budget:.0%}, "
              f"{episodes_per_budget} episodes", flush=True)

        model = model_factory(budget)
        evaluator = SimplerEnvEvaluator(config.env, policy, model)
        rng = np.random.default_rng(123)

        episode_results = []
        for ep in range(episodes_per_budget):
            params = model.get_random_params(rng)
            t0 = time.time()
            success, _, dirty_frames = evaluator.run_episode(params, record=True)
            elapsed = time.time() - t0

            sample_frames = dirty_frames[::args.frame_stride]
            if sample_frames:
                p_failures = predictor.predict_batch(sample_frames)
                mean_p_fail = float(np.mean(p_failures))
                max_p_fail = float(np.max(p_failures))
            else:
                mean_p_fail = max_p_fail = 0.0

            ep_result = {
                "episode": ep,
                "actual_success": bool(success),
                "actual_failure": 0 if success else 1,
                "predicted_p_failure_mean": mean_p_fail,
                "predicted_p_failure_max": max_p_fail,
                "n_frames_scored": len(sample_frames),
            }

            # Score baselines on the same frames
            for bl in baselines:
                if sample_frames:
                    bl_score = bl.score_frames(sample_frames)
                else:
                    bl_score = 0.0
                ep_result[f"baseline_{bl.name}"] = bl_score

            episode_results.append(ep_result)

            print(f"  ep {ep+1}/{episodes_per_budget}: "
                  f"actual={'OK' if success else 'FAIL'}, "
                  f"pred_mean={mean_p_fail:.3f}, pred_max={max_p_fail:.3f}, "
                  f"{elapsed:.1f}s", flush=True)

        actual_sr = np.mean([1 - r["actual_failure"] for r in episode_results])
        mean_pred = np.mean([r["predicted_p_failure_mean"] for r in episode_results])

        budget_result = {
            "actual_success_rate": float(actual_sr),
            "mean_predicted_p_failure": float(mean_pred),
            "episodes": episode_results,
        }

        # Aggregate baseline scores per budget
        for bl in baselines:
            key = f"baseline_{bl.name}"
            bl_scores = [r[key] for r in episode_results]
            budget_result[f"mean_{key}"] = float(np.mean(bl_scores))

        results[str(budget)] = budget_result

        print(f"  Budget {budget:.0%}: actual SR={actual_sr:.0%}, "
              f"mean predicted P(fail)={mean_pred:.3f}", flush=True)

    return results


def compute_calibration(all_results: dict, n_bins: int = 10) -> dict:
    """Compute Expected Calibration Error and reliability diagram data.

    Pools all episode predictions across budgets for each corruption type.
    Returns calibration metrics per corruption type.
    """
    calibration = {}

    for occ_type, budgets_data in all_results.items():
        if occ_type.endswith("_spearman"):
            continue

        preds, actuals = [], []
        for budget_str, budget_data in budgets_data.items():
            if "episodes" not in budget_data:
                continue
            for ep in budget_data["episodes"]:
                preds.append(ep["predicted_p_failure_mean"])
                actuals.append(ep["actual_failure"])

        if not preds:
            continue

        preds = np.array(preds)
        actuals = np.array(actuals)

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_data = []
        ece = 0.0

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 \
                else (preds >= lo) & (preds <= hi)
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                bin_data.append({
                    "bin_lower": float(lo), "bin_upper": float(hi),
                    "mean_predicted": None, "actual_fraction": None,
                    "count": 0,
                })
                continue

            mean_pred = float(preds[mask].mean())
            actual_frac = float(actuals[mask].mean())
            ece += abs(mean_pred - actual_frac) * n_in_bin / len(preds)

            bin_data.append({
                "bin_lower": float(lo), "bin_upper": float(hi),
                "mean_predicted": mean_pred,
                "actual_fraction": actual_frac,
                "count": int(n_in_bin),
            })

        calibration[occ_type] = {
            "ece": float(ece),
            "n_predictions": len(preds),
            "reliability_bins": bin_data,
        }

        print(f"  {occ_type.upper()} ECE: {ece:.3f} "
              f"({len(preds)} predictions)", flush=True)

    return calibration


def _make_eval_model_factory(occlusion_type, config, image_shape):
    """Create a model factory for evaluation using the unified factory."""
    from adversarial_dust.envelope_predictor import make_occlusion_model
    return lambda budget: make_occlusion_model(
        occlusion_type, config, image_shape, budget)


def stage_evaluate(config, policy, image_shape, predictor, output_dir, args):
    """Stage 3: Evaluate predictor on configurable corruption types."""
    eval_budgets = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}

    train_types = set(args.occlusion_types)

    for i, occ_type in enumerate(args.eval_corruptions):
        is_held_out = occ_type not in train_types
        label_suffix = "held-out" if is_held_out else "training distribution"

        print(f"\n{'=' * 60}", flush=True)
        print(f"STAGE 3{chr(97+i)}: Evaluating on {occ_type.upper()} "
              f"({label_suffix})", flush=True)
        print(f"{'=' * 60}", flush=True)

        factory = _make_eval_model_factory(occ_type, config, image_shape)
        results = _evaluate_occlusion(
            config, policy, image_shape, predictor, output_dir, args,
            occlusion_type=occ_type,
            model_factory=factory,
            budgets=eval_budgets,
            episodes_per_budget=args.eval_episodes,
            label=occ_type.capitalize(),
        )
        all_results[occ_type] = results

    # --- Calibration analysis ---
    print(f"\n{'=' * 60}", flush=True)
    print("CALIBRATION ANALYSIS", flush=True)
    print(f"{'=' * 60}", flush=True)
    calibration = compute_calibration(all_results)
    all_results["calibration"] = calibration

    # --- Save all evaluation results ---
    eval_path = output_dir / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation saved to {eval_path}", flush=True)

    # --- Summary table ---
    print("\n" + "=" * 60, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("=" * 60, flush=True)

    for occ_type, results in all_results.items():
        print(f"\n{occ_type.upper()}:", flush=True)
        print(f"  {'Budget':>8} {'Actual SR':>10} {'Pred P(fail)':>13}",
              flush=True)
        print(f"  {'-'*34}", flush=True)

        actuals, preds = [], []
        for budget in eval_budgets:
            r = results[str(budget)]
            actual_sr = r["actual_success_rate"]
            pred_pf = r["mean_predicted_p_failure"]
            actuals.append(1 - actual_sr)
            preds.append(pred_pf)
            print(f"  {budget:>7.0%} {actual_sr:>10.0%} {pred_pf:>13.3f}",
                  flush=True)

        # Rank correlation between predicted P(fail) and actual failure rate
        if len(set(actuals)) > 1:
            boot = bootstrap_spearman_ci(actuals, preds, n_boot=1000)
            print(f"  Spearman rho: {boot['rho']:.3f} "
                  f"(p={boot['p_value']:.3f}), "
                  f"95% CI: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]",
                  flush=True)
            all_results[f"{occ_type}_spearman"] = boot
        else:
            print(f"  (no variation in actual failure rate — "
                  f"correlation undefined)", flush=True)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Full CNN safety predictor pipeline"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str,
                        default="results/safety_predictor")
    parser.add_argument("--device", type=str, default="cuda")

    # Data collection (training corruptions)
    parser.add_argument("--occlusion-types", type=str, nargs="+",
                        default=["fingerprint", "glare"],
                        help="Corruption types for training data collection")
    parser.add_argument("--budget-levels", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--episodes-per-condition", type=int, default=5)
    parser.add_argument("--adversarial-episodes", type=int, default=3)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--force-recollect", action="store_true")
    parser.add_argument("--adv-results", type=str, nargs="*", default=None,
                        help="occlusion_type:results_dir pairs for "
                             "adversarial params")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Architecture (ablation support)
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vit_b_16"])
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                        action="store_false")
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="Separate LR for unfrozen backbone (default: lr/10)")
    parser.add_argument("--data-fraction", type=float, default=1.0,
                        help="Fraction of training data to use (learning curve ablation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training")

    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-corruptions", type=str, nargs="+",
                        default=["rain", "fingerprint"],
                        help="Corruption types to evaluate on "
                             "(first is primary held-out, rest are sanity checks)")

    # Baselines
    parser.add_argument("--run-baselines", action="store_true",
                        help="Evaluate baseline predictors alongside CNN")

    # Multi-seed (statistical rigor)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds for multi-seed runs "
                             "(e.g., '42,123,456'). Overrides --seed.")

    # Corruption diversity curve
    parser.add_argument("--diversity-curve", action="store_true",
                        help="Vary training set size {2,4,6,8} and report "
                             "mean rho vs number of training corruption types. "
                             "Requires --loo-types and collected LOO data.")
    parser.add_argument("--diversity-subsets", type=int, default=5,
                        help="Number of random subsets per training set size")

    # Leave-one-out cross-corruption analysis
    parser.add_argument("--loo", action="store_true",
                        help="Run leave-one-out cross-corruption analysis. "
                             "Collects data for all --occlusion-types, then "
                             "trains N models each holding out one type.")
    parser.add_argument("--loo-types", type=str, nargs="+",
                        default=["fingerprint", "glare", "rain",
                                 "gaussian_noise", "jpeg", "motion_blur",
                                 "defocus_blur", "dust_camera", "low_light"],
                        help="Corruption types for LOO analysis")

    # Stage control
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--only-eval", action="store_true",
                        help="Only run evaluation (requires trained model)")
    args = parser.parse_args()

    from adversarial_dust.config import load_envelope_config
    config = load_envelope_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from adversarial_dust.run_envelope import create_policy, get_image_shape

    print("Getting image shape...", flush=True)
    image_shape = get_image_shape(config)
    print(f"Image shape: {image_shape}", flush=True)

    print(f"Loading {config.env.policy_model} policy...", flush=True)
    policy = create_policy(config)

    # LOO mode: separate pipeline
    if args.loo:
        run_loo(config, policy, image_shape, output_dir, args)
        if args.diversity_curve:
            run_diversity_curve(config, policy, image_shape, output_dir, args)
        print("\nLOO pipeline complete!", flush=True)
        print(f"Results directory: {output_dir}", flush=True)
        return

    # Diversity curve standalone (requires LOO data already collected)
    if args.diversity_curve:
        run_diversity_curve(config, policy, image_shape, output_dir, args)
        print("\nDiversity curve complete!", flush=True)
        return

    # Parse seeds
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    # Stage 1: Collect (shared across seeds — data doesn't depend on train seed)
    if not args.skip_collect and not args.only_eval:
        stage_collect(config, policy, image_shape, output_dir, args)

    if len(seeds) == 1:
        # Single seed: original behavior
        args.seed = seeds[0]

        if not args.skip_train and not args.only_eval:
            predictor = stage_train(output_dir, args)
        else:
            from adversarial_dust.safety_predictor import SafetyPredictorCNN
            predictor = SafetyPredictorCNN()
            model_path = output_dir / "model" / "best_model.pt"
            predictor.load(str(model_path))
            predictor.to(args.device)
            print(f"Loaded existing model from {model_path}", flush=True)

        stage_evaluate(config, policy, image_shape, predictor, output_dir, args)
    else:
        # Multi-seed: train + eval for each seed, aggregate results
        all_seed_results = []

        for seed in seeds:
            print(f"\n{'#' * 60}", flush=True)
            print(f"# SEED {seed}", flush=True)
            print(f"{'#' * 60}", flush=True)

            args.seed = seed
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Symlink shared training data into seed dir
            shared_data = output_dir / "training_data.npz"
            seed_data = seed_dir / "training_data.npz"
            if shared_data.exists() and not seed_data.exists():
                seed_data.symlink_to(shared_data.resolve())

            if not args.skip_train and not args.only_eval:
                predictor = stage_train(seed_dir, args)
            else:
                from adversarial_dust.safety_predictor import SafetyPredictorCNN
                predictor = SafetyPredictorCNN()
                model_path = seed_dir / "model" / "best_model.pt"
                predictor.load(str(model_path))
                predictor.to(args.device)

            seed_results = stage_evaluate(
                config, policy, image_shape, predictor, seed_dir, args)
            seed_results["seed"] = seed
            all_seed_results.append(seed_results)

        # Aggregate across seeds
        _aggregate_multi_seed(all_seed_results, output_dir)

    print("\nPipeline complete!", flush=True)
    print(f"Results directory: {output_dir}", flush=True)


def run_loo(config, policy, image_shape, output_dir, args):
    """Run leave-one-out cross-corruption analysis.

    1. Collect episodes for ALL corruption types (shared dataset)
    2. For each fold: hold out one type, train on the rest, evaluate on held-out
    3. Aggregate: per-fold Spearman rho, mean across folds, AUROC
    """
    from adversarial_dust.collect_training_data import collect_episodes
    from adversarial_dust.safety_predictor import SafetyPredictorCNN, train_safety_predictor

    loo_types = args.loo_types
    n_folds = len(loo_types)

    print(f"\n{'#' * 60}", flush=True)
    print(f"# LEAVE-ONE-OUT ANALYSIS ({n_folds} folds)", flush=True)
    print(f"# Types: {', '.join(loo_types)}", flush=True)
    print(f"{'#' * 60}", flush=True)

    # --- Stage 1: Collect data for ALL types ---
    # Each type gets its own .npz so we can subset for training
    all_data_dir = output_dir / "loo_data"
    all_data_dir.mkdir(parents=True, exist_ok=True)

    adv_dirs = {}
    if args.adv_results:
        for pair in args.adv_results:
            occ_type, results_dir = pair.split(":", 1)
            adv_dirs[occ_type] = results_dir

    if not args.skip_collect:
        for occ_type in loo_types:
            data_path = all_data_dir / f"{occ_type}.npz"
            if data_path.exists() and not args.force_recollect:
                print(f"Data for {occ_type} exists, skipping.", flush=True)
                continue

            print(f"\n{'=' * 60}", flush=True)
            print(f"Collecting data: {occ_type}", flush=True)
            print(f"{'=' * 60}", flush=True)

            images, labels, metadata = collect_episodes(
                config=config,
                policy=policy,
                image_shape=image_shape,
                occlusion_types=[occ_type],
                budget_levels=args.budget_levels,
                episodes_per_condition=args.episodes_per_condition,
                frame_stride=args.frame_stride,
                adversarial_results_dirs={k: v for k, v in adv_dirs.items()
                                          if k == occ_type},
                adversarial_episodes=args.adversarial_episodes,
            )
            np.savez_compressed(str(data_path), images=images, labels=labels)
            meta_path = data_path.with_suffix(".json")
            with open(meta_path, "w") as f:
                json.dump({
                    "occlusion_type": occ_type,
                    "n_samples": len(labels),
                    "n_failures": int(labels.sum()),
                    "samples": metadata,
                }, f, indent=2)
            print(f"  {occ_type}: {len(labels)} samples, "
                  f"{int(labels.sum())} failures", flush=True)

    # --- Stage 2+3: LOO folds ---
    fold_results = []

    for fold_idx, held_out in enumerate(loo_types):
        train_types = [t for t in loo_types if t != held_out]

        print(f"\n{'#' * 60}", flush=True)
        print(f"# FOLD {fold_idx+1}/{n_folds}: hold out {held_out.upper()}", flush=True)
        print(f"# Train on: {', '.join(train_types)}", flush=True)
        print(f"{'#' * 60}", flush=True)

        fold_dir = output_dir / f"loo_fold_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Merge training data from all non-held-out types
        if not args.skip_train:
            train_images, train_labels = [], []
            for t in train_types:
                data_path = all_data_dir / f"{t}.npz"
                if not data_path.exists():
                    print(f"  WARNING: no data for {t}, skipping", flush=True)
                    continue
                d = np.load(str(data_path))
                train_images.append(d["images"])
                train_labels.append(d["labels"])

            if not train_images:
                print(f"  ERROR: no training data for fold {held_out}", flush=True)
                continue

            merged_images = np.concatenate(train_images)
            merged_labels = np.concatenate(train_labels)

            merged_path = fold_dir / "training_data.npz"
            np.savez_compressed(str(merged_path),
                                images=merged_images, labels=merged_labels)
            print(f"  Merged training data: {len(merged_labels)} samples from "
                  f"{len(train_types)} types", flush=True)

            # Train
            model_dir = fold_dir / "model"
            predictor = train_safety_predictor(
                dataset_path=str(merged_path),
                output_dir=str(model_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                frames_per_episode=100 // args.frame_stride,
                backbone=args.backbone,
                freeze_backbone=args.freeze_backbone,
                head_hidden=args.head_hidden,
                dropout=args.dropout,
                backbone_lr=args.backbone_lr,
                seed=args.seed,
            )
        else:
            predictor = SafetyPredictorCNN()
            model_path = fold_dir / "model" / "best_model.pt"
            predictor.load(str(model_path))
            predictor.to(args.device)

        # Evaluate on held-out type
        eval_budgets = args.budget_levels
        factory = _make_eval_model_factory(held_out, config, image_shape)
        eval_results = _evaluate_occlusion(
            config, policy, image_shape, predictor, fold_dir, args,
            occlusion_type=held_out,
            model_factory=factory,
            budgets=eval_budgets,
            episodes_per_budget=args.eval_episodes,
            label=f"[LOO] {held_out.capitalize()}",
        )

        # Compute Spearman rho for this fold
        actuals, preds = [], []
        for b in eval_budgets:
            r = eval_results[str(b)]
            actuals.append(1 - r["actual_success_rate"])
            preds.append(r["mean_predicted_p_failure"])

        fold_result = {
            "held_out": held_out,
            "train_types": train_types,
            "eval_results": eval_results,
        }

        if len(set(actuals)) > 1:
            boot = bootstrap_spearman_ci(actuals, preds, n_boot=1000)
            fold_result["spearman"] = boot
            print(f"\n  {held_out.upper()} Spearman rho: {boot['rho']:.3f} "
                  f"(p={boot['p_value']:.3f}), "
                  f"95% CI: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]",
                  flush=True)
        else:
            fold_result["spearman"] = {"rho": float("nan"), "note": "no variation"}
            print(f"\n  {held_out.upper()}: no variation in failure rate",
                  flush=True)

        # Episode-level AUROC
        ep_preds, ep_labels = [], []
        for b in eval_budgets:
            for ep in eval_results[str(b)]["episodes"]:
                ep_preds.append(ep["predicted_p_failure_mean"])
                ep_labels.append(ep["actual_failure"])
        if len(set(ep_labels)) > 1:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(ep_labels, ep_preds)
            fold_result["auroc"] = float(auroc)
            print(f"  {held_out.upper()} AUROC: {auroc:.3f}", flush=True)
        else:
            fold_result["auroc"] = float("nan")

        # Save fold results
        with open(fold_dir / "fold_results.json", "w") as f:
            json.dump(fold_result, f, indent=2, default=str)

        fold_results.append(fold_result)

    # --- Aggregate LOO results ---
    _aggregate_loo(fold_results, output_dir)
    return fold_results


def run_diversity_curve(config, policy, image_shape, output_dir, args):
    """Corruption diversity curve: vary training set size and measure OOD rho.

    For each size in {2, 4, 6, 8}, randomly sample that many corruption types
    for training, evaluate on the rest. Repeat `diversity_subsets` times.
    Reports mean rho vs training set size.

    Requires LOO data already collected (run --loo --skip-train first).
    """
    from adversarial_dust.safety_predictor import train_safety_predictor

    loo_types = args.loo_types
    n_types = len(loo_types)
    sizes = [s for s in [2, 4, 6, 8] if s < n_types]
    n_subsets = args.diversity_subsets
    rng = np.random.default_rng(args.seed)

    all_data_dir = output_dir / "loo_data"
    curve_dir = output_dir / "diversity_curve"
    curve_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 60}", flush=True)
    print(f"# CORRUPTION DIVERSITY CURVE", flush=True)
    print(f"# Sizes: {sizes}, Subsets per size: {n_subsets}", flush=True)
    print(f"{'#' * 60}", flush=True)

    curve_results = []

    for size in sizes:
        size_rhos = []

        for sub_idx in range(n_subsets):
            # Random subset of training types
            train_types = list(rng.choice(loo_types, size=size, replace=False))
            test_types = [t for t in loo_types if t not in train_types]

            run_name = f"size{size}_sub{sub_idx}"
            run_dir = curve_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Size={size}, Subset {sub_idx+1}/{n_subsets}: "
                  f"train={train_types}", flush=True)

            # Merge training data
            train_images, train_labels = [], []
            for t in train_types:
                data_path = all_data_dir / f"{t}.npz"
                if not data_path.exists():
                    continue
                d = np.load(str(data_path))
                train_images.append(d["images"])
                train_labels.append(d["labels"])

            if not train_images:
                continue

            merged_path = run_dir / "training_data.npz"
            np.savez_compressed(str(merged_path),
                                images=np.concatenate(train_images),
                                labels=np.concatenate(train_labels))

            # Train
            predictor = train_safety_predictor(
                dataset_path=str(merged_path),
                output_dir=str(run_dir / "model"),
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                frames_per_episode=100 // args.frame_stride,
                backbone=args.backbone,
                freeze_backbone=args.freeze_backbone,
                seed=args.seed + sub_idx,
            )

            # Evaluate on held-out types
            test_rhos = []
            for test_type in test_types:
                factory = _make_eval_model_factory(test_type, config, image_shape)
                eval_results = _evaluate_occlusion(
                    config, policy, image_shape, predictor, run_dir, args,
                    occlusion_type=test_type,
                    model_factory=factory,
                    budgets=args.budget_levels,
                    episodes_per_budget=args.eval_episodes,
                    label=f"[Div] {test_type}",
                )

                actuals, preds = [], []
                for b in args.budget_levels:
                    r = eval_results[str(b)]
                    actuals.append(1 - r["actual_success_rate"])
                    preds.append(r["mean_predicted_p_failure"])

                if len(set(actuals)) > 1:
                    from scipy.stats import spearmanr
                    rho, _ = spearmanr(actuals, preds)
                    test_rhos.append(rho)

            if test_rhos:
                mean_rho = float(np.mean(test_rhos))
                size_rhos.append(mean_rho)
                print(f"    Mean rho on {len(test_types)} held-out types: "
                      f"{mean_rho:.3f}", flush=True)

        curve_results.append({
            "train_size": size,
            "mean_rho": float(np.mean(size_rhos)) if size_rhos else None,
            "std_rho": float(np.std(size_rhos)) if size_rhos else None,
            "per_subset_rhos": [float(r) for r in size_rhos],
        })

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("DIVERSITY CURVE SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  {'Train Size':>12} {'Mean rho':>10} {'Std rho':>10}", flush=True)
    for cr in curve_results:
        m = cr["mean_rho"] or 0
        s = cr["std_rho"] or 0
        print(f"  {cr['train_size']:>12} {m:>10.3f} {s:>10.3f}", flush=True)

    with open(curve_dir / "diversity_curve.json", "w") as f:
        json.dump(curve_results, f, indent=2)
    print(f"\nDiversity curve saved to {curve_dir / 'diversity_curve.json'}",
          flush=True)


def _aggregate_loo(fold_results: list, output_dir: Path):
    """Aggregate leave-one-out results into a summary."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"LOO SUMMARY ({len(fold_results)} folds)", flush=True)
    print(f"{'=' * 60}", flush=True)

    rhos = []
    aurocs = []
    summary_rows = []

    print(f"\n  {'Held-out':>15} {'Spearman rho':>14} {'95% CI':>20} {'AUROC':>8}",
          flush=True)
    print(f"  {'-' * 60}", flush=True)

    for fr in fold_results:
        held_out = fr["held_out"]
        sp = fr.get("spearman", {})
        rho = sp.get("rho", float("nan"))
        ci_lo = sp.get("ci_lower", float("nan"))
        ci_hi = sp.get("ci_upper", float("nan"))
        auroc = fr.get("auroc", float("nan"))

        if not np.isnan(rho):
            rhos.append(rho)
        if not np.isnan(auroc):
            aurocs.append(auroc)

        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not np.isnan(ci_lo) else "N/A"
        auroc_str = f"{auroc:.3f}" if not np.isnan(auroc) else "N/A"
        print(f"  {held_out:>15} {rho:>14.3f} {ci_str:>20} {auroc_str:>8}",
              flush=True)

        summary_rows.append({
            "held_out": held_out,
            "spearman_rho": rho,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "auroc": auroc,
        })

    # Overall statistics
    rhos_arr = np.array(rhos)
    aurocs_arr = np.array(aurocs)

    print(f"\n  {'MEAN':>15} {np.mean(rhos_arr):>14.3f} "
          f"{'':>20} {np.mean(aurocs_arr):>8.3f}", flush=True)
    print(f"  {'STD':>15} {np.std(rhos_arr):>14.3f} "
          f"{'':>20} {np.std(aurocs_arr):>8.3f}", flush=True)

    summary = {
        "n_folds": len(fold_results),
        "per_fold": summary_rows,
        "mean_rho": float(np.mean(rhos_arr)) if len(rhos_arr) > 0 else None,
        "std_rho": float(np.std(rhos_arr)) if len(rhos_arr) > 0 else None,
        "mean_auroc": float(np.mean(aurocs_arr)) if len(aurocs_arr) > 0 else None,
        "std_auroc": float(np.std(aurocs_arr)) if len(aurocs_arr) > 0 else None,
    }

    summary_path = output_dir / "loo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nLOO summary saved to {summary_path}", flush=True)


def _aggregate_multi_seed(all_seed_results: list, output_dir: Path):
    """Aggregate evaluation results across multiple seeds."""
    print(f"\n{'=' * 60}", flush=True)
    print(f"MULTI-SEED AGGREGATION ({len(all_seed_results)} seeds)", flush=True)
    print(f"{'=' * 60}", flush=True)

    # Collect Spearman rho values per occlusion type
    rho_by_type = {}
    for result in all_seed_results:
        for key, val in result.items():
            if key.endswith("_spearman") and isinstance(val, dict):
                occ_type = key.replace("_spearman", "")
                rho_by_type.setdefault(occ_type, []).append(val["rho"])

    summary = {
        "n_seeds": len(all_seed_results),
        "seeds": [r["seed"] for r in all_seed_results],
    }

    for occ_type, rhos in rho_by_type.items():
        rhos_arr = np.array(rhos)
        stats = {
            "mean_rho": float(np.mean(rhos_arr)),
            "std_rho": float(np.std(rhos_arr)),
            "min_rho": float(np.min(rhos_arr)),
            "max_rho": float(np.max(rhos_arr)),
            "per_seed_rhos": [float(r) for r in rhos_arr],
        }
        summary[occ_type] = stats
        print(f"\n{occ_type.upper()} Spearman rho:", flush=True)
        print(f"  Mean: {stats['mean_rho']:.3f} +/- {stats['std_rho']:.3f}",
              flush=True)
        print(f"  Range: [{stats['min_rho']:.3f}, {stats['max_rho']:.3f}]",
              flush=True)
        print(f"  Per-seed: {stats['per_seed_rhos']}", flush=True)

    agg_path = output_dir / "multi_seed_summary.json"
    with open(agg_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAggregated results saved to {agg_path}", flush=True)


if __name__ == "__main__":
    main()
