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
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


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
    )

    return predictor


def _evaluate_occlusion(config, policy, image_shape, predictor, output_dir,
                         args, occlusion_type, model_factory, budgets,
                         episodes_per_budget, label):
    """Run evaluation episodes with a given occlusion type and score with CNN.

    Returns dict mapping budget_str -> results.
    """
    from adversarial_dust.evaluator import PolicyEvaluator

    results = {}

    for budget in budgets:
        print(f"\n{label} budget={budget:.0%}, "
              f"{episodes_per_budget} episodes", flush=True)

        model = model_factory(budget)
        evaluator = PolicyEvaluator(config.env, policy, model)
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

            episode_results.append({
                "episode": ep,
                "actual_success": bool(success),
                "actual_failure": 0 if success else 1,
                "predicted_p_failure_mean": mean_p_fail,
                "predicted_p_failure_max": max_p_fail,
                "n_frames_scored": len(sample_frames),
            })

            print(f"  ep {ep+1}/{episodes_per_budget}: "
                  f"actual={'OK' if success else 'FAIL'}, "
                  f"pred_mean={mean_p_fail:.3f}, pred_max={max_p_fail:.3f}, "
                  f"{elapsed:.1f}s", flush=True)

        actual_sr = np.mean([1 - r["actual_failure"] for r in episode_results])
        mean_pred = np.mean([r["predicted_p_failure_mean"] for r in episode_results])

        results[str(budget)] = {
            "actual_success_rate": float(actual_sr),
            "mean_predicted_p_failure": float(mean_pred),
            "episodes": episode_results,
        }

        print(f"  Budget {budget:.0%}: actual SR={actual_sr:.0%}, "
              f"mean predicted P(fail)={mean_pred:.3f}", flush=True)

    return results


def stage_evaluate(config, policy, image_shape, predictor, output_dir, args):
    """Stage 3: Evaluate predictor on rain (held-out) and fingerprint (sanity)."""
    from adversarial_dust.rain_model import RainOcclusionModel
    from adversarial_dust.envelope_predictor import make_occlusion_model

    eval_budgets = [0.1, 0.3, 0.5, 0.7, 0.9]
    all_results = {}

    # --- Rain evaluation (held-out occlusion type) ---
    print("\n" + "=" * 60, flush=True)
    print("STAGE 3a: Evaluating on held-out RAIN occlusion", flush=True)
    print("=" * 60, flush=True)

    rain_results = _evaluate_occlusion(
        config, policy, image_shape, predictor, output_dir, args,
        occlusion_type="rain",
        model_factory=lambda budget: RainOcclusionModel(budget_level=budget),
        budgets=eval_budgets,
        episodes_per_budget=args.eval_episodes,
        label="Rain",
    )
    all_results["rain"] = rain_results

    # --- Fingerprint evaluation (sanity check on training distribution) ---
    print("\n" + "=" * 60, flush=True)
    print("STAGE 3b: Evaluating on FINGERPRINT (training distribution)", flush=True)
    print("=" * 60, flush=True)

    fp_results = _evaluate_occlusion(
        config, policy, image_shape, predictor, output_dir, args,
        occlusion_type="fingerprint",
        model_factory=lambda budget: make_occlusion_model(
            "fingerprint", config, image_shape, budget),
        budgets=eval_budgets,
        episodes_per_budget=args.eval_episodes,
        label="Fingerprint",
    )
    all_results["fingerprint"] = fp_results

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
            from scipy.stats import spearmanr
            corr, pval = spearmanr(actuals, preds)
            print(f"  Spearman correlation: {corr:.3f} (p={pval:.3f})", flush=True)
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

    # Data collection
    parser.add_argument("--occlusion-types", type=str, nargs="+",
                        default=["fingerprint", "glare"])
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

    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=10)

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

    # Stage 1: Collect
    if not args.skip_collect and not args.only_eval:
        stage_collect(config, policy, image_shape, output_dir, args)

    # Stage 2: Train
    if not args.skip_train and not args.only_eval:
        predictor = stage_train(output_dir, args)
    else:
        from adversarial_dust.safety_predictor import SafetyPredictorCNN
        predictor = SafetyPredictorCNN()
        model_path = output_dir / "model" / "best_model.pt"
        predictor.load(str(model_path))
        predictor.to(args.device)
        print(f"Loaded existing model from {model_path}", flush=True)

    # Stage 3: Evaluate
    stage_evaluate(config, policy, image_shape, predictor, output_dir, args)

    print("\nPipeline complete!", flush=True)
    print(f"Results directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
