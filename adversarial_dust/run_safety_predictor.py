"""Full pipeline: collect data, train CNN safety predictor, evaluate on rain.

Orchestrates the three stages on a GPU machine:
  1. Collect training data (fingerprint + glare episodes)
  2. Train SafetyPredictorCNN on collected data
  3. Evaluate on held-out rain occlusion (generalization test)

Usage (on GPU machine):
    python -u adversarial_dust/run_safety_predictor.py \
        --config configs/extreme_fingerprint.yaml \
        --output-dir results/safety_predictor
"""

import argparse
import json
import sys
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
        data = np.load(str(data_path))
        return data["images"], data["labels"]

    print("=" * 60, flush=True)
    print("STAGE 1: Collecting training data", flush=True)
    print("=" * 60, flush=True)

    images, labels, metadata = collect_episodes(
        config=config,
        policy=policy,
        image_shape=image_shape,
        occlusion_types=["fingerprint", "glare"],
        budget_levels=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        episodes_per_condition=args.episodes_per_condition,
        frame_stride=args.frame_stride,
    )

    np.savez_compressed(str(data_path), images=images, labels=labels)

    meta_path = data_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "n_samples": len(labels),
            "n_failures": int(labels.sum()),
            "failure_rate": float(labels.mean()),
            "samples": metadata,
        }, f, indent=2)

    print(f"Saved {len(labels)} samples to {data_path}", flush=True)
    return images, labels


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
    )

    return predictor


def stage_evaluate_rain(config, policy, image_shape, predictor, output_dir, args):
    """Stage 3: Evaluate predictor on held-out rain occlusion."""
    from adversarial_dust.rain_model import RainOcclusionModel
    from adversarial_dust.evaluator import PolicyEvaluator

    print("\n" + "=" * 60, flush=True)
    print("STAGE 3: Evaluating on held-out rain occlusion", flush=True)
    print("=" * 60, flush=True)

    rain_budgets = [0.1, 0.3, 0.5, 0.7, 0.9]
    episodes_per_budget = args.rain_episodes
    rain_results = {}

    for budget in rain_budgets:
        print(f"\nRain budget={budget:.0%}, {episodes_per_budget} episodes",
              flush=True)

        model = RainOcclusionModel(budget_level=budget)
        evaluator = PolicyEvaluator(config.env, policy, model)
        rng = np.random.default_rng(123)

        episode_results = []
        for ep in range(episodes_per_budget):
            params = model.get_random_params(rng)
            t0 = time.time()
            success, _, dirty_frames = evaluator.run_episode(params, record=True)
            elapsed = time.time() - t0

            # Predict P(failure) on sampled frames
            sample_frames = dirty_frames[::args.frame_stride]
            if sample_frames:
                p_failures = predictor.predict_batch(sample_frames)
                mean_p_fail = float(np.mean(p_failures))
                max_p_fail = float(np.max(p_failures))
            else:
                mean_p_fail = max_p_fail = 0.0

            actual_fail = 0 if success else 1
            episode_results.append({
                "episode": ep,
                "actual_success": success,
                "actual_failure": actual_fail,
                "predicted_p_failure_mean": mean_p_fail,
                "predicted_p_failure_max": max_p_fail,
                "n_frames_scored": len(sample_frames),
            })

            print(f"  ep {ep+1}/{episodes_per_budget}: "
                  f"actual={'OK' if success else 'FAIL'}, "
                  f"pred_mean={mean_p_fail:.3f}, pred_max={max_p_fail:.3f}, "
                  f"{elapsed:.1f}s", flush=True)

        # Aggregate per-budget
        actual_sr = np.mean([1 - r["actual_failure"] for r in episode_results])
        mean_pred = np.mean([r["predicted_p_failure_mean"] for r in episode_results])

        rain_results[str(budget)] = {
            "actual_success_rate": float(actual_sr),
            "mean_predicted_p_failure": float(mean_pred),
            "episodes": episode_results,
        }

        print(f"  Budget {budget:.0%}: actual SR={actual_sr:.0%}, "
              f"mean predicted P(fail)={mean_pred:.3f}", flush=True)

    # Save rain evaluation results
    rain_path = output_dir / "rain_evaluation.json"
    with open(rain_path, "w") as f:
        json.dump(rain_results, f, indent=2)
    print(f"\nRain evaluation saved to {rain_path}", flush=True)

    # Print summary table
    print("\n" + "=" * 60, flush=True)
    print("RAIN GENERALIZATION RESULTS", flush=True)
    print("=" * 60, flush=True)
    print(f"{'Budget':>8} {'Actual SR':>10} {'Pred P(fail)':>13} {'Match':>6}",
          flush=True)
    print("-" * 40, flush=True)
    for budget in rain_budgets:
        r = rain_results[str(budget)]
        actual_sr = r["actual_success_rate"]
        pred_pf = r["mean_predicted_p_failure"]
        # "Match" = predictor correctly identifies danger (high pred when low SR)
        match = "Y" if (pred_pf > 0.5) == (actual_sr < 0.5) else "-"
        print(f"{budget:>7.0%} {actual_sr:>10.0%} {pred_pf:>13.3f} {match:>6}",
              flush=True)

    return rain_results


def main():
    parser = argparse.ArgumentParser(
        description="Full CNN safety predictor pipeline"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/safety_predictor")
    parser.add_argument("--device", type=str, default="cuda")

    # Data collection
    parser.add_argument("--episodes-per-condition", type=int, default=5)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--force-recollect", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Rain evaluation
    parser.add_argument("--rain-episodes", type=int, default=10)

    # Stage control
    parser.add_argument("--skip-collect", action="store_true",
                        help="Skip data collection (use existing data)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (use existing model)")
    parser.add_argument("--only-rain", action="store_true",
                        help="Only run rain evaluation (requires trained model)")
    args = parser.parse_args()

    config = load_envelope_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create policy and get image shape
    from adversarial_dust.run_envelope import create_policy, get_image_shape

    print("Getting image shape...", flush=True)
    image_shape = get_image_shape(config)
    print(f"Image shape: {image_shape}", flush=True)

    print(f"Loading {config.env.policy_model} policy...", flush=True)
    policy = create_policy(config)

    # Stage 1: Collect
    if not args.skip_collect and not args.only_rain:
        stage_collect(config, policy, image_shape, output_dir, args)

    # Stage 2: Train
    if not args.skip_train and not args.only_rain:
        predictor = stage_train(output_dir, args)
    else:
        # Load existing model
        from adversarial_dust.safety_predictor import SafetyPredictorCNN
        predictor = SafetyPredictorCNN()
        model_path = output_dir / "model" / "best_model.pt"
        predictor.load(str(model_path))
        predictor.to(args.device)
        print(f"Loaded existing model from {model_path}", flush=True)

    # Stage 3: Rain evaluation
    rain_results = stage_evaluate_rain(
        config, policy, image_shape, predictor, output_dir, args
    )

    print("\nPipeline complete!", flush=True)
    print(f"Results directory: {output_dir}", flush=True)


# Import here to avoid circular issues at module level
from adversarial_dust.config import load_envelope_config

if __name__ == "__main__":
    main()
