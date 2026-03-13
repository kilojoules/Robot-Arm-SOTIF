"""Collect (image, success/failure) training data for the CNN safety predictor.

Runs episodes with fingerprint and glare occlusion at various budget levels,
recording the occluded frames the policy sees. Saves as .npz dataset ready
for train_safety_predictor().

Usage (on GPU machine):
    python -u adversarial_dust/collect_training_data.py \
        --config configs/extreme_fingerprint.yaml \
        --output data/safety_training.npz \
        --episodes-per-condition 5 \
        --frame-stride 10
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from adversarial_dust.config import load_envelope_config
from adversarial_dust.envelope_predictor import make_occlusion_model
from adversarial_dust.evaluator import PolicyEvaluator


def collect_episodes(
    config,
    policy,
    image_shape,
    occlusion_types,
    budget_levels,
    episodes_per_condition,
    frame_stride,
):
    """Run episodes across occlusion types and budgets, collecting frames.

    Args:
        config: EnvelopeExperimentConfig
        policy: Policy inference object
        image_shape: (H, W, 3) from env
        occlusion_types: List of occlusion type names
        budget_levels: List of budget fractions
        episodes_per_condition: Episodes per (type, budget) pair
        frame_stride: Sample every Nth frame from each episode

    Returns:
        images: np.ndarray (N, H, W, 3) uint8
        labels: np.ndarray (N,) binary (1=failure, 0=success)
        metadata: list of dicts with per-sample info
    """
    all_images = []
    all_labels = []
    all_meta = []

    total_conditions = len(occlusion_types) * len(budget_levels)
    condition_idx = 0

    for occ_type in occlusion_types:
        for budget in budget_levels:
            condition_idx += 1
            print(f"\n[{condition_idx}/{total_conditions}] "
                  f"{occ_type} budget={budget:.0%}, "
                  f"{episodes_per_condition} episodes", flush=True)

            model = make_occlusion_model(occ_type, config, image_shape, budget)
            evaluator = PolicyEvaluator(config.env, policy, model)
            rng = np.random.default_rng(42 + condition_idx)

            for ep in range(episodes_per_condition):
                params = model.get_random_params(rng)
                t0 = time.time()
                success, clean_frames, dirty_frames = evaluator.run_episode(
                    params, record=True
                )
                elapsed = time.time() - t0
                label = 0 if success else 1

                # Sample frames at stride
                sampled = dirty_frames[::frame_stride]
                n_sampled = len(sampled)

                for frame in sampled:
                    all_images.append(frame)
                    all_labels.append(label)
                    all_meta.append({
                        "occlusion_type": occ_type,
                        "budget": budget,
                        "episode": ep,
                        "success": success,
                    })

                print(f"  ep {ep+1}/{episodes_per_condition}: "
                      f"{'OK' if success else 'FAIL'}, "
                      f"{len(dirty_frames)} frames -> {n_sampled} samples, "
                      f"{elapsed:.1f}s", flush=True)

    # Also collect clean baseline episodes
    print(f"\n[clean baseline] {episodes_per_condition} episodes", flush=True)
    from adversarial_dust.dust_model import AdversarialDustModel
    clean_model = AdversarialDustModel(config.dust, image_shape, budget_level=0.0)
    evaluator = PolicyEvaluator(config.env, policy, clean_model)

    for ep in range(episodes_per_condition):
        t0 = time.time()
        success, clean_frames, dirty_frames = evaluator.run_episode(
            None, record=True
        )
        elapsed = time.time() - t0
        label = 0 if success else 1

        sampled = dirty_frames[::frame_stride]
        for frame in sampled:
            all_images.append(frame)
            all_labels.append(label)
            all_meta.append({
                "occlusion_type": "clean",
                "budget": 0.0,
                "episode": ep,
                "success": success,
            })

        print(f"  ep {ep+1}/{episodes_per_condition}: "
              f"{'OK' if success else 'FAIL'}, "
              f"{len(dirty_frames)} frames -> {len(sampled)} samples, "
              f"{elapsed:.1f}s", flush=True)

    images = np.array(all_images)
    labels = np.array(all_labels, dtype=np.float32)
    return images, labels, all_meta


def main():
    parser = argparse.ArgumentParser(
        description="Collect training data for CNN safety predictor"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to envelope config YAML")
    parser.add_argument("--output", type=str, default="data/safety_training.npz",
                        help="Output .npz path")
    parser.add_argument("--episodes-per-condition", type=int, default=5,
                        help="Episodes per (occlusion_type, budget) pair")
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="Sample every Nth frame from each episode")
    parser.add_argument("--occlusion-types", type=str, nargs="+",
                        default=["fingerprint", "glare"],
                        help="Occlusion types to include")
    parser.add_argument("--budget-levels", type=float, nargs="+",
                        default=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
                        help="Budget levels to sweep")
    args = parser.parse_args()

    config = load_envelope_config(args.config)

    # Import and create policy (same pattern as run_envelope.py)
    from adversarial_dust.run_envelope import create_policy, get_image_shape

    print("Getting image shape...", flush=True)
    image_shape = get_image_shape(config)
    print(f"Image shape: {image_shape}", flush=True)

    print(f"Loading {config.env.policy_model} policy...", flush=True)
    policy = create_policy(config)

    print(f"\nCollection plan:", flush=True)
    print(f"  Occlusion types: {args.occlusion_types}", flush=True)
    print(f"  Budget levels: {args.budget_levels}", flush=True)
    print(f"  Episodes/condition: {args.episodes_per_condition}", flush=True)
    print(f"  Frame stride: {args.frame_stride}", flush=True)
    n_conditions = len(args.occlusion_types) * len(args.budget_levels) + 1  # +1 for clean
    total_eps = n_conditions * args.episodes_per_condition
    print(f"  Total episodes: ~{total_eps}", flush=True)

    images, labels, metadata = collect_episodes(
        config=config,
        policy=policy,
        image_shape=image_shape,
        occlusion_types=args.occlusion_types,
        budget_levels=args.budget_levels,
        episodes_per_condition=args.episodes_per_condition,
        frame_stride=args.frame_stride,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        images=images,
        labels=labels,
    )

    # Also save metadata as JSON
    import json
    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "n_samples": len(labels),
            "n_failures": int(labels.sum()),
            "failure_rate": float(labels.mean()),
            "occlusion_types": args.occlusion_types,
            "budget_levels": args.budget_levels,
            "episodes_per_condition": args.episodes_per_condition,
            "frame_stride": args.frame_stride,
            "samples": metadata,
        }, f, indent=2)

    print(f"\nDataset saved to {out_path}", flush=True)
    print(f"  {len(labels)} samples, {int(labels.sum())} failures "
          f"({labels.mean():.1%} failure rate)", flush=True)
    print(f"  Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
