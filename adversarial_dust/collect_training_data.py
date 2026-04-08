"""Collect (image, success/failure) training data for the CNN safety predictor.

Runs episodes with fingerprint and glare occlusion at various budget levels,
recording the occluded frames the policy sees. Uses both random and adversarial
occlusion params (from prior CMA-ES sweeps) to ensure sufficient failure examples.

Saves as .npz dataset ready for train_safety_predictor().

Usage (on GPU machine):
    python -u adversarial_dust/collect_training_data.py \
        --config configs/safety_predictor.yaml \
        --output data/safety_training.npz \
        --episodes-per-condition 5 \
        --frame-stride 10
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from adversarial_dust.config import load_envelope_config
from adversarial_dust.envelope_predictor import make_occlusion_model
from adversarial_dust.simpler_env_evaluator import SimplerEnvEvaluator


def _load_adversarial_params(results_dir: str, occlusion_type: str) -> dict:
    """Load adversarial params from a prior envelope sweep.

    Returns dict mapping budget_str -> np.ndarray of params.
    """
    results_path = Path(results_dir) / "envelope_results.json"
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        results = json.load(f)
    params = {}
    occ_results = results.get("occlusion_results", {}).get(occlusion_type, {})
    for budget_str, data in occ_results.items():
        if "adversarial_params" in data:
            params[budget_str] = np.array(data["adversarial_params"])
    return params


def collect_episodes(
    config,
    policy,
    image_shape,
    occlusion_types,
    budget_levels,
    episodes_per_condition,
    frame_stride,
    adversarial_results_dirs=None,
    adversarial_episodes=3,
):
    """Run episodes across occlusion types and budgets, collecting frames.

    Args:
        config: EnvelopeExperimentConfig.
        policy: Policy inference object.
        image_shape: (H, W, 3) from env.
        occlusion_types: List of occlusion type names.
        budget_levels: List of budget fractions (must not include 0.0 —
                      clean baseline is collected separately).
        episodes_per_condition: Episodes with random params per condition.
        frame_stride: Sample every Nth frame from each episode.
        adversarial_results_dirs: Dict mapping occlusion_type -> results dir
                                 containing envelope_results.json with
                                 adversarial params from prior CMA-ES sweeps.
        adversarial_episodes: Episodes to run per adversarial param set.

    Returns:
        images: np.ndarray (N, H, W, 3) uint8
        labels: np.ndarray (N,) binary (1=failure, 0=success)
        metadata: list of dicts with per-sample info
    """
    if adversarial_results_dirs is None:
        adversarial_results_dirs = {}

    all_images = []
    all_labels = []
    all_meta = []

    # Load adversarial params from prior sweeps
    adv_params_by_type = {}
    for occ_type in occlusion_types:
        if occ_type in adversarial_results_dirs:
            adv_params_by_type[occ_type] = _load_adversarial_params(
                adversarial_results_dirs[occ_type], occ_type
            )
            n_loaded = len(adv_params_by_type[occ_type])
            if n_loaded:
                print(f"Loaded adversarial params for {occ_type}: "
                      f"{n_loaded} budget levels", flush=True)

    # Filter out budget 0.0 — clean baseline is separate
    active_budgets = [b for b in budget_levels if b > 0.0]

    total_conditions = len(occlusion_types) * len(active_budgets)
    condition_idx = 0

    def _run_and_record(evaluator, model, params, occ_type, budget, ep_idx,
                        param_source):
        """Run one episode and append results."""
        t0 = time.time()
        success, _, dirty_frames = evaluator.run_episode(params, record=True)
        elapsed = time.time() - t0
        label = 0 if success else 1

        sampled = dirty_frames[::frame_stride]
        for frame in sampled:
            all_images.append(frame)
            all_labels.append(label)
            all_meta.append({
                "occlusion_type": occ_type,
                "budget": budget,
                "episode": ep_idx,
                "success": bool(success),
                "param_source": param_source,
            })

        print(f"  ep {ep_idx+1}: "
              f"{'OK' if success else 'FAIL'} [{param_source}], "
              f"{len(dirty_frames)} frames -> {len(sampled)} samples, "
              f"{elapsed:.1f}s", flush=True)

    for occ_type in occlusion_types:
        adv_params = adv_params_by_type.get(occ_type, {})

        for budget in active_budgets:
            condition_idx += 1
            budget_str = str(budget)

            n_adv = adversarial_episodes if budget_str in adv_params else 0
            n_rand = episodes_per_condition
            print(f"\n[{condition_idx}/{total_conditions}] "
                  f"{occ_type} budget={budget:.0%}: "
                  f"{n_rand} random + {n_adv} adversarial episodes",
                  flush=True)

            model = make_occlusion_model(occ_type, config, image_shape, budget)
            evaluator = SimplerEnvEvaluator(config.env, policy, model)
            rng = np.random.default_rng(42 + condition_idx)

            ep_idx = 0

            # Random params
            for _ in range(n_rand):
                params = model.get_random_params(rng)
                _run_and_record(evaluator, model, params, occ_type, budget,
                                ep_idx, "random")
                ep_idx += 1

            # Adversarial params (from prior CMA-ES sweep)
            if budget_str in adv_params:
                for _ in range(n_adv):
                    _run_and_record(evaluator, model, adv_params[budget_str],
                                    occ_type, budget, ep_idx, "adversarial")
                    ep_idx += 1

    # Clean baseline
    n_clean = episodes_per_condition
    print(f"\n[clean baseline] {n_clean} episodes", flush=True)
    from adversarial_dust.dust_model import AdversarialDustModel
    clean_model = AdversarialDustModel(config.dust, image_shape, budget_level=0.0)
    evaluator = SimplerEnvEvaluator(config.env, policy, clean_model)

    for ep in range(n_clean):
        t0 = time.time()
        success, _, dirty_frames = evaluator.run_episode(None, record=True)
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
                "success": bool(success),
                "param_source": "none",
            })

        print(f"  ep {ep+1}/{n_clean}: "
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
                        help="Random-param episodes per (type, budget) pair")
    parser.add_argument("--adversarial-episodes", type=int, default=3,
                        help="Adversarial-param episodes per condition")
    parser.add_argument("--frame-stride", type=int, default=10,
                        help="Sample every Nth frame from each episode")
    parser.add_argument("--occlusion-types", type=str, nargs="+",
                        default=["fingerprint", "glare"],
                        help="Occlusion types to include")
    parser.add_argument("--budget-levels", type=float, nargs="+",
                        default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help="Budget levels to sweep (0.0 excluded, "
                             "clean baseline collected separately)")
    parser.add_argument("--adv-results", type=str, nargs="*", default=None,
                        help="Pairs of occlusion_type:results_dir for "
                             "adversarial params, e.g. "
                             "fingerprint:results/coke_can_fingerprint")
    args = parser.parse_args()

    # Parse adversarial results dirs
    adv_dirs = {}
    if args.adv_results:
        for pair in args.adv_results:
            occ_type, results_dir = pair.split(":", 1)
            adv_dirs[occ_type] = results_dir

    config = load_envelope_config(args.config)

    from adversarial_dust.run_envelope import create_policy, get_image_shape

    print("Getting image shape...", flush=True)
    image_shape = get_image_shape(config)
    print(f"Image shape: {image_shape}", flush=True)

    print(f"Loading {config.env.policy_model} policy...", flush=True)
    policy = create_policy(config)

    n_conditions = len(args.occlusion_types) * len(args.budget_levels) + 1
    total_eps = (n_conditions * args.episodes_per_condition
                 + len(adv_dirs) * len(args.budget_levels) * args.adversarial_episodes)
    print(f"\nCollection plan:", flush=True)
    print(f"  Occlusion types: {args.occlusion_types}", flush=True)
    print(f"  Budget levels: {args.budget_levels}", flush=True)
    print(f"  Random episodes/condition: {args.episodes_per_condition}", flush=True)
    print(f"  Adversarial episodes/condition: {args.adversarial_episodes}", flush=True)
    print(f"  Frame stride: {args.frame_stride}", flush=True)
    print(f"  Adversarial params from: {adv_dirs or 'none'}", flush=True)
    print(f"  Estimated total episodes: ~{total_eps}", flush=True)

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

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), images=images, labels=labels)

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "n_samples": len(labels),
            "n_failures": int(labels.sum()),
            "failure_rate": float(labels.mean()),
            "occlusion_types": args.occlusion_types,
            "budget_levels": args.budget_levels,
            "episodes_per_condition": args.episodes_per_condition,
            "adversarial_episodes": args.adversarial_episodes,
            "frame_stride": args.frame_stride,
            "adversarial_results_dirs": adv_dirs,
            "samples": metadata,
        }, f, indent=2)

    print(f"\nDataset saved to {out_path}", flush=True)
    print(f"  {len(labels)} samples, {int(labels.sum())} failures "
          f"({labels.mean():.1%} failure rate)", flush=True)
    print(f"  Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
