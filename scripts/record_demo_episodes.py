#!/usr/bin/env python3
"""Record one real episode per corruption type for paper figures / demo GIF.

Saves full-resolution dirty frames as .npz and a montage image per type.
Run on GPU machine after LOO data collection.

Usage:
    python scripts/record_demo_episodes.py \
        --config configs/safety_predictor.yaml \
        --budget 0.7 \
        --output-dir results/demo_episodes
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Record demo episodes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--budget", type=float, default=0.7)
    parser.add_argument("--corruption-types", type=str, nargs="+",
                        default=["fingerprint", "glare", "rain",
                                 "gaussian_noise", "jpeg", "motion_blur",
                                 "defocus_blur", "fog", "low_light"])
    parser.add_argument("--output-dir", type=str, default="results/demo_episodes")
    parser.add_argument("--also-clean", action="store_true", default=True)
    args = parser.parse_args()

    from adversarial_dust.config import load_envelope_config
    from adversarial_dust.run_envelope import create_policy, get_image_shape
    from adversarial_dust.envelope_predictor import make_occlusion_model
    from adversarial_dust.simpler_env_evaluator import SimplerEnvEvaluator

    config = load_envelope_config(args.config)
    image_shape = get_image_shape(config)
    policy = create_policy(config)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    results = {}

    # Record clean episode first
    if args.also_clean:
        print("Recording CLEAN episode...", flush=True)
        from adversarial_dust.dust_model import AdversarialDustModel
        from adversarial_dust.config import DustGridConfig
        clean_model = AdversarialDustModel(DustGridConfig(), image_shape, 0.0)
        evaluator = SimplerEnvEvaluator(config.env, policy, clean_model)
        success, clean_frames, _ = evaluator.run_episode(None, record=True)
        print(f"  Clean: {'SUCCESS' if success else 'FAIL'}, "
              f"{len(clean_frames)} frames", flush=True)

        # Save frames
        np.savez_compressed(str(out / "clean_frames.npz"),
                            frames=np.array(clean_frames))
        # Save sample montage (every 10th frame)
        _save_montage(clean_frames[::10], out / "clean_montage.png",
                      f"Clean ({'SUCCESS' if success else 'FAIL'})")
        results["clean"] = {"success": success, "n_frames": len(clean_frames)}

    # Record one episode per corruption type
    for ctype in args.corruption_types:
        print(f"\nRecording {ctype} @ budget={args.budget}...", flush=True)
        model = make_occlusion_model(ctype, config, image_shape, args.budget)
        evaluator = SimplerEnvEvaluator(config.env, policy, model)
        params = model.get_random_params(rng)

        t0 = time.time()
        success, clean_frames, dirty_frames = evaluator.run_episode(
            params, record=True)
        elapsed = time.time() - t0

        print(f"  {ctype}: {'SUCCESS' if success else 'FAIL'}, "
              f"{len(dirty_frames)} frames, {elapsed:.1f}s", flush=True)

        # Save full frames
        np.savez_compressed(
            str(out / f"{ctype}_frames.npz"),
            dirty_frames=np.array(dirty_frames),
            clean_frames=np.array(clean_frames),
        )

        # Save montage of every 10th dirty frame
        _save_montage(dirty_frames[::10], out / f"{ctype}_montage.png",
                      f"{ctype} budget={args.budget:.0%} "
                      f"({'SUCCESS' if success else 'FAIL'})")

        # Save a single representative frame (midpoint of episode)
        mid = len(dirty_frames) // 2
        cv2.imwrite(str(out / f"{ctype}_sample.png"),
                    cv2.cvtColor(dirty_frames[mid], cv2.COLOR_RGB2BGR))

        results[ctype] = {
            "success": success,
            "n_frames": len(dirty_frames),
            "budget": args.budget,
            "elapsed": elapsed,
        }

    # Save summary
    with open(out / "demo_episodes.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create a combined sample grid (all corruption types side by side)
    _save_corruption_grid(out, args.corruption_types, args.budget)

    print(f"\nAll demo episodes saved to {out}", flush=True)


def _save_montage(frames, path, title=""):
    """Save a horizontal montage of frames."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    n = min(len(frames), 8)
    canvas = np.zeros((h + 30, w * n, 3), dtype=np.uint8)
    for i in range(n):
        canvas[30:30+h, i*w:(i+1)*w] = frames[i]

    # Title bar
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _save_corruption_grid(out_dir, corruption_types, budget):
    """Create a grid of sample frames from all corruption types."""
    samples = []
    labels = []
    for ctype in ["clean"] + list(corruption_types):
        sample_path = out_dir / f"{ctype}_sample.png"
        if not sample_path.exists():
            continue
        img = cv2.imread(str(sample_path))
        if img is not None:
            # Resize to uniform size
            img = cv2.resize(img, (320, 256))
            samples.append(img)
            labels.append(ctype.replace("_", " ").title())

    if not samples:
        return

    # Arrange in a grid (2 rows)
    n = len(samples)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    h, w = samples[0].shape[:2]
    label_h = 30
    grid = np.zeros((rows * (h + label_h), cols * w, 3), dtype=np.uint8)

    for idx, (img, label) in enumerate(zip(samples, labels)):
        r, c = divmod(idx, cols)
        y0 = r * (h + label_h)
        x0 = c * w
        grid[y0:y0+label_h, x0:x0+w] = 40  # dark gray label bar
        cv2.putText(grid, label, (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        grid[y0+label_h:y0+label_h+h, x0:x0+w] = img

    cv2.imwrite(str(out_dir / "corruption_grid.png"), grid)
    print(f"Saved corruption grid ({rows}x{cols}) to "
          f"{out_dir / 'corruption_grid.png'}", flush=True)


if __name__ == "__main__":
    main()
