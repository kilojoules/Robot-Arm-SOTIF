#!/usr/bin/env python3
"""Stage A: collect the per-corruption frame datasets (the sim-dependent part).

For each corruption type, run InternVLA-M1 episodes under that corruption across
budget levels and save the occluded frames + failure labels:
    <out-dir>/{type}.npz   images[N,H,W,3] uint8, labels[N] (1=failure)
    <out-dir>/{type}.json  per-frame metadata (occlusion_type, budget, episode, success)

This is the ONLY stage that needs SAPIEN rendering + the policy server. It is
deliberately kept torch-free in-process (it talks to the policy over a socket
and renders via SAPIEN) to avoid the torch-CUDA-vs-Vulkan svulkan2 segfault that
broke the monolithic pipeline. Persist <out-dir> off the compute node (it dies
with /tmp/$LSB_JOBID) — that dataset is the durable artifact Stage B reuses.

Usage (inside an LSF GPU job, after the InternVLA server is up on :10093):
    python gbar/stage_a_collect.py \
        --config configs/safety_predictor.yaml \
        --out-dir $WORK/loo_data \
        --episodes-per-condition 10 --frame-stride 10
"""
import argparse
import json
from pathlib import Path

import numpy as np

from adversarial_dust.config import load_envelope_config
from adversarial_dust.collect_training_data import collect_episodes
from adversarial_dust.run_envelope import create_policy, get_image_shape

DEFAULT_TYPES = ["fingerprint", "glare", "rain", "gaussian_noise", "jpeg",
                 "motion_blur", "defocus_blur", "dust_camera", "low_light"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--types", nargs="+", default=DEFAULT_TYPES)
    ap.add_argument("--budget-levels", nargs="+", type=float,
                    default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--episodes-per-condition", type=int, default=10)
    ap.add_argument("--frame-stride", type=int, default=10)
    args = ap.parse_args()

    cfg = load_envelope_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # First render happens here (get_image_shape) — before any torch import.
    image_shape = get_image_shape(cfg)
    print(f"image shape: {image_shape}", flush=True)
    policy = create_policy(cfg)

    for t in args.types:
        out = args.out_dir / f"{t}.npz"
        if out.exists():
            print(f"{t}: exists, skipping", flush=True)
            continue
        print(f"\n=== collecting {t} ===", flush=True)
        images, labels, meta = collect_episodes(
            config=cfg, policy=policy, image_shape=image_shape,
            occlusion_types=[t], budget_levels=args.budget_levels,
            episodes_per_condition=args.episodes_per_condition,
            frame_stride=args.frame_stride,
        )
        np.savez_compressed(str(out), images=images, labels=labels)
        (args.out_dir / f"{t}.json").write_text(json.dumps({
            "occlusion_type": t,
            "n_samples": int(len(labels)),
            "n_failures": int(np.sum(labels)),
            "budget_levels": args.budget_levels,
            "episodes_per_condition": args.episodes_per_condition,
            "frame_stride": args.frame_stride,
            "samples": meta,
        }, indent=2))
        print(f"{t}: {len(labels)} frames, {int(np.sum(labels))} failures -> {out}",
              flush=True)

    print(f"\nStage A complete. Dataset in {args.out_dir} "
          f"({len(args.types)} types). PERSIST THIS off the compute node.")


if __name__ == "__main__":
    main()
