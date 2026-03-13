#!/usr/bin/env python3
"""Re-record adversarial animations from a saved generator checkpoint.

Usage:
    python scripts/rerecord_adversarial_animation.py \
        --checkpoint results/adversarial_move_near/generator_checkpoint.pt \
        --config configs/adversarial_move_near.yaml \
        --output results/adversarial_move_near/animations/rerecord_4blobs.mp4 \
        --budget 4
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Re-record adversarial animation from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator_checkpoint.pt")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, required=True, help="Output MP4 path")
    parser.add_argument("--budget", type=int, default=4, help="Number of blobs to render")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from adversarial_dust.config import load_adversarial_config
    from adversarial_dust.blob_generator import BlobGenerator
    from adversarial_dust.animation import record_adversarial_episode

    config = load_adversarial_config(args.config)

    # Load Octo policy
    logger.info("Loading Octo policy...")
    from simpler_env.policies.octo.octo_model import OctoInference

    task_name = config.env.task_name
    policy_setup = "widowx_bridge" if task_name.startswith("widowx") else "google_robot"
    policy = OctoInference(model_type=config.env.policy_model, policy_setup=policy_setup)

    # Get reference image
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    env = simpler_env.make(config.env.task_name, **config.env.make_kwargs())
    obs, _ = env.reset()
    reference_image = get_image_from_maniskill2_obs_dict(env, obs)
    env.close()
    image_shape = reference_image.shape
    logger.info(f"Image shape: {image_shape}")

    # Load generator from checkpoint
    logger.info(f"Loading generator from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    generator = BlobGenerator(config.blob_generator, image_shape)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator = generator.to(args.device)
    generator.eval()

    # Generate alpha mask
    ref_torch = (
        torch.from_numpy(reference_image).float() / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(args.device)

    with torch.no_grad():
        mask = generator.generate_mask(ref_torch, k=args.budget)
    alpha_mask = mask.squeeze(1).squeeze(0).cpu().numpy()  # (H, W)

    logger.info(f"Alpha mask stats: min={alpha_mask.min():.4f}, max={alpha_mask.max():.4f}, "
                f"mean={alpha_mask.mean():.4f}, nonzero={np.count_nonzero(alpha_mask > 0.01)}")

    # Record animation with 3-panel view (clean | policy view | dust highlighted)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    success = record_adversarial_episode(
        config.env,
        policy,
        alpha_mask,
        args.output,
        budget_label=f"{args.budget} blobs",
        dust_color=config.blob_generator.dust_color,
        two_panel=True,
    )
    logger.info(f"Done! success={success}, saved to {args.output}")


if __name__ == "__main__":
    main()
