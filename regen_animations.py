"""Regenerate animations from existing sweep_results.json with improved dust visualization.

Usage: python regen_animations.py --config configs/medium.yaml
"""

import argparse
import json
import logging

import numpy as np

from adversarial_dust.config import load_config
from adversarial_dust.animation import record_all_budget_animations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Regenerate animations from existing results")
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument("--video-fmt", default="mp4", choices=["mp4", "gif"])
    args = parser.parse_args()

    config = load_config(args.config)
    results_path = f"{config.output_dir}/sweep_results.json"
    logger.info(f"Loading results from {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    # Get image shape from environment
    import simpler_env
    env = simpler_env.make(config.env.task_name)
    obs, _ = env.reset()
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
    image = get_image_from_maniskill2_obs_dict(env, obs)
    image_shape = image.shape
    logger.info(f"Image shape: {image_shape}")

    # Load Octo policy
    logger.info("Loading Octo policy...")
    from simpler_env.policies.octo.octo_model import OctoInference
    policy = OctoInference(model_type=config.env.policy_model, policy_setup="google_robot")

    logger.info("Regenerating animations with improved dust visualization...")
    record_all_budget_animations(config, policy, image_shape, results, config.output_dir, fmt=args.video_fmt)
    logger.info("Done!")


if __name__ == "__main__":
    main()
