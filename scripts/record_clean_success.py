#!/usr/bin/env python3
"""Record an animation of the baseline Octo policy succeeding on a clean
(no-dust) episode of google_robot_pick_coke_can.

Runs episodes until a success with real arm movement is found, then saves
an MP4 to results/clean_baseline_success.mp4.

Usage:
    python scripts/record_clean_success.py [--max-attempts 100]
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Record a clean baseline success animation")
    parser.add_argument("--task", type=str, default="google_robot_move_near", help="SimplerEnv task name")
    parser.add_argument("--max-attempts", type=int, default=100, help="Max episodes to try")
    parser.add_argument("--output", type=str, default="results/clean_baseline_success.mp4")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    args = parser.parse_args()

    import simpler_env
    from simpler_env.policies.octo.octo_model import OctoInference
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

    # --- Load environment (use SimplerEnv defaults for freq) ---
    logger.info(f"Creating environment: {args.task}")
    env = simpler_env.make(
        args.task,
        max_episode_steps=args.max_episode_steps,
    )

    logger.info("Loading baseline Octo policy (octo-base)...")
    policy = OctoInference(model_type="octo-base", policy_setup="google_robot")

    # --- Run episodes until a real success ---
    success_frames = None
    for attempt in range(1, args.max_attempts + 1):
        obs, _ = env.reset()
        instruction = env.get_language_instruction()
        policy.reset(instruction)

        frames = []
        all_actions = []
        truncated = False
        step_count = 0
        info = {}

        while not truncated and step_count < args.max_episode_steps:
            image = get_image_from_maniskill2_obs_dict(env, obs)
            frames.append(image.copy())

            raw_action, action = policy.step(image, instruction)
            action_array = np.concatenate([
                action["world_vector"],
                action["rot_axangle"],
                action["gripper"].flatten(),
            ])
            all_actions.append(action_array.copy())

            obs, reward, done, truncated, info = env.step(action_array)
            step_count += 1

            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction
                policy.reset(instruction)

        # Only check success from the FINAL step's info dict
        success = info.get("success", False)

        # Compute action statistics
        actions_np = np.stack(all_actions)
        mean_norm = np.mean(np.linalg.norm(actions_np[:, :3], axis=1))  # xyz displacement
        max_norm = np.max(np.linalg.norm(actions_np[:, :3], axis=1))
        total_displacement = np.linalg.norm(np.sum(actions_np[:, :3], axis=0))

        status = "SUCCESS" if success else "FAIL"
        logger.info(
            f"  Attempt {attempt}/{args.max_attempts}: {status} "
            f"({step_count} steps, "
            f"mean_action_norm={mean_norm:.4f}, "
            f"max_action_norm={max_norm:.4f}, "
            f"total_displacement={total_displacement:.4f})"
        )

        # Log info dict details
        for k in ["consecutive_grasp", "lifted_object", "lifted_object_significantly", "success"]:
            if k in info:
                logger.info(f"    info[{k}] = {info[k]}")

        # Accept success only if the arm actually moved
        if success and mean_norm > 0.001:
            success_frames = frames
            logger.info(f"  -> Accepted! Arm moved meaningfully.")
            break
        elif success:
            logger.info(f"  -> Rejected: success=True but arm barely moved (mean_norm={mean_norm:.6f})")

    env.close()

    if success_frames is None:
        logger.error(f"No real successful episode found in {args.max_attempts} attempts.")
        sys.exit(1)

    # --- Write video ---
    logger.info(f"Recording {len(success_frames)} frames to {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    h, w = success_frames[0].shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h + 30))

    for i, frame in enumerate(success_frames):
        canvas = np.full((h + 30, w, 3), 255, dtype=np.uint8)
        canvas[30:30 + h, :w] = frame
        label = f"Clean Baseline (step {i+1}/{len(success_frames)})"
        cv2.putText(canvas, label, (10, 22), font, 0.6, (0, 128, 0), 2)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    logger.info(f"Saved: {args.output} ({len(success_frames)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
