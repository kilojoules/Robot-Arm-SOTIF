#!/usr/bin/env python3
"""End-to-end pipeline: generate 3D meshes, convert to URDF, build task matrix.

Usage:
    python run_pipeline.py                     # Full pipeline
    python run_pipeline.py --skip-generation   # Only URDF conversion + task matrix
    python run_pipeline.py --tasks-only        # Only show task matrix from existing URDFs
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


def step_generate(args):
    """Step 1: Generate 3D meshes with Shap-E."""
    from generate_meshes import generate_all

    import torch

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    results = generate_all(
        output_dir=args.assets_dir,
        device=device,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
    )
    print(f"\n=== Step 1: Generated {len(results)} meshes ===")
    for key, path in results.items():
        print(f"  {key}: {path}")
    return results


def step_convert(args):
    """Step 2: Convert .ply meshes to URDF."""
    from mesh_to_urdf import convert_all

    results = convert_all(args.assets_dir, args.urdf_dir)
    print(f"\n=== Step 2: Converted {len(results)} meshes to URDF ===")
    for key, path in results.items():
        print(f"  {key}: {path}")
    return results


def step_tasks(args):
    """Step 3: Build task matrix and export task specs."""
    from task_factory import TaskFactory

    factory = TaskFactory(urdf_dir=args.urdf_dir)

    print("\n=== Step 3: Task Matrix ===")
    factory.print_task_matrix()

    tasks = factory.generate_all_tasks()

    # Export task specs to JSON
    specs_path = BASE_DIR / "task_specs.json"
    specs = [t.to_dict() for t in tasks]
    with open(specs_path, "w") as f:
        json.dump(specs, f, indent=2)
    print(f"\nExported {len(tasks)} task specs to {specs_path}")

    # Print language instructions
    print("\nLanguage instructions:")
    for task in tasks:
        print(f"  {task.task_id}: '{task.language_instruction}'")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D assets and build manipulation task matrix"
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=BASE_DIR / "assets",
        help="Directory for raw generated meshes (.ply)",
    )
    parser.add_argument(
        "--urdf-dir",
        type=Path,
        default=BASE_DIR / "urdf_assets",
        help="Directory for converted URDF bundles",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip mesh generation (use existing .ply files)",
    )
    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Only build task matrix from existing URDFs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.tasks_only:
        step_tasks(args)
        return

    if not args.skip_generation:
        step_generate(args)

    step_convert(args)
    step_tasks(args)

    print("\n=== Pipeline complete ===")
    print(f"  Raw meshes:  {args.assets_dir}")
    print(f"  URDF assets: {args.urdf_dir}")
    print(f"  Task specs:  {BASE_DIR / 'task_specs.json'}")


if __name__ == "__main__":
    main()
