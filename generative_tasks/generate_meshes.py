"""Generate 3D meshes from text prompts using OpenAI's Shap-E."""

import argparse
import logging
from pathlib import Path

import torch
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.diffusion.sample import sample_latents
from shap_e.models.download import load_config, load_model
from shap_e.util.notebooks import decode_latent_mesh

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = {
    "objects": ["a scoop of ice cream in a waffle cone", "a cooked shrimp"],
    "containers": ["a golden chalice", "a small metal bucket"],
}


def generate_mesh(
    prompt: str,
    output_path: Path,
    device: torch.device,
    xm: object,
    diffusion: object,
    model: object,
    guidance_scale: float = 15.0,
    num_steps: int = 64,
    batch_size: int = 1,
):
    """Generate a single 3D mesh from a text prompt and save as .ply."""
    logger.info(f"Generating mesh for: '{prompt}'")

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=num_steps,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    # Take the first latent from the batch
    latent = latents[0]
    t = decode_latent_mesh(xm, latent).tri_mesh()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        t.write_ply(f)

    logger.info(f"Saved mesh to {output_path}")
    return output_path


def load_shap_e_models(device: torch.device):
    """Load all Shap-E models needed for text-to-3D generation."""
    logger.info("Loading Shap-E models...")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))
    logger.info("Models loaded.")
    return xm, model, diffusion


def generate_all(
    output_dir: Path,
    prompts: dict[str, list[str]] | None = None,
    device: torch.device | None = None,
    guidance_scale: float = 15.0,
    num_steps: int = 64,
):
    """Generate meshes for all configured prompts.

    Returns dict mapping category/name to output path.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xm, model, diffusion = load_shap_e_models(device)

    results = {}
    for category, prompt_list in prompts.items():
        for prompt in prompt_list:
            # Derive filename from prompt
            name = prompt.lower().replace(" ", "_")
            # Trim common prefixes for cleaner names
            for prefix in ["a_", "an_", "a_scoop_of_", "a_cooked_", "a_golden_", "a_small_"]:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    break
            out_path = output_dir / category / f"{name}.ply"

            generate_mesh(
                prompt=prompt,
                output_path=out_path,
                device=device,
                xm=xm,
                diffusion=diffusion,
                model=model,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
            )
            results[f"{category}/{name}"] = out_path

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate 3D meshes with Shap-E")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "assets",
        help="Directory to save generated meshes",
    )
    parser.add_argument("--guidance-scale", type=float, default=15.0)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument(
        "--device", type=str, default=None, help="torch device (default: auto)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    results = generate_all(
        output_dir=args.output_dir,
        device=device,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
    )

    print(f"\nGenerated {len(results)} meshes:")
    for key, path in results.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
