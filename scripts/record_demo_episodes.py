#!/usr/bin/env python3
"""Record real episodes per corruption type for paper figures and demo animations.

For each corruption type (+ clean), records 10 episodes where the policy
sees the corrupted frames. Saves:
  - Per-type GIF animation with green (success) / red (failure) borders
  - Per-type sample montage PNG
  - Corruption grid of sample frames
  - Summary JSON with success rates

Usage (on GPU machine):
    python scripts/record_demo_episodes.py \
        --config configs/safety_predictor.yaml \
        --budget 0.7 \
        --episodes 10
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_font(size=14):
    for path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def add_border(frame_pil, color, thickness=4):
    """Add colored border to a PIL image."""
    w, h = frame_pil.size
    draw = ImageDraw.Draw(frame_pil)
    for t in range(thickness):
        draw.rectangle([(t, t), (w - 1 - t, h - 1 - t)], outline=color)
    return frame_pil


def add_label(frame_pil, text, bg_color, position="top-right"):
    """Add text label on the frame."""
    draw = ImageDraw.Draw(frame_pil)
    font = get_font(12)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    w, h = frame_pil.size
    pad = 3
    if position == "top-right":
        x, y = w - tw - pad - 6, pad
    else:
        x, y = pad + 2, pad
    draw.rectangle([(x - pad, y - 1), (x + tw + pad, y + th + pad)], fill=bg_color)
    draw.text((x, y), text, fill="white", font=font)
    return frame_pil


def record_episodes(config, policy, image_shape, corruption_type, budget,
                    n_episodes, rng):
    """Record n_episodes with a given corruption type. Returns list of
    (success, dirty_frames) tuples."""
    from adversarial_dust.envelope_predictor import make_occlusion_model
    from adversarial_dust.simpler_env_evaluator import SimplerEnvEvaluator

    if corruption_type == "clean":
        from adversarial_dust.dust_model import AdversarialDustModel
        from adversarial_dust.config import DustGridConfig
        model = AdversarialDustModel(DustGridConfig(), image_shape, 0.0)
    else:
        model = make_occlusion_model(corruption_type, config, image_shape, budget)

    evaluator = SimplerEnvEvaluator(config.env, policy, model)
    episodes = []

    for ep in range(n_episodes):
        if corruption_type == "clean":
            params = None
        else:
            params = model.get_random_params(rng)
        t0 = time.time()
        success, _, dirty_frames = evaluator.run_episode(params, record=True)
        success_step = getattr(evaluator, "last_success_step", -1)
        elapsed = time.time() - t0
        episodes.append((success, dirty_frames, success_step))
        step_info = f", lifted at step {success_step}" if success_step > 0 else ""
        print(f"    ep {ep+1}/{n_episodes}: "
              f"{'SUCCESS' if success else 'FAIL'} ({elapsed:.1f}s, "
              f"{len(dirty_frames)} frames{step_info})", flush=True)

    return episodes


def make_episode_gif(episodes, output_path, label, fps=5, frame_size=(220, 176)):
    """Create GIF showing all episodes with green/red borders.

    Green border appears at the actual timestep the can is lifted
    (from env success signal), not at a fixed fraction.
    Red border appears at episode end for failures.
    """
    pw, ph = frame_size
    n_eps = len(episodes)
    gap = 2
    border_t = 4
    label_h = 24

    # Layout: episodes side by side (max 5 per row)
    cols = min(5, n_eps)
    rows = (n_eps + cols - 1) // cols
    col_w = pw + border_t * 2
    row_h = ph + border_t * 2 + label_h
    total_w = cols * col_w + (cols - 1) * gap
    total_h = rows * row_h + (rows - 1) * gap + 30  # 30 for title

    # Find max frames across episodes
    max_frames = max(len(frames) for _, frames, _ in episodes)
    stride = max(1, max_frames // 40)
    frame_indices = list(range(0, max_frames, stride))

    GREEN = "#2e7d32"
    RED = "#c62828"

    gif_frames = []
    for fi in frame_indices:
        canvas = Image.new("RGB", (total_w, total_h), "#1a1a2e")
        draw = ImageDraw.Draw(canvas)

        # Title
        font = get_font(13)
        sr = sum(1 for s, _, _ in episodes if s) / n_eps
        title = f"{label} — {sr:.0%} success rate"
        draw.text((8, 6), title, fill="white", font=font)

        for ep_idx, (success, frames, success_step) in enumerate(episodes):
            r, c = divmod(ep_idx, cols)
            x0 = c * (col_w + gap)
            y0 = r * (row_h + gap) + 30

            # Get frame (clamp to episode length)
            src_i = min(fi, len(frames) - 1)
            frame_np = frames[src_i]
            panel = Image.fromarray(frame_np).resize((pw, ph), Image.LANCZOS)

            ep_label = f"Ep {ep_idx+1}"
            color = GREEN if success else RED

            # Green border at actual success step, red at episode end
            if success and success_step > 0 and fi >= success_step:
                panel = add_border(panel, color, border_t)
                panel = add_label(panel, "LIFTED", GREEN)
            elif not success and fi >= max_frames - 5:
                panel = add_border(panel, color, border_t)
                panel = add_label(panel, "FAIL", RED)
            else:
                panel = add_border(panel, "#37474f", border_t)

            canvas.paste(panel, (x0, y0 + label_h))

            # Episode number label
            is_active = (success and success_step > 0 and fi >= success_step) or \
                        (not success and fi >= max_frames - 5)
            ep_color = color if is_active else "#9e9e9e"
            draw_canvas = ImageDraw.Draw(canvas)
            small_font = get_font(10)
            draw_canvas.text((x0 + 4, y0 + 4), ep_label, fill=ep_color,
                             font=small_font)

        gif_frames.append(canvas)

    # Hold last frame
    for _ in range(10):
        gif_frames.append(gif_frames[-1])

    # Save GIF
    gif_frames[0].save(
        str(output_path), save_all=True, append_images=gif_frames[1:],
        duration=200, loop=0, optimize=True)
    size_kb = output_path.stat().st_size / 1024
    print(f"  Saved GIF: {output_path} ({size_kb:.0f} KB, "
          f"{len(gif_frames)} frames)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Record demo episodes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--budget", type=float, default=0.7)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--corruption-types", type=str, nargs="+",
                        default=["clean", "fingerprint", "rain", "glare",
                                 "defocus_blur", "motion_blur",
                                 "gaussian_noise", "dust_camera",
                                 "low_light", "jpeg"])
    parser.add_argument("--output-dir", type=str, default="results/demo_episodes")
    args = parser.parse_args()

    from adversarial_dust.config import load_envelope_config
    from adversarial_dust.run_envelope import create_policy, get_image_shape

    config = load_envelope_config(args.config)
    image_shape = get_image_shape(config)
    policy = create_policy(config)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    summary = {}

    for ctype in args.corruption_types:
        print(f"\n{'=' * 50}", flush=True)
        budget_str = "N/A" if ctype == "clean" else f"{args.budget:.0%}"
        print(f"Recording {ctype} (budget={budget_str}, "
              f"{args.episodes} episodes)", flush=True)
        print(f"{'=' * 50}", flush=True)

        episodes = record_episodes(
            config, policy, image_shape, ctype, args.budget,
            args.episodes, rng)

        sr = sum(1 for s, _, _ in episodes if s) / len(episodes)
        summary[ctype] = {
            "success_rate": sr,
            "n_episodes": len(episodes),
            "budget": args.budget if ctype != "clean" else None,
        }

        # Save GIF
        label = ctype.replace("_", " ").title()
        if ctype != "clean":
            label += f" ({args.budget:.0%})"
        make_episode_gif(episodes, out / f"{ctype}.gif", label)

        # Save sample frame from middle of first episode
        first_frames = episodes[0][1]
        mid = len(first_frames) // 2
        cv2.imwrite(str(out / f"{ctype}_sample.png"),
                    cv2.cvtColor(first_frames[mid], cv2.COLOR_RGB2BGR))

    # Save summary
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll done! Results in {out}", flush=True)
    print(f"Summary: {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
