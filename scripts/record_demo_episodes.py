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


def make_episode_animation(episodes, output_dir, name, label,
                            fps=5, frame_size=(420, 336)):
    """Create GIF + MP4 playing episodes sequentially with a running counter.

    Each episode plays one at a time. A running "X / Y successes" counter
    updates after each episode resolves. Green border + "LIFTED" at the
    actual success step; red border + "FAIL" at episode end.
    """
    pw, ph = frame_size
    border_t = 5
    header_h = 40
    counter_h = 36
    total_w = pw + border_t * 2
    total_h = header_h + ph + border_t * 2 + counter_h

    GREEN = "#2e7d32"
    RED = "#c62828"
    GRAY = "#37474f"

    n_eps = len(episodes)
    frames_per_ep = 30  # subsample each episode to this many frames
    hold_result = 8     # hold frames after each episode resolves

    all_frames = []
    successes_so_far = 0

    for ep_idx, (success, ep_frames, success_step) in enumerate(episodes):
        n_raw = len(ep_frames)
        stride = max(1, n_raw // frames_per_ep)
        sampled_indices = list(range(0, n_raw, stride))[:frames_per_ep]

        # Map success_step to sampled frame index
        if success and success_step > 0:
            success_frame_idx = success_step // stride
        else:
            success_frame_idx = -1

        resolved = False
        for si, raw_idx in enumerate(sampled_indices):
            canvas = Image.new("RGB", (total_w, total_h), "#1a1a2e")
            draw = ImageDraw.Draw(canvas)

            # Header: condition label + episode number
            font = get_font(14)
            draw.text((8, 8), label, fill="white", font=font)
            small_font = get_font(11)
            ep_text = f"Episode {ep_idx + 1}/{n_eps}"
            bbox = draw.textbbox((0, 0), ep_text, font=small_font)
            draw.text((total_w - (bbox[2] - bbox[0]) - 10, 12),
                      ep_text, fill="#9e9e9e", font=small_font)

            # Episode frame
            frame_np = ep_frames[raw_idx]
            panel = Image.fromarray(frame_np).resize((pw, ph), Image.LANCZOS)

            # Border + label based on actual success step
            if success and success_frame_idx >= 0 and si >= success_frame_idx:
                panel = add_border(panel, GREEN, border_t)
                panel = add_label(panel, "LIFTED", GREEN)
                if not resolved:
                    successes_so_far += 1
                    resolved = True
            elif not success and si >= len(sampled_indices) - 3:
                panel = add_border(panel, RED, border_t)
                panel = add_label(panel, "FAIL", RED)
                resolved = True
            else:
                panel = add_border(panel, GRAY, border_t)

            canvas.paste(panel, (0, header_h))

            # Running counter at bottom
            completed = ep_idx if not resolved else ep_idx + 1
            attempts = ep_idx + 1
            counter_text = f"{successes_so_far} / {attempts} successes"
            counter_font = get_font(13)
            bbox = draw.textbbox((0, 0), counter_text, font=counter_font)
            cx = (total_w - (bbox[2] - bbox[0])) // 2
            cy = header_h + ph + border_t * 2 + 8
            # Color the counter based on success rate so far
            if attempts > 0:
                rate = successes_so_far / attempts
                if rate >= 0.8:
                    counter_color = "#66bb6a"
                elif rate >= 0.5:
                    counter_color = "#ffa726"
                else:
                    counter_color = "#ef5350"
            else:
                counter_color = "#9e9e9e"
            draw.text((cx, cy), counter_text, fill=counter_color,
                      font=counter_font)

            all_frames.append(canvas)

        # Hold the final frame of each episode (shows result)
        if not resolved:
            # Episode ended without explicit success/fail — count as fail
            resolved = True
        for _ in range(hold_result):
            all_frames.append(all_frames[-1])

    # Hold final summary for longer
    for _ in range(20):
        all_frames.append(all_frames[-1])

    # Save GIF
    gif_path = output_dir / f"{name}.gif"
    all_frames[0].save(
        str(gif_path), save_all=True, append_images=all_frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True)
    size_kb = gif_path.stat().st_size / 1024
    print(f"  Saved GIF: {gif_path} ({size_kb:.0f} KB, "
          f"{len(all_frames)} frames)", flush=True)

    # Save MP4
    mp4_path = output_dir / f"{name}.mp4"
    try:
        import imageio
        writer = imageio.get_writer(str(mp4_path), fps=fps,
                                     codec="libx264", quality=8)
        for frame in all_frames:
            writer.append_data(np.array(frame))
        writer.close()
        size_kb = mp4_path.stat().st_size / 1024
        print(f"  Saved MP4: {mp4_path} ({size_kb:.0f} KB)", flush=True)
    except Exception as e:
        # Fall back to saving frames as individual PNGs if imageio fails
        print(f"  MP4 save failed ({e}), trying ffmpeg...", flush=True)
        import tempfile, subprocess
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, frame in enumerate(all_frames):
                frame.save(f"{tmpdir}/f_{i:05d}.png")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", f"{tmpdir}/f_%05d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(mp4_path)
            ], capture_output=True)
        if mp4_path.exists():
            size_kb = mp4_path.stat().st_size / 1024
            print(f"  Saved MP4 (ffmpeg): {mp4_path} ({size_kb:.0f} KB)",
                  flush=True)
        else:
            print(f"  MP4 failed", flush=True)


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

        # Save GIF + MP4
        label = ctype.replace("_", " ").title()
        if ctype != "clean":
            label += f" ({args.budget:.0%})"
        make_episode_animation(episodes, out, ctype, label)

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
