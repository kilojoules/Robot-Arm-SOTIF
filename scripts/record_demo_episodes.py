#!/usr/bin/env python3
"""Record real episodes per corruption type for paper figures and demo animations.

For each corruption type (+ clean), records 10 episodes where the policy
sees the corrupted frames. Saves:
  - Per-type GIF + MP4 animation with actual success/failure labels
  - Real-time P(failure) gauge from the safety predictor
  - Running success counter
  - Per-type sample PNG
  - Summary JSON

Uses the LOO model specific to each corruption type for P(fail) predictions
(the model that was trained WITHOUT that corruption type), so the gauge
shows out-of-distribution predictions — the paper's central claim.

Usage (on GPU machine):
    python scripts/record_demo_episodes.py \
        --config configs/safety_predictor.yaml \
        --episodes 10 \
        --loo-dir results/loo_analysis
"""

import argparse
import json
import time
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Per-type budget targeting 40-70% success rate regime (most informative).
# Derived from LOO evaluation data. Policy-robust types use 0.9 to show
# the policy handles them even at extreme severity.
DEFAULT_BUDGETS = {
    "clean": 0.0,
    "rain": 0.5,            # ~50% SR at 0.5
    "fingerprint": 0.9,     # needs high budget to cause failures
    "motion_blur": 0.9,     # needs high budget
    "defocus_blur": 0.9,    # needs high budget
    "low_light": 0.9,       # needs high budget
    "glare": 0.9,           # policy-robust — show at high severity
    "gaussian_noise": 0.9,  # policy-robust
    "dust_camera": 0.9,     # policy-robust
    "jpeg": 0.9,            # policy-robust
}


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
    w, h = frame_pil.size
    draw = ImageDraw.Draw(frame_pil)
    for t in range(thickness):
        draw.rectangle([(t, t), (w - 1 - t, h - 1 - t)], outline=color)
    return frame_pil


def add_label(frame_pil, text, bg_color, position="top-right"):
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
    draw.rectangle([(x - pad, y - 1), (x + tw + pad, y + th + pad)],
                   fill=bg_color)
    draw.text((x, y), text, fill="white", font=font)
    return frame_pil


def load_loo_predictor(loo_dir, corruption_type, device="cuda"):
    """Load the LOO model that was trained WITHOUT this corruption type.

    This ensures the P(fail) gauge shows OOD predictions, matching the
    paper's experimental setup.
    """
    from adversarial_dust.safety_predictor import SafetyPredictorCNN

    fold_dir = Path(loo_dir) / f"loo_fold_{corruption_type}" / "model"
    model_path = fold_dir / "best_model.pt"

    if model_path.exists():
        predictor = SafetyPredictorCNN()
        predictor.load(str(model_path))
        predictor.to(device)
        print(f"  Loaded LOO model (trained without {corruption_type}) "
              f"from {model_path}", flush=True)
        return predictor

    # No fallback — wrong model would produce misleading OOD claim
    print(f"  WARNING: No LOO model found for {corruption_type} "
          f"at {fold_dir}. P(fail) gauge will be disabled.", flush=True)
    return None


def record_episodes(config, policy, image_shape, corruption_type, budget,
                    n_episodes, rng, predictor=None):
    """Record n_episodes with a given corruption type.

    Returns list of (success, dirty_frames, success_step, p_failures) tuples.
    """
    from adversarial_dust.envelope_predictor import make_occlusion_model
    from adversarial_dust.simpler_env_evaluator import SimplerEnvEvaluator

    if corruption_type == "clean":
        from adversarial_dust.dust_model import AdversarialDustModel
        from adversarial_dust.config import DustGridConfig
        model = AdversarialDustModel(DustGridConfig(), image_shape, 0.0)
    else:
        model = make_occlusion_model(corruption_type, config, image_shape,
                                      budget)

    evaluator = SimplerEnvEvaluator(config.env, policy, model)
    episodes = []

    for ep in range(n_episodes):
        params = None if corruption_type == "clean" else \
            model.get_random_params(rng)
        t0 = time.time()
        success, _, dirty_frames = evaluator.run_episode(params, record=True)
        success_step = getattr(evaluator, "last_success_step", -1)

        p_failures = []
        if predictor is not None and dirty_frames:
            p_failures = list(predictor.predict_batch(dirty_frames))

        elapsed = time.time() - t0
        mean_pf = np.mean(p_failures) if p_failures else 0
        episodes.append((success, dirty_frames, success_step, p_failures))
        step_info = f", lifted at step {success_step}" if success_step > 0 \
            else ""
        print(f"    ep {ep+1}/{n_episodes}: "
              f"{'SUCCESS' if success else 'FAIL'}, "
              f"mean P(fail)={mean_pf:.3f}"
              f"{step_info} ({elapsed:.1f}s, "
              f"{len(dirty_frames)} frames)", flush=True)

    return episodes


def _draw_pfail_gauge(draw, x, y, w, h, p_fail, font):
    """Draw a horizontal P(failure) gauge bar with numeric label."""
    draw.rectangle([(x, y), (x + w, y + h)], fill="#263238")
    fill_w = max(1, int(w * p_fail))
    if p_fail < 0.3:
        bar_color = "#66bb6a"
    elif p_fail < 0.6:
        bar_color = "#ffa726"
    else:
        bar_color = "#ef5350"
    draw.rectangle([(x, y), (x + fill_w, y + h)], fill=bar_color)
    draw.rectangle([(x, y), (x + w, y + h)], outline="#546e7a", width=1)
    draw.text((x + 4, y + 2), f"P(fail) = {p_fail:.0%}", fill="white",
              font=font)


def make_episode_animation(episodes, output_dir, name, label, model_label="",
                            fps=5, frame_size=(420, 336)):
    """Create GIF + MP4 playing episodes sequentially.

    - Episodes play one at a time
    - P(failure) gauge updates per frame from the safety predictor
    - "LIFTED" label at the actual env success timestep
    - "FAIL" label at episode end
    - Running "X / Y successes" counter updates ONLY between episodes
    """
    pw, ph = frame_size
    border_t = 5
    header_h = 40
    gauge_h = 22
    counter_h = 32
    total_w = pw + border_t * 2
    total_h = header_h + ph + border_t * 2 + gauge_h + 4 + counter_h

    GREEN = "#2e7d32"
    RED = "#c62828"
    GRAY = "#37474f"

    n_eps = len(episodes)
    frames_per_ep = 25  # keep GIFs compact
    hold_result = 10

    all_frames = []
    successes_so_far = 0
    completed_eps = 0

    for ep_idx, (success, ep_frames, success_step, p_failures) in \
            enumerate(episodes):
        n_raw = len(ep_frames)
        stride = max(1, n_raw // frames_per_ep)
        sampled_indices = list(range(0, n_raw, stride))[:frames_per_ep]

        # Map success_step to sampled frame index using bisect.
        # success_step is 1-indexed (post-increment in evaluator);
        # searching 0-indexed sampled_indices means LIFTED shows on the
        # result frame (object visibly lifted), not the action frame.
        if success and success_step > 0:
            success_frame_idx = bisect_left(sampled_indices, success_step)
            success_frame_idx = min(success_frame_idx,
                                     len(sampled_indices) - 1)
        else:
            success_frame_idx = -1

        ep_resolved = False
        for si, raw_idx in enumerate(sampled_indices):
            canvas = Image.new("RGB", (total_w, total_h), "#1a1a2e")
            draw = ImageDraw.Draw(canvas)

            # Header: condition label + episode counter
            font = get_font(14)
            draw.text((8, 8), label, fill="white", font=font)
            small_font = get_font(11)
            ep_text = f"Episode {ep_idx + 1}/{n_eps}"
            bbox = draw.textbbox((0, 0), ep_text, font=small_font)
            draw.text((total_w - (bbox[2] - bbox[0]) - 10, 12),
                      ep_text, fill="#9e9e9e", font=small_font)
            # Model annotation
            if model_label:
                tiny_font = get_font(9)
                draw.text((8, 28), model_label, fill="#78909c",
                          font=tiny_font)

            # Episode frame
            frame_np = ep_frames[raw_idx]
            panel = Image.fromarray(frame_np).resize((pw, ph), Image.LANCZOS)

            # Border at actual success step or episode end
            if success and success_frame_idx >= 0 and \
                    si >= success_frame_idx:
                panel = add_border(panel, GREEN, border_t)
                panel = add_label(panel, "LIFTED", GREEN)
                if not ep_resolved:
                    successes_so_far += 1
                    completed_eps += 1
                    ep_resolved = True
            elif not success and si >= len(sampled_indices) - 3:
                panel = add_border(panel, RED, border_t)
                panel = add_label(panel, "FAIL", RED)
                if not ep_resolved:
                    completed_eps += 1
                    ep_resolved = True
            else:
                panel = add_border(panel, GRAY, border_t)

            canvas.paste(panel, (0, header_h))

            # P(failure) gauge
            gauge_y = header_h + ph + border_t * 2 + 2
            gauge_font = get_font(10)
            if p_failures and raw_idx < len(p_failures):
                pf = p_failures[raw_idx]
            elif p_failures:
                pf = p_failures[-1]
            else:
                pf = 0.0
            _draw_pfail_gauge(draw, 8, gauge_y, total_w - 16, gauge_h,
                              pf, gauge_font)

            # Running counter — only shows completed episodes
            counter_y = gauge_y + gauge_h + 6
            counter_font = get_font(13)
            if completed_eps > 0:
                counter_text = (f"{successes_so_far} / {completed_eps} "
                                f"successes")
                rate = successes_so_far / completed_eps
                counter_color = "#66bb6a" if rate >= 0.8 else (
                    "#ffa726" if rate >= 0.5 else "#ef5350")
            else:
                counter_text = "—"
                counter_color = "#9e9e9e"
            bbox = draw.textbbox((0, 0), counter_text, font=counter_font)
            cx = (total_w - (bbox[2] - bbox[0])) // 2
            draw.text((cx, counter_y), counter_text, fill=counter_color,
                      font=counter_font)

            all_frames.append(canvas)

        # Count unresolved episodes as failures
        if not ep_resolved:
            completed_eps += 1

        # Hold result frame between episodes
        for _ in range(hold_result):
            all_frames.append(all_frames[-1])

    # Hold final summary longer
    for _ in range(20):
        all_frames.append(all_frames[-1])

    # Save GIF
    gif_path = output_dir / f"{name}.gif"
    all_frames[0].save(
        str(gif_path), save_all=True, append_images=all_frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True)
    size_kb = gif_path.stat().st_size / 1024
    print(f"  GIF: {gif_path} ({size_kb:.0f} KB, "
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
        print(f"  MP4: {mp4_path} ({mp4_path.stat().st_size/1024:.0f} KB)",
              flush=True)
    except Exception as e:
        print(f"  MP4 failed ({e}), trying ffmpeg...", flush=True)
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
            print(f"  MP4 (ffmpeg): {mp4_path} "
                  f"({mp4_path.stat().st_size/1024:.0f} KB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Record demo episodes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--corruption-types", type=str, nargs="+",
                        default=["clean", "fingerprint", "rain", "glare",
                                 "defocus_blur", "motion_blur",
                                 "gaussian_noise", "dust_camera",
                                 "low_light", "jpeg"])
    parser.add_argument("--output-dir", type=str,
                        default="results/demo_episodes")
    parser.add_argument("--loo-dir", type=str,
                        default="results/loo_analysis",
                        help="LOO results directory (for per-type models)")
    parser.add_argument("--budget", type=float, default=None,
                        help="Override budget for all types (default: "
                             "per-type budgets targeting informative regime)")
    parser.add_argument("--device", type=str, default="cuda")
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
        # Per-type budget
        if args.budget is not None:
            budget = args.budget
        else:
            budget = DEFAULT_BUDGETS.get(ctype, 0.7)

        budget_str = "N/A" if ctype == "clean" else f"{budget:.0%}"
        print(f"\n{'=' * 50}", flush=True)
        print(f"Recording {ctype} (budget={budget_str}, "
              f"{args.episodes} episodes)", flush=True)
        print(f"{'=' * 50}", flush=True)

        # Load the LOO model for this type (trained WITHOUT this type)
        if ctype == "clean":
            # For clean, use any available model
            predictor = load_loo_predictor(args.loo_dir, "rain",
                                           device=args.device)
            model_label = ""
        else:
            predictor = load_loo_predictor(args.loo_dir, ctype,
                                           device=args.device)
            model_label = f"Model trained without {ctype}" if predictor \
                else ""

        episodes = record_episodes(
            config, policy, image_shape, ctype, budget,
            args.episodes, rng, predictor=predictor)

        sr = sum(1 for s, _, _, _ in episodes if s) / len(episodes)
        summary[ctype] = {
            "success_rate": sr,
            "n_episodes": len(episodes),
            "budget": budget if ctype != "clean" else None,
        }

        # Save GIF + MP4
        label = ctype.replace("_", " ").title()
        if ctype != "clean":
            label += f" ({budget:.0%})"
        make_episode_animation(episodes, out, ctype, label,
                                model_label=model_label)

        # Save sample frame
        first_frames = episodes[0][1]
        mid = len(first_frames) // 2
        cv2.imwrite(str(out / f"{ctype}_sample.png"),
                    cv2.cvtColor(first_frames[mid], cv2.COLOR_RGB2BGR))

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll done! Results in {out}", flush=True)
    print(f"Summary: {json.dumps(summary, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
