#!/usr/bin/env python3
"""Generate simulation GIF from REAL episodes (policy sees occluded images).

Records actual manipulation episodes under different corruption types.
The policy receives corrupted frames and its behavior reflects the
corruption — honest visualization with no post-hoc overlays.

Requires GPU + InternVLA-M1 server running.

Usage (on GPU machine):
    python scripts/generate_sim_animation.py --config configs/safety_predictor.yaml

Layout: N columns, one per corruption type.
Green border on success, red border on failure.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"


def get_font(size=15):
    for font_path in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()



def crop_resize(img_np, crop_top=30, size=(220, 176)):
    h, w = img_np.shape[:2]
    return Image.fromarray(img_np[crop_top:h, :, :]).resize(size, Image.LANCZOS)


def draw_glow_border(img, color, thickness=4, alpha=1.0):
    """Draw a glowing border on a PIL Image."""
    result = img.copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = result.size

    r, g, b = color
    a = int(255 * alpha)

    # Multiple border layers for glow effect
    for t in range(thickness, 0, -1):
        layer_alpha = int(a * (t / thickness) ** 0.5)
        c = (r, g, b, layer_alpha)
        draw.rectangle([(thickness - t, thickness - t),
                        (w - 1 - (thickness - t), h - 1 - (thickness - t))],
                       outline=c, width=1)

    result = result.convert("RGBA")
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def add_label(img, text, font_size=12, position="bottom-center",
              bg_color=None, text_color="white", bar_height=22):
    """Add a text label on the image."""
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    w, h = img.size

    if position == "bottom-center" and bg_color:
        y0 = h - bar_height
        draw.rectangle([(0, y0), (w, h)], fill=bg_color)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, y0 + 4), text, fill=text_color, font=font)
    elif position == "top-right":
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 4
        x = w - tw - pad - 6
        y = pad
        draw.rectangle([(x - pad, y - 2), (x + tw + pad, y + th + pad)],
                       fill=bg_color or "#000000")
        draw.text((x, y), text, fill=text_color, font=font)

    return img


def make_header(width, text, bg_color, height=26):
    header = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(header)
    font = get_font(14)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 5), text, fill="white", font=font)
    return header


def record_real_episode(config, policy, image_shape, occlusion_type, budget):
    """Record a real episode where the policy sees corrupted frames."""
    from adversarial_dust.envelope_predictor import make_occlusion_model
    from adversarial_dust.simpler_env_evaluator import SimplerEnvEvaluator

    model = make_occlusion_model(occlusion_type, config, image_shape, budget)
    evaluator = SimplerEnvEvaluator(config.env, policy, model)
    rng = np.random.default_rng(42)
    params = model.get_random_params(rng)
    success, clean_frames, dirty_frames = evaluator.run_episode(params, record=True)
    return success, dirty_frames


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate simulation demo GIF from real episodes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--corruptions", type=str, nargs="+",
                        default=["fingerprint", "glare", "rain"],
                        help="Corruption types to show")
    parser.add_argument("--budget", type=float, default=0.7,
                        help="Budget level for corruptions")
    parser.add_argument("--sections", type=str, nargs="+", default=None,
                        help="Section labels per corruption (train/test)")
    cli_args = parser.parse_args()

    from adversarial_dust.config import load_envelope_config
    from adversarial_dust.run_envelope import create_policy, get_image_shape

    config = load_envelope_config(cli_args.config)
    image_shape = get_image_shape(config)
    policy = create_policy(config)

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "simulation_demo.gif"

    pw, ph = 220, 176
    gap = 3
    header_h = 26
    border_t = 5

    # Default section assignment: first two are "train", rest are "test"
    sections = cli_args.sections
    if sections is None:
        sections = ["train"] * min(2, len(cli_args.corruptions))
        sections += ["test"] * (len(cli_args.corruptions) - len(sections))

    # --- Record REAL episodes ---
    columns = []
    for i, occ_type in enumerate(cli_args.corruptions):
        print(f"Recording real episode: {occ_type} @ budget={cli_args.budget}",
              flush=True)
        success, frames = record_real_episode(
            config, policy, image_shape, occ_type, cli_args.budget)
        label = f"{occ_type.replace('_', ' ').title()} {cli_args.budget:.0%}"
        section = sections[i] if i < len(sections) else "test"
        columns.append((label, frames, success, section))
        print(f"  {occ_type}: {'SUCCESS' if success else 'FAIL'}, "
              f"{len(frames)} frames", flush=True)

    # Layout — dynamic number of columns
    n_cols = len(columns)
    col_total_w = pw + border_t * 2
    total_w = col_total_w * n_cols + gap * (n_cols - 1)
    panel_total_h = ph + border_t * 2

    n_train = sum(1 for _, _, _, s in columns if s == "train")
    n_test = n_cols - n_train
    train_w = col_total_w * n_train + gap * max(0, n_train - 1)
    test_w = col_total_w * n_test + gap * max(0, n_test - 1)
    train_header = make_header(train_w, "TRAINED ON", "#2e7d32", header_h) \
        if n_train > 0 else None
    test_header = make_header(test_w, "TESTED ON (never seen)", "#1565c0",
                               header_h) if n_test > 0 else None
    total_h = header_h + panel_total_h

    # Subsample frames to keep GIF reasonable
    n = min(len(f) for _, f, _, _ in columns)
    stride = max(1, n // 50)
    frame_indices = list(range(0, n, stride))

    print(f"Compositing {len(frame_indices)} frames from {n} total...",
          flush=True)

    GREEN = (46, 125, 50)
    RED = (198, 40, 40)
    GRAY = (55, 71, 79)

    combined = []
    for fi, i in enumerate(frame_indices):
        frame = Image.new("RGB", (total_w, total_h), "#1a1a2e")

        # Paste headers
        if train_header:
            frame.paste(train_header, (0, 0))
        if test_header:
            test_x = (col_total_w + gap) * n_train
            frame.paste(test_header, (test_x, 0))

        frac = fi / max(len(frame_indices) - 1, 1)

        for col_idx, (label, src_frames, success, section) in enumerate(columns):
            src_i = min(i, len(src_frames) - 1)
            panel = crop_resize(src_frames[src_i], size=(pw, ph))

            # Border color based on actual episode outcome
            if success and frac >= 0.45:
                phase = (frac - 0.45) / 0.1
                pulse = 0.5 + 0.5 * np.sin(phase * np.pi * 2)
                alpha = 0.4 + 0.6 * pulse
                border_color = GREEN
                add_label(panel, "SUCCESS", font_size=13,
                          position="top-right", bg_color="#1b5e20",
                          text_color="white")
            elif not success and frac >= 0.85:
                phase = (frac - 0.85) / 0.1
                pulse = 0.5 + 0.5 * np.sin(phase * np.pi * 2)
                alpha = 0.5 + 0.5 * pulse
                border_color = RED
                add_label(panel, "FAILED", font_size=13,
                          position="top-right", bg_color="#b71c1c",
                          text_color="white")
            elif not success and frac >= 0.6:
                border_color = (255, 152, 0)
                alpha = 0.4
            else:
                border_color = GRAY
                alpha = 0.3

            sec_color = "#2e7d32" if section == "train" else "#1565c0"
            add_label(panel, label, font_size=11,
                      position="bottom-center", bg_color=sec_color)

            bordered = draw_glow_border(panel, border_color,
                                        thickness=border_t, alpha=alpha)

            x = col_idx * (col_total_w + gap)
            frame.paste(bordered, (x, header_h))

        combined.append(frame)

    # Hold last frame
    for _ in range(8):
        combined.append(combined[-1])

    # Save
    print("Saving GIF...", flush=True)
    quantized = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
                 for f in combined]
    quantized[0].save(
        output, save_all=True, append_images=quantized[1:],
        duration=200, loop=0,
    )
    size_kb = output.stat().st_size / 1024
    print(f"  -> {output} ({size_kb:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
