#!/usr/bin/env python3
"""Generate simulation GIF showing pick_coke_can episodes with success/failure flash.

Layout: 3 columns showing episodes under different occlusion types.
  Column 1: Fingerprint — robot succeeds, green flash at lift
  Column 2: Glare — robot succeeds, green flash at lift
  Column 3: Rain (heavy, never seen) — robot fails, red flash at timeout

Section headers separate "TRAINED ON" vs "TESTED ON".
Green border pulses when the can is lifted. Red border pulses on failure.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "camera_occlusion"))

from PIL import Image, ImageDraw, ImageFont
from camera_occlusion.camera_noise import Rain, Glare

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"

# Success happens around frame 40-50 out of 100 (can lifted off table)
LIFT_FRAME_FRAC = 0.45


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


def extract_frames(video_path, fps=5):
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["ffmpeg", "-y", "-i", str(video_path),
               "-vf", f"fps={fps}", f"{tmpdir}/f_%04d.png"]
        subprocess.run(cmd, capture_output=True, check=True)
        frames = []
        for p in sorted(Path(tmpdir).glob("f_*.png")):
            frames.append(np.array(Image.open(p).convert("RGB")))
    return frames


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "simulation_demo.gif"

    pw, ph = 220, 176
    gap = 3
    header_h = 26
    border_t = 5  # border thickness

    # --- Load source frames ---
    print("Extracting frames...", flush=True)
    fps = 5

    clean_frames = extract_frames(
        RESULTS / "coke_can_fingerprint" / "animations" / "clean.mp4", fps=fps)
    fp90_frames = extract_frames(
        RESULTS / "coke_can_fingerprint" / "animations" /
        "worst_fingerprint_budget_0.90.mp4", fps=fps)
    glare90_frames = extract_frames(
        RESULTS / "coke_can_glare" / "animations" /
        "worst_glare_budget_0.90.mp4", fps=fps)

    n = min(len(clean_frames), len(fp90_frames), len(glare90_frames))
    lift_frame = int(n * LIFT_FRAME_FRAC)
    print(f"  {n} frames, lift at frame {lift_frame}", flush=True)

    # --- Generate rain (heavy, causes failure) ---
    print("Generating heavy rain frames...", flush=True)
    rain_frames = []
    for f in clean_frames[:n]:
        rain = Rain(num_drops=200, radius_range=(5, 18))
        rain_frames.append(rain.apply(f.copy()))

    # --- Apply real glare model ---
    print("Applying glare model...", flush=True)
    glare_frames = []
    for f in clean_frames[:n]:
        g = Glare(intensity=0.9, num_streaks=6, ghost_count=4,
                  streak_length_factor=2.0, chromatic_aberration=1.8,
                  manual_source=(350, 220))
        glare_frames.append(g.apply(f.copy()))

    # --- Column definitions ---
    # (label, frames, success, section)
    columns = [
        ("Fingerprint 90%", fp90_frames, True, "train"),
        ("Glare (heavy)", glare_frames, True, "train"),
        ("Rain 50%", rain_frames, False, "test"),
    ]

    # Layout
    col_total_w = pw + border_t * 2
    total_w = col_total_w * 3 + gap * 2
    panel_total_h = ph + border_t * 2

    train_header = make_header(
        col_total_w * 2 + gap, "TRAINED ON", "#2e7d32", header_h)
    test_header = make_header(
        col_total_w, "TESTED ON (never seen)", "#1565c0", header_h)
    total_h = header_h + panel_total_h

    # --- Build frames ---
    print("Compositing...", flush=True)

    GREEN = (46, 125, 50)
    RED = (198, 40, 40)
    GRAY = (55, 71, 79)

    combined = []
    for i in range(n):
        frame = Image.new("RGB", (total_w, total_h), "#1a1a2e")

        # Headers
        frame.paste(train_header, (0, 0))
        frame.paste(test_header, (col_total_w * 2 + gap * 2, 0))

        for col_idx, (label, src_frames, success, section) in enumerate(columns):
            panel = crop_resize(src_frames[i], size=(pw, ph))

            # Determine border state
            if success:
                if i >= lift_frame:
                    # Pulsing green — sinusoidal intensity
                    phase = (i - lift_frame) / 4.0
                    pulse = 0.5 + 0.5 * np.sin(phase * np.pi * 2)
                    alpha = 0.4 + 0.6 * pulse
                    border_color = GREEN
                    # Add "PICKED UP" label at lift moment
                    if i == lift_frame:
                        pass  # label added below
                    add_label(panel, "PICKED UP", font_size=13,
                              position="top-right", bg_color="#1b5e20",
                              text_color="white")
                else:
                    border_color = GRAY
                    alpha = 0.3
            else:
                if i >= n - 5:
                    # Pulsing red at the end — failure
                    phase = (i - (n - 5)) / 2.0
                    pulse = 0.5 + 0.5 * np.sin(phase * np.pi * 2)
                    alpha = 0.5 + 0.5 * pulse
                    border_color = RED
                    add_label(panel, "FAILED", font_size=13,
                              position="top-right", bg_color="#b71c1c",
                              text_color="white")
                elif i >= n * 0.7:
                    # Amber warning zone
                    border_color = (255, 152, 0)
                    alpha = 0.4
                else:
                    border_color = GRAY
                    alpha = 0.3

            # Occlusion type label at bottom
            sec_color = "#2e7d32" if section == "train" else "#1565c0"
            add_label(panel, label, font_size=11,
                      position="bottom-center", bg_color=sec_color)

            # Draw glowing border
            bordered = draw_glow_border(panel, border_color,
                                        thickness=border_t, alpha=alpha)

            x = col_idx * (col_total_w + gap)
            frame.paste(bordered, (x, header_h))

        combined.append(frame)

    # Add a few held frames at the end for the loop
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
