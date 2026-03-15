#!/usr/bin/env python3
"""Generate side-by-side simulation GIF: clean vs fingerprint vs glare."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "figures"


def make_side_by_side_gif():
    """Create a 3-panel GIF: clean | fingerprint | glare with labels."""
    clean = RESULTS / "coke_can_fingerprint" / "animations" / "clean.mp4"
    fp = RESULTS / "coke_can_fingerprint" / "animations" / "worst_fingerprint_budget_0.90.mp4"
    glare = RESULTS / "coke_can_glare" / "animations" / "worst_glare_budget_0.90.mp4"

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "simulation_demo.gif"

    # Each source video is 640x542. Crop off the top 30px (baked-in labels)
    # then scale to 280px wide, add our own clean labels, hstack.
    pw = 280  # panel width
    crop_top = 30
    # Cropped aspect: 640 x (542-30) = 640x512 -> scale to 280 x 224
    ph = int(pw * 512 / 640)

    filter_complex = (
        # Crop baked-in labels, scale down
        f"[0:v]crop=iw:ih-{crop_top}:0:{crop_top},scale={pw}:{ph},setsar=1[v0];"
        f"[1:v]crop=iw:ih-{crop_top}:0:{crop_top},scale={pw}:{ph},setsar=1[v1];"
        f"[2:v]crop=iw:ih-{crop_top}:0:{crop_top},scale={pw}:{ph},setsar=1[v2];"

        # Add label bars at bottom
        f"[v0]drawbox=x=0:y=ih-28:w=iw:h=28:color=0x1b5e20@0.85:t=fill,"
        f"drawtext=text='Clean  -  Success':fontcolor=white:fontsize=15:"
        f"x=(w-text_w)/2:y=h-22[l0];"

        f"[v1]drawbox=x=0:y=ih-28:w=iw:h=28:color=0xb71c1c@0.85:t=fill,"
        f"drawtext=text='Fingerprint 90%  -  Failure':fontcolor=white:fontsize=14:"
        f"x=(w-text_w)/2:y=h-22[l1];"

        f"[v2]drawbox=x=0:y=ih-28:w=iw:h=28:color=0xb71c1c@0.85:t=fill,"
        f"drawtext=text='Glare 90%  -  Failure':fontcolor=white:fontsize=14:"
        f"x=(w-text_w)/2:y=h-22[l2];"

        # Stack horizontally with 2px black dividers
        "[l0][l1][l2]hstack=inputs=3[stacked];"

        # GIF palette for quality + small file
        "[stacked]split[a][b];"
        "[b]palettegen=max_colors=96:stats_mode=diff[p];"
        "[a][p]paletteuse=dither=bayer:bayer_scale=4"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(clean),
        "-i", str(fp),
        "-i", str(glare),
        "-filter_complex", filter_complex,
        "-r", "8",  # slightly lower fps for smaller file
        "-loop", "0",
        str(output),
    ]

    print("Running ffmpeg...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg stderr: {result.stderr}", flush=True)
        raise RuntimeError("ffmpeg failed")

    size_kb = output.stat().st_size / 1024
    print(f"  -> {output} ({size_kb:.0f} KB)", flush=True)


if __name__ == "__main__":
    make_side_by_side_gif()
