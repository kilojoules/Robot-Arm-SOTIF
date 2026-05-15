#!/usr/bin/env python3
"""Cinematic 3D animation: beauty shots + pick-and-place with post-processing.

Run on GPU with: PYOPENGL_PLATFORM=egl python animate_cinematic.py --all
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v3 as iio
import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = Path(__file__).parent
W, H = 1280, 720
FPS = 30
BG = [12, 12, 30, 255]
ACCENT = (100, 180, 240)
TEXT_COL = (220, 225, 255)


# ── Fonts ──────────────────────────────────────────────────────────────

def font(size):
    for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ── Mesh loading ───────────────────────────────────────────────────────

TARGET = {"objects": 0.07, "containers": 0.10}


def load_mesh(name, category):
    ply = BASE_DIR / "assets" / category / f"{name}.ply"
    obj = BASE_DIR / "urdf_assets" / category / name / f"{name}_visual.obj"
    m = trimesh.load(ply if ply.exists() else obj, process=True)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(m.dump())
    m.vertices -= m.centroid
    m.apply_scale(TARGET.get(category, 0.07) / max(m.extents))
    return m


# ── Camera ─────────────────────────────────────────────────────────────

def cam_pose(dist, elev, azim, target=(0, 0, 0)):
    er, ar = np.radians(elev), np.radians(azim)
    t = np.array(target)
    pos = t + dist * np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)])
    fwd = t - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    T = np.eye(4); T[:3, 0] = right; T[:3, 1] = up; T[:3, 2] = -fwd; T[:3, 3] = pos
    return T


# ── Scene setup ────────────────────────────────────────────────────────

def make_table():
    t = trimesh.creation.box(extents=[0.40, 0.30, 0.004])
    t.apply_translation([0.07, 0, -0.005])
    t.visual.face_colors = [85, 72, 60, 255]
    return t


def make_gripper(pos, gw=0.018):
    x, y, z = pos
    parts = []
    s = trimesh.creation.cylinder(radius=0.003, height=0.08)
    s.apply_translation([x, y, z + 0.10]); s.visual.face_colors = [170, 175, 185, 255]
    parts.append(s)
    b = trimesh.creation.box(extents=[gw*2+0.008, 0.006, 0.006])
    b.apply_translation([x, y, z+0.058]); b.visual.face_colors = [170, 175, 185, 255]
    parts.append(b)
    for sign in [-1, 1]:
        f = trimesh.creation.box(extents=[0.004, 0.004, 0.045])
        f.apply_translation([x+sign*gw, y, z+0.035]); f.visual.face_colors = [190, 195, 205, 255]
        parts.append(f)
    return trimesh.util.concatenate(parts)


def build_scene(meshes, azim=-50, dist=0.36, elev=26, target=(0.07, 0, 0.03)):
    sc = pyrender.Scene(bg_color=BG, ambient_light=[0.15, 0.15, 0.20])
    for m in meshes:
        vc = (m.visual.kind == "vertex")
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=vc))
    cam = pyrender.PerspectiveCamera(yfov=np.pi/4.5)
    cp = cam_pose(dist, elev, azim, target)
    sc.add(cam, pose=cp)
    # Three-point lighting
    sc.add(pyrender.DirectionalLight(color=[1.0, 0.95, 0.88], intensity=5.0),
           pose=cam_pose(0.5, 50, azim-35, target))
    sc.add(pyrender.DirectionalLight(color=[0.6, 0.7, 1.0], intensity=2.5),
           pose=cam_pose(0.5, 15, azim+100, target))
    sc.add(pyrender.DirectionalLight(color=[0.9, 0.85, 0.8], intensity=1.5),
           pose=cam_pose(0.5, -10, azim+180, target))
    return sc


# ── Post-processing ───────────────────────────────────────────────────

def post_process(frame):
    """Vignette + subtle color grade + bloom."""
    img = Image.fromarray(frame)

    # Bloom on bright areas
    bright = img.point(lambda p: p if p > 180 else 0)
    bloom = bright.filter(ImageFilter.GaussianBlur(radius=20))
    img = Image.blend(img, bloom, 0.15)

    # Vignette
    vig = Image.new("L", (W, H), 0)
    vd = ImageDraw.Draw(vig)
    cx, cy = W//2, H//2
    for r in range(max(W, H), 0, -2):
        brightness = int(255 * min(1.0, (r / (W * 0.55)) ** 0.4))
        brightness = 255 - brightness
        vd.ellipse([cx-r, cy-r, cx+r, cy+r], fill=brightness)
    img = Image.composite(img, Image.new("RGB", (W, H), (5, 5, 15)), vig)

    return np.array(img)


def add_text_overlay(frame, text, y=24):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    f = font(20)
    q = f'"{text}"'
    bb = draw.textbbox((0, 0), q, font=f)
    tw = bb[2] - bb[0]
    x = (W - tw) // 2
    pad = 14
    draw.rounded_rectangle([x-pad, y-6, x+tw+pad, y+30], radius=10,
                            fill=(12, 20, 45, 200), outline=(30, 60, 120), width=2)
    draw.text((x, y), q, fill=TEXT_COL, font=f)
    return np.array(img)


# ── Easing ─────────────────────────────────────────────────────────────

def ease(t): return t*t*(3-2*t)
def ease_out(t): return 1-(1-t)**3


# ── Prompt sequence ───────────────────────────────────────────────────

def render_prompt(text, n=50):
    f_big, f_sm, f_lbl = font(34), font(15), font(13)
    frames = []
    for i in range(n):
        img = Image.new("RGB", (W, H), (12, 12, 30))
        draw = ImageDraw.Draw(img)
        # Subtle gradient
        for y in range(H):
            t = y / H
            draw.line([(0,y),(W,y)], fill=(int(12+6*t), int(12+8*t), int(30+10*t)))
        # Dot grid
        for gx in range(0, W, 50):
            for gy in range(0, H, 50):
                draw.ellipse([gx-1, gy-1, gx+1, gy+1], fill=(25, 28, 50))

        frac = min(1.0, i / (n * 0.45))
        shown = text[:int(len(text) * frac)]
        cursor = "|" if (frac < 1.0 and i % 8 < 4) else ""

        # Label
        a = min(1.0, i/12)
        draw.text((W//2, int(H*0.32)), "TASK  PROMPT",
                  fill=(int(80*a), int(90*a), int(130*a)), font=f_lbl, anchor="mm",
                  spacing=4)
        # Accent line
        lw = int(ease_out(min(1.0, i/18)) * 180)
        if lw > 0:
            draw.line([(W//2-lw, int(H*0.365)), (W//2+lw, int(H*0.365))],
                      fill=ACCENT, width=2)
        # Text
        draw.text((W//2, int(H*0.45)), f'"{shown}{cursor}"',
                  fill=TEXT_COL, font=f_big, anchor="mm")
        # Subtitle
        if i > n*0.55:
            sa = min(1.0, (i - n*0.55) / (n*0.35))
            draw.text((W//2, int(H*0.60)),
                      "Shap-E Generated Assets  \u00b7  WidowX Arm",
                      fill=(int(90*sa), int(180*sa), int(210*sa)), font=f_sm, anchor="mm")
        frames.append(np.array(img))
    return frames


# ── Beauty shots ──────────────────────────────────────────────────────

def render_beauty(renderer, mesh, label, n=40):
    """Slow turntable rotation of a single object."""
    f_lbl = font(16)
    frames = []
    for i in range(n):
        t = i / n
        azim = -60 + t * 120
        fade = ease(min(1.0, i / 8)) * (1 - ease(max(0, (i - n + 8)) / 8))

        sc = pyrender.Scene(bg_color=BG, ambient_light=[0.2, 0.2, 0.25])
        vc = (mesh.visual.kind == "vertex")
        sc.add(pyrender.Mesh.from_trimesh(mesh, smooth=vc))
        cam = pyrender.PerspectiveCamera(yfov=np.pi/5)
        sc.add(cam, pose=cam_pose(0.22, 15, azim, [0, 0, 0]))
        sc.add(pyrender.DirectionalLight(color=[1, 0.95, 0.9], intensity=6),
               pose=cam_pose(0.3, 45, azim-40))
        sc.add(pyrender.DirectionalLight(color=[0.5, 0.6, 1.0], intensity=3),
               pose=cam_pose(0.3, 10, azim+120))

        color, _ = renderer.render(sc)
        frame = post_process(color)

        # Label
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        a = int(255 * fade)
        draw.text((W//2, H - 60), label, fill=(*TEXT_COL, a), font=f_lbl, anchor="mm")
        # Thin accent underline
        bb = draw.textbbox((0, 0), label, font=f_lbl)
        lw = (bb[2] - bb[0]) // 2
        draw.line([(W//2 - lw, H-45), (W//2 + lw, H-45)],
                  fill=(*ACCENT, int(a*0.7)), width=1)
        frames.append(np.array(img))
    return frames


# ── Pick-and-place ────────────────────────────────────────────────────

def make_trajectory(os, cp):
    ox, oy, oz = os; cx, cy, cz = cp
    lift, start = 0.13, 0.14
    traj = []
    def add(n, fn):
        for i in range(n):
            t = ease(i/max(1, n-1)); traj.append(fn(t))
    add(18, lambda t: ((ox,oy, start+(oz-start)*t), 0.020, None))
    add(6, lambda t: ((ox,oy,oz), 0.020*(1-t)+0.007*t, None))
    add(14, lambda t: ((ox,oy, oz+(lift-oz)*t), 0.007, (ox,oy, oz+(lift-oz)*t)))
    add(18, lambda t: ((ox+(cx-ox)*t, oy+(cy-oy)*t, lift), 0.007,
                        (ox+(cx-ox)*t, oy+(cy-oy)*t, lift)))
    dz = cz+0.035
    add(14, lambda t: ((cx,cy, lift+(dz-lift)*t), 0.007, (cx,cy, lift+(dz-lift)*t)))
    add(6, lambda t: ((cx,cy,dz), 0.007*(1-t)+0.020*t, None))
    add(12, lambda t: ((cx,cy, dz+(start-dz)*t), 0.020, None))
    return traj


def render_pickplace(renderer, obj_mesh, cont_mesh, instruction):
    table = make_table()
    obj_start = np.array([0.0, 0.0, 0.0])
    cont_pos = np.array([0.15, 0.0, 0.0])
    traj = make_trajectory(obj_start.tolist(), cont_pos.tolist())

    frames = []
    for i, (gpos, gw, held) in enumerate(traj):
        geoms = [table]
        c = cont_mesh.copy(); c.apply_translation(cont_pos); geoms.append(c)
        o = obj_mesh.copy()
        if held:
            o.apply_translation(held)
        elif i < 24:
            o.apply_translation(obj_start)
        else:
            o.apply_translation([cont_pos[0], cont_pos[1], cont_pos[2]+0.035])
        geoms.append(o)
        geoms.append(make_gripper(gpos, gw))

        azim = -52 + (i/len(traj)) * 22
        sc = build_scene(geoms, azim=azim)
        color, _ = renderer.render(sc)
        frame = post_process(color)
        frame = add_text_overlay(frame, instruction)
        frames.append(frame)

        if (i+1) % 20 == 0:
            print(f"    Scene {i+1}/{len(traj)}")
    return frames


# ── Transitions ───────────────────────────────────────────────────────

def crossfade(a, b, n=12):
    return [np.array(Image.blend(Image.fromarray(a), Image.fromarray(b), ease(i/(n-1))))
            for i in range(n)]


# ── Main ──────────────────────────────────────────────────────────────

def animate_task(spec, output_path):
    obj_name, cont_name = spec["object_name"], spec["container_name"]
    instruction = spec["language_instruction"]

    obj_mesh = load_mesh(obj_name, "objects")
    cont_mesh = load_mesh(cont_name, "containers")

    renderer = pyrender.OffscreenRenderer(W, H)

    print("  Prompt...")
    prompt = render_prompt(instruction)

    print("  Beauty: object...")
    beauty_obj = render_beauty(renderer, obj_mesh,
                               obj_name.replace("_", " ").upper())

    print("  Beauty: container...")
    beauty_cont = render_beauty(renderer, cont_mesh,
                                 cont_name.replace("_", " ").upper())

    print("  Pick-and-place...")
    scene = render_pickplace(renderer, obj_mesh, cont_mesh, instruction)

    renderer.delete()

    # Assemble with transitions
    print("  Compositing...")
    all_frames = (
        prompt
        + crossfade(prompt[-1], beauty_obj[0])
        + beauty_obj
        + crossfade(beauty_obj[-1], beauty_cont[0])
        + beauty_cont
        + crossfade(beauty_cont[-1], scene[0])
        + scene
        + [scene[-1]] * int(FPS * 1.5)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, all_frames, fps=FPS, codec="libx264")
    dur = len(all_frames) / FPS
    kb = output_path.stat().st_size / 1024
    print(f"  Done: {output_path.name} ({len(all_frames)} frames, {dur:.1f}s, {kb:.0f}KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "animations")
    args = parser.parse_args()

    with open(BASE_DIR / "task_specs.json") as f:
        specs = json.load(f)

    for idx in (range(len(specs)) if args.all else [args.task_index]):
        spec = specs[idx]
        out = args.output_dir / f"{spec['task_id']}_cinematic.mp4"
        print(f"\n{'='*60}\n{spec['task_id']}\n{'='*60}")
        animate_task(spec, out)


if __name__ == "__main__":
    main()
