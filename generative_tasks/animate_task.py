#!/usr/bin/env python3
"""Animate a pick-and-place task using pyrender (EGL) for proper 3D rendering.

Run on a GPU instance with: PYOPENGL_PLATFORM=egl python animate_task.py --all
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
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
BG = [26, 26, 46, 255]
RES = (960, 576)


TARGET_SIZE = {"objects": 0.08, "containers": 0.12}


def load_mesh(name, category):
    ply = BASE_DIR / "assets" / category / f"{name}.ply"
    obj = BASE_DIR / "urdf_assets" / category / name / f"{name}_visual.obj"
    mesh = trimesh.load(ply if ply.exists() else obj, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    # Normalize: center and scale to target size
    mesh.vertices -= mesh.centroid
    max_extent = max(mesh.extents)
    if max_extent > 0:
        target = TARGET_SIZE.get(category, 0.08)
        mesh.apply_scale(target / max_extent)
    return mesh


def make_table():
    t = trimesh.creation.box(extents=[0.40, 0.30, 0.006])
    t.apply_translation([0.07, 0.0, -0.006])
    t.visual.face_colors = [100, 82, 65, 255]
    return t


def make_gripper(pos, grip_w=0.02):
    x, y, z = pos
    parts = []
    # Shaft
    s = trimesh.creation.cylinder(radius=0.004, height=0.10)
    s.apply_translation([x, y, z + 0.11])
    s.visual.face_colors = [160, 160, 170, 255]
    parts.append(s)
    # Bar
    b = trimesh.creation.box(extents=[grip_w * 2 + 0.01, 0.008, 0.008])
    b.apply_translation([x, y, z + 0.06])
    b.visual.face_colors = [160, 160, 170, 255]
    parts.append(b)
    # Fingers
    for sign in [-1, 1]:
        f = trimesh.creation.box(extents=[0.006, 0.006, 0.055])
        f.apply_translation([x + sign * grip_w, y, z + 0.033])
        f.visual.face_colors = [180, 180, 190, 255]
        parts.append(f)
    return trimesh.util.concatenate(parts)


def ease(t):
    return t * t * (3 - 2 * t)


def make_trajectory(obj_start, cont_pos):
    ox, oy, oz = obj_start
    cx, cy, cz = cont_pos
    lift_h = 0.14
    start_h = 0.15
    traj = []

    def add(n, pos_fn, grip_fn, held_fn):
        for i in range(n):
            t = ease(i / max(1, n - 1))
            traj.append((pos_fn(t), grip_fn(t), held_fn(t)))

    add(16, lambda t: (ox, oy, start_h + (oz - start_h) * t), lambda t: 0.022, lambda t: None)
    add(5, lambda t: (ox, oy, oz), lambda t: 0.022 * (1 - t) + 0.008 * t, lambda t: None)
    add(12, lambda t: (ox, oy, oz + (lift_h - oz) * t), lambda t: 0.008, lambda t: (ox, oy, oz + (lift_h - oz) * t))
    add(16, lambda t: (ox + (cx - ox) * t, oy + (cy - oy) * t, lift_h), lambda t: 0.008, lambda t: (ox + (cx - ox) * t, oy + (cy - oy) * t, lift_h))
    drop_z = cz + 0.04
    add(12, lambda t: (cx, cy, lift_h + (drop_z - lift_h) * t), lambda t: 0.008, lambda t: (cx, cy, lift_h + (drop_z - lift_h) * t))
    add(5, lambda t: (cx, cy, drop_z), lambda t: 0.008 * (1 - t) + 0.022 * t, lambda t: None)
    add(10, lambda t: (cx, cy, drop_z + (start_h - drop_z) * t), lambda t: 0.022, lambda t: None)
    return traj


def cam_pose(distance, elevation, azimuth, target):
    er, ar = np.radians(elevation), np.radians(azimuth)
    pos = np.array(target) + distance * np.array([
        np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)
    ])
    fwd = np.array(target) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = up
    T[:3, 2] = -fwd
    T[:3, 3] = pos
    return T


def render_frame(renderer, meshes, azim=-55):
    scene = pyrender.Scene(bg_color=BG, ambient_light=[0.25, 0.25, 0.30])

    for m in meshes:
        has_vertex_colors = (m.visual.kind == "vertex")
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=has_vertex_colors))

    # Camera
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.5)
    cp = cam_pose(0.38, 28, azim, [0.07, 0, 0.04])
    scene.add(cam, pose=cp)

    # Key light (warm)
    kl = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=4.5)
    kl_pose = cam_pose(0.5, 55, azim - 30, [0.07, 0, 0])
    scene.add(kl, pose=kl_pose)

    # Fill light (cool)
    fl = pyrender.DirectionalLight(color=[0.7, 0.8, 1.0], intensity=2.0)
    fl_pose = cam_pose(0.5, 20, azim + 90, [0.07, 0, 0])
    scene.add(fl, pose=fl_pose)

    color, _ = renderer.render(scene)
    return color


def add_text(frame, text, y=16):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except Exception:
            font = ImageFont.load_default()

    q = f'"{text}"'
    bb = draw.textbbox((0, 0), q, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    x = (frame.shape[1] - tw) // 2
    p = 12
    draw.rounded_rectangle([x - p, y - 4, x + tw + p, y + th + 8], radius=8,
                            fill=(22, 33, 62), outline=(15, 52, 96), width=2)
    draw.text((x, y), q, fill=(224, 224, 255), font=font)
    return np.array(img)


def prompt_frames(text, n=35):
    frames = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        sfont = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
            sfont = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        except Exception:
            font = sfont = ImageFont.load_default()

    for i in range(n):
        img = Image.new("RGB", RES, (26, 26, 46))
        draw = ImageDraw.Draw(img)

        frac = min(1.0, i / (n * 0.55))
        shown = text[:int(len(text) * frac)]
        if frac < 1.0 and i % 4 < 2:
            shown += "|"

        draw.text((RES[0] // 2, int(RES[1] * 0.35)), "Task Prompt:",
                  fill=(136, 136, 136), font=sfont, anchor="mm")
        draw.text((RES[0] // 2, int(RES[1] * 0.48)), f'"{shown}"',
                  fill=(224, 224, 255), font=font, anchor="mm")
        if i > n * 0.65:
            draw.text((RES[0] // 2, int(RES[1] * 0.68)),
                      "Shap-E Generated Assets + WidowX Arm",
                      fill=(126, 200, 227), font=sfont, anchor="mm")
        frames.append(np.array(img))
    return frames


def animate_task(spec, output_path, fps=20):
    obj_name, cont_name = spec["object_name"], spec["container_name"]
    instruction = spec["language_instruction"]

    print(f"Loading meshes...")
    obj_mesh = load_mesh(obj_name, "objects")
    obj_mesh.apply_scale(1.0)
    cont_mesh = load_mesh(cont_name, "containers")
    cont_mesh.apply_scale(1.0)
    table = make_table()

    obj_start = np.array([0.0, 0.0, 0.0])
    cont_pos = np.array([0.15, 0.0, 0.0])
    traj = make_trajectory(obj_start.tolist(), cont_pos.tolist())

    print("Rendering prompt...")
    pframes = prompt_frames(instruction)

    print(f"Rendering {len(traj)} scene frames...")
    renderer = pyrender.OffscreenRenderer(*RES)

    sframes = []
    for i, (grip_xyz, grip_w, held_pos) in enumerate(traj):
        meshes = [table]

        c = cont_mesh.copy()
        c.apply_translation(cont_pos)
        meshes.append(c)

        o = obj_mesh.copy()
        if held_pos is not None:
            o.apply_translation(held_pos)
        elif i < 21:
            o.apply_translation(obj_start)
        else:
            o.apply_translation([cont_pos[0], cont_pos[1], cont_pos[2] + 0.04])
        meshes.append(o)

        meshes.append(make_gripper(grip_xyz, grip_w))

        azim = -55 + (i / len(traj)) * 25
        frame = render_frame(renderer, meshes, azim)
        frame = add_text(frame, instruction)
        sframes.append(frame)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(traj)}")

    renderer.delete()

    hold = [sframes[-1]] * 15
    all_frames = pframes + sframes + hold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, all_frames, fps=fps, codec="libx264")
    kb = output_path.stat().st_size / 1024
    print(f"Saved: {output_path} ({len(all_frames)} frames, {len(all_frames)/fps:.1f}s, {kb:.0f}KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "animations")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    with open(BASE_DIR / "task_specs.json") as f:
        specs = json.load(f)

    for idx in (range(len(specs)) if args.all else [args.task_index]):
        spec = specs[idx]
        out = args.output_dir / f"{spec['task_id']}.mp4"
        print(f"\n{'='*60}\n{spec['task_id']}\n{'='*60}")
        animate_task(spec, out, fps=args.fps)


if __name__ == "__main__":
    main()
