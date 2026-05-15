#!/usr/bin/env python3
"""Cinematic motion-graphics animation of pick-and-place tasks.

Renders stylized 2D silhouettes from Shap-E meshes with gradient fills,
glow effects, motion trails, and smooth typography — all locally via PIL.
"""

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.spatial import ConvexHull, Delaunay

BASE_DIR = Path(__file__).parent

# Palette
BG_DARK = (12, 12, 30)
BG_MID = (18, 22, 48)
TABLE_COLOR = (55, 48, 42)
TABLE_HIGHLIGHT = (75, 65, 55)
TEXT_COLOR = (220, 225, 255)
ACCENT = (100, 180, 240)
GRIPPER_COLOR = (170, 175, 190)
GRIPPER_DARK = (120, 125, 140)

W, H = 1280, 720
FPS = 30


def load_font(size):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def get_silhouette(name, category, scale=1.0):
    """Extract a smooth 2D outline from a 3D mesh."""
    ply = BASE_DIR / "assets" / category / f"{name}.ply"
    obj_path = BASE_DIR / "urdf_assets" / category / name / f"{name}_visual.obj"
    mesh = trimesh.load(ply if ply.exists() else obj_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    mesh.vertices -= mesh.centroid
    mesh.apply_scale(scale / max(mesh.extents))

    # Project to XZ plane (front view)
    pts = mesh.vertices[:, [0, 2]]

    # Concave hull via alpha shape
    outline = _concave_outline(pts)

    # Extract colors
    colors = _get_colors(mesh)
    return outline, colors


def _concave_outline(pts, subsample=80):
    """Concave hull approximation for organic silhouettes."""
    try:
        tri = Delaunay(pts)
        edges = {}
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0) + 1

        # Boundary edges appear in only one triangle
        boundary_edges = [e for e, count in edges.items() if count == 1]

        # Build ordered boundary
        adj = {}
        for a, b in boundary_edges:
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        if not adj:
            raise ValueError("No boundary")

        ordered = [list(adj.keys())[0]]
        visited = {ordered[0]}
        while True:
            curr = ordered[-1]
            found = False
            for nxt in adj.get(curr, []):
                if nxt not in visited:
                    ordered.append(nxt)
                    visited.add(nxt)
                    found = True
                    break
            if not found:
                break

        outline = pts[ordered]

        # Subsample evenly
        if len(outline) > subsample:
            indices = np.linspace(0, len(outline) - 1, subsample, dtype=int)
            outline = outline[indices]

        return outline

    except Exception:
        hull = ConvexHull(pts)
        return pts[hull.vertices]


def _get_colors(mesh):
    if mesh.visual.kind == "vertex":
        vc = mesh.visual.vertex_colors[:, :3].astype(float)
        return {
            "light": tuple(np.clip(np.percentile(vc, 85, axis=0) * 1.2, 0, 255).astype(int)),
            "mid": tuple(np.clip(vc.mean(axis=0) * 1.1, 0, 255).astype(int)),
            "dark": tuple(np.clip(np.percentile(vc, 15, axis=0) * 0.8, 0, 255).astype(int)),
        }
    return {"light": (180, 180, 180), "mid": (130, 130, 130), "dark": (80, 80, 80)}


def ease_in_out(t):
    return t * t * (3 - 2 * t)


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def draw_gradient_bg(img):
    """Radial gradient background."""
    draw = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2
    for y in range(H):
        for x in range(0, W, 4):  # step 4 for speed
            dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            t = min(1.0, dist / (W * 0.7))
            r = int(BG_MID[0] * (1 - t) + BG_DARK[0] * t)
            g = int(BG_MID[1] * (1 - t) + BG_DARK[1] * t)
            b = int(BG_MID[2] * (1 - t) + BG_DARK[2] * t)
            draw.rectangle([x, y, x + 3, y], fill=(r, g, b))


def make_bg():
    """Pre-render background with gradient and subtle grid."""
    bg = Image.new("RGB", (W, H), BG_DARK)
    draw = ImageDraw.Draw(bg)

    # Vertical gradient
    for y in range(H):
        t = y / H
        r = int(BG_DARK[0] * (1 - t * 0.3) + 8 * t)
        g = int(BG_DARK[1] * (1 - t * 0.3) + 10 * t)
        b = int(BG_DARK[2] * (1 - t * 0.2) + 20 * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # Subtle dot grid
    for x in range(0, W, 40):
        for y in range(0, H, 40):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=(30, 30, 55))

    return bg


def draw_table(draw, y_base):
    """Isometric table surface."""
    pts = [(80, y_base), (W - 80, y_base), (W - 40, y_base + 60), (40, y_base + 60)]
    draw.polygon(pts, fill=TABLE_COLOR)
    draw.line([pts[0], pts[1]], fill=TABLE_HIGHLIGHT, width=2)
    # Edge
    draw.polygon([pts[1], pts[2], (W - 40, y_base + 65), (W - 80, y_base + 5)],
                 fill=(45, 38, 32))


def draw_silhouette(img, outline, pos, colors, glow=True, scale=1.0):
    """Draw a filled silhouette with gradient and optional glow."""
    # Scale and translate outline
    scaled = outline * scale
    cx = scaled[:, 0].mean()
    cy = scaled[:, 1].mean()
    shifted = scaled.copy()
    shifted[:, 0] += pos[0] - cx
    shifted[:, 1] = -shifted[:, 1]  # flip Y
    shifted[:, 1] += pos[1] - (-scaled[:, 1]).mean()

    pts = [(float(p[0]), float(p[1])) for p in shifted]

    if glow:
        # Glow layer
        glow_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_img)
        glow_color = (*colors["light"], 60)
        glow_draw.polygon(pts, fill=glow_color)
        glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=15))
        img.paste(Image.alpha_composite(
            img.convert("RGBA"), glow_img
        ).convert("RGB"))

    draw = ImageDraw.Draw(img)
    # Main fill
    draw.polygon(pts, fill=colors["mid"], outline=colors["dark"])

    # Highlight edge (top portion)
    top_pts = [p for p in pts if p[1] < pos[1]]
    if len(top_pts) > 2:
        draw.line(top_pts, fill=colors["light"], width=2)


def draw_gripper(draw, pos, grip_w=20, alpha=255):
    """Sleek gripper visualization."""
    x, y = pos
    # Shaft
    draw.rounded_rectangle([x - 4, y - 80, x + 4, y - 10], radius=3,
                            fill=GRIPPER_COLOR, outline=GRIPPER_DARK)
    # Crossbar
    draw.rounded_rectangle([x - grip_w - 2, y - 14, x + grip_w + 2, y - 6],
                            radius=3, fill=GRIPPER_COLOR, outline=GRIPPER_DARK)
    # Fingers
    for sign in [-1, 1]:
        fx = x + sign * grip_w
        draw.rounded_rectangle([fx - 3, y - 8, fx + 3, y + 30], radius=2,
                                fill=(190, 195, 210), outline=GRIPPER_DARK)
        # Finger tips
        draw.rounded_rectangle([fx - 4, y + 25, fx + 4, y + 35], radius=2,
                                fill=(200, 205, 220), outline=GRIPPER_DARK)


def draw_motion_trail(draw, positions, color, width=3, max_trail=8):
    """Fading motion trail."""
    n = min(len(positions), max_trail)
    for i in range(1, n):
        alpha = int(255 * (i / n) * 0.3)
        t_color = (*color[:3], alpha) if len(color) == 4 else (*color, alpha)
        # We can't do alpha lines in RGB, so fade color toward bg
        fade = 1 - (n - i) / n
        r = int(color[0] * fade + BG_DARK[0] * (1 - fade))
        g = int(color[1] * fade + BG_DARK[1] * (1 - fade))
        b = int(color[2] * fade + BG_DARK[2] * (1 - fade))
        w = max(1, int(width * (i / n)))
        draw.line([positions[-i - 1], positions[-i]], fill=(r, g, b), width=w)


def draw_text_banner(draw, text, y=28):
    font = load_font(22)
    bb = draw.textbbox((0, 0), f'"{text}"', font=font)
    tw = bb[2] - bb[0]
    x = (W - tw) // 2
    pad = 16
    # Banner bg
    draw.rounded_rectangle([x - pad, y - 8, x + tw + pad, y + 34], radius=10,
                            fill=(16, 24, 52, 220), outline=(30, 60, 110), width=2)
    draw.text((x, y), f'"{text}"', fill=TEXT_COLOR, font=font)


def render_prompt_sequence(instruction, n_frames=45):
    """Cinematic prompt reveal."""
    font_big = load_font(32)
    font_sm = load_font(15)
    font_label = load_font(14)
    bg = make_bg()
    frames = []

    for i in range(n_frames):
        frame = bg.copy()
        draw = ImageDraw.Draw(frame)

        # Typing
        frac = min(1.0, i / (n_frames * 0.5))
        shown = instruction[:int(len(instruction) * frac)]
        cursor = "|" if (frac < 1.0 and i % 6 < 3) else ""

        # Label fade in
        label_alpha = min(1.0, i / 10)
        if label_alpha > 0.1:
            lbl = "TASK PROMPT"
            draw.text((W // 2, int(H * 0.33)), lbl, fill=(100, 110, 140),
                      font=font_label, anchor="mm")

        # Accent line
        line_w = int(min(1.0, i / 15) * 200)
        if line_w > 0:
            draw.line([(W // 2 - line_w, int(H * 0.38)),
                       (W // 2 + line_w, int(H * 0.38))], fill=ACCENT, width=2)

        # Main text
        quoted = f'"{shown}{cursor}"'
        draw.text((W // 2, int(H * 0.47)), quoted, fill=TEXT_COLOR,
                  font=font_big, anchor="mm")

        # Subtitle
        if i > n_frames * 0.6:
            sub_alpha = min(1.0, (i - n_frames * 0.6) / (n_frames * 0.3))
            c = int(126 * sub_alpha)
            draw.text((W // 2, int(H * 0.62)),
                      "Shap-E Generated Assets  \u00b7  WidowX Arm",
                      fill=(c, int(200 * sub_alpha), int(227 * sub_alpha)),
                      font=font_sm, anchor="mm")

        frames.append(np.array(frame))
    return frames


def make_trajectory(obj_pos, cont_pos, table_y):
    """Smooth pick-and-place trajectory in screen coords."""
    ox, oy = obj_pos
    cx, cy = cont_pos
    start_y = table_y - 160
    traj = []

    def add(n, fn):
        for i in range(n):
            t = ease_in_out(i / max(1, n - 1))
            traj.append(fn(t))

    # Approach
    add(20, lambda t: ((ox, start_y + (oy - 35 - start_y) * t), 22, None))
    # Grip
    add(8, lambda t: ((ox, oy - 35), 22 * (1 - t) + 8 * t, None))
    # Lift
    add(15, lambda t: ((ox, oy - 35 + (start_y - oy + 35) * t), 8,
                        (ox, oy + (start_y + 50 - oy) * t)))
    # Transit
    add(20, lambda t: ((ox + (cx - ox) * t, start_y), 8,
                        (ox + (cx - ox) * t, start_y + 50)))
    # Lower
    add(15, lambda t: ((cx, start_y + (cy - 25 - start_y) * t), 8,
                        (cx, start_y + 50 + (cy - start_y - 50) * t)))
    # Release
    add(8, lambda t: ((cx, cy - 25), 8 * (1 - t) + 22 * t, None))
    # Retreat
    add(12, lambda t: ((cx, cy - 25 + (start_y - cy + 25) * t), 22, None))

    return traj


def animate_task(spec, output_path):
    obj_name, cont_name = spec["object_name"], spec["container_name"]
    instruction = spec["language_instruction"]

    print(f"  Loading silhouettes...")
    obj_outline, obj_colors = get_silhouette(obj_name, "objects", scale=120)
    cont_outline, cont_colors = get_silhouette(cont_name, "containers", scale=160)

    table_y = 420
    obj_pos = (380, table_y - 10)
    cont_pos = (780, table_y - 10)

    bg = make_bg()

    # Prompt frames
    print(f"  Rendering prompt...")
    prompt_frames = render_prompt_sequence(instruction)

    # Transition: fade from prompt to scene
    print(f"  Rendering scene...")
    traj = make_trajectory(obj_pos, cont_pos, table_y)

    # Pre-render static scene base
    scene_base = bg.copy()
    draw_table(ImageDraw.Draw(scene_base), table_y)
    # Container (always visible)
    draw_silhouette(scene_base, cont_outline, cont_pos, cont_colors, glow=True)

    gripper_trail = []
    scene_frames = []

    for i, (grip_pos, grip_w, obj_held_pos) in enumerate(traj):
        frame = scene_base.copy()

        # Object
        if obj_held_pos is not None:
            draw_silhouette(frame, obj_outline, obj_held_pos, obj_colors, glow=True)
        elif i < 28:  # before pickup
            draw_silhouette(frame, obj_outline, obj_pos, obj_colors, glow=True)
        else:  # after release — in container
            draw_silhouette(frame, obj_outline,
                            (cont_pos[0], cont_pos[1] - 15), obj_colors, glow=False)

        draw = ImageDraw.Draw(frame)

        # Motion trail
        gripper_trail.append(grip_pos)
        if len(gripper_trail) > 2:
            draw_motion_trail(draw, gripper_trail, ACCENT)

        # Gripper
        draw_gripper(draw, grip_pos, grip_w=int(grip_w))

        # Text banner
        draw_text_banner(draw, instruction)

        scene_frames.append(np.array(frame))

        if (i + 1) % 20 == 0:
            print(f"    Frame {i + 1}/{len(traj)}")

    # Fade transition (prompt -> scene)
    transition = []
    for i in range(15):
        t = ease_in_out(i / 14)
        p = Image.fromarray(prompt_frames[-1])
        s = Image.fromarray(scene_frames[0])
        blended = Image.blend(p, s, t)
        transition.append(np.array(blended))

    # Hold final
    hold = [scene_frames[-1]] * int(FPS * 1.5)

    all_frames = prompt_frames + transition + scene_frames + hold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, all_frames, fps=FPS, codec="libx264")
    kb = output_path.stat().st_size / 1024
    dur = len(all_frames) / FPS
    print(f"  Saved: {output_path.name} ({len(all_frames)} frames, {dur:.1f}s, {kb:.0f}KB)")


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
        out = args.output_dir / f"{spec['task_id']}_motion.mp4"
        print(f"\n{'='*60}\n{spec['task_id']}\n{'='*60}")
        animate_task(spec, out)


if __name__ == "__main__":
    main()
