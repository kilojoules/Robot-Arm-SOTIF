#!/usr/bin/env python3
"""Generate SVG vector animations of pick-and-place tasks.

Projects 3D Shap-E meshes to 2D silhouettes, renders as SVG paths with
gradient fills, and animates via CSS keyframes.

Usage:
    python animate_svg.py                   # First task
    python animate_svg.py --all             # All tasks
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

BASE_DIR = Path(__file__).parent
TARGET_SIZE = {"objects": 80, "containers": 120}  # px-scale units


def load_and_project(name: str, category: str, view="front"):
    """Load mesh, normalize, project to 2D, return SVG path + color."""
    ply = BASE_DIR / "assets" / category / f"{name}.ply"
    obj = BASE_DIR / "urdf_assets" / category / name / f"{name}_visual.obj"
    mesh = trimesh.load(ply if ply.exists() else obj, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    mesh.vertices -= mesh.centroid
    target = TARGET_SIZE.get(category, 80)
    mesh.apply_scale(target / max(mesh.extents))

    # Project based on view
    if view == "front":
        pts = mesh.vertices[:, [0, 2]]  # XZ
    elif view == "side":
        pts = mesh.vertices[:, [1, 2]]  # YZ
    else:
        pts = mesh.vertices[:, [0, 1]]  # XY (top)

    # Multi-layer silhouette: slice at several depths for detail
    path_data = _outline_path(pts)

    # Extract colors
    colors = _extract_colors(mesh)

    return path_data, colors, target


def _outline_path(pts_2d, alpha_fraction=0.15):
    """Create an SVG path from 2D point cloud using concave hull approximation.

    Uses multiple convex hull layers for a more detailed silhouette.
    """
    from scipy.spatial import Delaunay

    try:
        tri = Delaunay(pts_2d)
        # Filter long edges to approximate concave hull
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                edges.add((min(a, b), max(a, b)))

        # Compute edge lengths
        edge_list = list(edges)
        lengths = [np.linalg.norm(pts_2d[a] - pts_2d[b]) for a, b in edge_list]
        threshold = np.percentile(lengths, 85)

        # Keep only short edges, find boundary
        short_edges = [e for e, l in zip(edge_list, lengths) if l < threshold]

        # Count edge usage per vertex to find boundary vertices
        from collections import Counter
        vertex_count = Counter()
        for a, b in short_edges:
            vertex_count[a] += 1
            vertex_count[b] += 1

        # Boundary vertices are those with fewer connections
        boundary = sorted(vertex_count.keys(),
                          key=lambda v: np.arctan2(pts_2d[v, 1] - pts_2d[:, 1].mean(),
                                                    pts_2d[v, 0] - pts_2d[:, 0].mean()))

        if len(boundary) < 10:
            raise ValueError("Not enough boundary points")

        # Subsample for smooth SVG path
        step = max(1, len(boundary) // 60)
        boundary = boundary[::step]
        outline = pts_2d[boundary]
    except Exception:
        # Fallback: convex hull
        hull = ConvexHull(pts_2d)
        outline = pts_2d[hull.vertices]

    return _points_to_svg_path(outline)


def _points_to_svg_path(points):
    """Convert ordered 2D points to an SVG cubic bezier path for smooth curves."""
    if len(points) < 3:
        return ""

    # Close the loop
    pts = np.vstack([points, points[0:2]])

    d = f"M {pts[0, 0]:.2f},{-pts[0, 1]:.2f} "

    # Catmull-Rom to cubic bezier
    for i in range(len(pts) - 2):
        p0 = pts[max(0, i - 1)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(len(pts) - 1, i + 2)]

        # Control points
        cp1x = p1[0] + (p2[0] - p0[0]) / 6
        cp1y = -(p1[1] + (p2[1] - p0[1]) / 6)
        cp2x = p2[0] - (p3[0] - p1[0]) / 6
        cp2y = -(p2[1] - (p3[1] - p1[1]) / 6)

        d += f"C {cp1x:.2f},{cp1y:.2f} {cp2x:.2f},{cp2y:.2f} {p2[0]:.2f},{-p2[1]:.2f} "

    d += "Z"
    return d


def _extract_colors(mesh):
    """Extract representative colors from mesh."""
    if mesh.visual.kind == "vertex" and hasattr(mesh.visual, "vertex_colors"):
        vc = mesh.visual.vertex_colors[:, :3].astype(float)
        avg = vc.mean(axis=0)
        bright = np.percentile(vc, 80, axis=0)
        dark = np.percentile(vc, 20, axis=0)
        return {
            "avg": _rgb(avg),
            "bright": _rgb(bright),
            "dark": _rgb(dark),
        }
    return {"avg": "#888888", "bright": "#aaaaaa", "dark": "#555555"}


def _rgb(arr):
    r, g, b = int(arr[0]), int(arr[1]), int(arr[2])
    return f"rgb({r},{g},{b})"


def build_svg(task_spec: dict) -> str:
    """Build a complete animated SVG for a task."""
    obj_name = task_spec["object_name"]
    cont_name = task_spec["container_name"]
    instruction = task_spec["language_instruction"]

    # Load silhouettes
    obj_path, obj_colors, obj_size = load_and_project(obj_name, "objects")
    cont_path, cont_colors, cont_size = load_and_project(cont_name, "containers")

    W, H = 800, 500
    table_y = 340
    obj_x, obj_y = 250, table_y
    cont_x, cont_y = 560, table_y

    # Gripper dimensions
    gw, gh = 6, 50
    bar_w = 40

    anim_dur = "5s"

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <style>
    @keyframes gripper-move {{
      0%   {{ transform: translate({obj_x}px, {obj_y - 140}px); }}
      15%  {{ transform: translate({obj_x}px, {obj_y - 50}px); }}
      25%  {{ transform: translate({obj_x}px, {obj_y - 50}px); }}
      40%  {{ transform: translate({obj_x}px, {obj_y - 130}px); }}
      65%  {{ transform: translate({cont_x}px, {cont_y - 130}px); }}
      80%  {{ transform: translate({cont_x}px, {cont_y - 55}px); }}
      90%  {{ transform: translate({cont_x}px, {cont_y - 55}px); }}
      100% {{ transform: translate({cont_x}px, {cont_y - 140}px); }}
    }}
    @keyframes object-move {{
      0%   {{ transform: translate({obj_x}px, {obj_y}px); }}
      24%  {{ transform: translate({obj_x}px, {obj_y}px); }}
      25%  {{ transform: translate({obj_x}px, {obj_y - 50}px); }}
      40%  {{ transform: translate({obj_x}px, {obj_y - 130}px); }}
      65%  {{ transform: translate({cont_x}px, {cont_y - 130}px); }}
      80%  {{ transform: translate({cont_x}px, {cont_y - 55}px); }}
      85%  {{ transform: translate({cont_x}px, {cont_y - 10}px); }}
      100% {{ transform: translate({cont_x}px, {cont_y - 10}px); }}
    }}
    @keyframes grip-close {{
      0%   {{ transform: scaleX(1); }}
      18%  {{ transform: scaleX(1); }}
      25%  {{ transform: scaleX(0.4); }}
      82%  {{ transform: scaleX(0.4); }}
      88%  {{ transform: scaleX(1); }}
      100% {{ transform: scaleX(1); }}
    }}
    @keyframes fade-in {{
      0%   {{ opacity: 0; }}
      100% {{ opacity: 1; }}
    }}
    @keyframes cursor-blink {{
      0%, 50% {{ opacity: 1; }}
      51%, 100% {{ opacity: 0; }}
    }}
    .gripper {{ animation: gripper-move {anim_dur} ease-in-out infinite; }}
    .object  {{ animation: object-move {anim_dur} ease-in-out infinite; }}
    .grip-fingers {{ animation: grip-close {anim_dur} ease-in-out infinite; transform-origin: center; }}
  </style>

  <defs>
    <linearGradient id="obj-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{obj_colors['bright']}" />
      <stop offset="100%" stop-color="{obj_colors['dark']}" />
    </linearGradient>
    <linearGradient id="cont-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{cont_colors['bright']}" />
      <stop offset="100%" stop-color="{cont_colors['dark']}" />
    </linearGradient>
    <filter id="shadow" x="-10%" y="-10%" width="130%" height="130%">
      <feDropShadow dx="2" dy="3" stdDeviation="4" flood-opacity="0.3" />
    </filter>
  </defs>

  <!-- Background -->
  <rect width="{W}" height="{H}" fill="#1a1a2e" />

  <!-- Prompt text -->
  <rect x="{W//2 - 280}" y="18" width="560" height="36" rx="8" fill="#16213e" stroke="#0f3460" stroke-width="2" />
  <text x="{W//2}" y="42" text-anchor="middle" font-family="system-ui, sans-serif"
        font-size="16" font-weight="bold" fill="#e0e0ff">"{instruction}"</text>

  <!-- Table surface -->
  <path d="M 60,{table_y + 10} L 200,{table_y + 55} L 740,{table_y + 55} L 600,{table_y + 10} Z"
        fill="#6b5b4f" stroke="#7d6d61" stroke-width="1" />
  <rect x="60" y="{table_y + 10}" width="540" height="0" fill="none" />
  <path d="M 60,{table_y + 10} L 740,{table_y + 10}" stroke="#8a7a6e" stroke-width="2" opacity="0.6" />

  <!-- Container (static) -->
  <g transform="translate({cont_x}, {cont_y})" filter="url(#shadow)">
    <path d="{cont_path}" fill="url(#cont-grad)" stroke="{cont_colors['dark']}" stroke-width="1.5" opacity="0.95" />
  </g>

  <!-- Object (animated) -->
  <g class="object" filter="url(#shadow)">
    <path d="{obj_path}" fill="url(#obj-grad)" stroke="{obj_colors['dark']}" stroke-width="1" opacity="0.95" />
  </g>

  <!-- Gripper (animated) -->
  <g class="gripper">
    <!-- Shaft -->
    <rect x="-3" y="-70" width="6" height="55" rx="2" fill="#9898a2" stroke="#7a7a84" stroke-width="1" />
    <!-- Crossbar -->
    <rect x="-20" y="-18" width="40" height="7" rx="2" fill="#a0a0aa" stroke="#8888" stroke-width="1" />
    <!-- Fingers -->
    <g class="grip-fingers">
      <rect x="-20" y="-12" width="5" height="35" rx="1.5" fill="#b0b0ba" stroke="#9090" stroke-width="1" />
      <rect x="15" y="-12" width="5" height="35" rx="1.5" fill="#b0b0ba" stroke="#9090" stroke-width="1" />
    </g>
  </g>

  <!-- Label -->
  <text x="{W//2}" y="{H - 15}" text-anchor="middle" font-family="monospace"
        font-size="12" fill="#7ec8e3" opacity="0.7">Shap-E Generated Assets + WidowX Arm</text>
</svg>"""

    return svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "animations")
    args = parser.parse_args()

    with open(BASE_DIR / "task_specs.json") as f:
        specs = json.load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx in (range(len(specs)) if args.all else [args.task_index]):
        spec = specs[idx]
        svg = build_svg(spec)
        out = args.output_dir / f"{spec['task_id']}.svg"
        out.write_text(svg)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
