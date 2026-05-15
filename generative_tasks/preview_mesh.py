"""Quick visualization of generated meshes using trimesh's built-in viewer."""

import argparse
import sys
from pathlib import Path

import trimesh


def preview(mesh_path: Path, wireframe: bool = False):
    """Load and display a mesh file interactively."""
    mesh = trimesh.load(mesh_path)

    print(f"Mesh: {mesh_path.name}")
    if isinstance(mesh, trimesh.Trimesh):
        print(f"  Vertices: {len(mesh.vertices)}")
        print(f"  Faces:    {len(mesh.faces)}")
        print(f"  Extents:  {mesh.extents}")
        print(f"  Watertight: {mesh.is_watertight}")
    elif isinstance(mesh, trimesh.Scene):
        print(f"  Geometries: {len(mesh.geometry)}")

    if wireframe and isinstance(mesh, trimesh.Trimesh):
        mesh.visual.face_colors = [100, 100, 100, 80]

    mesh.show()


def main():
    parser = argparse.ArgumentParser(description="Preview a generated mesh")
    parser.add_argument("mesh", type=Path, help="Path to .ply or .obj file")
    parser.add_argument("--wireframe", action="store_true")
    args = parser.parse_args()

    if not args.mesh.exists():
        print(f"File not found: {args.mesh}", file=sys.stderr)
        sys.exit(1)

    preview(args.mesh, wireframe=args.wireframe)


if __name__ == "__main__":
    main()
