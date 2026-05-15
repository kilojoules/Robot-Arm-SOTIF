"""Convert generated .ply meshes to URDF files suitable for ManiSkill2."""

import argparse
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

logger = logging.getLogger(__name__)


@dataclass
class PhysicsParams:
    """Physical properties for URDF generation."""

    mass: float = 0.1  # kg
    friction: float = 0.5
    restitution: float = 0.1
    target_height: float = 0.08  # meters — scale mesh so tallest dim matches this
    density: float | None = None  # if set, overrides mass based on volume


# Reasonable defaults per category
CATEGORY_PHYSICS = {
    "objects": PhysicsParams(mass=0.05, friction=0.6, target_height=0.08),
    "containers": PhysicsParams(mass=0.15, friction=0.4, target_height=0.12),
}


def normalize_mesh(mesh: trimesh.Trimesh, target_height: float) -> trimesh.Trimesh:
    """Center mesh at origin and scale to target height."""
    # Center at centroid
    mesh.vertices -= mesh.centroid

    # Scale so the largest dimension matches target_height
    extents = mesh.extents
    max_extent = max(extents)
    if max_extent > 0:
        scale = target_height / max_extent
        mesh.apply_scale(scale)

    logger.info(
        f"  Normalized: extents={mesh.extents}, "
        f"bounds=[{mesh.bounds[0]}, {mesh.bounds[1]}]"
    )
    return mesh


def compute_inertia(mesh: trimesh.Trimesh, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute center of mass and inertia tensor for URDF."""
    # Use trimesh's moment_inertia if the mesh is watertight
    if mesh.is_watertight:
        mesh.density = mass / mesh.volume
        inertia = mesh.moment_inertia
        com = mesh.center_mass
    else:
        # Approximate as bounding box
        extents = mesh.extents
        ix = mass / 12.0 * (extents[1] ** 2 + extents[2] ** 2)
        iy = mass / 12.0 * (extents[0] ** 2 + extents[2] ** 2)
        iz = mass / 12.0 * (extents[0] ** 2 + extents[1] ** 2)
        inertia = np.diag([ix, iy, iz])
        com = mesh.centroid
        logger.info("  Mesh not watertight; using bounding-box inertia approximation.")

    return com, inertia


def build_urdf_xml(
    name: str,
    obj_filename: str,
    collision_filename: str,
    mass: float,
    com: np.ndarray,
    inertia: np.ndarray,
    friction: float,
    restitution: float,
) -> ET.Element:
    """Build URDF XML ElementTree for a single-link object."""
    robot = ET.Element("robot", name=name)

    link = ET.SubElement(robot, "link", name=f"{name}_link")

    # --- Inertial ---
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia[0, 0]:.8f}",
        ixy=f"{inertia[0, 1]:.8f}",
        ixz=f"{inertia[0, 2]:.8f}",
        iyy=f"{inertia[1, 1]:.8f}",
        iyz=f"{inertia[1, 2]:.8f}",
        izz=f"{inertia[2, 2]:.8f}",
    )

    # --- Visual ---
    visual = ET.SubElement(link, "visual")
    v_geom = ET.SubElement(visual, "geometry")
    ET.SubElement(v_geom, "mesh", filename=obj_filename)

    # --- Collision (simplified) ---
    collision = ET.SubElement(link, "collision")
    c_geom = ET.SubElement(collision, "geometry")
    ET.SubElement(c_geom, "mesh", filename=collision_filename)

    # --- Contact (ManiSkill2 extension) ---
    contact = ET.SubElement(robot, "contact")
    ET.SubElement(
        contact,
        "lateral_friction",
        value=f"{friction:.4f}",
    )
    ET.SubElement(
        contact,
        "restitution",
        value=f"{restitution:.4f}",
    )

    return robot


def simplify_collision_mesh(mesh: trimesh.Trimesh, face_count: int = 500) -> trimesh.Trimesh:
    """Create a simplified collision mesh via decimation or convex hull."""
    if len(mesh.faces) <= face_count:
        return mesh.copy()
    try:
        simplified = mesh.simplify_quadric_decimation(face_count)
        logger.info(
            f"  Collision mesh: {len(mesh.faces)} -> {len(simplified.faces)} faces"
        )
        return simplified
    except Exception as e:
        logger.warning(f"  Decimation failed ({e}), using convex hull for collision.")
        return mesh.convex_hull


def convert_ply_to_urdf(
    ply_path: Path,
    output_dir: Path,
    category: str = "objects",
    physics: PhysicsParams | None = None,
) -> Path:
    """Convert a .ply mesh to a URDF with .obj visual and collision meshes.

    Returns the path to the generated .urdf file.
    """
    if physics is None:
        physics = CATEGORY_PHYSICS.get(category, PhysicsParams())

    name = ply_path.stem
    asset_dir = output_dir / name
    asset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {ply_path.name} -> {asset_dir}/")

    # Load and normalize
    mesh = trimesh.load(ply_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh = normalize_mesh(mesh, physics.target_height)

    # Compute mass (from density if provided)
    mass = physics.mass
    if physics.density and mesh.is_watertight:
        mass = physics.density * mesh.volume
        logger.info(f"  Mass from density: {mass:.4f} kg")

    # Export visual mesh as .obj
    visual_path = asset_dir / f"{name}_visual.obj"
    mesh.export(visual_path)

    # Export simplified collision mesh
    collision_mesh = simplify_collision_mesh(mesh)
    collision_path = asset_dir / f"{name}_collision.obj"
    collision_mesh.export(collision_path)

    # Compute inertia
    com, inertia = compute_inertia(mesh, mass)

    # Build and write URDF
    urdf_xml = build_urdf_xml(
        name=name,
        obj_filename=f"{name}_visual.obj",
        collision_filename=f"{name}_collision.obj",
        mass=mass,
        com=com,
        inertia=inertia,
        friction=physics.friction,
        restitution=physics.restitution,
    )

    urdf_path = asset_dir / f"{name}.urdf"
    tree = ET.ElementTree(urdf_xml)
    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="unicode", xml_declaration=True)

    logger.info(f"  URDF written to {urdf_path}")
    return urdf_path


def convert_all(assets_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Convert all .ply files in assets_dir/{objects,containers}/ to URDFs."""
    results = {}
    for category in ["objects", "containers"]:
        ply_dir = assets_dir / category
        if not ply_dir.exists():
            continue
        for ply_file in sorted(ply_dir.glob("*.ply")):
            urdf_path = convert_ply_to_urdf(
                ply_file,
                output_dir=output_dir / category,
                category=category,
            )
            results[f"{category}/{ply_file.stem}"] = urdf_path

    return results


def main():
    parser = argparse.ArgumentParser(description="Convert .ply meshes to URDF")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(__file__).parent / "assets",
        help="Directory containing objects/ and containers/ with .ply files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "urdf_assets",
        help="Directory to write URDF bundles",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = convert_all(args.assets_dir, args.output_dir)
    print(f"\nConverted {len(results)} meshes to URDF:")
    for key, path in results.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
