"""Task factory: combine generated objects + containers into ManiSkill2 tasks.

Creates manipulation task variants by pairing object URDFs with container URDFs,
generating language instructions, and registering custom environments compatible
with SimplerEnv's interface.
"""

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TaskSpec:
    """Specification for a single manipulation task variant."""

    task_id: str  # e.g. "put_shrimp_in_chalice"
    object_name: str  # e.g. "shrimp"
    object_urdf: Path
    container_name: str  # e.g. "chalice"
    container_urdf: Path
    language_instruction: str  # e.g. "put the shrimp in the chalice"
    object_initial_pos: tuple[float, float, float] = (0.0, 0.0, 0.05)
    container_pos: tuple[float, float, float] = (0.15, 0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "object_name": self.object_name,
            "object_urdf": str(self.object_urdf),
            "container_name": self.container_name,
            "container_urdf": str(self.container_urdf),
            "language_instruction": self.language_instruction,
            "object_initial_pos": list(self.object_initial_pos),
            "container_pos": list(self.container_pos),
        }


@dataclass
class TaskFactory:
    """Generates task combinations from object and container URDFs."""

    urdf_dir: Path
    object_names: list[str] = field(default_factory=list)
    container_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.object_names:
            self.object_names = self._discover_names("objects")
        if not self.container_names:
            self.container_names = self._discover_names("containers")

    def _discover_names(self, category: str) -> list[str]:
        """Discover available assets by scanning urdf_dir/category/."""
        cat_dir = self.urdf_dir / category
        if not cat_dir.exists():
            return []
        names = sorted(
            d.name for d in cat_dir.iterdir() if d.is_dir() and (d / f"{d.name}.urdf").exists()
        )
        logger.info(f"Discovered {category}: {names}")
        return names

    def _find_urdf(self, category: str, name: str) -> Path:
        """Locate the URDF file for a given asset."""
        urdf_path = self.urdf_dir / category / name / f"{name}.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        return urdf_path

    def _make_instruction(self, obj_name: str, container_name: str) -> str:
        """Generate a natural language instruction for the task."""
        # Clean underscores for readable names
        obj_readable = obj_name.replace("_", " ")
        cont_readable = container_name.replace("_", " ")
        return f"put the {obj_readable} in the {cont_readable}"

    def _make_task_id(self, obj_name: str, container_name: str) -> str:
        return f"widowx_put_{obj_name}_in_{container_name}"

    def generate_task(self, obj_name: str, container_name: str) -> TaskSpec:
        """Create a single TaskSpec for a given object-container pair."""
        return TaskSpec(
            task_id=self._make_task_id(obj_name, container_name),
            object_name=obj_name,
            object_urdf=self._find_urdf("objects", obj_name),
            container_name=container_name,
            container_urdf=self._find_urdf("containers", container_name),
            language_instruction=self._make_instruction(obj_name, container_name),
        )

    def generate_all_tasks(self) -> list[TaskSpec]:
        """Generate TaskSpecs for all object x container combinations."""
        tasks = []
        for obj_name, cont_name in itertools.product(
            self.object_names, self.container_names
        ):
            try:
                task = self.generate_task(obj_name, cont_name)
                tasks.append(task)
                logger.info(f"Task: {task.task_id} -> '{task.language_instruction}'")
            except FileNotFoundError as e:
                logger.warning(f"Skipping {obj_name} x {cont_name}: {e}")
        return tasks

    def print_task_matrix(self):
        """Print a matrix of all object x container combinations."""
        header = [""] + self.container_names
        print(" | ".join(f"{h:>20s}" for h in header))
        print("-" * (22 * len(header)))
        for obj in self.object_names:
            row = [obj]
            for cont in self.container_names:
                task_id = self._make_task_id(obj, cont)
                row.append(task_id[:20])
            print(" | ".join(f"{c:>20s}" for c in row))
