"""Data format and validation for real-world lens contamination captures.

Defines the expected structure for paired clean/contaminated photographs
used to calibrate the Isaac Sim shader parameters against real-world
contamination patterns.

Expected directory structure::

    calibration_data/
      fingerprint_01/
        clean.png              # Clean reference image
        contaminated.png       # Same pose, with real fingerprint on lens
        metadata.yaml          # Contamination description + lighting
      glare_01/
        clean.png
        contaminated.png
        metadata.yaml
      ...

Each metadata.yaml contains::

    contamination_type: fingerprint  # or "glare", "dust"
    lighting: "bright_diffuse"       # or "directional", "dim_ambient"
    notes: "index finger, light pressure, center of lens"
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationSample:
    """A single paired clean/contaminated capture."""

    name: str
    clean_image: np.ndarray  # (H, W, 3) uint8
    contaminated_image: np.ndarray  # (H, W, 3) uint8
    contamination_type: str  # "fingerprint", "glare", "dust"
    lighting: str  # "bright_diffuse", "directional", "dim_ambient"
    notes: str = ""


@dataclass
class CalibrationDataset:
    """Collection of paired calibration samples."""

    samples: List[CalibrationSample]

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def filter_by_type(self, contamination_type: str) -> "CalibrationDataset":
        """Return subset of samples matching the given contamination type."""
        return CalibrationDataset(
            samples=[s for s in self.samples if s.contamination_type == contamination_type]
        )

    def split(self, validation_fraction: float = 0.2, seed: int = 42):
        """Split into train/validation sets.

        Returns:
            (train_dataset, val_dataset)
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self.samples))
        n_val = max(1, int(len(self.samples) * validation_fraction))

        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        return (
            CalibrationDataset(samples=[self.samples[i] for i in train_indices]),
            CalibrationDataset(samples=[self.samples[i] for i in val_indices]),
        )


def load_calibration_directory(data_dir: str) -> CalibrationDataset:
    """Load calibration samples from a directory of paired images.

    Args:
        data_dir: Path to directory containing subdirectories, each
            with clean.png, contaminated.png, and metadata.yaml.

    Returns:
        CalibrationDataset with all valid samples.
    """
    import cv2
    import yaml

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Calibration data directory not found: {data_dir}")

    samples = []
    for sample_dir in sorted(data_path.iterdir()):
        if not sample_dir.is_dir():
            continue

        clean_path = sample_dir / "clean.png"
        contaminated_path = sample_dir / "contaminated.png"
        metadata_path = sample_dir / "metadata.yaml"

        if not clean_path.exists() or not contaminated_path.exists():
            logger.warning(f"Skipping {sample_dir.name}: missing images")
            continue

        clean = cv2.imread(str(clean_path))
        contaminated = cv2.imread(str(contaminated_path))

        if clean is None or contaminated is None:
            logger.warning(f"Skipping {sample_dir.name}: failed to read images")
            continue

        # Convert BGR -> RGB
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        contaminated = cv2.cvtColor(contaminated, cv2.COLOR_BGR2RGB)

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f) or {}

        samples.append(CalibrationSample(
            name=sample_dir.name,
            clean_image=clean,
            contaminated_image=contaminated,
            contamination_type=metadata.get("contamination_type", "unknown"),
            lighting=metadata.get("lighting", "unknown"),
            notes=metadata.get("notes", ""),
        ))

    logger.info(f"Loaded {len(samples)} calibration samples from {data_dir}")
    return CalibrationDataset(samples=samples)


def save_calibration_npz(dataset: CalibrationDataset, output_path: str):
    """Save calibration dataset to a single .npz file for fast loading."""
    clean_images = np.stack([s.clean_image for s in dataset.samples])
    contaminated_images = np.stack([s.contaminated_image for s in dataset.samples])
    types = np.array([s.contamination_type for s in dataset.samples])
    lightings = np.array([s.lighting for s in dataset.samples])
    names = np.array([s.name for s in dataset.samples])

    np.savez_compressed(
        output_path,
        clean_images=clean_images,
        contaminated_images=contaminated_images,
        contamination_types=types,
        lightings=lightings,
        names=names,
    )
    logger.info(f"Saved {len(dataset.samples)} samples to {output_path}")


def load_calibration_npz(npz_path: str) -> CalibrationDataset:
    """Load calibration dataset from .npz file."""
    data = np.load(npz_path, allow_pickle=False)

    required_keys = {"clean_images", "contaminated_images", "contamination_types"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing keys in calibration .npz: {missing}")

    samples = []
    n = len(data["clean_images"])
    for i in range(n):
        samples.append(CalibrationSample(
            name=str(data["names"][i]) if "names" in data else f"sample_{i}",
            clean_image=data["clean_images"][i],
            contaminated_image=data["contaminated_images"][i],
            contamination_type=str(data["contamination_types"][i]),
            lighting=str(data["lightings"][i]) if "lightings" in data else "unknown",
        ))

    return CalibrationDataset(samples=samples)
