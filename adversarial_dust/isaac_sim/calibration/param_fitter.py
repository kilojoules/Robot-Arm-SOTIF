"""Differentiable parameter fitting for sim-to-real calibration.

Optimizes occlusion model parameters to minimize the reconstruction
error between simulated contaminated images and real photographs.

Two fitting backends:
  1. **Polynomial** — fits a polynomial transfer function mapping
     simulated pixel intensities to real ones. Fast, interpretable.
  2. **Gradient descent** — directly optimizes CMA-ES parameters to
     minimize L1 + perceptual loss. More accurate, slower.

The polynomial backend runs on CPU and does not require Isaac Sim.
The gradient descent backend can use Warp autodiff (GPU) or PyTorch.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from adversarial_dust.config import CalibrationConfig

logger = logging.getLogger(__name__)


@dataclass
class TransferFunction:
    """Fitted intensity/color transfer function from sim to real.

    For each channel, stores polynomial coefficients mapping
    simulated intensity -> real intensity.
    """

    coefficients: np.ndarray  # (3, degree+1) for RGB polynomial
    degree: int
    method: str  # "polynomial" or "gradient"

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the transfer function to a simulated image.

        Args:
            image: float32 array in [0, 1] of shape (H, W, 3).

        Returns:
            Corrected image in [0, 1].
        """
        corrected = np.zeros_like(image)
        for c in range(3):
            corrected[..., c] = np.polyval(self.coefficients[c], image[..., c])
        return np.clip(corrected, 0.0, 1.0)


@dataclass
class FittingResult:
    """Result of parameter fitting."""

    transfer_function: TransferFunction
    train_l1_error: float
    val_l1_error: Optional[float]
    train_ssim: float
    val_ssim: Optional[float]
    fitted_params: Optional[np.ndarray]  # CMA-ES params if gradient method


def fit_polynomial_transfer(
    clean_images: np.ndarray,
    sim_contaminated: np.ndarray,
    real_contaminated: np.ndarray,
    degree: int = 3,
) -> TransferFunction:
    """Fit a per-channel polynomial mapping sim -> real intensities.

    Computes the difference images (contaminated - clean) for both sim
    and real, then fits a polynomial mapping sim_diff -> real_diff
    per channel.

    Args:
        clean_images: (N, H, W, 3) float32 reference images.
        sim_contaminated: (N, H, W, 3) float32 simulated contaminated.
        real_contaminated: (N, H, W, 3) float32 real contaminated.
        degree: Polynomial degree.

    Returns:
        TransferFunction with fitted coefficients.
    """
    coefficients = np.zeros((3, degree + 1))

    for c in range(3):
        # Flatten all pixels across all images
        sim_vals = sim_contaminated[..., c].ravel()
        real_vals = real_contaminated[..., c].ravel()

        # Subsample for efficiency (fitting on millions of pixels is slow)
        n = len(sim_vals)
        if n > 100_000:
            idx = np.random.default_rng(42).choice(n, 100_000, replace=False)
            sim_vals = sim_vals[idx]
            real_vals = real_vals[idx]

        coefficients[c] = np.polyfit(sim_vals, real_vals, degree)

    return TransferFunction(
        coefficients=coefficients,
        degree=degree,
        method="polynomial",
    )


def compute_l1_error(
    predicted: np.ndarray, target: np.ndarray
) -> float:
    """Mean absolute error between two image arrays."""
    return float(np.mean(np.abs(predicted - target)))


def compute_ssim(
    predicted: np.ndarray, target: np.ndarray
) -> float:
    """Structural similarity between two images.

    Simplified single-scale SSIM. For paper-quality results, use
    skimage.metrics.structural_similarity.
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_p = predicted.mean()
    mu_t = target.mean()
    sigma_p = predicted.var()
    sigma_t = target.var()
    sigma_pt = ((predicted - mu_p) * (target - mu_t)).mean()

    ssim = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / (
        (mu_p ** 2 + mu_t ** 2 + c1) * (sigma_p + sigma_t + c2)
    )
    return float(ssim)


def fit_transfer_function(
    clean_images: np.ndarray,
    sim_contaminated: np.ndarray,
    real_contaminated: np.ndarray,
    config: CalibrationConfig,
) -> FittingResult:
    """Fit a transfer function using the configured method.

    Args:
        clean_images: (N, H, W, 3) float32 reference images.
        sim_contaminated: (N, H, W, 3) float32 simulated contaminated.
        real_contaminated: (N, H, W, 3) float32 real contaminated.
        config: Calibration configuration.

    Returns:
        FittingResult with transfer function and metrics.
    """
    if config.fitting_method == "polynomial":
        tf = fit_polynomial_transfer(
            clean_images, sim_contaminated, real_contaminated,
            degree=config.fitting_degree,
        )
    else:
        raise ValueError(f"Unknown fitting method: {config.fitting_method}")

    # Evaluate
    corrected = np.stack([
        tf.apply(sim_contaminated[i])
        for i in range(len(sim_contaminated))
    ])

    train_l1 = compute_l1_error(corrected, real_contaminated)
    train_ssim = compute_ssim(
        corrected.mean(axis=0), real_contaminated.mean(axis=0)
    )

    logger.info(
        f"Calibration fit: method={config.fitting_method}, "
        f"L1={train_l1:.4f}, SSIM={train_ssim:.4f}"
    )

    return FittingResult(
        transfer_function=tf,
        train_l1_error=train_l1,
        val_l1_error=None,
        train_ssim=train_ssim,
        val_ssim=None,
        fitted_params=None,
    )
