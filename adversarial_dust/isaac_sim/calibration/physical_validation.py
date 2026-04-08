"""Sim-to-real correlation analysis for physical validation.

Compares predicted success rates from simulation against measured
success rates on real hardware to quantify the sim-to-real gap.

Key metrics reported for the paper:
  - Parameter reconstruction error (L2 on synthetic ground-truth)
  - Image reconstruction quality (SSIM / PSNR)
  - Success rate prediction correlation (Spearman rho)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of sim-to-real validation for one contamination level."""

    budget_level: float
    sim_success_rate: float
    real_success_rate: float
    n_sim_episodes: int
    n_real_episodes: int


@dataclass
class ValidationReport:
    """Complete sim-to-real validation report."""

    results: List[ValidationResult]
    spearman_rho: float
    mean_absolute_error: float
    max_absolute_error: float

    def summary(self) -> str:
        lines = [
            "Sim-to-Real Validation Report",
            "=" * 40,
            f"Spearman rho: {self.spearman_rho:.3f}",
            f"Mean |sim - real|: {self.mean_absolute_error:.3f}",
            f"Max  |sim - real|: {self.max_absolute_error:.3f}",
            "",
            f"{'Budget':>8}  {'Sim SR':>8}  {'Real SR':>8}  {'Delta':>8}",
            "-" * 40,
        ]
        for r in self.results:
            delta = r.sim_success_rate - r.real_success_rate
            lines.append(
                f"{r.budget_level:8.2%}  {r.sim_success_rate:8.3f}  "
                f"{r.real_success_rate:8.3f}  {delta:+8.3f}"
            )
        return "\n".join(lines)


def compute_spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation coefficient.

    Uses the ranking approach without scipy dependency.
    """
    def _rank(arr):
        order = arr.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
        return ranks

    n = len(x)
    if n < 3:
        return float("nan")

    rx = _rank(x)
    ry = _rank(y)

    d = rx - ry
    rho = 1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1))
    return float(rho)


def validate_sim_to_real(
    sim_results: Dict[float, float],
    real_results: Dict[float, float],
    sim_episodes: int = 25,
    real_episodes: int = 10,
) -> ValidationReport:
    """Compare simulated vs real success rates across budget levels.

    Args:
        sim_results: {budget_level: mean_success_rate} from simulation.
        real_results: {budget_level: mean_success_rate} from real hardware.
        sim_episodes: Number of simulation episodes per budget.
        real_episodes: Number of real-world episodes per budget.

    Returns:
        ValidationReport with correlation metrics.
    """
    # Find common budget levels
    common_budgets = sorted(set(sim_results.keys()) & set(real_results.keys()))
    if not common_budgets:
        raise ValueError("No overlapping budget levels between sim and real results")

    results = []
    sim_srs = []
    real_srs = []

    for budget in common_budgets:
        sim_sr = sim_results[budget]
        real_sr = real_results[budget]
        results.append(ValidationResult(
            budget_level=budget,
            sim_success_rate=sim_sr,
            real_success_rate=real_sr,
            n_sim_episodes=sim_episodes,
            n_real_episodes=real_episodes,
        ))
        sim_srs.append(sim_sr)
        real_srs.append(real_sr)

    sim_arr = np.array(sim_srs)
    real_arr = np.array(real_srs)

    rho = compute_spearman_rho(sim_arr, real_arr)
    errors = np.abs(sim_arr - real_arr)

    report = ValidationReport(
        results=results,
        spearman_rho=rho,
        mean_absolute_error=float(errors.mean()),
        max_absolute_error=float(errors.max()),
    )

    logger.info(f"\n{report.summary()}")
    return report


def compute_psnr(
    predicted: np.ndarray, target: np.ndarray, max_val: float = 1.0
) -> float:
    """Peak signal-to-noise ratio between two images."""
    mse = np.mean((predicted - target) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(max_val ** 2 / mse))
