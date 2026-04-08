"""Warp GPU kernel for fast budget projection via opacity histogram.

Instead of 30 binary-search renders, this kernel computes a 256-bin
histogram of the attenuation mask in a single GPU pass. The host then
walks the histogram to find the scale factor where coverage equals
the budget — replacing O(30 renders) with O(1 render + trivial CPU math).
"""

import numpy as np

_wp = None


def _ensure_warp():
    global _wp
    if _wp is None:
        import warp as wp
        wp.init()
        _wp = wp


def compute_budget_scale_gpu(
    attenuation_mask,  # wp.array2d on GPU
    budget_level: float,
    dirty_threshold: float,
    num_bins: int = 256,
) -> float:
    """Find the largest scale factor s such that coverage(mask*s) <= budget.

    Args:
        attenuation_mask: 2D GPU array of opacity values in [0, 1].
        budget_level: Maximum fraction of dirty pixels.
        dirty_threshold: Opacity threshold for "dirty" classification.
        num_bins: Histogram resolution.

    Returns:
        Scale factor in [0, 1].
    """
    _ensure_warp()
    wp = _wp

    h, w = attenuation_mask.shape
    total_pixels = h * w

    # Build histogram on GPU
    histogram = wp.zeros(num_bins, dtype=int)

    @wp.kernel
    def histogram_kernel(
        mask: wp.array2d(dtype=float),
        hist: wp.array(dtype=int),
        n_bins: int,
    ):
        i, j = wp.tid()
        val = wp.clamp(mask[i, j], 0.0, 1.0)
        bin_idx = wp.min(int(val * float(n_bins - 1)), n_bins - 1)
        wp.atomic_add(hist, bin_idx, 1)

    wp.launch(histogram_kernel, dim=(h, w), inputs=[attenuation_mask, histogram, num_bins])

    # Read histogram back to CPU
    hist_cpu = histogram.numpy()

    # Walk from top to find scale factor
    return _find_scale_from_histogram(
        hist_cpu, num_bins, total_pixels, budget_level, dirty_threshold
    )


def _find_scale_from_histogram(
    histogram: np.ndarray,
    num_bins: int,
    total_pixels: int,
    budget_level: float,
    dirty_threshold: float,
) -> float:
    """CPU-side binary search on the histogram to find budget-compliant scale.

    This is also usable standalone for testing without GPU.
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lo, hi = 0.0, 1.0
    for _ in range(30):
        s = (lo + hi) / 2.0
        # Count pixels that would exceed threshold after scaling
        dirty_count = 0
        for b in range(num_bins):
            if bin_centers[b] * s > dirty_threshold:
                dirty_count += histogram[b]
        coverage = dirty_count / total_pixels
        if coverage <= budget_level:
            lo = s
        else:
            hi = s

    return lo
