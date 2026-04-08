"""Translate existing occlusion model configs to Warp kernel uniforms.

These adapters convert FingerprintConfig / GlareConfig parameters (the same
parameter space optimized by CMA-ES) into dictionaries of Warp kernel
uniform values. No GPU or Isaac Sim runtime is required — the translation
is pure data manipulation, making it fully testable without hardware.
"""

from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fingerprint adapter
# ---------------------------------------------------------------------------

# Parameter offsets within the flat CMA-ES vector (per-print block of 8)
_FP_CX = 0
_FP_CY = 1
_FP_ANGLE = 2
_FP_SCALE = 3
_FP_RIDGE_FREQ = 4
_FP_RIDGE_AMP = 5
_FP_OPACITY = 6
_FP_SMEAR_ANGLE = 7
_FP_PARAMS_PER_PRINT = 8

# Physics constants matching fingerprint_model.py
_GREASE_COLOR = np.array([0.38, 0.40, 0.42], dtype=np.float32)
_SCATTER_TINT = np.array([0.90, 0.95, 1.00], dtype=np.float32)
_DEFOCUS_FRAC = 0.015  # fraction of image diagonal


class IsaacFingerprintAdapter:
    """Translate FingerprintConfig CMA-ES params to Warp kernel uniforms.

    The uniform dict mirrors the struct fields expected by
    ``warp_kernels.fingerprint_kernel``.
    """

    def __init__(self, config, image_shape: tuple):
        self.config = config
        self.num_prints = config.num_prints
        self.image_h, self.image_w = image_shape[:2]
        diag = np.sqrt(self.image_h**2 + self.image_w**2)
        self.defocus_sigma = _DEFOCUS_FRAC * diag

    def translate(
        self, params: np.ndarray, timestep: int = 0
    ) -> Dict[str, object]:
        """Convert flat CMA-ES param vector to Warp kernel uniform dict.

        Args:
            params: Flat array of shape ``(num_prints * 8,)``.
            timestep: Episode timestep (unused for fingerprint, kept for API).

        Returns:
            Dictionary with keys matching the Warp kernel struct.
        """
        params = np.asarray(params, dtype=np.float64)
        centers = []
        angles = []
        scales = []
        ridge_freqs = []
        ridge_amps = []
        opacities = []
        smear_angles = []

        for i in range(self.num_prints):
            off = i * _FP_PARAMS_PER_PRINT
            p = params[off : off + _FP_PARAMS_PER_PRINT]

            centers.append([
                float(np.clip(p[_FP_CX], 0.0, 1.0)),
                float(np.clip(p[_FP_CY], 0.0, 1.0)),
            ])
            angles.append(float(p[_FP_ANGLE] % (2 * np.pi)))
            scales.append(float(np.clip(
                p[_FP_SCALE],
                self.config.scale_range[0],
                self.config.scale_range[1],
            )))
            ridge_freqs.append(float(np.clip(
                p[_FP_RIDGE_FREQ],
                self.config.freq_range[0],
                self.config.freq_range[1],
            )))
            ridge_amps.append(float(np.clip(p[_FP_RIDGE_AMP], 0.0, 1.0)))
            opacities.append(float(np.clip(
                p[_FP_OPACITY], 0.0, self.config.max_opacity
            )))
            smear_angles.append(float(p[_FP_SMEAR_ANGLE] % np.pi))

        return {
            "num_prints": self.num_prints,
            "centers": np.array(centers, dtype=np.float32),
            "angles": np.array(angles, dtype=np.float32),
            "scales": np.array(scales, dtype=np.float32),
            "ridge_freqs": np.array(ridge_freqs, dtype=np.float32),
            "ridge_amps": np.array(ridge_amps, dtype=np.float32),
            "opacities": np.array(opacities, dtype=np.float32),
            "smear_angles": np.array(smear_angles, dtype=np.float32),
            "defocus_sigma": float(self.defocus_sigma),
            "grease_color": _GREASE_COLOR.copy(),
            "scatter_tint": _SCATTER_TINT.copy(),
            "image_height": self.image_h,
            "image_width": self.image_w,
        }


# ---------------------------------------------------------------------------
# Glare adapter
# ---------------------------------------------------------------------------

_GL_SOURCE_X = 0
_GL_SOURCE_Y = 1
_GL_INTENSITY = 2
_GL_STREAK_ANGLE = 3
_GL_HAZE_SPREAD = 4
_GL_STREAK_LENGTH = 5
_GL_NUM_PARAMS = 6


class IsaacGlareAdapter:
    """Translate GlareConfig CMA-ES params to Warp kernel uniforms.

    The uniform dict mirrors the struct fields expected by
    ``warp_kernels.glare_kernel``.
    """

    def __init__(self, config, image_shape: tuple):
        self.config = config
        self.image_h, self.image_w = image_shape[:2]

    def translate(
        self, params: np.ndarray, timestep: int = 0
    ) -> Dict[str, object]:
        """Convert flat CMA-ES param vector to Warp kernel uniform dict.

        Args:
            params: Flat array of shape ``(6,)``.
            timestep: Episode timestep (unused for glare, kept for API).

        Returns:
            Dictionary with keys matching the Warp kernel struct.
        """
        params = np.asarray(params, dtype=np.float64)

        source_x = float(np.clip(params[_GL_SOURCE_X], 0.0, 1.0))
        source_y = float(np.clip(params[_GL_SOURCE_Y], 0.0, 1.0))
        intensity = float(np.clip(
            params[_GL_INTENSITY], 0.0, self.config.max_intensity
        ))
        streak_angle = float(params[_GL_STREAK_ANGLE] % 360.0)
        haze_spread = float(np.clip(params[_GL_HAZE_SPREAD], 0.5, 3.0))
        streak_length = float(np.clip(params[_GL_STREAK_LENGTH], 0.5, 3.0))

        return {
            "source_x": source_x,
            "source_y": source_y,
            "intensity": intensity,
            "streak_angle_deg": streak_angle,
            "haze_spread": haze_spread,
            "streak_length": streak_length,
            "num_streaks": self.config.num_streaks,
            "chromatic_aberration": self.config.chromatic_aberration,
            "ghost_count": self.config.ghost_count,
            "ghost_spacing": self.config.ghost_spacing,
            "ghost_decay": self.config.ghost_decay,
            "image_height": self.image_h,
            "image_width": self.image_w,
        }
