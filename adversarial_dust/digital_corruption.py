"""Corruption models for cross-corruption generalization experiments.

Provides image-space corruption transforms as OcclusionModel subclasses,
following the ImageNet-C taxonomy (Hendrycks & Dietterich, ICLR 2019)
adapted for fixed-mount robot cameras.

Models:
  Sensor:       GaussianNoiseModel
  Digital:      JPEGCompressionModel
  Blur:         MotionBlurModel, DefocusBlurModel
  Atmospheric:  FogModel
  Illumination: LowLightModel
"""

import cv2
import numpy as np

from adversarial_dust.occlusion_model import ContaminationMode, OcclusionModel


# ===========================================================================
# Helper
# ===========================================================================

def _parse_shape(image_shape):
    h = image_shape[0]
    w = image_shape[1] if len(image_shape) > 1 else image_shape[0]
    return h, w


# ===========================================================================
# Gaussian Noise
# ===========================================================================

_NOISE_IDX_SIGMA = 0
_NOISE_IDX_BIAS = 1


class GaussianNoiseModel(OcclusionModel):
    """Additive Gaussian noise corruption.

    budget_level scales the maximum noise sigma. At budget=1.0, noise
    sigma can reach max_sigma (default 80/255 in uint8 space).
    """

    def __init__(self, budget_level: float, max_sigma: float = 80.0,
                 image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.max_sigma = max_sigma
        self.image_h, self.image_w = _parse_shape(image_shape)
        self.n_params = 2
        self.contamination_mode = ContaminationMode.POST_PROCESS
        self.lower_bounds = np.array([0.0, -0.5])
        self.upper_bounds = np.array([1.0, 0.5])

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        sigma_scale = np.clip(params[_NOISE_IDX_SIGMA], 0.0, 1.0)
        bias = np.clip(params[_NOISE_IDX_BIAS], -0.5, 0.5)
        sigma = sigma_scale * self.max_sigma * self.budget_level
        noise = np.random.default_rng(timestep).normal(
            0, max(sigma, 1e-6), image.shape)
        result = image.astype(np.float32) + noise + bias * 255.0 * self.budget_level
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        coverage = np.clip(params[_NOISE_IDX_SIGMA], 0.0, 1.0) * self.budget_level
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return np.array([0.5, 0.0])

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)


# ===========================================================================
# JPEG Compression
# ===========================================================================

_JPEG_IDX_QUALITY = 0


class JPEGCompressionModel(OcclusionModel):
    """JPEG compression artifact corruption.

    budget_level controls the minimum quality factor. At budget=1.0,
    quality can drop to min_quality (default 1, maximum artifacts).
    """

    def __init__(self, budget_level: float, min_quality: int = 1,
                 max_quality: int = 95, image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.image_h, self.image_w = _parse_shape(image_shape)
        self.n_params = 1
        self.contamination_mode = ContaminationMode.POST_PROCESS
        self.lower_bounds = np.array([0.0])
        self.upper_bounds = np.array([1.0])

    def _get_quality(self, params: np.ndarray) -> int:
        severity = np.clip(params[_JPEG_IDX_QUALITY], 0.0, 1.0)
        effective = severity * self.budget_level
        quality = int(self.max_quality - effective *
                      (self.max_quality - self.min_quality))
        return max(self.min_quality, min(self.max_quality, quality))

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        quality = self._get_quality(params)
        _, encoded = cv2.imencode(".jpg", image,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        quality = self._get_quality(params)
        coverage = 1.0 - quality / self.max_quality
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return np.array([0.5])

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)


# ===========================================================================
# Motion Blur
# ===========================================================================

_MBLUR_IDX_INTENSITY = 0  # kernel size scale [0, 1]
_MBLUR_IDX_ANGLE = 1      # blur direction [0, 1] -> [0, 180] degrees


class MotionBlurModel(OcclusionModel):
    """Linear motion blur from camera shake or object motion.

    budget_level scales the maximum kernel size. Kernel is a 1D line
    rotated to the specified angle.
    """

    def __init__(self, budget_level: float, max_kernel: int = 45,
                 image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.max_kernel = max_kernel
        self.image_h, self.image_w = _parse_shape(image_shape)
        self.n_params = 2
        self.contamination_mode = ContaminationMode.POST_PROCESS
        self.lower_bounds = np.array([0.0, 0.0])
        self.upper_bounds = np.array([1.0, 1.0])

    def _get_kernel(self, params: np.ndarray) -> np.ndarray:
        intensity = np.clip(params[_MBLUR_IDX_INTENSITY], 0.0, 1.0)
        angle_frac = np.clip(params[_MBLUR_IDX_ANGLE], 0.0, 1.0)

        ksize = max(3, int(intensity * self.max_kernel * self.budget_level))
        if ksize % 2 == 0:
            ksize += 1
        angle = angle_frac * 180.0

        # Create motion blur kernel: horizontal line rotated by angle
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        kernel[ksize // 2, :] = 1.0 / ksize
        center = (ksize / 2.0, ksize / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
        # Re-normalize after rotation (warpAffine can lose mass)
        s = kernel.sum()
        if s > 0:
            kernel /= s
        return kernel

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        kernel = self._get_kernel(params)
        return cv2.filter2D(image, -1, kernel)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        intensity = np.clip(params[_MBLUR_IDX_INTENSITY], 0.0, 1.0)
        coverage = intensity * self.budget_level
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return np.array([0.5, 0.25])

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)


# ===========================================================================
# Defocus Blur
# ===========================================================================

_DEFOCUS_IDX_INTENSITY = 0  # blur sigma scale [0, 1]


class DefocusBlurModel(OcclusionModel):
    """Defocus (out-of-focus) blur via Gaussian kernel.

    budget_level scales the maximum blur sigma. Simulates focus
    mismatch on a fixed-mount camera.
    """

    def __init__(self, budget_level: float, max_sigma: float = 12.0,
                 image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.max_sigma = max_sigma
        self.image_h, self.image_w = _parse_shape(image_shape)
        self.n_params = 1
        self.contamination_mode = ContaminationMode.POST_PROCESS
        self.lower_bounds = np.array([0.0])
        self.upper_bounds = np.array([1.0])

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        intensity = np.clip(params[_DEFOCUS_IDX_INTENSITY], 0.0, 1.0)
        sigma = intensity * self.max_sigma * self.budget_level
        if sigma < 0.5:
            return image.copy()
        ksize = int(sigma * 6) | 1  # ensure odd
        ksize = max(3, ksize)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        intensity = np.clip(params[_DEFOCUS_IDX_INTENSITY], 0.0, 1.0)
        coverage = intensity * self.budget_level
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return np.array([0.5])

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)


# ===========================================================================
# Fog / Haze
# ===========================================================================

_FOG_IDX_INTENSITY = 0  # fog opacity [0, 1]
_FOG_IDX_BRIGHTNESS = 1  # fog brightness offset [-0.3, 0.3]


class FogModel(OcclusionModel):
    """Uniform fog / haze via alpha blend with white.

    Simulates atmospheric scattering that reduces contrast and shifts
    brightness toward white. budget_level scales maximum fog opacity.
    """

    def __init__(self, budget_level: float,
                 image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.image_h, self.image_w = _parse_shape(image_shape)
        self.n_params = 2
        self.contamination_mode = ContaminationMode.POST_PROCESS
        self.lower_bounds = np.array([0.0, -0.3])
        self.upper_bounds = np.array([1.0, 0.3])

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        intensity = np.clip(params[_FOG_IDX_INTENSITY], 0.0, 1.0)
        brightness = np.clip(params[_FOG_IDX_BRIGHTNESS], -0.3, 0.3)

        alpha = intensity * self.budget_level
        fog_value = 255.0 * (0.85 + brightness)  # slightly off-white default

        img_f = image.astype(np.float32)
        result = img_f * (1.0 - alpha) + fog_value * alpha
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        intensity = np.clip(params[_FOG_IDX_INTENSITY], 0.0, 1.0)
        coverage = intensity * self.budget_level
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return np.array([0.5, 0.0])

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)


# ===========================================================================
# Low-Light / Darkness
# ===========================================================================

_LOWLIGHT_IDX_GAMMA = 0      # gamma exponent scale [0, 1]
_LOWLIGHT_IDX_NOISE = 1      # read noise scale [0, 1]


class LowLightModel(OcclusionModel):
    """Low-light / darkness simulation via gamma darkening + sensor read noise.

    Gamma > 1 nonlinearly darkens the image (simulates reduced photon count),
    preserving highlight detail while crushing shadows. Read noise (Gaussian)
    is amplified in dark regions, matching real low-light camera behavior.
    budget_level scales both the gamma exponent and noise level.
    """

    def __init__(self, budget_level: float, max_gamma: float = 5.0,
                 max_noise_std: float = 25.0,
                 image_shape: tuple = (480, 640)):
        self.budget_level = budget_level
        self.max_gamma = max_gamma
        self.max_noise_std = max_noise_std
        self.image_h, self.image_w = _parse_shape(image_shape)
        self.n_params = 2
        self.contamination_mode = ContaminationMode.POST_PROCESS
        self.lower_bounds = np.array([0.0, 0.0])
        self.upper_bounds = np.array([1.0, 1.0])

    def apply(self, image: np.ndarray, params: np.ndarray,
              timestep: int = 0) -> np.ndarray:
        gamma_scale = np.clip(params[_LOWLIGHT_IDX_GAMMA], 0.0, 1.0)
        noise_scale = np.clip(params[_LOWLIGHT_IDX_NOISE], 0.0, 1.0)

        # Gamma darkening: gamma > 1 reduces brightness nonlinearly
        gamma = 1.0 + gamma_scale * (self.max_gamma - 1.0) * self.budget_level
        img_f = image.astype(np.float32) / 255.0
        darkened = np.power(img_f, gamma)

        # Sensor read noise (more visible in dark regions)
        noise_std = noise_scale * self.max_noise_std * self.budget_level / 255.0
        rng = np.random.default_rng(timestep)
        noise = rng.normal(0, max(noise_std, 1e-8), darkened.shape)
        result = darkened + noise

        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def get_alpha_mask(self, params: np.ndarray, timestep: int = 0) -> np.ndarray:
        gamma_scale = np.clip(params[_LOWLIGHT_IDX_GAMMA], 0.0, 1.0)
        coverage = gamma_scale * self.budget_level
        return np.full((self.image_h, self.image_w), coverage, dtype=np.float32)

    def compute_coverage(self, alpha_mask: np.ndarray) -> float:
        return float(np.mean(alpha_mask > 0.05))

    def get_cma_bounds(self):
        return self.lower_bounds.copy(), self.upper_bounds.copy()

    def get_cma_x0(self) -> np.ndarray:
        return np.array([0.5, 0.5])

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.lower_bounds, self.upper_bounds)
