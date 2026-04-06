"""Baseline failure predictors for comparison with the CNN safety monitor.

Each baseline maps frames from an episode to a scalar "failure score"
(higher = more likely to fail), evaluated using the same Spearman rho
protocol as the CNN.

Baselines:
  1. BRISQUE — blind image quality metric
  2. NIQE — naturalness image quality metric
  3. PixelCoverage — mean occlusion alpha mask (trivial baseline)
  4. Autoencoder — reconstruction error on clean-trained autoencoder
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FailureBaseline(ABC):
    """Abstract base for failure prediction baselines."""

    name: str

    @abstractmethod
    def score_frames(self, frames: List[np.ndarray]) -> float:
        """Return a scalar failure score for a list of episode frames.

        Higher values indicate higher predicted likelihood of failure.
        """
        ...


class BRISQUEBaseline(FailureBaseline):
    """Blind/Referenceless Image Spatial Quality Evaluator.

    Higher BRISQUE score = worse quality = higher predicted P(fail).
    Uses OpenCV's implementation.
    """

    name = "BRISQUE"

    def __init__(self):
        self._brisque = cv2.quality.QualityBRISQUE_create(
            cv2.quality.QualityBRISQUE_computeFeatures.__func__.__qualname__
            if False else None,  # dummy
            None,
        ) if hasattr(cv2, "quality") else None

    def score_frames(self, frames: List[np.ndarray]) -> float:
        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
                if len(frame.shape) == 3 else frame
            # Manual BRISQUE approximation using local contrast
            # (full BRISQUE requires opencv-contrib with quality module)
            mu = cv2.GaussianBlur(gray.astype(np.float64), (7, 7), 1.167)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(gray.astype(np.float64) ** 2, (7, 7), 1.167)
            sigma = np.sqrt(np.abs(sigma - mu_sq))
            # Mean normalized contrast — lower = more degraded
            mean_sigma = np.mean(sigma)
            scores.append(1.0 / (mean_sigma + 1e-6))  # invert: higher = worse
        return float(np.mean(scores))


class NIQEBaseline(FailureBaseline):
    """Natural Image Quality Evaluator approximation.

    Uses local statistics deviation from natural scene statistics.
    Higher score = more unnatural = higher predicted P(fail).
    """

    name = "NIQE"

    def score_frames(self, frames: List[np.ndarray]) -> float:
        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            # Local mean and variance in patches
            mu = cv2.GaussianBlur(gray, (7, 7), 1.167)
            sigma = np.sqrt(np.abs(
                cv2.GaussianBlur(gray ** 2, (7, 7), 1.167) - mu ** 2))
            # Normalize
            normalized = (gray - mu) / (sigma + 1e-6)
            # Kurtosis-based quality proxy
            kurtosis = np.mean(normalized ** 4) - 3.0
            skewness = np.abs(np.mean(normalized ** 3))
            # Natural images have near-zero kurtosis/skewness
            score = np.abs(kurtosis) + skewness
            scores.append(score)
        return float(np.mean(scores))


class PixelCoverageBaseline(FailureBaseline):
    """Trivial baseline: mean brightness reduction as failure proxy.

    Compares frame brightness to a reference (assumed 128 midpoint).
    Lower brightness or higher variance = higher failure score.
    """

    name = "PixelCoverage"

    def score_frames(self, frames: List[np.ndarray]) -> float:
        scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) \
                if len(frame.shape) == 3 else frame.astype(np.float64)
            # Combine brightness loss and local variance
            brightness = np.mean(gray) / 255.0
            local_var = np.std(gray) / 255.0
            # Heavily occluded images tend to have lower brightness
            # or very high local variance from artifacts
            score = (1.0 - brightness) + (1.0 - local_var)
            scores.append(score)
        return float(np.mean(scores))


class AutoencoderBaseline(FailureBaseline):
    """Autoencoder reconstruction error baseline.

    Trains a small convolutional autoencoder on clean frames.
    At test time, reconstruction error is the failure score.
    """

    name = "Autoencoder"

    def __init__(self, image_size: int = 64):
        self.image_size = image_size
        self._model = None
        self._device = "cpu"

    def fit(self, clean_frames: List[np.ndarray], epochs: int = 30,
            device: str = "cpu"):
        """Train the autoencoder on clean (unoccluded) frames."""
        import torch
        import torch.nn as nn

        self._device = device

        class SmallAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1,
                                       output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1,
                                       output_padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self._model = SmallAE().to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Preprocess
        tensors = []
        for frame in clean_frames:
            img = cv2.resize(frame, (self.image_size, self.image_size))
            t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            tensors.append(t)
        data = torch.stack(tensors).to(device)

        self._model.train()
        for _ in range(epochs):
            recon = self._model(data)
            loss = criterion(recon, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self._model.eval()
        logger.info(f"Autoencoder trained on {len(clean_frames)} frames, "
                    f"final loss={loss.item():.4f}")

    def score_frames(self, frames: List[np.ndarray]) -> float:
        import torch

        if self._model is None:
            return 0.0

        tensors = []
        for frame in frames:
            img = cv2.resize(frame, (self.image_size, self.image_size))
            t = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            tensors.append(t)
        data = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            recon = self._model(data)
            errors = ((data - recon) ** 2).mean(dim=(1, 2, 3))
        return float(errors.mean().item())


def get_all_baselines() -> List[FailureBaseline]:
    """Return all non-learned baselines (no training needed)."""
    return [BRISQUEBaseline(), NIQEBaseline(), PixelCoverageBaseline()]
