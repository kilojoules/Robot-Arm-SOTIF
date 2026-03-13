"""CNN-based visual safety predictor for robot manipulation.

Predicts P(failure) from a single camera frame — the occluded image the policy
actually sees. Trained on (image, success/failure) pairs from adversarial
CMA-ES sweeps with fingerprint and glare occlusion. Evaluated on held-out
raindrop occlusion to test generalization.

The key idea: the predictor only has access to the perceived camera image,
with no knowledge of the occlusion type, budget, or parameters.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Defer torch imports to avoid breaking non-GPU environments
_torch = None
_nn = None
_F = None


def _ensure_torch():
    global _torch, _nn, _F
    if _torch is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        _torch = torch
        _nn = nn
        _F = F


class SafetyPredictorCNN:
    """Lightweight CNN that predicts P(failure) from a camera image.

    Architecture: 4 conv blocks → global avg pool → 2 FC layers → sigmoid.
    Input: (B, 3, H, W) normalized image.
    Output: (B, 1) probability of failure.
    """

    def __init__(self, image_size: Tuple[int, int] = (128, 128)):
        _ensure_torch()
        self.image_size = image_size
        self.model = _make_safety_net()
        self.device = "cpu"

    def to(self, device: str):
        self.device = device
        self.model = self.model.to(device)
        return self

    def preprocess(self, image: np.ndarray) -> "torch.Tensor":
        """Convert HWC uint8 image to normalized CHW tensor."""
        _ensure_torch()
        import cv2
        img = cv2.resize(image, self.image_size)
        # Normalize to ImageNet stats
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        # HWC -> CHW
        tensor = _torch.from_numpy(img.transpose(2, 0, 1)).float()
        return tensor

    def predict(self, image: np.ndarray) -> float:
        """Predict P(failure) for a single image. Returns float in [0, 1]."""
        self.model.eval()
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with _torch.no_grad():
            p_fail = self.model(tensor).item()
        return p_fail

    def predict_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Predict P(failure) for a batch of images."""
        self.model.eval()
        tensors = _torch.stack([self.preprocess(img) for img in images])
        tensors = tensors.to(self.device)
        with _torch.no_grad():
            p_fail = self.model(tensors).cpu().numpy().flatten()
        return p_fail

    def save(self, path: str):
        _ensure_torch()
        _torch.save({
            "model_state": self.model.state_dict(),
            "image_size": self.image_size,
        }, path)
        logger.info(f"Saved safety predictor to {path}")

    def load(self, path: str):
        _ensure_torch()
        ckpt = _torch.load(path, map_location=self.device, weights_only=True)
        self.image_size = ckpt["image_size"]
        self.model.load_state_dict(ckpt["model_state"])
        self.model = self.model.to(self.device)
        logger.info(f"Loaded safety predictor from {path}")


def _make_safety_net():
    """Build the SafetyNet nn.Module class after torch is imported."""
    _ensure_torch()

    class SafetyNet(_nn.Module):
        """4-layer CNN for binary safety classification."""

        def __init__(self):
            super().__init__()
            # Conv blocks: 3 -> 32 -> 64 -> 128 -> 256
            self.features = _nn.Sequential(
                _nn.Conv2d(3, 32, 3, padding=1), _nn.BatchNorm2d(32), _nn.ReLU(),
                _nn.MaxPool2d(2),
                _nn.Conv2d(32, 64, 3, padding=1), _nn.BatchNorm2d(64), _nn.ReLU(),
                _nn.MaxPool2d(2),
                _nn.Conv2d(64, 128, 3, padding=1), _nn.BatchNorm2d(128), _nn.ReLU(),
                _nn.MaxPool2d(2),
                _nn.Conv2d(128, 256, 3, padding=1), _nn.BatchNorm2d(256), _nn.ReLU(),
                _nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = _nn.Sequential(
                _nn.Flatten(),
                _nn.Linear(256, 64),
                _nn.ReLU(),
                _nn.Dropout(0.3),
                _nn.Linear(64, 1),
                _nn.Sigmoid(),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SafetyNet()


def train_safety_predictor(
    dataset_path: str,
    output_dir: str,
    image_size: Tuple[int, int] = (128, 128),
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cuda",
) -> SafetyPredictorCNN:
    """Train the safety predictor on collected episode data.

    Args:
        dataset_path: Path to .npz file with 'images' (N,H,W,3 uint8)
                      and 'labels' (N,) binary (1=failure, 0=success).
        output_dir: Where to save the trained model and training log.
        image_size: Resize images to this before feeding to CNN.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        val_split: Fraction of data for validation.
        device: 'cuda' or 'cpu'.

    Returns:
        Trained SafetyPredictorCNN.
    """
    _ensure_torch()
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import cv2

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = np.load(dataset_path)
    images_raw = data["images"]  # (N, H, W, 3) uint8
    labels = data["labels"].astype(np.float32)  # (N,) binary

    print(f"Dataset: {len(labels)} samples, {labels.sum():.0f} failures "
          f"({labels.mean():.1%} failure rate)")

    # Preprocess all images
    predictor = SafetyPredictorCNN(image_size=image_size)
    tensors = _torch.stack([predictor.preprocess(img) for img in images_raw])
    label_tensor = _torch.from_numpy(labels).float().unsqueeze(1)

    dataset = TensorDataset(tensors, label_tensor)

    # Train/val split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=_torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Training
    predictor.to(device)
    optimizer = _torch.optim.Adam(predictor.model.parameters(), lr=lr)
    criterion = _nn.BCELoss()

    history = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        predictor.model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = predictor.model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= n_train

        # Validate
        predictor.model.eval()
        val_loss = 0.0
        val_correct = 0
        with _torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = predictor.model(X)
                val_loss += criterion(pred, y).item() * len(X)
                val_correct += ((pred > 0.5) == y).sum().item()
        val_loss /= max(n_val, 1)
        val_acc = val_correct / max(n_val, 1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            predictor.save(str(out / "best_model.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.1%}")

    # Save training log
    with open(out / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model
    predictor.load(str(out / "best_model.pt"))
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")

    return predictor
