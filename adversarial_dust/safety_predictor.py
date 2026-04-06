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
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Defer torch imports to avoid breaking non-GPU environments
_torch = None
_nn = None


def _ensure_torch():
    global _torch, _nn
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn


class SafetyPredictorCNN:
    """CNN that predicts P(failure) from a camera image.

    Architecture: backbone (frozen or fine-tuned) -> FC layers -> logit.
    Input: (B, 3, H, W) normalized image.
    Output: (B, 1) probability of failure (after sigmoid).
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        backbone: str = "resnet18",
        freeze_backbone: bool = True,
        head_hidden: int = 64,
        dropout: float = 0.3,
    ):
        _ensure_torch()
        self.image_size = image_size
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.model = _make_safety_net(
            backbone=backbone,
            freeze_backbone=freeze_backbone,
            head_hidden=head_hidden,
            dropout=dropout,
        )
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
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        tensor = _torch.from_numpy(img.transpose(2, 0, 1)).float()
        return tensor

    def predict(self, image: np.ndarray) -> float:
        """Predict P(failure) for a single image. Returns float in [0, 1]."""
        self.model.eval()
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with _torch.no_grad():
            logit = self.model(tensor)
            p_fail = _torch.sigmoid(logit).item()
        return p_fail

    def predict_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Predict P(failure) for a batch of images."""
        self.model.eval()
        tensors = _torch.stack([self.preprocess(img) for img in images])
        tensors = tensors.to(self.device)
        with _torch.no_grad():
            logits = self.model(tensors)
            p_fail = _torch.sigmoid(logits).cpu().numpy().flatten()
        return p_fail

    def save(self, path: str):
        _ensure_torch()
        _torch.save({
            "model_state": self.model.state_dict(),
            "image_size": self.image_size,
            "backbone": self.backbone,
            "freeze_backbone": self.freeze_backbone,
            "head_hidden": self.head_hidden,
            "dropout": self.dropout,
        }, path)
        logger.info(f"Saved safety predictor to {path}")

    def load(self, path: str):
        _ensure_torch()
        ckpt = _torch.load(path, map_location=self.device, weights_only=True)
        self.image_size = ckpt["image_size"]
        # Rebuild model if architecture params are in checkpoint
        if "backbone" in ckpt:
            self.backbone = ckpt["backbone"]
            self.freeze_backbone = ckpt.get("freeze_backbone", True)
            self.head_hidden = ckpt.get("head_hidden", 64)
            self.dropout = ckpt.get("dropout", 0.3)
            self.model = _make_safety_net(
                backbone=self.backbone,
                freeze_backbone=self.freeze_backbone,
                head_hidden=self.head_hidden,
                dropout=self.dropout,
            )
        self.model.load_state_dict(ckpt["model_state"])
        self.model = self.model.to(self.device)
        logger.info(f"Loaded safety predictor from {path}")


def _make_safety_net(
    backbone: str = "resnet18",
    freeze_backbone: bool = True,
    head_hidden: int = 64,
    dropout: float = 0.3,
):
    """Build a pretrained backbone + trainable classification head.

    Output is a raw logit (no sigmoid). Use BCEWithLogitsLoss for training
    and apply sigmoid at inference time for numerical stability.

    Args:
        backbone: One of "resnet18", "resnet50", "vit_b_16".
        freeze_backbone: If True, freeze all backbone weights.
        head_hidden: Hidden layer size in classification head.
        dropout: Dropout rate in classification head.
    """
    _ensure_torch()
    from torchvision import models

    class SafetyNet(_nn.Module):

        def __init__(self):
            super().__init__()

            if backbone == "resnet18":
                base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                feat_dim = base.fc.in_features  # 512
                base.fc = _nn.Identity()
            elif backbone == "resnet50":
                base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                feat_dim = base.fc.in_features  # 2048
                base.fc = _nn.Identity()
            elif backbone == "vit_b_16":
                base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
                feat_dim = base.heads.head.in_features  # 768
                base.heads = _nn.Identity()
            else:
                raise ValueError(f"Unknown backbone: {backbone}")

            if freeze_backbone:
                for param in base.parameters():
                    param.requires_grad = False

            self.backbone = base
            self.head = _nn.Sequential(
                _nn.Linear(feat_dim, head_hidden),
                _nn.ReLU(),
                _nn.Dropout(dropout),
                _nn.Linear(head_hidden, 1),
            )

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    return SafetyNet()


def train_safety_predictor(
    dataset_path: str,
    output_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.2,
    device: str = "cuda",
    frames_per_episode: int = 10,
    backbone: str = "resnet18",
    freeze_backbone: bool = True,
    head_hidden: int = 64,
    dropout: float = 0.3,
    backbone_lr: float = None,
    seed: int = 42,
    data_fraction: float = 1.0,
) -> SafetyPredictorCNN:
    """Train the safety predictor on collected episode data.

    Args:
        dataset_path: Path to .npz file with 'images' (N,H,W,3 uint8)
                      and 'labels' (N,) binary (1=failure, 0=success).
        output_dir: Where to save the trained model and training log.
        image_size: Resize images to this before feeding to CNN.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate for head (and backbone if backbone_lr is None).
        val_split: Fraction of episodes held out for validation.
        device: 'cuda' or 'cpu'.
        frames_per_episode: Number of consecutive frames per episode in the
                           dataset (used for episode-level train/val split).
        backbone: Backbone architecture ("resnet18", "resnet50", "vit_b_16").
        freeze_backbone: If True, freeze backbone weights.
        head_hidden: Hidden layer size in classification head.
        dropout: Dropout rate in classification head.
        backbone_lr: Separate learning rate for backbone when unfrozen.
                     Defaults to lr/10 if None and backbone is unfrozen.
        seed: Random seed for reproducibility.
        data_fraction: Fraction of training data to use (for learning curve ablation).

    Returns:
        Trained SafetyPredictorCNN.
    """
    _ensure_torch()
    from torch.utils.data import DataLoader, TensorDataset, Subset

    # Set seeds for reproducibility
    _torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    data = np.load(dataset_path)
    images_raw = data["images"]  # (N, H, W, 3) uint8
    labels = data["labels"].astype(np.float32)  # (N,) binary
    n_samples = len(labels)
    n_failures = int(labels.sum())
    n_success = n_samples - n_failures

    print(f"Dataset: {n_samples} samples, {n_failures} failures "
          f"({labels.mean():.1%} failure rate)", flush=True)
    print(f"Architecture: {backbone} (freeze={freeze_backbone}), "
          f"head_hidden={head_hidden}, dropout={dropout}", flush=True)

    # Preprocess all images
    predictor = SafetyPredictorCNN(
        image_size=image_size,
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        head_hidden=head_hidden,
        dropout=dropout,
    )
    tensors = _torch.stack([predictor.preprocess(img) for img in images_raw])
    label_tensor = _torch.from_numpy(labels).float().unsqueeze(1)

    # --- Episode-level train/val split ---
    # Frames are stored in contiguous blocks of frames_per_episode.
    # Split by episode to prevent data leakage between correlated frames.
    n_episodes = n_samples // frames_per_episode
    rng = np.random.default_rng(seed)
    episode_indices = rng.permutation(n_episodes)
    n_val_episodes = max(1, int(n_episodes * val_split))
    val_episodes = set(episode_indices[:n_val_episodes])

    train_idx, val_idx = [], []
    for ep in range(n_episodes):
        start = ep * frames_per_episode
        end = start + frames_per_episode
        if ep in val_episodes:
            val_idx.extend(range(start, end))
        else:
            train_idx.extend(range(start, end))

    # --- Data fraction subsampling (for learning curve ablation) ---
    if data_fraction < 1.0:
        n_keep = max(1, int(len(train_idx) * data_fraction))
        train_idx = list(rng.choice(train_idx, size=n_keep, replace=False))
        print(f"Data fraction: using {n_keep}/{len(train_idx) + n_keep - n_keep} "
              f"training samples ({data_fraction:.0%})", flush=True)

    dataset = TensorDataset(tensors, label_tensor)
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    n_train = len(train_idx)
    n_val = len(val_idx)

    print(f"Split: {n_train} train, {n_val} val "
          f"({n_val_episodes}/{n_episodes} episodes held out)", flush=True)

    # --- Class-weighted loss to handle imbalance ---
    if n_failures > 0 and n_success > 0:
        pos_weight = _torch.tensor([n_success / n_failures]).to(device)
    else:
        pos_weight = _torch.tensor([1.0]).to(device)
    print(f"Class balance: pos_weight={pos_weight.item():.1f} "
          f"(upweights failures {pos_weight.item():.1f}x)", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Training
    predictor.to(device)
    n_total = sum(p.numel() for p in predictor.model.parameters())

    if freeze_backbone:
        trainable = [p for p in predictor.model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        optimizer = _torch.optim.Adam(trainable, lr=lr)
    else:
        # Separate param groups: lower LR for backbone, full LR for head
        bb_lr = backbone_lr if backbone_lr is not None else lr / 10.0
        param_groups = [
            {"params": predictor.model.backbone.parameters(), "lr": bb_lr},
            {"params": predictor.model.head.parameters(), "lr": lr},
        ]
        n_trainable = n_total
        optimizer = _torch.optim.Adam(param_groups)
        print(f"Backbone LR: {bb_lr:.1e}, Head LR: {lr:.1e}", flush=True)

    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total "
          f"({100*n_trainable/n_total:.1f}%)", flush=True)
    scheduler = _torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = _nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Train
        predictor.model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = predictor.model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= n_train

        # Validate
        predictor.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_tp = val_fp = val_fn = val_tn = 0
        with _torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = predictor.model(X)
                val_loss += criterion(logits, y).item() * len(X)
                preds = (_torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y).sum().item()
                val_tp += ((preds == 1) & (y == 1)).sum().item()
                val_fp += ((preds == 1) & (y == 0)).sum().item()
                val_fn += ((preds == 0) & (y == 1)).sum().item()
                val_tn += ((preds == 0) & (y == 0)).sum().item()

        val_loss /= max(n_val, 1)
        val_acc = val_correct / max(n_val, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": precision,
            "val_recall": recall,
            "val_f1": f1,
            "lr": optimizer.param_groups[0]["lr"],
        })

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            predictor.save(str(out / "best_model.pt"))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, "
                  f"val_acc={val_acc:.1%}, "
                  f"P={precision:.2f} R={recall:.2f} F1={f1:.2f}",
                  flush=True)

    # Save training log
    with open(out / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best model
    predictor.load(str(out / "best_model.pt"))

    # Report baseline comparison
    majority_acc = max(n_failures, n_success) / n_samples
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}", flush=True)
    print(f"Majority-class baseline accuracy: {majority_acc:.1%}", flush=True)

    return predictor
