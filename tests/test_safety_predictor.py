"""Tests for the CNN safety predictor (CPU only, no SimplerEnv needed)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from adversarial_dust.safety_predictor import SafetyPredictorCNN, train_safety_predictor


@pytest.fixture
def sample_images():
    """Generate fake images: half 'clean' (bright), half 'occluded' (dark)."""
    rng = np.random.default_rng(42)
    n = 40
    h, w = 64, 64
    # Success images: bright
    success_imgs = rng.integers(150, 256, size=(n // 2, h, w, 3), dtype=np.uint8)
    # Failure images: dark (simulating heavy occlusion)
    failure_imgs = rng.integers(0, 80, size=(n // 2, h, w, 3), dtype=np.uint8)
    images = np.concatenate([success_imgs, failure_imgs])
    labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)
    return images, labels


@pytest.fixture
def dataset_path(sample_images, tmp_path):
    path = tmp_path / "test_data.npz"
    images, labels = sample_images
    np.savez(str(path), images=images, labels=labels)
    return str(path)


class TestSafetyPredictorCNN:
    def test_predict_returns_probability(self):
        predictor = SafetyPredictorCNN(image_size=(32, 32))
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        p = predictor.predict(img)
        assert 0.0 <= p <= 1.0

    def test_predict_batch(self):
        predictor = SafetyPredictorCNN(image_size=(32, 32))
        imgs = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                for _ in range(4)]
        preds = predictor.predict_batch(imgs)
        assert preds.shape == (4,)
        assert all(0 <= p <= 1 for p in preds)

    def test_save_load_roundtrip(self, tmp_path):
        predictor = SafetyPredictorCNN(image_size=(32, 32))
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        p_before = predictor.predict(img)

        path = str(tmp_path / "model.pt")
        predictor.save(path)

        loaded = SafetyPredictorCNN(image_size=(32, 32))
        loaded.load(path)
        p_after = loaded.predict(img)

        assert abs(p_before - p_after) < 1e-6

    def test_preprocess_shape(self):
        predictor = SafetyPredictorCNN(image_size=(64, 64))
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = predictor.preprocess(img)
        assert tensor.shape == (3, 64, 64)


class TestTraining:
    def test_train_produces_model(self, dataset_path, tmp_path):
        output_dir = str(tmp_path / "model_output")
        predictor = train_safety_predictor(
            dataset_path=dataset_path,
            output_dir=output_dir,
            image_size=(32, 32),
            epochs=3,
            batch_size=8,
            lr=1e-3,
            device="cpu",
            frames_per_episode=1,
        )
        assert Path(output_dir, "best_model.pt").exists()
        assert Path(output_dir, "training_log.json").exists()

        with open(Path(output_dir, "training_log.json")) as f:
            log = json.load(f)
        assert len(log) == 3
        assert "val_f1" in log[0]
        assert "val_precision" in log[0]

    def test_train_with_episode_split(self, tmp_path):
        """Test that episode-level splitting works with multi-frame episodes."""
        rng = np.random.default_rng(0)
        n_episodes = 10
        frames_per_ep = 5
        n = n_episodes * frames_per_ep
        images = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        labels = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)

        data_path = str(tmp_path / "data.npz")
        np.savez(data_path, images=images, labels=labels)

        predictor = train_safety_predictor(
            dataset_path=data_path,
            output_dir=str(tmp_path / "model"),
            image_size=(32, 32),
            epochs=2,
            batch_size=8,
            device="cpu",
            frames_per_episode=frames_per_ep,
        )
        assert predictor is not None
