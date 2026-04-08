"""Tests for the sim-to-real calibration pipeline (no GPU needed)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from adversarial_dust.config import CalibrationConfig
from adversarial_dust.isaac_sim.calibration.param_fitter import (
    TransferFunction,
    compute_l1_error,
    compute_ssim,
    fit_polynomial_transfer,
    fit_transfer_function,
)
from adversarial_dust.isaac_sim.calibration.physical_validation import (
    ValidationReport,
    compute_psnr,
    compute_spearman_rho,
    validate_sim_to_real,
)
from adversarial_dust.isaac_sim.calibration.capture_protocol import (
    CalibrationDataset,
    CalibrationSample,
    load_calibration_npz,
    save_calibration_npz,
)


# -----------------------------------------------------------------------
# Transfer function fitting
# -----------------------------------------------------------------------


class TestPolynomialFit:
    def test_identity_fit(self):
        """Fitting y=x data should produce near-identity polynomial."""
        rng = np.random.default_rng(42)
        n = 5
        h, w = 64, 64
        clean = rng.uniform(0, 1, (n, h, w, 3)).astype(np.float32)
        # Sim and real are identical
        sim = clean * 0.8  # some attenuation
        real = clean * 0.8

        tf = fit_polynomial_transfer(clean, sim, real, degree=1)

        assert tf.degree == 1
        assert tf.method == "polynomial"
        # Coefficients should be close to [1, 0] (y = 1*x + 0)
        for c in range(3):
            assert abs(tf.coefficients[c, 0] - 1.0) < 0.1
            assert abs(tf.coefficients[c, 1]) < 0.1

    def test_known_quadratic(self):
        """Fit y = x^2 data, verify polynomial captures it."""
        rng = np.random.default_rng(0)
        n = 10
        h, w = 32, 32
        clean = rng.uniform(0, 1, (n, h, w, 3)).astype(np.float32)
        sim = clean.copy()
        real = clean ** 2  # quadratic relationship

        tf = fit_polynomial_transfer(clean, sim, real, degree=2)

        # Verify transform roughly matches x^2
        test_vals = np.linspace(0, 1, 100).astype(np.float32)
        for c in range(3):
            predicted = np.polyval(tf.coefficients[c], test_vals)
            expected = test_vals ** 2
            assert np.mean(np.abs(predicted - expected)) < 0.1


class TestTransferFunction:
    def test_apply_preserves_shape(self):
        coeffs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        tf = TransferFunction(coefficients=coeffs, degree=1, method="polynomial")

        image = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
        result = tf.apply(image)

        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_apply_clips_to_01(self):
        # Coefficients that would produce values > 1
        coeffs = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        tf = TransferFunction(coefficients=coeffs, degree=1, method="polynomial")

        image = np.ones((10, 10, 3), dtype=np.float32) * 0.8
        result = tf.apply(image)

        assert result.max() <= 1.0
        assert result.min() >= 0.0


class TestFitTransferFunction:
    def test_polynomial_method(self):
        config = CalibrationConfig(fitting_method="polynomial", fitting_degree=2)
        rng = np.random.default_rng(42)
        n, h, w = 3, 32, 32
        clean = rng.uniform(0, 1, (n, h, w, 3)).astype(np.float32)
        sim = clean * 0.9
        real = clean * 0.85

        result = fit_transfer_function(clean, sim, real, config)

        assert result.transfer_function.method == "polynomial"
        assert result.train_l1_error >= 0.0
        assert -1.0 <= result.train_ssim <= 1.0

    def test_unknown_method_raises(self):
        config = CalibrationConfig(fitting_method="unknown_method")
        dummy = np.zeros((1, 10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown fitting method"):
            fit_transfer_function(dummy, dummy, dummy, config)


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------


class TestMetrics:
    def test_l1_error_identical(self):
        a = np.ones((10, 10, 3), dtype=np.float32)
        assert compute_l1_error(a, a) == pytest.approx(0.0)

    def test_l1_error_positive(self):
        a = np.ones((10, 10, 3), dtype=np.float32)
        b = np.zeros((10, 10, 3), dtype=np.float32)
        assert compute_l1_error(a, b) == pytest.approx(1.0)

    def test_ssim_identical(self):
        a = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
        assert compute_ssim(a, a) == pytest.approx(1.0, abs=0.01)

    def test_psnr_identical(self):
        a = np.random.uniform(0, 1, (10, 10, 3)).astype(np.float32)
        assert compute_psnr(a, a) == float("inf")

    def test_psnr_finite(self):
        a = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        b = np.ones((10, 10, 3), dtype=np.float32) * 0.6
        psnr = compute_psnr(a, b)
        assert 0 < psnr < 100


# -----------------------------------------------------------------------
# Spearman correlation
# -----------------------------------------------------------------------


class TestSpearmanRho:
    def test_perfect_positive(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([10, 20, 30, 40, 50], dtype=float)
        assert compute_spearman_rho(x, y) == pytest.approx(1.0)

    def test_perfect_negative(self):
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        y = np.array([50, 40, 30, 20, 10], dtype=float)
        assert compute_spearman_rho(x, y) == pytest.approx(-1.0)

    def test_too_few_points(self):
        x = np.array([1, 2], dtype=float)
        y = np.array([3, 4], dtype=float)
        assert np.isnan(compute_spearman_rho(x, y))


# -----------------------------------------------------------------------
# Sim-to-real validation
# -----------------------------------------------------------------------


class TestValidateSimToReal:
    def test_basic_report(self):
        sim = {0.05: 0.95, 0.10: 0.85, 0.20: 0.60}
        real = {0.05: 0.90, 0.10: 0.80, 0.20: 0.55}

        report = validate_sim_to_real(sim, real)

        assert isinstance(report, ValidationReport)
        assert len(report.results) == 3
        assert report.mean_absolute_error >= 0
        assert -1 <= report.spearman_rho <= 1

    def test_no_overlap_raises(self):
        sim = {0.05: 0.9}
        real = {0.10: 0.8}

        with pytest.raises(ValueError, match="No overlapping"):
            validate_sim_to_real(sim, real)

    def test_summary_string(self):
        sim = {0.10: 0.80}
        real = {0.10: 0.75}
        report = validate_sim_to_real(sim, real)
        summary = report.summary()
        assert "Sim-to-Real" in summary
        assert "10.00%" in summary


# -----------------------------------------------------------------------
# Calibration data I/O
# -----------------------------------------------------------------------


class TestCalibrationDataIO:
    def test_save_and_load_npz(self, tmp_path):
        samples = [
            CalibrationSample(
                name="test_sample",
                clean_image=np.zeros((64, 64, 3), dtype=np.uint8),
                contaminated_image=np.ones((64, 64, 3), dtype=np.uint8) * 128,
                contamination_type="fingerprint",
                lighting="bright_diffuse",
            ),
        ]
        dataset = CalibrationDataset(samples=samples)
        npz_path = str(tmp_path / "test_calib.npz")

        save_calibration_npz(dataset, npz_path)
        loaded = load_calibration_npz(npz_path)

        assert loaded.num_samples == 1
        assert loaded.samples[0].contamination_type == "fingerprint"
        np.testing.assert_array_equal(
            loaded.samples[0].clean_image,
            np.zeros((64, 64, 3), dtype=np.uint8),
        )

    def test_filter_by_type(self):
        samples = [
            CalibrationSample("a", np.zeros((4, 4, 3), dtype=np.uint8),
                              np.zeros((4, 4, 3), dtype=np.uint8), "fingerprint", ""),
            CalibrationSample("b", np.zeros((4, 4, 3), dtype=np.uint8),
                              np.zeros((4, 4, 3), dtype=np.uint8), "glare", ""),
            CalibrationSample("c", np.zeros((4, 4, 3), dtype=np.uint8),
                              np.zeros((4, 4, 3), dtype=np.uint8), "fingerprint", ""),
        ]
        dataset = CalibrationDataset(samples=samples)
        fp_only = dataset.filter_by_type("fingerprint")
        assert fp_only.num_samples == 2

    def test_split(self):
        samples = [
            CalibrationSample(f"s{i}", np.zeros((4, 4, 3), dtype=np.uint8),
                              np.zeros((4, 4, 3), dtype=np.uint8), "fp", "")
            for i in range(10)
        ]
        dataset = CalibrationDataset(samples=samples)
        train, val = dataset.split(validation_fraction=0.3)
        assert train.num_samples + val.num_samples == 10
        assert val.num_samples >= 1

    def test_load_missing_npz_raises(self):
        with pytest.raises(FileNotFoundError):
            load_calibration_npz("/nonexistent/path.npz")

    def test_load_missing_keys_raises(self, tmp_path):
        npz_path = str(tmp_path / "bad.npz")
        np.savez(npz_path, something_else=np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="Missing keys"):
            load_calibration_npz(npz_path)
