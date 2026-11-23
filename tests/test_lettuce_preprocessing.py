"""Tests for the benchmark's leakage-aware lettuce preprocessing."""

import numpy as np
import pytest

from benchmarks.lettuce_preprocessing import (
    LettucePreprocessor,
    PCAMahalanobisOutlierDetector,
    preprocessing_configurations,
)


@pytest.fixture
def spectra():
    wavelengths = np.linspace(0, 1, 31)
    return np.vstack(
        [
            1.0 + wavelengths + 0.1 * np.sin(8 * wavelengths),
            1.2 + 0.8 * wavelengths + 0.1 * np.sin(8 * wavelengths),
            0.9 + 1.1 * wavelengths + 0.1 * np.sin(8 * wavelengths),
            1.1 + 0.9 * wavelengths + 0.1 * np.sin(8 * wavelengths),
        ]
    )


@pytest.mark.parametrize("name", ["SG+MSC", "SG+SNV", "SG1+MSC", "SG1+SNV"])
def test_all_paper_preprocessing_combinations_are_finite(spectra, name):
    transformed = preprocessing_configurations()[name].fit_transform(spectra)
    assert transformed.shape == spectra.shape
    assert np.isfinite(transformed).all()


def test_msc_training_reference_is_not_recomputed_for_test_batch(spectra):
    preprocessor = LettucePreprocessor(scatter="msc").fit(spectra[:3])
    reference = preprocessor.reference_.copy()
    preprocessor.transform(spectra[3:])
    np.testing.assert_array_equal(preprocessor.reference_, reference)


def test_pca_mahalanobis_detector_exposes_threshold_and_mask():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(40, 12))
    detector = PCAMahalanobisOutlierDetector(n_components=5).fit(X)
    assert detector.inlier_mask_.shape == (40,)
    assert detector.threshold_ == pytest.approx(11.0704976935)
    assert np.isfinite(detector.squared_distances(X)).all()


def test_pca_mahalanobis_detector_supports_one_component():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 4))
    detector = PCAMahalanobisOutlierDetector(n_components=1).fit(X)
    assert detector.squared_distances(X).shape == (20,)
