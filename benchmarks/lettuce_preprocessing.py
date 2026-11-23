"""Leakage-aware preprocessing helpers for the lettuce benchmark."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted


class LettucePreprocessor(TransformerMixin, BaseEstimator):
    """Apply SG or SG1 followed by MSC or SNV.

    Parameters are deliberately explicit because the article reports the four
    preprocessing combinations but does not state the SG window and polynomial
    order. The defaults (11, 3) reproduce the article's reported outlier counts,
    but those counts do not uniquely identify the original SG parameters.

    ``msc_reference_scope='training'`` is the leakage-safe default: the mean
    reference is learned by :meth:`fit` and reused for validation and test data.
    ``'batch'`` computes a new reference for every transformed matrix. This can
    reproduce workflows that normalize training and testing sets separately,
    but it is transductive because predictions depend on the other test samples.
    """

    def __init__(
        self,
        *,
        derivative=0,
        scatter="msc",
        window_length=11,
        polyorder=3,
        mode="nearest",
        msc_reference_scope="training",
    ):
        self.derivative = derivative
        self.scatter = scatter
        self.window_length = window_length
        self.polyorder = polyorder
        self.mode = mode
        self.msc_reference_scope = msc_reference_scope

    def fit(self, X, y=None):
        """Validate settings and learn the MSC training reference when needed."""
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        self._validate_parameters(X.shape[1])
        filtered = self._filter(X)
        if self.scatter == "msc":
            self.reference_ = filtered.mean(axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Apply the configured filtering and scatter correction."""
        check_is_fitted(self, "n_features_in_")
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} wavelengths; expected {self.n_features_in_}."
            )
        filtered = self._filter(X)
        if self.scatter == "snv":
            return self._snv(filtered)
        reference = (
            filtered.mean(axis=0)
            if self.msc_reference_scope == "batch"
            else self.reference_
        )
        return self._msc(filtered, reference)

    def _validate_parameters(self, n_features):
        if self.derivative not in (0, 1):
            raise ValueError("derivative must be 0 (SG) or 1 (SG1).")
        if self.scatter not in {"msc", "snv"}:
            raise ValueError("scatter must be 'msc' or 'snv'.")
        if self.msc_reference_scope not in {"training", "batch"}:
            raise ValueError("msc_reference_scope must be 'training' or 'batch'.")
        if not isinstance(self.window_length, int) or self.window_length < 3:
            raise ValueError("window_length must be an integer of at least 3.")
        if self.window_length > n_features:
            raise ValueError("window_length cannot exceed the wavelength count.")
        if not isinstance(self.polyorder, int) or not (
            0 <= self.polyorder < self.window_length
        ):
            raise ValueError("polyorder must be smaller than window_length.")

    def _filter(self, X):
        return savgol_filter(
            X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.derivative,
            axis=1,
            mode=self.mode,
        )

    @staticmethod
    def _snv(X):
        means = X.mean(axis=1, keepdims=True)
        scales = X.std(axis=1, keepdims=True)
        if np.any(scales <= np.finfo(np.float64).eps):
            raise ValueError("SNV cannot normalize a constant spectrum.")
        return (X - means) / scales

    @staticmethod
    def _msc(X, reference):
        reference_centered = reference - reference.mean()
        denominator = reference_centered @ reference_centered
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("MSC reference spectrum is constant.")
        means = X.mean(axis=1)
        slopes = ((X - means[:, None]) @ reference_centered) / denominator
        if np.any(np.abs(slopes) <= np.finfo(np.float64).eps):
            raise ValueError("MSC encountered a zero regression slope.")
        intercepts = means - slopes * reference.mean()
        return (X - intercepts[:, None]) / slopes[:, None]


class PCAMahalanobisOutlierDetector(BaseEstimator):
    """Detect spectra using PCA scores and squared Mahalanobis distance."""

    def __init__(self, *, n_components=10, confidence=0.95):
        self.n_components = n_components
        self.confidence = confidence

    def fit(self, X, y=None):
        """Fit PCA and the score-space covariance model."""
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        maximum = min(X.shape[0] - 1, X.shape[1])
        if self.n_components > maximum:
            raise ValueError(f"n_components cannot exceed {maximum} for this X.")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie strictly between zero and one.")
        self.pca_ = PCA(n_components=self.n_components).fit(X)
        scores = self.pca_.transform(X)
        self.score_mean_ = scores.mean(axis=0)
        score_covariance = np.atleast_2d(np.cov(scores, rowvar=False))
        self.score_precision_ = np.linalg.pinv(score_covariance)
        self.threshold_ = float(chi2.ppf(self.confidence, self.n_components))
        self.n_features_in_ = X.shape[1]
        self.training_distances_ = self.squared_distances(X)
        self.inlier_mask_ = self.training_distances_ <= self.threshold_
        return self

    def squared_distances(self, X):
        """Return squared Mahalanobis distances in the fitted PCA score space."""
        check_is_fitted(self, "score_precision_")
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} wavelengths; expected {self.n_features_in_}."
            )
        centered = self.pca_.transform(X) - self.score_mean_
        return np.einsum("ij,jk,ik->i", centered, self.score_precision_, centered)


def preprocessing_configurations(
    *, window_length=11, polyorder=3, msc_reference_scope="training"
):
    """Return the four SG/SG1 and MSC/SNV combinations used in the paper."""
    configurations = {}
    for derivative, filter_name in ((0, "SG"), (1, "SG1")):
        for scatter in ("msc", "snv"):
            name = f"{filter_name}+{scatter.upper()}"
            configurations[name] = LettucePreprocessor(
                derivative=derivative,
                scatter=scatter,
                window_length=window_length,
                polyorder=polyorder,
                msc_reference_scope=msc_reference_scope,
            )
    return configurations
