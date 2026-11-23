"""PLS-DA classifiers shared by the benchmark scripts.

Benchmarks use chemotools' PLS-DA (``chemotools.classification.PLSDA``,
nearest class centroid in PLS score space) whenever the installed chemotools
provides it. It is not part of chemotools 0.5.0 on PyPI, so older installations
fall back to :class:`spectralselector.PLSDA` (PLS scores followed by LDA).
``PLSDA_IMPLEMENTATION`` names the class actually used and is written into every
benchmark manifest, so cached results never mix the two silently.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from spectralselector import PLSDA as PackagePLSDA

try:
    from chemotools.classification import PLSDA as ChemotoolsPLSDA
except ImportError:  # chemotools <= 0.5.0 or not installed
    ChemotoolsPLSDA = None

PLSDA_IMPLEMENTATION = (
    "chemotools.classification.PLSDA"
    if ChemotoolsPLSDA is not None
    else "spectralselector.PLSDA"
)


def make_plsda(n_components: int = 2):
    """Return the benchmark PLS-DA without internal X scaling.

    Every benchmark pipeline autoscales X in an earlier ``StandardScaler`` step,
    so the classifier itself must not scale again.
    """
    if ChemotoolsPLSDA is not None:
        return ChemotoolsPLSDA(n_components=n_components, scale=False)
    return PackagePLSDA(n_components=n_components, scale_pls=False)


class CappedPLSDA(ClassifierMixin, BaseEstimator):
    """Benchmark PLS-DA whose component count never exceeds the input rank.

    Selectors such as MI-VIF may keep fewer wavelengths than the largest
    ``n_components`` in the tuning grid. The wrapper fits
    ``min(n_components, n_features, n_samples - 1)`` components instead of
    failing, and records the value used as ``n_components_``.
    """

    def __init__(self, n_components: int = 2):
        """Store the requested number of PLS components."""
        self.n_components = n_components

    def fit(self, X, y):
        """Fit the capped PLS-DA."""
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64)
        self.n_components_ = int(
            max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        )
        self.model_ = make_plsda(self.n_components_).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def _scores(self, X):
        check_is_fitted(self, "model_")
        return validate_data(self, X, reset=False, ensure_2d=True, dtype=np.float64)

    def predict(self, X):
        """Predict class labels."""
        return self.model_.predict(self._scores(X))

    def predict_proba(self, X):
        """Return the wrapped model's class probabilities."""
        return self.model_.predict_proba(self._scores(X))

    def decision_function(self, X):
        """Return the wrapped model's decision function."""
        return self.model_.decision_function(self._scores(X))
