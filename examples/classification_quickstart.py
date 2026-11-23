"""Select wavelengths for PLS-DA classification inside a leakage-safe pipeline.

The classifier is chemotools' PLS-DA (``chemotools.classification.PLSDA``),
which assigns samples to the nearest class centroid in PLS score space. It is
available in chemotools releases after 0.5.0; with older releases the example
falls back to :class:`spectralselector.PLSDA` (PLS scores followed by LDA).
Both default to autoscaled X, so the example only sets ``n_components``.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from spectralselector import MIVIFSelector, VIPSelector

try:
    from chemotools.classification import PLSDA

    PLSDA_SOURCE = "chemotools.classification.PLSDA"
except ImportError:  # chemotools <= 0.5.0 or not installed
    from spectralselector import PLSDA

    PLSDA_SOURCE = "spectralselector.PLSDA (chemotools PLS-DA unavailable)"


def make_classification_spectra(
    random_state: int = 21,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create smooth spectra whose class signal lies in two absorption bands."""
    rng = np.random.default_rng(random_state)
    wavelengths = np.linspace(650.0, 950.0, 90)
    n_samples = 180
    baseline = rng.normal(1.0, 0.1, size=(n_samples, 1)) * np.exp(
        -((wavelengths - 800.0) ** 2) / 2.0e4
    )
    band_a = np.exp(-((wavelengths - 720.0) ** 2) / 60.0)
    band_b = np.exp(-((wavelengths - 880.0) ** 2) / 60.0)
    y = np.repeat(["controlled", "stressed"], n_samples // 2)
    stressed = (y == "stressed")[:, None]
    X = (
        baseline
        + 0.08 * stressed * band_a
        - 0.05 * stressed * band_b
        + rng.normal(scale=0.02, size=(n_samples, wavelengths.size))
    )
    return X, y, wavelengths


def main() -> None:
    """Compare model-dependent (VIP) and model-free (MI-VIF) selection."""
    X, y, wavelengths = make_classification_spectra()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=5
    )
    print(f"Classifier: {PLSDA_SOURCE}")

    selectors = {
        # VIP is model-dependent: it reads the weights of a fitted PLS-DA.
        "VIP": VIPSelector(
            estimator=PLSDA(n_components=2),
            task="classification",
            n_features_to_select=12,
            wavelengths=wavelengths,
        ),
        # MI-VIF is a filter: it never fits the downstream model.
        "MI-VIF": MIVIFSelector(
            task="classification",
            random_state=0,
            wavelengths=wavelengths,
        ),
    }
    for name, selector in selectors.items():
        pipeline = Pipeline([("select", selector), ("model", PLSDA(n_components=2))])
        pipeline.fit(X_train, y_train)
        score = balanced_accuracy_score(y_test, pipeline.predict(X_test))
        fitted = pipeline.named_steps["select"]
        print(
            f"{name:>6}: {fitted.selected_indices_.size:2d} wavelengths, "
            f"held-out balanced accuracy {score:.3f}"
        )
        print("        nm:", np.round(fitted.selected_wavelengths_, 1).tolist())


if __name__ == "__main__":
    main()
