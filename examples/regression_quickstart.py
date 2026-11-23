"""Fit and evaluate a leakage-safe VIP/PLSR regression pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import VIPSelector


def make_spectra(random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create smooth synthetic spectra with three informative wavelength areas."""
    rng = np.random.default_rng(random_state)
    wavelengths = np.linspace(700.0, 900.0, 80)
    latent = rng.normal(size=(120, 5))
    smooth_basis = np.vstack(
        [
            np.exp(-0.5 * ((wavelengths - center) / width) ** 2)
            for center, width in [(720, 14), (760, 20), (805, 12), (850, 18), (885, 10)]
        ]
    )
    X = latent @ smooth_basis + rng.normal(scale=0.05, size=(120, 80))
    y = 2.5 * latent[:, 1] + rng.normal(scale=0.15, size=120)
    return X, y, wavelengths


def main() -> None:
    """Train on one split and print held-out RMSE and selected wavelengths."""
    X, y, wavelengths = make_spectra()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "select",
                VIPSelector(
                    estimator=PLSRegression(n_components=4, scale=False),
                    task="regression",
                    n_features_to_select=12,
                    wavelengths=wavelengths,
                ),
            ),
            ("model", PLSRegression(n_components=4, scale=False)),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test).ravel()
    selector = pipeline.named_steps["select"]

    print(f"Held-out RMSE: {root_mean_squared_error(y_test, predictions):.3f}")
    print("Selected indices:", selector.selected_indices_.tolist())
    print(
        "Selected wavelengths (nm):", selector.selected_wavelengths_.round(1).tolist()
    )


if __name__ == "__main__":
    main()
