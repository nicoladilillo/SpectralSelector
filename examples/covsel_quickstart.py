"""Fit and evaluate a leakage-safe CovSel regression pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import CovSelSelector


def main() -> None:
    """Select synthetic wavelengths using only the training split."""
    rng = np.random.default_rng(42)
    wavelengths = np.linspace(700.0, 900.0, 100)
    X = rng.normal(size=(160, wavelengths.size))
    y = 4 * X[:, 18] - 2 * X[:, 72] + rng.normal(scale=0.2, size=X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
    )

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "select",
                CovSelSelector(
                    n_features_to_select=6,
                    task="regression",
                    wavelengths=wavelengths,
                ),
            ),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    selector = pipeline.named_steps["select"]

    print(f"Held-out RMSE: {root_mean_squared_error(y_test, predictions):.3f}")
    print("Selection order:", selector.selection_order_.tolist())
    selected_wavelengths = selector.wavelengths_in_[selector.selection_order_]
    print("Wavelength order:", selected_wavelengths.round(1).tolist())
    print("Cumulative Y variance:", selector.y_explained_variance_.round(3).tolist())
    print(f"Selection time: {selector.selection_time_:.4f} s")


if __name__ == "__main__":
    main()
