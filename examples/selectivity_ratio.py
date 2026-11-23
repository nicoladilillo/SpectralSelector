"""Select synthetic wavelengths using target-projection selectivity ratios."""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import SelectivityRatioSelector


def main() -> None:
    """Fit SR inside a preprocessing pipeline and print its diagnostics."""
    rng = np.random.default_rng(42)
    wavelengths = np.linspace(700.0, 900.0, 100)
    X = rng.normal(size=(120, wavelengths.size))
    y = 3 * X[:, 22] - 2 * X[:, 71] + rng.normal(scale=0.15, size=X.shape[0])

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "select",
                SelectivityRatioSelector(
                    estimator=PLSRegression(n_components=3, scale=False),
                    n_features_to_select=8,
                    wavelengths=wavelengths,
                ),
            ),
        ]
    )
    X_selected = pipeline.fit_transform(X, y)
    selector = pipeline.named_steps["select"]

    print("Selected matrix shape:", X_selected.shape)
    print("Selected indices:", selector.selected_indices_.tolist())
    print("Selected wavelengths:", selector.selected_wavelengths_.round(1).tolist())
    print("Selected SR scores:", selector.scores_[selector.support_].round(3).tolist())
    print(f"Selection time: {selector.selection_time_:.4f} s")


if __name__ == "__main__":
    main()
