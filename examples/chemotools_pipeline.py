"""Demonstrate ndarray interoperability between chemotools and spectralselector."""

from __future__ import annotations

import numpy as np
from chemotools.scatter import StandardNormalVariate
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from spectralselector import VIPSelector


def main() -> None:
    """Cross-validate a chemotools preprocessing, VIP, and PLSR pipeline."""
    rng = np.random.default_rng(31)
    wavelengths = np.linspace(700.0, 900.0, 50)
    X = rng.uniform(0.5, 2.0, size=(60, 50))
    y = 2 * X[:, 12] - X[:, 37] + rng.normal(scale=0.05, size=60)

    pipeline = Pipeline(
        [
            ("snv", StandardNormalVariate()),
            (
                "select",
                VIPSelector(
                    n_components=2,
                    n_features_to_select=8,
                    wavelengths=wavelengths,
                ),
            ),
            ("model", PLSRegression(n_components=2)),
        ]
    )
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=4),
        scoring="neg_root_mean_squared_error",
    )
    print(f"Mean CV RMSE: {-scores.mean():.3f}")


if __name__ == "__main__":
    main()
