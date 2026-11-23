"""Remove raw-data Hotelling T² outliers before a train/test split.

This reproduces a *global* filtering protocol. Because the PCA and outlier
threshold see the complete dataset, do not use this protocol to claim an
unbiased estimate of performance on unseen samples. For predictive validation,
fit preprocessing and outlier detection on each training fold instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from chemotools.outliers import HotellingT2
from numpy.typing import ArrayLike, NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y


def filter_then_split(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_components: int = 10,
    confidence: float = 0.95,
    test_size: float = 0.3,
    random_state: int = 42,
    stratify: bool = False,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[Any],
    NDArray[Any],
    NDArray[np.bool_],
]:
    """Fit Hotelling T² on all raw samples, remove outliers, then split."""
    X_array, y_array = check_X_y(X, y, dtype=np.float64, ensure_all_finite=True)
    maximum = min(X_array.shape[0] - 1, X_array.shape[1])
    if not 1 <= n_components <= maximum:
        raise ValueError(f"n_components must be between 1 and {maximum}.")

    pca = PCA(n_components=n_components).fit(X_array)
    detector = HotellingT2(model=pca, confidence=confidence).fit(X_array)
    inlier_mask = detector.predict(X_array) == 1

    X_filtered = X_array[inlier_mask]
    y_filtered = y_array[inlier_mask]
    stratification = y_filtered if stratify else None
    split = train_test_split(
        X_filtered,
        y_filtered,
        test_size=test_size,
        random_state=random_state,
        stratify=stratification,
    )
    return (*split, inlier_mask)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    spectra = rng.normal(size=(80, 30))
    target = 2 * spectra[:, 4] - spectra[:, 19] + rng.normal(size=80)
    spectra[0] += 12  # Deliberate raw-spectrum outlier for the demonstration.

    X_train, X_test, y_train, y_test, kept = filter_then_split(
        spectra,
        target,
        n_components=5,
    )
    print(f"Removed {(~kept).sum()} samples before splitting.")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
