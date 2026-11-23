"""Fit a compact regression BOSS example with component optimization."""

import numpy as np

from spectralselector import BOSSSelector

rng = np.random.default_rng(21)
X = rng.normal(size=(48, 20))
y = 2.5 * X[:, 3] - 1.8 * X[:, 14] + rng.normal(scale=0.2, size=X.shape[0])
wavelengths = np.linspace(900.0, 1700.0, X.shape[1])

selector = BOSSSelector(
    cv=3,
    max_components=4,
    n_submodels=20,
    top_fraction=0.2,
    max_iterations=4,
    random_state=21,
    wavelengths=wavelengths,
)
X_selected = selector.fit_transform(X, y)

print("Selected indices:", selector.selected_indices_)
print("Selected wavelengths:", selector.selected_wavelengths_)
print("Best PLS components:", selector.best_n_components_)
print("Reduced shape:", X_selected.shape)
