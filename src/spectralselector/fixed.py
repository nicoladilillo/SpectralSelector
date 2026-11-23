"""Fixed feature-index wavelength selection.

Reference status
----------------
``IndexSelector`` is a deterministic utility, not a reproduction of a named
scientific algorithm. No primary-paper equivalence is claimed.

Method implemented here
-----------------------
The user supplies unique zero-based column indices. ``fit`` validates those
indices against the observed feature count, constructs the support mask, and
records optional physical wavelengths. ``transform`` returns the same columns
in their original source order.

Scientific scope
----------------
This selector is useful for applying a wavelength set fixed by prior knowledge
or for constructing leakage-free random/fixed benchmark baselines. It does not
learn from X or y and does not assess whether the supplied variables are
predictive.

Computational cost
------------------
``fit`` validates the requested indices and constructs one Boolean support mask;
``transform`` copies only the selected columns. No predictive model or internal
cross-validation is fitted.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ._base import BaseWavelengthSelector


class IndexSelector(BaseWavelengthSelector):
    """Select explicitly supplied zero-based feature indices.

    Parameters
    ----------
    indices : array-like of int
        Unique zero-based column indices to retain.
    wavelengths : array-like of float, optional
        Optional wavelength value for every input column. Values are metadata;
        selection remains index based.
    """

    def __init__(self, indices: ArrayLike, *, wavelengths: ArrayLike | None = None):
        """Store fixed-index selection configuration without inspecting data."""
        self.indices = indices
        self.wavelengths = wavelengths

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> IndexSelector:
        """Validate requested indices against the fitted input feature count."""
        self._validate_X(X, reset=True)
        indices = np.asarray(self.indices)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("indices must be a non-empty one-dimensional sequence.")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("indices must contain integers.")
        indices = indices.astype(np.int64, copy=False)
        if np.any(indices < 0) or np.any(indices >= self.n_features_in_):
            raise ValueError(
                f"indices must be between 0 and {self.n_features_in_ - 1}."
            )
        if np.unique(indices).size != indices.size:
            raise ValueError("indices must not contain duplicates.")

        # No search: the user-supplied indices are the selection, in given order.
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[indices] = True
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=indices,
        )
        return self
