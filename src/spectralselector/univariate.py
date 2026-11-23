"""Univariate wavelength selection with scikit-learn score functions.

Reference status
----------------
This module is a scikit-learn-compatible filter wrapper rather than a new named
chemometric algorithm. Statistical scoring is delegated to the configured
scikit-learn function, such as ``f_regression``, ``f_classif``, or mutual
information.

Method implemented here
-----------------------
Every wavelength is scored independently against y during ``fit``. The selector
retains the k highest finite scores, breaks exact ties by original column index,
and exposes both score order and source-order support. The score function sees
only the training fold when this selector is placed inside a pipeline.

Scientific scope
----------------
Univariate filters cannot discover interactions that are informative only as a
joint wavelength combination. The score definition, target assumptions, and
multiple-testing implications belong to the configured statistical function.

Computational cost
------------------
Every feature is scored once. F tests are linear in the input matrix size;
mutual-information estimators are typically more expensive and may use
``random_state`` for reproducible neighbor-based estimation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.feature_selection import (
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector

_SCORE_FUNCTIONS = {
    "f_classif": f_classif,
    "f_regression": f_regression,
    "mutual_info_classif": mutual_info_classif,
    "mutual_info_regression": mutual_info_regression,
}


class WavelengthSelectKBest(BaseWavelengthSelector):
    """Select the highest-scoring individual variables.

    Parameters
    ----------
    k : int or {"all"}, default=10
        Number of variables to select, or ``"all"``.
    score_func : {"f_regression", "f_classif", "mutual_info_regression",
            "mutual_info_classif"}, default="f_regression"
        One of ``"f_regression"``, ``"f_classif"``,
        ``"mutual_info_regression"``, or ``"mutual_info_classif"``.
    wavelengths : array-like of shape (n_features,), optional
        Optional wavelength value for every input column.
    random_state : int, optional
        Seed used by mutual-information score functions.
    """

    def __init__(
        self,
        k: int | str = 10,
        *,
        score_func: str = "f_regression",
        wavelengths: ArrayLike | None = None,
        random_state: int | None = None,
    ):
        """Store the univariate scoring and output-size configuration."""
        self.k = k
        self.score_func = score_func
        self.wavelengths = wavelengths
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> WavelengthSelectKBest:
        """Score each column independently and retain the highest scores."""
        if self.score_func not in _SCORE_FUNCTIONS:
            choices = ", ".join(sorted(_SCORE_FUNCTIONS))
            raise ValueError(f"score_func must be one of: {choices}.")
        is_regression = self.score_func.endswith("regression")
        X_array, y_array = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
            multi_output=False,
            y_numeric=is_regression,
        )
        self._validate_X(X_array, reset=True)
        if not is_regression:
            target_kind = type_of_target(y_array, raise_unknown=True)
            if target_kind not in {"binary", "multiclass"}:
                raise ValueError(
                    "Classification score functions require binary or "
                    "multiclass labels."
                )
        if self.k != "all":
            if not isinstance(self.k, (int, np.integer)) or isinstance(self.k, bool):
                raise TypeError("k must be a positive integer or 'all'.")
            if self.k < 1 or self.k > self.n_features_in_:
                raise ValueError(
                    "k must be between 1 and the number of input features "
                    f"(n_features = {self.n_features_in_}), or 'all'."
                )

        # Step 1: score every wavelength on its own against y (F test or MI).
        score_function = _SCORE_FUNCTIONS[self.score_func]
        if self.score_func.startswith("mutual_info"):
            scores = score_function(X_array, y_array, random_state=self.random_state)
            pvalues = None
        else:
            scores, pvalues = score_function(X_array, y_array)

        self.scores_ = np.asarray(scores, dtype=np.float64)
        self.pvalues_ = (
            None if pvalues is None else np.asarray(pvalues, dtype=np.float64)
        )
        # Step 2: rank by decreasing score (NaN last, ties to the lower index)
        # and keep the top k.
        sortable = np.nan_to_num(self.scores_, nan=-np.inf)
        order = np.lexsort((np.arange(self.n_features_in_), -sortable))
        n_selected = self.n_features_in_ if self.k == "all" else int(self.k)
        selected_order = order[:n_selected]
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected_order] = True
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selected_order,
        )
        return self
