"""Successive Projections Algorithm (SPA) wavelength selection.

Reference
---------
Araújo, M. C. U., Saldanha, T. C. B., Galvão, R. K. H., Yoneyama, T.,
Chame, H. C., & Visani, V. (2001). The successive projections algorithm for
variable selection in spectroscopic multicomponent analysis. Chemometrics and
Intelligent Laboratory Systems, 57(1), 65-73.
https://doi.org/10.1016/S0169-7439(01)00119-8

Paper algorithm implemented here
--------------------------------
Given a starting wavelength ``k(0)`` and a length ``N`` (Section 2.2):

0. Let ``x_j`` be the j-th column of the calibration matrix.
1. ``S`` is the set of wavelengths not selected yet.
2. Project every ``x_j`` onto the subspace orthogonal to the last selected
   (already projected) vector: ``Px_j = x_j - (x_j' x_k) x_k / (x_k' x_k)``
   (Eq. 1).
3. Select ``k(n) = argmax ||Px_j||`` over ``S``.
4. Replace ``x_j`` by ``Px_j`` and repeat until ``N`` wavelengths are chosen.

``N*`` and ``k*(0)`` are chosen by validating an MLR model on every starting
wavelength and every ``N`` and keeping the pair with the smallest RMSE
(Section 2.2, Eqs. 2-3). As in the paper's application (Section 4), columns
are mean-centered and autoscaled before the projections by default, so the
selection depends on the angles between wavelength vectors.

Package behavior and extensions
-------------------------------
SPA uses only ``X``; the response enters only through the validation model.
The paper's ``k*(0)``/``N*`` search is run with the package's leakage-safe
prefix validation: in every inner fold the projection order is recomputed on
the training rows only, every prefix is scored with ``estimator`` (default
``LinearRegression``, the paper's MLR), and the (start, length) pair with the
best mean score is kept (ties: fewer variables, then the lower start index).
The final order is fitted on all rows from that start. A fixed ``initial``
skips the start search. Classification (default
``LinearDiscriminantAnalysis``, the classification analogue of MLR), any
scikit-learn scorer, ``param_grid``, ``preprocessor``, ``autoscale=False``
(raw columns, Step 0 of the paper) and the ``max_features`` cap are package
extensions. The paper's lower bound ``N >= number of analytes`` is not
enforced.

Computational cost
------------------
One projection order costs ``O(N * n_samples * n_features)``. With
``initial="auto"`` the length search is repeated for every one of the
``n_features`` starts, i.e. ``n_features * n_splits * max_features`` model fits
(times the grid size); the paper notes the same ``J^2`` growth. Fix ``initial``
or restrict the spectral range for wide spectra.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import is_classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._length_search import PrefixLengthSearchMixin


def _is_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


class SPASelector(PrefixLengthSearchMixin, BaseWavelengthSelector):
    """Select minimally collinear wavelengths by successive projections.

    Parameters
    ----------
    n_features_to_select : int or {"auto"}, default="auto"
        Number of wavelengths ``N``, or ``"auto"`` to choose it (jointly with
        the start when ``initial="auto"``) by cross-validation.
    initial : int or {"auto"}, default="auto"
        Starting wavelength index ``k(0)``, or ``"auto"`` to try every column
        as in the paper.
    max_features : int, default=20
        Largest ``N`` evaluated, clipped to the sample and feature limits.
    autoscale : bool, default=True
        Mean-center and scale columns to unit variance before projecting (the
        paper's application). ``False`` projects the raw columns.
    estimator : estimator object, optional
        Model used only to validate start and length. ``None`` uses
        ``LinearRegression`` (MLR) for regression and
        ``LinearDiscriminantAnalysis`` for classification.
    task : {"auto", "regression", "classification"}, default="auto"
        Prediction task; ``"auto"`` infers it from the estimator and target.
    scoring : str or callable, optional
        Any scikit-learn scorer. Defaults to negative RMSE for regression and
        accuracy for classification.
    cv : int, CV splitter, or -1, default=5
        Inner validation folds; ``-1`` selects leave-one-out.
    random_state : int, optional
        Seed for shuffled K-fold splits created from an integer ``cv``.
    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength associated with every input column.
    param_grid : dict or list of dict, optional
        Hyper-parameter grid of ``estimator`` searched jointly with the length.
    preprocessor : transformer, optional
        Column-preserving transformer refitted on the training rows of every
        fold and on all rows; ``transform`` applies it before selecting.
    n_jobs : int, optional
        Parallel jobs over validation folds and grid points.

    Attributes
    ----------
    initial_ : int
        Selected starting wavelength index ``k*(0)``.
    initial_scores_ : ndarray of shape (n_starts, n_lengths)
        Mean validation score for every evaluated start (rows, in column order
        when ``initial="auto"``) and length; constant columns are not tried as
        starts and score ``-inf``. Only the chosen start's row is detailed in
        ``validation_scores_`` and ``cv_results_``.
    full_selection_order_ : ndarray of shape (n_lengths,)
        Projection order fitted on all rows from ``initial_``.
    max_features_ : int
        Evaluated length ``n_lengths``, clipped to the sample and feature
        limits.
    n_features_selected_ : int
        Number of retained wavelengths.
    validation_scores_, validation_score_std_, rmsecv_, cv_results_,
    best_n_features_, best_params_, best_score_, estimator_ :
        Length-validation diagnostics of the chosen start (see
        :class:`FCovSelSelector`).
    """

    def __init__(
        self,
        n_features_to_select: int | str = "auto",
        *,
        initial: int | str = "auto",
        max_features: int = 20,
        autoscale: bool = True,
        estimator=None,
        task: str = "auto",
        scoring=None,
        cv=5,
        random_state: int | None = None,
        wavelengths: ArrayLike | None = None,
        param_grid=None,
        preprocessor=None,
        n_jobs: int | None = None,
    ):
        """Store SPA selection and validation configuration."""
        self.n_features_to_select = n_features_to_select
        self.initial = initial
        self.max_features = max_features
        self.autoscale = autoscale
        self.estimator = estimator
        self.task = task
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.wavelengths = wavelengths
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = None):
        """Choose start and length by validation, then fit the full order."""
        X_array, y_array = check_X_y(
            X, y, dtype=np.float64, multi_output=True, y_numeric=False
        )
        self._validate_X(X_array, reset=True)
        n_samples, n_features = X_array.shape
        if not _is_int(self.max_features) or self.max_features < 1:
            raise ValueError("max_features must be a positive integer.")
        if self.initial != "auto" and (
            not _is_int(self.initial) or not 0 <= self.initial < n_features
        ):
            raise ValueError("initial must be 'auto' or a column index.")
        splits = self._setup_length_search(X_array, y_array, groups)
        self.max_features_ = min(
            int(self.max_features),
            self._max_prefix_length(n_samples, n_features, splits),
        )
        requested = self.n_features_to_select
        if requested != "auto" and (
            not _is_int(requested) or not 1 <= requested <= self.max_features_
        ):
            raise ValueError(
                "n_features_to_select must be 'auto' or an integer between 1 "
                "and max_features."
            )

        # Paper Section 2.2 criterion: try every start k(0) (or the fixed one) and
        # score every length N with fold-local validation.
        starts = range(n_features) if self.initial == "auto" else [int(self.initial)]
        # A constant (autoscaled) or all-zero (raw) column projects nothing.
        usable = np.ptp(X_array, axis=0) > 0 if self.autoscale else X_array.any(axis=0)
        table = []
        for start in starts:
            if len(starts) > 1 and not usable[start]:
                table.append(np.full(self.max_features_, -np.inf))
                continue
            self.initial_ = start
            table.append(
                self._run_length_search(X_array, y_array, self.max_features_, splits)
            )
        self.initial_scores_ = np.vstack(table)
        # N* = best length over all starts, k*(0) = best start for that length.
        if requested == "auto":
            # argmax returns the first maximum: fewer variables, then lower start.
            best_per_length = self.initial_scores_.max(axis=0)
            length = int(np.argmax(best_per_length))
        else:
            length = int(requested) - 1
        self.initial_ = int(starts[int(np.argmax(self.initial_scores_[:, length]))])
        if len(starts) > 1:  # restore the chosen start's diagnostics
            self._run_length_search(X_array, y_array, self.max_features_, splits)
        requested = self._choose_length(length + 1)

        # Final selection: SPA order on all rows from k*(0), cut at N*.
        X_model = self._fit_full_preprocessor(X_array, y_array)
        self.full_selection_order_ = self._fold_order(
            X_model, y_array, self.max_features_
        )
        selected_order = self.full_selection_order_[:requested]
        mask = np.zeros(n_features, dtype=bool)
        mask[selected_order] = True
        self.n_features_selected_ = requested
        self._fit_final_estimator(X_model, y_array, selected_order)
        self._finalize_selection(
            mask, wavelengths=self.wavelengths, selection_order=selected_order
        )
        return self

    def _default_length_estimator(self, y):
        """MLR as in the paper; LDA, its classification analogue."""
        if is_classifier(super()._default_length_estimator(y)):
            return LinearDiscriminantAnalysis()
        return LinearRegression()

    def _fold_order(self, X: np.ndarray, y: np.ndarray, n_steps: int) -> np.ndarray:
        """SPA order from ``initial_`` on the supplied rows (y is unused)."""
        # Paper Section 4: mean-center and autoscale so only angles matter.
        if self.autoscale:
            std = X.std(axis=0)
            X = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)
        return self._projection_order(X, self.initial_, n_steps)

    @staticmethod
    def _projection_order(X: np.ndarray, initial: int, n_steps: int) -> np.ndarray:
        """Paper SPA steps 0-5 from column ``initial``."""
        # Step 0: x_j = j-th column of X (copied, the caller's X is untouched).
        projected = np.array(X, dtype=np.float64)
        tolerance = np.finfo(float).eps * max(np.max(np.sum(X**2, axis=0)), 1.0)
        order = np.empty(n_steps, dtype=np.int64)
        order[0] = initial
        for n in range(1, n_steps + 1):
            last = projected[:, order[n - 1]].copy()
            last_norm = last @ last
            if last_norm <= tolerance:
                raise ValueError(
                    "SPA reached a zero-norm projection. Reduce max_features, "
                    "choose another initial column, or remove constant columns."
                )
            if n == n_steps:
                break
            # Steps 2 and 4: project every column onto the subspace orthogonal to
            # x_k(n-1) (Eq. 1) and replace x_j by Px_j.
            projected -= np.outer(last, last @ projected) / last_norm
            # Step 1: only wavelengths not yet selected (set S) can be chosen.
            norms = np.sum(projected**2, axis=0)
            norms[order[:n]] = -np.inf
            # Step 3: k(n) = argmax ||Px_j|| (squared norm, same argmax).
            order[n] = np.argmax(norms)
            # Step 5: n = n + 1, repeat until N wavelengths are selected.
        return order
