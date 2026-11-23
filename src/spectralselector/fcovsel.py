"""Faster Covariate Selection (fCovSel) wavelength selection.

Reference
---------
Mishra, P. (2022).
A brief note on a new faster covariate's selection (fCovSel) algorithm.
Journal of Chemometrics, 36(5), e3397.
https://doi.org/10.1002/cem.3397

Paper algorithm implemented here
--------------------------------
fCovSel avoids repeatedly deflating the full predictor matrix. It selects the
variable with the largest covariance to the current Y residual, constructs an
orthonormal score sequence through Gram-Schmidt projection, and deflates Y along
each new score direction.

Package behavior and extensions
-------------------------------
The final support mask is a prefix of the paper's forward sequence fitted on all
data supplied to ``fit``. The implementation additionally repeats selection in
user-configurable validation folds and evaluates every sequence length with
PLSR for regression or PLSDA for classification. Any compatible scikit-learn
scorer can be used, and ``cv=-1`` selects leave-one-out validation. Validation
chooses only the length when ``n_features_to_select="auto"``; it does not replace
the paper order. Indicator encoding extends the selection criterion to binary
and multiclass classification. Cross-validated length selection and all
classification behavior are package extensions.

Model-specific sequence length
------------------------------
The forward order depends only on covariances, so it is the same whichever
model is used afterwards. The number of variables to keep, however, depends on
the model that will use them. With ``param_grid`` the length is therefore
chosen *jointly* with the hyper-parameters of ``estimator`` (which may be a
scikit-learn ``Pipeline``, e.g. autoscaling followed by a classifier): in every
validation fold the fCovSel order is recomputed on the training rows only, and
every prefix length is scored with every point of the grid. The length and
hyper-parameters with the best mean validation score are retained (ties: fewer
variables, then the first grid point). An optional ``preprocessor`` (for
example SG + MSC) is re-fitted on the training rows of every fold before the
fold order is computed, so data-dependent preprocessing never sees validation
rows. Both options are package extensions.

Computational cost
------------------
The forward order uses dense covariance/projection operations. Validation then
fits one PLSR or PLSDA model for every sequence prefix in every fold, plus one
final full-data model; cost therefore grows with ``n_splits * max_features``
predictive fits (multiplied by the number of grid points when ``param_grid``
is given; ``n_jobs`` parallelises the folds and grid points).

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._length_search import PrefixLengthSearchMixin


class FCovSelSelector(PrefixLengthSearchMixin, BaseWavelengthSelector):
    """Select variables by direct fCovSel with validated sequence length.

    The selector produces an ordered sequence of variable indices and an inner
    cross-validation score for every evaluated sequence length. Selection is
    performed entirely during ``fit``, so putting this estimator inside an outer
    pipeline prevents target leakage during model evaluation.

    Parameters
    ----------
    n_features_to_select : int or {"auto"}, default="auto"
        Number of variables to retain, or ``"auto"`` to use the sequence length
        with the best mean validation score.
    max_features : int, default=20
        Maximum sequence length evaluated by fCovSel.
    estimator : estimator object, optional
        Predictive estimator used only to validate sequence length. ``None``
        creates one-component PLSR for regression or PLSDA for classification.
    task : {"auto", "regression", "classification"}, default="auto"
        ``"regression"``, ``"classification"``, or ``"auto"`` inference from
        the estimator and target.
    scoring : str or callable, optional
        Any scikit-learn scorer accepted by ``check_scoring``. Defaults to
        negative root mean squared error for regression and accuracy for
        classification.
    cv : int, CV splitter, or -1, default=5
        Integer fold count, CV splitter, or ``-1`` for leave-one-out validation.
    random_state : int, optional
        Seed used when an integer ``cv`` creates shuffled K-fold splits.
    wavelengths : array-like of shape (n_features,), optional
        Optional wavelength value for every input column.
    param_grid : dict or list of dict, optional
        Hyper-parameter grid of ``estimator`` (scikit-learn ``ParameterGrid``
        syntax, e.g. ``{"clf__C": [0.1, 1, 10]}`` for a pipeline). When given,
        the sequence length and the hyper-parameters are selected jointly by the
        inner validation. Any parameter whose name ends with ``n_components`` is
        capped at the number of selected variables.
    preprocessor : transformer, optional
        Transformer applied before fCovSel (for example a chemotools SG + MSC
        pipeline). It is fitted on the training rows of every validation fold
        and on all rows for the final order; it must keep the number of columns.
        ``transform`` then applies the fitted preprocessor before selecting the
        columns, so ``fit`` and ``transform`` both expect unprocessed spectra.
    n_jobs : int, optional
        Parallel jobs over validation folds and grid points.

    Attributes
    ----------
    validation_scores_ : ndarray of shape (evaluated sequence length,)
        Mean higher-is-better validation score for every sequence length.
    validation_score_std_ : ndarray of shape (evaluated sequence length,)
        Standard deviation of fold scores for every sequence length.
    rmsecv_ : ndarray of shape (evaluated sequence length,) or None
        Mean fold RMSE when the selected scorer is
        ``"neg_root_mean_squared_error"``; otherwise ``None``.
    fold_selection_orders_ : ndarray of shape (n_splits, evaluated sequence length)
        Feature order fitted independently in every validation training fold.
    consensus_order_ : ndarray of shape (evaluated sequence length,)
        Unique deterministic aggregation of fold orders. This is a stability
        diagnostic and does not define the final support mask.
    full_selection_order_ : ndarray of shape (evaluated sequence length,)
        Paper fCovSel sequence fitted on all data supplied to ``fit``.
    max_features_ : int
        Evaluated sequence length, clipped to the sample and feature limits.
    n_features_selected_ : int
        Number of unique variables retained in the final mask.
    task_ : {"regression", "classification"}
        Resolved prediction task.
    scoring_ : str or callable
        Scorer used for sequence-length validation.
    cv_ : CV splitter
        Resolved cross-validation splitter.
    best_params_ : dict
        Hyper-parameters of ``estimator`` selected together with the length
        (empty when ``param_grid`` is None).
    best_n_features_ : int
        Selected sequence length.
    cv_results_ : dict
        ``n_features``, ``params``, ``mean_test_score`` and ``std_test_score``
        for every evaluated (length, grid point) pair.
    validation_params_ : list of dict
        Best grid point for every sequence length (the one reported in
        ``validation_scores_``).
    preprocessor_ : transformer or None
        Preprocessor fitted on all rows passed to ``fit``.
    """

    def __init__(
        self,
        n_features_to_select: int | str = "auto",
        *,
        max_features: int = 20,
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
        """Store fCovSel selection and validation configuration."""
        self.n_features_to_select = n_features_to_select
        self.max_features = max_features
        self.estimator = estimator
        self.task = task
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.wavelengths = wavelengths
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.n_jobs = n_jobs

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: ArrayLike | None = None,
    ) -> FCovSelSelector:
        """Fit the paper sequence and validate its length inside CV folds."""
        X_array, y_array = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
            multi_output=True,
            y_numeric=False,
        )
        self._validate_X(X_array, reset=True)
        n_samples, n_features = X_array.shape
        splits = self._setup_length_search(X_array, y_array, groups)
        max_allowed = self._max_prefix_length(n_samples, n_features, splits)
        if (
            not isinstance(self.max_features, (int, np.integer))
            or isinstance(self.max_features, bool)
            or self.max_features < 1
        ):
            raise ValueError("max_features must be a positive integer.")
        self.max_features_ = min(int(self.max_features), max_allowed)
        if self.n_features_to_select != "auto" and (
            not isinstance(self.n_features_to_select, (int, np.integer))
            or isinstance(self.n_features_to_select, bool)
            or not 1 <= self.n_features_to_select <= self.max_features_
        ):
            raise ValueError(
                "n_features_to_select must be 'auto' or an integer between 1 "
                "and max_features."
            )

        # The publication defines one forward sequence on all data passed to
        # fit. Inner CV below evaluates sequence length and stability; it must not
        # replace that paper sequence with cross-fold modal variables.
        X_model = self._fit_full_preprocessor(X_array, y_array)
        self.full_selection_order_ = self._fold_order(
            X_model, y_array, self.max_features_
        )
        self._run_length_search(X_array, y_array, self.max_features_, splits)
        requested = self._choose_length(self.n_features_to_select)

        selected_order = self.full_selection_order_[:requested]
        mask = np.zeros(n_features, dtype=bool)
        mask[selected_order] = True
        self.n_features_selected_ = int(selected_order.size)
        self._fit_final_estimator(X_model, y_array, selected_order)
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selected_order,
        )
        return self

    def _fold_order(self, X: np.ndarray, y: np.ndarray, n_steps: int) -> np.ndarray:
        """Paper fCovSel order fitted on the supplied (training) rows."""
        # Center X and Y (indicator-encoded classes for classification).
        Y = self._numeric_response(y)
        return self._forward_order(X - X.mean(axis=0), Y - Y.mean(axis=0), n_steps)

    def _numeric_response(self, y: np.ndarray) -> np.ndarray:
        """Return regression responses or indicator-encoded class responses."""
        if self.task_ == "classification":
            y = LabelBinarizer().fit_transform(y)
        return np.asarray(y, dtype=np.float64).reshape(len(y), -1)

    @staticmethod
    def _forward_order(
        X_centered: np.ndarray, Y_centered: np.ndarray, n_steps: int
    ) -> np.ndarray:
        """Return the paper fCovSel order (Mishra 2022, Table 2)."""
        n_samples, n_features = X_centered.shape
        if n_steps > n_features:
            raise ValueError(
                "fCovSel exhausted unselected variables before max_features."
            )
        scores = np.empty((n_samples, n_steps))
        order = np.empty(n_steps, dtype=np.int64)
        Y_residual = Y_centered.copy()

        for k in range(n_steps):
            # Mishra (2022) Table 2, steps 1-2: largest total absolute covariance
            # to the current Y residual among the unselected variables.
            criterion = np.abs(X_centered.T @ Y_residual).sum(axis=1)
            criterion[order[:k]] = -np.inf
            order[k] = np.argmax(criterion)

            # Score: the selected column orthogonalized against earlier scores
            # (Gram-Schmidt replaces X deflation), then normalized.
            column = X_centered[:, order[k]]
            score = column - scores[:, :k] @ (scores[:, :k].T @ column)
            norm = np.linalg.norm(score)
            if norm <= np.finfo(float).eps:
                raise ValueError(
                    "fCovSel encountered a zero-variance orthogonal component. "
                    "Reduce max_features or remove collinear variables."
                )
            scores[:, k] = score / norm
            # Mishra Table 2 response deflation along the score; X is not deflated.
            Y_residual -= np.outer(scores[:, k], scores[:, k] @ Y_residual)

        return order
