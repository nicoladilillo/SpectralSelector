"""Covariance Selection (CovSel) wavelength selection.

Reference
---------
Roger, J.-M., Palagos, B., Bertrand, D., & Fernandez-Ahumada, E. (2011).
CovSel: Variable selection for highly multivariate and multi-response
calibration: Application to IR spectroscopy.
Chemometrics and Intelligent Laboratory Systems, 106(2), 216-223.
https://doi.org/10.1016/j.chemolab.2010.10.003

Paper algorithm implemented here
--------------------------------
CovSel centers X and Y, selects the X column maximizing global squared
covariance with all Y responses, and projects both residual blocks onto the
orthogonal complement of that selected X direction. Selection and joint X/Y
deflation repeat until the requested sequence length is reached.

Package behavior and scope
--------------------------
Regression supports one or multiple responses. ``scale_y=True`` provides the
autoscaled multi-response global stage described in the paper; the optional
second response-specific refinement is not implemented. Classification encodes
classes as indicator responses for selection. The ranking never uses a
predictive model.

Model-validated length (package extension)
------------------------------------------
With ``n_features_to_select="auto"`` the number of variables is chosen with a
user-supplied ``estimator`` (the model that will use the wavelengths): the
CovSel order is recomputed on the training rows of every validation fold, every
prefix is scored with the estimator (and optional ``param_grid``), and the best
mean score fixes the length. The order itself stays the model-free paper order
fitted on all rows. Roger et al. likewise choose the number of variables from
the validation error of the calibration model built on them.

Computational cost
------------------
CovSel performs one dense covariance calculation and two rank-one residual
projections per selected variable. With a fixed count it fits no predictive model;
``"auto"`` adds ``n_splits * max_features`` estimator fits (times the grid
size).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._length_search import PrefixLengthSearchMixin
from ._modeling import VALID_TASKS


class CovSelSelector(PrefixLengthSearchMixin, BaseWavelengthSelector):
    """Select variables by maximum covariance and orthogonal deflation.

    At each step, CovSel selects the X column with the largest squared global
    covariance with Y. Both matrices are then projected onto the orthogonal
    complement of that selected column. Repeating this process favors variables
    that explain new response information instead of redundant neighboring
    variables.

    Parameters
    ----------
    n_features_to_select : int or {"auto"}, default=10
        Exact number of variables to select, or ``"auto"`` to choose the
        number with ``estimator`` by cross-validation over the CovSel prefixes.
    task : {"auto", "regression", "classification"}, default="auto"
        Interpretation of the response. In automatic mode, binary and multiclass
        targets are indicator encoded; continuous targets are treated as
        regression. Set this explicitly when an integer response is quantitative.
    scale_y : bool, default=False
        Standardize each numeric response to unit sample standard deviation
        before selection. Roger et al.'s multi-response regression procedure
        uses autoscaled Y so responses in different units contribute comparably.
        Classification indicators are left unscaled.
    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength associated with every input column.
    max_features : int, default=20
        Longest prefix evaluated when ``n_features_to_select="auto"``.
    estimator : estimator object, optional
        Predictive model used *only* with ``n_features_to_select="auto"`` to
        choose how many CovSel variables to keep (the CovSel order never uses
        it). Pass the model you will fit on the selected wavelengths, e.g.
        ``PLSRegression``, a PLS-DA, or a ``Pipeline``. ``None`` creates
        one-component PLSR for regression or the package PLSDA for
        classification.
    scoring : str or callable, optional
        Scorer for length validation. Defaults to negative RMSE for regression
        and accuracy for classification.
    cv : int, CV splitter, or -1, default=5
        Inner validation folds for ``"auto"``; ``-1`` selects leave-one-out.
    random_state : int, optional
        Seed for shuffled K-fold splits created from an integer ``cv``.
    param_grid : dict or list of dict, optional
        Hyper-parameter grid of ``estimator`` searched jointly with the length.
        Parameters ending in ``n_components`` are capped at the prefix length.
    preprocessor : transformer, optional
        Column-preserving transformer fitted before CovSel on all rows (and on
        the training rows of every validation fold); ``transform`` applies it
        before selecting columns.
    n_jobs : int, optional
        Parallel jobs over validation folds and grid points.

    Attributes
    ----------
    selection_order_ : ndarray of shape (n_selected,)
        Selected indices in sequential CovSel order.
    full_selection_order_ : ndarray of shape (n_steps,)
        CovSel order fitted on all rows (``max_features`` long with
        ``"auto"``); the support is its first ``n_selected`` entries.
    covariance_scores_ : ndarray of shape (n_features_to_select,)
        Winning global squared covariance at each selection step.
    criterion_history_ : ndarray of shape (n_features_to_select, n_features)
        Global squared covariance for every available variable at every step.
        Previously selected variables contain ``NaN``.
    x_explained_variance_ : ndarray of shape (n_features_to_select,)
        Cumulative fraction of centered X sum of squares removed by deflation.
    y_explained_variance_ : ndarray of shape (n_features_to_select,)
        Cumulative fraction of centered Y sum of squares removed by deflation.
    task_ : {"regression", "classification"}
        Resolved response interpretation.
    y_scale_ : ndarray of shape (n_responses,)
        Per-response divisor applied before CovSel. All ones when response
        scaling is disabled or the task is classification.
    validation_scores_, best_n_features_, best_params_, cv_results_, estimator_
        Length-validation results, only with ``n_features_to_select="auto"``
        (same meaning as in :class:`FCovSelSelector`).

    Notes
    -----
    Selection is fitted only on the data passed to ``fit``. Put CovSel inside a
    scikit-learn pipeline for leakage-safe cross-validation. The desired number
    of variables should be tuned around that complete pipeline. For
    multi-response regression, response scaling affects the global covariance
    criterion. Set ``scale_y=True`` to reproduce the paper's autoscaled-Y global
    selection. The optional per-response refinement described by Roger et al. is
    outside this selector's current scope. For classification, labels are
    indicator encoded for selection; the downstream classifier remains a user
    pipeline decision.

    References
    ----------
    Roger, J.-M., Palagos, B., Bertrand, D., and Fernandez-Ahumada, E.
    (2011). CovSel: Variable selection for highly multivariate and
    multi-response calibration: Application to IR spectroscopy.
    *Chemometrics and Intelligent Laboratory Systems*, 106(2), 216-223.
    https://doi.org/10.1016/j.chemolab.2010.10.003
    """

    def __init__(
        self,
        n_features_to_select: int = 10,
        *,
        task: str = "auto",
        scale_y: bool = False,
        wavelengths: ArrayLike | None = None,
        max_features: int = 20,
        estimator=None,
        scoring=None,
        cv=5,
        random_state: int | None = None,
        param_grid=None,
        preprocessor=None,
        n_jobs: int | None = None,
    ):
        """Store the selection count, task, and length-validation settings."""
        self.n_features_to_select = n_features_to_select
        self.task = task
        self.scale_y = scale_y
        self.wavelengths = wavelengths
        self.max_features = max_features
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.n_jobs = n_jobs

    def fit(
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = None
    ) -> CovSelSelector:
        """Build the sequential CovSel order and support mask.

        ``groups`` is used only by group-aware ``cv`` splitters when
        ``n_features_to_select="auto"``.
        """
        X_array, y_array = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
            ensure_min_samples=2,
            multi_output=True,
            y_numeric=False,
        )
        self._validate_X(X_array, reset=True)
        n_samples, n_features = X_array.shape
        if self.task not in VALID_TASKS:
            raise ValueError("task must be 'auto', 'regression', or 'classification'.")
        if not isinstance(self.scale_y, (bool, np.bool_)):
            raise TypeError("scale_y must be a boolean.")

        auto = self.n_features_to_select == "auto"
        if auto:
            splits = self._setup_length_search(X_array, y_array, groups)
            max_allowed = self._max_prefix_length(n_samples, n_features, splits)
            if (
                not isinstance(self.max_features, (int, np.integer))
                or isinstance(self.max_features, bool)
                or self.max_features < 1
            ):
                raise ValueError("max_features must be a positive integer.")
            n_steps = min(int(self.max_features), max_allowed)
            order_task = self.task_
        else:
            max_count = min(n_samples - 1, n_features)
            if (
                not isinstance(self.n_features_to_select, (int, np.integer))
                or isinstance(self.n_features_to_select, bool)
                or not 1 <= self.n_features_to_select <= max_count
            ):
                raise ValueError(
                    f"n_features_to_select must be 'auto' or between 1 and {max_count}."
                )
            n_steps = int(self.n_features_to_select)
            order_task = self.task

        X_model = self._fit_full_preprocessor(X_array, y_array)
        Y, self.task_, classes = self._prepare_response(y_array, order_task)
        if classes is not None:
            self.classes_ = classes
        self.y_scale_ = np.ones(Y.shape[1], dtype=np.float64)
        if self.scale_y and self.task_ == "regression":
            self.y_scale_ = self._response_scale(Y)
            Y = Y / self.y_scale_

        (
            self.full_selection_order_,
            self.covariance_scores_,
            self.criterion_history_,
            self.x_explained_variance_,
            self.y_explained_variance_,
        ) = self._covsel(X_model, Y, n_steps)

        selected_order = self.full_selection_order_
        if auto:
            self._run_length_search(X_array, y_array, n_steps, splits)
            selected_order = selected_order[: self._choose_length("auto")]
            self._fit_final_estimator(X_model, y_array, selected_order)

        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected_order] = True
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selected_order,
        )
        return self

    def _fold_order(self, X: np.ndarray, y: np.ndarray, n_steps: int) -> np.ndarray:
        """CovSel order fitted on one validation training fold."""
        Y, _, _ = self._prepare_response(y, self.task_)
        if self.scale_y and self.task_ == "regression":
            Y = Y / self._response_scale(Y)
        return self._covsel(X, Y, n_steps)[0]

    @staticmethod
    def _response_scale(Y: np.ndarray) -> np.ndarray:
        """Sample standard deviation of every response column."""
        scale = np.std(Y, axis=0, ddof=1)
        if np.any(scale <= np.finfo(float).eps):
            raise ValueError(
                "scale_y=True requires variation in every response column."
            )
        return scale

    @staticmethod
    def _prepare_response(
        y: np.ndarray, task: str
    ) -> tuple[np.ndarray, str, np.ndarray | None]:
        """Return a centered-ready numeric response matrix and resolved task."""
        target_kind = type_of_target(y, raise_unknown=True)
        if task == "auto":
            resolved = (
                "classification"
                if target_kind in {"binary", "multiclass", "multilabel-indicator"}
                else "regression"
            )
        else:
            resolved = task

        if resolved == "classification":
            if target_kind in {"binary", "multiclass"}:
                encoder = LabelBinarizer()
                Y = encoder.fit_transform(y)
                return np.asarray(Y, dtype=np.float64), resolved, encoder.classes_
            if target_kind == "multilabel-indicator":
                return np.asarray(y, dtype=np.float64), resolved, np.arange(y.shape[1])
            raise ValueError(
                "Classification requires binary, multiclass, or multilabel targets."
            )

        if np.asarray(y).dtype.kind not in "biufc":
            raise ValueError("Regression requires a numeric response.")
        if target_kind not in {
            "continuous",
            "continuous-multioutput",
            "binary",
            "multiclass",
            "multiclass-multioutput",
        }:
            raise ValueError("Regression requires one or more numeric responses.")
        Y = np.asarray(y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return Y, resolved, None

    @staticmethod
    def _covsel(
        X: np.ndarray,
        Y: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run centered CovSel and return its order and diagnostic histories."""
        # Step 0: center X and Y; the residual blocks start as the centered data.
        X_residual = X - np.mean(X, axis=0, keepdims=True)
        Y_residual = Y - np.mean(Y, axis=0, keepdims=True)
        total_x = float(np.sum(X_residual**2))
        total_y = float(np.sum(Y_residual**2))
        if total_x <= np.finfo(float).eps:
            raise ValueError("CovSel requires variation in X.")
        if total_y <= np.finfo(float).eps:
            raise ValueError("CovSel requires variation in y.")

        n_features = X.shape[1]
        selected = np.zeros(n_features, dtype=bool)
        order = np.empty(n_steps, dtype=np.int64)
        winning_scores = np.empty(n_steps, dtype=np.float64)
        history = np.full((n_steps, n_features), np.nan, dtype=np.float64)
        explained_x = np.empty(n_steps, dtype=np.float64)
        explained_y = np.empty(n_steps, dtype=np.float64)
        criterion_tolerance = 0.0

        for step in range(n_steps):
            # Step 1, Roger et al. Eq. (1): x_i.T @ Y @ Y.T @ x_i is the squared
            # Euclidean norm of x_i's covariances with all Y columns. The common
            # covariance divisor does not affect the selected argmax.
            covariance = X_residual.T @ Y_residual / (X.shape[0] - 1)
            criterion = np.sum(covariance**2, axis=1)
            criterion[selected] = -np.inf
            available = criterion[~selected]
            maximum = float(np.max(available)) if available.size else 0.0
            if step == 0:
                criterion_tolerance = np.finfo(float).eps * maximum
            if available.size == 0 or maximum <= criterion_tolerance:
                raise ValueError(
                    "CovSel exhausted predictive covariance before selecting the "
                    "requested number of variables. Reduce n_features_to_select."
                )

            # Step 2: select the unselected column with the largest criterion.
            index = int(np.argmax(criterion))
            history[step, ~selected] = criterion[~selected]
            order[step] = index
            winning_scores[step] = criterion[index]
            selected[index] = True

            # Step 3, Roger et al. Eqs. (2)-(3): apply the same orthogonal projector to
            # both residual blocks. Later variables must therefore explain Y
            # information not already carried by a selected X direction.
            score = X_residual[:, index].copy()
            score_sum_squares = float(score @ score)
            if score_sum_squares <= np.finfo(float).eps:
                raise ValueError(
                    "CovSel selected a zero-variance residual variable. Reduce "
                    "n_features_to_select or remove duplicate variables."
                )
            x_projection = np.outer(score, score @ X_residual / score_sum_squares)
            y_projection = np.outer(score, score @ Y_residual / score_sum_squares)
            X_residual -= x_projection
            Y_residual -= y_projection
            # Step 4: record explained variance; repeat until n_steps are chosen.
            explained_x[step] = 1.0 - float(np.sum(X_residual**2)) / total_x
            explained_y[step] = 1.0 - float(np.sum(Y_residual**2)) / total_y

        return order, winning_scores, history, explained_x, explained_y
