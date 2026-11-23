"""Model-validated length for selectors that produce a model-free variable order.

CovSel, fCovSel and MI-VIF rank wavelengths without fitting a predictive model,
but *how many* of the ranked wavelengths to keep depends on the model that will
use them. :class:`PrefixLengthSearchMixin` chooses that number with the user's
estimator while keeping the ranking itself model-free:

1. Split the data with the configured cross-validation scheme.
2. In every fold, refit the optional ``preprocessor`` and recompute the
   selector's own order on the *training rows only* (``_fold_order``), so no
   validation row influences either preprocessing or ranking.
3. Fit the estimator on every prefix ``order[:k]`` of the fold order (and every
   point of ``param_grid``) and score it on the validation rows.
4. Keep the length (and hyper-parameters) with the best mean validation score;
   ties prefer fewer variables, then the first grid point.
5. The final support is the first ``k`` variables of the order fitted on all
   rows, and ``estimator_`` is refitted on them.

Subclasses implement ``_fold_order(X, y, n_steps)`` and must define the
constructor parameters ``estimator``, ``task``, ``scoring``, ``cv``,
``random_state``, ``param_grid``, ``preprocessor`` and ``n_jobs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import ParameterGrid
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_consistent_length

from ._modeling import default_scoring, resolve_cv, resolve_task
from .plsda import PLSDA

LENGTH_SEARCH_PARAMETERS_DOC = """
    estimator : estimator object, optional
        Predictive model used *only* to choose how many ranked wavelengths to
        keep (the ranking itself never uses it). Pass the model you will fit on
        the selected wavelengths, for example ``PLSRegression``, a PLS-DA, or a
        ``Pipeline``. ``None`` creates one-component PLSR for regression or the
        package PLSDA for classification.
    scoring : str or callable, optional
        Any scikit-learn scorer. Defaults to negative RMSE for regression and
        accuracy for classification.
    cv : int, CV splitter, or -1, default=5
        Inner validation folds; ``-1`` selects leave-one-out.
    random_state : int, optional
        Seed for shuffled K-fold splits created from an integer ``cv``.
    param_grid : dict or list of dict, optional
        Hyper-parameter grid of ``estimator`` searched jointly with the length
        (``ParameterGrid`` syntax). Parameters ending in ``n_components`` are
        capped at the number of selected variables.
    preprocessor : transformer, optional
        Column-preserving transformer (for example SG + MSC) refitted on the
        training rows of every validation fold and on all rows for the final
        order. ``transform`` applies the fitted preprocessor before selecting.
    n_jobs : int, optional
        Parallel jobs over validation folds and grid points.
"""

LENGTH_SEARCH_ATTRIBUTES_DOC = """
    validation_scores_ : ndarray of shape (n_evaluated,)
        Mean higher-is-better validation score for every prefix length.
    validation_score_std_ : ndarray of shape (n_evaluated,)
        Standard deviation of fold scores for every prefix length.
    rmsecv_ : ndarray of shape (n_evaluated,) or None
        Mean fold RMSE when ``scoring="neg_root_mean_squared_error"``.
    fold_selection_orders_ : ndarray of shape (n_splits, n_evaluated)
        Order recomputed on the training rows of every validation fold.
    consensus_order_ : ndarray of shape (n_evaluated,)
        Deterministic aggregation of fold orders (a stability diagnostic).
    best_n_features_ : int
        Selected number of wavelengths.
    best_params_ : dict
        Hyper-parameters chosen together with the length.
    best_score_ : float
        Mean validation score at ``best_n_features_``.
    cv_results_ : dict
        ``n_features``, ``params``, ``mean_test_score``, ``std_test_score``
        for every evaluated (length, grid point) pair.
    estimator_ : estimator object
        ``estimator`` refitted on the selected wavelengths.
"""


class PrefixLengthSearchMixin:
    """Choose how many variables of a model-free order a model should use."""

    # ------------------------------------------------------------------
    # Hook implemented by each selector
    # ------------------------------------------------------------------
    def _fold_order(
        self, X: NDArray[np.float64], y: NDArray[Any], n_steps: int
    ) -> NDArray[np.int64]:
        """Return at most ``n_steps`` ranked indices fitted on these rows."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Set-up
    # ------------------------------------------------------------------
    def _default_length_estimator(self, y: NDArray[Any]):
        """One-component PLSR or PLSDA, matching the requested/inferred task."""
        requested = self.task
        if requested == "auto":
            target_kind = type_of_target(y, raise_unknown=True)
            requested = (
                "classification"
                if target_kind in {"binary", "multiclass"}
                else "regression"
            )
        if requested == "classification":
            return PLSDA(n_components=1, scale_pls=False)
        return PLSRegression(n_components=1, scale=False)

    def _setup_length_search(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        groups: ArrayLike | None,
    ) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Resolve task, model, scorer, folds and grid; return the splits."""
        estimator = self.estimator
        if estimator is None:
            estimator = self._default_length_estimator(y)
        self.task_ = resolve_task(estimator, y, self.task)
        self.estimator_template_ = estimator
        self.scoring_ = self.scoring or default_scoring(self.task_)
        check_scoring(estimator, scoring=self.scoring_)
        self.cv_ = resolve_cv(self.cv, self.task_, self.random_state)
        group_array = None if groups is None else np.asarray(groups)
        if group_array is not None:
            if group_array.ndim != 1:
                raise ValueError("groups must be a one-dimensional array-like.")
            check_consistent_length(y, group_array)
        self.param_list_ = (
            list(ParameterGrid(self.param_grid))
            if self.param_grid is not None
            else [{}]
        )
        return self._split_indices(X, y, group_array)

    def _max_prefix_length(
        self,
        n_samples: int,
        n_features: int,
        splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
    ) -> int:
        """Largest prefix every training fold can support."""
        smallest_training = min(len(train) for train, _ in splits)
        max_allowed = min(n_samples - 1, smallest_training - 1, n_features)
        if max_allowed < 1:
            raise ValueError(
                "Cross-validation training folds are too small for length validation."
            )
        return max_allowed

    def _split_indices(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Materialize and validate the configured inner-validation splits."""
        if groups is None:
            splits = list(self.cv_.split(X, y))
        else:
            splits = list(self.cv_.split(X, y, groups))
        if not splits:
            raise ValueError("cv produced no validation splits.")
        for train, test in splits:
            if len(train) < 2 or len(test) < 1:
                raise ValueError(
                    "Every cv split must contain at least two training samples "
                    "and one validation sample."
                )
        return splits

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _fit_full_preprocessor(
        self, X: NDArray[np.float64], y: NDArray[Any]
    ) -> NDArray[np.float64]:
        """Fit ``preprocessor`` on all rows; return the matrix used for ranking."""
        if self.preprocessor is None:
            self.preprocessor_ = None
            return X
        self.preprocessor_ = clone(self.preprocessor).fit(X, y)
        return self._apply_preprocessor(self.preprocessor_, X)

    @staticmethod
    def _apply_preprocessor(preprocessor, X: np.ndarray) -> np.ndarray:
        """Transform with a fitted preprocessor and check the column layout."""
        X_out = np.asarray(preprocessor.transform(X), dtype=np.float64)
        if X_out.shape != X.shape:
            raise ValueError(
                "preprocessor must keep the number of rows and columns "
                f"(got {X_out.shape} from {X.shape})."
            )
        return X_out

    def transform(self, X: ArrayLike):
        """Select the fitted columns (after the fitted preprocessor, if any)."""
        if getattr(self, "preprocessor_", None) is None:
            return super().transform(X)
        X_array = self._validate_X(X, reset=False)
        return self._apply_preprocessor(self.preprocessor_, X_array)[:, self.support_]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _candidate_estimator(
        self,
        n_features: int,
        n_samples: int,
        params: dict | None = None,
    ):
        """Clone the validation model, apply grid parameters, cap components."""
        candidate = clone(self.estimator_template_)
        if params:
            candidate.set_params(**params)
        parameters = candidate.get_params(deep=True)
        caps = {
            name: min(int(value), n_features, n_samples - 1)
            for name, value in parameters.items()
            if (name == "n_components" or name.endswith("__n_components"))
            and isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
        }
        if caps:
            candidate.set_params(**caps)
        return candidate

    def _fold_scores(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        order: np.ndarray,
        params: dict,
        n_steps: int,
        scorer,
    ) -> np.ndarray:
        """Score every prefix length of one fold order for one grid point."""
        scores = np.empty(n_steps, dtype=np.float64)
        for length in range(1, n_steps + 1):
            indices = order[:length]
            candidate = self._candidate_estimator(length, len(y_train), params)
            candidate.fit(X_train[:, indices], y_train)
            scores[length - 1] = scorer(candidate, X_test[:, indices], y_test)
        return scores

    @staticmethod
    def _unique_fold_consensus(
        fold_orders: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """Aggregate fold positions without repeating a variable index."""
        n_steps = fold_orders.shape[1]
        consensus = np.empty(n_steps, dtype=np.int64)
        selected = np.zeros(n_features, dtype=bool)
        # Mean fold rank is a deterministic fallback if all variables observed
        # at one position have already won an earlier consensus position.
        aggregate_rank = np.full(n_features, n_steps + 1.0)
        for feature in range(n_features):
            locations = np.argwhere(fold_orders == feature)
            if locations.size:
                aggregate_rank[feature] = float(np.mean(locations[:, 1]))

        for step in range(n_steps):
            counts = np.bincount(fold_orders[:, step], minlength=n_features).astype(
                float
            )
            counts[selected] = -1.0
            if np.max(counts) <= 0.0:
                candidates = np.flatnonzero(~selected)
                index = int(candidates[np.argmin(aggregate_rank[candidates])])
            else:
                # np.argmax provides a stable lowest-source-index tie break.
                index = int(np.argmax(counts))
            consensus[step] = index
            selected[index] = True
        return consensus

    def _validate_sequence(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_steps: int,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Score fold-local prefixes without leaking validation rows.

        In every fold the optional preprocessor and the selector order are
        fitted on the training rows only; every prefix length is then scored
        with every grid point. Orders of different lengths (MI-VIF) are
        truncated to the shortest one. Returns the per-length best mean score
        and its standard deviation, the fold-order consensus and the fold
        orders; the full (length x grid point) tables are stored for ``fit``.
        """
        n_features = X.shape[1]
        scorer = check_scoring(self.estimator_template_, scoring=self.scoring_)
        expected_classes = np.unique(y) if self.task_ == "classification" else None

        # Step 1: one entry per validation split.
        fold_data = []
        for train, test in splits:
            if expected_classes is not None and not np.array_equal(
                np.unique(y[train]), expected_classes
            ):
                raise ValueError(
                    "Every classification training fold must contain every class. "
                    "Use stratified folds or reduce the number of folds."
                )
            # Step 2: preprocessing and the selector order use training rows only.
            X_train, X_test = X[train], X[test]
            if self.preprocessor is not None:
                fold_pre = clone(self.preprocessor).fit(X_train, y[train])
                X_train = self._apply_preprocessor(fold_pre, X_train)
                X_test = self._apply_preprocessor(fold_pre, X_test)
            order = np.asarray(
                self._fold_order(X_train, y[train], n_steps), dtype=np.int64
            )
            fold_data.append([X_train, y[train], X_test, y[test], order])

        n_steps = min(n_steps, *(len(data[4]) for data in fold_data))
        if n_steps < 1:
            raise ValueError("A validation fold produced an empty variable order.")
        for data in fold_data:
            data[4] = data[4][:n_steps]
        fold_orders = np.vstack([data[4] for data in fold_data])

        # Step 3: score every prefix length for every (fold, grid point).
        tasks = [
            (fold, k)
            for fold in range(len(splits))
            for k in range(len(self.param_list_))
        ]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fold_scores)(
                *fold_data[fold], self.param_list_[k], n_steps, scorer
            )
            for fold, k in tasks
        )
        fold_scores = np.empty(
            (len(splits), n_steps, len(self.param_list_)), dtype=np.float64
        )
        for (fold, k), scores in zip(tasks, results, strict=True):
            fold_scores[fold, :, k] = scores

        # Step 4 (input): mean and spread of fold scores per (length, grid point).
        grid_mean = np.mean(fold_scores, axis=0)
        grid_std = (
            np.std(fold_scores, axis=0, ddof=1)
            if len(splits) > 1
            else np.zeros_like(grid_mean)
        )
        best = np.argmax(grid_mean, axis=1)
        rows = np.arange(n_steps)
        validation_scores = grid_mean[rows, best]
        score_std = grid_std[rows, best]
        consensus = self._unique_fold_consensus(fold_orders, n_features)
        self._grid_tables = (grid_mean, grid_std)
        return validation_scores, score_std, consensus, fold_orders

    def _run_length_search(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        n_steps: int,
        splits: list[tuple[NDArray[np.int64], NDArray[np.int64]]],
        *,
        full_order_length: int | None = None,
    ) -> NDArray[np.float64]:
        """Validate prefix lengths and store the shared diagnostics.

        ``full_order_length`` truncates the evaluated lengths when the order
        fitted on all rows is shorter than the fold orders.
        """
        self._grid_tables = None
        validation_scores, score_std, consensus, fold_orders = self._validate_sequence(
            X, y, n_steps, splits
        )
        if self._grid_tables is not None:
            grid_mean, grid_std = self._grid_tables
        else:  # single-grid-point fallback (e.g. a custom _validate_sequence)
            grid_mean = np.asarray(validation_scores, dtype=np.float64)[:, None]
            grid_std = np.asarray(score_std, dtype=np.float64)[:, None]
        del self._grid_tables

        n_evaluated = len(validation_scores)
        if full_order_length is not None:
            n_evaluated = min(n_evaluated, full_order_length)
        validation_scores = np.asarray(validation_scores)[:n_evaluated]
        self.validation_scores_ = validation_scores
        self.validation_score_std_ = np.asarray(score_std)[:n_evaluated]
        grid_mean, grid_std = grid_mean[:n_evaluated], grid_std[:n_evaluated]
        self.rmsecv_ = (
            -validation_scores
            if self.scoring_ == "neg_root_mean_squared_error"
            else None
        )
        self.fold_selection_orders_ = np.asarray(fold_orders)[:, :n_evaluated]
        self.consensus_order_ = np.asarray(consensus)[:n_evaluated]
        best_param_index = np.argmax(grid_mean, axis=1)
        self.validation_params_ = [self.param_list_[int(k)] for k in best_param_index]
        self.cv_results_ = {
            "n_features": np.repeat(
                np.arange(1, n_evaluated + 1), len(self.param_list_)
            ),
            "params": [p for _ in range(n_evaluated) for p in self.param_list_],
            "mean_test_score": grid_mean.ravel(),
            "std_test_score": grid_std.ravel(),
        }
        return validation_scores

    def _choose_length(self, requested: int | str) -> int:
        """Resolve ``"auto"`` or a fixed count against the validation table."""
        # Step 4: best mean validation score sets the length (or use a fixed one).
        if requested == "auto":
            # First maximum: fewer variables win ties, then the first grid point.
            length = int(np.argmax(self.validation_scores_)) + 1
        else:
            length = int(requested)
        self.best_n_features_ = length
        self.best_params_ = dict(self.validation_params_[length - 1])
        self.best_score_ = float(self.validation_scores_[length - 1])
        return length

    def _fit_final_estimator(
        self,
        X_model: NDArray[np.float64],
        y: NDArray[Any],
        selected_order: NDArray[np.int64],
    ) -> None:
        """Refit the validation model on the selected wavelengths."""
        # Step 5: refit on the selected prefix of the full-data order.
        self.estimator_ = self._candidate_estimator(
            len(selected_order), len(y), self.best_params_
        ).fit(X_model[:, selected_order], y)
