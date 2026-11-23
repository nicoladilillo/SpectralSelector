"""Reusable contract for model-evaluated feature selectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_consistent_length

from ._modeling import (
    resolve_cv,
    resolve_n_jobs,
    resolve_task,
    score_subset,
)


class WrapperSelectorMixin:
    """Internal mixin shared by CARS, BOSS, and genetic selectors.

    Concrete selectors expose the annotated constructor parameters. The mixin
    resolves them once during fitting and provides a common subset-scoring
    method.
    """

    estimator: Any
    task: str
    scoring: str | Any | None
    cv: Any
    random_state: int | None
    n_jobs: int
    max_components: int
    verbose: int

    def _prepare_wrapper(
        self,
        y: ArrayLike,
        groups: ArrayLike | None = None,
        *,
        estimator_override: Any | None = None,
    ) -> tuple[Any, str, Any, NDArray[Any] | None]:
        """Resolve the estimator, task, CV splitter, and scoring configuration."""
        if estimator_override is not None:
            estimator = estimator_override
        elif self.estimator is None:
            estimator = PLSRegression(n_components=1, scale=True)
        else:
            estimator = self.estimator
        task = resolve_task(estimator, y, self.task)
        cv = resolve_cv(self.cv, task, self.random_state)
        n_jobs = resolve_n_jobs(self.n_jobs)
        self.estimator_template_ = estimator
        self.task_ = task
        self.cv_ = cv
        self.n_jobs_ = n_jobs
        self.scoring_ = self.scoring or (
            "accuracy" if task == "classification" else "neg_root_mean_squared_error"
        )
        if groups is None:
            group_array = None
        else:
            group_array = np.asarray(groups)
            if group_array.ndim != 1:
                raise ValueError("groups must be a one-dimensional array-like.")
            check_consistent_length(y, group_array)
        return estimator, task, cv, group_array

    def _score_indices(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        indices: NDArray[np.int64],
        groups: ArrayLike | None = None,
    ) -> float:
        """Evaluate selected indices with the resolved wrapper configuration."""
        return score_subset(
            self.estimator_template_,
            X,
            y,
            indices,
            cv=self.cv_,
            scoring=self.scoring_,
            task=self.task_,
            # Candidate searches own the process pool. Keeping fold evaluation
            # serial prevents nested parallelism and CPU oversubscription.
            n_jobs=1,
            groups=groups,
        )

    def _resolve_scoring(self) -> str | Any:
        """Return the configured scorer or the task-specific default."""
        if self.scoring is not None:
            return self.scoring
        if self.task_ == "classification":
            return "accuracy"
        return "neg_root_mean_squared_error"

    @staticmethod
    def _coefficient_importance(
        coefficients: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reduce PLS coefficients to one non-negative value per feature."""
        coefficients = np.asarray(coefficients, dtype=np.float64)
        if coefficients.ndim != 2:
            raise ValueError("coefficients must have shape (n_features, n_responses).")
        if coefficients.shape[1] == 1:
            return np.abs(coefficients[:, 0])
        return np.linalg.norm(coefficients, ord=2, axis=1)

    _importance_from_coefficients = _coefficient_importance

    @staticmethod
    def _coefficient_matrix(
        estimator: Any,
        *,
        n_features: int,
    ) -> NDArray[np.float64]:
        """Normalize ``estimator.coef_`` to features by responses."""
        if not hasattr(estimator, "coef_"):
            raise AttributeError(f"{estimator.__class__.__name__} must expose coef_.")

        coefficients = np.asarray(estimator.coef_, dtype=np.float64)
        if coefficients.ndim == 1:
            if coefficients.size != n_features:
                raise ValueError("coef_ length does not match active feature count.")
            return coefficients.reshape(n_features, 1)
        if coefficients.ndim != 2:
            raise ValueError("coef_ must be one- or two-dimensional.")
        if coefficients.shape[0] == n_features:
            return coefficients.copy()
        if coefficients.shape[1] == n_features:
            return coefficients.T.copy()
        raise ValueError("Neither coef_ axis matches active feature count.")

    def _log(self, level: int, message: str) -> None:
        """Emit an algorithm progress message at the requested level."""
        if self.verbose >= level:
            print(message, flush=True)
