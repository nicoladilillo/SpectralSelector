"""Modelling utilities for supervised wrapper and embedded selectors.

This private module centralizes estimator cloning, task inference, scoring,
cross-validation, and coefficient extraction. Keeping these operations in one
place gives CARS, BOSS, GA, VIP, and SR consistent behavior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import cpu_count
from numpy.typing import ArrayLike, NDArray
from sklearn import get_config
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target

VALID_TASKS = {"auto", "regression", "classification"}


def resolve_n_jobs(n_jobs: int) -> int:
    """Validate and resolve the number of search worker processes.

    Parameters
    ----------
    n_jobs : int
        ``-1`` requests every CPU available to the current process. Otherwise,
        the value must be between one and the available CPU count, inclusive.

    Returns
    -------
    n_jobs_resolved : int
        Positive number of workers passed to Joblib.

    Raises
    ------
    TypeError
        If ``n_jobs`` is not an integer or is a Boolean value.
    ValueError
        If ``n_jobs`` is zero, below ``-1``, or greater than the available CPU
        count.
    """
    if not isinstance(n_jobs, (int, np.integer)) or isinstance(n_jobs, bool):
        raise TypeError("n_jobs must be an integer: -1 or a positive CPU count.")

    available = int(cpu_count())
    if n_jobs == -1:
        return available
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1 or an integer greater than or equal to 1.")
    if n_jobs > available:
        raise ValueError(
            f"n_jobs={n_jobs} exceeds the {available} CPU cores available to this "
            "process. Use -1 to select all available cores."
        )
    return int(n_jobs)


def resolve_task(estimator: Any, y: ArrayLike, task: str) -> str:
    """Resolve and validate regression versus classification behavior.

    Parameters
    ----------
    estimator : estimator object
        Unfitted scikit-learn-compatible predictive estimator.
    y : array-like of shape (n_samples,)
        Target used to validate the inferred task.
    task : {"auto", "regression", "classification"}
        Requested task. ``"auto"`` first inspects estimator tags and falls back
        to the target type for estimators without classifier/regressor tags.

    Returns
    -------
    task : str
        Resolved task name.
    """
    if task not in VALID_TASKS:
        raise ValueError("task must be 'auto', 'regression', or 'classification'.")

    estimator_is_classifier = is_classifier(estimator)
    estimator_is_regressor = is_regressor(estimator)
    if task == "auto":
        if estimator_is_classifier:
            resolved = "classification"
        elif estimator_is_regressor:
            resolved = "regression"
        else:
            target_kind = type_of_target(y, raise_unknown=True)
            resolved = (
                "classification"
                if target_kind in {"binary", "multiclass"}
                else "regression"
            )
    else:
        resolved = task

    if resolved == "classification" and estimator_is_regressor:
        raise ValueError(
            "task='classification' requires a classifier such as PLSDA, logistic "
            "regression, SVC, or RandomForestClassifier; a regressor was supplied."
        )
    if resolved == "regression" and estimator_is_classifier:
        raise ValueError("task='regression' cannot be used with a classifier.")

    target_kind = type_of_target(y, raise_unknown=True)
    if resolved == "classification" and target_kind not in {"binary", "multiclass"}:
        raise ValueError("Classification requires binary or multiclass labels.")
    if resolved == "regression" and target_kind not in {
        "continuous",
        "continuous-multioutput",
        "binary",
        "multiclass",
    }:
        raise ValueError("Regression requires one or more numeric responses.")
    if resolved == "regression" and np.asarray(y).dtype.kind not in "biufc":
        raise ValueError(
            "A regression estimator requires numeric y values. For class labels, "
            "pass a classifier and task='classification'."
        )
    return resolved


def resolve_cv(cv: Any, task: str, random_state: int | None) -> Any:
    """Create a deterministic cross-validation splitter.

    ``cv=-1`` creates :class:`~sklearn.model_selection.LeaveOneOut`; positive
    integers create stratified folds for classification and ordinary shuffled
    folds for regression. Existing splitter objects are returned unchanged.
    """
    if cv == -1:
        return LeaveOneOut()
    if isinstance(cv, (int, np.integer)):
        if isinstance(cv, bool) or cv < 2:
            raise ValueError("cv must be -1, an integer >= 2, or a CV splitter.")
        if task == "classification":
            return StratifiedKFold(
                n_splits=int(cv),
                shuffle=True,
                random_state=random_state,
            )
        return KFold(
            n_splits=int(cv),
            shuffle=True,
            random_state=random_state,
        )
    if not hasattr(cv, "split"):
        raise TypeError("cv must be -1, an integer >= 2, or an object with split().")
    return cv


def default_scoring(task: str) -> str:
    """Return accuracy for classification or negative RMSE for regression."""
    return "accuracy" if task == "classification" else "neg_root_mean_squared_error"


def score_subset(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[Any],
    indices: NDArray[np.int64],
    *,
    cv: Any,
    scoring: str | Any | None,
    task: str,
    n_jobs: int | None,
    groups: ArrayLike | None = None,
) -> float:
    """Evaluate one feature subset and return a higher-is-better mean CV score.

    The estimator is cloned before evaluation. Negative loss scorers follow
    scikit-learn's convention, so a value closer to zero is better.
    """
    candidate = clone(estimator)
    scorer = check_scoring(candidate, scoring=scoring or default_scoring(task))
    cv_arguments = {}
    if groups is not None:
        if get_config()["enable_metadata_routing"]:
            cv_arguments["params"] = {"groups": groups}
        else:
            cv_arguments["groups"] = groups
    scores = cross_val_score(
        candidate,
        X[:, indices],
        y,
        cv=cv,
        scoring=scorer,
        n_jobs=n_jobs,
        error_score="raise",
        **cv_arguments,
    )
    return float(np.mean(scores))


def unwrap_fitted_estimator(estimator: Any) -> Any:
    """Find the predictive model inside common fitted meta-estimators.

    Supported containers include ``GridSearchCV``-like objects, pipelines, and
    the local PLS-DA implementation, which exposes its fitted PLS model as
    ``pls_``.
    """
    current = estimator
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, "best_estimator_"):
            current = current.best_estimator_
        elif isinstance(current, Pipeline):
            current = current.steps[-1][1]
        elif hasattr(current, "pls_"):
            current = current.pls_
        else:
            break
    return current


def feature_importances(estimator: Any, n_features: int) -> NDArray[np.float64]:
    """Extract a one-dimensional absolute importance vector.

    Multi-target coefficient matrices are averaged across all non-feature axes.
    Tree-style ``feature_importances_`` vectors are accepted directly.
    """
    model = unwrap_fitted_estimator(estimator)
    if hasattr(model, "coef_"):
        values = np.asarray(model.coef_, dtype=np.float64)
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    else:
        raise TypeError(
            "The estimator must expose coef_ or feature_importances_ after fit."
        )

    values = np.abs(np.squeeze(values))
    if values.ndim == 1:
        if values.size != n_features:
            raise ValueError("The estimator importance vector has the wrong length.")
        return values
    feature_axes = [
        axis for axis, size in enumerate(values.shape) if size == n_features
    ]
    if not feature_axes:
        raise ValueError("Cannot identify the feature axis in estimator importances.")
    feature_axis = feature_axes[-1]
    values = np.moveaxis(values, feature_axis, -1)
    return np.mean(values.reshape(-1, n_features), axis=0)


def fit_feature_importances(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[Any],
) -> tuple[Any, NDArray[np.float64]]:
    """Clone and fit an estimator, then return it with feature importances."""
    fitted = clone(estimator).fit(X, y)
    return fitted, feature_importances(fitted, X.shape[1])
