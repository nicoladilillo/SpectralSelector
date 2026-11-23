from types import SimpleNamespace

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline

from spectralselector import PLSDA
from spectralselector._modeling import (
    default_scoring,
    feature_importances,
    fit_feature_importances,
    resolve_cv,
    resolve_task,
    score_subset,
    unwrap_fitted_estimator,
)
from spectralselector._wrapper import WrapperSelectorMixin


class _UntaggedEstimator(BaseEstimator):
    pass


class _CustomSplitter:
    def split(self, X, y=None, groups=None):
        yield np.arange(len(X) - 1), np.asarray([len(X) - 1])


class _WrapperFixture(WrapperSelectorMixin):
    def __init__(self):
        self.estimator = None
        self.task = "auto"
        self.scoring = None
        self.cv = 2
        self.random_state = 7
        self.n_jobs = 1
        self.verbose = 0


def test_task_resolution_covers_tags_fallbacks_and_incompatible_targets():
    binary = np.asarray([0, 1, 0, 1])
    continuous = np.asarray([0.1, 0.4, 0.2, 0.8])

    assert resolve_task(DummyClassifier(), binary, "auto") == "classification"
    assert resolve_task(DummyRegressor(), continuous, "auto") == "regression"
    assert resolve_task(_UntaggedEstimator(), binary, "auto") == "classification"
    assert resolve_task(_UntaggedEstimator(), continuous, "auto") == "regression"
    assert resolve_task(DummyRegressor(), binary, "regression") == "regression"

    with pytest.raises(ValueError, match="task must be"):
        resolve_task(DummyRegressor(), continuous, "forecasting")
    with pytest.raises(ValueError, match="cannot be used with a classifier"):
        resolve_task(DummyClassifier(), binary, "regression")
    with pytest.raises(ValueError, match="requires a classifier"):
        resolve_task(DummyRegressor(), binary, "classification")
    with pytest.raises(ValueError, match="binary or multiclass"):
        resolve_task(DummyClassifier(), continuous, "classification")
    with pytest.raises(ValueError, match="one or more numeric responses"):
        resolve_task(
            DummyRegressor(),
            np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]]),
            "regression",
        )
    with pytest.raises(ValueError, match="requires numeric y"):
        resolve_task(DummyRegressor(), np.asarray(["a", "b", "a", "b"]), "regression")


def test_cv_resolution_and_default_scoring_are_task_aware():
    assert isinstance(resolve_cv(-1, "regression", 3), LeaveOneOut)
    assert isinstance(resolve_cv(3, "regression", 3), KFold)
    assert isinstance(resolve_cv(np.int64(3), "classification", 3), StratifiedKFold)
    custom = _CustomSplitter()
    assert resolve_cv(custom, "regression", 3) is custom
    assert default_scoring("classification") == "accuracy"
    assert default_scoring("regression") == "neg_root_mean_squared_error"

    for invalid in (True, 0, 1):
        with pytest.raises(ValueError, match="cv must be"):
            resolve_cv(invalid, "regression", 3)
    with pytest.raises(TypeError, match="object with split"):
        resolve_cv(2.5, "regression", 3)


@pytest.mark.parametrize("metadata_routing", [False, True])
def test_subset_scoring_routes_groups_in_both_sklearn_modes(metadata_routing):
    rng = np.random.default_rng(91)
    X = rng.normal(size=(18, 4))
    y = 2 * X[:, 0] - X[:, 2]
    groups = np.repeat(np.arange(6), 3)

    with config_context(enable_metadata_routing=metadata_routing):
        score = score_subset(
            Ridge(),
            X,
            y,
            np.asarray([0, 2]),
            cv=GroupKFold(3),
            scoring="neg_mean_absolute_error",
            task="regression",
            n_jobs=1,
            groups=groups,
        )

    assert np.isfinite(score)


def test_unwrap_and_importance_extraction_cover_supported_model_shapes():
    rng = np.random.default_rng(92)
    X = rng.normal(size=(20, 3))
    y = (X[:, 0] > 0).astype(int)
    plsda = PLSDA(n_components=1).fit(X, y)
    assert unwrap_fitted_estimator(plsda) is plsda.pls_

    final = Ridge()
    wrapped = SimpleNamespace(
        best_estimator_=Pipeline([("identity", "passthrough"), ("model", final)])
    )
    assert unwrap_fitted_estimator(wrapped) is final

    one_dimensional = SimpleNamespace(coef_=np.asarray([-1.0, 2.0, -3.0]))
    np.testing.assert_array_equal(
        feature_importances(one_dimensional, 3), np.asarray([1.0, 2.0, 3.0])
    )
    targets_by_features = SimpleNamespace(
        coef_=np.asarray([[1.0, -2.0, 3.0], [-3.0, 4.0, -5.0]])
    )
    np.testing.assert_array_equal(
        feature_importances(targets_by_features, 3), np.asarray([2.0, 3.0, 4.0])
    )
    tree_style = SimpleNamespace(feature_importances_=np.asarray([0.1, 0.7, 0.2]))
    np.testing.assert_array_equal(
        feature_importances(tree_style, 3), tree_style.feature_importances_
    )

    fitted, importances = fit_feature_importances(LinearRegression(), X, X[:, 0])
    assert isinstance(fitted, LinearRegression)
    assert importances.shape == (3,)


def test_importance_extraction_rejects_missing_or_ambiguous_feature_axes():
    with pytest.raises(TypeError, match="coef_ or feature_importances_"):
        feature_importances(SimpleNamespace(), 3)
    with pytest.raises(ValueError, match="wrong length"):
        feature_importances(SimpleNamespace(coef_=np.ones(2)), 3)
    with pytest.raises(ValueError, match="identify the feature axis"):
        feature_importances(SimpleNamespace(coef_=np.ones((2, 2))), 3)


def test_wrapper_shared_coefficient_guards_and_group_validation(capsys):
    wrapper = _WrapperFixture()
    y = np.arange(6, dtype=float)
    estimator, task, cv, groups = wrapper._prepare_wrapper(y)
    assert estimator.n_components == 1
    assert task == "regression"
    assert isinstance(cv, KFold)
    assert groups is None

    override = Ridge().fit(np.ones((6, 2)), y)
    estimator, _, _, groups = wrapper._prepare_wrapper(
        y, np.asarray([0, 0, 1, 1, 2, 2]), estimator_override=override
    )
    assert estimator is override
    np.testing.assert_array_equal(groups, [0, 0, 1, 1, 2, 2])

    with pytest.raises(ValueError, match="one-dimensional"):
        wrapper._prepare_wrapper(y, np.ones((6, 1)), estimator_override=override)

    np.testing.assert_array_equal(
        wrapper._coefficient_matrix(
            SimpleNamespace(coef_=np.asarray([1.0, 2.0])), n_features=2
        ),
        np.asarray([[1.0], [2.0]]),
    )
    transposed = wrapper._coefficient_matrix(
        SimpleNamespace(coef_=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
        n_features=2,
    )
    assert transposed.shape == (2, 3)
    np.testing.assert_array_equal(
        wrapper._coefficient_importance(np.asarray([[3.0, 4.0], [5.0, 12.0]])),
        [5.0, 13.0],
    )

    with pytest.raises(AttributeError, match="must expose coef_"):
        wrapper._coefficient_matrix(SimpleNamespace(), n_features=2)
    with pytest.raises(ValueError, match="length"):
        wrapper._coefficient_matrix(SimpleNamespace(coef_=np.ones(3)), n_features=2)
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        wrapper._coefficient_matrix(
            SimpleNamespace(coef_=np.ones((2, 2, 2))), n_features=2
        )
    with pytest.raises(ValueError, match="Neither coef_ axis"):
        wrapper._coefficient_matrix(
            SimpleNamespace(coef_=np.ones((3, 4))), n_features=2
        )
    with pytest.raises(ValueError, match="n_features, n_responses"):
        wrapper._coefficient_importance(np.ones(2))

    wrapper.verbose = 1
    wrapper._log(1, "visible")
    assert capsys.readouterr().out.strip() == "visible"
