from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from spectralselector import SPASelector


def _reference_order(X, initial, n_steps):
    """SPA as residual norms after least-squares projection on the selected span."""
    order = [initial]
    while len(order) < n_steps:
        basis = X[:, order]
        residual = X - basis @ np.linalg.lstsq(basis, X, rcond=None)[0]
        norms = np.sum(residual**2, axis=0)
        norms[order] = -np.inf
        order.append(int(np.argmax(norms)))
    return order


def test_projection_order_matches_least_squares_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 40)) @ rng.normal(size=(40, 40))
    for initial in (0, 7, 39):
        order = SPASelector._projection_order(X, initial, 12)
        assert order.tolist() == _reference_order(X, initial, 12)


def test_hand_calculated_projection_step():
    # After projecting out x0=(1,0,0), x1=(1,1,0) keeps norm 1 and x2=(1,0,2) norm 2.
    X = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    assert SPASelector._projection_order(X, 0, 3).tolist() == [0, 2, 1]


def test_auto_choice_is_best_start_and_length_of_the_score_table():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 12))
    y = X[:, 3] - 2 * X[:, 8] + rng.normal(scale=0.1, size=40)

    selector = SPASelector(max_features=5, cv=KFold(4)).fit(X, y)

    table = selector.initial_scores_
    assert table.shape == (12, 5)
    length = int(np.argmax(table.max(axis=0))) + 1
    assert selector.n_features_selected_ == length
    assert selector.initial_ == int(np.argmax(table[:, length - 1]))
    np.testing.assert_allclose(selector.validation_scores_, table[selector.initial_])
    assert selector.selection_order_[0] == selector.initial_
    assert isinstance(selector.estimator_, LinearRegression)


def test_fixed_initial_and_length_use_the_full_data_order():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 8))
    y = X[:, 1] + rng.normal(scale=0.1, size=25)

    selector = SPASelector(3, initial=5, autoscale=False, cv=3).fit(X, y)

    assert selector.initial_scores_.shape[0] == 1
    assert selector.selection_order_.tolist() == _reference_order(X, 5, 3)
    assert selector.transform(X).shape == (25, 3)


def test_classification_defaults_to_lda():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 6))
    y = (X[:, 2] > 0).astype(int)

    selector = SPASelector(initial=2, max_features=3, cv=3).fit(X, y)

    assert selector.task_ == "classification"
    assert isinstance(selector.estimator_, LinearDiscriminantAnalysis)


@pytest.mark.parametrize(
    "params",
    [
        {"initial": 8},
        {"initial": True},
        {"max_features": 0},
        {"n_features_to_select": 0},
    ],
)
def test_invalid_parameters_are_rejected(params):
    X = np.random.default_rng(4).normal(size=(10, 8))
    with pytest.raises(ValueError):
        SPASelector(**params).fit(X, X[:, 0])


def _regression_problem(seed=5, n=40, p=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2 * X[:, 1] - X[:, -1] + rng.normal(scale=0.1, size=n)
    return X, y


def test_spa_selects_indices_and_records_diagnostics():
    X, y = _regression_problem()
    wavelengths = 400.0 + 2.0 * np.arange(X.shape[1])

    selector = SPASelector(max_features=4, cv=KFold(4), wavelengths=wavelengths).fit(
        X, y
    )

    k = selector.n_features_selected_
    assert selector.support_.sum() == k == selector.best_n_features_
    assert selector.selected_indices_.tolist() == sorted(selector.selection_order_)
    assert selector.full_selection_order_[:k].tolist() == (
        selector.selection_order_.tolist()
    )
    np.testing.assert_array_equal(
        selector.selected_wavelengths_, wavelengths[selector.selected_indices_]
    )
    assert selector.fold_selection_orders_.shape == (4, 4)
    assert (selector.fold_selection_orders_[:, 0] == selector.initial_).all()
    assert len(selector.cv_results_["mean_test_score"]) == 4
    np.testing.assert_allclose(selector.rmsecv_, -selector.validation_scores_)


def test_spa_order_uses_only_x():
    X, y = _regression_problem()
    rng = np.random.default_rng(6)

    first = SPASelector(4, initial=3, cv=3).fit(X, y)
    second = SPASelector(4, initial=3, cv=3).fit(X, rng.normal(size=len(y)))

    assert first.selection_order_.tolist() == second.selection_order_.tolist()


def test_spa_autoscaled_selection_is_invariant_to_column_rescaling():
    X, y = _regression_problem()
    scales = np.geomspace(0.01, 100.0, X.shape[1])

    original = SPASelector(max_features=4, cv=KFold(4)).fit(X, y)
    rescaled = SPASelector(max_features=4, cv=KFold(4)).fit(X * scales + 5.0, y)

    assert original.initial_ == rescaled.initial_
    assert original.selection_order_.tolist() == rescaled.selection_order_.tolist()


def test_spa_never_starts_from_a_constant_column():
    X, y = _regression_problem()
    X[:, 0] = 3.0

    selector = SPASelector(max_features=3, cv=KFold(4)).fit(X, y)

    assert np.isneginf(selector.initial_scores_[0]).all()
    assert 0 not in selector.selection_order_


def test_spa_supports_multiclass_string_labels():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 8))
    y = np.array(["low", "mid", "high"])[np.digitize(X[:, 2], [-0.5, 0.5])]

    selector = SPASelector(max_features=3, cv=StratifiedKFold(3)).fit(X, y)

    assert selector.task_ == "classification"
    assert selector.scoring_ == "accuracy"
    assert set(selector.estimator_.classes_) == {"low", "mid", "high"}


def test_spa_cv_minus_one_resolves_to_leave_one_out():
    X, y = _regression_problem(n=12, p=6)

    selector = SPASelector(2, initial=0, max_features=3, cv=-1).fit(X, y)

    assert isinstance(selector.cv_, LeaveOneOut)
    assert selector.fold_selection_orders_.shape == (12, 3)


def test_spa_accepts_group_aware_custom_splitter():
    X, y = _regression_problem(n=30, p=6)
    groups = np.repeat(np.arange(3), 10)

    selector = SPASelector(2, initial=1, max_features=3, cv=GroupKFold(3)).fit(
        X, y, groups=groups
    )

    assert selector.fold_selection_orders_.shape == (3, 3)


def test_spa_does_not_mutate_max_features_when_data_require_clipping():
    X, y = _regression_problem(n=8, p=5)

    selector = SPASelector(1, initial=0, max_features=20, cv=2).fit(X, y)

    assert selector.max_features == 20
    assert selector.max_features_ == 3  # smallest training fold (4) - 1


def test_spa_param_grid_selects_length_and_hyperparameters_jointly():
    X, y = _regression_problem(p=6)
    grid = {"alpha": [0.01, 100.0]}

    selector = SPASelector(
        initial=1, max_features=4, cv=KFold(4), estimator=Ridge(), param_grid=grid
    ).fit(X, y)

    assert len(selector.cv_results_["params"]) == 4 * 2
    assert selector.best_params_ in [{"alpha": 0.01}, {"alpha": 100.0}]
    assert selector.estimator_.alpha == selector.best_params_["alpha"]


def test_spa_runs_inside_cross_validated_pipeline():
    X, y = _regression_problem()
    pipeline = Pipeline(
        [
            ("select", SPASelector(max_features=4, cv=3)),
            ("model", LinearRegression()),
        ]
    )

    scores = cross_val_score(
        pipeline, X, y, cv=KFold(3, shuffle=True, random_state=0), scoring="r2"
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_spa_preprocessor_is_refitted_inside_every_fold():
    class RecordingCentering(TransformerMixin, BaseEstimator):
        fitted_sizes: ClassVar[list[int]] = []

        def fit(self, X, y=None):
            RecordingCentering.fitted_sizes.append(len(X))
            self.mean_ = np.asarray(X).mean(axis=0)
            return self

        def transform(self, X):
            return np.asarray(X) - self.mean_

    X, y = _regression_problem()
    X = X + 5.0
    RecordingCentering.fitted_sizes = []
    selector = SPASelector(
        initial=0, max_features=3, cv=KFold(4), preprocessor=RecordingCentering()
    ).fit(X, y)

    assert sorted(RecordingCentering.fitted_sizes) == [30, 30, 30, 30, 40]
    expected = X - X.mean(axis=0)
    np.testing.assert_allclose(selector.transform(X), expected[:, selector.support_])


def test_spa_preprocessor_must_keep_columns():
    X, y = _regression_problem()

    with pytest.raises(ValueError, match="preprocessor must keep"):
        SPASelector(initial=0, preprocessor=PCA(n_components=3)).fit(X, y)


def test_spa_rejects_a_constant_fixed_start():
    X, y = _regression_problem()
    X[:, 2] = 1.0

    with pytest.raises(ValueError, match="zero-norm projection"):
        SPASelector(initial=2, cv=3).fit(X, y)
