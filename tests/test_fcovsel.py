import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from spectralselector import PLSDA, CovSelSelector, FCovSelSelector


def _plsvarsel_reference_case():
    """Recreate the external non-deflating CovSel/fCovSel comparison inputs."""
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "reference"
        / "plsvarsel_0_10_0_covsel.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = np.arange(1.0, 19.0)
    X = np.column_stack(
        (
            np.sin(rows * 0.31),
            np.cos(rows * 0.17),
            np.sin(rows * 0.73) + 0.1 * rows,
            np.cos(rows * 0.47),
            (rows.astype(int) % 5) - 2.0,
            np.sin(rows * 0.11) * np.cos(rows * 0.23),
        )
    )
    Y = np.column_stack(
        (
            1.5 * X[:, 1] - 0.8 * X[:, 4] + 0.3 * X[:, 5] + 0.05 * np.sin(rows * 1.3),
            -0.7 * X[:, 0] + 1.2 * X[:, 2] + 0.1 * np.cos(rows * 0.9),
        )
    )
    return fixture, X, Y


def test_fcovsel_selects_indices_and_records_diagnostics():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(24, 8))
    y = 3 * X[:, 2] - X[:, 6] + rng.normal(scale=0.1, size=24)
    selector = FCovSelSelector(
        n_features_to_select=3,
        max_features=4,
        wavelengths=np.arange(700, 708),
    ).fit(X, y)

    assert selector.selected_indices_.size == 3
    np.testing.assert_array_equal(
        selector.selection_order_, selector.full_selection_order_[:3]
    )
    assert selector.rmsecv_.shape == (4,)
    assert selector.validation_scores_.shape == (4,)
    assert selector.validation_score_std_.shape == (4,)
    assert selector.fold_selection_orders_.shape == (5, 4)
    assert np.unique(selector.consensus_order_).size == selector.max_features_
    assert selector.transform(X).shape[1] == selector.selected_indices_.size
    assert selector.selected_wavelengths_ is not None
    assert not hasattr(clone(selector), "rmsecv_")


def test_fcovsel_accepts_binary_string_labels_and_classification_metric():
    rng = np.random.default_rng(22)
    X = rng.normal(size=(20, 6))
    labels = np.where(X[:, 1] > 0, "positive", "negative")

    selector = FCovSelSelector(
        n_features_to_select=2,
        max_features=3,
        task="classification",
        scoring="balanced_accuracy",
        cv=4,
        random_state=3,
    ).fit(X, labels)

    assert selector.selected_indices_.size == 2
    assert selector.task_ == "classification"
    assert selector.scoring_ == "balanced_accuracy"
    assert selector.rmsecv_ is None
    assert selector.fold_selection_orders_.shape == (4, 3)
    assert isinstance(selector.estimator_, PLSDA)


def test_fcovsel_supports_multiclass_classification():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(90, 8))
    logits = np.column_stack((2 * X[:, 1], -2 * X[:, 4], 1.5 * X[:, 6]))
    y = np.asarray(["first", "second", "third"])[np.argmax(logits, axis=1)]

    selector = FCovSelSelector(
        n_features_to_select="auto",
        max_features=4,
        task="classification",
        scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=7),
    ).fit(X, y)

    assert selector.task_ == "classification"
    assert 1 <= selector.n_features_selected_ <= 4
    assert selector.validation_scores_.shape == (4,)
    assert np.isfinite(selector.validation_scores_).all()


def test_fcovsel_supports_multiresponse_regression_and_custom_metric():
    rng = np.random.default_rng(28)
    X = rng.normal(size=(50, 7))
    y = np.column_stack((2 * X[:, 1] - X[:, 5], -3 * X[:, 3] + X[:, 6]))

    selector = FCovSelSelector(
        n_features_to_select=3,
        max_features=4,
        task="regression",
        scoring="r2",
        cv=3,
        random_state=8,
    ).fit(X, y)

    assert selector.task_ == "regression"
    assert selector.scoring_ == "r2"
    assert selector.rmsecv_ is None
    assert selector.validation_scores_.shape == (4,)


def test_fcovsel_cv_minus_one_resolves_to_leave_one_out():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(12, 6))
    y = 2 * X[:, 2] + rng.normal(scale=0.1, size=12)

    selector = FCovSelSelector(
        n_features_to_select=2,
        max_features=3,
        task="regression",
        scoring="neg_mean_absolute_error",
        cv=-1,
    ).fit(X, y)

    assert isinstance(selector.cv_, LeaveOneOut)
    assert selector.fold_selection_orders_.shape == (X.shape[0], 3)
    assert selector.scoring_ == "neg_mean_absolute_error"


def test_fcovsel_accepts_group_aware_custom_splitter():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(30, 6))
    groups = np.repeat(np.arange(3), 10)
    y = 2 * X[:, 1] - X[:, 4] + rng.normal(scale=0.1, size=30)

    selector = FCovSelSelector(
        n_features_to_select=2,
        max_features=3,
        task="regression",
        scoring="neg_mean_absolute_error",
        cv=GroupKFold(3),
    ).fit(X, y, groups=groups)

    assert selector.fold_selection_orders_.shape == (3, 3)


def test_fcovsel_rejects_invalid_fold_count():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(20, 5))
    y = X[:, 1] + rng.normal(scale=0.1, size=20)

    with pytest.raises(ValueError, match="cv must be -1"):
        FCovSelSelector(n_features_to_select=2, cv=1).fit(X, y)


def test_fcovsel_runs_inside_cross_validated_pipeline():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(30, 7))
    y = 2 * X[:, 1] + X[:, 5] + rng.normal(scale=0.1, size=30)
    pipeline = Pipeline(
        [
            ("select", FCovSelSelector(n_features_to_select=2, max_features=3)),
            ("model", LinearRegression()),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=8),
        scoring="neg_root_mean_squared_error",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_fcovsel_does_not_mutate_max_features_when_data_require_clipping():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(8, 5))
    y = 2 * X[:, 1] + rng.normal(scale=0.1, size=8)
    selector = FCovSelSelector(n_features_to_select=1, max_features=20)

    selector.fit(X, y)

    assert selector.max_features == 20
    assert selector.max_features_ == 5
    assert clone(selector).max_features == 20


def test_fcovsel_forward_order_matches_covsel_for_single_response():
    rng = np.random.default_rng(25)
    X = rng.normal(size=(60, 9))
    y = 3.0 * X[:, 1] - 2.0 * X[:, 6] + 0.5 * X[:, 8] + rng.normal(scale=0.1, size=60)

    covsel_order = CovSelSelector._covsel(X, y[:, None], 4)[0]
    X_centered = X - np.mean(X, axis=0, keepdims=True)
    y_centered = (y - np.mean(y)).reshape(-1, 1)
    fcovsel_order = FCovSelSelector._forward_order(X_centered, y_centered, 4)

    np.testing.assert_array_equal(fcovsel_order, covsel_order)


def test_fcovsel_order_matches_versioned_plsvarsel_reference_fixture():
    fixture, X, Y = _plsvarsel_reference_case()
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    order = FCovSelSelector._forward_order(
        X_centered,
        Y_centered,
        fixture["n_features_to_select"],
    )

    assert order.tolist() == fixture["selected_indices"]


def test_fcovsel_fold_consensus_is_unique_when_position_modes_repeat():
    fold_orders = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
        ]
    )

    consensus = FCovSelSelector._unique_fold_consensus(fold_orders, n_features=3)

    np.testing.assert_array_equal(consensus, [0, 1, 2])


def test_fcovsel_fixed_count_uses_full_data_order_not_fold_consensus(monkeypatch):
    rng = np.random.default_rng(26)
    X = rng.normal(size=(20, 5))
    y = 2.0 * X[:, 3] - X[:, 1] + rng.normal(scale=0.1, size=20)

    def diagnostic_with_different_consensus(self, X, y, n_steps, splits):
        return (
            np.linspace(1.0, 3.0, n_steps),
            np.zeros(n_steps),
            np.arange(n_steps, dtype=np.int64),
            np.tile(np.arange(n_steps, dtype=np.int64), (len(splits), 1)),
        )

    monkeypatch.setattr(
        FCovSelSelector,
        "_validate_sequence",
        diagnostic_with_different_consensus,
    )
    selector = FCovSelSelector(n_features_to_select=2, max_features=3).fit(X, y)

    np.testing.assert_array_equal(
        selector.selection_order_, selector.full_selection_order_[:2]
    )


def test_fcovsel_auto_count_maximizes_configured_validation_score(monkeypatch):
    rng = np.random.default_rng(30)
    X = rng.normal(size=(24, 5))
    y = 2 * X[:, 2] + rng.normal(scale=0.1, size=24)

    def known_scores(self, X, y, n_steps, splits):
        return (
            np.asarray([0.1, 0.8, 0.3]),
            np.zeros(n_steps),
            np.arange(n_steps, dtype=np.int64),
            np.tile(np.arange(n_steps, dtype=np.int64), (len(splits), 1)),
        )

    monkeypatch.setattr(FCovSelSelector, "_validate_sequence", known_scores)
    selector = FCovSelSelector(
        n_features_to_select="auto",
        max_features=3,
        scoring="r2",
        cv=3,
    ).fit(X, y)

    assert selector.n_features_selected_ == 2
    assert selector.best_score_ == 0.8


def _binary_problem(seed=40, n=60, p=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.where(X[:, 3] - 0.8 * X[:, 7] + 0.3 * rng.normal(size=n) > 0, "b", "a")
    return X, y


def test_fcovsel_param_grid_selects_length_and_hyperparameters_jointly():
    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = _binary_problem()
    cv = StratifiedKFold(4, shuffle=True, random_state=0)
    grid = {"clf__C": [0.01, 1.0, 100.0], "clf__kernel": ["linear", "rbf"]}
    estimator = Pipeline([("scale", StandardScaler()), ("clf", SVC())])
    selector = FCovSelSelector(
        max_features=5,
        estimator=estimator,
        param_grid=grid,
        task="classification",
        scoring="accuracy",
        cv=cv,
    ).fit(X, y)

    params = list(ParameterGrid(grid))
    assert len(selector.cv_results_["params"]) == 5 * len(params)
    assert selector.cv_results_["mean_test_score"].shape == (5 * len(params),)
    assert selector.best_params_ in params
    assert selector.best_n_features_ == selector.n_features_selected_

    # Brute-force reference: fold-local orders, every (length, grid point).
    expected = np.zeros((5, len(params)))
    for train, test in cv.split(X, y):
        Xc = X[train] - X[train].mean(axis=0)
        Yc = selector._numeric_response(y[train])
        Yc = Yc - Yc.mean(axis=0)
        order = FCovSelSelector._forward_order(Xc, Yc, 5)
        for length in range(1, 6):
            cols = order[:length]
            for k, p in enumerate(params):
                model = (
                    clone(estimator).set_params(**p).fit(X[train][:, cols], y[train])
                )
                expected[length - 1, k] += (
                    np.mean(model.predict(X[test][:, cols]) == y[test]) / 4
                )
    np.testing.assert_allclose(
        selector.cv_results_["mean_test_score"], expected.ravel(), atol=1e-12
    )
    best_length = int(np.argmax(expected.max(axis=1))) + 1
    assert selector.best_n_features_ == best_length
    np.testing.assert_allclose(selector.best_score_, expected.max())
    assert selector.best_params_ == params[int(np.argmax(expected[best_length - 1]))]
    # The final estimator uses the selected hyper-parameters.
    assert selector.estimator_.get_params()["clf__C"] == selector.best_params_["clf__C"]


def test_fcovsel_param_grid_caps_pipeline_components():
    from sklearn.preprocessing import StandardScaler

    X, y = _binary_problem(seed=41)
    selector = FCovSelSelector(
        max_features=4,
        estimator=Pipeline(
            [("scale", StandardScaler()), ("clf", PLSDA(scale_pls=False))]
        ),
        param_grid={"clf__n_components": [1, 3, 6]},
        task="classification",
        cv=StratifiedKFold(3, shuffle=True, random_state=1),
    ).fit(X, y)

    fitted = selector.estimator_.get_params()["clf__n_components"]
    assert fitted <= selector.n_features_selected_
    # Grid points above the length are capped, so they score like the cap.
    scores = selector.cv_results_["mean_test_score"].reshape(4, 3)
    np.testing.assert_allclose(scores[0, 0], scores[0, 1])
    np.testing.assert_allclose(scores[0, 0], scores[0, 2])


def test_fcovsel_single_grid_point_matches_no_grid():
    X, y = _binary_problem(seed=42)
    kwargs = dict(
        max_features=4,
        estimator=PLSDA(n_components=2, scale_pls=False),
        task="classification",
        cv=StratifiedKFold(3, shuffle=True, random_state=2),
    )
    plain = FCovSelSelector(**kwargs).fit(X, y)
    gridded = FCovSelSelector(param_grid={"n_components": [2]}, **kwargs).fit(X, y)
    np.testing.assert_allclose(plain.validation_scores_, gridded.validation_scores_)
    np.testing.assert_array_equal(plain.selected_indices_, gridded.selected_indices_)
    assert plain.best_params_ == {}


def test_fcovsel_preprocessor_is_refitted_inside_every_fold():
    from sklearn.base import BaseEstimator, TransformerMixin

    class RecordingCentering(TransformerMixin, BaseEstimator):
        fitted_sizes: ClassVar[list[int]] = []

        def fit(self, X, y=None):
            RecordingCentering.fitted_sizes.append(len(X))
            self.mean_ = np.asarray(X).mean(axis=0)
            return self

        def transform(self, X):
            return np.asarray(X) - self.mean_

    X, y = _binary_problem(seed=43, n=40)
    X = X + 5.0
    RecordingCentering.fitted_sizes = []
    cv = StratifiedKFold(4, shuffle=True, random_state=3)
    selector = FCovSelSelector(
        max_features=3,
        task="classification",
        cv=cv,
        preprocessor=RecordingCentering(),
    ).fit(X, y)

    train_sizes = sorted(len(train) for train, _ in cv.split(X, y))
    assert sorted(RecordingCentering.fitted_sizes[1:]) == train_sizes
    assert RecordingCentering.fitted_sizes[0] == len(X)
    expected = X - X.mean(axis=0)
    np.testing.assert_allclose(selector.transform(X), expected[:, selector.support_])


def test_fcovsel_preprocessor_must_keep_columns():
    from sklearn.decomposition import PCA

    X, y = _binary_problem(seed=44)
    with pytest.raises(ValueError, match="preprocessor must keep"):
        FCovSelSelector(
            max_features=2, task="classification", preprocessor=PCA(n_components=3)
        ).fit(X, y)
