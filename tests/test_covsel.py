import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import CovSelSelector


def _plsvarsel_reference_case():
    """Recreate inputs used for the independent plsVarSel CovSel output."""
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


def test_covsel_matches_two_step_equation_fixture():
    X = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    y = 2.0 * X[:, 0] + X[:, 1]

    order, scores, _, explained_x, explained_y = CovSelSelector._covsel(
        X, y[:, None], 2
    )

    np.testing.assert_array_equal(order, [0, 1])
    np.testing.assert_allclose(scores, [16.0 / 9.0, 4.0 / 9.0])
    np.testing.assert_allclose(explained_x, [0.5, 1.0])
    np.testing.assert_allclose(explained_y, [0.8, 1.0])


def test_covsel_matches_versioned_plsvarsel_reference_fixture():
    fixture, X, Y = _plsvarsel_reference_case()

    selector = CovSelSelector(
        n_features_to_select=fixture["n_features_to_select"],
        task="regression",
        scale_y=False,
    ).fit(X, Y)

    assert selector.selection_order_.tolist() == fixture["selected_indices"]
    np.testing.assert_allclose(
        selector.x_explained_variance_,
        fixture["x_explained_variance"],
        rtol=0.0,
        atol=fixture["absolute_tolerance"],
    )
    np.testing.assert_allclose(
        selector.y_explained_variance_,
        fixture["y_explained_variance"],
        rtol=0.0,
        atol=fixture["absolute_tolerance"],
    )


def test_covsel_selects_regression_variables_and_records_diagnostics():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(140, 12))
    y = 5 * X[:, 2] - 3 * X[:, 9] + rng.normal(scale=0.15, size=140)
    wavelengths = np.linspace(700, 900, X.shape[1])

    selector = CovSelSelector(
        n_features_to_select=2,
        task="regression",
        wavelengths=wavelengths,
    ).fit(X, y)

    assert set(selector.selection_order_) == {2, 9}
    assert selector.selected_indices_.tolist() == [2, 9]
    assert selector.selected_wavelengths_.shape == (2,)
    assert selector.covariance_scores_.shape == (2,)
    assert selector.criterion_history_.shape == (2, 12)
    assert np.all(np.diff(selector.x_explained_variance_) >= -1e-12)
    assert np.all(np.diff(selector.y_explained_variance_) >= -1e-12)
    assert not hasattr(clone(selector), "selection_order_")


def test_covsel_supports_multiresponse_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 10))
    Y = np.column_stack(
        (
            4 * X[:, 1] + rng.normal(scale=0.1, size=120),
            -3 * X[:, 7] + rng.normal(scale=0.1, size=120),
        )
    )

    selector = CovSelSelector(
        n_features_to_select=2,
        task="regression",
    ).fit(X, Y)

    assert set(selector.selected_indices_) == {1, 7}
    assert selector.task_ == "regression"


def test_covsel_can_autoscale_multiresponse_y():
    X = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    Y = np.column_stack((X[:, 0], 50.0 * X[:, 1]))

    unscaled = CovSelSelector(1, task="regression").fit(X, Y)
    scaled = CovSelSelector(1, task="regression", scale_y=True).fit(X, Y)

    assert unscaled.selection_order_.tolist() == [1]
    assert scaled.selection_order_.tolist() == [0]
    np.testing.assert_allclose(scaled.y_scale_, np.std(Y, axis=0, ddof=1))


def test_covsel_autoscaling_rejects_a_constant_response_column():
    rng = np.random.default_rng(46)
    X = rng.normal(size=(40, 5))
    Y = np.column_stack((X[:, 1], np.ones(X.shape[0])))

    with pytest.raises(ValueError, match="variation in every response"):
        CovSelSelector(1, task="regression", scale_y=True).fit(X, Y)


def test_covsel_breaks_exact_covariance_ties_by_source_index():
    rng = np.random.default_rng(47)
    signal = rng.normal(size=60)
    X = np.column_stack((signal, signal.copy(), rng.normal(size=60)))

    selector = CovSelSelector(1, task="regression").fit(X, signal)

    assert selector.selection_order_.tolist() == [0]
    assert selector.criterion_history_[0, 0] == pytest.approx(
        selector.criterion_history_[0, 1]
    )


def test_covsel_reports_predictive_covariance_exhaustion_for_duplicate_columns():
    rng = np.random.default_rng(48)
    signal = rng.normal(size=60)
    X = np.column_stack((signal, signal.copy()))

    with pytest.raises(ValueError, match="exhausted predictive covariance"):
        CovSelSelector(2, task="regression").fit(X, signal)


def test_covsel_handles_correlated_rank_deficient_predictors():
    rng = np.random.default_rng(49)
    first = rng.normal(size=100)
    second = rng.normal(size=100)
    X = np.column_stack(
        (first, first + 1e-5 * rng.normal(size=100), second, first + second)
    )
    y = 3.0 * first - 2.0 * second

    selector = CovSelSelector(2, task="regression").fit(X, y)

    assert selector.selected_indices_.size == 2
    assert np.isfinite(selector.covariance_scores_).all()
    assert np.all(np.diff(selector.x_explained_variance_) >= -1e-12)
    assert np.all(np.diff(selector.y_explained_variance_) >= -1e-12)


def test_covsel_rejects_non_boolean_scale_y():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.arange(20, dtype=float)

    with pytest.raises(TypeError, match="scale_y"):
        CovSelSelector(1, scale_y="yes").fit(X, y)


def test_covsel_indicator_encodes_multiclass_response():
    rng = np.random.default_rng(43)
    X = rng.normal(size=(150, 9))
    logits = np.column_stack((2 * X[:, 1], 2 * X[:, 4], 2 * X[:, 7]))
    y = np.asarray(["a", "b", "c"])[np.argmax(logits, axis=1)]

    selector = CovSelSelector(
        n_features_to_select=3,
        task="classification",
    ).fit(X, y)

    assert selector.task_ == "classification"
    assert selector.classes_.tolist() == ["a", "b", "c"]
    assert selector.selected_indices_.size == 3
    assert {1, 4, 7}.intersection(selector.selected_indices_)


def test_covsel_runs_inside_regression_pipeline():
    rng = np.random.default_rng(44)
    X = rng.normal(size=(100, 14))
    y = 3 * X[:, 3] - X[:, 11] + rng.normal(scale=0.2, size=100)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", CovSelSelector(4, task="regression")),
            ("model", LinearRegression()),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(4, shuffle=True, random_state=8),
        scoring="neg_root_mean_squared_error",
    )

    assert scores.shape == (4,)
    assert np.isfinite(scores).all()


def test_covsel_runs_inside_classification_pipeline():
    rng = np.random.default_rng(45)
    X = rng.normal(size=(150, 14))
    y = (3 * X[:, 2] - 2 * X[:, 10] + rng.normal(scale=0.2, size=150) > 0).astype(int)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", CovSelSelector(4, task="classification")),
            ("model", LogisticRegression()),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(3, shuffle=True, random_state=9),
        scoring="balanced_accuracy",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.mean(scores) > 0.8


def test_covsel_rejects_constant_response():
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.ones(20)

    with pytest.raises(ValueError, match="variation in y"):
        CovSelSelector(1, task="regression").fit(X, y)


@pytest.mark.parametrize(
    ("params", "y_kind", "match"),
    [
        ({"task": "ranking"}, "numeric", "task must be"),
        (
            {"n_features_to_select": "auto", "max_features": 0},
            "numeric",
            "max_features",
        ),
        ({"n_features_to_select": 0}, "numeric", "n_features_to_select"),
        ({"task": "regression"}, "strings", "numeric response"),
        ({}, "constant-x", "variation in X"),
    ],
)
def test_covsel_rejects_invalid_configuration(params, y_kind, match):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 5))
    y = X[:, 0]
    if y_kind == "strings":
        y = np.where(y > 0, "high", "low")
    if y_kind == "constant-x":
        X = np.ones_like(X)
    with pytest.raises(ValueError, match=match):
        CovSelSelector(**{"n_features_to_select": 2, **params}).fit(X, y)


def test_covsel_auto_length_with_scaled_response():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 6))
    Y = np.column_stack((X[:, 1], 100 * X[:, 4])) + rng.normal(scale=0.05, size=(30, 2))
    selector = CovSelSelector(
        "auto", scale_y=True, max_features=3, cv=3, random_state=0
    ).fit(X, Y)

    assert set(selector.selected_indices_) <= set(range(6))
    assert 1 <= selector.selected_indices_.size <= 3
