import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from spectralselector import MIVIFSelector
from spectralselector.mivif import _vif_from_correlation


def _ols_vif(X: np.ndarray) -> np.ndarray:
    """Cheng et al. Eq. (3) computed literally with intercept OLS regressions."""
    values = []
    for column in range(X.shape[1]):
        others = np.delete(X, column, axis=1)
        design = np.column_stack((np.ones(X.shape[0]), others))
        coefficients, *_ = np.linalg.lstsq(design, X[:, column], rcond=None)
        residual = X[:, column] - design @ coefficients
        centered = X[:, column] - X[:, column].mean()
        r_squared = 1.0 - (residual @ residual) / (centered @ centered)
        values.append(1.0 / (1.0 - r_squared))
    return np.asarray(values)


def test_vif_from_correlation_matches_ols_definition():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    X[:, 3] = X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.3, size=50)

    np.testing.assert_allclose(
        _vif_from_correlation(np.corrcoef(X, rowvar=False)), _ols_vif(X), rtol=1e-10
    )


def test_vif_is_infinite_for_perfect_collinearity():
    x = np.arange(10.0)
    correlation = np.corrcoef(np.column_stack((x, 2 * x + 1)), rowvar=False)

    assert np.isinf(_vif_from_correlation(correlation)).all()


def _collinear_problem(random_state: int = 3):
    rng = np.random.default_rng(random_state)
    n = 120
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    noise = rng.normal(size=(n, 6))
    X = np.column_stack(
        (
            noise[:, 0],
            a,  # most informative
            a + rng.normal(scale=0.02, size=n),  # near duplicate of column 1
            noise[:, 1],
            b,  # second, independent signal
            noise[:, 2:],
        )
    )
    y = 3.0 * a + 1.5 * b + rng.normal(scale=0.05, size=n)
    return X, y


def test_mivif_follows_published_forward_procedure():
    X, y = _collinear_problem()

    selector = MIVIFSelector(random_state=0).fit(X, y)

    # The higher-MI member of the duplicated pair is accepted first.
    first = int(selector.selection_order_[0])
    assert first in {1, 2}
    duplicate = 3 - first
    assert duplicate not in selector.selected_indices_
    assert 4 in selector.selected_indices_
    assert selector.n_rejected_ >= 1
    assert selector.n_rejected_ + selector.selected_indices_.size == X.shape[1]
    # Examination order is decreasing MI; acceptances preserve that order.
    assert np.all(np.diff(selector.scores_[selector.mi_order_]) <= 0)
    positions = {int(v): i for i, v in enumerate(selector.mi_order_)}
    assert [positions[int(v)] for v in selector.selection_order_] == sorted(
        positions[int(v)] for v in selector.selection_order_
    )
    # The final set satisfies the multicollinearity criterion.
    assert np.all(selector.vif_ <= 10.0)
    np.testing.assert_allclose(
        selector.vif_, _ols_vif(X[:, selector.selection_order_]), rtol=1e-8
    )


def test_mivif_lower_threshold_keeps_fewer_or_equal_variables():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(80, 3))
    X = base @ rng.normal(size=(3, 25)) + rng.normal(scale=0.3, size=(80, 25))
    y = base[:, 0] - base[:, 2]

    loose = MIVIFSelector(vif_threshold=50.0, random_state=1).fit(X, y)
    strict = MIVIFSelector(vif_threshold=2.0, random_state=1).fit(X, y)

    assert strict.selected_indices_.size <= loose.selected_indices_.size
    assert np.all(strict.vif_ <= 2.0)


def test_mivif_max_features_stops_forward_selection():
    X, y = _collinear_problem()

    full = MIVIFSelector(random_state=0).fit(X, y)
    capped = MIVIFSelector(max_features=1, random_state=0).fit(X, y)

    assert capped.selection_order_.tolist() == full.selection_order_[:1].tolist()


def test_mivif_never_selects_a_constant_column():
    X, y = _collinear_problem()
    X[:, 0] = 7.0

    selector = MIVIFSelector(random_state=0).fit(X, y)

    assert 0 not in selector.selected_indices_


def test_mivif_selection_is_bounded_by_sample_count():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(12, 60))
    y = X[:, 5] + rng.normal(scale=0.1, size=12)

    selector = MIVIFSelector(random_state=0).fit(X, y)

    # n samples support at most n - 1 linearly independent centered columns.
    assert selector.selected_indices_.size <= X.shape[0] - 1
    assert np.isfinite(selector.vif_).all()


def test_mivif_is_invariant_to_column_rescaling():
    X, y = _collinear_problem()
    scales = np.geomspace(0.01, 100.0, X.shape[1])

    original = MIVIFSelector(random_state=2).fit(X, y)
    rescaled = MIVIFSelector(random_state=2).fit(X * scales + 5.0, y)

    assert original.selection_order_.tolist() == rescaled.selection_order_.tolist()


def test_mivif_classification_with_string_labels():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(150, 12))
    y = np.where(X[:, 4] + 0.5 * X[:, 9] > 0, "healthy", "stressed")

    selector = MIVIFSelector(random_state=0, wavelengths=np.arange(12) * 5.0).fit(X, y)

    assert selector.task_ == "classification"
    assert selector.selection_order_[0] == 4
    assert selector.selected_wavelengths_ is not None


def test_mivif_explicit_regression_for_integer_response():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 6))
    y = np.round(3 * X[:, 2]).astype(int)

    assert MIVIFSelector(random_state=0).fit(X, y).task_ == "classification"
    assert (
        MIVIFSelector(task="regression", random_state=0).fit(X, y).task_ == "regression"
    )


def test_mivif_runs_inside_cross_validated_pipeline():
    X, y = _collinear_problem()
    pipeline = Pipeline(
        [
            ("select", MIVIFSelector(random_state=0)),
            ("model", LinearRegression()),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=0),
        scoring="r2",
    )

    assert np.all(scores > 0.9)


def test_mivif_selection_works_with_pls_downstream():
    X, y = _collinear_problem()
    selector = MIVIFSelector(random_state=0).fit(X, y)

    model = PLSRegression(n_components=1).fit(selector.transform(X), y)

    assert model.score(selector.transform(X), y) > 0.5


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"vif_threshold": 1.0}, "vif_threshold"),
        ({"vif_threshold": np.inf}, "vif_threshold"),
        ({"max_features": 0}, "max_features"),
        ({"n_neighbors": 0}, "n_neighbors"),
        ({"task": "ranking"}, "task"),
    ],
)
def test_mivif_rejects_invalid_parameters(parameters, message):
    X, y = _collinear_problem()

    with pytest.raises(ValueError, match=message):
        MIVIFSelector(**parameters).fit(X, y)


def test_mivif_rejects_multioutput_target():
    X, y = _collinear_problem()

    with pytest.raises(ValueError):
        MIVIFSelector().fit(X, np.column_stack((y, y)))
