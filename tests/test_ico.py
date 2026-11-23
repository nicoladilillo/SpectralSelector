import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from spectralselector import ICOSelector


def _data(n=60, p=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 5:10].sum(axis=1) + 0.05 * rng.normal(size=n)
    return X, y


def _estimator():
    return PLSRegression(n_components=3, scale=False)


def test_ico_selects_the_informative_interval_and_is_reproducible():
    X, y = _data()
    kwargs = dict(
        estimator=_estimator(),
        n_intervals=8,
        n_combinations=60,
        best_fraction=0.1,
        random_state=0,
    )
    first = ICOSelector(**kwargs).fit(X, y)
    second = ICOSelector(**kwargs).fit(X, y)

    np.testing.assert_array_equal(first.selected_indices_, second.selected_indices_)
    # Columns 5-9 carry the signal: at least most of them must be retained.
    assert set(range(5, 10)) <= set(first.selected_indices_.tolist())
    assert first.n_iterations_ >= 1
    assert len(first.intervals_) == 8
    assert first.transform(X).shape[1] == first.selected_indices_.size


def test_wbs_matrix_shape_and_zero_weight_intervals_are_never_drawn():
    selector = ICOSelector(n_combinations=50)
    weights = np.asarray([1.0, 0.0, 1.0, 0.0])
    combos = selector._weighted_bootstrap(weights, 4, np.random.default_rng(0))

    assert combos.shape == (50, 4)
    assert combos[:, [1, 3]].sum() == 0
    assert (combos.sum(axis=1) >= 1).all()


def test_first_iteration_combinations_keep_about_632_per_mille_of_intervals():
    selector = ICOSelector(n_combinations=400)
    combos = selector._weighted_bootstrap(np.ones(40), 40, np.random.default_rng(1))
    assert combos.sum(axis=1).mean() / 40 == pytest.approx(0.632, abs=0.03)


def test_physical_equal_width_intervals_with_irregular_axis():
    axis = np.asarray([400, 401, 402, 403, 500, 501, 700, 701, 800, 1000])
    selector = ICOSelector(n_intervals=3, wavelengths=axis)
    selector.n_features_in_ = 10
    groups = selector._make_intervals(axis.astype(float))
    # Equal physical width: edges at 400, 600, 800, 1000.
    assert [g.tolist() for g in groups] == [
        [0, 1, 2, 3, 4, 5],
        [6, 7],
        [8, 9],
    ]


def test_local_search_never_selects_fewer_than_one_variable():
    X, y = _data()
    selector = ICOSelector(
        estimator=_estimator(),
        n_intervals=4,
        n_combinations=20,
        random_state=2,
    ).fit(X, y)
    assert selector.selected_indices_.size >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_intervals": 0},
        {"n_combinations": 0},
        {"best_fraction": 0.0},
        {"best_fraction": 1.5},
        {"max_iterations": 0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    X, y = _data()
    with pytest.raises(ValueError):
        ICOSelector(estimator=_estimator(), **kwargs).fit(X, y)


def test_interval_count_is_capped_at_the_number_of_features():
    X, y = _data(p=6)
    selector = ICOSelector(
        estimator=_estimator(), n_intervals=500, n_combinations=10, random_state=0
    ).fit(X, y)
    assert len(selector.intervals_) == 6
