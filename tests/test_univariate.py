import numpy as np
import pytest
from sklearn.base import clone

from spectralselector import WavelengthSelectKBest


def test_f_regression_selects_informative_variable():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 6))
    y = 8.0 * X[:, 3] + rng.normal(scale=0.05, size=80)
    selector = WavelengthSelectKBest(k=1, score_func="f_regression").fit(X, y)

    np.testing.assert_array_equal(selector.selected_indices_, [3])
    assert selector.transform(X).shape == (80, 1)
    assert clone(selector).get_params()["k"] == 1


def test_mutual_information_is_reproducible():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 5))
    y = X[:, 0] ** 2 + rng.normal(scale=0.1, size=50)
    kwargs = {"k": 2, "score_func": "mutual_info_regression", "random_state": 12}

    first = WavelengthSelectKBest(**kwargs).fit(X, y)
    second = WavelengthSelectKBest(**kwargs).fit(X, y)

    np.testing.assert_array_equal(first.selection_order_, second.selection_order_)
    np.testing.assert_allclose(first.scores_, second.scores_)


@pytest.mark.parametrize(
    ("params", "labels", "error", "match"),
    [
        ({"score_func": "chi2"}, False, ValueError, "score_func must be one of"),
        ({"score_func": "f_classif"}, False, ValueError, "binary or multiclass"),
        ({"k": True}, False, TypeError, "positive integer or 'all'"),
        ({"k": 0}, False, ValueError, "between 1 and the number"),
    ],
)
def test_select_k_best_rejects_invalid_configuration(params, labels, error, match):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    y = (X[:, 0] > 0).astype(int) if labels else X[:, 0]
    with pytest.raises(error, match=match):
        WavelengthSelectKBest(**params).fit(X, y)
