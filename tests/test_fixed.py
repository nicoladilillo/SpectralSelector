import numpy as np
import pytest
from sklearn.base import clone

from spectralselector import IndexSelector


def test_index_selector_returns_numpy_columns_and_metadata():
    X = np.arange(24, dtype=float).reshape(4, 6)
    selector = IndexSelector([4, 1], wavelengths=np.arange(700, 706)).fit(X)

    transformed = selector.transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (4, 2)
    np.testing.assert_array_equal(transformed, X[:, [1, 4]])
    np.testing.assert_array_equal(selector.selected_indices_, [1, 4])
    np.testing.assert_array_equal(selector.selection_order_, [4, 1])
    np.testing.assert_array_equal(selector.selected_wavelengths_, [701, 704])
    np.testing.assert_array_equal(
        selector.get_feature_names_out(["a", "b", "c", "d", "e", "f"]),
        ["b", "e"],
    )
    assert clone(selector).get_params()["indices"] == [4, 1]


def test_index_selector_rejects_invalid_indices():
    X = np.ones((3, 4))
    with pytest.raises(ValueError, match="duplicates"):
        IndexSelector([1, 1]).fit(X)
    with pytest.raises(ValueError, match="between"):
        IndexSelector([4]).fit(X)


def test_wavelength_length_must_match_features():
    with pytest.raises(ValueError, match="one value per input feature"):
        IndexSelector([0], wavelengths=[700, 701]).fit(np.ones((3, 4)))


@pytest.mark.parametrize(
    ("indices", "wavelengths", "error", "match"),
    [
        ([], None, ValueError, "non-empty one-dimensional"),
        ([[0, 1]], None, ValueError, "non-empty one-dimensional"),
        ([0.5], None, TypeError, "integers"),
        ([0], np.ones((2, 2)), ValueError, "one-dimensional array"),
        ([0], [700, np.nan, 702, 703], ValueError, "finite numeric"),
    ],
)
def test_index_selector_rejects_malformed_inputs(indices, wavelengths, error, match):
    with pytest.raises(error, match=match):
        IndexSelector(indices, wavelengths=wavelengths).fit(np.ones((3, 4)))
