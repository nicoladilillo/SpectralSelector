import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import NotFittedError

from spectralselector import (
    BOSSSelector,
    CARSSelector,
    CovSelSelector,
    FCovSelSelector,
    GeneticAlgorithmSelector,
    IndexSelector,
    SelectivityRatioSelector,
    VIPSelector,
    WavelengthSelectKBest,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(scope="module")
def fitted_selectors():
    """Fit one compact deterministic instance of every diagnostic family."""
    rng = np.random.default_rng(51)
    X = rng.normal(size=(36, 8))
    y = 3 * X[:, 1] - 2 * X[:, 6] + rng.normal(scale=0.2, size=X.shape[0])
    wavelengths = np.linspace(700, 900, X.shape[1])
    model = PLSRegression(n_components=1, scale=False)

    selectors = [
        IndexSelector([1, 6], wavelengths=wavelengths),
        WavelengthSelectKBest(k=3, wavelengths=wavelengths),
        VIPSelector(n_components=2, n_features_to_select=3, wavelengths=wavelengths),
        SelectivityRatioSelector(
            n_components=2,
            n_features_to_select=3,
            wavelengths=wavelengths,
        ),
        CovSelSelector(3, task="regression", wavelengths=wavelengths),
        FCovSelSelector(2, max_features=3, wavelengths=wavelengths),
        CARSSelector(
            cv=2,
            max_components=2,
            n_runs=2,
            n_iterations=3,
            min_features=2,
            random_state=2,
            wavelengths=wavelengths,
        ),
        BOSSSelector(
            estimator=model,
            cv=2,
            max_components=2,
            n_submodels=3,
            top_fraction=0.5,
            max_iterations=2,
            min_features=2,
            random_state=3,
            wavelengths=wavelengths,
        ),
        GeneticAlgorithmSelector(
            estimator=model,
            cv=2,
            population_size=4,
            generations=2,
            tournament_size=2,
            min_groups=2,
            max_groups=3,
            random_state=4,
            wavelengths=wavelengths,
        ),
    ]
    return [selector.fit(X, y) for selector in selectors]


def test_every_selector_can_plot_its_diagnostics(fitted_selectors):
    expected_panels = [1, 1, 1, 1, 2, 2, 2, 2, 2]

    for selector, panel_count in zip(fitted_selectors, expected_panels, strict=True):
        measured_time = selector.selection_time_
        assert isinstance(measured_time, float)
        assert np.isfinite(measured_time)
        assert measured_time >= 0
        assert not hasattr(clone(selector), "selection_time_")
        figure, axes = selector.plot_diagnostics()
        assert axes.shape == (panel_count,)
        assert axes[0].figure is figure
        assert any(axis.get_title() for axis in axes)
        assert selector.selection_time_ == measured_time
        plt.close(figure)


def test_plot_diagnostics_accepts_existing_axes(fitted_selectors):
    figure, axes = plt.subplots(2, 1)

    returned_figure, returned_axes = fitted_selectors[4].plot_diagnostics(axes=axes)

    assert returned_figure is figure
    assert np.array_equal(returned_axes, axes)
    plt.close(figure)


def test_plot_diagnostics_validates_axis_count(fitted_selectors):
    figure, axis = plt.subplots()

    with pytest.raises(ValueError, match="requires 2 axes"):
        fitted_selectors[4].plot_diagnostics(axes=axis)
    plt.close(figure)


def test_plot_diagnostics_requires_a_fitted_selector():
    with pytest.raises(NotFittedError):
        IndexSelector([0]).plot_diagnostics()
