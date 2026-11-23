"""Optional Matplotlib diagnostics for fitted wavelength selectors."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_is_fitted


def plot_selector_diagnostics(
    selector: Any,
    *,
    axes: Any = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Any, NDArray[Any]]:
    """Plot diagnostics chosen for the fitted selector type.

    This function is normally called through ``selector.plot_diagnostics()``.
    Matplotlib remains optional because it is imported only inside this call.
    """
    check_is_fitted(selector, "support_")
    # Runtime imports avoid making plotting or Matplotlib part of package import.
    from .boss import BOSSSelector
    from .cars import CARSSelector
    from .covsel import CovSelSelector
    from .fcovsel import FCovSelSelector
    from .genetic import GeneticAlgorithmSelector

    # fCovSel always, and CovSel/MI-VIF with n_features_to_select="auto",
    # chose their length with a model: show the validation curve.
    if isinstance(selector, FCovSelSelector) or hasattr(selector, "validation_scores_"):
        return _plot_fcovsel(selector, axes, figsize)
    if isinstance(selector, CovSelSelector):
        return _plot_covsel(selector, axes, figsize)
    if isinstance(selector, CARSSelector):
        return _plot_cars(selector, axes, figsize)
    if isinstance(selector, BOSSSelector):
        return _plot_boss(selector, axes, figsize)
    if isinstance(selector, GeneticAlgorithmSelector):
        return _plot_genetic(selector, axes, figsize)
    if hasattr(selector, "scores_"):
        return _plot_feature_scores(selector, axes, figsize)
    return _plot_support(selector, axes, figsize)


def _matplotlib():
    """Import pyplot lazily and provide an actionable optional-extra error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Diagnostic plots require Matplotlib. Install it with "
            "`pip install 'spectralselector[plot]'`."
        ) from error
    return plt


def _make_axes(
    axes: Any,
    n_panels: int,
    figsize: tuple[float, float] | None,
) -> tuple[Any, NDArray[Any]]:
    """Create or validate a flat array containing the requested axes."""
    plt = _matplotlib()
    if axes is None:
        size = figsize or (9.0, 3.2 * n_panels)
        figure, created = plt.subplots(n_panels, 1, figsize=size, squeeze=False)
        axes_array = created.ravel()
    else:
        axes_array = np.asarray(axes, dtype=object).reshape(-1)
        if axes_array.size != n_panels:
            raise ValueError(
                f"This diagnostic plot requires {n_panels} axes; received "
                f"{axes_array.size}."
            )
        figure = axes_array[0].figure
    return figure, axes_array


def _feature_axis(selector: Any) -> tuple[NDArray[np.float64], str]:
    """Return physical wavelengths when available, otherwise column indices."""
    if selector.wavelengths_in_ is None:
        return np.arange(selector.n_features_in_, dtype=np.float64), "Feature index"
    return np.asarray(selector.wavelengths_in_, dtype=np.float64), "Wavelength"


def _finite_for_plot(values: Any) -> NDArray[np.float64]:
    """Replace infinite diagnostics with a finite plotting height."""
    result = np.asarray(values, dtype=np.float64).copy()
    finite = result[np.isfinite(result)]
    ceiling = 1.0 if finite.size == 0 else max(1.0, float(np.max(finite)) * 1.05)
    result[np.isposinf(result)] = ceiling
    result[np.isneginf(result)] = -ceiling
    return result


def _plot_selected_rank(axis: Any, selector: Any) -> None:
    """Show sequential selection rank at each selected feature position."""
    x, x_label = _feature_axis(selector)
    order = np.asarray(selector.selection_order_, dtype=np.int64)
    ranks = np.arange(1, order.size + 1)
    axis.scatter(x[order], ranks, color="#D1495B", zorder=3)
    axis.vlines(x[order], 0, ranks, color="#D1495B", alpha=0.35, linewidth=1)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Selection rank")
    axis.set_ylim(order.size + 0.7, 0.3)
    axis.grid(alpha=0.2)


def _plot_feature_scores(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot per-feature scores for univariate, VIP, and SR selectors."""
    figure, axes_array = _make_axes(axes, 1, figsize)
    axis = axes_array[0]
    x, x_label = _feature_axis(selector)
    scores = _finite_for_plot(selector.scores_)
    axis.plot(x, scores, color="#2962A3", linewidth=1.5, label="Feature score")
    axis.scatter(
        x[selector.support_],
        scores[selector.support_],
        color="#D1495B",
        s=28,
        zorder=3,
        label="Selected",
    )
    if hasattr(selector, "threshold_") and np.isfinite(selector.threshold_):
        axis.axhline(
            selector.threshold_,
            color="#3A7D44",
            linestyle="--",
            linewidth=1.2,
            label="Threshold",
        )
    elif (
        hasattr(selector, "threshold")
        and selector.n_features_to_select is None
        and isinstance(selector.threshold, (int, float, np.integer, np.floating))
    ):
        axis.axhline(
            float(selector.threshold),
            color="#3A7D44",
            linestyle="--",
            linewidth=1.2,
            label="Threshold",
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel("Selection score")
    axis.set_title(f"{type(selector).__name__} feature importance")
    axis.grid(alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _plot_support(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot the selected-feature mask for selectors without score diagnostics."""
    figure, axes_array = _make_axes(axes, 1, figsize)
    axis = axes_array[0]
    x, x_label = _feature_axis(selector)
    axis.step(x, selector.support_.astype(int), where="mid", color="#2962A3")
    axis.scatter(
        x[selector.support_], np.ones(selector.support_.sum()), color="#D1495B"
    )
    axis.set(xlabel=x_label, ylabel="Selected", yticks=[0, 1])
    axis.set_title(f"{type(selector).__name__} selected variables")
    axis.grid(alpha=0.2)
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _plot_covsel(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot CovSel rank and cumulative deflation diagnostics."""
    figure, axes_array = _make_axes(axes, 2, figsize)
    _plot_selected_rank(axes_array[0], selector)
    axes_array[0].set_title("CovSel variable order")
    steps = np.arange(1, np.size(selector.x_explained_variance_) + 1)
    axes_array[1].plot(steps, selector.x_explained_variance_, marker="o", label="X")
    axes_array[1].plot(steps, selector.y_explained_variance_, marker="s", label="Y")
    axes_array[1].set(
        xlabel="Number of selected variables",
        ylabel="Cumulative explained fraction",
        title="CovSel orthogonal deflation",
        ylim=(0, 1.05),
    )
    axes_array[1].grid(alpha=0.2)
    axes_array[1].legend(loc="best")
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _plot_fcovsel(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot length-validation scores and the final full-data selection."""
    figure, axes_array = _make_axes(axes, 2, figsize)
    counts = np.arange(1, selector.validation_scores_.size + 1)
    axes_array[0].plot(
        counts,
        selector.validation_scores_,
        marker="o",
        color="#2962A3",
    )
    best = (
        int(
            getattr(selector, "best_n_features_", 0)
            or np.argmax(selector.validation_scores_) + 1
        )
        - 1
    )
    name = {
        "FCovSelSelector": "fCovSel",
        "CovSelSelector": "CovSel",
        "MIVIFSelector": "MI-VIF",
        "SPASelector": "SPA",
    }.get(type(selector).__name__, type(selector).__name__)
    axes_array[0].scatter(
        counts[best], selector.validation_scores_[best], color="#D1495B", zorder=3
    )
    axes_array[0].set(
        xlabel="Sequence length",
        ylabel="Mean CV score",
        title=f"{name} length validation ({selector.scoring_})",
    )
    axes_array[0].grid(alpha=0.2)
    _plot_selected_rank(axes_array[1], selector)
    axes_array[1].set_title(f"{name} full-data selection")
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _plot_cars(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot repeated-run frequency and mean inner-search convergence."""
    figure, axes_array = _make_axes(axes, 2, figsize)
    x, x_label = _feature_axis(selector)
    axes_array[0].bar(
        x,
        selector.selection_frequency_,
        color="#2962A3",
        width=_bar_width(x),
    )
    axes_array[0].scatter(
        x[selector.support_],
        selector.selection_frequency_[selector.support_],
        color="#D1495B",
        zorder=3,
    )
    axes_array[0].set(
        xlabel=x_label,
        ylabel="Selection frequency",
        title="CARS repeated-run stability",
    )
    axes_array[1].plot(
        selector.score_history_,
        color="#3A7D44",
        label="Mean candidate score",
    )
    axes_array[1].set(
        xlabel="Reduction iteration",
        ylabel="Internal CV score",
        title="CARS convergence",
    )
    for axis in axes_array:
        axis.grid(alpha=0.2)
    axes_array[1].legend(loc="best")
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _plot_boss(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot BOSS adaptive weights and score/subset histories."""
    figure, axes_array = _make_axes(axes, 2, figsize)
    x, x_label = _feature_axis(selector)
    axes_array[0].plot(x, selector.variable_weights_, color="#2962A3")
    axes_array[0].scatter(
        x[selector.support_],
        selector.variable_weights_[selector.support_],
        color="#D1495B",
        zorder=3,
    )
    axes_array[0].set(
        xlabel=x_label,
        ylabel="Adaptive weight",
        title="BOSS variable weights",
    )
    steps = np.arange(1, selector.score_history_.size + 1)
    axes_array[1].plot(steps, selector.score_history_, marker="o", color="#3A7D44")
    axes_array[1].set(
        xlabel="Iteration",
        ylabel="Internal CV score",
        title="BOSS convergence",
    )
    retained_axis = axes_array[1].twinx()
    retained_axis.plot(
        steps,
        selector.retained_history_,
        color="#D1495B",
        linestyle="--",
        alpha=0.7,
    )
    retained_axis.set_ylabel("Bootstrap draws")
    for axis in axes_array:
        axis.grid(alpha=0.2)
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _plot_genetic(
    selector: Any, axes: Any, figsize: tuple[float, float] | None
) -> tuple[Any, NDArray[Any]]:
    """Plot GA support and best-fitness convergence."""
    figure, axes_array = _make_axes(axes, 2, figsize)
    x, x_label = _feature_axis(selector)
    axes_array[0].fill_between(
        x,
        selector.support_.astype(int),
        step="mid",
        color="#D1495B",
        alpha=0.55,
    )
    axes_array[0].set(
        xlabel=x_label,
        ylabel="Selected",
        yticks=[0, 1],
        title="Genetic-algorithm support mask",
    )
    axes_array[1].plot(selector.best_score_history_, color="#2962A3", marker="o")
    axes_array[1].set(
        xlabel="Generation",
        ylabel="Best internal CV score",
        title="Genetic-algorithm convergence",
    )
    for axis in axes_array:
        axis.grid(alpha=0.2)
    figure.tight_layout(h_pad=1.5)
    return figure, axes_array


def _bar_width(x: NDArray[np.float64]) -> float:
    """Return a readable bar width for index or wavelength coordinates."""
    if x.size < 2:
        return 0.8
    positive_steps = np.diff(np.sort(np.unique(x)))
    return 0.8 * float(np.min(positive_steps)) if positive_steps.size else 0.8
