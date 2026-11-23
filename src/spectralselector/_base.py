"""Base interfaces and validation shared by every wavelength selector.

The classes in this module define the stable contract of :mod:`spectralselector`.
Algorithm modules only need to learn a support mask and pass it to
:meth:`BaseWavelengthSelector._finalize_selection`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class BaseWavelengthSelector(SelectorMixin, BaseEstimator, ABC):
    """Base class for selectors that reduce columns of a spectral matrix.

    Concrete selectors learn a Boolean support mask during ``fit``. The default
    output is always a two-dimensional NumPy array, which matches the normal
    output of chemotools transformers. Inheriting scikit-learn's
    :class:`~sklearn.feature_selection.SelectorMixin` also supplies
    ``fit_transform``, ``get_support``, and ``get_feature_names_out``.

    Notes
    -----
    Subclasses must keep only configuration in ``__init__`` and create fitted
    attributes with a trailing underscore in ``fit``. This is required for
    cloning inside scikit-learn pipelines and model-selection tools.

    Attributes
    ----------
    selection_time_ : float
        Wall-clock seconds spent fitting the selector, measured after initial
        method-specific input coercion. For wrappers this includes their internal
        model fitting and cross-validation search.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> BaseWavelengthSelector:
        """Learn a support mask from the supplied data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Spectral or other ordered feature matrix.
        y : array-like of shape (n_samples,), optional
            Target values. Unsupervised selectors may ignore this argument.

        Returns
        -------
        self : BaseWavelengthSelector
            Fitted selector.
        """

    def _validate_X(self, X: ArrayLike, *, reset: bool) -> NDArray[np.float64]:
        """Validate X and record/check the number and names of input features.

        ``reset=True`` is used during ``fit``; ``reset=False`` ensures that a
        later ``transform`` receives the same feature layout.
        """
        if reset:
            # Start after method-specific input coercion so the reported value
            # measures selection work consistently across all implementations.
            self._selection_started_at = perf_counter()
        return validate_data(
            self,
            X,
            reset=reset,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
        )

    def _validate_wavelengths(
        self, wavelengths: ArrayLike | None
    ) -> NDArray[np.float64] | None:
        """Validate optional wavelength metadata against ``n_features_in_``."""
        if wavelengths is None:
            return None
        axis = np.asarray(wavelengths, dtype=np.float64)
        if axis.ndim != 1:
            raise ValueError("wavelengths must be a one-dimensional array.")
        if axis.shape[0] != self.n_features_in_:
            raise ValueError(
                "wavelengths must contain one value per input feature; "
                f"received {axis.shape[0]} values for {self.n_features_in_} features."
            )
        if not np.isfinite(axis).all():
            raise ValueError("wavelengths must contain only finite numeric values.")
        return axis.copy()

    def _finalize_selection(
        self,
        mask: ArrayLike,
        *,
        wavelengths: ArrayLike | None,
        selection_order: ArrayLike | None = None,
    ) -> None:
        """Store common fitted attributes and finish the selection timer.

        Parameters
        ----------
        mask : array-like of bool, shape (n_features,)
            Boolean mask in original input-column order.
        wavelengths : array-like of shape (n_features,), optional
            Physical wavelength associated with every original input column.
        selection_order : array-like of int, optional
            Algorithm ranking or forward-selection order. It may differ from
            ``selected_indices_``, which is always in source-column order.
        """
        support = np.asarray(mask, dtype=bool)
        if support.shape != (self.n_features_in_,):
            raise ValueError("The learned support mask has an invalid shape.")
        if not support.any():
            raise ValueError("The selector did not select any input features.")

        self.support_ = support
        self.selected_indices_ = np.flatnonzero(support)
        self.wavelengths_in_ = self._validate_wavelengths(wavelengths)
        self.selected_wavelengths_ = (
            None
            if self.wavelengths_in_ is None
            else self.wavelengths_in_[self.selected_indices_]
        )

        if selection_order is None:
            self.selection_order_ = self.selected_indices_.copy()
        else:
            order = np.asarray(selection_order, dtype=np.int64)
            if (
                order.ndim != 1
                or np.any(order < 0)
                or np.any(order >= self.n_features_in_)
            ):
                raise ValueError("selection_order contains invalid feature indices.")
            self.selection_order_ = order.copy()

        started_at = getattr(self, "_selection_started_at", None)
        if started_at is None:
            raise RuntimeError(
                "Selection timing was not initialized. Selectors must call "
                "_validate_X(..., reset=True) before _finalize_selection()."
            )
        self.selection_time_ = max(0.0, perf_counter() - started_at)
        del self._selection_started_at

    def _get_support_mask(self) -> NDArray[np.bool_]:
        """Return a defensive copy of the fitted Boolean support mask."""
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return selected columns as a two-dimensional NumPy array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Matrix with the same feature layout used during fitting.

        Returns
        -------
        X_selected : ndarray of shape (n_samples, n_selected_features)
            Selected columns in their original source order.
        """
        check_is_fitted(self, "support_")
        X_array = self._validate_X(X, reset=False)
        return X_array[:, self.support_]

    def plot_diagnostics(
        self,
        *,
        axes: Any = None,
        figsize: tuple[float, float] | None = None,
    ) -> tuple[Any, NDArray[Any]]:
        """Plot the most informative fitted diagnostics for this selector.

        Parameters
        ----------
        axes : matplotlib Axes or array-like of Axes, optional
            Existing axes to draw on. The required number depends on the
            selector. When omitted, a new figure and axes are created.
        figsize : tuple of float, optional
            Figure size used only when new axes are created.

        Returns
        -------
        figure : matplotlib.figure.Figure
            Figure containing the diagnostic plot.
        axes : ndarray of matplotlib.axes.Axes
            One-dimensional array of axes used by the selector.

        Notes
        -----
        Matplotlib is an optional dependency. Install ``spectralselector[plot]``
        to use this method. The import occurs only when plotting is requested.
        """
        check_is_fitted(self, "support_")
        from .plotting import plot_selector_diagnostics

        return plot_selector_diagnostics(self, axes=axes, figsize=figsize)
