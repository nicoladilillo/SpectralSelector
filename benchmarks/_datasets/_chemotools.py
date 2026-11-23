"""Adapters for datasets owned and distributed by chemotools."""

from __future__ import annotations

import numpy as np
from sklearn.utils import Bunch

from ._base import _dataset


def _load_from_chemotools(loader_name, *, return_X_y, as_frame, task):
    """Delegate storage to chemotools and normalize its result metadata."""
    try:
        from chemotools import datasets as chemotools_datasets
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "This loader requires chemotools; install chemotools."
        ) from error
    X_frame, y_frame = getattr(chemotools_datasets, loader_name)()
    X = X_frame.to_numpy()
    y = y_frame.to_numpy()
    if y.shape[1] == 1:
        y = y[:, 0]
    try:
        wavelengths = np.asarray(X_frame.columns, dtype=np.float64)
    except (TypeError, ValueError):
        wavelengths = np.arange(X.shape[1], dtype=np.float64)
    target_names = np.asarray(y_frame.columns, dtype=str)
    names = {
        "load_coffee": "coffee",
        "load_fermentation_train": "fermentation_train",
        "load_fermentation_test": "fermentation_test",
    }
    return _dataset(
        data=X,
        target=y,
        wavelengths=wavelengths,
        name=names[loader_name],
        task=task,
        target_names=target_names,
        description="Dataset loaded from the installed chemotools package.",
        source="https://github.com/paucablop/chemotools",
        citation="See the chemotools dataset documentation for source references.",
        license_name="Distributed by chemotools (MIT package licence).",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def load_coffee(*, return_X_y=False, as_frame=False):
    """Load the chemotools ATR-FTIR coffee-origin classification dataset."""
    return _load_from_chemotools(
        "load_coffee",
        return_X_y=return_X_y,
        as_frame=as_frame,
        task="classification",
    )


def load_fermentation_train(*, return_X_y=False, as_frame=False):
    """Load the chemotools synthetic fermentation regression training set."""
    return _load_from_chemotools(
        "load_fermentation_train",
        return_X_y=return_X_y,
        as_frame=as_frame,
        task="regression",
    )


def load_fermentation_test(*, return_X_y=False, as_frame=False):
    """Load the chemotools real fermentation spectra and HPLC references.

    Chemotools provides 1,629 spectra but only 34 HPLC measurements. Consequently,
    this source is a time-alignment dataset rather than a directly row-aligned
    supervised matrix. The two original arrays are returned without interpolation.
    """
    try:
        from chemotools.datasets import load_fermentation_test as upstream_loader
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "This loader requires chemotools; install chemotools."
        ) from error
    X_frame, y_frame = upstream_loader()
    if as_frame:
        X, y = X_frame, y_frame
    else:
        X, y = X_frame.to_numpy(), y_frame.to_numpy()
    if return_X_y:
        return X, y
    return Bunch(
        data=X,
        target=y,
        wavelengths=np.asarray(X_frame.columns, dtype=np.float64),
        feature_names=np.asarray([str(value) for value in X_frame.columns]),
        target_names=np.asarray(y_frame.columns, dtype=str),
        sample_metadata=None,
        frame=None,
        name="fermentation_test",
        task="time_aligned_regression",
        DESCR=(
            "Real fermentation spectra sampled every 1.25 minutes and separate "
            "hourly HPLC references. Rows are not aligned one-to-one."
        ),
        source="https://github.com/paucablop/chemotools",
        citation="See the chemotools fermentation dataset documentation.",
        license="Distributed by chemotools (MIT package licence).",
    )
