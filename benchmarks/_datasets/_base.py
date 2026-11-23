"""Common dataset containers, output conversion, and cache management."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from sklearn.utils import Bunch

from spectralselector import __version__


def get_data_home(data_home: str | os.PathLike[str] | None = None) -> Path:
    """Return and create the directory used for downloaded datasets.

    The location is selected in this order: ``data_home``, the
    ``SPECTRALSELECTOR_DATA`` environment variable, then
    ``~/.cache/spectralselector``.
    """
    if data_home is None:
        data_home = os.environ.get(
            "SPECTRALSELECTOR_DATA", Path.home() / ".cache" / "spectralselector"
        )
    path = Path(data_home).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(
    url: str,
    destination: Path,
    *,
    checksum: str | None = None,
    algorithm: str = "sha256",
) -> Path:
    """Download a file atomically and optionally verify its checksum."""
    if destination.exists() and _checksum_matches(destination, checksum, algorithm):
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        url, headers={"User-Agent": f"spectralselector-datasets/{__version__}"}
    )
    try:
        with urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if not _checksum_matches(temporary, checksum, algorithm):
            raise OSError(f"Checksum verification failed for {url}.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _checksum_matches(path: Path, expected: str | None, algorithm: str) -> bool:
    """Return whether ``path`` matches ``expected``; no checksum means valid."""
    if expected is None:
        return True
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected.lower()


def _dataset(
    *,
    data: Any,
    target: Any,
    wavelengths: Any,
    name: str,
    task: str,
    target_names: Any,
    description: str,
    source: str,
    citation: str,
    license_name: str,
    sample_metadata: Any = None,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch | tuple[Any, Any]:
    """Build the public dataset result with optional pandas conversion."""
    X = np.asarray(data)
    y = np.asarray(target)
    wavelength_array = np.asarray(wavelengths, dtype=np.float64)
    if X.ndim != 2 or wavelength_array.shape != (X.shape[1],):
        raise ValueError(
            "Dataset spectra and wavelength axis have incompatible shapes."
        )
    if y.shape[0] != X.shape[0]:
        raise ValueError("Dataset spectra and target have different sample counts.")

    feature_names = np.asarray([f"{value:g}" for value in wavelength_array])
    frame = None
    if as_frame:
        try:
            import pandas as pd
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "as_frame=True requires the pandas runtime dependency."
            ) from error
        X = pd.DataFrame(X, columns=wavelength_array)
        if y.ndim == 1:
            y = pd.Series(y, name=np.asarray(target_names).reshape(-1)[0])
        else:
            y = pd.DataFrame(y, columns=np.asarray(target_names))
        frame = X.copy()
        for column in np.asarray(target_names).reshape(-1):
            frame[column] = y[column] if y.ndim > 1 else y

    if return_X_y:
        return X, y
    return Bunch(
        data=X,
        target=y,
        wavelengths=wavelength_array,
        feature_names=feature_names,
        target_names=np.asarray(target_names),
        sample_metadata=sample_metadata,
        frame=frame,
        name=name,
        task=task,
        DESCR=description,
        source=source,
        citation=citation,
        license=license_name,
    )
