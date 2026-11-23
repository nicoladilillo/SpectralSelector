"""Parsers for spectroscopy datasets supplied by the user."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ._base import _dataset


def load_lettuce_water_stress(
    path,
    *,
    wavelength_range=None,
    return_X_y=False,
    as_frame=False,
):
    """Load a local copy of the lettuce water-stress supplementary CSV.

    The CSV must contain ``Class`` and ``Cycle`` as its first two columns,
    followed by numeric wavelength column names. Spectra are returned unchanged.
    Source labels are exposed in ``sample_metadata['source_class']`` and mapped
    to the display labels ``Controlled`` and ``Water Stressed`` in ``target``.

    The dataset is not downloaded or distributed by SpectralSelector. Users are
    responsible for obtaining it from an authorized source and complying with
    its CC BY-NC-ND 4.0 terms.

    Parameters
    ----------
    path : path-like
        Local supplementary CSV supplied by the user.
    wavelength_range : pair of float, optional
        Inclusive lower and upper wavelength bounds. ``None`` retains the full
        source axis.
    return_X_y : bool, default=False
        Return ``(data, target)`` instead of a metadata-rich Bunch.
    as_frame : bool, default=False
        Return pandas objects instead of NumPy arrays.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Lettuce dataset CSV not found: {source_path}")

    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("Lettuce dataset CSV is empty.") from error
        if len(header) < 3 or header[:2] != ["Class", "Cycle"]:
            raise ValueError(
                "Expected 'Class' and 'Cycle' followed by wavelength columns."
            )
        try:
            wavelengths = np.asarray(header[2:], dtype=np.float64)
        except ValueError as error:
            raise ValueError("Wavelength column names must be numeric.") from error
        if not np.isfinite(wavelengths).all() or np.unique(wavelengths).size != len(
            wavelengths
        ):
            raise ValueError("Wavelength columns must be finite and unique.")

        source_labels = []
        cycles = []
        spectra = []
        expected_columns = len(header)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != expected_columns:
                raise ValueError(
                    f"Row {row_number} has {len(row)} columns; "
                    f"expected {expected_columns}."
                )
            source_labels.append(row[0])
            try:
                cycles.append(int(row[1]))
                spectra.append([float(value) for value in row[2:]])
            except ValueError as error:
                raise ValueError(
                    f"Row {row_number} contains a non-numeric cycle or spectrum."
                ) from error

    if not spectra:
        raise ValueError("Lettuce dataset CSV contains no observations.")
    label_mapping = {
        "Control (C)": "Controlled",
        "Water Stress (W)": "Water Stressed",
    }
    unknown = sorted(set(source_labels).difference(label_mapping))
    if unknown:
        raise ValueError(f"Unknown lettuce class labels: {unknown}.")

    if wavelength_range is None:
        keep = np.ones(wavelengths.size, dtype=bool)
    else:
        try:
            lower, upper = wavelength_range
        except (TypeError, ValueError) as error:
            raise ValueError("wavelength_range must be an increasing pair.") from error
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("wavelength_range must be an increasing finite pair.")
        keep = (wavelengths >= lower) & (wavelengths <= upper)
        if not keep.any():
            raise ValueError("wavelength_range does not overlap the dataset axis.")

    target = np.asarray([label_mapping[label] for label in source_labels])
    return _dataset(
        data=np.asarray(spectra, dtype=np.float64)[:, keep],
        target=target,
        wavelengths=wavelengths[keep],
        name="lettuce_water_stress",
        task="classification",
        target_names=["water_stress_class"],
        sample_metadata={
            "cycle": np.asarray(cycles, dtype=np.int64),
            "source_class": np.asarray(source_labels),
        },
        description=(
            "User-supplied raw lettuce reflectance spectra. No preprocessing, "
            "outlier removal, or train/test split is applied."
        ),
        source=str(source_path),
        citation=(
            "Dilillo et al. (2026), Advanced optical-based water stress detection "
            "in lettuce: A novel wavelength selection technique."
        ),
        license_name="CC BY-NC-ND 4.0; user-supplied and not redistributed",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )
