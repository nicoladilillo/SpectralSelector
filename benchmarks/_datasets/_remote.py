"""Checksum-aware fetchers for public spectroscopy datasets."""

from __future__ import annotations

import csv
import io
import zipfile

import numpy as np
from scipy.io import loadmat

from ._base import _dataset, _download, get_data_home

_CORN_URL = "https://www.eigenvector.com/data/Corn/corn.mat"
_DIESEL_URL = "https://www.eigenvector.com/data/SWRI/SWRI_Diesel_NIR_CSV.zip"
_DIESEL_WANG_URL = "https://www.eigenvector.com/data/SWRI/bp50gatest.zip"
_SHOOTOUT_URL = "https://www.eigenvector.com/data/tablets/nir_shootout_2002.mat"


def _axis(dataset_object):
    """Extract the column axis from an Eigenvector MATLAB Dataset Object."""
    return np.asarray(dataset_object["axisscale"][1, 0], dtype=np.float64)


def fetch_corn(
    *,
    target="oil",
    instrument="m5",
    data_home=None,
    return_X_y=False,
    as_frame=False,
):
    """Fetch the Eigenvector corn NIR regression dataset.

    ``target`` is one of ``'moisture'``, ``'oil'``, ``'protein'``, or
    ``'starch'``. ``instrument`` is one of ``'m5'``, ``'mp5'``, or ``'mp6'``.
    Wang et al. (2025) used the oil target and 80 spectra by 700 wavelengths.
    """
    targets = {"moisture": 0, "oil": 1, "protein": 2, "starch": 3}
    instruments = {"m5": "m5spec", "mp5": "mp5spec", "mp6": "mp6spec"}
    if target not in targets:
        raise ValueError(f"target must be one of {sorted(targets)}.")
    if instrument not in instruments:
        raise ValueError(f"instrument must be one of {sorted(instruments)}.")
    path = _download(
        _CORN_URL,
        get_data_home(data_home) / "corn" / "corn.mat",
        checksum="e28fd4be274a54ca57b1f2c67ef5a8bf4981f8314bcc73e4c64836fe658c46b5",
    )
    raw = loadmat(path, simplify_cells=True)
    spectrum = raw[instruments[instrument]]
    return _dataset(
        data=spectrum["data"],
        target=raw["propvals"]["data"][:, targets[target]],
        wavelengths=_axis(spectrum),
        name="corn",
        task="regression",
        target_names=[target],
        description="NIR spectra of 80 corn samples measured on three instruments.",
        source="https://www.eigenvector.com/data/Corn/index.html",
        citation="Eigenvector corn data; used by Wang et al. (2025).",
        license_name="Distributed with permission by Eigenvector Research.",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def _read_eigenvector_csv(rows, *, axis_row, data_start):
    """Read a numeric matrix from Eigenvector's annotated CSV representation."""
    axis = np.asarray(rows[axis_row][2:-1], dtype=np.float64)
    identifiers = np.asarray([row[1] for row in rows[data_start:]])
    data = np.asarray(
        [row[2 : 2 + axis.size] for row in rows[data_start:]], dtype=float
    )
    return identifiers, axis, data


def fetch_diesel(
    *,
    target="BP50",
    subset="wang2025",
    data_home=None,
    return_X_y=False,
    as_frame=False,
):
    """Fetch the SWRI diesel NIR regression dataset.

    The default ``subset='wang2025'`` loads the dedicated BP50 benchmark with
    246 samples and 401 wavelengths. Use ``subset='all'`` to load all finite
    rows for any property in the broader SWRI table.
    """
    if subset not in ("wang2025", "all"):
        raise ValueError("subset must be 'wang2025' or 'all'.")
    if subset == "wang2025":
        if target.upper() != "BP50":
            raise ValueError("subset='wang2025' supports only target='BP50'.")
        path = _download(
            _DIESEL_WANG_URL,
            get_data_home(data_home) / "diesel" / "bp50gatest.zip",
            checksum=(
                "811c3a9c8317166f95077e4646f389bdea418b74a0d1a14a5c14ba99ec7dca47"
            ),
        )
        with (
            zipfile.ZipFile(path) as archive,
            archive.open("BP50GATEST.mat") as stream,
        ):
            raw = loadmat(io.BytesIO(stream.read()), simplify_cells=True)
        groups = ("hl", "ll_a", "ll_b")
        X = np.vstack([raw[f"bp50_s1d_{group}"] for group in groups])
        values = np.concatenate([raw[f"bp50_y1_{group}"] for group in groups])
        labels = np.concatenate(
            [np.full(raw[f"bp50_y1_{group}"].size, group) for group in groups]
        )
        return _dataset(
            data=X,
            target=values,
            wavelengths=np.arange(750.0, 1552.0, 2.0),
            name="diesel_bp50",
            task="regression",
            target_names=["BP50"],
            sample_metadata={"source_group": labels},
            description="SWRI BP50 diesel benchmark used by Wang et al. (2025).",
            source="https://www.eigenvector.com/data/SWRI/index.html",
            citation="SWRI/Eigenvector BP50 data; used by Wang et al. (2025).",
            license_name="Distributed by Eigenvector Research with SWRI attribution.",
            return_X_y=return_X_y,
            as_frame=as_frame,
        )

    path = _download(
        _DIESEL_URL,
        get_data_home(data_home) / "diesel" / "SWRI_Diesel_NIR_CSV.zip",
        checksum="c5eb0f7f5aacba47c40e2dbbbc81db944bedf568f1d3e7c6c8e381f610563c7a",
    )
    with zipfile.ZipFile(path) as archive:
        spectra_rows = list(
            csv.reader(io.TextIOWrapper(archive.open("diesel_spec.csv")))
        )
        property_rows = list(
            csv.reader(io.TextIOWrapper(archive.open("diesel_prop.csv")))
        )
    property_names = property_rows[8][2:-1]
    lookup = {name.upper(): index for index, name in enumerate(property_names)}
    if target.upper() not in lookup:
        raise ValueError(f"target must be one of {sorted(lookup)}.")
    identifiers, wavelengths, X = _read_eigenvector_csv(
        spectra_rows, axis_row=9, data_start=10
    )
    property_ids = np.asarray([row[1] for row in property_rows[9:]])
    if not np.array_equal(identifiers, property_ids):
        raise ValueError("Diesel spectra and property identifiers are misaligned.")
    column = lookup[target.upper()]
    values = np.asarray(
        [row[2 + column] for row in property_rows[9:]], dtype=np.float64
    )
    keep = np.isfinite(values)
    return _dataset(
        data=X[keep],
        target=values[keep],
        wavelengths=wavelengths,
        name="diesel",
        task="regression",
        target_names=[target.upper()],
        sample_metadata={"sample_id": identifiers[keep]},
        description="SWRI NIR spectra and laboratory properties of diesel fuels.",
        source="https://www.eigenvector.com/data/SWRI/index.html",
        citation="SWRI/Eigenvector diesel data; used by Wang et al. (2025).",
        license_name="Distributed by Eigenvector Research with SWRI attribution.",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def fetch_shootout(
    *,
    target="assay",
    instrument=1,
    split="all",
    data_home=None,
    return_X_y=False,
    as_frame=False,
):
    """Fetch the 2002 NIR Shootout pharmaceutical-tablet regression data."""
    targets = {"weight": 0, "hardness": 1, "assay": 2}
    if target not in targets:
        raise ValueError(f"target must be one of {sorted(targets)}.")
    if instrument not in (1, 2):
        raise ValueError("instrument must be 1 or 2.")
    valid_splits = ("calibrate", "validate", "test")
    if split != "all" and split not in valid_splits:
        raise ValueError(f"split must be 'all' or one of {valid_splits}.")
    path = _download(
        _SHOOTOUT_URL,
        get_data_home(data_home) / "shootout" / "nir_shootout_2002.mat",
        checksum="129a32ec9e194e568cbd96a200c88aabeb347156081b5dfbb0120372fdd6c22a",
    )
    raw = loadmat(path, simplify_cells=True)
    selected_splits = valid_splits if split == "all" else (split,)
    X = np.vstack([raw[f"{part}_{instrument}"]["data"] for part in selected_splits])
    y = np.concatenate(
        [raw[f"{part}_Y"]["data"][:, targets[target]] for part in selected_splits]
    )
    spectrum = raw[f"{selected_splits[0]}_{instrument}"]
    labels = np.concatenate(
        [np.full(raw[f"{part}_Y"]["data"].shape[0], part) for part in selected_splits]
    )
    return _dataset(
        data=X,
        target=y,
        wavelengths=_axis(spectrum),
        name="shootout_2002",
        task="regression",
        target_names=[target],
        sample_metadata={"split": labels, "instrument": instrument},
        description="IDRC 2002 Shootout NIR spectra of pharmaceutical tablets.",
        source="https://www.eigenvector.com/data/tablets/index.html",
        citation="IDRC 2002 Shootout data; Shootout-3 in Wang et al. (2025).",
        license_name="Public benchmark distributed by Eigenvector Research.",
        return_X_y=return_X_y,
        as_frame=as_frame,
    )


def fetch_wang2025(name, **kwargs):
    """Fetch a verified benchmark used by Wang et al. (2025).

    Available names are ``'corn'``, ``'diesel'``, and ``'shootout'``. The paper's
    beer, wheat, tobacco, soy, and UCPH-tablet sources do not currently expose a
    stable, licence-explicit download suitable for an automatic package fetcher.
    """
    loaders = {"corn": fetch_corn, "diesel": fetch_diesel, "shootout": fetch_shootout}
    if name not in loaders:
        raise ValueError(f"name must be one of {sorted(loaders)}.")
    return loaders[name](**kwargs)
