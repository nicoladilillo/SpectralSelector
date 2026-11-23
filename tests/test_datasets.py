import hashlib
import zipfile

import numpy as np
import pytest

from benchmarks._datasets import (
    fetch_corn,
    fetch_diesel,
    fetch_shootout,
    fetch_wang2025,
    get_data_home,
    load_coffee,
    load_fermentation_test,
    load_fermentation_train,
    load_lettuce_water_stress,
)
from benchmarks._datasets._base import _download


def test_chemotools_dataset_adapters():
    pytest.importorskip("chemotools")

    coffee = load_coffee()
    train = load_fermentation_train()
    test = load_fermentation_test()

    assert coffee.data.shape == (60, 1841)
    assert coffee.target.shape == (60,)
    assert train.data.shape == (21, 1047)
    assert train.target.shape == (21,)
    assert test.data.shape == (1629, 1047)
    assert test.target.shape == (34, 6)
    assert test.task == "time_aligned_regression"
    X_frame, y_frame = load_fermentation_test(as_frame=True, return_X_y=True)
    assert X_frame.shape == (1629, 1047)
    assert y_frame.shape == (34, 6)


def test_local_lettuce_loader_reads_user_supplied_csv(tmp_path):
    source = tmp_path / "lettuce.csv"
    source.write_text(
        "Class,Cycle,350,351,352\n"
        "Control (C),1,1.0,2.0,3.0\n"
        "Water Stress (W),1,4.0,5.0,6.0\n",
        encoding="utf-8",
    )

    dataset = load_lettuce_water_stress(source, wavelength_range=(351, 352))

    assert dataset.data.shape == (2, 2)
    assert dataset.wavelengths.tolist() == [351.0, 352.0]
    assert dataset.target.tolist() == ["Controlled", "Water Stressed"]
    assert dataset.sample_metadata["source_class"].tolist() == [
        "Control (C)",
        "Water Stress (W)",
    ]
    assert "not redistributed" in dataset.license


def test_local_lettuce_loader_rejects_invalid_schema(tmp_path):
    source = tmp_path / "lettuce.csv"
    source.write_text("Label,Cycle,350\nControl (C),1,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Class"):
        load_lettuce_water_stress(source)


def test_get_data_home_honors_environment(monkeypatch, tmp_path):
    expected = tmp_path / "spectra"
    monkeypatch.setenv("SPECTRALSELECTOR_DATA", str(expected))

    assert get_data_home() == expected
    assert expected.is_dir()


def test_download_verifies_checksum_and_reuses_cache(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"spectra")
    expected = hashlib.sha256(b"spectra").hexdigest()
    destination = tmp_path / "cache" / "data.txt"

    result = _download(source.as_uri(), destination, checksum=expected)
    source.write_bytes(b"changed")
    cached = _download(source.as_uri(), destination, checksum=expected)

    assert result == cached
    assert cached.read_bytes() == b"spectra"


def test_download_rejects_wrong_checksum(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"spectra")

    with pytest.raises(OSError, match="Checksum"):
        _download(source.as_uri(), tmp_path / "out.txt", checksum="0" * 64)


def test_wang_dispatch_rejects_unverified_source():
    with pytest.raises(ValueError, match="corn"):
        fetch_wang2025("tobacco")


@pytest.mark.remote_data
def test_remote_wang_datasets(tmp_path):
    corn = fetch_corn(data_home=tmp_path)
    diesel = fetch_diesel(data_home=tmp_path)
    shootout = fetch_shootout(data_home=tmp_path)

    assert corn.data.shape == (80, 700)
    assert diesel.data.shape == (246, 401)
    assert shootout.data.shape == (655, 650)
    assert corn.target.shape == (80,)
    assert diesel.target.shape == (246,)
    assert shootout.target.shape == (655,)
    assert diesel.target.min() == 197
    assert diesel.target.max() == 293


def test_remote_argument_validation_occurs_before_download(tmp_path):
    with pytest.raises(ValueError, match="target"):
        fetch_corn(target="invalid", data_home=tmp_path)
    with pytest.raises(ValueError, match="instrument"):
        fetch_shootout(instrument=3, data_home=tmp_path)


_LETTUCE_HEADER = "Class,Cycle,350,351,352\n"
_LETTUCE_ROW = "Control (C),1,1.0,2.0,3.0\n"


@pytest.mark.parametrize(
    ("content", "kwargs", "match"),
    [
        ("", {}, "empty"),
        ("Class,Cycle,a,b\n", {}, "numeric"),
        ("Class,Cycle,350,350\n", {}, "finite and unique"),
        (_LETTUCE_HEADER + "Control (C),1,1.0\n", {}, "has 3 columns"),
        (_LETTUCE_HEADER + "Control (C),x,1,2,3\n", {}, "non-numeric"),
        (_LETTUCE_HEADER, {}, "no observations"),
        (_LETTUCE_HEADER + "Drought,1,1,2,3\n", {}, "Unknown lettuce class"),
        (_LETTUCE_HEADER + _LETTUCE_ROW, {"wavelength_range": 351}, "pair"),
        (
            _LETTUCE_HEADER + _LETTUCE_ROW,
            {"wavelength_range": (352, 351)},
            "increasing",
        ),
        (_LETTUCE_HEADER + _LETTUCE_ROW, {"wavelength_range": (1, 2)}, "overlap"),
    ],
)
def test_local_lettuce_loader_rejects_malformed_files(tmp_path, content, kwargs, match):
    source = tmp_path / "lettuce.csv"
    source.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_lettuce_water_stress(source, **kwargs)


def test_local_lettuce_loader_missing_file_and_frame_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_lettuce_water_stress(tmp_path / "missing.csv")

    source = tmp_path / "lettuce.csv"
    source.write_text(_LETTUCE_HEADER + _LETTUCE_ROW, encoding="utf-8")
    frame = load_lettuce_water_stress(source, as_frame=True)
    X, y = load_lettuce_water_stress(source, return_X_y=True)

    assert frame.data.columns.tolist() == [350.0, 351.0, 352.0]
    assert frame.frame.shape == (1, 4)
    assert X.shape == (1, 3)
    assert y.tolist() == ["Controlled"]


def _diesel_csv_zip(path, property_ids=("s1", "s2", "s3")):
    """Write a miniature archive with Eigenvector's annotated CSV layout."""
    preamble = [",,\n"] * 8
    spectra = [*preamble, ",,\n", ",,750,752,\n"] + [
        f",s{i},{i}.0,{i}.5,\n" for i in (1, 2, 3)
    ]
    properties = [*preamble, ",,BP50,CN,\n"] + [
        f",{sid},{value},40,\n"
        for sid, value in zip(property_ids, ("200", "NaN", "220"), strict=True)
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("diesel_spec.csv", "".join(spectra))
        archive.writestr("diesel_prop.csv", "".join(properties))
    return path


def test_diesel_full_table_parser_drops_missing_targets(tmp_path, monkeypatch):
    archive = _diesel_csv_zip(tmp_path / "diesel.zip")
    monkeypatch.setattr(
        "benchmarks._datasets._remote._download", lambda *a, **k: archive
    )

    dataset = fetch_diesel(subset="all", target="bp50", data_home=tmp_path)

    np.testing.assert_array_equal(dataset.data, [[1.0, 1.5], [3.0, 3.5]])
    np.testing.assert_array_equal(dataset.target, [200.0, 220.0])
    assert dataset.wavelengths.tolist() == [750.0, 752.0]
    assert dataset.sample_metadata["sample_id"].tolist() == ["s1", "s3"]
    dispatched = fetch_wang2025("diesel", subset="all", data_home=tmp_path)
    np.testing.assert_array_equal(dispatched.target, dataset.target)
    with pytest.raises(ValueError, match="target must be one of"):
        fetch_diesel(subset="all", target="density", data_home=tmp_path)


def test_diesel_full_table_parser_rejects_misaligned_ids(tmp_path, monkeypatch):
    archive = _diesel_csv_zip(tmp_path / "diesel.zip", property_ids=("s1", "s3", "s2"))
    monkeypatch.setattr(
        "benchmarks._datasets._remote._download", lambda *a, **k: archive
    )

    with pytest.raises(ValueError, match="misaligned"):
        fetch_diesel(subset="all", data_home=tmp_path)


@pytest.mark.parametrize(
    ("loader", "kwargs", "match"),
    [
        (fetch_corn, {"instrument": "nir"}, "instrument"),
        (fetch_diesel, {"subset": "train"}, "subset"),
        (fetch_diesel, {"target": "CN"}, "only target='BP50'"),
        (fetch_shootout, {"target": "colour"}, "target"),
        (fetch_shootout, {"split": "train"}, "split"),
    ],
)
def test_remote_loaders_reject_invalid_options(tmp_path, loader, kwargs, match):
    with pytest.raises(ValueError, match=match):
        loader(data_home=tmp_path, **kwargs)
