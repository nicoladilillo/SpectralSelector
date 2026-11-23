from pathlib import Path

import pytest

from scripts.check_release_metadata import (
    is_development_version,
    validate_release_metadata,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def test_current_metadata_is_consistent():
    metadata = validate_release_metadata(REPOSITORY)

    if metadata["development"] == "true":
        with pytest.raises(ValueError, match="cannot be released"):
            validate_release_metadata(REPOSITORY, tag=f"v{metadata['version']}")
    else:
        validate_release_metadata(REPOSITORY, tag=metadata["tag"])
        assert metadata["tag"] == f"v{metadata['version']}"
        assert (REPOSITORY / metadata["benchmark_report"]).is_dir()


def test_release_tag_must_match_package_version():
    with pytest.raises(ValueError, match=r"must equal|cannot be released"):
        validate_release_metadata(REPOSITORY, tag="v9.9.9")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.1.0.dev0", True),
        ("1.2.3.dev12", True),
        ("0.1.0", False),
        ("0.1.0rc1", False),
        ("0.1.0.post1", False),
    ],
)
def test_development_version_detection(version, expected):
    assert is_development_version(version) is expected


def _write_release_tree(root: Path, version: str, *, unreleased: bool) -> None:
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8"
    )
    (root / "CITATION.cff").write_text(
        f"version: {version}\ndate-released: 2026-01-02\n", encoding="utf-8"
    )
    heading = "## [Unreleased]" if unreleased else f"## [{version}] - 2026-01-02"
    (root / "CHANGELOG.md").write_text(f"# Changelog\n\n{heading}\n", "utf-8")
    (root / "benchmarks" / "reports" / f"v{version}").mkdir(parents=True)


def test_final_release_requires_dated_heading(tmp_path):
    _write_release_tree(tmp_path, "0.1.0", unreleased=True)

    with pytest.raises(ValueError, match="no dated"):
        validate_release_metadata(tmp_path)


def test_final_release_with_dated_heading_validates_tag(tmp_path):
    _write_release_tree(tmp_path, "0.1.0", unreleased=False)

    metadata = validate_release_metadata(tmp_path, tag="v0.1.0")

    assert metadata["development"] == "false"
    assert metadata["date_released"] == "2026-01-02"


def test_development_version_requires_unreleased_section(tmp_path):
    _write_release_tree(tmp_path, "0.1.0.dev0", unreleased=False)

    with pytest.raises(ValueError, match="Unreleased"):
        validate_release_metadata(tmp_path)
