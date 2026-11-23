"""Validate version-linked release metadata without optional dependencies."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


def _cff_scalar(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*[\"']?([^\"'#]+?)[\"']?\s*$", text, re.M)
    if match is None:
        raise ValueError(f"CITATION.cff is missing the top-level {key!r} field.")
    return match.group(1).strip()


def _project_version(text: str) -> str:
    project = re.search(r"^\[project\]\s*$(.*?)(?=^\[|\Z)", text, re.M | re.S)
    if project is None:
        raise ValueError("pyproject.toml is missing its [project] table.")
    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']\s*$', project.group(1), re.M)
    if match is None:
        raise ValueError("pyproject.toml is missing project.version.")
    return match.group(1)


_DEVELOPMENT_VERSION = re.compile(r"\.dev\d+$")


def is_development_version(version: str) -> bool:
    """Return whether ``version`` is a PEP 440 development release (``X.Y.Z.devN``)."""
    return _DEVELOPMENT_VERSION.search(version) is not None


def validate_release_metadata(root: Path, tag: str | None = None) -> dict[str, str]:
    """Validate package, citation, changelog, report, and optional tag versions.

    A development version (for example ``0.1.0.dev0``) is valid between
    releases: it needs a matching ``CITATION.cff`` version and an
    ``## [Unreleased]`` changelog section, but it can never be tagged. A release
    version additionally needs a dated changelog heading, a matching citation
    date, and a versioned benchmark report.
    """
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    version = _project_version(pyproject)
    expected_tag = f"v{version}"
    development = is_development_version(version)
    if tag is not None and development:
        raise ValueError(
            f"Development version {version!r} cannot be released; set the final "
            "version in pyproject.toml and CITATION.cff before tagging."
        )
    if tag is not None and tag != expected_tag:
        raise ValueError(f"Release tag {tag!r} must equal {expected_tag!r}.")

    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_version = _cff_scalar(citation, "version")
    if citation_version != version:
        raise ValueError(
            f"CITATION.cff version {citation_version!r} does not match {version!r}."
        )
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if development:
        if re.search(r"^## \[Unreleased\]\s*$", changelog, re.M) is None:
            raise ValueError(
                "A development version requires an '## [Unreleased]' section in "
                "CHANGELOG.md."
            )
        return {
            "version": version,
            "tag": "",
            "date_released": "",
            "benchmark_report": "",
            "development": "true",
        }

    citation_date = _cff_scalar(citation, "date-released")
    date.fromisoformat(citation_date)

    release_match = re.search(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})$",
        changelog,
        re.M,
    )
    if release_match is None:
        raise ValueError(f"CHANGELOG.md has no dated [{version}] release heading.")
    if release_match.group(1) != citation_date:
        raise ValueError(
            "CITATION.cff date-released does not match the changelog release date."
        )

    report = root / "benchmarks" / "reports" / expected_tag
    if not report.is_dir():
        raise ValueError(f"The versioned benchmark report is missing: {report}")
    return {
        "version": version,
        "tag": expected_tag,
        "date_released": citation_date,
        "benchmark_report": str(report.relative_to(root)),
        "development": "false",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Release tag, for example v0.1.0.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (defaults to the parent of this script directory).",
    )
    args = parser.parse_args()
    metadata = validate_release_metadata(args.root.resolve(), args.tag)
    if metadata["development"] == "true":
        print(
            f"Development metadata is consistent: {metadata['version']} "
            "(not releasable until the .devN suffix is removed)."
        )
        return
    print(
        "Release metadata is consistent: "
        f"{metadata['tag']} ({metadata['date_released']}), "
        f"report={metadata['benchmark_report']}"
    )


if __name__ == "__main__":
    main()
