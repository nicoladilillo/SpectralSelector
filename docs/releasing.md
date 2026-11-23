# Release process

Releases are deliberately split between checks that can be completed locally
and actions that require access to hosted services.

## Prepare locally

1. Choose the version and update it once in `pyproject.toml`. The runtime
   `spectralselector.__version__` is read dynamically from installed package
   metadata.
2. Update `CHANGELOG.md` and `CITATION.cff`, regenerate
   `requirements-dev.lock` if dependencies changed, and generate the matching
   versioned technical benchmark report. Apply the repository/Zenodo boundary
   in `benchmarks/RESULTS_POLICY.md`; never commit a selectively truncated
   scientific result merely to satisfy the file-size limits.
3. Run Ruff, codespell, the non-network tests, strict Sphinx HTML and link
   checks, package build, Twine validation, and clean-environment wheel and
   source-distribution smoke tests.
4. Build the JOSS draft with the official Inara command documented in
   `paper/README.md` and inspect both PDF and JATS output. A Pandoc-only check is
   useful but does not replace Inara.
5. Run `python scripts/check_release_metadata.py --tag vX.Y.Z`. It verifies that
   the tag, package metadata, `CITATION.cff`, dated changelog heading, and
   versioned benchmark-report directory agree.
6. Commit a clean tree and create an annotated tag, for example
   `git tag -a v0.1.0 -m "SpectralSelector 0.1.0"`.

## Publish with hosted services

1. Push the release commit and tag to the final GitHub repository.
2. Create a GitHub release from the tag. The release workflow builds and checks
   the artifacts, installs both the wheel and source distribution in clean
   environments, publishes to PyPI through trusted publishing, generates a
   SHA-256 checksum manifest, and attaches the distributions and checksums to
   the GitHub release.
3. Confirm the Read the Docs build and public package installation.
4. Archive the GitHub release on Zenodo, then add the final software DOI to
   `CITATION.cff` in the next commit.

GitHub, PyPI, Read the Docs, and Zenodo configuration requires the maintainer's
accounts and cannot be completed solely by repository changes. Never publish a
dataset in the software release: create its own Zenodo record, choose its
licence, and add a checksum-verified loader only after the record is final.
