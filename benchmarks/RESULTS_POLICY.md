# Benchmark results and archival policy

This policy separates reviewable repository evidence from large, immutable
scientific archives. Benchmark inputs remain in their authoritative source or
the user's private data location; results never redistribute raw spectra.

## Commit to Git

Each released package version may have one directory under
`benchmarks/reports/<version>/`. Commit only compact files needed to review the
release claims and reproduce its tables:

- a short README stating the benchmark scope and limitations;
- the exact command/configuration and matrix manifest;
- aggregate and fold-level CSV tables;
- selected-index or stability summaries that contain no source observations;
- small final figures in an open format; and
- environment metadata plus checksums/DOIs for externally archived evidence.

Keep an individual committed file below 5 MiB and the complete report directory
below 20 MiB. These are repository-maintenance limits, not scientific sampling
rules. A larger result must be archived rather than truncated selectively.

## Archive on Zenodo

Deposit the complete successful and failed attempt records for a publication
run in a separate immutable Zenodo record. Include the matrix manifest, exact
split hashes, fold results, selections, warnings, failures, dependency
environment, source revision, cache schema version, and a SHA-256 checksum
manifest. Include model checkpoints only when they are necessary to reproduce a
reported result and safe to deserialize under the documented software version.

The record description must identify the SpectralSelector release and Git
commit, commands used, hardware/parallel configuration, dataset citations, and
any unavailable method/dataset pairs. After deposition, add the archive DOI and
file checksum to the compact Git report. Never rewrite an archive associated
with a released claim; create a new Zenodo version for a changed configuration,
dataset, dependency environment, or source revision.

## Never publish in either location

- restricted, confidential, or user-supplied raw spectra;
- source dataset rows embedded in logs or debugging artifacts;
- local absolute paths, credentials, tokens, or private download URLs;
- caches unrelated to the declared benchmark run;
- Python bytecode, temporary files, serialized interactive sessions, or
  untrusted pickle/joblib objects; and
- hand-edited aggregate results that cannot be regenerated from fold artifacts.

Before archiving, inspect the file list, scan text metadata for private paths and
secrets, regenerate aggregates using `aggregate_cached_results.py`, and verify
the checksum manifest from a clean extraction.
