# JOSS readiness checklist

This file separates work that can be completed locally from evidence that must
come from a public, actively developed project.

## Implemented locally

- [x] OSI-approved software licence.
- [x] Installable Python package with declared dependencies.
- [x] Automated tests, coverage threshold, linting, strict documentation build,
  and package validation in continuous integration.
- [x] Contribution, conduct, security, citation, changelog, and data-licensing
  documents.
- [x] User documentation, API reference, examples, and development guide.
- [x] Draft JOSS paper with the required major sections and AI-use disclosure.
- [x] Reproducible nested-CV benchmark entry point.

## Required before public release

- [ ] Confirm the final GitHub repository and documentation URLs.
- [ ] Validate every named scientific method numerically against publications,
  author code, or independently reproduced reference calculations.
- [x] Avoid bundling third-party research datasets and verify remote downloads
  with fixed checksums.
- [ ] Run the complete supported-Python CI matrix on the public repository.
- [ ] Configure Read the Docs and PyPI trusted publishing.
- [ ] Publish a tagged release and archive it on Zenodo with a DOI.
- [ ] Replace the provisional citation metadata with the release DOI.

## Required before JOSS submission

- [ ] Maintain at least six months of genuine public development history with
  work distributed over time; JOSS checks this at pre-review.
- [ ] Use public issues and pull requests to document design decisions and
  community feedback.
- [ ] Produce a versioned technical benchmark report with regression and
  classification, repeated nested CV, full-spectrum and random-subset baselines,
  stability, runtime, and environment metadata. The internal 0.8.0 report is synthetic
  engineering evidence; the complete publication matrix is implemented but has
  not yet been executed and archived.
- [ ] Extend technical benchmarks into independent scientific validation against
  authoritative implementations and public spectroscopy datasets.
- [ ] Demonstrate research impact or independent use beyond package-development
  tests and update the paper's research-impact statement.
- [ ] Ask at least one domain expert to reproduce the installation and one
  tutorial from a clean environment.
- [ ] Run the official JOSS paper checker and resolve all manuscript warnings.
  The source passes a local Pandoc citation build. The documented Inara command
  still requires a running Docker-compatible daemon and remains mandatory.
- [ ] Ensure the paper describes released behavior and contains no new research
  result that belongs in a conventional research article.

The public-development-history requirement cannot be created retrospectively by
uploading a mature local repository. Publish early and continue development in
the open before submitting.
