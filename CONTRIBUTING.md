# Contributing to SpectralSelector

Contributions to algorithms, validation, documentation, and examples are
welcome. Please use a public issue before starting a substantial change so the
scientific assumptions and API can be discussed early.

## Development setup

```bash
git clone https://github.com/nicoladilillo/spectralselector.git
cd spectralselector
conda env create -f environment.yml
conda activate spectralselector
```

Run the local checks with:

```bash
python -m ruff check src tests examples benchmarks
python -m pytest --cov=spectralselector --cov-report=term-missing
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
```

## Scientific contributions

Read the [selector implementation guide](docs/adding_selectors.md) before adding
a method. It defines the module-level reference presentation, selector-family
classification, fitted API, validation fixtures, and optional neural-method
requirements.

New selectors should include:

- a source or methodological reference when the method is published;
- a scikit-learn-compatible implementation derived from
  `BaseWavelengthSelector`;
- deterministic behavior for a fixed `random_state`, when stochastic;
- regression and/or classification tests matching the claimed task support;
- tests showing that selection occurs during `fit` and transforms unseen data;
- documentation of scores, masks, stopping rules, and computational cost;
- a small example and, where appropriate, a diagnostic plot.

Avoid copying implementations from incompatibly licensed projects. State when
an implementation is an independent interpretation of a publication.

## Pull requests

Keep changes focused and add an entry under `Unreleased` in `CHANGELOG.md`.
Pull requests must pass tests, linting, documentation, and package-build checks.
Reviewers may request a numerical comparison against a reference implementation
or a synthetic dataset with known informative wavelengths.

## Reporting problems

Use the issue templates for reproducible bugs and feature requests. Report
security-sensitive problems according to `SECURITY.md` instead of opening a
public issue.

By participating, contributors agree to follow `CODE_OF_CONDUCT.md`.
