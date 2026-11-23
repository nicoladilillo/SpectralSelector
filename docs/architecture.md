# Architecture and file guide

## Runtime package

| File | Responsibility |
|---|---|
| `src/spectralselector/__init__.py` | Stable public imports and aliases |
| `src/spectralselector/_version.py` | Installed distribution version exposed as `__version__` |
| `src/spectralselector/_base.py` | Common selector validation, support masks, indices, wavelengths, and transformation |
| `src/spectralselector/_modeling.py` | Task inference, CV construction, scoring, estimator unwrapping, and importance extraction |
| `src/spectralselector/_wrapper.py` | Shared configuration used by CV-driven wrapper selectors |
| `src/spectralselector/_length_search.py` | Fold-local, model-validated prefix length shared by CovSel, fCovSel, MI-VIF, and SPA |
| `src/spectralselector/plsda.py` | Built-in PLS projection plus LDA classifier |
| `src/spectralselector/fixed.py` | Selection of explicitly supplied indices |
| `src/spectralselector/univariate.py` | F-test and mutual-information ranking |
| `src/spectralselector/covsel.py` | Sequential covariance selection with X/Y deflation |
| `src/spectralselector/fcovsel.py` | Direct fCovSel sequence with configurable CV diagnostics |
| `src/spectralselector/mivif.py` | Mutual-information ranking with variance-inflation-factor screening (MI-VIF) |
| `src/spectralselector/spa.py` | Successive projections algorithm with validated start and length |
| `src/spectralselector/model_based.py` | Leakage-safe VIP and selectivity-ratio (mean or F-test threshold) selection with cloned PLS/PLS-DA estimators |
| `src/spectralselector/plotting.py` | Optional algorithm-specific Matplotlib diagnostics |
| `src/spectralselector/cars.py` | Competitive adaptive reweighted sampling |
| `src/spectralselector/boss.py` | Bootstrapping soft shrinkage |
| `src/spectralselector/genetic.py` | Genetic optimization of individual-variable or interval masks |

Files beginning with `_` are internal. Users should import selectors from
`spectralselector`, not from those modules directly.

## Project files

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Build system, package metadata, dependencies, and pytest settings |
| `environment.yml` | Conda/Python bootstrap for the development environment |
| `requirements-dev.lock` | Exact Python dependency versions for reproducible development |
| `.pre-commit-config.yaml` | Local lint, format, and pre-push test hooks |
| `README.md` | GitHub/PyPI landing page and user quick start |
| `LICENSE` | MIT licence for `spectralselector` code |
| `.readthedocs.yaml` | Hosted Sphinx build environment and strict warning policy |
| `docs/conf.py` | Sphinx extensions, theme, version, and intersphinx settings |
| `docs/index.md` | Documentation home page and navigation tree |
| `docs/api.rst` | Autosummary definition for the public selector API |
| `docs/*.md` | Hand-written user, algorithm, and development guides |
| `examples/` | Small executable regression, classification, and chemotools workflows |
| `benchmarks/` | Benchmark programs, the shared PLS-DA factory (`_models.py`), and small versioned technical reports; temporary runs are ignored |
| `assets/branding/` | Editable SpectralSelector SVGs and generated PNG assets |
| `tests/` | Unit, estimator-contract, pipeline, and interoperability tests |
| `.gitignore` | Excludes generated caches, builds, coverage data, and documentation output |

## Fitted selector contract

Every selector learns a one-dimensional Boolean `support_` mask. The common base
class derives `selected_indices_` in source-column order and applies the mask in
`transform`. Algorithms may additionally store `selection_order_`, scores,
convergence histories, fitted internal models, and wavelength metadata.
Successful fitting also records selector-only wall-clock duration in
`selection_time_`.

Training data are accepted only by `fit(X, y)`. Constructors contain configuration
only, which allows scikit-learn to clone selectors inside pipelines and refit them
within each cross-validation training fold.

## Adding an algorithm

Inherit `BaseWavelengthSelector`, validate data through `_validate_X`, calculate a
mask, save method-specific diagnostics, and call `_finalize_selection`. A method
that produces a model-free variable order can also inherit
`PrefixLengthSearchMixin` and implement `_fold_order(X, y, n_steps)`; it then
gains the same model-validated `n_features_to_select="auto"` behavior as CovSel,
fCovSel, and MI-VIF. A
model-evaluated method should also reuse the task/CV/scoring helpers so that
regression, classification, LOOCV, and score direction remain consistent. CARS
and BOSS inherit `WrapperSelectorMixin`, but intentionally use only its
coefficient normalization and progress logging. Their apparently similar task,
CV, estimator, and subset-scoring methods encode different public contracts:
CARS always constructs its own PLSR/PLS-DA model, while BOSS may clone a supplied
PLSR/PLS-DA configuration, and both couple component optimization to their own
candidate search. The genetic selector instead uses the mixin's generic wrapper
preparation and scoring path. Moving the CARS/BOSS methods into the generic path
without equivalence fixtures would risk changing controlled behavior, so that
deduplication is postponed with the broader scientific-validation work.
