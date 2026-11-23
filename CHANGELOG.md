# Changelog

All notable changes to SpectralSelector are documented here. The project follows
[Semantic Versioning](https://semver.org/) and [PEP 440](https://peps.python.org/pep-0440/)
from its first public release.

Version numbering restarts at `0.1.0` for the first public release. Versions
`0.6.0` to `0.8.0` below were internal pre-publication milestones: they were
never uploaded to PyPI, archived on Zenodo, or pushed as public tags, and are
kept only as development history. Between releases the package carries a
`.devN` version, which the release validator refuses to tag.

## [0.1.0] - 2026-11-21

### Added

- `SPASelector`: Successive Projections Algorithm (Araújo et al., 2001,
  doi:10.1016/S0169-7439(01)00119-8). Wavelengths are chosen by successive
  projections onto the orthogonal complement of the last selected column; the
  start wavelength and the length are chosen jointly by leakage-safe
  cross-validated MLR (LDA for classification). Exposes `initial_`,
  `initial_scores_`, and the shared length-validation diagnostics.
- `ICOSelector`: Interval Combination Optimization (Song et al., 2016,
  doi:10.1016/j.aca.2016.10.041). Equal-width intervals are combined by
  weighted bootstrap sampling, weights are updated from the best combinations
  until their mean CV score stops improving, and boundary variables are trimmed
  by local search. Exposes `intervals_`, `interval_weights_`,
  `selected_intervals_`, `mean_score_history_`, and `n_evaluated_subsets_`.
- `MIVIFSelector`: Mutual Information-Variance Inflation Factor selection
  (Cheng et al., 2022, doi:10.1016/j.saa.2021.120652). Variables are ranked by
  k-nearest-neighbor mutual information with the response and accepted by
  forward selection only while every selected variable keeps VIF <= 10. Exposes
  `scores_`, `mi_order_`, `vif_`, and `n_rejected_`; classification targets and
  a `max_features` stopping rule are documented package extensions.
- `SelectivityRatioSelector(threshold="f_test", alpha=0.05)`: the published SR
  significance boundary `F(1 - alpha; N - 2, N - 3)` (Farrés et al., 2015,
  Eq. 8), plus `SelectivityRatioSelector.f_critical_value`. Tests reproduce the
  paper's critical values 2.04, 3.73, and 3.14.
- `VIPSelector` and `SelectivityRatioSelector` accept
  `chemotools.classification.PLSDA` (in addition to `spectralselector.PLSDA`)
  for classification. SR reduces the two collinear binary dummy-coefficient
  rows to one target-projection direction.
- `CovSelSelector("auto", estimator=...)` and
  `MIVIFSelector("auto", estimator=...)`: the number of wavelengths is chosen
  with the model that will use them. The model-free order is recomputed on the
  training rows of every validation fold, every prefix is scored with the
  estimator (optionally jointly with `param_grid`), and the best length is kept.
  Both also gain `scoring`, `cv`, `groups`, `preprocessor`, and `n_jobs`, and
  `plot_diagnostics` shows the validation curve. Fixed counts keep the previous
  model-free behavior.
- A shared prefix-length search (`_length_search.PrefixLengthSearchMixin`) now
  drives fCovSel, CovSel, and MI-VIF; fCovSel results are unchanged.
- Benchmarks: classification now uses PLS-DA (chemotools'
  `chemotools.classification.PLSDA` when installed, otherwise
  `spectralselector.PLSDA`) as the downstream model, tuned over 1-5
  components, and inside VIP, SR, and the genetic selector. The implementation
  used is recorded in every manifest (matrix schema 4, nested schema 3). The
  dataset matrix adds `covsel_auto`, `mivif`, `mivif_auto`, and
  `selectivity_ratio_ftest`; length-validated selectors (including fCovSel) now
  choose their size with the downstream model. The lettuce scripts gain
  `--classifier {plsda,logistic}` and MI-VIF.
- Benchmarks: the dataset matrix adds 27 NIRBENCH-DL v0.1 regression tasks
  (`nirbench_<id>`, read from a local clone with `--nirbench-path`), each using
  NIRBENCH's fixed train/test partition as its single outer split. CovSel and
  fCovSel in the nested benchmark, and every length-validated selector in the
  matrix, now choose their length inside the selector up to 20 wavelengths
  instead of a 6/12/20 grid.
- A method-taxonomy guide separating model-dependent from model-independent
  selectors and describing the main families of wavelength selection methods.
- A classification quick start using chemotools' PLS-DA when it is available.
- `FCovSelSelector(param_grid=...)`: the sequence length is selected jointly
  with the hyper-parameters of `estimator` (which may be a `Pipeline`) on
  fold-local orders, so the number of variables can be chosen for the model
  that will use them. Adds `cv_results_`, `best_params_`, `best_n_features_` and
  `validation_params_`; parameters ending in `n_components` are capped at the
  prefix length.
- `FCovSelSelector(preprocessor=...)`: a transformer re-fitted on the training
  rows of every validation fold (and on all rows for the final order), applied
  again by `transform`, so data-dependent preprocessing never sees validation
  rows.
- `FCovSelSelector(n_jobs=...)`: parallel scoring of folds and grid points.

- Structured scientific module introductions for every selector, separating
  primary references, the implemented algorithm, and package-specific scope.
- Equation-level regression fixtures for VIP, Selectivity Ratio, and CovSel,
  plus optional global response autoscaling for multi-response CovSel.
- A resumable dataset-by-method benchmark matrix with content-addressed result
  caching, immutable attempts, failure records, and cached-result aggregation.
- Contributor guidance for implementing and validating new selectors, including
  a leakage-safe roadmap for future Transformer-based methods.
- Deterministic genetic-operator tests covering interval boundaries, decoding,
  repair, crossover, mutation, tournament tie-breaking, cache reuse, and
  fixed-seed complete runs.
- Version-checked benchmark-attempt schemas and expanded matrix provenance,
  warnings, selected wavelengths, component diagnostics, confidence intervals,
  pairwise support stability, within-dataset predictive ranks, and paired
  full-spectrum comparisons.
- Task-aware fCovSel validation with configurable scikit-learn scoring, integer
  folds or custom splitters, `cv=-1` leave-one-out behavior, multi-response
  regression, and binary or multiclass PLSDA classification.
- Full scikit-learn estimator-contract coverage for every public selector and
  PLSDA.
- Fixed repeated-seed execution for stochastic methods in the publication
  matrix, with seed-aware caching, provenance, fold records, and stability
  summaries.
- Cross-platform fixed-seed support-mask and tolerant trajectory checks for
  CARS, BOSS, and the genetic selector on Linux, macOS, and Windows CI.
- Clean installation checks for both wheel and source-distribution artifacts in
  CI and the trusted-publishing release workflow.
- Equation-based selector documentation with hand-checkable CovSel/fCovSel and
  VIP examples, explicit Selectivity Ratio target projection, predictive-search
  notation, and consistent computational-cost disclosures.
- Explicit benchmark search-budget manifests, symmetric PLS-component tuning
  for VIP/SR, a common downstream regularization grid, and per-fold CPU,
  prediction-time, and Python-traced-memory measurements.
- Focused branch tests for shared task/CV/coefficient validation and PLSDA,
  raising the enforced coverage floor to 80%.
- Public API contract tests for canonical selectors, compatibility aliases,
  constructor documentation, dataset exports, and metadata-derived versions.
- A benchmark-results policy defining compact Git evidence, complete immutable
  Zenodo archives, size limits, provenance, checksums, and excluded data.
- Reproducible JOSS Inara/Pandoc manuscript-check instructions with generated
  publishing artifacts excluded from version control.
- A stdlib-only release metadata validator and a gated release workflow that
  reruns quality checks, creates SHA-256 checksums, and attaches verified
  distributions to the GitHub release before archival.
- A staged publication policy separating the public `0.x` software release,
  independent scientific validation, the dataset record, and later JOSS/API
  stability milestones.
- Versioned external numerical fixtures showing VIP and Selectivity Ratio
  agreement with chemotools 0.4.3, plus CovSel diagnostics and fCovSel sequence
  agreement with the R package plsVarSel 0.10.0.
- A versioned direct-PLSR component-search fixture plus bounds and tie-breaking
  tests for the PLS optimization shared by CARS and BOSS.

### Changed

- The dataset loaders are no longer part of the installed package:
  `spectralselector.datasets` moved to `benchmarks/_datasets/` in the source
  repository, where only the benchmark scripts use them. The dataset guide
  moved to `benchmarks/DATASETS.md`.
- Restarted public version numbering at `0.1.0.dev0`; the release validator now
  distinguishes development versions (require an `[Unreleased]` section and can
  never be tagged) from final releases.
- `SelectivityRatioSelector(threshold="mean")` averages only finite ratios, so a
  single zero-residual variable no longer raises the threshold to infinity.
- Classification in `VIPSelector`/`SelectivityRatioSelector` now requires a
  fitted PLS-DA (exposing PLS scores, weights and y-loadings) instead of the
  package `PLSDA` class specifically; non-PLS classifiers are still rejected.
- Shared input validation, model fitting, and score-based selection between
  VIP and SR.
- Document the exact boundary between CARSSelector and the official libPLS
  author code, including component selection, sampling, rounding, deterministic
  mode, repeated-run aggregation, and the archived source checksum.
- Record and checksum the official BOSS 1.0.0.0 MATLAB archive, and distinguish
  its executable reference contract from the package's tested extensions.
- Record the dated, provisional PyPI name-availability check while keeping
  account control and the final pre-upload check as maintainer responsibilities.
- Use macro-averaged F1 consistently in classification benchmarks so binary,
  multiclass, numeric, and string labels share one valid metric definition.
- Continue the full benchmark matrix after an individual method/fold failure;
  the complete traceback remains available in the persistent result store.
- Permit multi-response numeric targets in shared regression task validation,
  making PLS multi-response VIP usable, and test VIP across PLS/PLSDA scaling
  and binary, multiclass, single-response, and multi-response modes.
- Make fCovSel's final mask follow the direct paper sequence fitted on all
  selection data; leave-one-out RMSECV now chooses only sequence length, and
  fold consensus is unique and diagnostic-only.
- Require the package's PLSDA implementation for VIP and Selectivity Ratio
  classification instead of accepting arbitrary classifier coefficients.

### Fixed

- Removed benchmark configurations that applied Selectivity Ratio to
  logistic-regression coefficients (`run_nested_benchmarks.py`,
  `run_lettuce_classification.py`); SR rejects them, so those jobs always
  failed. SR classification is benchmarked through PLS-DA.
- Unwrap PLSDA through its actual fitted `pls_` attribute when extracting model
  coefficients from nested estimators.
- Convert scaled PLSDA coefficients back to the caller's feature coordinates
  before Selectivity Ratio target projection.

## Internal milestone 0.8.0 (unpublished, 2026-08-26)

### Changed

- Renamed the distribution, import package, documentation identity, and visual
  assets from SpectraSelect to SpectralSelector.
- Consolidated common CARS/BOSS scoring, coefficient, and logging utilities
  without changing their algorithm-specific searches.
- Aligned examples, plots, tests, and benchmarks with the current CARS/BOSS API
  and internal PLS component optimization.
- Declared the `wavelength_selection_package` Conda development environment and
  made `pyproject.toml` the dependency source of truth.
- Reorganized documentation, branding, examples, tests, and generated benchmark
  outputs for a publishable `src`-layout library.
- Made the runtime version derive from installed distribution metadata.
- Deferred the lettuce-health fetcher until its final Zenodo dataset record is
  available.
- Added exact development dependency locking, local pre-commit hooks, clean-wheel
  CI smoke tests, release documentation, and a versioned synthetic benchmark
  report.

### Added

- Release, citation, governance, and JOSS submission scaffolding.
- Leakage-safe benchmark framework for regression and classification.
- Optional sample-group routing for CARS, BOSS, and GA internal
  cross-validation.
- A 400-2,500 nm lettuce preprocessing benchmark covering SG/SG1 with MSC/SNV,
  PCA-Mahalanobis outlier removal, training/global/batch MSC references, and
  held-out confusion matrices.
- Leakage-aware benchmark utilities for training-reference MSC and
  training-only outlier detection.
- A chemotools Hotelling T² example that filters the complete raw dataset before
  the train/test split for explicitly global reproduction workflows.

### Removed

- Bundled lettuce water-stress supplementary data and its loader because the
  CC BY-NC-ND 4.0 terms did not provide sufficiently clear redistribution and
  adaptation permission for an open-source package.

## Internal milestone 0.7.0 (unpublished, 2026-08-17)

### Added

- Dataset loaders for bundled, remote, and chemotools datasets.
- Diagnostic plots and fitted selection-time measurements.
- CovSel and selectivity-ratio selectors.
- Sphinx documentation and examples for all public selectors.

### Changed

- Public lettuce water-stress classes are named `Controlled` and
  `Water Stressed`.

## Internal milestone 0.6.0 (unpublished)

### Added

- CARS, BOSS, genetic-algorithm, VIP, fCovSel, fixed-index, and univariate
  selectors with a shared scikit-learn-compatible API.

[Unreleased]: https://github.com/nicoladilillo/spectralselector/commits/main
