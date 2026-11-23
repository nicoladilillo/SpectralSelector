# Scientific validation

Wavelength selection is a supervised learning step whenever it uses `y`, model
coefficients, or model performance. It must therefore be fitted on training data
only. Selecting variables once before cross-validation leaks information from
validation samples and can make the reported model appear better than it is.

Every public selector and PLSDA is exercised with scikit-learn's estimator
contract checks. Deterministic reduced search budgets keep the CARS, BOSS, and
genetic checks fast without changing their public behavior. Scikit-learn may
skip only its optional Array API checks when `SCIPY_ARRAY_API` is not enabled.
Fixed-seed CARS, BOSS, and genetic complete-run fixtures assert stable support
masks and compare floating-point trajectories with declared tolerances. CI runs
these fixtures on Linux, macOS, and Windows with Python 3.12.

The offline suite enforces at least 90% combined statement-and-branch coverage.
Tests concentrate on scientific calculations, shared task/CV/model validation,
numerical guards, and failure paths; the threshold is not a substitute for the
independent reference comparisons listed below.

## Required evaluation structure

Use nested cross-validation when selector settings are tuned:

1. The outer split reserves data used only for final fold evaluation.
2. An inner search fits preprocessing, selection, and modelling pipelines on
   subdivisions of the outer training fold.
3. The best inner configuration is refitted on the complete outer training fold.
4. That fitted pipeline predicts the untouched outer test fold once.
5. Performance is aggregated over outer folds.

The selector belongs inside the pipeline passed to `GridSearchCV`; a precomputed
support mask derived from the complete dataset is not equivalent.

## Evidence expected for an algorithm

An algorithm is not scientifically validated merely because it returns a mask
and passes API tests. Validation should include:

- synthetic spectra with known informative bands;
- regression and every claimed classification mode;
- comparison with equations, published outputs, or authoritative source code;
- full-spectrum and target-independent random-subset baselines;
- repeated nested cross-validation on public spectral datasets;
- selected-variable recovery and pairwise stability;
- sensitivity to noise, sample size, collinearity, and wavelength resolution;
- runtime, peak memory, and parallel reproducibility;
- fixed seeds and complete package/environment metadata.

The runnable synthetic benchmark in `benchmarks/run_nested_benchmarks.py`
implements the leakage-safe structure and writes fold-level results, summaries,
selected indices, and environment information. It is a starting point for the
public validation record, not a substitute for method-by-method reference
validation.

For the complete dataset matrix, stochastic selectors must be repeated over a
declared fixed seed set. `run_dataset_matrix.py --profile publication` defaults
to five selector seeds, or accepts an explicit comma-separated set through
`--selector-seeds`. Deterministic methods are not redundantly refitted. Report
predictive uncertainty across saved evaluations together with within-fold
selection Jaccard across seeds; do not present only the most favorable seed.

Classification benchmarks use PLS-DA as the downstream model (chemotools'
implementation when installed) and tune its component count; regression uses
ridge regression. CovSel, fCovSel, and MI-VIF are also benchmarked with their
length chosen by that same downstream model, so the number of wavelengths is
validated for the model that uses them. MI-VIF (VIF <= 10) and SR with the
F-test boundary are benchmarked with their published stopping rules and no
size grid.

The publication profile records an explicit budget for every job. Candidate
subset sizes, validation partitions, task scoring, and downstream
regularization candidates are shared across compatible methods; intrinsic CARS,
BOSS, fCovSel, and GA operations retain method-specific declared limits. Wall
time, prediction time, main-process CPU time, and peak Python-traced memory are
saved per fold. The traced-memory value excludes some native allocations and
parallel-worker memory, so resource comparisons should use `--n-jobs 1` and
identical hardware.
