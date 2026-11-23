# SpectralSelector technical benchmarks

This folder contains benchmark programs, protocols, and small versioned result
reports. It is not a data directory and it does not provide training data to the
library. Inputs come from the script's synthetic generator, the benchmark
loaders in `benchmarks/_datasets/` (see `DATASETS.md`), or an explicitly supplied local path whose use is
authorized by the researcher.

The benchmark program evaluates selectors without fitting them on an outer test
fold. It creates synthetic spectral matrices with known informative bands and
runs separate regression and classification experiments.

For every outer fold, `GridSearchCV` receives a pipeline containing autoscaling,
wavelength selection, and the predictive model. Hyperparameters are selected by
inner cross-validation using only the outer training data. The chosen pipeline
is then evaluated once on the untouched outer test fold.

```bash
python benchmarks/run_nested_benchmarks.py --output benchmarks/results
```

The nested synthetic benchmark also accepts `--outer-splits -1` and
`--inner-splits -1` for LOOCV, plus `--regression-scoring` and
`--classification-scoring` for task-specific model selection.

Outputs are:

- `fold_results.csv`: one row per task, selector, and outer fold;
- `summary.csv`: aggregate predictive, recovery, stability, and runtime metrics;
- `environment.json`: package versions and run configuration;
- `selected_indices.json`: fitted masks from every outer training fold.

## Persistent execution and resume

The nested benchmark stores every method/fold job under
`benchmarks/.cache/nested/` by default. A stable SHA-256 run key includes the
dataset values, split, selector/model configuration, package version, and a
fingerprint of the package and benchmark source files. Exact successful jobs are
reused automatically, including after an interrupted process:

```bash
python benchmarks/run_nested_benchmarks.py --resume
```

Use `--rerun-failed` to retry only jobs whose latest attempt failed. Use
`--force` or `--no-resume` to append a new attempt even when a successful result
already exists. Attempts are immutable and written atomically; a forced run
does not overwrite prior scientific evidence.

Aggregate the latest cached attempts without fitting any estimator:

```bash
python benchmarks/aggregate_cached_results.py \
  --cache-dir benchmarks/.cache/nested \
  --output benchmarks/results/cached
```

The aggregation writes successes and failures separately. Cache directories are
local execution state and are ignored by Git; compact versioned reports remain
under `benchmarks/reports/`.

## Complete dataset-by-method matrix

`run_dataset_matrix.py` schedules every task-compatible method across the
available supervised catalogue. The matrix includes full-spectrum and random
baselines, univariate selection, CovSel, fCovSel, MI-VIF, VIP, SR, CARS, BOSS,
and the generic genetic selector:

| Method key | Selector and size rule |
|---|---|
| `full_spectrum`, `random_subset` | All variables; seeded random subset without access to `y` |
| `univariate`, `covsel`, `vip`, `selectivity_ratio` | Size tuned over the shared 6/12/20 grid |
| `covsel_auto`, `fcovsel`, `mivif_auto` | Model-free order; length chosen inside the selector with the downstream model |
| `mivif` | Published MI-VIF rule: every variable passing VIF <= 10, no model |
| `selectivity_ratio_ftest` | Published SR F-test boundary `F(0.95; N-2, N-3)`, PLS components tuned |
| `cars`, `boss`, `genetic` | Stochastic wrapper searches with declared budgets |

Classification uses PLS-DA as the downstream model, tuned over 1-5 components,
and as the internal model of VIP, SR, and the genetic selector. This is
chemotools' `chemotools.classification.PLSDA` when installed (chemotools > 0.5.0),
otherwise `spectralselector.PLSDA`. `benchmarks/_models.py` caps the component
count at the number of selected wavelengths, and every manifest records
`plsda_implementation`. CARS and BOSS build `spectralselector.PLSDA` internally.
Regression uses ridge regression tuned over five penalties.

```bash
python benchmarks/run_dataset_matrix.py \
  --datasets all \
  --methods all \
  --profile publication \
  --lettuce-path /path/to/authorized/lettuce.csv \
  --output benchmarks/results/publication-matrix
```

Both `--outer-splits` and `--inner-splits` accept `-1` for leave-one-out
validation when no sample-group structure is present. For grouped datasets,
group-disjoint splitting takes precedence and `-1` means leave one group out.
Use `--regression-scoring` and `--classification-scoring` to choose any scorer
compatible with the downstream estimator; the same metric is passed to CARS,
BOSS, the genetic wrapper, and fCovSel for their internal candidate evaluation.
`--selector-seeds 42,43,44` requests independent complete fits for stochastic
methods. The smoke profile defaults to one seed; the publication profile
defaults to five consecutive seeds beginning at `--random-state`. Random
subset, CARS, BOSS, and the genetic selector are repeated, while deterministic
methods are fitted once per outer fold. Fold records, cache keys, selections,
and the matrix manifest preserve each seed. The summary reports evaluation,
outer-fold, and selector-seed counts plus within-fold seed Jaccard stability.

The public/adapter catalogue covers synthetic regression and classification,
coffee, fermentation training, corn, diesel, Shootout, and an authorized local
lettuce water-stress file. The fermentation test source is deliberately excluded
because its spectra and HPLC rows are time-aligned rather than one-to-one
supervised observations. Lettuce water-stress is recorded as unavailable when
`--lettuce-path` is omitted. Unsupported pairs, currently including multiclass
SR, are recorded as not applicable instead of being silently discarded.

All 27 regression tasks of [NIRBENCH-DL v0.1](https://github.com/DeepLightLab/NIRBENCH_DL_v0.1)
that ship their data are also available as `nirbench_<task id>` (tasks 1-3,
5-11, 13 and 15-30), read from a local clone passed with `--nirbench-path`.
They range from 88 to 5,046 samples and 100 to 875 wavelengths. Tasks 4, 12
and 14 are omitted because NIRBENCH requires rebuilding them from Eigenvector
sources; task 21 is the trimmed, SNV-transformed copy of task 15.

```bash
git clone https://github.com/DeepLightLab/NIRBENCH_DL_v0.1 /path/to/NIRBENCH_DL_v0.1
python benchmarks/run_dataset_matrix.py \
  --datasets nirbench_1,nirbench_18 \
  --nirbench-path /path/to/NIRBENCH_DL_v0.1
```

NIRBENCH fixes its train/test partition, so these tasks use that partition as
their single outer split instead of `--outer-splits`; inner CV still runs on
the training rows only. Rows with missing values are dropped and no spectral
preprocessing is applied. The data are not redistributed by this package: each
manifest records the task's source and license, and the original source must be
cited (see NIRBENCH's `DATA_LICENSES.md`). Tasks are added by extending
`NIRBENCH_TASKS` in `run_dataset_matrix.py`. Without `--nirbench-path` they are
recorded as unavailable.

The default `smoke` profile validates execution with reduced stochastic search
budgets. The `publication` profile uses the declared full CARS/BOSS/GA budgets
and can be very expensive. Both profiles persist one immutable attempt per
dataset, method, outer fold, configuration, and code fingerprint under
`benchmarks/.cache/matrix/`; rerunning the same command resumes exact completed
jobs. Use `--force` only when a genuinely new attempt is required.

The declared stochastic budgets are:

| Search setting | Smoke | Publication |
|---|---:|---:|
| External selector seeds | 1 | 5 |
| CARS complete runs x iterations | 1 x 3 | 50 x 100 |
| BOSS runs x maximum iterations x submodels | 1 x 2 x 8 | 10 x 100 x 1,000 |
| GA population x generations | 6 x 2 | 30 x 100 |

All methods receive the same inner folds, task metric, and five-value downstream
regularization grid. Methods with an explicit selected-count hyperparameter use
the same feasible subset-size candidates, drawn from 6, 12, and 20 variables;
length-validated methods evaluate every prefix up to 20 wavelengths, clipped by
the selector to what its inner training folds allow.
VIP and SR additionally search one through at most five PLS components for both
regression and classification. CARS and BOSS search one through at most ten
components internally, while CovSel-auto, fCovSel, and MI-VIF-auto validate every
feasible prefix of their forward sequence with the downstream model. Equal raw model-fit counts would not represent these
different algorithms faithfully; instead, every job manifest records the joint
grid size, number of pipeline fits, intrinsic selector limits, and a
conservative internal-CV fit upper bound. Early stopping, invalid component
counts, and GA cache reuse can make realized work smaller.

`fold_results.csv`, `summary.csv`, `selections.json`, and
`matrix_manifest.json` are regenerated from completed jobs. The selections file
records indices, physical wavelengths, and captured warnings. Summary rows add
fold-level 95% Student-t intervals, mean pairwise Jaccard support stability, a
predictive rank within each dataset, and paired primary-metric differences from
the full-spectrum baseline. Fold records also contain total nested-fit and
prediction wall time, main-process CPU time, and peak Python-traced memory.
Python tracing is portable but does not fully measure native-library buffers or
joblib worker processes; use `--n-jobs 1` when comparing memory or CPU fields.
Rankings are predictive summaries for one dataset, not a claim of one
universally best selector.

The manifest records unavailable datasets, scientifically unsupported pairs,
content fingerprints, citations, licences, loader arguments and versions,
wavelength/target/group metadata, preprocessing, dependency versions, source
fingerprints, Git provenance, and cache statistics. Cache reuse is invalidated
when this environment or any scientific configuration changes.

The `full_spectrum` baseline uses every variable. `random_subset` uses a fixed,
seeded subset generated without access to `y`; its size is the smallest feasible
common candidate (normally six variables). Its purpose is to test whether a
selector adds value beyond merely reducing dimensionality.

This synthetic benchmark is a technical validation, not sufficient scientific
evidence by itself. Before a JOSS submission, versioned results should also
cover public NIR datasets, repeated nested CV, stochastic selectors, resolution
and noise sensitivity, and numerical comparisons with authoritative reference
implementations.

Committed reports live under `benchmarks/reports/<package-version>/`. Temporary
experiments belong under the ignored `benchmarks/results/` paths. The report in
`reports/v0.8.0/` comes from an internal pre-publication milestone (public
numbering restarts at 0.1.0) and establishes engineering reproducibility only;
broader scientific equivalence testing remains future work.

See [RESULTS_POLICY.md](RESULTS_POLICY.md) for the exact boundary between compact
reviewable files committed to Git and complete immutable benchmark evidence
archived separately on Zenodo.

## Local lettuce classification

The lettuce water-stress supplementary data are not distributed by
SpectralSelector. With an authorized local copy, run a leave-one-cycle-out nested
classification benchmark using:

```bash
python benchmarks/run_lettuce_classification.py \
    /path/to/1-s2.0-S1574954126003511-mmc1.csv \
    --output benchmarks/lettuce_results
```

The script uses the paper's 400-2,500 nm analysis range. The outer evaluation
holds out one complete experimental cycle. Inner tuning
uses `GroupKFold` on the two remaining cycles, and autoscaling, wavelength
selection, and the classifier are fitted only inside those training folds. The
default classifier is PLS-DA (chemotools' implementation when installed);
`--classifier logistic` reproduces the earlier logistic-regression protocol used
for the results recorded in `LETTUCE_ANALYSIS.md`. Both lettuce scripts include
MI-VIF and write the classifier used to `metadata.json`.
No source observations are copied into benchmark outputs.
When Matplotlib is installed, the script also writes `summary.png` comparing
cycle-disjoint accuracy and the mean number of retained variables. The summary
also reports exact pairwise Jaccard agreement between the wavelength masks
fitted in the three outer folds. `confusion_matrices.json` contains aggregate
held-out predictions as counts, without copying any source spectra.

## Paper preprocessing and outlier sensitivity

The second analysis applies all four preprocessing branches described in the
paper (`SG+MSC`, `SG+SNV`, `SG1+MSC`, and `SG1+SNV`) and evaluates every model
with and without PCA-Mahalanobis outlier removal:

```bash
python -m benchmarks.run_lettuce_paper_analysis \
    /path/to/1-s2.0-S1574954126003511-mmc1.csv \
    --msc-reference-scope training \
    --output benchmarks/lettuce_paper_results/training
```

Use `--msc-reference-scope global` to reproduce preprocessing and outlier
removal before the 70:30 split described in the manuscript. This mode uses test
data during preprocessing and must be labelled as manuscript reproduction, not
as a leakage-safe performance estimate. `training` learns the MSC reference and
outlier model from training data only. `batch` calculates independent training
and test means; it is a transductive sensitivity analysis because one test
sample's transformation depends on the other test samples.

SNV always centers and scales each spectrum independently and therefore does
not use a mean spectrum shared across samples. The configurable SG defaults are
window length 11 and polynomial order 3. These values reproduce the paper's
reported outlier totals, but the paper does not state them and the totals do not
identify them uniquely.
