# SpectralSelector 0.1.0 technical benchmark

Dataset-by-method matrix on 30 public regression datasets: the three Wang et
al. (2025) benchmarks with a stable download (corn oil, diesel BP50, Shootout
2002 assay) and the 27 NIRBENCH-DL v0.1 tasks that ship their data. Generated
from the repository root with:

```bash
python benchmarks/run_dataset_matrix.py \
  --datasets corn,diesel,shootout,nirbench_1,nirbench_2,nirbench_3,nirbench_5,nirbench_6,nirbench_7,nirbench_8,nirbench_9,nirbench_10,nirbench_11,nirbench_13,nirbench_15,nirbench_16,nirbench_17,nirbench_18,nirbench_19,nirbench_20,nirbench_21,nirbench_22,nirbench_23,nirbench_24,nirbench_25,nirbench_26,nirbench_27,nirbench_28,nirbench_29,nirbench_30 \
  --methods all --profile smoke --n-jobs 4 \
  --nirbench-path /path/to/NIRBENCH_DL_v0.1 \
  --output benchmarks/reports/v0.1.0
```

Files: `fold_results.csv` (one row per dataset, method, outer fold),
`summary.csv` (aggregates), `selections.json` (selected indices and
wavelengths), and `matrix_manifest.json` (configuration, dataset provenance,
split hashes, environment, and the 48 failed or not-applicable jobs).

## Scope and limitations

- **Smoke profile.** CARS, BOSS, and the genetic selector ran with reduced
  stochastic budgets and one seed. Their results are execution checks, not
  method rankings; CARS in particular kept a single wavelength per fold here.
  The full-budget `publication` profile has not been run for this release.
- **Not scientific validation.** See `docs/algorithm_validation.md` for the
  evidence behind each method.
- NIRBENCH tasks use their fixed train/test partition as one outer split;
  corn and diesel use three shuffled outer folds; Shootout uses its
  calibrate/validate/test groups.
- No spectral preprocessing was applied. Downstream model: ridge regression
  tuned over five penalties. Length-validated selectors searched up to 20
  wavelengths.

## Observations

- BOSS had the best mean test R² among selectors on 20 of 30 datasets, in line
  with Wang et al. (2025). CovSel, CovSel-auto, and fCovSel usually came close
  to the full spectrum; their best length was often the 20-wavelength cap.
- MI-VIF kept a median of two wavelengths (VIF <= 10 on collinear spectra) and
  predicted poorly on most tasks.
- `selectivity_ratio_ftest` found no significant wavelength on 26 of the 30
  datasets.
- On Shootout, the length-validated selectors, CARS, BOSS, and the genetic
  selector failed: inside a training fold only one sample group remains, so
  their group-aware inner validation cannot split. This is a benchmark
  configuration limitation recorded in the manifest.

## Provenance notes

The run used source revision `953cc20` plus the uncommitted 0.1.0 changes
(recorded as `dirty` and `0.1.0.dev0`). Changes made after the run and before
the release commit are comments, docstrings, a lint-only loop-variable rename,
and manifest path redaction; none alter computed results. The local NIRBENCH
path and cache directory were replaced by their portable names in
`matrix_manifest.json` before committing.
The loaders recorded in `matrix_manifest.json` as `spectralselector.datasets.*`
were moved unchanged to `benchmarks/_datasets/` before the release; the package
itself no longer ships dataset loaders.
