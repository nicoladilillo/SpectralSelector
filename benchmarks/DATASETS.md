# Benchmark datasets

`benchmarks._datasets` loads the spectroscopy datasets used by the benchmark
scripts. It is part of the source repository, not of the installed
`spectralselector` package: run the code below from the repository root. It
includes both classification and regression tasks and uses a consistent
NumPy-first interface.

## What a loader returns

The default result is a scikit-learn `Bunch`, which supports both attribute and
dictionary access:

```python
from benchmarks._datasets import fetch_corn

dataset = fetch_corn(target="oil", instrument="m5")

X = dataset.data  # Spectra: samples x wavelengths
y = dataset.target  # Class labels or regression values
wavelengths = dataset.wavelengths

print(dataset.name)  # corn
print(dataset.task)  # regression
print(X.shape)  # (80, 700)
print(wavelengths[[0, -1]])  # [1100.0, 2498.0]
```

All row-aligned datasets expose:

| Field | Meaning |
|---|---|
| `data` | Numeric spectral matrix shaped `(n_samples, n_wavelengths)` |
| `target` | Class labels or continuous reference measurements |
| `wavelengths` | Physical wavelength or wavenumber value for every column |
| `feature_names` | String representation of the spectral axis |
| `target_names` | Name of each predicted property |
| `sample_metadata` | Cycle, source split, instrument, or sample identifiers |
| `task` | `classification`, `regression`, or `time_aligned_regression` |
| `DESCR` | Short description of the unmodified loader output |
| `source`, `citation`, `license` | Provenance and reuse information |

Use `return_X_y=True` for the compact scikit-learn convention:

```python
X, y = fetch_corn(return_X_y=True)
```

Use `as_frame=True` to obtain pandas objects while retaining numeric wavelength
column names:

```python
dataset = fetch_corn(as_frame=True)
print(dataset.data.iloc[:3, :5])
print(dataset.target.describe())
```

Pandas-backed output is available through `as_frame=True`.

## Dataset catalogue

| Loader | Task | Default output | Spectral axis | Storage |
|---|---|---:|---|---|
| `load_coffee` | Classification | 60 x 1,841 | ATR-FTIR axis from chemotools | chemotools |
| `load_fermentation_train` | Regression | 21 x 1,047 | ATR-FTIR wavenumbers | chemotools |
| `load_fermentation_test` | Time-aligned regression | 1,629 spectra; 34 references | ATR-FTIR wavenumbers | chemotools |
| `load_lettuce_water_stress(path)` | Classification | 216 x 2,151 | 350-2,500 nm, 1 nm | User-supplied CSV |
| `fetch_corn` | Regression | 80 x 700 | 1,100-2,498 nm, 2 nm | Eigenvector cache |
| `fetch_diesel` | Regression | 246 x 401 | 750-1,550 nm, 2 nm | Eigenvector cache |
| `fetch_shootout` | Regression | 655 x 650 | 600-1,898 nm, 2 nm | Eigenvector cache |

Loaders return raw source values. They do not smooth, normalize, scale, remove
outliers, select wavelengths, or create a random train/test split. This keeps
the datasets suitable for comparing alternative preprocessing and selection
pipelines without hidden transformations.

## Local lettuce water-stress data

The CC BY-NC-ND 4.0 supplementary CSV is not included in SpectralSelector and is
not downloaded by the package. A user who has obtained an authorized copy can
parse it locally:

```python
from benchmarks._datasets import load_lettuce_water_stress

dataset = load_lettuce_water_stress("/path/to/1-s2.0-S1574954126003511-mmc1.csv")
print(dataset.data.shape)  # (216, 2151)
print(dataset.wavelengths[[0, -1]])  # [350.0, 2500.0]
print(set(dataset.target))  # {'Controlled', 'Water Stressed'}
```

The source class strings are retained in
`dataset.sample_metadata["source_class"]`; `target` uses the clearer display
labels. Pass `wavelength_range=(700, 900)` to select an inclusive local range at
load time. The parser does not preprocess, split, or redistribute the data.

## Future lettuce-health dataset

The lettuce-health dataset is intentionally not exposed by the package yet.
Before adding a public fetcher, the dataset must be published in a stable Zenodo
record with a final DOI, explicit licence, immutable filename, and checksum. A
future loader will preserve the raw data and record any label normalization in
its documentation and tests.

## Regression benchmarks

### Corn oil

The corn dataset contains spectra from 80 samples measured on three NIR
instruments. Moisture, oil, protein, and starch are available as regression
targets:

```python
from benchmarks._datasets import fetch_corn

dataset = fetch_corn(target="oil", instrument="m5")
print(dataset.data.shape)  # (80, 700)
print(dataset.target_names)  # ['oil']
```

Valid instruments are `m5`, `mp5`, and `mp6`. Valid targets are `moisture`,
`oil`, `protein`, and `starch`.

### Diesel BP50

The default loader reproduces the 246-sample BP50 benchmark used by Wang et al.:

```python
from benchmarks._datasets import fetch_diesel

dataset = fetch_diesel()
print(dataset.data.shape)  # (246, 401)
print(dataset.target.min())  # 197
print(dataset.target.max())  # 293
```

Pass `subset="all"` to use the broader SWRI property table and select another
available property with `target`.

### Shootout-3

The pharmaceutical-tablet benchmark provides two instrument measurements,
three continuous targets, and native calibration, validation, and test groups:

```python
from benchmarks._datasets import fetch_shootout

dataset = fetch_shootout(target="assay", instrument=1, split="all")
print(dataset.data.shape)  # (655, 650)
print(dataset.sample_metadata["split"][:3])  # native split labels
```

Targets are `weight`, `hardness`, and `assay`. Set `split` to `calibrate`,
`validate`, or `test` to retrieve one native group.

### Regression and selection example

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import VIPSelector
from benchmarks._datasets import fetch_corn

dataset = fetch_corn(target="oil")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

pipeline = make_pipeline(
    StandardScaler(),
    VIPSelector(
        n_components=5,
        n_features_to_select=20,
        wavelengths=dataset.wavelengths,
    ),
    PLSRegression(n_components=5, scale=False),
)

result = cross_validate(
    pipeline,
    dataset.data,
    dataset.target,
    cv=cv,
    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
)
print(-result["test_rmse"].mean())
print(result["test_r2"].mean())
```

## Chemotools datasets

The coffee and fermentation loaders delegate to the installed chemotools
package. This avoids copying approximately 17 MB of upstream package data and
keeps the values synchronized with chemotools:

```bash
pip install chemotools
```

```python
from benchmarks._datasets import load_coffee, load_fermentation_train

coffee = load_coffee()
fermentation = load_fermentation_train()
print(coffee.task, coffee.data.shape)
print(fermentation.task, fermentation.data.shape)
```

`load_fermentation_test()` is intentionally marked `time_aligned_regression`:
chemotools supplies 1,629 spectra sampled every 1.25 minutes and 34 separate
hourly HPLC measurements. These are not one-to-one rows. Spectra and references
must be aligned using acquisition time before supervised modelling.

## Wang et al. comparison

The 2025 comparison paper describes eight public regression benchmarks. The
verified convenience dispatcher currently exposes three:

```python
from benchmarks._datasets import fetch_wang2025

corn = fetch_wang2025("corn")
diesel = fetch_wang2025("diesel")
shootout = fetch_wang2025("shootout")
```

Beer, wheat protein, tobacco, soy moisture, and the UCPH tablet benchmark are
not silently substituted. Their paper links are publication references, retired
FTP sites, or sources without a stable, licence-explicit machine download. A
loader should be added only after its canonical file, schema, checksum, and
reuse terms are verified.

## Download cache

Remote files are downloaded atomically and checksum verified. The default cache
is `~/.cache/spectralselector`. Change it with either mechanism:

```bash
export SPECTRALSELECTOR_DATA=/path/to/dataset-cache
```

```python
dataset = fetch_corn(data_home="/path/to/dataset-cache")
```

## Avoiding leakage

Dataset loading and model evaluation are separate operations. For a valid
performance estimate:

1. Keep any final test set untouched.
2. Put learned preprocessing and wavelength selection inside a scikit-learn
   pipeline.
3. Fit that pipeline independently in every training fold.
4. Tune selector and model hyperparameters using inner cross-validation.
5. Evaluate the chosen procedure once on the held-out test set.

Selecting wavelengths on the complete dataset before cross-validation allows
test-fold target information to influence the selected variables and produces
an optimistic performance estimate.
