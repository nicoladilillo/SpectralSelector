# SpectralSelector

![SpectralSelector logo](https://raw.githubusercontent.com/nicoladilillo/spectralselector/main/assets/branding/spectralselector-logo.svg)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/nicoladilillo/spectralselector/actions/workflows/ci.yml/badge.svg)](https://github.com/nicoladilillo/spectralselector/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/spectralselector/badge/?version=latest)](https://spectralselector.readthedocs.io/)

SpectralSelector is a scikit-learn-compatible Python library for wavelength and
variable selection in spectroscopy and chemometrics. Every selector learns
column indices during `fit`, works inside a `Pipeline`, and exposes the selected
indices and physical wavelengths without coupling selection to a preprocessing
package.

The project currently has an alpha API. Use nested validation for scientific
comparisons and review the [algorithm-validation status](docs/algorithm_validation.md)
before making claims of numerical equivalence with a reference implementation.

## Included methods

| Family | Public class | Typical task | Uses a model? |
|---|---|---|---|
| Fixed selection | `IndexSelector` | Regression or classification | No |
| Univariate filtering | `WavelengthSelectKBest` | Regression or classification | No |
| Information + collinearity filter | `MIVIFSelector` | Regression or classification | No (optional: length) |
| Covariance selection | `CovSelSelector` | Regression or classification | No (optional: length) |
| Covariance selection, validated length | `FCovSelSelector` | Regression or classification | Length only |
| Successive projections (collinearity) | `SPASelector` | Regression or classification | Start and length only |
| Model importance | `VIPSelector`, `SelectivityRatioSelector` | Regression or classification | PLS / PLS-DA |
| Competitive sampling | `CARSSelector` | PLS regression or PLS-DA | PLS / PLS-DA |
| Bootstrapping soft shrinkage | `BOSSSelector` | PLS regression or PLS-DA | PLS / PLS-DA |
| Genetic search | `GeneticAlgorithmSelector` | Estimator-defined task | Any estimator |
| Interval combination optimization | `ICOSelector` | Estimator-defined task | Any estimator |

The package also provides a `PLSDA` classifier (PLS scores followed by LDA).
VIP and SR also accept chemotools' nearest-centroid
`chemotools.classification.PLSDA`. The
[method taxonomy](docs/method_taxonomy.md) explains which selectors depend on a
model and how the method families differ.
CARS and BOSS evaluate candidate subsets across component counts from 1 through
`max_components` and retain the best internally cross-validated count. CARS
constructs PLSR or PLS-DA from the inferred task; BOSS does the same by default
and accepts only an explicit `PLSRegression` or `PLSDA` estimator.

For predictive-search methods, `task` chooses regression or classification,
`scoring` accepts a scikit-learn metric name or scorer, and `cv` accepts an
integer, a splitter object, or `-1` for leave-one-out validation. These controls
are available in CARS, BOSS, the genetic wrapper, and fCovSel. fCovSel uses PLSR
or PLSDA to validate sequence length and supports multi-response regression and
multiclass classification. Direct-ranking methods such as CovSel, VIP, SR, and
univariate selection receive their evaluation metric and folds from the
surrounding pipeline or benchmark; SR classification is currently binary.

## Installation

Install the package from a clone:

```bash
python -m pip install .
```

Optional groups add plotting, chemotools interoperability, documentation, or
development tools:

```bash
python -m pip install '.[plot,chemotools]'
python -m pip install -e '.[dev]'
```

For this repository, the Conda bootstrap is declared in
[`environment.yml`](environment.yml), with exact Python dependencies recorded
in [`requirements-dev.lock`](requirements-dev.lock):

```bash
conda env create -f environment.yml
conda activate wavelength_selection_package
```

## Quick start

```python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import VIPSelector

wavelengths = np.linspace(700, 900, X_train.shape[1])
pipeline = Pipeline(
    [
        ("scale", StandardScaler()),
        (
            "select",
            VIPSelector(
                n_components=4,
                n_features_to_select=12,
                wavelengths=wavelengths,
            ),
        ),
        ("model", PLSRegression(n_components=4, scale=False)),
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

selector = pipeline.named_steps["select"]
print(selector.selected_indices_)
print(selector.selected_wavelengths_)
```

CARS chooses its PLS model and component count internally:

```python
from spectralselector import CARSSelector

selector = CARSSelector(
    task="regression",
    cv=5,
    max_components=10,
    n_runs=20,
    n_iterations=50,
    random_state=42,
    wavelengths=wavelengths,
)
X_selected = selector.fit_transform(X_train, y_train)
print(selector.best_n_components_, selector.selected_indices_)
```

### Classification with PLS-DA

For class labels, use a PLS-DA classifier. The quick start uses chemotools'
PLS-DA (available in chemotools releases after 0.5.0) and falls back to the
package implementation otherwise. Data are rows = samples and columns =
wavelengths; labels are a separate array.

```python
try:
    from chemotools.classification import PLSDA
except ImportError:  # chemotools <= 0.5.0
    from spectralselector import PLSDA
from sklearn.pipeline import Pipeline

from spectralselector import MIVIFSelector, VIPSelector

# Model-dependent: VIP reads the weights of a fitted PLS-DA.
vip_pipeline = Pipeline(
    [
        (
            "select",
            VIPSelector(
                estimator=PLSDA(n_components=2),
                task="classification",
                n_features_to_select=12,
                wavelengths=wavelengths,
            ),
        ),
        ("model", PLSDA(n_components=2)),
    ]
)

# Model-independent: MI-VIF ranks by mutual information and drops collinear
# wavelengths (VIF > 10) without fitting any predictive model.
mivif_pipeline = Pipeline(
    [
        ("select", MIVIFSelector(random_state=0, wavelengths=wavelengths)),
        ("model", PLSDA(n_components=2)),
    ]
)
vip_pipeline.fit(X_train, labels_train)
print(vip_pipeline.named_steps["select"].selected_wavelengths_)
```

### Let your model choose how many wavelengths to keep

CovSel, fCovSel and MI-VIF rank wavelengths without a model, but the best
*number* of wavelengths depends on the model that will use them. Pass that
model as `estimator` and set `n_features_to_select="auto"`. Inside every
validation fold the ranking is recomputed on the training rows only, every
prefix of the ranking is scored with your model (and optional `param_grid`),
and the best length is kept:

```python
from spectralselector import CovSelSelector

selector = CovSelSelector(
    "auto",
    max_features=30,
    estimator=PLSDA(),
    param_grid={"n_components": [2, 4, 6]},
    cv=5,
    scoring="balanced_accuracy",
    wavelengths=wavelengths,
)
selector.fit(X_train, labels_train)
print(selector.best_n_features_, selector.best_params_)
selector.plot_diagnostics()  # validation curve + selected wavelengths
```

`MIVIFSelector("auto", estimator=...)` and `FCovSelSelector(estimator=...)`
take the same arguments. With a fixed integer (CovSel, MI-VIF) or `None`
(MI-VIF, the published procedure) no model is fitted.

Selectivity Ratio can apply the published F-test boundary instead of a fixed
count:

```python
from spectralselector import SelectivityRatioSelector

sr = SelectivityRatioSelector(n_components=4, threshold="f_test", alpha=0.05)
sr.fit(X_train, y_train)
print(sr.threshold_, sr.selected_wavelengths_)
```

Small executable workflows are available in [`examples/`](examples/).

## Estimator contract

All fitted selectors expose a consistent core interface:

| Attribute | Meaning |
|---|---|
| `support_` | Boolean mask in the original column order |
| `selected_indices_` | Selected zero-based column indices |
| `selection_order_` | Ranking or search order, when meaningful |
| `wavelengths_in_` | Optional wavelength axis supplied at construction |
| `selected_wavelengths_` | Wavelengths corresponding to selected indices |
| `selection_time_` | Wall-clock time spent fitting the selector |

Selection belongs inside the training pipeline. Fitting a selector on the full
dataset before cross-validation leaks information into validation folds. The
[validation guide](docs/validation.md) shows the recommended nested-CV pattern.

## chemotools and outlier handling

SpectralSelector accepts the NumPy arrays returned by chemotools transformers.
The [chemotools pipeline example](examples/chemotools_pipeline.py) demonstrates
SNV preprocessing. The [Hotelling T² example](examples/hotelling_outlier_filter.py)
shows the explicitly global protocol requested for the lettuce analysis:
fit chemotools `HotellingT2` on raw data, remove outliers from the complete
dataset, and only then split into train and test sets. That protocol is useful
for reproducing a stated analysis, but it is not leakage-safe for estimating
performance on future samples; the example calls out this distinction.

## Datasets and benchmarks

The installed package contains no dataset loaders and no research data. The
loaders used by the benchmarks live in `benchmarks/_datasets/` in the source
repository; third-party data licences remain separate from the software's MIT
licence. See the [benchmark dataset guide](benchmarks/DATASETS.md) and
[`DATASET_LICENSES.md`](DATASET_LICENSES.md).

Benchmark programs live in [`benchmarks/`](benchmarks/). This folder contains
code and reports, not source datasets. Temporary outputs below
`benchmarks/results/` are ignored; small release-specific technical reports are
tracked below `benchmarks/reports/`.

## Development

Run the focused quality checks with the active environment:

```bash
python -m ruff check src tests examples benchmarks
codespell .
python -m pytest
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
python -m build
python -m twine check dist/*
```

Contribution and release guidance is in [`CONTRIBUTING.md`](CONTRIBUTING.md),
[`docs/development.md`](docs/development.md),
[`docs/publication_plan.md`](docs/publication_plan.md),
[`docs/releasing.md`](docs/releasing.md), and [`CHANGELOG.md`](CHANGELOG.md).

## Citation and licence

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). SpectralSelector
is distributed under the [MIT licence](LICENSE).
