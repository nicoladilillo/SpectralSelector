# Implementing a wavelength selector

This guide defines the engineering and scientific contract for adding a method
to SpectralSelector. A selector is a scikit-learn transformer that learns a
column support mask during `fit`; it is not necessarily a predictive model.

## Choose the method family

Classify a proposed method before designing its API:

| Family | Uses `y` | Fits a predictor | Typical examples |
|---|---:|---:|---|
| Manual/fixed | No | No | Known wavelength indices |
| Filter/univariate | Usually | No | F-score, mutual information |
| Relevance-redundancy filter | Yes | No | MI-VIF, mRMR |
| Projection/deflation | Yes | No separate predictor | CovSel, fCovSel |
| Embedded/model importance | Yes | Yes | VIP, selectivity ratio |
| Wrapper/search | Yes | Yes repeatedly | CARS, BOSS, genetic search |
| Interval/group | Depends | Depends | GA-iPLS, interval selectors |
| Stability/ensemble | Yes | Often repeatedly | Repeated-run consensus |
| Neural sequence | Usually | Usually | Sparse Transformer selector |

The family determines which assumptions and fitted diagnostics users need. Do
not give a generic optimizer the name of a published method unless its complete
operators and validation evidence match that method.

## Scientific specification first

Before writing code, record:

1. the primary methodological reference and DOI, if one exists;
2. the exact mathematical objective and selection rule;
3. regression and classification support;
4. required scaling, centering, preprocessing, and estimator behavior;
5. initialization, stopping, tie-breaking, and numerical guard rules;
6. every random process and how `random_state` controls it;
7. computational complexity and parallelism;
8. differences between the paper and proposed package extensions; and
9. the external or hand-calculated fixture used to validate the equations.

Passing an API test is not evidence of equivalence with a paper. Named
equivalence additionally requires an authoritative external result with a
declared numerical tolerance.

## Required module presentation

Every public method module begins with a scientific presentation following this
template:

```python
"""
Full algorithm name and purpose.

Reference
---------
Authors (year).
Title. Journal, volume, pages.
DOI URL.

Paper algorithm implemented here
--------------------------------
Concise ordered description of the published operations.

Package behavior and extensions
-------------------------------
Exact differences, additional tasks, defaults, diagnostics, and unsupported
parts of the paper.
"""
```

For generic utilities without one primary paper, use `Reference status` and
state explicitly that no named-method equivalence is claimed. Never invent a
reference merely to fill the section.

## Base selector contract

Inherit from `BaseWavelengthSelector` and keep the constructor free of data or
fitted work:

```python
from spectralselector._base import BaseWavelengthSelector


class NewSelector(BaseWavelengthSelector):
    def __init__(self, n_features_to_select=10, *, wavelengths=None):
        self.n_features_to_select = n_features_to_select
        self.wavelengths = wavelengths

    def fit(self, X, y=None):
        X_array = self._validate_X(X, reset=True)
        # Validate parameters, calculate a mask, and save diagnostics here.
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selection_order,
        )
        return self
```

The fitted selector must expose:

- `n_features_in_`;
- one-dimensional Boolean `support_`;
- source-ordered `selected_indices_`;
- `selection_order_` when ranking/search order is meaningful;
- `wavelengths_in_` and `selected_wavelengths_` when wavelengths are supplied;
- `selection_time_`; and
- method-specific diagnostics documented with shapes and units.

`transform` always preserves original column order. A ranking may be stored
separately in `selection_order_`.

## Estimator-based methods

Clone user estimators before fitting. State the required fitted attributes, such
as `coef_`, PLS scores, weights, or loadings. Validate that the estimator's task
matches the target and do not mutate the caller's instance.

Selection and hyperparameter tuning belong inside a pipeline and inside each
training fold. Internal CV used to compare candidate subsets is not a final
generalization estimate; final evaluation needs a separate outer split.

Avoid nested parallelism. When candidate subsets are evaluated in parallel,
keep the CV work inside each candidate serial unless oversubscription is
explicitly controlled.

## Tests required for a new selector

Add tests for:

- parameter validation before expensive fitting;
- exact mask, index, and wavelength-axis behavior;
- deterministic tie-breaking;
- cloning and the scikit-learn estimator contract;
- pipeline and cross-validation operation;
- unseen-data transformation without refitting;
- fixed-seed reproducibility and serial/parallel agreement;
- constant, rank-deficient, very small, and invalid inputs;
- regression and every claimed classification mode;
- a hand-calculated equation fixture;
- an external reference fixture for named scientific equivalence; and
- persisted benchmark execution when the method is expensive.

Tests should verify scientific diagnostics, not only output shapes.

## Documentation and release integration

For every public selector:

1. export it from `spectralselector.__init__`;
2. add it to `docs/api.rst` and the algorithm-family table;
3. document assumptions, scaling, cost, and extensions;
4. provide a small leakage-safe example;
5. add a diagnostic plot only when it helps interpret fitted behavior;
6. register task compatibility and benchmark settings;
7. add the primary reference and validation status; and
8. add a changelog entry.

## Neural and Transformer-based selectors

Here, *Transformer* means the neural attention architecture. It is distinct from
the scikit-learn term *transformer*, which means any object implementing
`transform`.

### Scientific design

Define the spectral token and sequence before choosing a network. A token may be
one wavelength, a contiguous patch, or a learned local embedding. Physical
wavelength values should inform positional encoding, especially when axes are
irregular or instruments use different resolutions.

The selection mechanism must produce an explicit support mask. Raw attention
weights are not automatically faithful feature importance. Prefer or compare
mechanisms such as differentiable gates, sparse attention, hard-concrete masks,
perturbation-based attribution, or stability aggregation. State whether the
network is only a selector or also the final predictor.

### Leakage and evaluation

All representation learning, normalization, mask learning, early stopping, and
hyperparameter tuning must use only an outer training fold. A pretrained model
must record its training sources so overlap with benchmark test data can be
excluded. Cache fold-specific checkpoints with the same dataset, split, code,
configuration, seed, and environment fingerprint as classical selectors.

Evaluate more than predictive accuracy:

- number and contiguity of selected wavelengths;
- mask stability across seeds and folds;
- attribution faithfulness under feature perturbation;
- robustness to noise, missing bands, resolution, and wavelength shifts;
- sample efficiency;
- CPU/GPU time, memory, energy where feasible, and checkpoint size; and
- performance against full-spectrum, random, univariate, and PLS baselines.

### Package integration

Deep-learning dependencies must remain optional, for example under a dedicated
extra. The estimator should expose explicit `device`, deterministic seed,
checkpoint, and inference settings; support CPU execution; and avoid importing
the deep-learning framework when the optional method is unused.

The public object still follows scikit-learn construction and cloning rules and
returns `support_`, `selected_indices_`, and wavelength metadata. Test variable
sequence lengths, masks/padding, irregular axes, serialization, regression,
binary and multiclass targets, device movement, and deterministic inference.

Label the method experimental until its selection mechanism and reference
benchmarks have independent scientific validation.
