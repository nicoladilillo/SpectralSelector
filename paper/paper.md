---
title: "SpectralSelector: Scikit-learn-compatible wavelength selection for chemometric workflows"
tags:
  - Python
  - chemometrics
  - spectroscopy
  - wavelength selection
  - variable selection
  - scikit-learn
authors:
  - name: Nicola Dilillo
    affiliation: 1
    corresponding: true
affiliations:
  - name: Politecnico di Torino, Italy
    index: 1
date: 26 August 2026
bibliography: paper.bib
---

# Summary

SpectralSelector is a Python package for selecting informative columns from
spectroscopic matrices. It provides a common estimator interface for
wavelength-selection methods that are otherwise often distributed as isolated
scripts with different input conventions and fitted outputs. Each selector
implements the scikit-learn `fit` and `transform` protocol, returns selected
columns as a two-dimensional NumPy array, and exposes the learned indices,
Boolean mask, optional wavelength values, selection order, diagnostics, and
wall-clock selection time. These properties allow wavelength selection to be
placed inside cross-validation and model-selection pipelines.

The initial implementation includes covariance selection (CovSel), faster
covariate selection (fCovSel), mutual-information selection with
variance-inflation-factor screening (MI-VIF) [@cheng2022mivif], competitive
adaptive reweighted sampling (CARS), bootstrapping soft shrinkage (BOSS), a
generic genetic mask optimizer, variable importance in projection (VIP),
selectivity ratio (SR) with its published F-test boundary, univariate
selection, and explicit index selection. It supports regression and
classification where the underlying method permits. CARS and BOSS use PLS
regression or the included PLS-DA model and optimize the latent-component count
through internal cross-validation.

# Statement of need

Spectroscopic measurements routinely contain hundreds or thousands of strongly
correlated variables. Selecting a smaller wavelength set can improve model
interpretability, reduce acquisition and computation costs, and support the
design of multispectral instruments. Reliable evaluation is difficult,
however, because supervised selection performed once on the complete dataset
leaks response information into validation folds and gives optimistic
performance estimates. Stochastic methods also need explicit random-state and
computational controls to be reproducible.

Scikit-learn provides a general estimator and pipeline ecosystem
[@pedregosa2011scikit], while chemotools provides scikit-learn-compatible
spectral preprocessing [@chemotools]. Neither currently offers a unified set of
chemometric wavelength-selection algorithms with common masks, wavelength
metadata, and method-specific diagnostics. SpectralSelector addresses this focused
gap. It accepts NumPy arrays produced by chemotools and can also follow any
transformer that emits a finite two-dimensional numeric matrix.

# State of the field

General feature-selection utilities in scikit-learn include univariate,
model-based, recursive, and sequential selectors, but do not implement several
methods widely used in chemometrics. Published approaches such as CovSel
[@roger2011covsel], fCovSel [@mishra2022fcovsel], CARS [@li2009cars], BOSS
[@deng2016boss], VIP, and SR [@rajalahti2009sr; @farres2015vip] are commonly
reimplemented for individual studies. Those implementations may differ in
deflation, resampling, stopping, and cross-validation behavior and may not
follow the estimator conventions required for leakage-safe composition. The
package's genetic selector is deliberately described as a generic optimizer,
not as a reference-equivalent implementation of a particular named genetic
algorithm.

Adding wavelength selection directly to chemotools was considered. A separate
package was chosen because selection has a distinct scientific validation
surface, includes computationally expensive supervised searches, and is useful
with preprocessing libraries other than chemotools. The boundary remains small:
NumPy arrays and the scikit-learn estimator protocol. This avoids a required
chemotools dependency while preserving natural interoperability.

# Software design

All selectors derive from `BaseWavelengthSelector`, which combines
scikit-learn's `BaseEstimator` and `SelectorMixin`. Constructors contain only
configuration, while `fit(X, y)` validates the selection data and learns a
support mask. `transform(X)` applies that fixed mask to new observations. The
base class exposes `get_support`, `fit_transform`, feature-name handling, and
common fitted attributes. Optional wavelength metadata is validated against the
input column count and mapped to selected indices without changing numerical
output.

The package separates low-cost filter methods, model-derived embedded methods,
and wrapper or hybrid search methods, and documents which selections depend on
a predictive model. Methods that rank wavelengths without a model (CovSel,
fCovSel, MI-VIF) can choose *how many* wavelengths to keep with the model that
will use them: the ranking is recomputed in every inner training fold and each
prefix is scored with that model, so the length is model-specific while the
ranking stays model-free. Model-dependent selectors clone accepted
estimators. CARS constructs a task-appropriate PLS model; BOSS optionally accepts
only PLS regression or PLS-DA. CARS, BOSS, and genetic searches use joblib for
bounded process-based parallelism, and stochastic selectors expose
`random_state`. Classification and regression behavior is explicit or inferred
from the target and estimator. Diagnostic plots present the most informative
method-specific trace, such as scores, selection order, votes, or convergence.

A selector belongs inside the fitted pipeline:

```python
pipeline = Pipeline(
    [
        ("preprocess", StandardScaler()),
        ("select", CovSelSelector(n_features_to_select=12, wavelengths=wavelengths)),
        ("model", PLSRegression(n_components=4, scale=False)),
    ]
)
scores = cross_validate(pipeline, X, y, cv=outer_cv)
```

Consequently, preprocessing, selection, and model fitting are repeated using
only each training fold. Selector hyperparameters must be tuned in an inner
cross-validation loop when reporting generalization performance. The test suite
checks estimator behavior, regression and classification use, deterministic
random states, diagnostic outputs, dataset loaders, and interoperability with
chemotools. NumPy [@harris2020numpy], SciPy [@virtanen2020scipy], scikit-learn,
joblib, pandas, and tqdm are the required runtime dependencies.

# Research impact statement

SpectralSelector is intended to make comparative wavelength-selection studies more
reproducible and to shorten the path from full-spectrum experiments to compact
sensor designs. Its uniform API enables the same train/test split, preprocessing
pipeline, estimator, and evaluation procedure to be applied across selector
families. The included benchmark framework compares selected models with
full-spectrum and random-subset baselines using nested cross-validation and
records predictive performance, selected-variable counts, stability, and
runtime.

At the time of this draft, the package is preparing for its first public release
and does not yet have evidence of independent external adoption. Before JOSS
submission, this section will be updated with a versioned public benchmark,
archived outputs, and concrete examples of use beyond package-development tests.

# AI usage disclosure

OpenAI Codex and Anthropic's Claude Code were used to assist with software
scaffolding, algorithm implementation, documentation and manuscript drafting,
test suggestions, and repository review. The author
reviewed and edited the generated material and is responsible for the software,
scientific claims, references, tests, and final manuscript.

# Acknowledgements

The author thanks the maintainers of chemotools and the scientific Python
ecosystem on which this work builds.

# References
