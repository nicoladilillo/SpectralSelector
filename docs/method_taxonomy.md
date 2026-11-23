# Method taxonomy: model dependence and selector families

Wavelength selectors differ in two independent ways:

1. **Model dependence**: whether the selection uses a fitted predictive model,
   and therefore whether it changes when that model or its settings change.
2. **Selection strategy**: how candidate wavelengths are scored and searched
   (one variable at a time, sequential projection, model importance, sampling,
   evolutionary search, or intervals).

Both axes matter when a selection is reported. A model-dependent selection
answers "which wavelengths does *this* model need?", while a model-independent
selection answers "which wavelengths carry information about *y*?" These
are different scientific questions.

## Data layout

Every selector uses the usual spectroscopy layout: `X` has one row per sample
and one column per wavelength, and the response is a separate array. That
array is `y` with shape `(n_samples,)` for one property or class label, or
`Y` with shape `(n_samples, n_targets)` for several properties where the method
supports it. The optional `wavelengths` argument attaches the physical axis so
that fitted selectors report `selected_wavelengths_` as well as column indices.

## Axis 1: model dependence

### Model-independent selectors

These selectors compute their criterion directly from `X` and `y`. No
regression or classification model is fitted, so the selected wavelengths are
the same whichever model is trained on them later (PLSR, MLR, PLS-DA, SVM,
random forest, and so on).

| Selector | Criterion | Response used |
|---|---|---|
| `IndexSelector` | Wavelengths fixed by prior knowledge | No |
| `WavelengthSelectKBest` | Univariate F statistic or mutual information | Yes |
| `MIVIFSelector` | Mutual information ranking + VIF collinearity screen | Yes |
| `CovSelSelector` | Maximum squared covariance with sequential deflation | Yes |

Strengths: cheap, deterministic or seedable, interpretable, and transferable
across models. The selection is not tuned to a model's quirks, so it suits a
paper that compares several downstream models on the *same* wavelengths.

Limitations: univariate criteria ignore combinations that are informative only
jointly. Linear criteria (F statistic, covariance) miss non-linear relevance,
and mutual information depends on an estimator and on sample size. Without a
model there is no direct guarantee that the subset predicts well, so the number
of retained variables must be fixed by the user, by a rule (VIF ≤ 10 in MI-VIF),
or by validation.

### Partly model-dependent selectors

| Selector | Model-free part | Model-dependent part |
|---|---|---|
| `FCovSelSelector` | The covariance-ordered variable sequence | The sequence *length*, always chosen by cross-validated PLSR/PLS-DA or a user estimator |
| `CovSelSelector("auto", estimator=...)` | The CovSel order | The length, chosen with the supplied estimator |
| `MIVIFSelector("auto", estimator=...)` | The MI order and VIF screen | How many accepted variables to keep |
| `SPASelector` | The X-only projection order from a given start | The start wavelength and the length, chosen by cross-validated MLR/LDA or a user estimator |

The order in which wavelengths enter does not depend on the model; only where
the order is cut does. Supplying a different `estimator` or `param_grid` can
therefore change how many wavelengths are kept, but never which ones come
first. To avoid leakage, the order is recomputed on the training rows of
every validation fold before its prefixes are scored. CovSel and MI-VIF stay
fully model-independent when given a fixed count (or `None` for MI-VIF's
published rule); they become partly model-dependent only when asked to
validate their length.

### Model-dependent selectors: importance from one fitted model

| Selector | Model quantity read after fitting |
|---|---|
| `VIPSelector` | PLS X weights, scores, and y-loadings (Farrés et al., 2015, Eq. 1–2) |
| `SelectivityRatioSelector` | PLS regression vector, used for target projection (Eq. 3–7) |

The ranking is a property of one fitted PLS or PLS-DA model. It changes with
the number of latent variables, X/Y scaling, and preprocessing, and it is only
defined for PLS-like models. VIP rewards variables that explain X variance
along y-relevant components, which can include orthogonal (y-unrelated)
variation. SR uses only the y-predictive direction. In Farrés et al. this made
VIP more inclusive (more false positives) and the SR F-test more conservative
(more false negatives). Because the PLS projection is the same, VIP and SR
scores are identical for `spectralselector.PLSDA` and
`chemotools.classification.PLSDA` at equal scaling; only the classifiers'
decision rules differ.

### Model-dependent selectors: search guided by predictive performance (wrappers)

| Selector | Model used in the search | Search |
|---|---|---|
| `CARSSelector` | Internal PLSR/PLS-DA, component count cross-validated | Monte Carlo sampling + exponentially decreasing, coefficient-weighted retention |
| `BOSSSelector` | Internal PLSR/PLS-DA, component count cross-validated | Weighted bootstrap sampling + soft shrinkage of coefficient-based weights |
| `GeneticAlgorithmSelector` | Any user estimator and scorer | Evolutionary search over interval masks |
| `ICOSelector` | Any user estimator and scorer | Weighted bootstrap sampling of interval combinations, soft shrinkage, boundary local search |

These selectors optimize an internal cross-validated score of a specific model.
They can find combinations that filters miss, but they are the most expensive
and the most prone to *selection overfitting*: the internal CV score is
optimistically biased because it was maximized. Their reported performance
must come from an outer validation loop that refits the whole pipeline
(preprocessing, selector, model) on each outer training fold. Stochastic
wrappers (CARS, BOSS, GA) should also be repeated with several seeds and their
selection stability reported.

### Summary

| | Model-independent | Partly dependent | Model importance | Wrapper search |
|---|---|---|---|---|
| Fits a model? | No | Yes, for length only | Yes, once | Yes, many times |
| Same selection for any downstream model? | Yes | Order yes, length no | No | No |
| Captures variable combinations? | MI-VIF/CovSel partly | Yes (sequential) | Through the PLS model | Yes |
| Typical cost | Low | Medium | Medium | High to very high |
| Main risk | Misses joint effects; count must be chosen | Length tuned to one model | Ranking specific to the PLS model | Selection overfitting, instability |
| Selectors here | Index, KBest, MI-VIF, CovSel | fCovSel; CovSel/MI-VIF with `"auto"` | VIP, SR | CARS, BOSS, GA |

Whatever the category, every data-dependent selector (all except a fixed
`IndexSelector`) uses `y` and must be fitted on training data only. Putting the
selector inside a scikit-learn `Pipeline` guarantees this during
cross-validation.

## Axis 2: families of wavelength selection methods

The chemometrics literature (for example Andersen and Bro, 2010; Mehmood et al.,
2012) groups variable-selection methods into *filter*, *wrapper*, and *embedded*
methods. For spectra it is also useful to separate methods that select
individual wavelengths from those that select contiguous intervals. The table
lists the main families, representative methods, and coverage in this package.

| Family | Idea | Representative methods | In SpectralSelector |
|---|---|---|---|
| Knowledge-based | Use known absorption bands | Literature band assignment | `IndexSelector` |
| Univariate filters | Score each wavelength alone against y | Correlation, F test, mutual information | `WavelengthSelectKBest` |
| Relevance–redundancy filters | Maximize relevance to y while limiting redundancy or collinearity between selected wavelengths | MIFS, mRMR, MI-VIF; SPA (collinearity only, X-based) | `SPASelector`, `MIVIFSelector` |
| Sequential covariance/projection | Pick the most y-covariant variable, deflate, repeat | CovSel, fCovSel | `CovSelSelector`, `FCovSelSelector` |
| Model-parameter importance (embedded) | Rank by a fitted model's parameters | Regression coefficients, VIP, SR, sMC, UVE | `VIPSelector`, `SelectivityRatioSelector` |
| Model population analysis | Fit many sub-models on sampled subsets, aggregate variable statistics, shrink | CARS, BOSS, MC-UVE, IRIV, VCPA, VISSA, random frog | `CARSSelector`, `BOSSSelector` |
| Evolutionary/stochastic search (wrapper) | Optimize a CV score over subsets with a population heuristic | GA, PSO, simulated annealing, ant colony | `GeneticAlgorithmSelector` (generic interval GA) |
| Interval methods | Select contiguous spectral windows instead of single wavelengths | iPLS, biPLS, siPLS, mwPLS, GA-iPLS, ICO | Interval encoding in the GA only |
| Sparse/regularized models (embedded) | Penalize coefficients so that most are exactly zero | LASSO, elastic net, sparse PLS | Not implemented; usable through scikit-learn |

How to read the families:

- **Filters** (univariate, relevance–redundancy, covariance) are
  model-independent and fast; use them for screening, interpretation, and
  comparing several models on one common subset.
- **Embedded** methods (model importance, sparse models) take the selection
  from a model fitted once; they are efficient but specific to that model
  family.
- **Wrappers** (model population analysis, evolutionary search) optimize
  predictive performance directly and are the most powerful and the most
  overfitting-prone; nested validation is mandatory.
- **Interval** methods are a choice of *search unit* rather than a separate
  mechanism. Contiguous windows respect the band structure of spectra, give
  more stable selections, and fit wavelength-resolved instruments, at the cost
  of resolution.

## Choosing a method

- To explain *which wavelengths relate to the property*, and to compare models
  on a common subset, start with a model-independent filter (MI-VIF for few,
  non-collinear wavelengths suited to MLR or multispectral sensors; CovSel for a
  covariance-ordered sequence).
- To interpret an existing PLS or PLS-DA model, use VIP for an inclusive
  ranking and SR (with `threshold="f_test"`) for a conservative,
  prediction-oriented one; report both when they disagree.
- To get the best prediction from a PLS model, use CARS or BOSS with several
  seeds and nested validation, and report selection frequencies.
- To design an instrument with a few bands, use interval-oriented or
  collinearity-aware selection and check the result with an outer validation.

## References

- Andersen, C. M., and Bro, R. (2010). Variable selection in regression—a
  tutorial. *Journal of Chemometrics*, 24(11–12), 728–737.
- Cheng, J., Sun, J., Yao, K., Xu, M., and Cao, Y. (2022). A variable selection
  method based on mutual information and variance inflation factor.
  *Spectrochimica Acta Part A*, 268, 120652.
  <https://doi.org/10.1016/j.saa.2021.120652>
- Deng, B.-C. et al. (2016). A bootstrapping soft shrinkage approach for
  variable selection in chemical modeling. *Analytica Chimica Acta*, 908,
  63–74. <https://doi.org/10.1016/j.aca.2016.01.001>
- Farrés, M., Platikanov, S., Tsakovski, S., and Tauler, R. (2015). Comparison
  of the variable importance in projection (VIP) and of the selectivity ratio
  (SR) methods for variable selection and interpretation. *Journal of
  Chemometrics*, 29(10), 528–536. <https://doi.org/10.1002/cem.2736>
- Li, H., Liang, Y., Xu, Q., and Cao, D. (2009). Key wavelengths screening
  using competitive adaptive reweighted sampling method for multivariate
  calibration. *Analytica Chimica Acta*, 648(1), 77–84.
  <https://doi.org/10.1016/j.aca.2009.06.046>
- Mehmood, T., Liland, K. H., Snipen, L., and Sæbø, S. (2012). A review of
  variable selection methods in partial least squares regression.
  *Chemometrics and Intelligent Laboratory Systems*, 118, 62–69.
- Mishra, P. (2022). A brief note on a new faster covariate's selection
  (fCovSel) algorithm. *Journal of Chemometrics*, 36(5), e3397.
  <https://doi.org/10.1002/cem.3397>
- Rajalahti, T. et al. (2009). Discriminating variable test and selectivity
  ratio plot. *Analytical Chemistry*, 81(7), 2581–2590.
- Roger, J. M., Palagos, B., Bertrand, D., and Fernandez-Ahumada, E. (2011).
  CovSel: Variable selection for highly multivariate and multi-response
  calibration. *Chemometrics and Intelligent Laboratory Systems*, 106(2),
  216–223. <https://doi.org/10.1016/j.chemolab.2010.10.003>
