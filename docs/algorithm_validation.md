# Algorithm validation status

Passing software tests establishes API consistency and deterministic behaviour;
it does not by itself establish scientific equivalence with a publication or
another implementation.

| Selector | API tests | Equation/synthetic fixture | External reference fixture | Status |
|---|---:|---:|---:|---|
| `IndexSelector` | Yes | N/A | N/A | Direct index mask |
| `WavelengthSelectKBest` | Yes | Yes | Delegated | Standard scikit-learn scores |
| `VIPSelector` | Yes | Yes | chemotools 0.4.3 | Scores/support agree |
| `SelectivityRatioSelector` | Yes | Yes | chemotools 0.4.3 | Centered-X scores/support agree; F-test boundary reproduces Farrés et al. (2015) critical values |
| `MIVIFSelector` | Yes | Yes | N/A | VIF equals the OLS definition; procedure follows Cheng et al. (2022); no author code available |
| `SPASelector` | Yes | Yes | N/A | Order equals least-squares residual-norm reference; start/length search follows Araújo et al. (2001); no external fixture yet |
| `CovSelSelector` | Yes | Yes | plsVarSel 0.10.0 | Order/diagnostics agree |
| `FCovSelSelector` | Yes | Yes | plsVarSel 0.10.0 | Non-deflating order agrees |
| `CARSSelector` | Yes | Yes | Component search only | Full trajectory equivalence unconfirmed |
| `BOSSSelector` | Yes | Yes | Component search only | Full trajectory equivalence unconfirmed |
| `GeneticAlgorithmSelector` | Yes | Yes | N/A | Generic GA, not a named GA-iPLS reproduction |

## VIP, selectivity ratio, and CovSel

Small hand-calculated fixtures now verify the equations independently of the
normal fitted-model paths:

- VIP uses component-normalized X weights and score/Y-loading explained-variance
  contributions; its mean squared score is one. Tests cover centered and
  autoscaled PLS with single and multiple responses, plus the explicit binary
  and multiclass PLSDA extension.
- SR follows the target-projection score, loading, explained, residual, and
  explained-to-residual calculations in Farrés et al. (2015).
- CovSel selects by global squared covariance and applies the same orthogonal
  projector to residual X and Y at every step.

The VIP and SR regression scores and threshold-one support masks also agree with
versioned outputs from the independent chemotools 0.4.3 implementations on a
stored analytic fixture. This comparison intentionally centers X and y because
the implementations perform centering at different boundaries. VIP conventions
can still differ across software over the use of weights versus rotations. The
SR `threshold="f_test"` boundary `F(1 - alpha; N - 2, N - 3)` reproduces the
published critical values 2.04, 3.73, and 3.14 for N = 25, 10, and 12; the
`"mean"` threshold is the less conservative boundary that Farrés et al. mention
as an alternative. CovSel exposes
`scale_y=True` for the autoscaled
multi-response global selection described by Roger et al.; it does not implement
the paper's optional second per-response refinement. Its multi-response selected
order and cumulative explained X/Y diagnostics agree with the independent R
package plsVarSel 0.10.0 on a stored analytic fixture.

Classification through PLSDA and user-selected downstream classifiers for
CovSel must be reported as package extensions rather than evidence from the
regression formulations alone. SR deliberately rejects arbitrary classifier
coefficients. Its binary PLSDA path is checked against direct PLS regression on
the same explicit dummy response, isolating the target-projection calculation
from the classifier's downstream LDA decision stage. VIP and SR computed with
`chemotools.classification.PLSDA` agree with the package `PLSDA` at equal
scaling, because both use the same PLS projection.

## MI-VIF

`MIVIFSelector` follows the six steps of Cheng et al. (2022): mutual-information
ranking, then forward selection that rejects a newly added variable whenever any
selected variable exceeds VIF 10. A fixture checks that the VIF used, the diagonal
of the inverse correlation matrix, equals the literal `1 / (1 - R_i^2)` from
intercept OLS regressions. Further tests cover rejection of near-duplicate
variables, invariance to column rescaling, the `n_samples - 1` rank bound, and
constant columns. The paper does not name its mutual-information estimator, so
the k-nearest-neighbor estimator from scikit-learn is a documented choice, not a
reproduction. Classification targets and model-validated length are package
extensions.

## SPA

`SPASelector` follows Section 2.2 of Araújo et al. (2001). A fixture checks the
projection order against an independent formulation: at every step the next
wavelength is the one with the largest least-squares residual after regressing
all columns on the already selected ones, which the repeated projections of
Eq. 1 compute incrementally. A three-column hand calculation, the joint choice
of start and length from `initial_scores_`, invariance of the autoscaled
selection to column rescaling and offsets, independence of the order from `y`,
and constant columns are also tested. No author code or published numerical
order is available as an external fixture, so numerical equivalence with the
paper's MATLAB implementation is not claimed. Classification and non-MLR
validation models are package extensions.

## Model-validated sequence length

CovSel, fCovSel, and MI-VIF share one length search. A test reconstructs it by
hand, recomputing the CovSel order on every training fold and scoring every
prefix with the supplied estimator, and matches the selector's validation scores
exactly. Refactoring fCovSel onto the shared search left its supports, fold
orders, score tables, and selected hyper-parameters unchanged.

## Genetic selector source audit

`GeneticAlgorithmSelector` implements a generic chromosome wrapper with
tournament selection, single-point crossover, mutation, elitism, constraint
repair, and within-fit caching. The attached Song et al. (2016) paper primarily
defines interval combination optimization (ICO). Its GA-iPLS comparison divides
spectra into equal-width intervals, repeats a stochastic GA procedure, and
aggregates selection frequencies, while delegating procedural details to older
references. It is therefore not a sufficient primary specification for claiming
that the current generic selector is GA-iPLS.

The source chain is now explicit. Song et al. cite Goicoechea and Olivieri
(2003) as the interval-GA source, Xiaobo et al. (2007) as a spectroscopy
application, and Yun et al. (2014) for the detailed GA strategy. The verified
bibliographic identifiers are:

- Goicoechea and Olivieri (2003), DOI `10.1002/cem.812`;
- Xiaobo et al. (2007), DOI `10.1016/j.chemolab.2006.09.003`; and
- Yun et al. (2014), DOI `10.1016/j.chemolab.2013.09.007`.

Song et al.'s comparison configuration uses a population of 30, approximately
five selected interval genes in the initial chromosome, no more than 30 active
genes, 50% crossover, 1% mutation, at most 10 PLS latent variables selected by
five-fold CV, 100 fitness evaluations, and 100 independent runs. It then ranks
intervals by selection frequency and chooses the number of top-frequency
intervals that minimizes RMSECV. Those values document a future reproduction
preset; they are not defaults for the current generic search. In particular,
the present `numpy.array_split` groups are equal in column count, which only
means equal physical wavelength width when the wavelength axis is uniformly
spaced.

Deterministic tests isolate interval boundaries, decoding, lower/upper-bound
repair, crossover, mutation, tournament sparsity tie-breaking, duplicate-mask
cache reuse, and complete fixed-seed reproducibility. These validate the stated
generic implementation but do not convert it into a GA-iPLS or ICO reference
fixture. One ``fit`` call is one stochastic run; ``generations`` counts
population updates, while ``n_evaluated_subsets_`` records distinct CV fitness
evaluations after caching.

Before adding a named GA-iPLS or ICO selector, the implementation needs a
separate method specification, canonical primary sources, published-setting
fixtures, and repeated-run frequency validation. The existing short
`GASelector` alias must not be described as a reference-equivalent named method.

## fCovSel

The final mask follows the direct full-data sequence in Mishra (2022), while
configurable fold-local validation is restricted to length selection and
stability diagnostics. An equation fixture verifies that single-response
fCovSel and CovSel return the same order on the same centered data. Previously,
per-position LOO modes could repeat one feature and silently reduce the requested
support size; fold consensus is now explicitly unique and diagnostic-only.

This is an internal equivalence fixture, not a comparison with the author's
published implementation. A second analytic fixture matches the selected order
from plsVarSel 0.10.0, whose implementation uses the non-deflating
score-orthogonalization structure. The fixture does not establish universal
equivalence between absolute- and squared-covariance aggregation for every
multi-response matrix. PLSR/PLSDA scoring, multiclass indicator encoding,
custom CV splitters, and automatic length selection remain package extensions.

## CARS

The implementation follows the main CARS operations described by Li et al.
(2009): Monte Carlo sample selection, PLS coefficient importance, exponentially
decreasing forced reduction, adaptive reweighted sampling, and cross-validated
subset selection. Complete runs are repeated for stability analysis, and the
final implementation adds a documented frequency-peak aggregation step.

Classification through PLS-DA, multiclass coefficient reduction, and frequency
peak post-processing are explicit extensions. A versioned input/output fixture
from an independently executed authoritative implementation is still required
before claiming numerical equivalence.

### Author-code comparison

The source-level comparison uses Hongdong Li's `libPLS_1.95.zip`, downloaded
from the official [libPLS site](https://www.libpls.net/). The archive SHA-256 is
`322e87a6f0fa58d2a46cd5de30797cfb96974d46212ac6ce2a7339b5d95426e3`.
Its `carspls.m` is author code for the cited method, but it has not been
executed in this validation environment; source inspection is not a numerical
reference fixture.

The author code and `CARSSelector` share the EDF, coefficient-weighting, ARS,
and final cross-validated subset-comparison concepts, but their operational
contracts are not identical:

| Decision | libPLS `carspls.m` | `CARSSelector` |
|---|---|---|
| Coefficient model | Configured maximum latent variables | Components tuned on the current support |
| Regression sampling | 90%, rounded down | Configurable 80% default, rounded up |
| EDF target | Rounded from the original feature count | Rounded up with `min_features` as the endpoint |
| ARS draws | Original feature count, with replacement | Current EDF-support size, with replacement |
| Simplified mode | Full data and deterministic hard retention | Not provided |
| Subset CV | Minimum RMSECV or optional 1-SD rule | Configurable higher-is-better scorer; fewer components win ties |
| Repeated executions | External repetition | Integrated `n_runs` with frequency diagnostics |
| Final support | Best trajectory subset | Peak(s) of run-best selection frequency |

Consequently, the current class should be described as a documented CARS-based
package variant, not as a line-for-line port. A future author-code fixture must
pin the MATLAB/libPLS version, input matrix, random-number state or deterministic
mode, preprocessing, CV fold order, selected support, latent-variable count,
and complete RMSECV trajectory. Matching only the final support would not
distinguish the implementation choices above.

## BOSS

The implementation follows the bootstrapping soft shrinkage and model population
analysis procedure of Deng et al. (2016): weighted variable-space bootstrap
sampling, cross-validated population ranking, coefficient-based weight updates,
and iterative soft shrinkage. It preserves the signed regression-coefficient
aggregation used by the reference MATLAB logic and exposes repeated complete
runs as a stability diagnostic.

PLS-DA classification and repeated-run summaries are extensions. Independent
fixtures are still required to validate complete trajectories and selected
subsets against a reference execution.

### Author-code comparison

The primary paper links to Baichuan Deng's [MATLAB Central BOSS
record](https://www.mathworks.com/matlabcentral/fileexchange/52770-boss).
Version 1.0.0.0 contains `boss.m`, the supporting PLS/CV routines, and the corn
example matrices. The downloaded archive SHA-256 is
`9ddfc5ff7c5797803fbc4c39cff6d45e62d12f405c134b91f8db9da9990e6992`;
the `boss.m` SHA-256 published by MATLAB Central's artifact manifest is
`ac49f74c66c593c09b94cda9f1254cfaca2c52a673301904ab5f09812a39f0e6`.
The archive was inspected but not executed in this environment.

The regression core intentionally preserves the author code's most consequential
rules: unique variables from bootstrap draws with replacement, population sizes
of at least 500 under the automatic top-fraction schedule, normalized signed
coefficient vectors summed before taking the absolute value, the `speed`
multiplier followed by flooring, a maximum of 100 iterations, and termination
at one retained replacement. The following differences prevent a claim of exact
trajectory equivalence:

- scikit-learn PLS and CV replace the bundled MATLAB `pls.m` and `plscvfold.m`;
- NumPy's random generator cannot reproduce MATLAB's `randsample` stream from
  the same integer seed;
- arbitrary higher-is-better scorers and custom/group-aware splitters extend
  the fixed RMSECV comparison;
- invalid sub-models and degenerate coefficient weights have explicit numerical
  safeguards;
- exact global-score ties use fewer variables and then the earlier candidate,
  whereas `boss.m` chooses the last occurrence of its minimum RMSECV; and
- `min_features`, repeated complete runs, PLSDA classification, parallelism,
  and optional population storage are package extensions.

An authoritative numerical fixture should therefore run the archived MATLAB
version on one included matrix and record the MATLAB release, random state,
preprocessing, fold order, per-iteration best RMSECV, replacement counts,
weights, selected support, and optimal component count. A comparison that
ignores the trajectory would be too weak to isolate these implementation
boundaries.

## Component optimization

CARS and BOSS search PLS component counts inside each candidate-subset
evaluation. A versioned fixture stores fold-level results from a direct
scikit-learn 1.9.0 `PLSRegression` search on an analytic matrix. Both selectors
recover its winning score and component count. Isolated tests also verify that
the search starts at one component, respects the configured, subset-width, and
sample-rank upper bounds, and prefers fewer components when scores tie. Tests
assert that the final fitted model uses the selected `best_n_components_`.

This evidence validates the common component-search contract only. It is not a
reference execution of either complete stochastic algorithm and does not prove
that a particular validation design is scientifically appropriate.

## Acceptance rule

A method can be labelled reference-equivalent only after the repository contains:

1. a precise primary methodological reference;
2. a small versioned matrix with expected indices and diagnostic values;
3. documented numerical tolerances against published output or an independently
   run authoritative implementation;
4. tests for every claimed regression and classification mode;
5. fixed-seed reproducibility for stochastic methods; and
6. leakage-safe pipeline and nested-validation tests.

Until those conditions are met, reports should say that SpectralSelector follows
the cited algorithm and state its extensions, rather than claim exact numerical
reproduction.
