# Algorithms and selector families

## Manual and filter methods

`IndexSelector` applies a known set of indices. `WavelengthSelectKBest` ranks
variables independently with an F test or mutual information.

`MIVIFSelector` implements MI-VIF (Cheng et al., 2022). Variables are visited in
decreasing order of estimated mutual information $I(x_j, y)$; each one is
tentatively added to the selected set $S$, and it is kept only if every member
of the enlarged set still satisfies

```{math}
\operatorname{VIF}_i = \frac{1}{1-R_i^2} = \left(R_{SS}^{-1}\right)_{ii}
\le \tau, \qquad \tau = 10,
```

where $R_i^2$ comes from regressing $x_i$ on the other members of $S$ and
$R_{SS}$ is their correlation matrix. Mutual information is estimated with
scikit-learn's k-nearest-neighbor estimators because the paper does not specify
an estimator for continuous spectra; set `random_state` for reproducibility.
No predictive model is fitted, so the selection is model-independent and
typically small, because spectra are strongly collinear.

CovSel and MI-VIF share fCovSel's length validation. With
`n_features_to_select="auto"` and an `estimator`, the selector's own order is
recomputed on the training rows of every inner fold, every prefix length is
scored with the estimator (and `param_grid`), and the best mean score sets the
length of the full-data order. The same `scoring`, `cv` (including `-1` and
group splitters through `fit(X, y, groups)`), `preprocessor`, and `n_jobs`
options apply to all three selectors.

`CovSelSelector` repeatedly selects the variable with maximum global squared
covariance and deflates both X and Y. It supports multi-response regression and
indicator-encoded classification and follows Roger et al. (2011). Set
`scale_y=True` when multi-response targets use incomparable units and the
paper's autoscaled-Y global selection is desired. The optional per-response
refinement described in the paper is not currently implemented.

For centered residual blocks at step $k$, the implemented global criterion and
joint deflation are:

```{math}
j_k = \operatorname*{arg\,max}_{j \notin S_k}
      \left\| {Y^{(k)}}^\mathsf{T} x_j^{(k)} \right\|_2^2,
\qquad
P_k = I - \frac{x_{j_k}^{(k)} {x_{j_k}^{(k)}}^\mathsf{T}}
                    {{x_{j_k}^{(k)}}^\mathsf{T} x_{j_k}^{(k)}},
```

```{math}
X^{(k+1)} = P_k X^{(k)},
\qquad
Y^{(k+1)} = P_k Y^{(k)}.
```

The common covariance divisor does not change the maximizing index.

`FCovSelSelector` fits the paper's non-X-deflating forward sequence on all data
supplied to `fit`: maximum covariance selects an unselected variable,
Gram-Schmidt reorthogonalizes its original X column against earlier scores, and
only Y is deflated. Fold-local paths evaluate sequence lengths with PLSR or
PLSDA and a configurable scikit-learn `scoring` value. `cv` accepts an integer,
a splitter, or `-1` for leave-one-out validation. Validation chooses a length
but does not replace the paper sequence used for the final mask. The package
supports single- and multi-response regression plus indicator-encoded binary
and multiclass classification.

For one response, fCovSel chooses the unselected variable maximizing
$|{x_j}^\mathsf{T}y^{(k)}|$. The package's multi-response/classification
extension uses the row-wise sum of absolute covariances. The selected original
column is orthogonalized against earlier unit score vectors and only Y is
deflated:

```{math}
t_k = \frac{(I - T_{k-1}T_{k-1}^\mathsf{T})x_{j_k}}
           {\left\|(I - T_{k-1}T_{k-1}^\mathsf{T})x_{j_k}\right\|_2},
\qquad
Y^{(k+1)} = (I - t_k t_k^\mathsf{T})Y^{(k)}.
```

As a three-sample check, let $x_1=(-1,0,1)^\mathsf{T}$,
$x_2=(1,-2,1)^\mathsf{T}$, and $y=(-1,0,1)^\mathsf{T}$. Then the absolute
covariance scores are 2 and 0, so both CovSel and first-step fCovSel select
$x_1$. Deflating Y along $x_1$ leaves zero response residual, correctly showing
that no second independent predictive direction exists in this toy problem.

### Model-specific length and fold-wise preprocessing

The forward order depends only on covariances, but the best number of variables
depends on the model that will use them. Passing `param_grid` (any
`ParameterGrid` specification for `estimator`, which may be a `Pipeline` such as
autoscaling followed by a classifier) makes the inner validation score every
pair of prefix length $k$ and grid point $\theta$ on fold-local orders,

```{math}
(\hat{k},\hat{\theta}) = \arg\max_{k,\theta}
\frac{1}{F}\sum_{f=1}^{F} s\!\left(\mathcal{M}_\theta,\;
X^{(f)}_{\mathrm{val}}[:, o^{(f)}_{1:k}],\; y^{(f)}_{\mathrm{val}}\right),
```

where $o^{(f)}$ is the fCovSel order fitted on the training rows of fold $f$
and $\mathcal{M}_\theta$ is fitted on the same rows. Ties prefer the smaller
$k$ and then the first grid point; any parameter ending in `n_components` is
capped at $k$. The final mask is the prefix of length $\hat{k}$ of the order
fitted on all rows, and `estimator_` is refitted with $\hat{\theta}$.
`cv_results_`, `best_params_` and `best_n_features_` expose the search.

An optional `preprocessor` (for example a chemotools SG + MSC pipeline) is
re-fitted on the training rows of every fold before the fold order is
computed, and on all rows for the final order; `transform` applies the fitted
preprocessor before selecting the columns. Data-dependent preprocessing, such
as the MSC reference spectrum, therefore never uses validation rows.

## Successive projections algorithm (SPA)

`SPASelector` implements the Successive Projections Algorithm of Araújo et al.
(2001). SPA does not look for the wavelengths most related to the response;
it looks for wavelengths that carry the *least redundant* information, so that
an MLR model built on them is well conditioned. The response is used only to
decide where to start and how many wavelengths to keep.

### The projection order

Treat every wavelength as a column vector $x_j$ of the calibration matrix
(one value per sample). Starting from a given wavelength $k(0)$:

1. Project every column onto the subspace orthogonal to the last selected
   column (Eq. 1 of the paper):

   ```{math}
   P x_j = x_j - \frac{x_j^\top x_{k(n-1)}}{x_{k(n-1)}^\top x_{k(n-1)}}\, x_{k(n-1)} .
   ```

2. Select the not-yet-selected wavelength with the largest projection,
   $k(n) = \arg\max_j \lVert P x_j \rVert$.
3. Replace every $x_j$ by $P x_j$ and repeat until $N$ wavelengths are chosen.

Because the columns are replaced by their projections at every step, $P x_j$
after $n$ steps is the part of the original $x_j$ that the $n$ selected
wavelengths cannot explain by least squares. Each new wavelength is therefore
the one *least collinear* with everything already chosen. A wavelength nearly
proportional to a selected one has a tiny projection and is skipped, which is
why SPA picks spread-out, complementary wavelengths from strongly overlapping
spectra. Unlike Gram-Schmidt, the projections are used only to rank: the
selected columns themselves are the original, physically meaningful
wavelengths.

After centering, $n$ samples span at most $n - 1$ directions, so at most
$n - 1$ wavelengths can be selected; further projections are zero. The
selector caps `max_features` accordingly (and at the smallest training fold).

By default (`autoscale=True`) every column is mean-centered and divided by its
standard deviation before projecting, as in the paper's application. All
vectors then have the same norm, so the choice depends only on the angles
between wavelengths, not on their absorbance level. `autoscale=False` projects
the raw columns (Step 0 of the paper).

### Choosing the start and the number of wavelengths

The order depends on $k(0)$, and the paper chooses both $k^*(0)$ and $N^*$ by
validation: for every possible start and every $N$, build an MLR model on the
selected wavelengths and keep the pair with the smallest RMSE. With
`initial="auto"` (default) `SPASelector` does the same, using the package's
leakage-safe length search:

- For every start column, the projection order is recomputed on the training
  rows of every inner fold, every prefix is scored on the validation rows with
  `estimator`, and the mean score per length is stored as one row of
  `initial_scores_`.
- The (start, length) pair with the best mean score is kept. Ties prefer fewer
  wavelengths, then the lower start index. With an integer
  `n_features_to_select`, only the best start for that length is chosen.
- The final wavelengths are the first $N^*$ of the order computed on all rows
  from $k^*(0)$, and `estimator_` is refitted on them.

The default validation model is `LinearRegression` (the paper's MLR) for
regression and `LinearDiscriminantAnalysis` for classification. Any
scikit-learn estimator or `Pipeline`, `scoring`, `cv` (including `-1` and group
splitters), `param_grid`, and `preprocessor` work as for fCovSel. Passing an
integer `initial` fixes the start and runs only one length search.

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from spectralselector import SPASelector

pipeline = make_pipeline(
    SPASelector(max_features=15, cv=5, wavelengths=wavelengths),
    LinearRegression(),
)
pipeline.fit(X_train, y_train)

spa = pipeline[0]
spa.initial_                # chosen start column k*(0)
spa.selection_order_        # wavelengths in the order SPA picked them
spa.selected_wavelengths_   # in spectral order
spa.initial_scores_         # (start, length) validation table
```

### Cost and limitations

One projection order costs $O(N \cdot n \cdot J)$ for $n$ samples and $J$
wavelengths. With `initial="auto"` the length search is repeated for each of
the $J$ starts, that is $J \times$ folds $\times N$ model fits, the same $J^2$
growth the paper reports. For wide spectra, restrict the range, fix `initial`,
or lower `max_features`. Constant columns are never tried as starts.

Because the order ignores the response, SPA can keep wavelengths that are
weakly collinear but carry mostly noise; the validated length limits this, and
the paper notes the same trade-off. Its lower bound $N \ge$ number of analytes
is not enforced. Classification, non-MLR estimators, `param_grid`,
`preprocessor`, and `autoscale=False` are package extensions.

## Tasks, metrics, and validation folds

Every learned selector has a regression and classification path where its
mathematical definition permits one. CARS, BOSS, the genetic wrapper, and
fCovSel evaluate predictive candidates internally and therefore expose
`task`, `scoring`, and `cv`; for all four, `cv=-1` means leave-one-out. CARS,
BOSS, and fCovSel use PLSR for regression and PLSDA for classification by
default.

CovSel, VIP, SR, and univariate filters rank variables directly rather than
choosing candidates by predictive CV. Their prediction metric and fold count
belong to the surrounding `Pipeline`, `GridSearchCV`, or benchmark protocol.
This keeps a requested metric from appearing to affect an algorithm that does
not use predictive performance in its selection equation. SR classification is
currently binary and requires PLSDA because the published target-projection
ratio requires one PLS coefficient direction; arbitrary classifier
coefficients are rejected. The other selectors support multiclass
classification.

## Model-importance methods

`VIPSelector` derives variable importance in projection from fitted PLS score,
weight, and y-loading matrices. `SelectivityRatioSelector` performs target
projection along a fitted coefficient vector and ranks the explained-to-residual
variance ratio. Both clone accepted estimators before fitting. VIP's threshold
of one is conventional rather than universally significant. SR offers the
published F-test boundary with `threshold="f_test"` (Rajalahti et al., 2009;
Farrés et al., 2015, Eq. 8), retaining variables with
$\operatorname{SR}_j > F_{1-\alpha}(N-2,\,N-3)$ for $N$ training samples; at
$\alpha=0.05$ this gives 2.04, 3.73, and 3.14 for $N=25, 10, 12$, matching the
paper. The default `"mean"` threshold is the less conservative boundary Farrés et
al. mention when the F-test discards too many relevant variables. Both
`spectralselector.PLSDA` and `chemotools.classification.PLSDA` are accepted for
classification. For scaled package PLSDA, SR converts the coefficient direction back
to the caller's feature coordinates before projecting centered X.

With $p$ input variables, component-normalized X weights $w_a$, and
$S_a=\lVert t_a\rVert_2^2\lVert q_a\rVert_2^2$, VIP is calculated as:

```{math}
\operatorname{VIP}_j =
\sqrt{p\,\frac{\sum_a S_a(w_{ja}/\lVert w_a\rVert_2)^2}
                    {\sum_a S_a}}.
```

For a one-component, two-variable example with normalized weights
$(0.8,0.6)$, the scores are approximately $(1.131,0.849)$ and their mean
squared value is exactly one. This illustrates both the implementation
invariant and why the conventional threshold of one separates the two variables
in this example without proving universal significance.

For SR, let $b$ be the fitted single-response coefficient vector and $X_c$ the
centered predictors. The implemented target-projection decomposition is:

```{math}
w = b/\lVert b\rVert_2,
\quad t=X_cw,
\quad p=X_c^\mathsf{T}t/(t^\mathsf{T}t),
\quad E=tp^\mathsf{T},
\quad R=X_c-E,
\quad \operatorname{SR}_j=\frac{\lVert E_{:j}\rVert_2^2}
                               {\lVert R_{:j}\rVert_2^2}.
```

The shared sample-variance divisor cancels in the final ratio. A zero residual
therefore produces an infinite SR, while a zero target-projection direction is
undefined and raises an explicit error.

## CARS

`CARSSelector` performs repeated complete Competitive Adaptive Reweighted
Sampling searches. Within each run it:

1. draws a Monte Carlo calibration subset;
2. fits PLSR or PLS-DA on the surviving wavelengths;
3. calculates coefficient importance;
4. applies exponentially decreasing forced reduction;
5. applies adaptive reweighted sampling; and
6. cross-validates the resulting subset.

For every candidate subset, component counts from 1 through `max_components`
are evaluated. The best inner iteration is retained for each complete run.
Repeated-run occurrence frequencies are then converted to one or more final
frequency peaks. Classification and this frequency-peak aggregation are
documented extensions to Li et al. (2009).

CARS creates the task-appropriate PLS estimator internally and therefore has no
user-estimator parameter.

## BOSS

`BOSSSelector` implements Bootstrapping Soft Shrinkage as a population search.
It samples variables with replacement, evaluates PLS submodels, retains the
best population fraction, updates sampling weights from coefficient vectors,
and progressively reduces the expected subset size. Each candidate also searches
PLS component counts from 1 through `max_components`.

The default regression path follows the signed coefficient update in the BOSS
reference logic. PLS-DA classification and repeated complete runs are explicit
extensions. BOSS creates a task-appropriate PLS estimator by default; an
explicit estimator must be `PLSRegression` or `PLSDA`.

Faithful BOSS populations are computationally expensive. Values below 500 for
`n_submodels` require an explicit `top_fraction`, making reduced smoke-test
settings intentional.

## Genetic wrapper

`GeneticAlgorithmSelector` treats a mean CV score as chromosome fitness. Genes
represent individual columns or contiguous equal-column groups created with
`numpy.array_split`. Supplied wavelength values are metadata and do not change
these group boundaries, so irregularly spaced axes do not produce equal-width
physical intervals. The selector uses tournament selection, single-point
crossover, mutation, elitism, constraint repair, and a cache for repeated masks.
This is a generic optimizer, not by itself a validated GA-iPLS reproduction.

The Song et al. (2016) paper introduces ICO. Its GA-iPLS section is a short
comparison-method summary that delegates the detailed GA procedure to earlier
references. A future faithful ICO or GA-iPLS implementation must therefore be a
separately specified selector rather than an undocumented reinterpretation of
this generic class. In particular, it must define equal-width interval behavior
on irregular wavelength axes instead of inheriting the generic grouping rule.

## Interval combination optimization

`ICOSelector` implements Song et al. (2016). The spectrum is split into
`n_intervals` equal-width intervals (equal physical width when a strictly
monotonic `wavelengths` axis is given, otherwise equal column count). Each
iteration draws `n_combinations` interval combinations by weighted bootstrap
sampling, scores them by cross-validation, and sets every interval weight to its
appearance frequency among the best `best_fraction` combinations. Iteration
stops when the mean score of those best combinations no longer improves, the
best combination of that final iteration is kept, and a boundary local search
removes edge variables of the selected blocks while the CV score improves. The
paper tunes PLS components per combination; pass an estimator that does so to
reproduce it.

## Group-aware validation

CARS, BOSS, and the genetic selector accept `groups` in `fit`, allowing an
internal splitter such as `GroupKFold`:

```python
from sklearn.model_selection import GroupKFold

from spectralselector import BOSSSelector

selector = BOSSSelector(cv=GroupKFold(5), random_state=42)
selector.fit(X, y, groups=sample_groups)
```

Grouping inside the selector protects its candidate comparison. Final
performance still requires a separate group-disjoint outer evaluation.

## Method comparison

The [method taxonomy](method_taxonomy.md) groups these selectors by model
dependence and search strategy.

| Family | Predictive model | Typical cost | Main risk |
|---|---:|---:|---|
| Fixed/filter | No | Low | Missing multivariate combinations |
| MI-VIF | No | Low to medium | MI-estimator noise; very few variables kept |
| SPA | MLR/LDA (start and length) | Medium; grows with $J^2$ | Ignores y in the order; may keep noisy, uncorrelated wavelengths |
| Model importance | Yes | Medium | Model-specific ranking |
| CARS/BOSS | Internal PLS | High to very high | Search instability and internal-CV overfit |
| Genetic wrapper | User supplied | Very high | Severe selection overfit without nested CV |

Internal CV guides selection and component choice; it is not an unbiased
performance estimate. Place the complete preprocessing, selector, and final
model pipeline inside an outer CV or evaluate it once on an untouched test set.

For a candidate support $S$ and component count $a$, predictive-search methods
maximize the mean higher-is-better scikit-learn fold score:

```{math}
Q(S,a)=\frac{1}{K}\sum_{k=1}^{K}
s\!\left(\hat f_{S,a}^{(-k)}, X_S^{(k)}, y^{(k)}\right).
```

CARS and BOSS compare all valid $a=1,\ldots,\texttt{max_components}$ for each
candidate support. fCovSel compares prefixes of its forward order with PLSR or
PLSDA. The generic genetic selector evaluates its configured estimator and does
not introduce a separate component search unless that estimator contains one.
Negative-loss scorers retain scikit-learn's convention, so larger values are
always preferred.

Every successful selector fit records `selection_time_`. For reproducible timing
comparisons, keep hardware, parallel settings, search budgets, and software
versions fixed.

The dominant computational cost of a predictive search is proportional to the
number of distinct candidate supports times the number of component settings
times the number of validation folds times the estimator-fit cost. GA mask
caching reduces only repeated supports; repeated publication seeds are separate
complete searches and must remain visible in the benchmark record.
