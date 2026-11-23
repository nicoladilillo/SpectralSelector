# Models, tasks, and PLS-DA

## Internal models

`CARSSelector` and `BOSSSelector` are PLS-based algorithms. With `task="auto"`,
a continuous target selects `PLSRegression` and a binary or multiclass target
selects the package's `PLSDA` classifier. Both selectors cross-validate every
feasible component count from 1 through `max_components` for each candidate
subset and retain the best score, breaking ties in favour of fewer components.

CARS deliberately has no estimator parameter. BOSS accepts an optional
estimator for configuration, but it must be `PLSRegression` or `PLSDA` and must
agree with an explicitly configured task.

## User-supplied models

- `GeneticAlgorithmSelector` accepts a cloneable predictor compatible with the
  task and scorer.
- `VIPSelector` requires fitted `x_scores_`, `x_weights_`, and `y_loadings_`
  attributes. The built-in `PLSDA` and scikit-learn `PLSRegression` satisfy this
  latent-model contract, as does `chemotools.classification.PLSDA`.
  Classification requires a PLS-DA: the fitted model must expose PLS scores,
  weights, and y-loadings.
- `SelectivityRatioSelector` requires a feature-aligned coefficient vector for
  regression and a binary PLS-DA (package or chemotools) for classification.
  Arbitrary classifier coefficients are not PLS target-projection directions.
- `MIVIFSelector` uses no predictive estimator; the downstream model only
  consumes the selected wavelengths.
- `FCovSelSelector`, and `CovSelSelector`/`MIVIFSelector` with
  `n_features_to_select="auto"`, use their `estimator` (default PLSR or PLSDA;
  any scikit-learn model, pipeline, or chemotools PLS-DA) only to choose the
  sequence length, jointly with an optional `param_grid`. The full-data order
  still defines which variables come first.
- `SPASelector` uses its `estimator` (default `LinearRegression`, the
  paper's MLR, or `LinearDiscriminantAnalysis` for classification) only to
  choose the start wavelength and the length; the projection order uses X
  alone.
- Fixed-index and univariate selectors, and CovSel/MI-VIF with a fixed count, do
  not use a predictive estimator internally.

Where an estimator is accepted, SpectralSelector clones it before fitting so
the caller's model is not modified.

## Built-in PLS-DA

SpectralSelector includes `PLSDA`, a scikit-learn classifier that combines a PLS
latent projection with linear discriminant analysis. It exposes the latent PLS
matrices needed by VIP and supports `predict`, `predict_proba`, `transform`, and
pipeline parameter tuning through `n_components`.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from spectralselector import PLSDA, VIPSelector

pipeline = Pipeline(
    [
        (
            "select",
            VIPSelector(
                estimator=PLSDA(n_components=2),
                task="classification",
                n_features_to_select=20,
            ),
        ),
        ("model", PLSDA()),
    ]
)
search = GridSearchCV(
    pipeline,
    {
        "select__estimator__n_components": [2, 3, 4],
        "model__n_components": [2, 3, 4],
    },
    scoring="balanced_accuracy",
    cv=5,
)
```

PLS-DA is not one universally standardized estimator. Reports should state the
exact projection, scaling, discriminant model, component search, and validation
scheme used.

## Package PLSDA versus chemotools PLSDA

PLS-DA is a family of classifiers rather than a single algorithm, and two
implementations are supported:

| | `spectralselector.PLSDA` | `chemotools.classification.PLSDA` |
|---|---|---|
| Latent projection | PLSR on a class-indicator matrix (one column for binary) | PLSR on a one-hot matrix (one column per class) |
| Decision rule | Linear discriminant analysis on PLS scores | Nearest class centroid (Euclidean) in PLS score space |
| Uses within-class covariance | Yes (LDA pooled covariance) | No |
| Probabilities | LDA posterior probabilities | Softmax of negative centroid distances (uncalibrated) |
| X scaling parameter | `scale_pls=True` | `scale=True` (also scales the dummy Y) |
| Availability | Always | chemotools releases after 0.5.0 |

Both expose the PLS matrices that VIP and SR need, and their PLS projections are
numerically identical for the same scaling, so **VIP and SR scores do not
depend on which classifier is chosen**. The predictions do: LDA accounts for the
shape of the class clouds in score space, while nearest-centroid treats every
score direction equally. Neither rule dominates in general. Treat the decision
rule as a hyper-parameter and compare both under the same outer validation when
the classifier itself is part of a reported result. The examples and quick start
use chemotools' PLS-DA when it is installed; CARS, BOSS, and fCovSel continue to
build the package `PLSDA` internally.
