# Examples

The `examples/` directory contains scripts that run without private datasets.

```bash
python examples/regression_quickstart.py
python examples/classification_quickstart.py
python examples/chemotools_pipeline.py
python examples/selectivity_ratio.py
python examples/covsel_quickstart.py
python examples/cars_quickstart.py
python examples/boss_quickstart.py
python examples/hotelling_outlier_filter.py
```

The regression example demonstrates VIP inside a preprocessing and PLSR pipeline.
The classification example compares model-dependent VIP with model-free MI-VIF
in front of a PLS-DA classifier. It uses chemotools' PLS-DA
(`chemotools.classification.PLSDA`, chemotools releases after 0.5.0) and falls
back to `spectralselector.PLSDA` when that class is unavailable.
The chemotools example verifies that a chemotools transformer returning an
`ndarray` can feed a selector directly.

The CARS and BOSS examples use deliberately small stochastic searches so they
run quickly; increase the run, iteration, and population settings for real
studies. Both examples report the internally selected PLS component count.

The Hotelling T² example intentionally filters the complete raw dataset before
splitting, matching the documented lettuce reproduction protocol. It also
explains why training-fold-only outlier fitting is preferred for unbiased
predictive validation.

These examples split data before reporting test performance. Selection is fitted
only on training data. For publication-grade model comparisons, replace the
single split with repeated nested CV and retain a final external test set when one
is available.
