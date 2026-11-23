# SpectralSelector 0.8.0 technical benchmark

Generate this report from the repository root with:

```bash
python benchmarks/run_nested_benchmarks.py \
  --output benchmarks/reports/v0.8.0
```

The default deterministic configuration evaluates regression and classification
with nested cross-validation, full-spectrum and random-subset baselines,
selection recovery, stability, and runtime. The spectra are synthetic and the
report is an engineering reproducibility artifact, not evidence that every
algorithm is numerically equivalent to its publication. Expensive stochastic
CARS and BOSS validation and independent reference comparisons are reserved for
future scientific work.
