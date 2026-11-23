# Dependencies and development files

## Runtime dependencies

| Dependency | Minimum | Purpose |
|---|---:|---|
| Python | 3.10 | Supported runtime and typing syntax |
| Joblib | 1.3 | Parallel candidate evaluation |
| NumPy | 1.26 | Arrays, masks, linear algebra, and random generators |
| pandas | 2.1 | CARS and BOSS diagnostic tables |
| SciPy | 1.11 | Statistical routines and frequency-peak detection |
| scikit-learn | 1.6 | Estimator API, PLS, scoring, CV, and pipelines |
| tqdm | 4.66 | BOSS progress reporting |

These requirements are declared once in `pyproject.toml`. The
`environment.yml` file installs the editable development extra from that source
rather than duplicating a manually maintained package list.

## Optional groups

- `plot` installs Matplotlib for `plot_diagnostics`.
- `chemotools` installs chemotools for preprocessing interoperability, the benchmark
  coffee and fermentation datasets, and the Hotelling T² example. Its PLS-DA classifier
  (`chemotools.classification.PLSDA`), used by the classification examples and
  benchmarks, first appears after chemotools 0.5.0; with older releases those
  scripts fall back to `spectralselector.PLSDA` and record which class ran.
- `test` installs the complete test stack.
- `docs` installs Sphinx, MyST Parser, and the PyData Sphinx Theme.
- `dev` combines release, test, documentation, formatting, and pre-commit tools.

Plotting imports remain lazy. Chemotools is imported only by its dataset adapters
or examples, so it is not needed for the core selector API.

## Parallel execution

CARS can parallelize cross-validation work, BOSS parallelizes population
submodels, and the genetic selector parallelizes candidate chromosomes. Avoid
running both an outer `GridSearchCV` and an inner selector with unrestricted
workers; set one level to `n_jobs=1` to prevent oversubscription.

## Reproducibility

Use `environment.yml` to create the named development environment. Release
artifacts are reproducible from `pyproject.toml`; a frozen platform-specific
environment export should be attached to an experiment or release only when
exact transitive versions are scientifically relevant.
