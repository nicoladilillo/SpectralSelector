# Documentation development

The documentation uses Sphinx. MyST renders the hand-written Markdown guides, Napoleon interprets
NumPy-style docstrings, and autodoc/autosummary generate the public API pages
from the installed Python package.

## Build the documentation

Create the repository's named Conda environment, then activate it:

```bash
conda env create -f environment.yml
conda activate wavelength_selection_package
```

`environment.yml` fixes the Python line and installs the exact Python package
versions in `requirements-dev.lock`; the package itself remains editable. To
refresh the lock after intentionally changing dependencies in `pyproject.toml`:

```bash
python -m piptools compile --extra dev --strip-extras \
  --output-file requirements-dev.lock pyproject.toml
```

Review dependency changes before committing the regenerated file. Do not use an
unfiltered `conda env export`: it includes unrelated packages and a
machine-specific prefix.

Build HTML with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The generated home page is `docs/_build/html/index.html`. Generated pages are
excluded from Git; commit the source guides, docstrings, configuration, and API
definitions instead.

Before opening a pull request, run:

```bash
python -m ruff check src tests examples benchmarks
codespell .
python -m pytest
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
python -m build
python -m twine check dist/*
```

Install the optional Git hooks with `python -m pre_commit install` and
`python -m pre_commit install --hook-type pre-push`. Commit-time hooks run Ruff;
the pre-push hook also runs the non-network test suite.

## Documentation map

- [Examples](examples.md) provides runnable usage patterns.
- [Algorithms](algorithms.md) describes selector behavior and trade-offs.
- [Visualization](visualization.md) documents algorithm-specific diagnostic plots.
- [Models and PLS-DA](models_and_plsda.md) explains estimator integration.
- [Dependencies](dependencies.md) separates runtime and optional tools.
- [Architecture](architecture.md) maps files and extension points.
- [Implementing a selector](adding_selectors.md) defines scientific, API, test,
  and Transformer-method requirements.
- [Release guide](releasing.md) defines local and hosted release steps.
- [API reference](api.rst) is generated from public class docstrings.

## Hosting

The repository includes `.readthedocs.yaml`. A Read the Docs project can build
the same strict Sphinx configuration by installing the `docs` optional
dependency group; no generated HTML needs to be committed.
