# SpectralSelector

`spectralselector` provides NumPy-first, scikit-learn-compatible wavelength
selectors for chemometric workflows. Selectors learn column indices during
`fit`, expose a support mask and selected wavelengths, and transform spectra to
the selected variables without coupling the package to a preprocessing library.

This design accepts the NumPy arrays returned by `chemotools` and places
selection safely inside scikit-learn pipelines and cross-validation.

Scientific equivalence is tracked separately in the
[algorithm-validation guide](algorithm_validation.md). The package is alpha
software while independent numerical fixtures are still being assembled.

```{toctree}
:maxdepth: 2
:caption: User guide

examples
algorithms
method_taxonomy
visualization
validation
algorithm_validation
models_and_plsda
dependencies
architecture
adding_selectors
development
publication_plan
releasing
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
