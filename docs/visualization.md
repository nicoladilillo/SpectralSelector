# Diagnostic visualization

Every fitted selector inherits `plot_diagnostics`. The method chooses diagnostics
that reflect how that algorithm made its selection rather than displaying the
same generic chart for every method.

Install the optional plotting dependency:

```bash
python -m pip install -e '.[plot]'
```

Then fit and plot a selector:

```python
selector.fit(X_train, y_train)
figure, axes = selector.plot_diagnostics()
figure.savefig("selection_diagnostics.png", dpi=200, bbox_inches="tight")
```

The call returns the Matplotlib figure and a one-dimensional array of axes. It
does not call `show`, so it works in scripts, notebooks, headless tests, and
custom figure layouts. In a notebook, leaving `figure` as the last expression
displays it; in a script, call `matplotlib.pyplot.show()` when interactive display
is wanted.

## Plot chosen for each selector

| Selector | Main diagnostics |
|---|---|
| `IndexSelector` | Selected-variable mask |
| `WavelengthSelectKBest` | Univariate score curve and selected variables |
| `VIPSelector` | VIP curve, threshold when active, and selected variables |
| `SelectivityRatioSelector` | SR curve, effective threshold, and selected variables |
| `MIVIFSelector` | Mutual-information curve and selected variables; validation curve with `"auto"` |
| `SPASelector` | Validation curve of the chosen start and selected wavelengths |
| `CovSelSelector` | Sequential wavelength rank and cumulative explained X/Y variance; validation curve with `"auto"` |
| `FCovSelSelector` | Configured CV score by sequence length and final full-data wavelength rank |
| `CARSSelector` | Run-vote stability and mean reduction trajectory |
| `BOSSSelector` | Adaptive variable weights, CV trajectory, and retained count |
| `GeneticAlgorithmSelector` | Selected mask and best fitness by generation |

Physical wavelengths are used when the selector was constructed with
`wavelengths`. Otherwise, plots use zero-based input-column indices.

## Existing axes

Pass an axis or array of axes to integrate diagnostics into another figure:

```python
import matplotlib.pyplot as plt

figure, axes = plt.subplots(2, 1, figsize=(9, 7))
selector.plot_diagnostics(axes=axes)
```

The required number of axes is one for manual and feature-score selectors and
two for sequential and stochastic search selectors. Passing the wrong count
raises a clear `ValueError`.
