# Lettuce water-stress benchmark analysis

This analysis uses the user-supplied supplementary CSV from Dilillo et al.
(2026). SpectralSelector does not redistribute the observations. Although the
source file covers 350-2,500 nm, every result below uses the paper's retained
range of 400-2,500 nm: 2,101 wavelength variables for 216 spectra.

## Two validation questions

Two protocols are intentionally reported separately.

1. **Cycle-transfer evaluation.** One complete experimental cycle is held out.
   Scaling, wavelength selection, model tuning, and fitting use only the other
   cycles. This asks whether a model transfers to a different cycle.
2. **Paper-preprocessing sensitivity.** A 70:30 class-and-cycle-stratified
   random split evaluates SG/SG1, MSC/SNV, and outlier removal. This is closer
   to the article's split and generally gives an easier prediction problem.

Accuracies from these protocols must not be compared as if they were repeated
estimates of the same quantity.

## Cycle-transfer baseline

This baseline uses the 400-2,500 nm raw signals followed by training-fold
autoscaling. No SG, scatter correction, or outlier removal is applied.

| Method | Mean wavelengths | Accuracy | Macro F1 | Mask Jaccard |
|---|---:|---:|---:|---:|
| Full spectrum | 2,101.0 | **0.657** | **0.631** | 1.000 |
| Univariate F | 10.0 | 0.495 | 0.472 | 0.000 |
| CovSel | 11.7 | 0.639 | 0.607 | 0.014 |
| Selectivity ratio | 8.3 | 0.505 | 0.493 | 0.000 |

CovSel retained approximately 0.56% of the wavelengths and lost 0.019 absolute
accuracy relative to the full spectrum. Its exact masks were highly unstable
between training-cycle pairs. The full-spectrum confusion matrix was
`[[58, 50], [24, 84]]`; CovSel produced `[[57, 51], [27, 81]]`, using the class
order Controlled, Water Stressed. Both models recognized stressed samples more
reliably than controlled samples.

## Spectral preprocessing

The paper describes four branches:

- **SG+MSC:** Savitzky-Golay smoothing, then multiplicative scatter correction.
- **SG+SNV:** Savitzky-Golay smoothing, then standard normal variate.
- **SG1+MSC:** first-derivative Savitzky-Golay, then MSC.
- **SG1+SNV:** first-derivative Savitzky-Golay, then SNV.

SNV subtracts each spectrum's own mean and divides by that spectrum's standard
deviation. It does not estimate a shared training or test mean. MSC regresses
each spectrum against a reference spectrum, so the reference definition does
matter:

- `global` uses the complete dataset mean before splitting, reproducing the
  order described in the paper but exposing the test set to preprocessing;
- `training` learns the mean from training spectra and applies that fixed
  reference to test spectra;
- `batch` computes independent train and test means, as a transductive
  sensitivity analysis rather than a deployable single-sample workflow.

The article does not report the SG window length or polynomial order. The
analysis uses window 11 and polynomial order 3 as an explicit assumption. This
combination reproduces the reported outlier totals, but several other settings
do too, so matching the counts does not prove that the parameters are original.

## Outlier removal

For every preprocessing branch, ten PCA scores are calculated and squared
Mahalanobis distances above the 95% chi-squared threshold (18.31) are flagged.
With global preprocessing, the implementation reproduces the paper exactly:

| Preprocessing | Removed | Paper |
|---|---:|---:|
| SG+MSC | 18 | 18 |
| SG+SNV | 20 | 20 |
| SG1+MSC | 24 | 24 |
| SG1+SNV | 23 | 23 |

In the prediction-safe protocol, the detector is fitted on training spectra
only. Training outliers are removed, while flagged test samples remain in the
test set. Removing difficult test observations would make the reported test
accuracy optimistic.

## Preprocessing sensitivity results

The following values were produced with the earlier logistic-regression
classifier, now available as `--classifier logistic`; the scripts default to
PLS-DA. The SR column came from an earlier configuration that applied SR to
logistic-regression coefficients, which SpectralSelector now rejects because
those coefficients are not a PLS target projection; reruns report SR through
`sr_plsda`. They are the best result within each selector family across the
four preprocessing branches. They come from one
fixed random split and should be treated as sensitivity results, not confidence
intervals.

| MSC scope | Outliers | Full | Univariate F | CovSel | SR |
|---|---|---:|---:|---:|---:|
| Global | Kept | **0.877** | 0.585 | 0.754 | 0.569 |
| Global | Removed | **0.850** | 0.695 | 0.831 | 0.672 |
| Training | Kept | **0.877** | 0.600 | 0.754 | 0.569 |
| Training | Removed | **0.862** | 0.600 | 0.738 | 0.600 |
| Separate batches | Kept | **0.877** | 0.615 | 0.754 | 0.631 |
| Separate batches | Removed | **0.862** | 0.646 | 0.738 | 0.631 |

Outlier removal does not consistently improve prediction. It helps some
compact selectors in global reproduction mode, but reduces the best
full-spectrum accuracy from 0.877 to 0.850 globally and to 0.862 under the
training-only protocol. This supports treating outlier removal as a model
choice that requires validation, not as an automatically beneficial cleaning
step.

The MSC reference also has a large effect. Under separate batch means,
full-spectrum SG+MSC accuracy is 0.569 without outlier removal, compared with
0.800 using a training reference and 0.877 using a global reference. SG1+MSC is
less sensitive in this split. A deployable model should therefore store and
reuse the training MSC reference rather than recomputing it from an arbitrary
test batch.

## Interpretation limits

The recorded preprocessing sensitivity results use logistic regression to test
the SpectralSelector selectors; they do not reproduce the paper's PLS-DA, BOSS,
GA-iPLS, or iGA-BOSS results. Rerunning with the default `--classifier plsda`
uses a PLS-DA classifier as in the paper, but the selector set still differs. The fixed random split and SG-parameter ambiguity
also prevent an exact numerical reproduction claim. The most defensible next
step is repeated nested validation in which preprocessing, outlier detection,
wavelength selection, and model tuning are all learned within each training
fold.
