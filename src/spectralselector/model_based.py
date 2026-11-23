"""PLS model-importance wavelength selection with VIP and Selectivity Ratio.

Reference
---------
Farres, M., Platikanov, S., Tsakovski, S., & Tauler, R. (2015).
Comparison of the variable importance in projection (VIP) and of the
selectivity ratio (SR) methods for variable selection and interpretation.
Journal of Chemometrics, 29(10), 528-536.
https://doi.org/10.1002/cem.2736

Paper algorithms implemented here
---------------------------------
VIP (Eq. 1-2) combines squared, component-normalized PLS X weights using each
component's explained-Y contribution. SR (Eq. 3-7) normalizes a fitted
regression vector, performs target projection, and ranks every X variable by
target-explained divided by residual variance. The SR F-test boundary
(Eq. 8, ``F(1 - alpha; N - 2, N - 3)``) is available as ``threshold="f_test"``.

Package behavior and extensions
-------------------------------
Each selector clones and fits its model during ``fit`` so the complete
calculation can remain inside a cross-validation training fold. Users may
select by score threshold or exact count. VIP=1 is a conventional heuristic;
SR's ``threshold="mean"`` is the lower boundary that Farres et al. mention as a
less conservative alternative to the F-test. PLS-DA support is an explicit
classification extension to the regression formulation: both
:class:`spectralselector.PLSDA` and ``chemotools.classification.PLSDA`` are
accepted, while non-PLS classifier coefficient vectors are deliberately
rejected.

Computational cost
------------------
Each selector clones and fits one configured PLS-like model on the complete
selection fold, then calculates feature-wise diagnostics. Model fitting usually
dominates the linear algebra used to calculate VIP or SR.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f as f_distribution
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._modeling import resolve_task, unwrap_fitted_estimator
from .plsda import PLSDA

_PLS_ATTRIBUTES = ("x_scores_", "x_weights_", "y_loadings_")


def _default_pls_estimator(
    task: str,
    *,
    n_components: int,
    scale: bool,
):
    """Create PLSR or PLSDA when an importance model is not supplied."""
    # Preserve the historical regression default for task="auto" because an
    # integer-valued numeric regression response is indistinguishable from
    # multiclass labels without an explicit task declaration.
    if task == "classification":
        return PLSDA(n_components=n_components, scale_pls=scale)
    return PLSRegression(n_components=n_components, scale=scale)


def _is_integer(value: Any) -> bool:
    """Return whether ``value`` is an integer but not a Boolean."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_importance_inputs(
    selector: BaseWavelengthSelector, X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Validate X/y and the configuration shared by VIP and SR."""
    X_array, y_array = check_X_y(
        X,
        y,
        ensure_2d=True,
        dtype=np.float64,
        ensure_all_finite=True,
        ensure_min_samples=2,
        multi_output=True,
        y_numeric=False,
    )
    selector._validate_X(X_array, reset=True)
    if selector.estimator is None:
        max_components = min(X_array.shape[0] - 1, X_array.shape[1])
        if (
            not _is_integer(selector.n_components)
            or not 1 <= selector.n_components <= max_components
        ):
            raise ValueError(f"n_components must be between 1 and {max_components}.")
    count = selector.n_features_to_select
    if count is not None and (
        not _is_integer(count) or not 1 <= count <= selector.n_features_in_
    ):
        raise ValueError(
            "n_features_to_select must be between 1 and the number of features."
        )
    return X_array, y_array


def _fit_importance_model(
    selector: BaseWavelengthSelector,
    X: NDArray[np.float64],
    y: NDArray[Any],
) -> Any:
    """Resolve the task, then clone and fit the PLS(-DA) importance model.

    Classification is restricted to PLS-DA estimators: after fitting, the
    predictive model must expose PLS scores, weights, and y-loadings. This keeps
    ordinary linear classifiers (whose coefficients are not a PLS projection)
    out of the VIP/SR equations while accepting both the package PLSDA and
    ``chemotools.classification.PLSDA``.
    """
    estimator = selector.estimator
    if estimator is None:
        estimator = _default_pls_estimator(
            selector.task,
            n_components=int(selector.n_components),
            scale=selector.scale,
        )
    selector.task_ = resolve_task(estimator, y, selector.task)
    selector.estimator_template_ = estimator
    fitted = clone(estimator).fit(X, y)
    if selector.task_ == "classification":
        model = unwrap_fitted_estimator(fitted)
        if any(not hasattr(model, name) for name in _PLS_ATTRIBUTES):
            raise TypeError(
                f"{type(selector).__name__} classification requires a PLS-DA "
                "classifier such as spectralselector.PLSDA or "
                "chemotools.classification.PLSDA; the fitted model does not expose "
                "x_scores_, x_weights_, and y_loadings_."
            )
    return fitted


def _select_from_scores(
    scores: NDArray[np.float64],
    *,
    threshold: float,
    n_features_to_select: int | None,
    method: str,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """Build the support mask and descending-score order.

    Scores are sorted in descending order with the source index as a
    deterministic tie breaker. Transformation preserves source-column order.
    """
    n_features = scores.shape[0]
    order = np.lexsort((np.arange(n_features), -scores))
    if n_features_to_select is None:
        mask = scores >= threshold
        if not mask.any():
            raise ValueError(
                f"No variable reached the {method} threshold. Lower threshold or "
                "set n_features_to_select."
            )
        return mask, order[mask[order]]
    selected_order = order[: int(n_features_to_select)]
    mask = np.zeros(n_features, dtype=bool)
    mask[selected_order] = True
    return mask, selected_order


class VIPSelector(BaseWavelengthSelector):
    """Select variables using PLS variable importance in projection (VIP).

    Unlike a post-fit diagnostic selector, this estimator fits its own PLS model
    during ``fit``. It can therefore be placed inside a scikit-learn pipeline and
    refitted independently in every cross-validation training fold.

    Parameters
    ----------
    estimator : estimator object, optional
        Cloneable PLS-like regression or classification estimator. After fitting,
        it must expose PLS scores, weights, and y-loadings directly. Defaults to
        scikit-learn PLS regression, or
        :class:`~spectralselector.PLSDA` when ``task="classification"``.
        ``chemotools.classification.PLSDA`` is also accepted.
    task : {"auto", "regression", "classification"}, default="auto"
        Task validation applied to the supplied estimator and target.
    n_components : int, default=2
        Number of PLS components used to calculate VIP scores when
        ``estimator`` is not supplied.
    threshold : float, default=1.0
        Retain scores greater than or equal to this value when
        ``n_features_to_select`` is ``None``.
    n_features_to_select : int, optional
        If provided, retain exactly this many highest-scoring variables.
    scale : bool, default=False
        Whether the internal PLS model scales X and y. Use ``False`` when
        autoscaling is already an earlier pipeline step.
    wavelengths : array-like of shape (n_features,), optional
        Optional wavelength value for every input column.

    Attributes
    ----------
    scores_ : ndarray of shape (n_features,)
        VIP score for every original input variable.
    task_ : str
        Resolved task, ``"regression"`` or ``"classification"``.
    estimator_ : estimator object
        Clone fitted on all selection data and used to calculate VIP.

    Notes
    -----
    This implementation uses component-normalized PLS X weights. Component
    contributions are the score sum of squares multiplied by the Y-loading sum
    of squares (Farrés et al., Eq. 2). Consequently, the mean squared VIP score
    is one, up to floating point error. The conventional threshold of one is a
    ranking heuristic, not a statistical significance boundary; Farrés et al.
    show it is sensitive to non-relevant X variation and may admit false
    positives. PLS regression supports one or multiple numeric responses.
    PLS-DA-based binary and multiclass selection is a package extension.

    References
    ----------
    Farrés, M., Platikanov, S., Tsakovski, S., and Tauler, R. (2015).
    Comparison of the variable importance in projection (VIP) and of the
    selectivity ratio (SR) methods for variable selection and interpretation.
    *Journal of Chemometrics*, 29(10), 528-536.
    https://doi.org/10.1002/cem.2736
    """

    def __init__(
        self,
        estimator=None,
        *,
        task: str = "auto",
        n_components: int = 2,
        threshold: float = 1.0,
        n_features_to_select: int | None = None,
        scale: bool = False,
        wavelengths: ArrayLike | None = None,
    ):
        """Store PLS estimator and VIP threshold/count configuration."""
        self.estimator = estimator
        self.task = task
        self.n_components = n_components
        self.threshold = threshold
        self.n_features_to_select = n_features_to_select
        self.scale = scale
        self.wavelengths = wavelengths

    def fit(self, X: ArrayLike, y: ArrayLike) -> VIPSelector:
        """Fit a cloned PLS-like estimator and calculate VIP scores."""
        X_array, y_array = _validate_importance_inputs(self, X, y)
        if not np.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be a finite non-negative number.")

        # Step 1: fit the PLS(-DA) model on the selection rows.
        self.estimator_ = _fit_importance_model(self, X_array, y_array)
        # Step 2: one VIP value per wavelength (Eqs. 1-2).
        self.scores_ = self._vip_scores(self.estimator_)
        if self.scores_.shape != (self.n_features_in_,):
            raise ValueError(
                "The supplied estimator produced VIP scores for a different "
                "feature space. Its internal preprocessing must preserve columns."
            )
        # Step 3: keep VIP >= threshold (conventionally 1), or the top-k scores.
        mask, selected_order = _select_from_scores(
            self.scores_,
            threshold=float(self.threshold),
            n_features_to_select=self.n_features_to_select,
            method="VIP",
        )
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selected_order,
        )
        return self

    @staticmethod
    def _vip_scores(model) -> np.ndarray:
        """Calculate one VIP value per feature from a fitted PLS-like model.

        For component ``a``, ``SSY_a = ||t_a||² ||q_a||²`` measures its
        contribution to fitted Y. Each feature receives the SSY-weighted sum of
        its squared, component-normalized X weights. Multiplication by the
        feature count makes the mean squared VIP equal to one.
        """
        model = unwrap_fitted_estimator(model)
        if any(not hasattr(model, attribute) for attribute in _PLS_ATTRIBUTES):
            raise TypeError(
                "VIPSelector requires a PLS-like estimator exposing x_scores_, "
                "x_weights_, and y_loadings_ after fit."
            )
        scores = np.asarray(model.x_scores_, dtype=np.float64)
        weights = np.asarray(model.x_weights_, dtype=np.float64)
        loadings_y = np.asarray(model.y_loadings_, dtype=np.float64)
        if scores.ndim != 2 or weights.ndim != 2 or loadings_y.ndim != 2:
            raise ValueError(
                "PLS scores, weights, and y-loadings must be two-dimensional."
            )
        n_components = scores.shape[1]
        if weights.shape[1] != n_components or loadings_y.shape[1] != n_components:
            raise ValueError(
                "PLS scores, weights, and y-loadings disagree on component count."
            )

        # Normalize each component's X weights to unit length (w_ja / ||w_a||).
        weight_norms = np.linalg.norm(weights, axis=0)
        if np.any(weight_norms <= np.finfo(float).eps):
            raise ValueError("VIP scores are undefined for a zero-norm PLS weight.")

        # Farrés et al. Eq. (1): weight each normalized squared X weight by
        # the corresponding component's explained-Y sum of squares (Eq. 2).
        explained = np.sum(scores**2, axis=0) * np.sum(loadings_y**2, axis=0)
        total = float(np.sum(explained))
        if total <= np.finfo(float).eps:
            raise ValueError(
                "VIP scores are undefined because PLS explained no y variance."
            )
        normalized_weights = weights / weight_norms
        # VIP_j = sqrt(J * sum_a SSY_a (w_ja/||w_a||)^2 / sum_a SSY_a).
        return np.sqrt(weights.shape[0] * (normalized_weights**2 @ explained) / total)


class SelectivityRatioSelector(BaseWavelengthSelector):
    """Select variables by target-projection selectivity ratio (SR).

    SR is the ratio of target-projection explained variance to residual variance
    for each input variable. The target-projection direction is the normalized
    regression-coefficient vector of a fitted PLS-like model. Larger ratios
    indicate variables whose variation is more strongly aligned with the fitted
    response.

    Parameters
    ----------
    estimator : estimator object, optional
        Cloneable regression estimator exposing one coefficient vector through
        ``coef_``, or a binary PLS-DA classifier
        (:class:`~spectralselector.PLSDA` or
        ``chemotools.classification.PLSDA``). Defaults to scikit-learn PLS
        regression, or binary PLSDA when ``task="classification"``. Non-PLS
        classifiers are rejected because their coefficients do not define the
        published PLS target projection.
    task : {"auto", "regression", "classification"}, default="auto"
        Task validation applied to the estimator and target. Classification
        requires one target-projection direction and is therefore binary.
    n_components : int, default=2
        Number of PLS components used only when constructing the default model.
    threshold : {"f_test", "mean"} or float, default="mean"
        Retain scores greater than or equal to this value when
        ``n_features_to_select`` is ``None``. ``"f_test"`` applies the published
        boundary of Rajalahti et al. used by Farrés et al. (Eq. 8),
        ``F(1 - alpha; N - 2, N - 3)`` with ``N`` training samples.
        ``"mean"`` retains variables above the mean SR, the less conservative
        boundary also discussed by Farrés et al.
    alpha : float, default=0.05
        Significance level used only by ``threshold="f_test"``.
    n_features_to_select : int, optional
        If provided, retain exactly this many highest-scoring variables and
        ignore ``threshold`` for support-mask construction.
    scale : bool, default=False
        Whether the default PLS model scales X and y. Use ``False`` when scaling
        is already performed by an earlier pipeline step.
    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength associated with every input column.

    Attributes
    ----------
    scores_ : ndarray of shape (n_features,)
        Selectivity ratio for every original variable.
    explained_variance_ : ndarray of shape (n_features,)
        Per-variable variance explained by the target-projection component.
    residual_variance_ : ndarray of shape (n_features,)
        Per-variable variance remaining after target projection.
    target_projection_weights_ : ndarray of shape (n_features,)
        Unit-length regression vector defining the target projection.
    target_projection_loadings_ : ndarray of shape (n_features,)
        Loadings of centered X on the target-projection score.
    target_projection_scores_ : ndarray of shape (n_samples,)
        Training-sample scores along the target-projection direction.
    threshold_ : float
        Effective fitted threshold: the resolved mean, the F critical value,
        or the numeric threshold.
    task_ : str
        Resolved task, ``"regression"`` or ``"classification"``.
    estimator_ : estimator object
        Clone fitted on all selection data.

    Notes
    -----
    SR is an embedded selector. Place it inside a pipeline so its model and
    support mask are fitted independently in each cross-validation training
    fold. The current implementation supports a single response because one
    target-projection direction is not uniquely defined for multiple responses.
    The binary classification path applies the same target-projection equations
    to the PLS regression vector fitted against the dummy-coded class response.
    For two-column (one column per class) dummy coding, the two coefficient rows
    are exact negatives of each other and define the same direction. This PLS-DA
    path is a package extension.

    The F-test is conservative: Farrés et al. report that it limits false
    positives at the cost of false negatives, especially for small ``N``.

    References
    ----------
    Rajalahti, T. et al. (2009). Discriminating variable test and selectivity
    ratio plot: quantitative tools for interpretation and variable (biomarker)
    selection in complex spectral or chromatographic profiles. *Analytical
    Chemistry*, 81(7), 2581-2590.

    Farrés, M. et al. (2015). Comparison of the variable importance in
    projection (VIP) and of the selectivity ratio (SR) methods for variable
    selection and interpretation. *Journal of Chemometrics*, 29(10), 528-536.
    https://doi.org/10.1002/cem.2736
    """

    def __init__(
        self,
        estimator=None,
        *,
        task: str = "auto",
        n_components: int = 2,
        threshold: str | float = "mean",
        alpha: float = 0.05,
        n_features_to_select: int | None = None,
        scale: bool = False,
        wavelengths: ArrayLike | None = None,
    ):
        """Store estimator and selectivity-ratio selection configuration."""
        self.estimator = estimator
        self.task = task
        self.n_components = n_components
        self.threshold = threshold
        self.alpha = alpha
        self.n_features_to_select = n_features_to_select
        self.scale = scale
        self.wavelengths = wavelengths

    def fit(self, X: ArrayLike, y: ArrayLike) -> SelectivityRatioSelector:
        """Fit the target model, calculate SR scores, and learn the mask."""
        X_array, y_array = _validate_importance_inputs(self, X, y)
        if self.threshold not in ("mean", "f_test") and (
            not isinstance(self.threshold, (int, float, np.integer, np.floating))
            or isinstance(self.threshold, bool)
            or not np.isfinite(self.threshold)
            or self.threshold < 0
        ):
            raise ValueError(
                "threshold must be 'f_test', 'mean', or a finite non-negative number."
            )
        if self.threshold == "f_test":
            if not (
                isinstance(self.alpha, (float, int, np.floating, np.integer))
                and not isinstance(self.alpha, bool)
                and 0.0 < self.alpha < 1.0
            ):
                raise ValueError("alpha must be strictly between 0 and 1.")
            if X_array.shape[0] < 4:
                raise ValueError(
                    "threshold='f_test' requires at least 4 samples because its "
                    "denominator has N - 3 degrees of freedom."
                )

        # Step 1: fit the PLS(-DA) model and take its regression vector b.
        self.estimator_ = _fit_importance_model(self, X_array, y_array)
        regression_vector = self._regression_vector(
            self.estimator_,
            self.n_features_in_,
            binary_pair=self.task_ == "classification",
        )
        # Step 2: target projection and one SR per wavelength (Eqs. 3-7).
        (
            self.scores_,
            self.explained_variance_,
            self.residual_variance_,
            self.target_projection_weights_,
            self.target_projection_loadings_,
            self.target_projection_scores_,
        ) = self._selectivity_ratios(X_array, regression_vector)

        # Step 3: SR boundary: mean SR, the Eq. (8) F-test, or a fixed value.
        if self.threshold == "mean":
            # An infinite SR (zero residual) would make the mean infinite and
            # discard every finite variable, so average finite ratios only.
            finite = self.scores_[np.isfinite(self.scores_)]
            self.threshold_ = float(np.mean(finite)) if finite.size else np.inf
        elif self.threshold == "f_test":
            self.threshold_ = self.f_critical_value(X_array.shape[0], self.alpha)
        else:
            self.threshold_ = float(self.threshold)
        # Step 4: keep SR >= boundary, or the top-k scores.
        mask, selected_order = _select_from_scores(
            self.scores_,
            threshold=self.threshold_,
            n_features_to_select=self.n_features_to_select,
            method="SR",
        )
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selected_order,
        )
        return self

    @staticmethod
    def f_critical_value(n_samples: int, alpha: float = 0.05) -> float:
        """Return the SR F-test boundary of Farrés et al. Eq. (8).

        Parameters
        ----------
        n_samples : int
            Number of calibration samples ``N`` (at least 4).
        alpha : float, default=0.05
            Significance level; ``0.05`` corresponds to the 95% test.

        Returns
        -------
        f_crit : float
            Upper ``1 - alpha`` quantile of ``F(N - 2, N - 3)``.
        """
        if n_samples < 4:
            raise ValueError("The SR F-test requires at least 4 samples.")
        return float(f_distribution.ppf(1.0 - alpha, n_samples - 2, n_samples - 3))

    @staticmethod
    def _regression_vector(
        model, n_features: int, *, binary_pair: bool = False
    ) -> np.ndarray:
        """Extract one feature-aligned coefficient vector from a fitted model.

        With ``binary_pair=True``, a two-row coefficient matrix whose rows are
        collinear (one dummy column per class) is reduced to its last row.
        """
        plsda_scale = None
        if isinstance(model, PLSDA) and model.scale_pls:
            # The package PLSDA fits PLS to StandardScaler coordinates. Convert
            # its coefficient direction back to the caller's X coordinates
            # before target projection; the intercept is irrelevant after
            # centering X. chemotools' PLSDA already reports X-unit coef_.
            plsda_scale = np.asarray(model.scaler_.scale_, dtype=np.float64)
        model = unwrap_fitted_estimator(model)
        if not hasattr(model, "coef_"):
            raise TypeError(
                "SelectivityRatioSelector requires an estimator exposing coef_ "
                "after fit."
            )
        coefficients = np.squeeze(np.asarray(model.coef_, dtype=np.float64))
        vector = None
        if coefficients.ndim == 0 and n_features == 1:
            vector = coefficients.reshape(1)
        elif coefficients.ndim == 1 and coefficients.size == n_features:
            vector = coefficients.copy()
        elif coefficients.ndim >= 2:
            feature_axes = [
                axis
                for axis, size in enumerate(coefficients.shape)
                if size == n_features
            ]
            if feature_axes:
                rows = np.moveaxis(coefficients, feature_axes[-1], -1).reshape(
                    -1, n_features
                )
                if rows.shape[0] == 1 or (
                    binary_pair and rows.shape[0] == 2 and _collinear(*rows)
                ):
                    vector = rows[-1].copy()
        if vector is None:
            raise ValueError(
                "SelectivityRatioSelector requires one coefficient vector with one "
                "value per input feature; multi-response coefficients are "
                "unsupported (including multiclass PLS-DA)."
            )
        return vector if plsda_scale is None else vector / plsda_scale

    @staticmethod
    def _selectivity_ratios(
        X: np.ndarray, regression_vector: np.ndarray
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Decompose centered X and return SR values and TP diagnostics.

        The calculation follows Farrés et al. Eqs. (3)-(7): normalize the
        regression vector, project X rows to target scores, project X columns
        back to target loadings, and divide each variable's explained sum of
        squares by its residual sum of squares. Sample variances are returned as
        diagnostics; their common ``n_samples - 1`` divisor cancels in the SR.
        """
        coefficient_norm = float(np.linalg.norm(regression_vector))
        if coefficient_norm <= np.finfo(float).eps:
            raise ValueError(
                "Selectivity ratios are undefined because the fitted regression "
                "coefficient vector has zero length."
            )
        # Eq. (3): the normalized fitted regression vector defines the target
        # direction. Centering X accounts for the fitted-model intercept.
        weights = regression_vector / coefficient_norm
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        target_scores = X_centered @ weights
        score_sum_squares = float(target_scores @ target_scores)
        if score_sum_squares <= np.finfo(float).eps:
            raise ValueError(
                "Selectivity ratios are undefined because the target-projection "
                "score has zero variance."
            )

        # Eqs. (4)-(6): rank-one target reconstruction and orthogonal residual.
        loadings = X_centered.T @ target_scores / score_sum_squares
        explained_matrix = np.outer(target_scores, loadings)
        residual_matrix = X_centered - explained_matrix
        explained_variance = np.var(explained_matrix, axis=0, ddof=1)
        residual_variance = np.var(residual_matrix, axis=0, ddof=1)

        tolerance = np.finfo(float).eps * max(
            1.0,
            float(np.max(explained_variance)),
            float(np.max(residual_variance)),
        )
        # Eq. (7): the ratio is identical whether calculated from sums of
        # squares or from sample variances because both use the same divisor.
        ratios = np.divide(
            explained_variance,
            residual_variance,
            out=np.zeros_like(explained_variance),
            where=residual_variance > tolerance,
        )
        ratios[(residual_variance <= tolerance) & (explained_variance > tolerance)] = (
            np.inf
        )
        return (
            ratios,
            explained_variance,
            residual_variance,
            weights,
            loadings,
            target_scores,
        )


def _collinear(first: NDArray[np.float64], second: NDArray[np.float64]) -> bool:
    """Return whether two non-zero vectors are parallel or anti-parallel."""
    norms = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norms <= np.finfo(float).eps:
        return False
    return abs(abs(float(first @ second)) / norms - 1.0) <= 1e-8


# Alternative spelling retained for users who call the method "VIPS".
VIPSSelector = VIPSelector

# Short chemometric name retained alongside the descriptive class name.
SRSelector = SelectivityRatioSelector
