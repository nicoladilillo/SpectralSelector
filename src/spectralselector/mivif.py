"""Mutual-information ranking with variance-inflation-factor screening (MI-VIF).

Reference
---------
Cheng, J., Sun, J., Yao, K., Xu, M., & Cao, Y. (2022). A variable selection
method based on mutual information and variance inflation factor.
Spectrochimica Acta Part A: Molecular and Biomolecular Spectroscopy, 268,
120652. https://doi.org/10.1016/j.saa.2021.120652

Paper algorithm implemented here
--------------------------------
1. Calculate the mutual information ``I(x_i, y)`` between every variable and
   the response (Eq. 2).
2. Sort variables by decreasing mutual information.
3. Walk that order with forward selection, tentatively adding one variable to
   the candidate set ``S``.
4. Calculate ``VIF_i = 1 / (1 - R_i^2)`` (Eq. 3) for every variable in ``S``.
5. If any VIF exceeds the threshold (10 in the paper), remove the newly added
   variable; otherwise keep it.
6. Repeat until every variable has been examined. The final ``S`` is the
   selection.

Package behavior and extensions
-------------------------------
The paper does not state which mutual-information estimator it applied to
continuous spectra. This implementation uses scikit-learn's k-nearest-neighbor
estimators (``mutual_info_regression`` / ``mutual_info_classif``), which
require no binning, and exposes ``n_neighbors`` and ``random_state`` for
reproducibility. The ``task="classification"`` path (MI against class labels)
and the optional ``max_features`` stopping rule are package extensions; the
paper examined every variable for single-response regression. VIF values are
computed exactly as the diagonal of the inverse correlation matrix of the
candidate set, which equals ``1 / (1 - R_i^2)`` from the OLS regression of each
member on the others with an intercept. Perfect collinearity, including a
candidate set with more members than ``n_samples - 1``, gives an infinite VIF
and rejects the candidate.

Model dependence
----------------
MI-VIF is a filter: neither ranking nor screening fits the downstream
predictive model, so by default (``n_features_to_select=None``, the published
procedure) the selection is identical for PLSR, MLR, PLS-DA or any other
estimator that later uses the selected wavelengths. As a package extension,
``n_features_to_select="auto"`` keeps only the best-validated prefix of the
accepted sequence for a user-supplied ``estimator``: MI-VIF is re-run on the
training rows of every validation fold and every prefix is scored with that
model. The ranking stays model-free; only its length becomes model-specific.

Computational cost
------------------
Mutual information is estimated once per variable, with roughly
``O(n_samples log n_samples)`` neighbor searches each. The forward pass then
examines every variable once; each test inverts a ``k x k`` correlation matrix,
where ``k`` is the current selection size, so the screening cost is
``O(n_features * (n_samples * k + k^3))``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._length_search import PrefixLengthSearchMixin

_VALID_TASKS = ("auto", "regression", "classification")


class MIVIFSelector(PrefixLengthSearchMixin, BaseWavelengthSelector):
    """Select wavelengths by mutual information with VIF collinearity screening.

    Variables are visited in decreasing order of mutual information with the
    response. A variable is kept only if, after adding it, every variable in the
    selected set still has a variance inflation factor at or below
    ``vif_threshold``. The result is a small set of individually informative
    and mutually non-collinear wavelengths, suited to MLR or PLSR.

    Parameters
    ----------
    n_features_to_select : None, int or {"auto"}, default=None
        ``None`` keeps every accepted variable (the published procedure, no
        model). An integer keeps the first ``k`` accepted variables. ``"auto"``
        chooses the number of accepted variables with ``estimator`` by
        cross-validation.
    task : {"auto", "regression", "classification"}, default="auto"
        Mutual-information estimator to use. ``"auto"`` chooses classification
        for non-numeric labels or binary/multiclass targets and regression
        otherwise (with ``n_features_to_select="auto"``, a supplied classifier
        or regressor decides). Pass ``task="regression"`` explicitly for an
        integer-valued numeric response.
    vif_threshold : float, default=10.0
        Largest acceptable VIF in the selected set. The paper uses 10, the
        usual multicollinearity rule of thumb; must be greater than 1.
    max_features : int, optional
        Stop once this many variables have been accepted. ``None`` (the
        published behavior) examines every variable.
    n_neighbors : int, default=3
        Number of neighbors used by the k-nearest-neighbor MI estimator.
    random_state : int or None, default=None
        Seed for the small noise the MI estimator adds to continuous variables
        and, with ``"auto"``, for shuffled K-fold splits. Set it for
        reproducible selections.
    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength associated with every input column.
    estimator : estimator object, optional
        Predictive model used *only* with ``n_features_to_select="auto"`` to
        choose how many accepted variables to keep. Pass the model you will fit
        on the selected wavelengths (e.g. ``LinearRegression``,
        ``PLSRegression``, a PLS-DA, or a ``Pipeline``). ``None`` creates
        one-component PLSR for regression or the package PLSDA for
        classification.
    scoring : str or callable, optional
        Scorer for length validation. Defaults to negative RMSE for regression
        and accuracy for classification.
    cv : int, CV splitter, or -1, default=5
        Inner validation folds for ``"auto"``; ``-1`` selects leave-one-out.
    param_grid : dict or list of dict, optional
        Hyper-parameter grid of ``estimator`` searched jointly with the length.
        Parameters ending in ``n_components`` are capped at the prefix length.
    preprocessor : transformer, optional
        Column-preserving transformer fitted before MI-VIF on all rows (and on
        the training rows of every validation fold); ``transform`` applies it
        before selecting columns.
    n_jobs : int, optional
        Parallel jobs over validation folds and grid points.

    Attributes
    ----------
    scores_ : ndarray of shape (n_features,)
        Estimated mutual information between each variable and y.
    mi_order_ : ndarray of shape (n_features,)
        Variable indices in the order they were examined (decreasing MI).
    full_selection_order_ : ndarray of shape (n_accepted,)
        Every variable accepted by the VIF screen on all rows, in acceptance
        order; the support is its first ``n_selected`` entries.
    vif_ : ndarray of shape (n_selected,)
        VIF of each selected variable within the final set, in
        ``selection_order_`` order.
    n_rejected_ : int
        Number of examined variables rejected by the VIF test.
    task_ : str
        Resolved task.
    selection_order_ : ndarray of shape (n_selected,)
        Selected variables in acceptance order (decreasing MI).
    validation_scores_, best_n_features_, best_params_, cv_results_, estimator_
        Length-validation results, only with ``n_features_to_select="auto"``
        (same meaning as in :class:`FCovSelSelector`).

    Notes
    -----
    Because the VIF test examines the joint candidate set, accepting an early
    variable can block a later, highly informative but collinear variable.
    Spectra are strongly collinear, so the method typically keeps few
    wavelengths (3 and 11 in the two data sets of Cheng et al.). Mutual
    information is invariant to feature scaling, and the VIF is invariant to
    affine rescaling of each column, so autoscaling X does not change the
    selection apart from estimator noise.

    References
    ----------
    Cheng, J., Sun, J., Yao, K., Xu, M., and Cao, Y. (2022). A variable
    selection method based on mutual information and variance inflation
    factor. *Spectrochimica Acta Part A*, 268, 120652.
    https://doi.org/10.1016/j.saa.2021.120652

    Examples
    --------
    >>> import numpy as np
    >>> from spectralselector import MIVIFSelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 20))
    >>> y = 2 * X[:, 3] - X[:, 11] + rng.normal(scale=0.1, size=60)
    >>> selector = MIVIFSelector(random_state=0).fit(X, y)
    >>> bool(np.isin([3, 11], selector.selected_indices_).all())
    True
    """

    def __init__(
        self,
        n_features_to_select: int | str | None = None,
        *,
        task: str = "auto",
        vif_threshold: float = 10.0,
        max_features: int | None = None,
        n_neighbors: int = 3,
        random_state: int | None = None,
        wavelengths: ArrayLike | None = None,
        estimator=None,
        scoring=None,
        cv=5,
        param_grid=None,
        preprocessor=None,
        n_jobs: int | None = None,
    ):
        """Store the MI-VIF, length-validation and preprocessing settings."""
        self.n_features_to_select = n_features_to_select
        self.task = task
        self.vif_threshold = vif_threshold
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.wavelengths = wavelengths
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.param_grid = param_grid
        self.preprocessor = preprocessor
        self.n_jobs = n_jobs

    def fit(
        self, X: ArrayLike, y: ArrayLike, groups: ArrayLike | None = None
    ) -> MIVIFSelector:
        """Rank variables by MI and keep those passing the VIF screen.

        ``groups`` is used only by group-aware ``cv`` splitters when
        ``n_features_to_select="auto"``.
        """
        if self.task not in _VALID_TASKS:
            raise ValueError("task must be 'auto', 'regression', or 'classification'.")
        X_array, y_array = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
            ensure_min_samples=3,
            multi_output=False,
            y_numeric=False,
        )
        self._validate_X(X_array, reset=True)
        self._validate_parameters()
        n_samples, n_features = X_array.shape
        limit = n_features if self.max_features is None else int(self.max_features)

        auto = self.n_features_to_select == "auto"
        if auto:
            splits = self._setup_length_search(X_array, y_array, groups)
            limit = min(limit, self._max_prefix_length(n_samples, n_features, splits))
        else:
            self.task_ = self._resolve_task(y_array)
            if self.n_features_to_select is not None:
                limit = min(limit, int(self.n_features_to_select))

        X_model = self._fit_full_preprocessor(X_array, y_array)
        self.scores_, self.mi_order_, accepted, self.n_rejected_ = self._mivif(
            X_model, y_array, limit
        )
        self.full_selection_order_ = accepted
        if (
            self.n_features_to_select not in (None, "auto")
            and accepted.size < self.n_features_to_select
        ):
            raise ValueError(
                f"MI-VIF accepted only {accepted.size} variables, fewer than "
                f"n_features_to_select={self.n_features_to_select}."
            )

        selected = accepted
        if auto:
            self._run_length_search(
                X_array, y_array, limit, splits, full_order_length=accepted.size
            )
            selected = accepted[: self._choose_length("auto")]
            self._fit_final_estimator(X_model, y_array, selected)
        self.vif_ = (
            np.ones(1)
            if selected.size == 1
            else _vif_from_correlation(np.corrcoef(X_model[:, selected], rowvar=False))
        )

        mask = np.zeros(n_features, dtype=bool)
        mask[selected] = True
        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selected,
        )
        return self

    def _fold_order(self, X: np.ndarray, y: np.ndarray, n_steps: int) -> np.ndarray:
        """Accepted MI-VIF sequence fitted on one validation training fold."""
        return self._mivif(X, y, n_steps)[2]

    def _mivif(
        self, X: NDArray[np.float64], y: NDArray, limit: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64], int]:
        """Steps 1-6: MI ranking followed by the forward VIF screen."""
        if self.task_ == "classification":
            mi = mutual_info_classif(
                X, y, n_neighbors=self.n_neighbors, random_state=self.random_state
            )
        else:
            mi = mutual_info_regression(
                X,
                np.asarray(y, dtype=np.float64),
                n_neighbors=self.n_neighbors,
                random_state=self.random_state,
            )
        # Step 1: mutual information between every variable and y (Eq. 2).
        scores = np.asarray(mi, dtype=np.float64)
        # Step 2: decreasing MI, source index as deterministic tie breaker.
        order = np.lexsort((np.arange(X.shape[1]), -scores)).astype(np.int64)
        accepted, rejected = self._forward_vif_screen(X, order, limit)
        return scores, order, accepted, rejected

    def _validate_parameters(self) -> None:
        """Validate numeric configuration before any expensive work."""
        if (
            isinstance(self.vif_threshold, bool)
            or not isinstance(self.vif_threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(self.vif_threshold)
            or self.vif_threshold <= 1.0
        ):
            raise ValueError("vif_threshold must be a finite number greater than 1.")
        for name in ("max_features", "n_features_to_select"):
            value = getattr(self, name)
            if value is None or (name == "n_features_to_select" and value == "auto"):
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or not 1 <= value <= self.n_features_in_
            ):
                raise ValueError(
                    f"{name} must be None"
                    + (", 'auto'," if name == "n_features_to_select" else "")
                    + " or between 1 and the number of features."
                )
        if (
            isinstance(self.n_neighbors, bool)
            or not isinstance(self.n_neighbors, (int, np.integer))
            or self.n_neighbors < 1
        ):
            raise ValueError("n_neighbors must be a positive integer.")

    def _resolve_task(self, y: NDArray) -> str:
        """Resolve ``task="auto"`` from the target type."""
        target_kind = type_of_target(y, raise_unknown=True)
        if self.task == "auto":
            if y.dtype.kind not in "biufc" or target_kind in {"binary", "multiclass"}:
                return "classification"
            return "regression"
        if self.task == "classification" and target_kind not in {
            "binary",
            "multiclass",
        }:
            raise ValueError("Classification requires binary or multiclass labels.")
        if self.task == "regression" and y.dtype.kind not in "biufc":
            raise ValueError("Regression requires a numeric response.")
        return self.task

    def _forward_vif_screen(
        self, X: NDArray[np.float64], order: NDArray[np.int64], limit: int
    ) -> tuple[NDArray[np.int64], int]:
        """Steps 3-6: forward selection with the multicollinearity test."""
        n_samples = X.shape[0]
        std = X.std(axis=0, ddof=1)
        # Standardized columns make Z.T @ Z / (n - 1) the correlation matrix.
        safe_std = np.where(std > 0, std, 1.0)
        Z = (X - X.mean(axis=0)) / safe_std
        threshold = float(self.vif_threshold)

        accepted: list[int] = []
        correlation = np.empty((0, 0))
        rejected = 0
        # Step 6: examine every variable in decreasing-MI order (or stop at limit).
        for index in order:
            if len(accepted) >= limit:
                break
            if std[index] <= 0:
                # A constant wavelength carries no information and has an
                # undefined VIF; never admit it.
                rejected += 1
                continue
            # Step 3: tentatively add the variable (grow the correlation matrix).
            cross = Z[:, accepted].T @ Z[:, index] / (n_samples - 1)
            candidate = np.block(
                [[correlation, cross[:, None]], [cross[None, :], np.ones((1, 1))]]
            )
            # Steps 4-5: keep it only if every member still has VIF <= threshold.
            if np.all(_vif_from_correlation(candidate) <= threshold):
                accepted.append(int(index))
                correlation = candidate
            else:
                rejected += 1
        if not accepted:
            raise ValueError("MI-VIF did not accept any variable; X may be constant.")
        return np.asarray(accepted, dtype=np.int64), rejected


def _vif_from_correlation(correlation: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return ``diag(R^-1)``, the VIF of every column, or inf if R is singular.

    For a correlation matrix ``R``, ``(R^-1)_ii = 1 / (1 - R_i^2)`` where
    ``R_i^2`` is the coefficient of determination of column ``i`` regressed on
    the remaining columns with an intercept (Cheng et al. Eq. 3).
    """
    size = correlation.shape[0]
    if size == 1:
        return np.ones(1)
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues[0] <= size * np.finfo(float).eps * eigenvalues[-1] * 1e3:
        return np.full(size, np.inf)
    return np.diag(np.linalg.inv(correlation)).copy()
