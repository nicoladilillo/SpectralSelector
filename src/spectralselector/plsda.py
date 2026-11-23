"""Partial Least Squares Discriminant Analysis (PLS-DA) classifier.

Reference status
----------------
PLS-DA names a family of related classification constructions rather than one
universally standardized estimator. This class therefore documents its exact
construction and does not claim equivalence with every published PLS-DA method.

Method implemented here
-----------------------
Labels are converted to a class-indicator response matrix. Partial Least Squares
Regression learns a supervised latent projection from X to those indicators,
then Linear Discriminant Analysis is fitted on the resulting PLS X scores.
Prediction transforms new X through the fitted PLS projection before LDA.

Package behavior and scope
--------------------------
The estimator follows scikit-learn cloning, prediction, probability, and
pipeline conventions and exposes fitted PLS matrices required by VIP. It is the
package's internal classification model used by CARS, BOSS, fCovSel validation,
and the default VIP/SR classification paths. Reports must state the PLS scaling,
LDA scaling, component count, class encoding, and validation scheme because
those choices materially define a PLS-DA implementation.

Computational cost
------------------
One PLS regression and one LDA model are fitted. Selector-level component
searches repeat that complete fit for every valid component count and validation
fold, so their reported search budgets must include both quantities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y, validate_data


class PLSDA(ClassifierMixin, TransformerMixin, BaseEstimator):
    """PLSR + LDA based discriminant analysis classifier.

    The classifier first applies Partial Least Squares Regression (PLSR)
    using a dummy-coded representation of the class labels as the response
    matrix. The resulting PLS X-scores are then used as input features for
    Linear Discriminant Analysis (LDA).

    Unlike PCA-LDA, the latent-variable extraction step is supervised because
    the PLS components are estimated using both X and the class-response
    matrix Y.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS latent components.

    scale_pls : bool, default=True
        Whether to standardize the input features before fitting PLS.
        Scaling is performed explicitly using StandardScaler, while the
        internal PLSRegression scaler is disabled.

    scale_lda : bool, default=False
        Whether to standardize the PLS X-scores before fitting LDA.

    store_scores : bool, default=False
        Whether to store the training PLS X-scores as ``X_scores_``.
    """

    def __init__(
        self,
        n_components: int = 2,
        scale_pls: bool = True,
        scale_lda: bool = False,
        store_scores: bool = False,
    ):
        self.n_components = n_components
        self.scale_pls = scale_pls
        self.scale_lda = scale_lda
        self.store_scores = store_scores

    def fit(self, X: ArrayLike, y: ArrayLike) -> PLSDA:
        """Fit the PLSR + LDA classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.

        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : PLSDA
            Fitted estimator.
        """
        X_array, y_array = check_X_y(
            X,
            y,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
        )

        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.unique(y_array)
        self.n_classes_ = len(self.classes_)

        if self.n_classes_ < 2:
            raise ValueError(
                f"PLSDA requires at least two classes. Found {self.n_classes_} class."
            )

        if not isinstance(self.n_components, (int, np.integer)):
            raise TypeError(
                "n_components must be an integer. "
                f"Got {type(self.n_components).__name__}."
            )

        if self.n_components < 1:
            raise ValueError(
                f"n_components must be at least 1. Got {self.n_components}."
            )

        # ------------------------------------------------------------------
        # Encode class labels as the PLS response Y
        # ------------------------------------------------------------------
        self.label_binarizer_ = LabelBinarizer()
        Y = self.label_binarizer_.fit_transform(y_array)

        # LabelBinarizer returns (n_samples, 1) for binary problems and
        # (n_samples, n_classes) for multiclass problems.
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        Y = Y.astype(np.float64, copy=False)

        # ------------------------------------------------------------------
        # Scale X before PLS if requested
        # ------------------------------------------------------------------
        if self.scale_pls:
            self.scaler_ = StandardScaler()
            X_pls = self.scaler_.fit_transform(X_array)
        else:
            X_pls = X_array

        # Because PLS centers the data, the effective maximum rank cannot
        # normally exceed n_samples - 1.
        max_components = min(
            X_pls.shape[1],
            X_pls.shape[0] - 1,
        )

        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} is too large for "
                f"the provided data. The maximum allowed value is "
                f"{max_components}."
            )

        # ------------------------------------------------------------------
        # Fit PLS regression
        #
        # scale=False because X scaling is handled explicitly above.
        # This also avoids automatically rescaling the dummy-coded Y matrix.
        # ------------------------------------------------------------------
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=False,
        )

        self.pls_.fit(X_pls, Y)

        # PLS latent X-scores
        X_scores = self.pls_.transform(X_pls)

        # ------------------------------------------------------------------
        # Optional scaling of PLS scores before LDA
        # ------------------------------------------------------------------
        if self.scale_lda:
            self.lda_scaler_ = StandardScaler()
            X_lda = self.lda_scaler_.fit_transform(X_scores)
        else:
            X_lda = X_scores

        # ------------------------------------------------------------------
        # Fit LDA on PLS latent scores
        # ------------------------------------------------------------------
        self.lda_ = LinearDiscriminantAnalysis()
        self.lda_.fit(X_lda, y_array)

        if self.store_scores:
            self.X_scores_ = X_scores

            if self.scale_lda:
                self.X_scores_scaled_ = X_lda

        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_scores = self._transform_to_scores(X)

        if self.scale_lda:
            X_scores = self.lda_scaler_.transform(X_scores)

        return self.lda_.predict(X_scores)

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict class probabilities using the LDA classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            LDA class probabilities.
        """
        X_scores = self._transform_to_scores(X)

        if self.scale_lda:
            X_scores = self.lda_scaler_.transform(X_scores)

        return self.lda_.predict_proba(X_scores)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Compute the LDA decision function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray
            LDA decision-function values.
        """
        X_scores = self._transform_to_scores(X)

        if self.scale_lda:
            X_scores = self.lda_scaler_.transform(X_scores)

        return self.lda_.decision_function(X_scores)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Transform samples into the PLS latent-variable space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            PLS X-scores.
        """
        return self._transform_to_scores(X)

    def _transform_to_scores(
        self,
        X: ArrayLike,
    ) -> NDArray[np.float64]:
        """Apply preprocessing and transform X to PLS X-scores."""
        check_is_fitted(self, ["pls_", "lda_"])

        X_array = validate_data(
            self,
            X,
            reset=False,
            ensure_2d=True,
            dtype=np.float64,
            ensure_all_finite=True,
        )

        X_pls = self.scaler_.transform(X_array) if self.scale_pls else X_array

        return self.pls_.transform(X_pls)

    # ------------------------------------------------------------------
    # PLS properties
    # ------------------------------------------------------------------

    @property
    def x_scores_(self) -> NDArray[np.float64]:
        """PLS X-scores from the training data."""
        check_is_fitted(self, "pls_")
        return self.pls_.x_scores_

    @property
    def x_weights_(self) -> NDArray[np.float64]:
        """PLS X-weights."""
        check_is_fitted(self, "pls_")
        return self.pls_.x_weights_

    @property
    def x_loadings_(self) -> NDArray[np.float64]:
        """PLS X-loadings."""
        check_is_fitted(self, "pls_")
        return self.pls_.x_loadings_

    @property
    def x_rotations_(self) -> NDArray[np.float64]:
        """PLS X-rotations."""
        check_is_fitted(self, "pls_")
        return self.pls_.x_rotations_

    @property
    def y_scores_(self) -> NDArray[np.float64]:
        """PLS Y-scores from the training data."""
        check_is_fitted(self, "pls_")
        return self.pls_.y_scores_

    @property
    def y_weights_(self) -> NDArray[np.float64]:
        """PLS Y-weights."""
        check_is_fitted(self, "pls_")
        return self.pls_.y_weights_

    @property
    def y_loadings_(self) -> NDArray[np.float64]:
        """PLS Y-loadings."""
        check_is_fitted(self, "pls_")
        return self.pls_.y_loadings_

    @property
    def coef_(self) -> NDArray[np.float64]:
        """PLS regression coefficients."""
        check_is_fitted(self, "pls_")
        return self.pls_.coef_

    # ------------------------------------------------------------------
    # LDA properties
    # ------------------------------------------------------------------

    @property
    def lda_coef_(self) -> NDArray[np.float64]:
        """LDA coefficients in the PLS latent space."""
        check_is_fitted(self, "lda_")
        return self.lda_.coef_

    @property
    def lda_intercept_(self) -> NDArray[np.float64]:
        """LDA intercept terms."""
        check_is_fitted(self, "lda_")
        return self.lda_.intercept_
