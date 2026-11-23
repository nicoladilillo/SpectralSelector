import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from spectralselector import PLSDA


def test_plsda_binary_classification():
    """Test PLSDA on binary classification problem."""
    # Generate binary classification data
    X, y = make_classification(
        n_samples=100, n_features=20, n_classes=2, n_informative=10, random_state=42
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit PLSDA
    plsda = PLSDA(n_components=2)
    plsda.fit(X_train, y_train)

    # Check that it's fitted
    assert hasattr(plsda, "pls_")
    assert hasattr(plsda, "classes_")
    assert hasattr(plsda, "n_classes_")

    # Check classes
    assert plsda.n_classes_ == 2
    assert set(plsda.classes_) == {0, 1}

    # Predict
    y_pred = plsda.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Check accuracy is reasonable (better than random)
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.5  # Should be better than random guessing

    # Check predict_proba
    probs = plsda.predict_proba(X_test)
    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1

    # Check score method
    score = plsda.score(X_test, y_test)
    assert np.isclose(score, acc)


def test_plsda_multiclass_classification():
    """Test PLSDA on multiclass classification problem."""
    # Generate multiclass classification data
    X, y = make_classification(
        n_samples=150,
        n_features=20,
        n_classes=3,
        n_informative=10,
        n_redundant=0,
        random_state=42,
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Fit PLSDA
    plsda = PLSDA(n_components=3)
    plsda.fit(X_train, y_train)

    # Check that it's fitted
    assert hasattr(plsda, "pls_")
    assert hasattr(plsda, "classes_")
    assert hasattr(plsda, "n_classes_")

    # Check classes
    assert plsda.n_classes_ == 3
    assert set(plsda.classes_) == {0, 1, 2}

    # Predict
    y_pred = plsda.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Check accuracy is reasonable
    acc = accuracy_score(y_test, y_pred)
    assert acc > 0.3  # Should be better than random guessing (1/3)

    # Check predict_proba
    probs = plsda.predict_proba(X_test)
    assert probs.shape == (len(X_test), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)  # Probabilities sum to 1

    # Check score method
    score = plsda.score(X_test, y_test)
    assert np.isclose(score, acc)


def test_plsda_exposed_attributes():
    """PLSDA exposes the PLS attributes used by model-based selectors."""
    # Generate binary classification data
    X, y = make_classification(
        n_samples=50, n_features=10, n_classes=2, n_informative=5, random_state=42
    )

    # Fit PLSDA
    plsda = PLSDA(n_components=2)
    plsda.fit(X, y)

    # Check that required attributes are exposed
    assert hasattr(plsda, "x_scores_")
    assert hasattr(plsda, "x_weights_")
    assert hasattr(plsda, "y_loadings_")

    # Check shapes
    assert plsda.x_scores_.shape == (X.shape[0], plsda.n_components)
    assert plsda.x_weights_.shape == (X.shape[1], plsda.n_components)
    # LabelBinarizer uses one response column for a binary target and one
    # response column per class for a multiclass target.
    assert plsda.y_loadings_.shape == (1, plsda.n_components)


def test_plsda_in_pipeline():
    """PLSDA works in a scikit-learn pipeline with a selector."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from spectralselector import VIPSelector

    # Generate classification data
    X, y = make_classification(
        n_samples=80, n_features=15, n_classes=2, n_informative=8, random_state=42
    )

    # Create pipeline with PLSDA as final estimator
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", VIPSelector(n_components=2, n_features_to_select=5)),
            ("model", PLSDA(n_components=2)),
        ]
    )

    # Fit pipeline
    pipeline.fit(X, y)

    # Check that it works
    assert hasattr(pipeline, "predict")

    # Predict
    y_pred = pipeline.predict(X)
    assert y_pred.shape == y.shape

    # Check score
    score = pipeline.score(X, y)
    assert score >= 0.0  # Should be valid accuracy


@pytest.mark.parametrize(
    ("n_components", "error_type", "message"),
    [
        (1.5, TypeError, "must be an integer"),
        (0, ValueError, "at least 1"),
        (10, ValueError, "maximum allowed"),
    ],
)
def test_plsda_rejects_invalid_component_counts(n_components, error_type, message):
    X = np.arange(18, dtype=float).reshape(6, 3)
    y = np.asarray([0, 1, 0, 1, 0, 1])

    with pytest.raises(error_type, match=message):
        PLSDA(n_components=n_components).fit(X, y)


def test_plsda_rejects_a_single_training_class():
    with pytest.raises(ValueError, match="at least two classes"):
        PLSDA(n_components=1).fit(np.ones((6, 2)), np.zeros(6))


def test_plsda_score_scaling_decisions_and_all_exposed_properties():
    X, y = make_classification(
        n_samples=60,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        random_state=93,
    )
    model = PLSDA(
        n_components=2,
        scale_pls=False,
        scale_lda=True,
        store_scores=True,
    ).fit(X, y)

    assert model.X_scores_.shape == (60, 2)
    assert model.X_scores_scaled_.shape == (60, 2)
    assert model.transform(X[:4]).shape == (4, 2)
    assert model.predict(X[:4]).shape == (4,)
    assert model.predict_proba(X[:4]).shape == (4, 2)
    assert model.decision_function(X[:4]).shape == (4,)
    assert model.x_loadings_.shape == (8, 2)
    assert model.x_rotations_.shape == (8, 2)
    assert model.y_scores_.shape == (60, 2)
    assert model.y_weights_.shape == (1, 2)
    assert model.coef_.shape == (1, 8)
    assert model.lda_coef_.shape == (1, 2)
    assert model.lda_intercept_.shape == (1,)


def test_plsda_fitted_properties_fail_clearly_before_fit():
    model = PLSDA(n_components=1)

    for attribute in (
        "x_scores_",
        "x_weights_",
        "x_loadings_",
        "x_rotations_",
        "y_scores_",
        "y_weights_",
        "y_loadings_",
        "coef_",
        "lda_coef_",
        "lda_intercept_",
    ):
        with pytest.raises(NotFittedError):
            getattr(model, attribute)
