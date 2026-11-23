"""Model-validated sequence length for CovSel, fCovSel and MI-VIF."""

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import check_scoring
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import PLSDA, CovSelSelector, FCovSelSelector, MIVIFSelector


def _regression_problem(n=90, p=25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 3.0 * X[:, 4] - 2.0 * X[:, 17] + rng.normal(scale=0.2, size=n)
    return X, y


def _classification_problem(n=120, p=20, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = np.where(1.5 * X[:, 2] - X[:, 11] > 0, "healthy", "stressed")
    return X, y


def test_covsel_fixed_count_is_unchanged_and_fits_no_model():
    X, y = _regression_problem()

    selector = CovSelSelector(4).fit(X, y)

    assert selector.selection_order_.size == 4
    assert np.array_equal(selector.full_selection_order_, selector.selection_order_)
    assert not hasattr(selector, "validation_scores_")
    assert not hasattr(selector, "estimator_")


def test_covsel_auto_matches_manual_fold_local_validation():
    X, y = _regression_problem()
    cv = KFold(4, shuffle=True, random_state=3)

    selector = CovSelSelector(
        "auto", max_features=6, estimator=LinearRegression(), cv=cv
    ).fit(X, y)

    scorer = check_scoring(LinearRegression(), scoring="neg_root_mean_squared_error")
    manual = np.zeros((4, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        order = CovSelSelector(6).fit(X[train], y[train]).selection_order_
        np.testing.assert_array_equal(selector.fold_selection_orders_[fold], order)
        for length in range(1, 7):
            columns = order[:length]
            model = LinearRegression().fit(X[train][:, columns], y[train])
            manual[fold, length - 1] = scorer(model, X[test][:, columns], y[test])

    np.testing.assert_allclose(selector.validation_scores_, manual.mean(axis=0))
    best = int(np.argmax(manual.mean(axis=0))) + 1
    assert selector.best_n_features_ == best
    np.testing.assert_array_equal(
        selector.selection_order_, selector.full_selection_order_[:best]
    )
    assert set(selector.selected_indices_) >= {4, 17}
    assert selector.estimator_.coef_.shape == (best,)


def test_covsel_auto_param_grid_selects_length_and_components_jointly():
    X, y = _regression_problem()

    selector = CovSelSelector(
        "auto",
        max_features=5,
        estimator=PLSRegression(scale=False),
        param_grid={"n_components": [1, 2, 3]},
        cv=3,
        random_state=0,
    ).fit(X, y)

    assert selector.best_params_["n_components"] in {1, 2, 3}
    assert len(selector.cv_results_["params"]) == 5 * 3
    assert selector.estimator_.n_components <= selector.best_n_features_


def test_covsel_auto_classification_with_plsda_and_groups():
    X, y = _classification_problem()
    groups = np.repeat(np.arange(12), 10)

    selector = CovSelSelector(
        "auto",
        max_features=5,
        estimator=PLSDA(n_components=2),
        cv=GroupKFold(4),
        scoring="balanced_accuracy",
    ).fit(X, y, groups=groups)

    assert selector.task_ == "classification"
    assert selector.validation_scores_.shape == (5,)
    assert {2, 11} <= set(selector.selected_indices_)


def test_length_model_can_be_chemotools_plsda():
    module = pytest.importorskip("chemotools.classification")
    if not hasattr(module, "PLSDA"):
        pytest.skip("installed chemotools does not provide PLSDA")
    X, y = _classification_problem()

    for selector in (
        CovSelSelector("auto", max_features=4, estimator=module.PLSDA(n_components=3)),
        MIVIFSelector("auto", estimator=module.PLSDA(n_components=3), random_state=0),
        FCovSelSelector(max_features=4, estimator=module.PLSDA(n_components=3)),
    ):
        selector.fit(X, y)
        assert selector.task_ == "classification"
        assert selector.estimator_.n_components <= selector.best_n_features_


def test_covsel_preprocessor_is_applied_in_transform():
    X, y = _regression_problem()
    X = X * np.linspace(1.0, 50.0, X.shape[1]) + 10.0

    selector = CovSelSelector(
        "auto", max_features=4, preprocessor=StandardScaler(), cv=3, random_state=0
    ).fit(X, y)

    expected = selector.preprocessor_.transform(X)[:, selector.support_]
    np.testing.assert_allclose(selector.transform(X), expected)


def test_mivif_default_is_published_procedure_without_model():
    X, y = _regression_problem()

    selector = MIVIFSelector(random_state=0).fit(X, y)

    np.testing.assert_array_equal(
        selector.selection_order_, selector.full_selection_order_
    )
    assert not hasattr(selector, "estimator_")


def test_mivif_auto_selects_validated_prefix_of_accepted_sequence():
    X, y = _regression_problem()

    selector = MIVIFSelector(
        "auto", estimator=LinearRegression(), cv=4, random_state=0
    ).fit(X, y)

    k = selector.best_n_features_
    np.testing.assert_array_equal(
        selector.selection_order_, selector.full_selection_order_[:k]
    )
    assert selector.validation_scores_.size <= selector.full_selection_order_.size
    assert selector.fold_selection_orders_.shape[1] == selector.validation_scores_.size
    assert {4, 17} <= set(selector.selected_indices_)
    assert selector.vif_.shape == (k,)


def test_mivif_integer_count_keeps_first_accepted_variables():
    X, y = _regression_problem()
    full = MIVIFSelector(random_state=0).fit(X, y)

    first_two = MIVIFSelector(2, random_state=0).fit(X, y)

    np.testing.assert_array_equal(first_two.selection_order_, full.selection_order_[:2])


def test_mivif_integer_count_larger_than_accepted_set_is_rejected():
    rng = np.random.default_rng(4)
    base = rng.normal(size=(40, 1))
    X = base + rng.normal(scale=1e-3, size=(40, 6))  # all columns collinear
    y = base.ravel()

    with pytest.raises(ValueError, match="accepted only"):
        MIVIFSelector(3, random_state=0).fit(X, y)


@pytest.mark.parametrize(
    "selector",
    [
        CovSelSelector("auto", max_features=4, cv=3, random_state=0),
        MIVIFSelector("auto", cv=3, random_state=0),
        FCovSelSelector(max_features=4, cv=3, random_state=0),
    ],
    ids=["covsel", "mivif", "fcovsel"],
)
def test_auto_length_selectors_run_inside_outer_cross_validation(selector):
    X, y = _regression_problem()
    pipeline = Pipeline([("select", selector), ("model", LinearRegression())])

    scores = cross_val_score(
        pipeline, X, y, cv=KFold(3, shuffle=True, random_state=1), scoring="r2"
    )

    assert np.all(scores > 0.9)


@pytest.mark.parametrize(
    "selector",
    [
        CovSelSelector("auto", max_features=4, cv=3, random_state=0),
        MIVIFSelector("auto", cv=3, random_state=0),
    ],
    ids=["covsel", "mivif"],
)
def test_auto_length_selectors_plot_validation_curve(selector):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y = _regression_problem()
    figure, axes = selector.fit(X, y).plot_diagnostics()

    assert "length validation" in axes[0].get_title()
    plt.close(figure)
