import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import PLSDA, SelectivityRatioSelector, SRSelector, VIPSelector


def _chemotools_reference_case():
    """Recreate the analytic inputs stored with the chemotools 0.4.3 outputs."""
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "reference"
        / "chemotools_0_4_3_vip_sr.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = np.arange(1.0, 19.0)
    X = np.column_stack(
        (
            np.sin(rows * 0.31),
            np.cos(rows * 0.17),
            np.sin(rows * 0.73) + 0.1 * rows,
            np.cos(rows * 0.47),
            (rows.astype(int) % 5) - 2.0,
            np.sin(rows * 0.11) * np.cos(rows * 0.23),
        )
    )
    X -= X.mean(axis=0)
    y = 1.5 * X[:, 1] - 0.8 * X[:, 4] + 0.3 * X[:, 5] + 0.05 * np.sin(rows * 1.3)
    y -= y.mean()
    return fixture, X, y


def test_vip_matches_published_equation_fixture_and_unit_mean_square():
    model = SimpleNamespace(
        x_scores_=np.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, -2.0]]),
        x_weights_=np.asarray([[1.0, 1.0], [1.0, -1.0], [0.0, 1.0]]),
        y_loadings_=np.asarray([[2.0, 1.0]]),
    )

    scores = VIPSelector._vip_scores(model)

    np.testing.assert_allclose(scores, np.sqrt([1.25, 1.25, 0.5]))
    assert np.mean(scores**2) == pytest.approx(1.0)


def test_vip_matches_versioned_chemotools_reference_fixture():
    fixture, X, y = _chemotools_reference_case()
    model_parameters = fixture["source"]["model_parameters"]

    selector = VIPSelector(
        estimator=PLSRegression(**model_parameters),
        task="regression",
        threshold=fixture["threshold"],
    ).fit(X, y)

    np.testing.assert_allclose(
        selector.scores_,
        fixture["vip_scores"],
        rtol=0.0,
        atol=fixture["absolute_tolerance"],
    )
    assert selector.selected_indices_.tolist() == fixture["vip_selected_indices"]


def test_vip_rejects_zero_norm_component_weight():
    model = SimpleNamespace(
        x_scores_=np.asarray([[1.0], [-1.0]]),
        x_weights_=np.zeros((2, 1)),
        y_loadings_=np.ones((1, 1)),
    )

    with pytest.raises(ValueError, match="zero-norm"):
        VIPSelector._vip_scores(model)


@pytest.mark.parametrize("scale", [False, True], ids=["centered", "autoscaled"])
@pytest.mark.parametrize("multiresponse", [False, True])
def test_vip_component_contributions_cover_pls_response_and_scaling_modes(
    scale, multiresponse
):
    rng = np.random.default_rng(92)
    X = rng.normal(size=(80, 9)) * np.linspace(0.5, 3.0, 9)
    first = 3.0 * X[:, 1] - X[:, 6] + rng.normal(scale=0.1, size=80)
    y = (
        np.column_stack((first, -2.0 * X[:, 3] + rng.normal(scale=0.1, size=80)))
        if multiresponse
        else first
    )
    estimator = PLSRegression(n_components=3, scale=scale)

    selector = VIPSelector(
        estimator=estimator,
        task="regression",
        n_features_to_select=4,
    ).fit(X, y)

    assert selector.scores_.shape == (X.shape[1],)
    assert np.isfinite(selector.scores_).all()
    assert np.mean(selector.scores_**2) == pytest.approx(1.0)


@pytest.mark.parametrize("scale_pls", [False, True], ids=["centered", "autoscaled"])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_vip_plsda_extension_supports_binary_and_multiclass_targets(
    scale_pls, n_classes
):
    rng = np.random.default_rng(93 + n_classes)
    X = rng.normal(size=(120, 10)) * np.linspace(0.5, 2.5, 10)
    if n_classes == 2:
        y = np.where(2.0 * X[:, 1] - X[:, 7] > 0, "positive", "negative")
    else:
        logits = np.column_stack((2.0 * X[:, 1], -X[:, 4], 1.5 * X[:, 8]))
        y = np.asarray(["first", "second", "third"])[np.argmax(logits, axis=1)]

    selector = VIPSelector(
        estimator=PLSDA(n_components=2, scale_pls=scale_pls),
        task="classification",
        n_features_to_select=4,
    ).fit(X, y)

    assert selector.task_ == "classification"
    assert isinstance(selector.estimator_, PLSDA)
    assert selector.selected_indices_.size == 4
    assert np.isfinite(selector.scores_).all()
    assert np.mean(selector.scores_**2) == pytest.approx(1.0)


def test_vip_automatically_creates_plsda_for_classification():
    rng = np.random.default_rng(98)
    X = rng.normal(size=(90, 9))
    logits = np.column_stack((2 * X[:, 1], -X[:, 4], 1.5 * X[:, 7]))
    y = np.asarray(["a", "b", "c"])[np.argmax(logits, axis=1)]

    selector = VIPSelector(
        task="classification",
        n_components=2,
        n_features_to_select=4,
    ).fit(X, y)

    assert isinstance(selector.estimator_, PLSDA)
    assert selector.task_ == "classification"
    assert selector.selected_indices_.size == 4


def test_vip_rejects_non_plsda_classifier():
    rng = np.random.default_rng(102)
    X = rng.normal(size=(80, 7))
    y = (X[:, 1] > 0).astype(int)

    with pytest.raises(TypeError, match=r"classification requires.*PLSDA"):
        VIPSelector(
            estimator=LogisticRegression(random_state=4),
            task="classification",
            n_features_to_select=3,
        ).fit(X, y)


def test_vip_selector_fits_internal_model_and_selects_requested_count():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(70, 10))
    y = 4 * X[:, 2] - 2 * X[:, 7] + rng.normal(scale=0.2, size=70)
    selector = VIPSelector(n_components=2, n_features_to_select=3).fit(X, y)

    assert selector.selected_indices_.shape == (3,)
    assert selector.scores_.shape == (10,)
    assert hasattr(selector, "estimator_")
    assert not hasattr(clone(selector), "estimator_")


def test_vip_selector_runs_inside_cross_validated_pipeline():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(60, 12))
    y = X[:, 1] - X[:, 8] + rng.normal(scale=0.2, size=60)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", VIPSelector(n_components=2, n_features_to_select=5)),
            ("model", PLSRegression(n_components=2)),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=3),
        scoring="neg_root_mean_squared_error",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_vip_selector_accepts_supplied_pls_estimator():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(50, 9))
    y = X[:, 2] + rng.normal(scale=0.1, size=50)
    supplied = PLSRegression(n_components=1, scale=False)

    selector = VIPSelector(
        estimator=supplied,
        task="regression",
        n_features_to_select=2,
    ).fit(X, y)

    assert selector.estimator_template_ is supplied
    assert selector.estimator_ is not supplied
    assert selector.selected_indices_.size == 2


def test_vip_tied_scores_use_source_index_as_selection_tie_break():
    rng = np.random.default_rng(96)
    signal = rng.normal(size=60)
    X = np.column_stack((signal, signal.copy(), rng.normal(size=(60, 3))))
    y = 2.0 * signal

    selector = VIPSelector(
        n_components=1,
        n_features_to_select=1,
        scale=False,
    ).fit(X, y)

    assert selector.scores_[0] == pytest.approx(selector.scores_[1], abs=1e-14)
    assert selector.selection_order_.tolist() == [0]


def test_vip_rejects_constant_response_but_handles_nearly_constant_response():
    rng = np.random.default_rng(97)
    X = rng.normal(size=(50, 6))

    with (
        pytest.warns(UserWarning, match="y residual is constant"),
        pytest.raises(ValueError, match="VIP scores are undefined"),
    ):
        VIPSelector(n_components=1, n_features_to_select=2).fit(X, np.ones(50))

    y_nearly_constant = 1.0 + 1e-10 * X[:, 2]
    selector = VIPSelector(
        n_components=1,
        n_features_to_select=2,
        scale=True,
    ).fit(X, y_nearly_constant)

    assert np.isfinite(selector.scores_).all()
    assert np.mean(selector.scores_**2) == pytest.approx(1.0)


def test_selectivity_ratio_selects_informative_variables_and_diagnostics():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(120, 12))
    y = 5 * X[:, 3] - 3 * X[:, 9] + rng.normal(scale=0.1, size=120)

    selector = SelectivityRatioSelector(
        n_components=2,
        n_features_to_select=2,
    ).fit(X, y)

    assert set(selector.selected_indices_) == {3, 9}
    assert selector.scores_.shape == (12,)
    assert selector.explained_variance_.shape == (12,)
    assert selector.residual_variance_.shape == (12,)
    assert selector.target_projection_scores_.shape == (120,)
    assert np.isclose(np.linalg.norm(selector.target_projection_weights_), 1.0)
    assert np.allclose(
        np.var(X, axis=0, ddof=1),
        selector.explained_variance_ + selector.residual_variance_,
    )


def test_selectivity_ratio_matches_target_projection_equation_fixture():
    X = np.asarray(
        [
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
        ]
    )

    ratios, explained, residual, weights, loadings, scores = (
        SelectivityRatioSelector._selectivity_ratios(X, np.asarray([1.0, 1.0]))
    )

    root_half = np.sqrt(0.5)
    np.testing.assert_allclose(weights, [root_half, root_half])
    np.testing.assert_allclose(loadings, [root_half, root_half])
    np.testing.assert_allclose(scores, [-root_half, root_half, root_half, -root_half])
    np.testing.assert_allclose(explained, [1.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(residual, [1.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(ratios, [1.0, 1.0])


def test_selectivity_ratio_matches_versioned_chemotools_reference_fixture():
    fixture, X, y = _chemotools_reference_case()
    model_parameters = fixture["source"]["model_parameters"]

    selector = SelectivityRatioSelector(
        estimator=PLSRegression(**model_parameters),
        task="regression",
        threshold=fixture["threshold"],
    ).fit(X, y)

    np.testing.assert_allclose(
        selector.scores_,
        fixture["sr_scores"],
        rtol=0.0,
        atol=fixture["absolute_tolerance"],
    )
    assert selector.selected_indices_.tolist() == fixture["sr_selected_indices"]


def test_selectivity_ratio_is_invariant_to_direction_sign_and_x_translation():
    rng = np.random.default_rng(98)
    X = rng.normal(size=(70, 5))
    coefficient = np.asarray([1.0, -0.5, 0.2, 0.0, 0.7])

    original = SelectivityRatioSelector._selectivity_ratios(X, coefficient)
    reversed_direction = SelectivityRatioSelector._selectivity_ratios(X, -coefficient)
    translated = SelectivityRatioSelector._selectivity_ratios(
        X + np.asarray([10.0, -3.0, 4.0, 2.0, 8.0]), coefficient
    )

    for index in (0, 1, 2):
        np.testing.assert_allclose(original[index], reversed_direction[index])
        np.testing.assert_allclose(original[index], translated[index])
    np.testing.assert_allclose(original[3], -reversed_direction[3])
    np.testing.assert_allclose(original[5], -reversed_direction[5])


def test_selectivity_ratio_is_invariant_to_consistent_feature_rescaling():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(80, 4))
    coefficient = np.asarray([1.5, -0.3, 0.8, 0.2])
    feature_scale = np.asarray([0.25, 2.0, 5.0, 1.5])

    original = SelectivityRatioSelector._selectivity_ratios(X, coefficient)
    rescaled = SelectivityRatioSelector._selectivity_ratios(
        X * feature_scale,
        coefficient / feature_scale,
    )

    np.testing.assert_allclose(original[0], rescaled[0])
    assert not np.isnan(rescaled[0]).any()


def test_selectivity_ratio_mean_threshold_and_alias():
    rng = np.random.default_rng(32)
    X = rng.normal(size=(80, 9))
    y = 2 * X[:, 4] + rng.normal(scale=0.2, size=80)

    selector = SRSelector(n_components=1, threshold="mean").fit(X, y)

    assert isinstance(selector, SelectivityRatioSelector)
    assert np.isclose(selector.threshold_, np.mean(selector.scores_))
    assert np.array_equal(selector.support_, selector.scores_ >= selector.threshold_)


def test_selectivity_ratio_runs_inside_cross_validated_pipeline():
    rng = np.random.default_rng(33)
    X = rng.normal(size=(75, 14))
    y = X[:, 2] - X[:, 11] + rng.normal(scale=0.2, size=75)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "select",
                SelectivityRatioSelector(
                    n_components=2,
                    n_features_to_select=5,
                ),
            ),
            ("model", PLSRegression(n_components=2, scale=False)),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=KFold(3, shuffle=True, random_state=3),
        scoring="neg_root_mean_squared_error",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()


def test_selectivity_ratio_rejects_non_plsda_classifier():
    rng = np.random.default_rng(34)
    X = rng.normal(size=(100, 10))
    y = (2 * X[:, 1] - X[:, 7] > 0).astype(int)

    with pytest.raises(TypeError, match=r"classification requires.*PLSDA"):
        SelectivityRatioSelector(
            estimator=LogisticRegression(random_state=4),
            task="classification",
            n_features_to_select=3,
        ).fit(X, y)


def test_selectivity_ratio_automatically_creates_plsda_for_binary_classification():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(100, 10))
    y = np.where(2 * X[:, 2] - X[:, 8] > 0, "positive", "negative")

    selector = SelectivityRatioSelector(
        task="classification",
        n_components=2,
        n_features_to_select=3,
    ).fit(X, y)

    assert isinstance(selector.estimator_, PLSDA)
    assert selector.task_ == "classification"
    assert selector.selected_indices_.size == 3


@pytest.mark.parametrize("scale_pls", [False, True])
def test_selectivity_ratio_plsda_matches_explicit_dummy_response_plsr(scale_pls):
    rng = np.random.default_rng(101)
    X = rng.normal(size=(90, 8)) * np.geomspace(0.2, 5.0, 8)
    y_numeric = (1.8 * X[:, 1] - 0.7 * X[:, 6] > 0).astype(float)
    y_labels = np.where(y_numeric == 1.0, "positive", "negative")

    selector = SelectivityRatioSelector(
        estimator=PLSDA(n_components=2, scale_pls=scale_pls),
        task="classification",
        n_features_to_select=3,
    ).fit(X, y_labels)
    reference_scaler = StandardScaler(with_std=scale_pls).fit(X)
    X_reference = reference_scaler.transform(X)
    reference_plsr = PLSRegression(n_components=2, scale=False).fit(
        X_reference, y_numeric
    )
    reference_coefficient = np.asarray(reference_plsr.coef_).reshape(-1)
    if scale_pls:
        reference_coefficient /= reference_scaler.scale_
    reference = SelectivityRatioSelector._selectivity_ratios(
        X,
        reference_coefficient,
    )

    np.testing.assert_allclose(selector.scores_, reference[0], atol=1e-12)
    np.testing.assert_allclose(selector.explained_variance_, reference[1], atol=1e-12)
    np.testing.assert_allclose(selector.residual_variance_, reference[2], atol=1e-12)


def test_selectivity_ratio_runs_inside_classification_pipeline():
    rng = np.random.default_rng(37)
    X = rng.normal(size=(150, 16))
    y = (3 * X[:, 2] - 2 * X[:, 12] + rng.normal(scale=0.2, size=150) > 0).astype(int)
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "select",
                SelectivityRatioSelector(
                    task="classification",
                    n_components=2,
                    n_features_to_select=5,
                ),
            ),
            ("model", LogisticRegression(random_state=4)),
        ]
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=StratifiedKFold(3, shuffle=True, random_state=5),
        scoring="balanced_accuracy",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.mean(scores) > 0.8


def test_selectivity_ratio_supports_a_single_input_variable():
    X = np.arange(20, dtype=float).reshape(-1, 1)
    y = 2 * X.ravel()

    selector = SelectivityRatioSelector(
        n_components=1,
        n_features_to_select=1,
    ).fit(X, y)

    assert selector.selected_indices_.tolist() == [0]
    assert np.isinf(selector.scores_[0])


def test_selectivity_ratio_rejects_invalid_threshold():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.arange(10, dtype=float)

    with pytest.raises(ValueError, match="threshold"):
        SelectivityRatioSelector(threshold="median").fit(X, y)


def test_selectivity_ratio_rejects_multiresponse_target():
    rng = np.random.default_rng(35)
    X = rng.normal(size=(40, 7))
    y = np.column_stack((X[:, 1], X[:, 5]))

    with pytest.raises(ValueError, match="multi-response coefficients are unsupported"):
        SelectivityRatioSelector(
            estimator=PLSRegression(n_components=2),
            n_features_to_select=2,
        ).fit(X, y)


def test_selectivity_ratio_requires_estimator_coefficients():
    rng = np.random.default_rng(36)
    X = rng.normal(size=(40, 7))
    y = X[:, 2]

    with pytest.raises(TypeError, match="coef_"):
        SelectivityRatioSelector(
            estimator=KNeighborsRegressor(),
            n_features_to_select=2,
        ).fit(X, y)


@pytest.mark.parametrize(
    ("n_samples", "published"),
    # Farrés et al. (2015): water (N=25), climate TIC (N=10), transcriptome (N=12).
    [(25, 2.04), (10, 3.73), (12, 3.14)],
)
def test_selectivity_ratio_f_test_matches_published_critical_values(
    n_samples, published
):
    value = SelectivityRatioSelector.f_critical_value(n_samples, alpha=0.05)

    assert round(value, 2) == published


def test_selectivity_ratio_f_test_threshold_is_applied():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(30, 12))
    y = 2 * X[:, 3] - X[:, 8] + rng.normal(scale=0.1, size=30)

    selector = SelectivityRatioSelector(threshold="f_test", alpha=0.05).fit(X, y)

    assert selector.threshold_ == pytest.approx(
        SelectivityRatioSelector.f_critical_value(30, 0.05)
    )
    np.testing.assert_array_equal(
        selector.support_, selector.scores_ >= selector.threshold_
    )
    # The F-test is conservative (Farrés et al.): the dominant variable passes,
    # while the minor contributor (about 20% of y variance, SR near 0.25) does not.
    assert selector.selected_indices_.tolist() == [3]
    assert selector.scores_[8] < selector.threshold_


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, True])
def test_selectivity_ratio_f_test_rejects_invalid_alpha(alpha):
    X = np.random.default_rng(0).normal(size=(10, 3))
    y = X[:, 0]

    with pytest.raises(ValueError, match="alpha"):
        SelectivityRatioSelector(threshold="f_test", alpha=alpha).fit(X, y)


def test_selectivity_ratio_mean_threshold_ignores_infinite_ratios():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = 2 * X[:, 0] + X[:, 1]

    selector = SelectivityRatioSelector(n_components=2, scale=False).fit(X, y)

    assert np.isfinite(selector.threshold_)


def _chemotools_plsda():
    module = pytest.importorskip("chemotools.classification")
    if not hasattr(module, "PLSDA"):
        pytest.skip("installed chemotools does not provide PLSDA")
    return module.PLSDA


@pytest.mark.parametrize("n_classes", [2, 3])
def test_vip_accepts_chemotools_plsda(n_classes):
    ChemotoolsPLSDA = _chemotools_plsda()
    rng = np.random.default_rng(140 + n_classes)
    X = rng.normal(size=(120, 10))
    logits = np.column_stack((2.0 * X[:, 1], -X[:, 4], 1.5 * X[:, 8]))[:, :n_classes]
    y = np.asarray(["a", "b", "c"])[np.argmax(logits, axis=1)]

    selector = VIPSelector(
        estimator=ChemotoolsPLSDA(n_components=2, scale=False),
        task="classification",
        n_features_to_select=3,
    ).fit(X, y)

    assert selector.task_ == "classification"
    assert np.mean(selector.scores_**2) == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [False, True])
def test_selectivity_ratio_chemotools_plsda_matches_package_plsda(scale):
    ChemotoolsPLSDA = _chemotools_plsda()
    rng = np.random.default_rng(143)
    X = rng.normal(size=(90, 8)) * np.geomspace(0.2, 5.0, 8)
    y = np.where(1.8 * X[:, 1] - 0.7 * X[:, 6] > 0, "positive", "negative")

    chemotools_selector = SelectivityRatioSelector(
        estimator=ChemotoolsPLSDA(n_components=2, scale=scale),
        task="classification",
        n_features_to_select=3,
    ).fit(X, y)
    package_selector = SelectivityRatioSelector(
        estimator=PLSDA(n_components=2, scale_pls=scale),
        task="classification",
        n_features_to_select=3,
    ).fit(X, y)

    np.testing.assert_allclose(
        chemotools_selector.scores_, package_selector.scores_, rtol=1e-8
    )


def test_selectivity_ratio_rejects_multiclass_chemotools_plsda():
    ChemotoolsPLSDA = _chemotools_plsda()
    rng = np.random.default_rng(144)
    X = rng.normal(size=(90, 6))
    y = np.asarray(["a", "b", "c"])[np.argmax(X[:, :3], axis=1)]

    with pytest.raises(ValueError, match="multiclass"):
        SelectivityRatioSelector(
            estimator=ChemotoolsPLSDA(n_components=2),
            task="classification",
            n_features_to_select=2,
        ).fit(X, y)


@pytest.mark.parametrize(
    ("factory", "n_samples", "match"),
    [
        (lambda: VIPSelector(n_components=0), 12, "n_components must be between"),
        (lambda: VIPSelector(n_features_to_select=0), 12, "n_features_to_select"),
        (lambda: VIPSelector(threshold=-1.0), 12, "finite non-negative"),
        (lambda: VIPSelector(threshold=1e9), 12, "No variable reached"),
        (lambda: SRSelector(threshold="f_test", alpha=2.0), 12, "alpha"),
        (lambda: SRSelector(threshold="f_test", n_components=1), 3, "at least 4"),
    ],
)
def test_importance_selectors_reject_invalid_configuration(factory, n_samples, match):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, 5))
    with pytest.raises(ValueError, match=match):
        factory().fit(X, X[:, 0] - X[:, 2])
