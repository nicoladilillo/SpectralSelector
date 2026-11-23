import json
from pathlib import Path

import numpy as np
import pytest
from joblib import cpu_count
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold, LeaveOneOut, cross_val_score

from spectralselector import (
    PLSDA,
    BOSSSelector,
    CARSSelector,
    GeneticAlgorithmSelector,
)


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(24, 8))
    y = 3 * X[:, 1] - 2 * X[:, 6] + rng.normal(scale=0.15, size=24)
    return X, y


def _pls_component_reference_case():
    """Recreate the analytic component-search fixture inputs."""
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "reference"
        / "sklearn_1_9_0_pls_component_search.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    rows = np.arange(1.0, 33.0)
    X = np.column_stack(
        (
            np.sin(rows * 0.17),
            np.cos(rows * 0.29),
            np.sin(rows * 0.41) + 0.02 * rows,
            ((rows.astype(int) % 7) - 3.0) / 3.0,
            np.cos(rows * 0.11) * np.sin(rows * 0.37),
            np.sin(rows * 0.07) + np.cos(rows * 0.53),
        )
    )
    y = 1.7 * X[:, 0] - 0.9 * X[:, 2] + 0.6 * X[:, 4] + 0.08 * np.sin(rows * 1.13)
    return fixture, X, y


def _prepare_component_search(selector, fixture):
    """Set only the fitted validation state required by ``_score_subset``."""
    cv = fixture["model"]["cv"]
    selector.task_ = "regression"
    selector.scoring_ = fixture["model"]["scoring"]
    selector.cv_ = KFold(
        n_splits=cv["n_splits"],
        shuffle=cv["shuffle"],
        random_state=cv["random_state"],
    )
    return selector


def test_cars_and_boss_component_search_matches_versioned_reference():
    fixture, X, y = _pls_component_reference_case()
    indices = np.asarray(fixture["input"]["selected_zero_based_indices"])
    expected = fixture["expected"]
    tolerance = expected["absolute_tolerance"]
    cv = fixture["model"]["cv"]

    # Recompute the stored, estimator-independent candidate scores explicitly.
    splitter = KFold(
        n_splits=cv["n_splits"],
        shuffle=cv["shuffle"],
        random_state=cv["random_state"],
    )
    for n_components in fixture["model"]["component_candidates"]:
        fold_scores = cross_val_score(
            PLSRegression(n_components=n_components, scale=False),
            X[:, indices],
            y,
            cv=splitter,
            scoring=fixture["model"]["scoring"],
            error_score="raise",
        )
        np.testing.assert_allclose(
            fold_scores,
            expected["fold_scores"][str(n_components)],
            rtol=0.0,
            atol=tolerance,
        )
        assert np.mean(fold_scores) == pytest.approx(
            expected["mean_scores"][str(n_components)],
            rel=0.0,
            abs=tolerance,
        )

    selectors = (
        CARSSelector(max_components=4, n_jobs=1),
        BOSSSelector(max_components=4, n_jobs=1),
    )
    for selector in selectors:
        score, n_components = _prepare_component_search(
            selector, fixture
        )._score_subset(X, y, indices, groups=None)
        assert n_components == expected["best_n_components"]
        assert score == pytest.approx(
            expected["best_mean_score"], rel=0.0, abs=tolerance
        )


@pytest.mark.parametrize(
    ("n_samples", "indices", "expected_candidates"),
    [
        (8, np.asarray([0, 1]), [1, 2]),
        (3, np.arange(5), [1, 2]),
    ],
    ids=["subset-width-bound", "sample-rank-bound"],
)
def test_cars_and_boss_component_bounds_and_ties(
    monkeypatch,
    n_samples,
    indices,
    expected_candidates,
):
    evaluated = []

    def equal_score(estimator, X, y, **kwargs):
        del X, y, kwargs
        evaluated.append(estimator.n_components)
        return np.asarray([-0.25, -0.25])

    monkeypatch.setattr("spectralselector.cars.cross_val_score", equal_score)
    monkeypatch.setattr("spectralselector.boss.cross_val_score", equal_score)
    X = np.arange(n_samples * 5, dtype=float).reshape(n_samples, 5)
    y = np.arange(n_samples, dtype=float)

    selectors = (
        CARSSelector(max_components=10, n_jobs=1),
        BOSSSelector(max_components=10, n_jobs=1),
    )
    for selector in selectors:
        selector.task_ = "regression"
        selector.scoring_ = "neg_root_mean_squared_error"
        selector.cv_ = 2
        score, n_components = selector._score_subset(X, y, indices, groups=None)
        assert score == pytest.approx(-0.25)
        assert n_components == 1

    assert evaluated == expected_candidates * 2


def test_cars_uses_plsr_and_tunes_components_for_regression(regression_data):
    X, y = regression_data
    selector = CARSSelector(
        cv=3,
        max_components=3,
        n_runs=2,
        n_iterations=3,
        min_features=2,
        random_state=4,
    ).fit(X, y)

    assert selector.task_ == "regression"
    assert isinstance(selector.estimator_, PLSRegression)
    assert 1 <= selector.best_n_components_ <= 3
    assert selector.estimator_.n_components == selector.best_n_components_
    assert selector.run_best_n_components_.shape == (2,)
    assert selector.selected_indices_.size >= 1
    assert selector.selection_counts_.shape == (X.shape[1],)
    assert np.isfinite(selector.best_score_)
    assert not hasattr(clone(selector), "best_score_")


def test_boss_uses_plsr_and_tunes_components_for_regression(regression_data):
    X, y = regression_data
    selector = BOSSSelector(
        cv=3,
        max_components=3,
        n_submodels=3,
        top_fraction=0.5,
        max_iterations=2,
        min_features=2,
        random_state=5,
    ).fit(X, y)

    assert selector.task_ == "regression"
    assert isinstance(selector.estimator_, PLSRegression)
    assert 1 <= selector.best_n_components_ <= 3
    assert selector.estimator_.n_components == selector.best_n_components_
    assert 1 <= selector.component_history_.size <= 2
    assert selector.selected_indices_.size >= 2
    assert selector.score_history_.shape == (2,)
    assert np.isfinite(selector.best_score_)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CARSSelector(
            task="classification",
            scoring="balanced_accuracy",
            cv=3,
            max_components=2,
            n_runs=1,
            n_iterations=2,
            random_state=16,
        ),
        lambda: BOSSSelector(
            task="classification",
            scoring="balanced_accuracy",
            cv=3,
            max_components=2,
            n_submodels=2,
            top_fraction=0.5,
            max_iterations=1,
            random_state=16,
        ),
    ],
    ids=["cars", "boss"],
)
def test_pls_selectors_use_plsda_for_classification(factory):
    rng = np.random.default_rng(116)
    X = rng.normal(size=(36, 8))
    y = (2 * X[:, 1] - X[:, 5] > 0).astype(int)

    selector = factory().fit(X, y)

    assert selector.task_ == "classification"
    assert selector.scoring_ == "balanced_accuracy"
    assert isinstance(selector.estimator_, PLSDA)
    assert 1 <= selector.best_n_components_ <= 2
    assert selector.estimator_.n_components == selector.best_n_components_


def test_cars_does_not_accept_a_user_estimator():
    with pytest.raises(TypeError):
        CARSSelector(Ridge())


def test_boss_rejects_non_pls_estimators(regression_data):
    X, y = regression_data
    selector = BOSSSelector(Ridge(), n_submodels=2, top_fraction=0.5, max_iterations=1)

    with pytest.raises(TypeError, match="PLSRegression, PLSDA, or None"):
        selector.fit(X, y)


@pytest.mark.parametrize(
    "selector",
    [
        BOSSSelector(
            PLSDA(),
            task="regression",
            n_submodels=2,
            top_fraction=0.5,
            max_iterations=1,
        ),
    ],
    ids=["boss-needs-plsr"],
)
def test_pls_selectors_require_task_appropriate_pls_model(regression_data, selector):
    X, y = regression_data
    target = (y > np.median(y)).astype(int) if selector.task == "classification" else y

    with pytest.raises(ValueError, match="incompatible"):
        selector.fit(X, target)


def test_genetic_selector_supports_classification_and_intervals():
    rng = np.random.default_rng(102)
    X = rng.normal(size=(30, 10))
    y = np.where(X[:, 2] + X[:, 3] > 0, "affected", "healthy")
    selector = GeneticAlgorithmSelector(
        LogisticRegression(max_iter=500),
        task="classification",
        scoring="balanced_accuracy",
        cv=3,
        population_size=6,
        generations=2,
        tournament_size=2,
        n_intervals=5,
        min_groups=1,
        max_groups=3,
        random_state=6,
    ).fit(X, y)

    assert selector.task_ == "classification"
    assert 1 <= selector.best_chromosome_.sum() <= 3
    assert selector.selected_indices_.size >= 2
    assert selector.best_score_history_.shape == (2,)
    assert np.isfinite(selector.best_score_)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CARSSelector(
            cv=-1,
            max_components=1,
            n_runs=1,
            n_iterations=2,
            min_features=2,
            random_state=7,
        ),
        lambda: BOSSSelector(
            cv=-1,
            max_components=1,
            n_submodels=2,
            top_fraction=0.5,
            max_iterations=1,
            min_features=2,
            random_state=7,
        ),
        lambda: GeneticAlgorithmSelector(
            Ridge(),
            cv=-1,
            population_size=4,
            generations=1,
            tournament_size=2,
            min_groups=1,
            max_groups=2,
            random_state=7,
        ),
    ],
    ids=["cars", "boss", "genetic"],
)
def test_cv_minus_one_means_leave_one_out(regression_data, factory):
    X, y = regression_data
    selector = factory().fit(X[:12, :4], y[:12])

    assert isinstance(selector.cv_, LeaveOneOut)


def test_task_must_match_estimator(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="requires a classifier"):
        GeneticAlgorithmSelector(
            Ridge(),
            task="classification",
            population_size=4,
            generations=1,
            tournament_size=2,
        ).fit(X, (y > np.median(y)).astype(int))


@pytest.mark.parametrize(
    ("n_jobs", "error_type"),
    [
        (0, ValueError),
        (-2, ValueError),
        (cpu_count() + 1, ValueError),
        (None, TypeError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_invalid_n_jobs_fails_before_search(regression_data, n_jobs, error_type):
    X, y = regression_data
    selector = GeneticAlgorithmSelector(
        Ridge(),
        population_size=4,
        generations=1,
        tournament_size=2,
        n_jobs=n_jobs,
        random_state=8,
    )

    with pytest.raises(error_type, match="n_jobs"):
        selector.fit(X, y)


def test_n_jobs_minus_one_resolves_to_all_available_cores(regression_data):
    X, y = regression_data
    selector = GeneticAlgorithmSelector(
        Ridge(),
        cv=2,
        population_size=4,
        generations=1,
        tournament_size=2,
        n_jobs=-1,
        random_state=9,
    ).fit(X, y)

    assert selector.n_jobs_ == cpu_count()


@pytest.mark.parametrize(
    "factory",
    [
        lambda cv: CARSSelector(
            cv=cv,
            max_components=2,
            n_runs=1,
            n_iterations=2,
            random_state=12,
        ),
        lambda cv: BOSSSelector(
            cv=cv,
            max_components=2,
            n_submodels=2,
            top_fraction=0.5,
            max_iterations=1,
            random_state=12,
        ),
        lambda cv: GeneticAlgorithmSelector(
            Ridge(alpha=0.1),
            cv=cv,
            population_size=4,
            generations=1,
            tournament_size=2,
            random_state=12,
        ),
    ],
    ids=["cars", "boss", "genetic"],
)
def test_wrapper_selectors_support_group_aware_cv(regression_data, factory):
    X, y = regression_data
    groups = np.repeat(np.arange(8), 3)

    selector = factory(GroupKFold(n_splits=4)).fit(X, y, groups=groups)

    assert selector.selected_indices_.size >= 1
    assert np.isfinite(selector.best_score_)


def test_wrapper_selector_rejects_misaligned_groups(regression_data):
    X, y = regression_data
    selector = GeneticAlgorithmSelector(
        Ridge(),
        cv=GroupKFold(n_splits=3),
        population_size=4,
        generations=1,
        tournament_size=2,
        random_state=13,
    )

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        selector.fit(X, y, groups=np.arange(X.shape[0] - 1))


@pytest.mark.skipif(cpu_count() < 2, reason="Parallel comparison needs two CPUs")
@pytest.mark.parametrize(
    "factory",
    [
        lambda n_jobs: CARSSelector(
            cv=2,
            max_components=2,
            n_runs=2,
            n_iterations=3,
            n_jobs=n_jobs,
            random_state=10,
        ),
        lambda n_jobs: BOSSSelector(
            cv=2,
            max_components=2,
            n_submodels=3,
            top_fraction=0.5,
            max_iterations=2,
            n_jobs=n_jobs,
            random_state=10,
        ),
        lambda n_jobs: GeneticAlgorithmSelector(
            Ridge(alpha=0.1),
            cv=2,
            population_size=6,
            generations=2,
            tournament_size=2,
            max_groups=4,
            n_jobs=n_jobs,
            random_state=10,
        ),
    ],
    ids=["cars", "boss", "genetic"],
)
def test_parallel_and_serial_searches_are_deterministic(regression_data, factory):
    X, y = regression_data
    serial = factory(1).fit(X, y)
    parallel = factory(2).fit(X, y)

    np.testing.assert_array_equal(serial.selected_indices_, parallel.selected_indices_)
    assert serial.best_score_ == pytest.approx(parallel.best_score_)


@pytest.mark.determinism
@pytest.mark.parametrize(
    ("factory", "expected_indices", "history_attribute"),
    [
        (
            lambda: CARSSelector(
                cv=3,
                max_components=2,
                n_runs=2,
                n_iterations=4,
                min_features=2,
                random_state=314,
            ),
            [7],
            "run_best_scores_",
        ),
        (
            lambda: BOSSSelector(
                cv=3,
                max_components=2,
                n_submodels=5,
                top_fraction=0.4,
                n_runs=2,
                min_features=2,
                max_iterations=3,
                random_state=314,
            ),
            [1, 7],
            "run_best_scores_",
        ),
        (
            lambda: GeneticAlgorithmSelector(
                Ridge(alpha=0.1),
                cv=3,
                population_size=8,
                generations=3,
                tournament_size=2,
                min_groups=1,
                max_groups=4,
                random_state=314,
            ),
            [1, 4, 5, 6],
            "best_score_history_",
        ),
    ],
    ids=["cars", "boss", "genetic"],
)
def test_fixed_seed_complete_runs_are_reproducible_across_supported_platforms(
    factory,
    expected_indices,
    history_attribute,
):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(36, 10))
    y = 5.0 * X[:, 1] - 4.0 * X[:, 7] + 0.25 * X[:, 4]

    first = factory().fit(X, y)
    second = factory().fit(X, y)

    np.testing.assert_array_equal(first.selected_indices_, expected_indices)
    np.testing.assert_array_equal(second.selected_indices_, expected_indices)
    np.testing.assert_allclose(
        getattr(first, history_attribute),
        getattr(second, history_attribute),
        rtol=1e-8,
        atol=1e-10,
    )
    assert first.best_score_ == pytest.approx(
        second.best_score_,
        rel=1e-8,
        abs=1e-10,
    )


_FAST_BOSS = dict(cv=3, max_components=2, n_submodels=2, top_fraction=0.5)
_FAST_CARS = dict(cv=3, max_components=2, n_runs=1, n_iterations=2)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"max_components": 0}, "max_components must be a positive integer"),
        ({"n_runs": True}, "n_runs must be a positive integer"),
        ({"min_features": 9}, "min_features cannot exceed"),
        ({"top_fraction": 1.5}, "top_fraction must lie"),
        ({"top_fraction": None}, "requires n_submodels >= 500"),
        ({"speed": 6}, "speed must be an integer"),
        ({"max_iterations": 0}, "max_iterations must be"),
        ({"n_jobs": 0}, "n_jobs must be a non-zero"),
        ({"verbose": -1}, "verbose must be"),
        ({"store_populations": 1}, "store_populations must be boolean"),
        ({"wavelengths": np.ones((2, 4))}, "one-dimensional"),
        ({"wavelengths": np.arange(3)}, "length must match"),
        ({"task": "ranking"}, "task must be"),
        ({"cv": 1}, "cv must be >= 2"),
    ],
)
def test_boss_rejects_invalid_parameters(regression_data, params, match):
    X, y = regression_data
    with pytest.raises(ValueError, match=match):
        BOSSSelector(**{**_FAST_BOSS, **params}).fit(X, y)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"peak_distance": 0}, "peak_distance must be a positive integer"),
        ({"n_iterations": 1}, "n_iterations must be at least 2"),
        ({"min_features": 9}, "min_features cannot exceed"),
        ({"sample_fraction": 0}, "sample_fraction must lie"),
        ({"peak_selection": "worst"}, "peak_selection must be"),
        ({"peak_prominence": -1}, "peak_prominence must be"),
        ({"peak_min_frequency": 2}, "peak_min_frequency must be"),
        ({"max_peaks": 0}, "max_peaks must be"),
        ({"n_jobs": 0}, "n_jobs must be a non-zero"),
        ({"verbose": -1}, "verbose must be"),
        ({"wavelengths": np.ones((2, 4))}, "one-dimensional"),
        ({"wavelengths": np.arange(3)}, "length must match"),
        ({"task": "ranking"}, "task must be"),
        ({"cv": 1}, "cv must be >= 2"),
    ],
)
def test_cars_rejects_invalid_parameters(regression_data, params, match):
    X, y = regression_data
    with pytest.raises(ValueError, match=match):
        CARSSelector(**{**_FAST_CARS, **params}).fit(X, y)


def test_boss_rejects_misaligned_groups_and_negative_rounding(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="one entry for every sample"):
        BOSSSelector(**_FAST_BOSS).fit(X, y, groups=np.arange(3))
    with pytest.raises(ValueError, match="non-negative"):
        BOSSSelector._matlab_round_positive(-1)
    assert BOSSSelector(n_submodels=7).n_bootstraps == 7


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("wavelengths", [None, np.linspace(900, 1700, 8)])
def test_boss_verbose_progress_and_stored_populations(
    regression_data, capsys, n_jobs, wavelengths
):
    X, y = regression_data
    selector = BOSSSelector(
        **_FAST_BOSS,
        max_iterations=2,
        n_runs=2,
        verbose=4,
        n_jobs=n_jobs,
        store_populations=True,
        wavelengths=wavelengths,
        random_state=3,
    ).fit(X, y)

    log = capsys.readouterr()
    assert "Selected" in log.out + log.err
    assert len(selector.run_populations_) == 2
    # The tqdm/joblib hook must restore Joblib's callback after the run.
    import joblib.parallel

    assert joblib.parallel.BatchCompletionCallBack.__name__ == "BatchCompletionCallBack"


@pytest.mark.parametrize("wavelengths", [None, np.linspace(900, 1700, 8)])
def test_cars_verbose_run_logs_selection(regression_data, capsys, wavelengths):
    X, y = regression_data
    CARSSelector(**_FAST_CARS, verbose=4, wavelengths=wavelengths, random_state=3).fit(
        X, y
    )

    log = capsys.readouterr()
    assert "CARS fit complete" in log.out + log.err


def test_cars_rejects_misaligned_groups(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="one entry for every sample"):
        CARSSelector(**_FAST_CARS).fit(X, y, groups=np.arange(3))
    assert CARSSelector(n_runs=7).n_sampling_runs == 7


@pytest.mark.parametrize(
    "params",
    [
        {"peak_selection": "all", "peak_prominence": 0.0, "max_peaks": 2},
        {"sample_fraction": 1.0, "min_features": 6},
    ],
    ids=["all-peaks", "full-sample-min-features"],
)
def test_cars_peak_and_sampling_options(regression_data, params):
    X, y = regression_data
    selector = CARSSelector(
        **{**_FAST_CARS, "n_runs": 3, "n_iterations": 4, **params}, random_state=6
    ).fit(X, y)

    assert 1 <= selector.selected_indices_.size <= params.get("max_peaks", 8)
    assert np.isfinite(selector.best_score_)


@pytest.mark.parametrize(
    ("n_submodels", "expected"), [(2000, 0.05), (1000, 0.10), (500, 0.20), (499, None)]
)
def test_boss_default_top_fraction_follows_matlab_reference(n_submodels, expected):
    selector = BOSSSelector(n_submodels=n_submodels)
    if expected is None:
        with pytest.raises(ValueError, match="n_submodels >= 500"):
            selector._resolve_top_fraction()
    else:
        assert selector._resolve_top_fraction() == expected


def test_boss_accepts_user_pls_models(regression_data):
    X, y = regression_data
    regression = BOSSSelector(
        PLSRegression(scale=False), **_FAST_BOSS, max_iterations=1, random_state=2
    ).fit(X, y)
    assert isinstance(regression.estimator_, PLSRegression)

    labels = (y > np.median(y)).astype(int)
    classification = BOSSSelector(
        PLSDA(), **_FAST_BOSS, max_iterations=1, random_state=2
    ).fit(X, labels)
    assert classification.task_ == "classification"
    assert isinstance(classification.estimator_, PLSDA)
