import numpy as np
import pytest
from sklearn.model_selection import KFold, LeaveOneOut

from benchmarks.run_dataset_matrix import (
    DATASET_NAMES,
    METHOD_NAMES,
    PROFILE_METHOD_SETTINGS,
    DatasetCase,
    NotApplicable,
    candidate_sizes,
    confidence_interval_95,
    dataset_provenance,
    fit_matrix_fold,
    inner_cv_for_fold,
    method_configuration,
    method_seeds,
    outer_split_list,
    parse_requested,
    parse_selector_seeds,
    prediction_configuration,
    prediction_parameter_grid,
    search_budget,
    selection_jaccard,
    summarize,
)
from benchmarks.run_nested_benchmarks import predictive_metrics


@pytest.fixture
def regression_case():
    rng = np.random.default_rng(81)
    X = rng.normal(size=(30, 12))
    y = X[:, 2] - X[:, 8]
    return DatasetCase(
        name="fixture",
        task="regression",
        X=X,
        y=y,
        wavelengths=np.arange(X.shape[1], dtype=float),
        source="unit test",
    )


def test_every_registered_method_has_a_smoke_configuration(regression_case):
    cv = KFold(2, shuffle=True, random_state=3)

    configured = {
        name: method_configuration(
            name=name,
            case=regression_case,
            sizes=[3, 6],
            internal_cv=cv,
            profile="smoke",
            random_state=4,
            n_jobs=1,
        )
        for name in METHOD_NAMES
    }

    assert set(configured) == set(METHOD_NAMES)
    assert configured["full_spectrum"] == ("passthrough", {})


def test_search_methods_receive_selected_metric_and_cv(regression_case):
    cv = KFold(3, shuffle=True, random_state=3)
    selectors = {
        name: method_configuration(
            name=name,
            case=regression_case,
            sizes=[3, 6],
            internal_cv=cv,
            profile="smoke",
            random_state=4,
            n_jobs=1,
            scoring="neg_mean_absolute_error",
        )[0]
        for name in ("fcovsel", "cars", "boss", "genetic")
    }

    for selector in selectors.values():
        assert selector.scoring == "neg_mean_absolute_error"
        assert selector.cv is cv


def test_matrix_minus_one_uses_leave_one_out(regression_case):
    outer = outer_split_list(regression_case, -1, random_state=2)
    inner, smallest = inner_cv_for_fold(
        "regression",
        regression_case.y,
        groups=None,
        n_splits=-1,
        random_state=2,
    )

    assert len(outer) == regression_case.X.shape[0]
    assert isinstance(inner, LeaveOneOut)
    assert smallest == regression_case.X.shape[0] - 1


def test_prediction_configuration_accepts_selected_metric():
    _, regression_scoring = prediction_configuration(
        "regression", "neg_mean_absolute_error"
    )
    _, classification_scoring = prediction_configuration("classification", "f1_macro")

    assert regression_scoring == "neg_mean_absolute_error"
    assert classification_scoring == "f1_macro"


def test_downstream_regularization_budget_is_symmetric_between_tasks():
    regression_grid = prediction_parameter_grid("regression")
    classification_grid = prediction_parameter_grid("classification")

    assert set(regression_grid) == {"model__alpha"}
    assert set(classification_grid) == {"model__n_components"}
    assert len(regression_grid["model__alpha"]) == 5
    assert len(classification_grid["model__n_components"]) == 5


def test_vip_and_sr_tune_pls_components_for_regression(regression_case):
    cv = KFold(2, shuffle=True, random_state=3)

    for method in ("vip", "selectivity_ratio"):
        _, grid = method_configuration(
            name=method,
            case=regression_case,
            sizes=[3, 6],
            internal_cv=cv,
            profile="smoke",
            random_state=4,
            n_jobs=1,
        )
        assert grid["select__n_components"] == [1, 2, 3]


def test_vip_and_sr_tune_plsda_components_for_binary_classification(
    regression_case,
):
    case = DatasetCase(
        **{
            **regression_case.__dict__,
            "task": "classification",
            "y": np.where(regression_case.y >= np.median(regression_case.y), "a", "b"),
        }
    )
    cv = KFold(2, shuffle=True, random_state=3)

    for method in ("vip", "selectivity_ratio"):
        _, grid = method_configuration(
            name=method,
            case=case,
            sizes=[3, 6],
            internal_cv=cv,
            profile="smoke",
            random_state=4,
            n_jobs=1,
        )
        assert grid["select__estimator__n_components"] == [1, 2, 3]


def test_declared_search_budget_matches_cars_smoke_configuration(regression_case):
    cv = KFold(2, shuffle=True, random_state=3)
    selector, selector_grid = method_configuration(
        name="cars",
        case=regression_case,
        sizes=[3, 6],
        internal_cv=cv,
        profile="smoke",
        random_state=4,
        n_jobs=1,
    )
    prediction_grid = prediction_parameter_grid("regression")

    budget = search_budget(
        method="cars",
        selector=selector,
        selector_grid=selector_grid,
        prediction_grid=prediction_grid,
        n_inner_splits=2,
        n_selector_seeds=1,
    )

    assert PROFILE_METHOD_SETTINGS["smoke"]["cars"] == {
        "n_runs": 1,
        "n_iterations": 3,
    }
    assert budget["joint_outer_candidates"] == 5
    assert budget["pipeline_fits_including_refit"] == 11
    assert budget["intrinsic_selector_cv_fit_upper_bound_per_selector_fit"] == 18
    assert budget["intrinsic_cv_fit_upper_bound_per_outer_fold_and_seed"] == 198


def test_fold_record_reports_cpu_prediction_and_traced_memory(regression_case):
    model, scoring = prediction_configuration("regression")
    payload = fit_matrix_fold(
        case=regression_case,
        method="full_spectrum",
        fold=1,
        train=np.arange(20),
        test=np.arange(20, 30),
        selector="passthrough",
        selector_grid={},
        prediction_grid={"model__alpha": [1.0]},
        model=model,
        scoring=scoring,
        inner_cv=KFold(2, shuffle=True, random_state=3),
        n_jobs=1,
        selector_seed=4,
    )

    record = payload["record"]
    assert record["main_process_cpu_time_seconds"] > 0
    assert record["prediction_time_seconds"] > 0
    assert record["peak_traced_memory_mebibytes"] > 0


def test_multiclass_unsupported_pairs_are_explicit():
    X = np.arange(90, dtype=float).reshape(30, 3)
    case = DatasetCase(
        name="three_class",
        task="classification",
        X=X,
        y=np.tile([0, 1, 2], 10),
        wavelengths=np.arange(3, dtype=float),
        source="unit test",
    )

    for method in ("selectivity_ratio",):
        with pytest.raises(NotApplicable):
            method_configuration(
                name=method,
                case=case,
                sizes=[2],
                internal_cv=KFold(2),
                profile="smoke",
                random_state=4,
                n_jobs=1,
            )


def test_multiclass_fcovsel_is_configured_with_task_metric_and_cv():
    rng = np.random.default_rng(82)
    X = rng.normal(size=(45, 8))
    case = DatasetCase(
        name="three_class",
        task="classification",
        X=X,
        y=np.repeat(["a", "b", "c"], 15),
        wavelengths=np.arange(8, dtype=float),
        source="unit test",
    )
    cv = KFold(3, shuffle=True, random_state=3)

    selector, selector_grid = method_configuration(
        name="fcovsel",
        case=case,
        sizes=[2, 4],
        internal_cv=cv,
        profile="smoke",
        random_state=4,
        n_jobs=1,
    )

    assert selector.task == "classification"
    assert selector.scoring == "balanced_accuracy"
    assert selector.cv is cv
    assert selector.n_features_to_select == "auto"
    assert selector.max_features == 20
    assert selector_grid == {}


def test_matrix_name_parsing_and_small_fold_sizes():
    assert parse_requested("all", DATASET_NAMES, "datasets") == list(DATASET_NAMES)
    assert parse_requested("corn,diesel", DATASET_NAMES, "datasets") == [
        "corn",
        "diesel",
    ]
    assert candidate_sizes(100, 5) == [4]
    with pytest.raises(ValueError, match="Unknown methods"):
        parse_requested("unknown", METHOD_NAMES, "methods")


def test_selector_seed_defaults_validation_and_method_repeats():
    assert parse_selector_seeds(None, profile="smoke", fallback=9) == [9]
    assert parse_selector_seeds(None, profile="publication", fallback=9) == [
        9,
        10,
        11,
        12,
        13,
    ]
    assert parse_selector_seeds("7, 8,7", profile="smoke", fallback=9) == [7, 8]
    assert method_seeds("cars", [7, 8], fallback=9) == [7, 8]
    assert method_seeds("vip", [7, 8], fallback=9) == [9]
    with pytest.raises(ValueError, match="At least one"):
        parse_selector_seeds(",", profile="smoke", fallback=9)
    with pytest.raises(ValueError, match="comma-separated"):
        parse_selector_seeds("invalid", profile="smoke", fallback=9)
    with pytest.raises(ValueError, match="between 0"):
        parse_selector_seeds("-1", profile="smoke", fallback=9)


def test_classification_metrics_support_multiclass_and_string_labels():
    metrics = predictive_metrics(
        "classification",
        np.asarray(["a", "b", "c", "a"]),
        np.asarray(["a", "b", "a", "a"]),
    )

    assert set(metrics) == {"accuracy", "balanced_accuracy", "f1_macro"}
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_dataset_provenance_records_scientific_and_loader_metadata(regression_case):
    case = DatasetCase(
        **{
            **regression_case.__dict__,
            "citation": "Fixture et al. (2026)",
            "license_name": "CC0-1.0",
            "target_names": ("property",),
            "loader": "tests.load_fixture",
            "loader_arguments": (("target", "property"),),
        }
    )
    runtime = {"packages": {"spectralselector": "0.8.0", "chemotools": "0.2.1"}}

    metadata = dataset_provenance(case, content_fingerprint="abc123", runtime=runtime)

    assert metadata["content_sha256"] == "abc123"
    assert metadata["citation"] == "Fixture et al. (2026)"
    assert metadata["license"] == "CC0-1.0"
    assert metadata["loader_arguments"] == {"target": "property"}
    assert metadata["target"] == {"names": ["property"], "n_responses": 1}
    assert metadata["wavelength_axis"]["n_features"] == case.X.shape[1]
    assert metadata["groups"] == {"present": False, "n_unique": 0}


def test_summary_adds_fold_confidence_intervals_paired_delta_and_dataset_rank():
    records = []
    for fold, baseline, selected in ((1, 0.5, 0.7), (2, 0.6, 0.8), (3, 0.7, 0.9)):
        common = {
            "dataset": "fixture",
            "task": "regression",
            "outer_fold": fold,
            "n_selected": 10,
            "selection_time_seconds": 0.1,
            "nested_fit_time_seconds": 0.2,
            "selector_n_components": None,
            "best_parameters": "{}",
            "rmse": 1.0,
            "mae": 0.8,
        }
        records.append(common | {"method": "full_spectrum", "r2": baseline})
        records.append(common | {"method": "vip", "r2": selected})

    rows = summarize(records)
    by_method = {row["method"]: row for row in rows}

    assert by_method["vip"]["predictive_rank_within_dataset"] == 1
    assert by_method["full_spectrum"]["predictive_rank_within_dataset"] == 2
    assert by_method["vip"]["mean_primary_metric_delta_vs_full_spectrum"] == (
        pytest.approx(0.2)
    )
    assert by_method["vip"]["ci95_low_r2"] < by_method["vip"]["mean_r2"]
    assert by_method["vip"]["ci95_high_r2"] > by_method["vip"]["mean_r2"]


def test_single_value_confidence_interval_has_zero_width():
    assert confidence_interval_95(np.asarray([0.75])) == (0.75, 0.75)


def test_selection_jaccard_and_summary_stability():
    assert selection_jaccard([[0, 1], [1, 2], [0, 1]]) == pytest.approx(5 / 9)
    records = [
        {
            "dataset": "fixture",
            "task": "regression",
            "method": "vip",
            "outer_fold": fold,
            "r2": score,
        }
        for fold, score in ((1, 0.5), (2, 0.6), (3, 0.7))
    ]
    selections = [
        {
            "dataset": "fixture",
            "task": "regression",
            "method": "vip",
            "outer_fold": fold,
            "selected_indices": indices,
        }
        for fold, indices in ((1, [0, 1]), (2, [1, 2]), (3, [0, 1]))
    ]

    [row] = summarize(records, selections)

    assert row["mean_pairwise_selection_jaccard"] == pytest.approx(5 / 9)


def test_summary_reports_repeated_seed_counts_and_within_fold_stability():
    records = []
    selections = []
    masks = {
        (1, 10): [0, 1],
        (1, 11): [0, 1],
        (2, 10): [1, 2],
        (2, 11): [1, 3],
    }
    for (fold, seed), indices in masks.items():
        records.append(
            {
                "dataset": "fixture",
                "task": "regression",
                "method": "cars",
                "outer_fold": fold,
                "selector_seed": seed,
                "r2": 0.5 + 0.01 * seed,
            }
        )
        selections.append(
            {
                "dataset": "fixture",
                "task": "regression",
                "method": "cars",
                "outer_fold": fold,
                "selector_seed": seed,
                "selected_indices": indices,
            }
        )

    [row] = summarize(records, selections)

    assert row["n_evaluations"] == 4
    assert row["n_outer_folds"] == 2
    assert row["n_selector_seeds"] == 2
    assert row["mean_within_fold_seed_jaccard"] == pytest.approx(2 / 3)


def test_classification_uses_plsda_downstream_and_inside_selectors():
    from benchmarks._models import PLSDA_IMPLEMENTATION, CappedPLSDA

    model, scoring = prediction_configuration("classification")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 6))
    y = np.where(X[:, 0] > 0, "a", "b")
    case = DatasetCase(
        name="binary",
        task="classification",
        X=X,
        y=y,
        wavelengths=np.arange(6, dtype=float),
        source="unit test",
    )
    vip, _ = method_configuration(
        name="vip",
        case=case,
        sizes=[2, 4],
        internal_cv=KFold(3, shuffle=True, random_state=0),
        profile="smoke",
        random_state=0,
        n_jobs=1,
    )

    assert isinstance(model, CappedPLSDA)
    assert scoring == "balanced_accuracy"
    # VIP's internal PLS-DA comes from the same package as the downstream model.
    package = type(vip.estimator).__module__.split(".")[0]
    assert package == PLSDA_IMPLEMENTATION.split(".")[0]
    # The wrapper never asks for more components than selected wavelengths.
    fitted = CappedPLSDA(n_components=5).fit(X[:, :2], y)
    assert fitted.n_components_ == 2
    assert set(fitted.predict(X[:, :2])) <= {"a", "b"}


def test_length_validated_methods_use_the_downstream_model(regression_case):
    cv = KFold(3, shuffle=True, random_state=0)
    for name in ("covsel_auto", "fcovsel", "mivif_auto"):
        selector, grid = method_configuration(
            name=name,
            case=regression_case,
            sizes=[2, 4],
            internal_cv=cv,
            profile="smoke",
            random_state=0,
            n_jobs=1,
        )
        assert selector.n_features_to_select == "auto"
        assert type(selector.estimator).__name__ == "Ridge"
        assert selector.cv is cv
        assert grid == {}


def test_new_published_rules_need_no_size_grid(regression_case):
    cv = KFold(3, shuffle=True, random_state=0)
    mivif, mivif_grid = method_configuration(
        name="mivif",
        case=regression_case,
        sizes=[2, 4],
        internal_cv=cv,
        profile="smoke",
        random_state=0,
        n_jobs=1,
    )
    sr, sr_grid = method_configuration(
        name="selectivity_ratio_ftest",
        case=regression_case,
        sizes=[2, 4],
        internal_cv=cv,
        profile="smoke",
        random_state=0,
        n_jobs=1,
    )

    assert mivif.n_features_to_select is None and mivif_grid == {}
    assert sr.threshold == "f_test" and sr.n_features_to_select is None
    assert set(sr_grid) == {"select__n_components"}


def test_nirbench_task_keeps_its_fixed_partition(tmp_path):
    from benchmarks.run_dataset_matrix import load_dataset_case, outer_split_list

    folder = tmp_path / "datasets" / "18-Barley_sensAIfood_Perten_protein"
    folder.mkdir(parents=True)
    header = ",950,952,954,Protein\n"
    (folder / "data_train.csv").write_text(
        header + "0,0.1,0.2,0.3,10\n1,0.2,0.3,0.4,11\n2,0.3,0.4,0.6,12\n"
    )
    (folder / "data_test.csv").write_text(header + "0,0.4,0.5,0.7,13\n")

    case = load_dataset_case(
        "nirbench_18",
        random_state=0,
        data_home=None,
        lettuce_path=None,
        nirbench_path=tmp_path,
    )

    np.testing.assert_array_equal(case.wavelengths, [950.0, 952.0, 954.0])
    np.testing.assert_array_equal(case.y, [10.0, 11.0, 12.0, 13.0])
    assert case.target_names == ("Protein",)
    ((train, test),) = outer_split_list(case, n_splits=3, random_state=0)
    assert train.tolist() == [0, 1, 2]
    assert test.tolist() == [3]
