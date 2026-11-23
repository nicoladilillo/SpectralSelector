import json

import numpy as np
import pytest

from benchmarks._result_store import (
    BenchmarkResultStore,
    BenchmarkSchemaError,
    CachedBenchmarkFailure,
    arrays_fingerprint,
    run_key,
    validate_attempt_record,
)
from benchmarks.aggregate_cached_results import collect_cached_results


def test_identical_success_is_reused_without_retraining(tmp_path):
    store = BenchmarkResultStore(tmp_path)
    calls = []

    def work():
        calls.append("fit")
        return {"score": 0.9, "indices": np.asarray([1, 4])}

    first = store.execute({"dataset": "corn", "method": "vip"}, work)
    second = store.execute({"method": "vip", "dataset": "corn"}, work)

    assert calls == ["fit"]
    assert not first.cached
    assert second.cached
    assert second.result == {"score": 0.9, "indices": [1, 4]}
    assert second.run_key == first.run_key
    assert second.attempt == 1


def test_changed_configuration_creates_a_new_run_key(tmp_path):
    store = BenchmarkResultStore(tmp_path)

    first = store.execute({"components": 2}, lambda: {"score": 1.0})
    second = store.execute({"components": 3}, lambda: {"score": 1.0})

    assert first.run_key != second.run_key
    assert len(list(store.iter_attempts())) == 2


def test_changed_selector_seed_creates_independent_cached_job(tmp_path):
    store = BenchmarkResultStore(tmp_path)

    first = store.execute(
        {"method": "cars", "selector_seed": 42},
        lambda: {"score": 1.0},
    )
    second = store.execute(
        {"method": "cars", "selector_seed": 43},
        lambda: {"score": 1.0},
    )

    assert first.run_key != second.run_key
    assert not first.cached
    assert not second.cached


def test_failure_is_saved_and_only_retried_explicitly(tmp_path):
    store = BenchmarkResultStore(tmp_path)
    manifest = {"dataset": "diesel", "method": "boss"}

    with pytest.raises(RuntimeError, match="failed fit"):
        store.execute(
            manifest, lambda: (_ for _ in ()).throw(RuntimeError("failed fit"))
        )

    with pytest.raises(CachedBenchmarkFailure, match="previously failed"):
        store.execute(manifest, lambda: {"score": 0.1})

    recovered = store.execute(manifest, lambda: {"score": 0.8}, rerun_failed=True)
    records = list(store.iter_attempts())

    assert recovered.attempt == 2
    assert [record["status"] for record in records] == ["failure", "success"]
    assert records[0]["error"]["type"] == "builtins.RuntimeError"


def test_force_appends_attempt_without_overwriting_evidence(tmp_path):
    store = BenchmarkResultStore(tmp_path)
    manifest = {"dataset": "shootout", "method": "cars"}

    store.execute(manifest, lambda: {"score": 0.5})
    forced = store.execute(manifest, lambda: {"score": 0.6}, force=True)

    assert forced.attempt == 2
    assert forced.result["score"] == 0.6
    assert len(list(store.iter_attempts())) == 2


def test_manifest_and_array_hashes_are_stable_and_sensitive():
    left_key, left_manifest = run_key(
        {"parameters": {"alpha": np.float64(1.0), "sizes": (2, 4)}}
    )
    right_key, right_manifest = run_key({"parameters": {"sizes": [2, 4], "alpha": 1.0}})

    assert left_key == right_key
    assert left_manifest == right_manifest
    assert arrays_fingerprint(np.arange(4)) == arrays_fingerprint(np.arange(4))
    assert arrays_fingerprint(np.arange(4)) != arrays_fingerprint(np.arange(5))


def test_array_hashes_object_values_instead_of_memory_addresses():
    first = np.asarray(["arabica", "robusta"], dtype=object)
    second = np.asarray(
        ["".join(["ara", "bica"]), "".join(["ro", "busta"])], dtype=object
    )

    assert first.tolist() == second.tolist()
    assert arrays_fingerprint(first) == arrays_fingerprint(second)


def test_saved_attempt_is_strict_json_and_has_no_temporary_file(tmp_path):
    store = BenchmarkResultStore(tmp_path)
    outcome = store.execute({"job": 1}, lambda: {"metric": np.float64(0.75)})
    run_directory = tmp_path / outcome.run_key
    record_path = run_directory / "attempt-0001.json"

    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["result"] == {"metric": 0.75}
    assert list(run_directory.glob("*.tmp")) == []
    assert not (run_directory / "active.lock").exists()


def test_attempt_schema_rejects_unknown_versions_and_incomplete_records():
    with pytest.raises(BenchmarkSchemaError, match="Migrate or regenerate"):
        validate_attempt_record({"schema_version": 999})

    with pytest.raises(BenchmarkSchemaError, match="missing required fields"):
        validate_attempt_record({"schema_version": 1})


def test_store_rejects_an_unsupported_cached_schema(tmp_path):
    store = BenchmarkResultStore(tmp_path)
    outcome = store.execute({"job": "schema"}, lambda: {"metric": 1.0})
    attempt = tmp_path / outcome.run_key / "attempt-0001.json"
    record = json.loads(attempt.read_text(encoding="utf-8"))
    record["schema_version"] = 2
    attempt.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(BenchmarkSchemaError, match="Unsupported"):
        list(store.iter_attempts())


def test_cached_results_can_be_aggregated_without_running_work(tmp_path):
    store = BenchmarkResultStore(tmp_path)
    store.execute(
        {
            "dataset": {"name": "corn"},
            "task": "regression",
            "method": "vip",
            "selector_seed": 42,
        },
        lambda: {
            "record": {"r2": 0.8, "n_selected": 4},
            "selected_indices": [1, 3, 5, 8],
            "selected_wavelengths": [410.0, 430.0, 450.0, 480.0],
            "warnings": [{"category": "UserWarning", "message": "fixture"}],
        },
    )
    with pytest.raises(ValueError):
        store.execute(
            {
                "dataset": {"name": "corn"},
                "task": "regression",
                "method": "boss",
            },
            lambda: (_ for _ in ()).throw(ValueError("invalid candidate")),
        )

    successes, failures = collect_cached_results(store.iter_attempts())

    assert len(successes) == 1
    assert successes[0]["dataset"] == "corn"
    assert successes[0]["selector_seed"] == 42
    assert successes[0]["selected_indices"] == "[1, 3, 5, 8]"
    assert successes[0]["selected_wavelengths"] == "[410.0, 430.0, 450.0, 480.0]"
    assert "UserWarning" in successes[0]["warnings"]
    assert len(failures) == 1
    assert failures[0]["error_type"] == "builtins.ValueError"
