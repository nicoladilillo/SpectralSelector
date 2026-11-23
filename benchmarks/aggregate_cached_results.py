"""Aggregate persistent benchmark attempts without fitting any model."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

try:
    from benchmarks._result_store import BenchmarkResultStore
except ModuleNotFoundError:  # Support direct ``python benchmarks/script.py`` use.
    from _result_store import BenchmarkResultStore  # type: ignore[no-redef]


def latest_attempts(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only the highest-numbered immutable attempt for every run key."""
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record["run_key"])
        if key not in latest or int(record["attempt"]) > int(latest[key]["attempt"]):
            latest[key] = dict(record)
    return [latest[key] for key in sorted(latest)]


def collect_cached_results(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten latest successes and failures into report-friendly records."""
    successes = []
    failures = []
    for record in latest_attempts(records):
        manifest = record["manifest"]
        identity = {
            "run_key": record["run_key"],
            "attempt": record["attempt"],
            "dataset": manifest.get("dataset", {}).get("name", "unknown"),
            "task": manifest.get("task", "unknown"),
            "method": manifest.get("method", "unknown"),
            "outer_fold": manifest.get("outer_fold", "unknown"),
            "selector_seed": manifest.get("selector_seed", "not applicable"),
            "duration_seconds": record["duration_seconds"],
        }
        if record["status"] == "success":
            payload = record["result"]
            result = payload.get("record", payload)
            row = dict(identity)
            for key, value in result.items():
                row[key] = (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
            for field in ("selected_indices", "selected_wavelengths", "warnings"):
                if field in payload:
                    row[field] = json.dumps(payload[field], sort_keys=True)
            successes.append(row)
        else:
            error = record.get("error", {})
            failures.append(
                identity
                | {
                    "error_type": error.get("type", "unknown"),
                    "error_message": error.get("message", "unknown"),
                }
            )
    return successes, failures


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the union of all result columns."""
    fields = []
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        if not fields:
            return
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Read immutable attempts and emit tables without executing estimators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("benchmarks/.cache/nested")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/cached")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    store = BenchmarkResultStore(args.cache_dir)
    successes, failures = collect_cached_results(store.iter_attempts())
    write_csv(args.output / "cached_results.csv", successes)
    (args.output / "cached_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(successes)} successes and {len(failures)} failures")


if __name__ == "__main__":
    main()
