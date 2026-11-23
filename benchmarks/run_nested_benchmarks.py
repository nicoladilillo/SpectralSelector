"""Run leakage-safe regression and classification selector benchmarks.

Results are written with the Python standard library to keep the benchmark
artifacts explicit and portable.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import (
    CovSelSelector,
    FCovSelSelector,
    IndexSelector,
    MIVIFSelector,
    SelectivityRatioSelector,
    VIPSelector,
    WavelengthSelectKBest,
)
from spectralselector import __version__ as spectralselector_version

try:
    from benchmarks._models import PLSDA_IMPLEMENTATION, CappedPLSDA, make_plsda
    from benchmarks._result_store import (
        BenchmarkResultStore,
        arrays_fingerprint,
        source_tree_fingerprint,
    )
except ModuleNotFoundError:  # Support direct ``python benchmarks/script.py`` use.
    from _models import (  # type: ignore[no-redef]
        PLSDA_IMPLEMENTATION,
        CappedPLSDA,
        make_plsda,
    )
    from _result_store import (  # type: ignore[no-redef]
        BenchmarkResultStore,
        arrays_fingerprint,
        source_tree_fingerprint,
    )


def installed_versions(distributions: tuple[str, ...]) -> dict[str, str]:
    """Return installed versions without requiring optional dependencies."""
    packages = {}
    for distribution in distributions:
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "not installed"
    return packages


def runtime_environment() -> dict[str, object]:
    """Return cache-relevant interpreter, platform, and dependency versions."""
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "spectralselector": spectralselector_version,
            **installed_versions(
                (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "pandas",
                    "joblib",
                    "tqdm",
                    "chemotools",
                )
            ),
        },
    }


def git_state() -> dict[str, str | bool]:
    """Capture repository provenance when Git metadata is available."""
    repository = Path(__file__).resolve().parents[1]

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    status = run("status", "--porcelain")
    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": status not in {"", "unavailable"},
    }


def make_synthetic_spectra(
    *, task: str, n_samples: int, n_features: int, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create smooth spectra and a target with known informative bands."""
    rng = np.random.default_rng(random_state)
    wavelengths = np.linspace(400.0, 1000.0, n_features)
    centers = np.array([475.0, 680.0, 875.0])
    widths = np.array([16.0, 24.0, 20.0])
    profiles = np.exp(
        -0.5 * ((wavelengths[None, :] - centers[:, None]) / widths[:, None]) ** 2
    )

    latent = rng.normal(size=(n_samples, centers.size))
    baseline = rng.normal(scale=0.25, size=(n_samples, 1))
    slope = rng.normal(scale=0.12, size=(n_samples, 1))
    scaled_axis = (wavelengths - wavelengths.mean()) / np.ptp(wavelengths)
    X = latent @ profiles + baseline + slope * scaled_axis
    X += rng.normal(scale=0.18, size=X.shape)

    signal = 1.5 * latent[:, 0] - 1.1 * latent[:, 1] + 1.3 * latent[:, 2]
    if task == "regression":
        y = signal + rng.normal(scale=0.35, size=n_samples)
    elif task == "classification":
        noisy_signal = signal + rng.normal(scale=0.45, size=n_samples)
        y = (noisy_signal > np.median(noisy_signal)).astype(int)
    else:
        raise ValueError("task must be 'regression' or 'classification'.")

    informative = np.any(
        np.abs(wavelengths[:, None] - centers[None, :]) <= widths[None, :], axis=1
    )
    return X, y, wavelengths, informative


def selector_candidates(
    task: str,
    wavelengths: np.ndarray,
    random_state: int,
    candidate_sizes: list[int],
    scoring: str,
    cv: int,
) -> list[tuple[str, object, dict[str, list[object]]]]:
    # Build base items common to both tasks
    rng = np.random.default_rng(random_state)
    random_indices = np.sort(rng.choice(wavelengths.size, size=12, replace=False))
    if task == "regression":
        score_func = "f_regression"
        model_based = VIPSelector(
            n_components=3,
            n_features_to_select=candidate_sizes[0],
            wavelengths=wavelengths,
        )
        model_grid = {"select__n_features_to_select": candidate_sizes}
    else:
        # SR/VIP for classification are configured below with PLS-DA; SR does
        # not accept non-PLS classifier coefficients.
        score_func = "f_classif"
    base_items = [
        ("full_spectrum", "passthrough", {}),
        (
            "random_12",
            IndexSelector(random_indices, wavelengths=wavelengths),
            {},
        ),
        (
            "univariate",
            WavelengthSelectKBest(
                k=candidate_sizes[0],
                score_func=score_func,
                wavelengths=wavelengths,
            ),
            {"select__k": candidate_sizes},
        ),
        *(
            (
                name,
                # Length chosen inside the selector; 20 is clipped to the folds.
                selector_class(
                    n_features_to_select="auto",
                    max_features=20,
                    task=task,
                    scoring=scoring,
                    cv=cv,
                    random_state=random_state,
                    wavelengths=wavelengths,
                ),
                {},
            )
            for name, selector_class in (
                ("covsel", CovSelSelector),
                ("fcovsel", FCovSelSelector),
            )
        ),
        (
            "mivif",
            MIVIFSelector(
                task=task, random_state=random_state, wavelengths=wavelengths
            ),
            {},
        ),
    ]
    if task == "regression":
        base_items.append(("vip", model_based, model_grid))
    else:
        # VIPSelector with PLSDA estimator
        base_items.append(
            (
                "vip_plsda",
                VIPSelector(
                    estimator=make_plsda(),
                    n_features_to_select=candidate_sizes[0],
                    wavelengths=wavelengths,
                ),
                {
                    "select__n_features_to_select": candidate_sizes,
                    "select__estimator__n_components": list(range(2, 9)),
                },
            )
        )
        # SelectivityRatioSelector with PLSDA estimator
        base_items.append(
            (
                "sr_plsda",
                SelectivityRatioSelector(
                    estimator=make_plsda(),
                    task="classification",
                    n_features_to_select=candidate_sizes[0],
                    wavelengths=wavelengths,
                ),
                {
                    "select__n_features_to_select": candidate_sizes,
                    "select__estimator__n_components": list(range(2, 9)),
                },
            )
        )
    return base_items


def pairwise_jaccard(masks: list[np.ndarray]) -> float:
    """Return mean pairwise Jaccard similarity, or one for a single mask."""
    if len(masks) < 2:
        return 1.0
    values = []
    for left_index, left in enumerate(masks[:-1]):
        for right in masks[left_index + 1 :]:
            union = np.logical_or(left, right).sum()
            values.append(np.logical_and(left, right).sum() / union)
    return float(np.mean(values))


def predictive_metrics(task: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate task-appropriate held-out metrics."""
    if task == "regression":
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
            "mae": mean_absolute_error(y_true, y_pred),
        }
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def fit_nested_fold(
    *,
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    informative: np.ndarray,
    task: str,
    method: str,
    fold: int,
    train: np.ndarray,
    test: np.ndarray,
    selector: object,
    selector_grid: dict,
    model: object,
    model_grid: dict,
    scoring: str,
    inner_cv: object,
    n_jobs: int,
) -> dict:
    """Fit one outer-fold job and return its complete serializable result."""
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", selector),
            ("model", model),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {**selector_grid, **model_grid},
        scoring=scoring,
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    started = perf_counter()
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        search.fit(X[train], y[train])
    elapsed = perf_counter() - started
    prediction = search.predict(X[test])

    fitted_selector = search.best_estimator_.named_steps["select"]
    if fitted_selector == "passthrough":
        mask = np.ones(X.shape[1], dtype=bool)
        selection_time = 0.0
        selector_n_components = None
    else:
        mask = fitted_selector.get_support()
        selection_time = fitted_selector.selection_time_
        selector_n_components = getattr(fitted_selector, "best_n_components_", None)
        if selector_n_components is None and hasattr(fitted_selector, "estimator_"):
            selector_n_components = getattr(
                fitted_selector.estimator_, "n_components", None
            )
    overlap = np.logical_and(mask, informative).sum()
    record = {
        "task": task,
        "method": method,
        "outer_fold": fold,
        "n_selected": int(mask.sum()),
        "informative_recall": overlap / informative.sum(),
        "selection_precision": overlap / mask.sum(),
        "selection_time_seconds": selection_time,
        "nested_fit_time_seconds": elapsed,
        "selector_n_components": selector_n_components,
        "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        **predictive_metrics(task, y[test], prediction),
    }
    return {
        "record": record,
        "selected_indices": np.flatnonzero(mask),
        "selected_wavelengths": wavelengths[mask],
        "warnings": [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in captured_warnings
        ],
    }


def run_task(
    *,
    task: str,
    n_samples: int,
    n_features: int,
    outer_splits: int,
    inner_splits: int,
    random_state: int,
    n_jobs: int,
    store: BenchmarkResultStore,
    code_fingerprint: str,
    runtime: dict[str, object],
    provenance: dict[str, str | bool],
    scoring: str,
    force: bool,
    rerun_failed: bool,
) -> tuple[list[dict], dict[str, list[list[int]]], dict[str, int]]:
    """Run one nested-CV benchmark and return fold-level records and masks."""
    X, y, wavelengths, informative = make_synthetic_spectra(
        task=task,
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
    dataset_fingerprint = arrays_fingerprint(X, y, wavelengths, informative)
    if task == "regression":
        outer_cv = (
            LeaveOneOut()
            if outer_splits == -1
            else KFold(outer_splits, shuffle=True, random_state=random_state)
        )
        model = Ridge()
        model_grid = {"model__alpha": [0.1, 1.0, 10.0]}
        split_iterator = outer_cv.split(X, y)
    else:
        outer_cv = (
            LeaveOneOut()
            if outer_splits == -1
            else StratifiedKFold(outer_splits, shuffle=True, random_state=random_state)
        )
        model = CappedPLSDA()
        model_grid = {"model__n_components": [1, 2, 3]}
        split_iterator = outer_cv.split(X, y)

    splits = list(split_iterator)
    records: list[dict] = []
    masks_by_method: dict[str, list[list[int]]] = {}
    cache_statistics = {"reused": 0, "executed": 0}
    smallest_outer_training = min(len(train) for train, _ in splits)
    smallest_inner_training = (
        smallest_outer_training - 1
        if inner_splits == -1
        else int(np.floor(smallest_outer_training * (inner_splits - 1) / inner_splits))
    )
    maximum_selection_size = max(1, smallest_inner_training - 1)
    candidate_sizes = [size for size in (6, 12, 20) if size <= maximum_selection_size]
    if not candidate_sizes:
        candidate_sizes = [maximum_selection_size]

    for method, selector, selector_grid in selector_candidates(
        task,
        wavelengths,
        random_state,
        candidate_sizes,
        scoring,
        inner_splits,
    ):
        masks_by_method[method] = []
        for fold, (train, test) in enumerate(splits, start=1):
            if inner_splits == -1:
                inner_cv = LeaveOneOut()
            elif task == "regression":
                inner_cv = KFold(
                    inner_splits, shuffle=True, random_state=random_state + fold
                )
            else:
                inner_cv = StratifiedKFold(
                    inner_splits, shuffle=True, random_state=random_state + fold
                )
            manifest = {
                "benchmark": "synthetic_nested_cv",
                "benchmark_schema_version": 3,
                "plsda_implementation": PLSDA_IMPLEMENTATION,
                "spectralselector_version": spectralselector_version,
                "code_fingerprint": code_fingerprint,
                "runtime": runtime,
                "git": provenance,
                "dataset": {
                    "name": f"synthetic_{task}",
                    "content_sha256": dataset_fingerprint,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "source": "SpectralSelector deterministic synthetic generator",
                    "citation": "Generated data; no external dataset citation.",
                    "license": "Generated by the MIT-licensed benchmark code.",
                    "target": "synthetic signal",
                    "wavelength_range": [
                        float(np.min(wavelengths)),
                        float(np.max(wavelengths)),
                    ],
                    "preprocessing": "none before the fold-local pipeline",
                },
                "task": task,
                "method": method,
                "outer_fold": fold,
                "split_fingerprint": arrays_fingerprint(train, test),
                "selector": selector,
                "selector_grid": selector_grid,
                "model": model,
                "model_grid": model_grid,
                "scoring": scoring,
                "preprocessing": {
                    "spectral": "none",
                    "outlier_removal": "none",
                    "pipeline_scaler": StandardScaler(),
                },
                "inner_cv": {
                    "class": inner_cv.__class__.__name__,
                    "n_splits": inner_splits,
                    "shuffle": getattr(inner_cv, "shuffle", False),
                    "random_state": getattr(inner_cv, "random_state", None),
                },
                "random_state": random_state,
            }

            outcome = store.execute(
                manifest,
                partial(
                    fit_nested_fold,
                    X=X,
                    y=y,
                    wavelengths=wavelengths,
                    informative=informative,
                    task=task,
                    method=method,
                    fold=fold,
                    train=train,
                    test=test,
                    selector=selector,
                    selector_grid=selector_grid,
                    model=model,
                    model_grid=model_grid,
                    scoring=scoring,
                    inner_cv=inner_cv,
                    n_jobs=n_jobs,
                ),
                force=force,
                rerun_failed=rerun_failed,
            )
            statistic = "reused" if outcome.cached else "executed"
            cache_statistics[statistic] += 1
            records.append(outcome.result["record"])
            masks_by_method[method].append(outcome.result["selected_indices"])
    return records, masks_by_method, cache_statistics


def summarize(records: list[dict], masks: dict[str, list[list[int]]]) -> list[dict]:
    """Aggregate numeric fold metrics and calculate selection stability."""
    summary = []
    keys = sorted({(row["task"], row["method"]) for row in records})
    identifier_fields = {
        "task",
        "method",
        "outer_fold",
        "best_parameters",
        "selector_n_components",
    }
    for task, method in keys:
        rows = [
            row for row in records if row["task"] == task and row["method"] == method
        ]
        numeric_fields = [key for key in rows[0] if key not in identifier_fields]
        result = {"task": task, "method": method, "n_outer_folds": len(rows)}
        for field in numeric_fields:
            values = np.asarray([row[field] for row in rows], dtype=float)
            result[f"mean_{field}"] = float(np.mean(values))
            result[f"std_{field}"] = float(np.std(values, ddof=1))
        key = f"{task}:{method}"
        method_masks = masks[key]
        n_features = len(masks[f"{task}:full_spectrum"][0])
        boolean_masks = []
        for indices in method_masks:
            mask = np.zeros(n_features, dtype=bool)
            mask[indices] = True
            boolean_masks.append(mask)
        result["mean_pairwise_jaccard"] = pairwise_jaccard(boolean_masks)
        summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write records with the union of their fields as a CSV table."""
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Parse command-line settings, run both tasks, and save all artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--n-samples", type=int, default=120)
    parser.add_argument("--n-features", type=int, default=120)
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--regression-scoring",
        default="neg_root_mean_squared_error",
        help="Scikit-learn scorer used for regression tuning.",
    )
    parser.add_argument(
        "--classification-scoring",
        default="balanced_accuracy",
        help="Scikit-learn scorer used for classification tuning.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("benchmarks/.cache/nested"),
        help="Persistent per-fold result store.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse exact successful run keys (default: enabled).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Append a new attempt for every job even when a success exists.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Retry exact run keys whose latest saved attempt failed.",
    )
    args = parser.parse_args()
    for option, value in (
        ("--outer-splits", args.outer_splits),
        ("--inner-splits", args.inner_splits),
    ):
        if value != -1 and value < 2:
            parser.error(f"{option} must be -1 for leave-one-out or at least 2")
    provenance = git_state()
    started_at = datetime.now(timezone.utc).isoformat()
    args.output.mkdir(parents=True, exist_ok=True)
    store = BenchmarkResultStore(args.cache_dir)
    repository = Path(__file__).resolve().parents[1]
    code_fingerprint = source_tree_fingerprint(
        repository,
        ("src/spectralselector/**/*.py", "benchmarks/*.py"),
    )
    runtime = runtime_environment()

    started = perf_counter()
    records = []
    all_masks = {}
    cache_statistics = {"reused": 0, "executed": 0}
    for task in ("regression", "classification"):
        task_records, task_masks, task_cache_statistics = run_task(
            task=task,
            n_samples=args.n_samples,
            n_features=args.n_features,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            store=store,
            code_fingerprint=code_fingerprint,
            runtime=runtime,
            provenance=provenance,
            scoring=(
                args.classification_scoring
                if task == "classification"
                else args.regression_scoring
            ),
            force=args.force or not args.resume,
            rerun_failed=args.rerun_failed,
        )
        records.extend(task_records)
        all_masks.update(
            {f"{task}:{method}": values for method, values in task_masks.items()}
        )
        for key, value in task_cache_statistics.items():
            cache_statistics[key] += value

    write_csv(args.output / "fold_results.csv", records)
    write_csv(args.output / "summary.csv", summarize(records, all_masks))
    (args.output / "selected_indices.json").write_text(
        json.dumps(all_masks, indent=2) + "\n", encoding="utf-8"
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    environment = {
        "benchmark_schema_version": 3,
        "plsda_implementation": PLSDA_IMPLEMENTATION,
        "started_at_utc": started_at,
        "total_runtime_seconds": perf_counter() - started,
        "configuration": configuration,
        "code_fingerprint": code_fingerprint,
        "cache": cache_statistics | {"directory": str(store.root)},
        "git": provenance,
        **runtime,
    }
    (args.output / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved nested-CV results to {args.output}")
    print(
        "Cache: "
        f"{cache_statistics['reused']} reused, "
        f"{cache_statistics['executed']} executed"
    )
    print(f"Total runtime: {environment['total_runtime_seconds']:.2f} s")


if __name__ == "__main__":
    main()
