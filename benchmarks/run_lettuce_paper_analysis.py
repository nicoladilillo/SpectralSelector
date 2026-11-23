"""Evaluate lettuce preprocessing and outlier-removal configurations.

The script separates manuscript reproduction from prediction-safe analysis.
The ``global`` MSC scope preprocesses and removes outliers before the 70:30
split, matching the order described in the article. The ``training`` scope
learns the MSC reference and outlier model from training data only. The
``batch`` scope computes independent train and test MSC references and is
included only as a transductive sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from benchmarks._datasets import load_lettuce_water_stress
from benchmarks.lettuce_preprocessing import (
    PCAMahalanobisOutlierDetector,
    preprocessing_configurations,
)
from benchmarks.run_lettuce_classification import (
    PLSDA_IMPLEMENTATION,
    classifier_configuration,
    selector_candidates,
)

PAPER_OUTLIER_COUNTS = {
    "SG+MSC": 18,
    "SG+SNV": 20,
    "SG1+MSC": 24,
    "SG1+SNV": 23,
}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write records with stable column order."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def split_strata(target, cycles):
    """Preserve both class and experimental-cycle proportions."""
    return np.asarray(
        [f"{label}|cycle-{cycle}" for label, cycle in zip(target, cycles, strict=True)]
    )


def prepare_global_split(X, y, cycles, preprocessor, remove_outliers, random_state):
    """Reproduce preprocessing and outlier removal before the random split."""
    transformed = preprocessor.fit_transform(X)
    detector = PCAMahalanobisOutlierDetector().fit(transformed)
    keep = detector.inlier_mask_ if remove_outliers else np.ones(len(X), dtype=bool)
    indices = np.flatnonzero(keep)
    train, test = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=split_strata(y[indices], cycles[indices]),
    )
    return (
        transformed[train],
        transformed[test],
        y[train],
        y[test],
        int((~detector.inlier_mask_).sum()),
        0,
    )


def prepare_training_split(
    X,
    y,
    cycles,
    preprocessor,
    remove_outliers,
    random_state,
):
    """Split first and fit preprocessing and the outlier model on training only."""
    indices = np.arange(len(X))
    train, test = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_state,
        stratify=split_strata(y, cycles),
    )
    transformed_train = preprocessor.fit_transform(X[train])
    transformed_test = preprocessor.transform(X[test])
    detector = PCAMahalanobisOutlierDetector().fit(transformed_train)
    keep_train = (
        detector.inlier_mask_
        if remove_outliers
        else np.ones(len(transformed_train), dtype=bool)
    )
    test_flagged = detector.squared_distances(transformed_test) > detector.threshold_
    return (
        transformed_train[keep_train],
        transformed_test,
        y[train][keep_train],
        y[test],
        int((~detector.inlier_mask_).sum()),
        int(test_flagged.sum()),
    )


def plot_results(path: Path, records: list[dict]) -> bool:
    """Plot test accuracy for all preprocessing, outlier, and selector settings."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    methods = ["full_spectrum", "univariate_f", "covsel", "selectivity_ratio"]
    preprocessings = ["SG+MSC", "SG+SNV", "SG1+MSC", "SG1+SNV"]
    colors = {False: "#2A9D8F", True: "#C8553D"}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, preprocessing in zip(axes.flat, preprocessings, strict=True):
        positions = np.arange(len(methods))
        width = 0.36
        for offset, remove_outliers in ((-width / 2, False), (width / 2, True)):
            values = [
                next(
                    row["accuracy"]
                    for row in records
                    if row["preprocessing"] == preprocessing
                    and row["method"] == method
                    and row["outliers_removed"] == remove_outliers
                )
                for method in methods
            ]
            axis.bar(
                positions + offset,
                values,
                width,
                color=colors[remove_outliers],
                label="Outliers removed" if remove_outliers else "All samples",
            )
        axis.set(
            title=preprocessing,
            ylim=(0, 1),
            xticks=positions,
            xticklabels=[method.replace("_", " ") for method in methods],
        )
        axis.tick_params(axis="x", labelrotation=20)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Held-out accuracy")
    axes[1, 0].set_ylabel("Held-out accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=2,
        frameon=False,
    )
    figure.suptitle("Lettuce preprocessing and outlier-removal sensitivity", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def main() -> None:
    """Run the preprocessing and outlier-removal sensitivity analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/lettuce_paper_results"),
    )
    parser.add_argument(
        "--msc-reference-scope",
        choices=("global", "training", "batch"),
        default="training",
        help=(
            "global reproduces whole-dataset preprocessing; training is "
            "prediction-safe; batch computes independent train/test means"
        ),
    )
    parser.add_argument("--sg-window", type=int, default=11)
    parser.add_argument("--sg-polyorder", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--classifier",
        choices=("plsda", "logistic"),
        default="plsda",
        help="Downstream classifier (default: PLS-DA; 'logistic' reproduces "
        "the results recorded in LETTUCE_ANALYSIS.md).",
    )
    args = parser.parse_args()

    dataset = load_lettuce_water_stress(
        args.csv_path,
        wavelength_range=(400, 2500),
    )
    X = dataset.data
    y = dataset.target
    cycles = dataset.sample_metadata["cycle"]
    preprocessor_scope = (
        "training" if args.msc_reference_scope == "global" else args.msc_reference_scope
    )
    preprocessors = preprocessing_configurations(
        window_length=args.sg_window,
        polyorder=args.sg_polyorder,
        msc_reference_scope=preprocessor_scope,
    )
    records = []
    confusion_matrices = {}
    class_labels = np.unique(y).tolist()
    started_total = perf_counter()

    for preprocessing_name, configured_preprocessor in preprocessors.items():
        for remove_outliers in (False, True):
            preprocessor = clone(configured_preprocessor)
            if args.msc_reference_scope == "global":
                prepared = prepare_global_split(
                    X,
                    y,
                    cycles,
                    preprocessor,
                    remove_outliers,
                    args.random_state,
                )
            else:
                prepared = prepare_training_split(
                    X,
                    y,
                    cycles,
                    preprocessor,
                    remove_outliers,
                    args.random_state,
                )
            X_train, X_test, y_train, y_test, train_outliers, test_flagged = prepared

            for method, selector, selector_grid in selector_candidates(
                dataset.wavelengths
            ):
                model, model_grid = classifier_configuration(args.classifier)
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
                    scoring="balanced_accuracy",
                    cv=StratifiedKFold(
                        n_splits=5,
                        shuffle=True,
                        random_state=args.random_state,
                    ),
                    n_jobs=args.n_jobs,
                    refit=True,
                    error_score="raise",
                )
                started = perf_counter()
                search.fit(X_train, y_train)
                elapsed = perf_counter() - started
                predicted = search.predict(X_test)
                fitted_selector = search.best_estimator_.named_steps["select"]
                n_selected = (
                    X.shape[1]
                    if fitted_selector == "passthrough"
                    else int(fitted_selector.selected_indices_.size)
                )
                key = f"{preprocessing_name}|outliers={remove_outliers}|{method}"
                confusion_matrices[key] = confusion_matrix(
                    y_test, predicted, labels=class_labels
                ).tolist()
                records.append(
                    {
                        "msc_reference_scope": args.msc_reference_scope,
                        "preprocessing": preprocessing_name,
                        "outliers_removed": remove_outliers,
                        "method": method,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                        "training_outliers": train_outliers,
                        "test_samples_flagged_not_removed": test_flagged,
                        "paper_reported_global_outliers": PAPER_OUTLIER_COUNTS[
                            preprocessing_name
                        ],
                        "n_selected": n_selected,
                        "accuracy": accuracy_score(y_test, predicted),
                        "balanced_accuracy": balanced_accuracy_score(y_test, predicted),
                        "precision_macro": precision_score(
                            y_test, predicted, average="macro", zero_division=0
                        ),
                        "recall_macro": recall_score(
                            y_test, predicted, average="macro", zero_division=0
                        ),
                        "f1_macro": f1_score(
                            y_test, predicted, average="macro", zero_division=0
                        ),
                        "inner_best_score": search.best_score_,
                        "fit_time_seconds": elapsed,
                        "best_parameters": json.dumps(
                            search.best_params_, sort_keys=True
                        ),
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "results.csv", records)
    (args.output / "confusion_matrices.json").write_text(
        json.dumps({"labels": class_labels, "matrices": confusion_matrices}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    figure_created = plot_results(args.output / "summary.png", records)
    metadata = {
        "source": str(args.csv_path.expanduser().resolve()),
        "source_license": dataset.license,
        "data_shape": list(X.shape),
        "wavelength_range_nm": [400.0, 2500.0],
        "sg_window": args.sg_window,
        "sg_polyorder": args.sg_polyorder,
        "sg_parameter_status": (
            "Assumption: the article does not state window/polyorder; 11/3 "
            "reproduces its global outlier totals but is not uniquely identified."
        ),
        "msc_reference_scope": args.msc_reference_scope,
        "outlier_model": "PCA(10) plus squared Mahalanobis distance",
        "outlier_threshold": 18.307038053275146,
        "test_outlier_policy": (
            "Removed before splitting in global mode; flagged but retained in "
            "training and batch modes."
        ),
        "split": "70:30 class-and-cycle-stratified random holdout",
        "inner_validation": "5-fold stratified CV on training data",
        "classifier": args.classifier,
        "plsda_implementation": (
            PLSDA_IMPLEMENTATION if args.classifier == "plsda" else None
        ),
        "total_runtime_seconds": perf_counter() - started_total,
        "summary_figure_created": figure_created,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved paper preprocessing analysis to {args.output}")
    print(f"Total runtime: {metadata['total_runtime_seconds']:.2f} s")


if __name__ == "__main__":
    main()
