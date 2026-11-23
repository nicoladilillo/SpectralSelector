"""Benchmark wavelength selectors on a user-supplied lettuce CSV.

Experimental cycles are kept disjoint. Each outer fold holds out one complete
cycle, while two-fold group-aware inner CV tunes the selector and the
classifier using only the remaining cycles. The default classifier is PLS-DA
(chemotools' implementation when installed); ``--classifier logistic``
reproduces the earlier logistic-regression protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import (
    CovSelSelector,
    MIVIFSelector,
    SelectivityRatioSelector,
    VIPSelector,
    WavelengthSelectKBest,
)

try:
    from benchmarks._datasets import load_lettuce_water_stress
    from benchmarks._models import PLSDA_IMPLEMENTATION, CappedPLSDA, make_plsda
except ModuleNotFoundError:  # Support direct ``python benchmarks/script.py`` use.
    from _datasets import load_lettuce_water_stress  # type: ignore[no-redef]
    from _models import (  # type: ignore[no-redef]
        PLSDA_IMPLEMENTATION,
        CappedPLSDA,
        make_plsda,
    )


def classifier_configuration(name: str) -> tuple[object, dict[str, list]]:
    """Return the downstream classifier and its tuning grid."""
    if name == "plsda":
        return CappedPLSDA(), {"model__n_components": [1, 2, 3]}
    if name == "logistic":
        return (
            LogisticRegression(max_iter=3000, random_state=42),
            {"model__C": [0.1, 1.0, 10.0]},
        )
    raise ValueError("classifier must be 'plsda' or 'logistic'.")


def selector_candidates(wavelengths):
    """Return the full-spectrum baseline and leakage-safe selector searches."""
    feature_counts = [5, 10, 20]
    return [
        ("full_spectrum", "passthrough", {}),
        (
            "univariate_f",
            WavelengthSelectKBest(
                k=10,
                score_func="f_classif",
                wavelengths=wavelengths,
            ),
            {"select__k": feature_counts},
        ),
        (
            "covsel",
            CovSelSelector(
                n_features_to_select=10,
                task="classification",
                wavelengths=wavelengths,
            ),
            {"select__n_features_to_select": feature_counts},
        ),
        (
            "mivif",
            MIVIFSelector(
                task="classification", random_state=42, wavelengths=wavelengths
            ),
            {},
        ),
        (
            "vip_plsda",
            VIPSelector(
                estimator=make_plsda(),
                n_features_to_select=10,
                wavelengths=wavelengths,
            ),
            {
                "select__n_features_to_select": feature_counts,
                "select__estimator__n_components": [2, 3, 4],
            },
        ),
        (
            "sr_plsda",
            SelectivityRatioSelector(
                estimator=make_plsda(),
                task="classification",
                n_features_to_select=10,
                wavelengths=wavelengths,
            ),
            {
                "select__n_features_to_select": feature_counts,
                "select__estimator__n_components": [2, 3, 4],
            },
        ),
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write records using their first row's stable field order."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_pairwise_jaccard(supports: list[np.ndarray]) -> float:
    """Return exact support agreement across fitted outer-fold selectors."""
    similarities = []
    for first_position, first in enumerate(supports):
        first_set = set(np.asarray(first, dtype=int))
        for second in supports[first_position + 1 :]:
            second_set = set(np.asarray(second, dtype=int))
            union = first_set | second_set
            similarities.append(len(first_set & second_set) / len(union))
    return float(np.mean(similarities)) if similarities else 1.0


def plot_summary(path: Path, summary: list[dict]) -> bool:
    """Save a compact accuracy and selected-variable comparison when available."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    labels = [row["method"].replace("_", " ") for row in summary]
    positions = np.arange(len(summary))
    colors = ["#28536B", "#C8553D", "#2A9D8F", "#6D597A"]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].bar(
        positions,
        [row["mean_accuracy"] for row in summary],
        yerr=[row["std_accuracy"] for row in summary],
        capsize=4,
        color=colors,
    )
    axes[0].set(
        title="Leave-one-cycle-out accuracy",
        ylabel="Accuracy",
        ylim=(0, 1),
        xticks=positions,
        xticklabels=labels,
    )
    axes[1].bar(
        positions,
        [row["mean_n_selected"] for row in summary],
        color=colors,
    )
    axes[1].set(
        title="Variables retained after inner tuning",
        ylabel="Mean number of variables",
        xticks=positions,
        xticklabels=labels,
        yscale="log",
    )
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=25)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def main() -> None:
    """Run leave-one-cycle-out nested evaluation and save fold results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Authorized local dataset CSV")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/lettuce_results")
    )
    parser.add_argument(
        "--wavelength-range",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(400.0, 2500.0),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--classifier",
        choices=("plsda", "logistic"),
        default="plsda",
        help="Downstream classifier (default: PLS-DA).",
    )
    args = parser.parse_args()

    dataset = load_lettuce_water_stress(
        args.csv_path,
        wavelength_range=args.wavelength_range,
    )
    X = dataset.data
    y = dataset.target
    groups = dataset.sample_metadata["cycle"]
    outer_cv = GroupKFold(n_splits=np.unique(groups).size)
    args.output.mkdir(parents=True, exist_ok=True)

    records = []
    selections = {}
    fitted_supports = {}
    confusion_counts = {}
    class_labels = np.unique(y).tolist()
    total_started = perf_counter()
    for method, selector, selector_grid in selector_candidates(dataset.wavelengths):
        selections[method] = []
        fitted_supports[method] = []
        confusion_counts[method] = np.zeros(
            (len(class_labels), len(class_labels)), dtype=int
        )
        for fold, (train, test) in enumerate(
            outer_cv.split(X, y, groups=groups), start=1
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
                cv=GroupKFold(n_splits=2),
                n_jobs=args.n_jobs,
                refit=True,
                error_score="raise",
            )
            fit_arguments = {"groups": groups[train]}
            started = perf_counter()
            search.fit(X[train], y[train], **fit_arguments)
            elapsed = perf_counter() - started
            predicted = search.predict(X[test])
            fitted_selector = search.best_estimator_.named_steps["select"]
            if fitted_selector == "passthrough":
                indices = np.arange(X.shape[1])
                selection_time = 0.0
            else:
                indices = fitted_selector.selected_indices_
                selection_time = fitted_selector.selection_time_
            fitted_supports[method].append(indices)
            selection_record = {
                "outer_fold": fold,
                "held_out_cycle": int(np.unique(groups[test]).item()),
                "n_features": int(indices.size),
            }
            if fitted_selector == "passthrough":
                selection_record["all_features"] = True
            else:
                selection_record["indices"] = indices.tolist()
                selection_record["wavelengths"] = dataset.wavelengths[indices].tolist()
            selections[method].append(selection_record)
            confusion_counts[method] += confusion_matrix(
                y[test], predicted, labels=class_labels
            )
            records.append(
                {
                    "method": method,
                    "outer_fold": fold,
                    "held_out_cycle": int(np.unique(groups[test]).item()),
                    "n_selected": int(indices.size),
                    "accuracy": accuracy_score(y[test], predicted),
                    "balanced_accuracy": balanced_accuracy_score(y[test], predicted),
                    "f1_macro": f1_score(y[test], predicted, average="macro"),
                    "inner_best_score": search.best_score_,
                    "selection_time_seconds": selection_time,
                    "nested_fit_time_seconds": elapsed,
                    "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                }
            )

    write_csv(args.output / "fold_results.csv", records)
    summary = []
    for method in selections:
        rows = [row for row in records if row["method"] == method]
        summary.append(
            {
                "method": method,
                "mean_n_selected": np.mean([row["n_selected"] for row in rows]),
                "mean_accuracy": np.mean([row["accuracy"] for row in rows]),
                "std_accuracy": np.std([row["accuracy"] for row in rows], ddof=1),
                "mean_balanced_accuracy": np.mean(
                    [row["balanced_accuracy"] for row in rows]
                ),
                "mean_f1_macro": np.mean([row["f1_macro"] for row in rows]),
                "mean_pairwise_jaccard": mean_pairwise_jaccard(fitted_supports[method]),
                "total_nested_fit_time_seconds": np.sum(
                    [row["nested_fit_time_seconds"] for row in rows]
                ),
            }
        )
    write_csv(args.output / "summary.csv", summary)
    figure_created = plot_summary(args.output / "summary.png", summary)
    (args.output / "selected_wavelengths.json").write_text(
        json.dumps(selections, indent=2) + "\n", encoding="utf-8"
    )
    confusion_output = {
        "labels": class_labels,
        "matrices": {
            method: matrix.tolist() for method, matrix in confusion_counts.items()
        },
    }
    (args.output / "confusion_matrices.json").write_text(
        json.dumps(confusion_output, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "source": str(args.csv_path.expanduser().resolve()),
        "source_license": dataset.license,
        "data_shape": list(X.shape),
        "wavelength_range": [
            float(dataset.wavelengths[0]),
            float(dataset.wavelengths[-1]),
        ],
        "outer_validation": "leave-one-cycle-out GroupKFold",
        "inner_validation": "two-cycle GroupKFold",
        "classifier": args.classifier,
        "plsda_implementation": (
            PLSDA_IMPLEMENTATION if args.classifier == "plsda" else None
        ),
        "total_runtime_seconds": perf_counter() - total_started,
        "summary_figure_created": figure_created,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved lettuce benchmark to {args.output}")
    print(f"Total runtime: {metadata['total_runtime_seconds']:.2f} s")


if __name__ == "__main__":
    main()
