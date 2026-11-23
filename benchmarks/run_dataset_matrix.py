"""Run all task-compatible selectors across the available dataset catalogue."""

from __future__ import annotations

import argparse
import json
import tracemalloc
import warnings
from dataclasses import dataclass
from functools import partial
from math import prod
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spectralselector import (
    BOSSSelector,
    CARSSelector,
    CovSelSelector,
    FCovSelSelector,
    GeneticAlgorithmSelector,
    IndexSelector,
    MIVIFSelector,
    SelectivityRatioSelector,
    VIPSelector,
    WavelengthSelectKBest,
)
from spectralselector import __version__ as spectralselector_version

try:
    from benchmarks._datasets import (
        fetch_corn,
        fetch_diesel,
        fetch_shootout,
        load_coffee,
        load_fermentation_train,
        load_lettuce_water_stress,
    )
    from benchmarks._models import PLSDA_IMPLEMENTATION, CappedPLSDA, make_plsda
    from benchmarks._result_store import (
        BenchmarkResultStore,
        arrays_fingerprint,
        source_tree_fingerprint,
    )
    from benchmarks.run_nested_benchmarks import (
        git_state,
        make_synthetic_spectra,
        predictive_metrics,
        runtime_environment,
        write_csv,
    )
except ModuleNotFoundError:  # Support direct ``python benchmarks/script.py`` use.
    from _datasets import (  # type: ignore[no-redef]
        fetch_corn,
        fetch_diesel,
        fetch_shootout,
        load_coffee,
        load_fermentation_train,
        load_lettuce_water_stress,
    )
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
    from run_nested_benchmarks import (  # type: ignore[no-redef]
        git_state,
        make_synthetic_spectra,
        predictive_metrics,
        runtime_environment,
        write_csv,
    )


DATASET_NAMES = (
    "synthetic_regression",
    "synthetic_classification",
    "coffee",
    "fermentation_train",
    "corn",
    "diesel",
    "shootout",
    "lettuce_water_stress",
)
# NIRBENCH-DL v0.1 regression tasks (fixed train/test partitions), read from a
# local clone passed with --nirbench-path. Tasks 4, 12 and 14 are left out: their
# CSVs must first be rebuilt from Eigenvector sources. Task id -> (directory,
# citation, license), as recorded in NIRBENCH's DATA_LICENSES.md.
_TOMATO = ("Ibanez et al., doi:10.5281/zenodo.10633732", "CC BY-SA 4.0")
_PERTEN = ("sensAIfood / M. Lagerholm, doi:10.5281/zenodo.15838136", "CC BY 4.0")
_GRAINIT = ("sensAIfood / P. Berzaghi, doi:10.5281/zenodo.15838272", "CC BY 4.0")
_MANGO = ("Anderson et al., doi:10.17632/46htwnp833.5", "CC BY 4.0")
_PEAR_2021 = ("Cruz et al. (2021), doi:10.1016/j.postharvbio.2021.111562", "CC BY 4.0")
_CHEMHOUSE = "Written redistribution permission (ChemHouse)"
NIRBENCH_TASKS = {
    1: (
        "1-Wheat_kernels_protein",
        "Nielsen et al. (2003), doi:10.1094/CCHEM.2003.80.3.274",
        "Written redistribution permission (University of Copenhagen)",
    ),
    2: ("2-Wheat_flours_protein", "D. Bertrand / INRA, ChemProject", _CHEMHOUSE),
    3: (
        "3-Tecator_moisture",
        "Tecator AB, StatLib Tecator dataset",
        "Public domain with attached permission note",
    ),
    5: (
        "5-Cucurbitaceae_fruit_ssc",
        "Kusumiyati et al., doi:10.17632/k55b8mvs84.2",
        "CC BY 4.0",
    ),
    6: ("6-NIR_tomato1_ssc", *_TOMATO),
    7: ("7-NIR_tomato2_ssc", *_TOMATO),
    8: ("8-NIR_tomato3_ssc", *_TOMATO),
    9: ("9-NIR_tomato4_ssc", *_TOMATO),
    10: ("10-NIR_tomato5_ssc", *_TOMATO),
    11: ("11-Olive_oils_c16", "N. Dupuy group, ChemProject", _CHEMHOUSE),
    13: (
        "13-NIR_milk_protein",
        "Diaz-Olivares et al., doi:10.5281/zenodo.8263430",
        "CC BY 4.0",
    ),
    15: ("15-CEOT_pear_2021_brix", *_PEAR_2021),
    16: (
        "16-CEOT_pears_2019_brix",
        "Passos et al. (2019), doi:10.3390/s19235165",
        "CC BY 4.0",
    ),
    17: ("17-Barley_sensAIfood_Perten_moisture", *_PERTEN),
    18: ("18-Barley_sensAIfood_Perten_protein", *_PERTEN),
    19: ("19-Wheat_sensAIfood_Grainit_moisture", *_GRAINIT),
    20: ("20-Wheat_sensAIfood_Grainit_protein", *_GRAINIT),
    # Same samples as task 15, trimmed and SNV-transformed by NIRBENCH.
    21: ("21-CEOT_pear_2021_snv_brix", *_PEAR_2021),
    22: ("22-Mango_S1_dm", *_MANGO),
    23: ("23-Mango_S2_dm", *_MANGO),
    24: ("24-Mango_S3_dm", *_MANGO),
    25: ("25-Mango_S4_dm", *_MANGO),
    26: ("26-Corn_sensAIfood_Perten_moisture", *_PERTEN),
    27: ("27-Corn_sensAIfood_Perten_protein", *_PERTEN),
    28: ("28-Wheat_sensAIfood_Perten_moisture", *_PERTEN),
    29: ("29-Wheat_sensAIfood_Perten_protein", *_PERTEN),
    30: (
        "30-Wheat_sensAIfood_CRAW_protein",
        "sensAIfood / CRA-W, doi:10.5281/zenodo.16108496",
        "CC BY 4.0",
    ),
}
DATASET_NAMES += tuple(f"nirbench_{task}" for task in NIRBENCH_TASKS)
METHOD_NAMES = (
    "full_spectrum",
    "random_subset",
    "univariate",
    "covsel",
    "covsel_auto",
    "fcovsel",
    "mivif",
    "mivif_auto",
    "vip",
    "selectivity_ratio",
    "selectivity_ratio_ftest",
    "cars",
    "boss",
    "genetic",
)
# Methods whose fit runs its own inner validation and therefore needs groups.
WRAPPER_METHODS = {"covsel_auto", "fcovsel", "mivif_auto", "cars", "boss", "genetic"}
# Selectors whose *length* is validated with the downstream prediction model.
LENGTH_VALIDATED_METHODS = frozenset({"covsel_auto", "fcovsel", "mivif_auto"})
STOCHASTIC_METHODS = frozenset({"random_subset", "cars", "boss", "genetic"})

# These are scientific protocol settings, not hidden performance defaults. Keep
# them centralized so reports and cache manifests can state the exact work given
# to each stochastic search.
PROFILE_METHOD_SETTINGS = {
    "smoke": {
        "cars": {"n_runs": 1, "n_iterations": 3},
        "boss": {
            "n_submodels": 8,
            "top_fraction": 0.25,
            "n_runs": 1,
            "max_iterations": 2,
        },
        "genetic": {
            "population_size": 6,
            "generations": 2,
            "tournament_size": 2,
        },
    },
    "publication": {
        "cars": {"n_runs": 50, "n_iterations": 100},
        "boss": {
            "n_submodels": 1000,
            "top_fraction": None,
            "n_runs": 10,
            "max_iterations": 100,
        },
        "genetic": {
            "population_size": 30,
            "generations": 100,
            "tournament_size": 3,
        },
    },
}

MODEL_REGULARIZATION_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
PLSDA_COMPONENT_VALUES = (1, 2, 3, 4, 5)


class NotApplicable(ValueError):
    """Signal a scientifically unsupported dataset/method pair."""


def parse_selector_seeds(
    value: str | None,
    *,
    profile: str,
    fallback: int,
) -> list[int]:
    """Resolve unique fixed seeds for independent stochastic-selector fits."""
    if value is None:
        return (
            list(range(fallback, fallback + 5))
            if profile == "publication"
            else [fallback]
        )
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("selector seeds must be comma-separated integers.") from error
    if not seeds:
        raise ValueError("At least one selector seed is required.")
    if any(seed < 0 or seed > np.iinfo(np.uint32).max for seed in seeds):
        raise ValueError("Selector seeds must be between 0 and 2**32 - 1.")
    return list(dict.fromkeys(seeds))


def method_seeds(name: str, seeds: list[int], fallback: int) -> list[int]:
    """Repeat stochastic methods only; fit deterministic methods once per fold."""
    return seeds if name in STOCHASTIC_METHODS else [fallback]


@dataclass(frozen=True)
class DatasetCase:
    """One row-aligned supervised dataset and its benchmark metadata."""

    name: str
    task: str
    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    source: str
    groups: np.ndarray | None = None
    # Boolean mask of a fixed external test partition (one outer split).
    test_mask: np.ndarray | None = None
    citation: str = "not provided"
    license_name: str = "not provided"
    target_names: tuple[str, ...] = ()
    loader: str = "not provided"
    loader_arguments: tuple[tuple[str, Any], ...] = ()
    preprocessing: str = "none; loader output used without spectral preprocessing"


def _bunch_case(
    dataset,
    *,
    loader: str,
    loader_arguments: tuple[tuple[str, Any], ...] = (),
) -> DatasetCase:
    groups = None
    if dataset.name == "lettuce_water_stress":
        groups = np.asarray(dataset.sample_metadata["cycle"])
    elif dataset.name == "shootout_2002":
        groups = np.asarray(dataset.sample_metadata["split"])
    return DatasetCase(
        name=str(dataset.name),
        task=str(dataset.task),
        X=np.asarray(dataset.data, dtype=np.float64),
        y=np.asarray(dataset.target),
        wavelengths=np.asarray(dataset.wavelengths, dtype=np.float64),
        source=str(dataset.source),
        groups=groups,
        citation=str(dataset.citation),
        license_name=str(dataset.license),
        target_names=tuple(np.asarray(dataset.target_names, dtype=str).reshape(-1)),
        loader=loader,
        loader_arguments=loader_arguments,
    )


def _nirbench_partition(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spectra and target of one NIRBENCH CSV (NIRBENCH src/data_loading.py rules)."""
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="latin1")
    # Drop a leading serialized index or sample identifier; the target is last.
    frame = frame.loc[:, ~frame.columns.str.lower().str.startswith("unnamed")]
    if pd.to_numeric(pd.Series([frame.columns[0]]), errors="coerce").isna()[0]:
        frame = frame.iloc[:, 1:]
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    return frame.iloc[:, :-1], frame.iloc[:, -1]


def load_nirbench_case(name: str, nirbench_path: Path | None) -> DatasetCase:
    """Concatenate a NIRBENCH task's fixed train and test partitions."""
    if nirbench_path is None:
        raise FileNotFoundError(
            f"{name} requires --nirbench-path to a NIRBENCH_DL_v0.1 clone"
        )
    directory, citation, license_name = NIRBENCH_TASKS[int(name.split("_")[1])]
    folder = Path(nirbench_path) / "datasets" / directory
    X_train, y_train = _nirbench_partition(folder / "data_train.csv")
    X_test, y_test = _nirbench_partition(folder / "data_test.csv")
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"{directory}: train and test wavelength axes differ.")
    return DatasetCase(
        name=name,
        task="regression",
        X=np.vstack([X_train.to_numpy(float), X_test.to_numpy(float)]),
        y=np.concatenate([y_train.to_numpy(float), y_test.to_numpy(float)]),
        wavelengths=X_train.columns.to_numpy(float),
        source=f"NIRBENCH-DL v0.1, datasets/{directory}",
        test_mask=np.r_[np.zeros(len(y_train), bool), np.ones(len(y_test), bool)],
        citation=citation,
        license_name=license_name,
        target_names=(str(y_train.name),),
        loader="benchmarks.run_dataset_matrix.load_nirbench_case",
        loader_arguments=(("task", directory), ("split", "fixed train/test")),
    )


def load_dataset_case(
    name: str,
    *,
    random_state: int,
    data_home: Path | None,
    lettuce_path: Path | None,
    nirbench_path: Path | None = None,
) -> DatasetCase:
    """Load one requested dataset without applying preprocessing or splitting."""
    if name.startswith("nirbench_"):
        return load_nirbench_case(name, nirbench_path)
    if name.startswith("synthetic_"):
        task = name.removeprefix("synthetic_")
        X, y, wavelengths, _ = make_synthetic_spectra(
            task=task,
            n_samples=120,
            n_features=120,
            random_state=random_state,
        )
        return DatasetCase(
            name=name,
            task=task,
            X=X,
            y=y,
            wavelengths=wavelengths,
            source="SpectralSelector deterministic synthetic generator",
            citation="Generated benchmark data; no external dataset citation.",
            license_name="Generated at runtime by the MIT-licensed benchmark code.",
            target_names=("target",),
            loader="benchmarks.run_nested_benchmarks.make_synthetic_spectra",
            loader_arguments=(
                ("n_samples", 120),
                ("n_features", 120),
                ("random_state", random_state),
                ("task", task),
            ),
        )
    if name == "coffee":
        return _bunch_case(load_coffee(), loader="benchmarks._datasets.load_coffee")
    if name == "fermentation_train":
        return _bunch_case(
            load_fermentation_train(),
            loader="benchmarks._datasets.load_fermentation_train",
        )
    if name == "corn":
        return _bunch_case(
            fetch_corn(data_home=data_home),
            loader="benchmarks._datasets.fetch_corn",
            loader_arguments=(("instrument", "m5"), ("target", "oil")),
        )
    if name == "diesel":
        return _bunch_case(
            fetch_diesel(data_home=data_home),
            loader="benchmarks._datasets.fetch_diesel",
            loader_arguments=(("subset", "wang2025"), ("target", "BP50")),
        )
    if name == "shootout":
        return _bunch_case(
            fetch_shootout(data_home=data_home),
            loader="benchmarks._datasets.fetch_shootout",
            loader_arguments=(
                ("instrument", 1),
                ("split", "all"),
                ("target", "assay"),
            ),
        )
    if name == "lettuce_water_stress":
        if lettuce_path is None:
            raise FileNotFoundError(
                "lettuce_water_stress requires --lettuce-path to an authorized CSV"
            )
        return _bunch_case(
            load_lettuce_water_stress(lettuce_path),
            loader="benchmarks._datasets.load_lettuce_water_stress",
            loader_arguments=(("wavelength_range", None),),
        )
    raise ValueError(f"Unknown dataset {name!r}.")


def parse_requested(value: str, available: tuple[str, ...], kind: str) -> list[str]:
    """Resolve comma-separated names or the special value ``all``."""
    requested = list(available) if value == "all" else value.split(",")
    unknown = sorted(set(requested).difference(available))
    if unknown:
        raise ValueError(f"Unknown {kind}: {unknown}.")
    return requested


def outer_split_list(
    case: DatasetCase, n_splits: int, random_state: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create reproducible sample- or group-disjoint outer splits."""
    if n_splits != -1 and n_splits < 2:
        raise ValueError("n_splits must be -1 for leave-one-out or at least 2.")
    if case.test_mask is not None:
        return [(np.flatnonzero(~case.test_mask), np.flatnonzero(case.test_mask))]
    if case.groups is not None:
        return list(LeaveOneGroupOut().split(case.X, case.y, case.groups))
    if n_splits == -1:
        return list(LeaveOneOut().split(case.X, case.y))
    if case.task == "classification":
        _, counts = np.unique(case.y, return_counts=True)
        resolved = min(n_splits, int(np.min(counts)))
        if resolved < 2:
            raise ValueError("Classification outer CV requires two samples per class.")
        cv = StratifiedKFold(resolved, shuffle=True, random_state=random_state)
    else:
        resolved = min(n_splits, case.X.shape[0])
        if resolved < 2:
            raise ValueError("Regression outer CV requires at least two samples.")
        cv = KFold(resolved, shuffle=True, random_state=random_state)
    return list(cv.split(case.X, case.y))


def inner_cv_for_fold(
    task: str,
    y: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    random_state: int,
):
    """Construct inner CV and return its smallest training-fold size."""
    if n_splits != -1 and n_splits < 2:
        raise ValueError("n_splits must be -1 for leave-one-out or at least 2.")
    placeholder = np.zeros((len(y), 1))
    if groups is not None:
        n_groups = np.unique(groups).size
        if n_groups < 2:
            raise ValueError("Group-aware inner CV requires at least two groups.")
        cv = (
            LeaveOneGroupOut()
            if n_splits == -1
            else GroupKFold(min(n_splits, n_groups))
        )
        splits = list(cv.split(placeholder, y, groups))
    elif n_splits == -1:
        cv = LeaveOneOut()
        splits = list(cv.split(placeholder, y))
    elif task == "classification":
        _, counts = np.unique(y, return_counts=True)
        resolved = min(n_splits, int(np.min(counts)))
        if resolved < 2:
            raise ValueError("Classification inner CV requires two samples per class.")
        cv = StratifiedKFold(resolved, shuffle=True, random_state=random_state)
        splits = list(cv.split(placeholder, y))
    else:
        resolved = min(n_splits, len(y))
        if resolved < 2:
            raise ValueError("Regression inner CV requires at least two samples.")
        cv = KFold(resolved, shuffle=True, random_state=random_state)
        splits = list(cv.split(placeholder, y))
    return cv, min(len(train) for train, _ in splits)


def candidate_sizes(n_features: int, smallest_training: int) -> list[int]:
    """Return feasible common subset sizes for every inner training fold."""
    maximum = max(1, min(n_features, smallest_training - 1))
    sizes = [size for size in (6, 12, 20) if size <= maximum]
    return sizes or [maximum]


def dataset_provenance(
    case: DatasetCase,
    *,
    content_fingerprint: str,
    runtime: dict[str, object],
) -> dict[str, Any]:
    """Return complete, serializable dataset and loader provenance."""
    packages = runtime["packages"]
    target_summary: dict[str, Any] = {"names": list(case.target_names)}
    if case.task == "classification":
        target_summary["classes"] = np.unique(case.y).tolist()
    else:
        target_summary["n_responses"] = 1 if case.y.ndim == 1 else case.y.shape[1]
    return {
        "name": case.name,
        "task": case.task,
        "shape": list(case.X.shape),
        "content_sha256": content_fingerprint,
        "source": case.source,
        "citation": case.citation,
        "license": case.license_name,
        "loader": case.loader,
        "loader_arguments": dict(case.loader_arguments),
        "loader_versions": {
            "spectralselector": packages["spectralselector"],
            "chemotools": packages["chemotools"],
        },
        "target": target_summary,
        "wavelength_axis": {
            "minimum": float(np.min(case.wavelengths)),
            "maximum": float(np.max(case.wavelengths)),
            "n_features": int(case.wavelengths.size),
            "sha256": arrays_fingerprint(case.wavelengths),
        },
        "groups": {
            "present": case.groups is not None,
            "n_unique": (
                0 if case.groups is None else int(np.unique(case.groups).size)
            ),
        },
        "fixed_test_rows": (
            0 if case.test_mask is None else int(np.count_nonzero(case.test_mask))
        ),
        "preprocessing": case.preprocessing,
    }


def method_configuration(
    *,
    name: str,
    case: DatasetCase,
    sizes: list[int],
    internal_cv: Any,
    profile: str,
    random_state: int,
    n_jobs: int,
    scoring: str | Any | None = None,
) -> tuple[Any, dict[str, list[Any]]]:
    """Construct one task-compatible selector and its tuning grid."""
    first_size = sizes[0]
    n_classes = np.unique(case.y).size if case.task == "classification" else None
    max_components = max(1, min(10, first_size))
    resolved_scoring = scoring or (
        "balanced_accuracy"
        if case.task == "classification"
        else "neg_root_mean_squared_error"
    )
    # Length-validated selectors choose their size with the downstream model.
    length_model = prediction_configuration(case.task)[0]
    if name == "full_spectrum":
        return "passthrough", {}
    if name == "random_subset":
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(case.X.shape[1], size=first_size, replace=False))
        return IndexSelector(indices, wavelengths=case.wavelengths), {}
    if name == "univariate":
        score = "f_classif" if case.task == "classification" else "f_regression"
        return (
            WavelengthSelectKBest(
                k=first_size, score_func=score, wavelengths=case.wavelengths
            ),
            {"select__k": sizes},
        )
    if name == "covsel":
        return (
            CovSelSelector(first_size, task=case.task, wavelengths=case.wavelengths),
            {"select__n_features_to_select": sizes},
        )
    if name in LENGTH_VALIDATED_METHODS:
        selector_class = {
            "covsel_auto": CovSelSelector,
            "fcovsel": FCovSelSelector,
            "mivif_auto": MIVIFSelector,
        }[name]
        return (
            selector_class(
                "auto",
                # The selector clips 20 to what its inner folds allow.
                max_features=20,
                estimator=length_model,
                task=case.task,
                scoring=resolved_scoring,
                cv=internal_cv,
                random_state=random_state,
                wavelengths=case.wavelengths,
            ),
            {},
        )
    if name == "mivif":
        # Published MI-VIF: keep every variable passing VIF <= 10; no model.
        return (
            MIVIFSelector(
                task=case.task,
                random_state=random_state,
                wavelengths=case.wavelengths,
            ),
            {},
        )
    if name == "vip":
        estimator = (
            make_plsda(min(2, max_components))
            if case.task == "classification"
            else None
        )
        selector = VIPSelector(
            estimator=estimator,
            task=case.task,
            n_components=min(3, max_components),
            n_features_to_select=first_size,
            wavelengths=case.wavelengths,
        )
        grid = {"select__n_features_to_select": sizes}
        if estimator is not None:
            grid["select__estimator__n_components"] = list(
                range(1, min(5, max_components) + 1)
            )
        else:
            grid["select__n_components"] = list(range(1, min(5, max_components) + 1))
        return selector, grid
    if name in {"selectivity_ratio", "selectivity_ratio_ftest"}:
        if case.task == "classification" and n_classes != 2:
            raise NotApplicable("SR currently requires one coefficient vector")
        f_test = name == "selectivity_ratio_ftest"
        estimator = (
            make_plsda(min(2, max_components))
            if case.task == "classification"
            else None
        )
        selector = SelectivityRatioSelector(
            estimator=estimator,
            task=case.task,
            n_components=min(3, max_components),
            # The F-test variant lets the published boundary set the size.
            threshold="f_test" if f_test else "mean",
            n_features_to_select=None if f_test else first_size,
            wavelengths=case.wavelengths,
        )
        grid = {} if f_test else {"select__n_features_to_select": sizes}
        if estimator is not None:
            grid["select__estimator__n_components"] = list(
                range(1, min(5, max_components) + 1)
            )
        else:
            grid["select__n_components"] = list(range(1, min(5, max_components) + 1))
        return selector, grid
    if name == "cars":
        settings = PROFILE_METHOD_SETTINGS[profile]["cars"]
        return (
            CARSSelector(
                task=case.task,
                scoring=resolved_scoring,
                cv=internal_cv,
                max_components=max_components,
                random_state=random_state,
                n_jobs=n_jobs,
                wavelengths=case.wavelengths,
                **settings,
            ),
            {},
        )
    if name == "boss":
        settings = PROFILE_METHOD_SETTINGS[profile]["boss"]
        return (
            BOSSSelector(
                task=case.task,
                scoring=resolved_scoring,
                cv=internal_cv,
                max_components=max_components,
                random_state=random_state,
                n_jobs=n_jobs,
                wavelengths=case.wavelengths,
                **settings,
            ),
            {},
        )
    if name == "genetic":
        estimator = (
            make_plsda(min(2, max_components))
            if case.task == "classification"
            else Ridge()
        )
        settings = PROFILE_METHOD_SETTINGS[profile]["genetic"]
        return (
            GeneticAlgorithmSelector(
                estimator,
                task=case.task,
                scoring=resolved_scoring,
                cv=internal_cv,
                min_groups=min(2, max(sizes)),
                max_groups=max(sizes),
                random_state=random_state,
                n_jobs=n_jobs,
                wavelengths=case.wavelengths,
                **settings,
            ),
            {},
        )
    raise ValueError(f"Unknown method {name!r}.")


def prediction_configuration(
    task: str,
    scoring: str | None = None,
) -> tuple[Any, str]:
    """Return the fixed downstream model and inner-search scoring criterion.

    Classification uses PLS-DA (chemotools' implementation when installed, see
    ``benchmarks._models``); regression uses ridge regression.
    """
    if task == "classification":
        return CappedPLSDA(n_components=2), scoring or "balanced_accuracy"
    return Ridge(alpha=1.0), scoring or "neg_root_mean_squared_error"


def prediction_parameter_grid(task: str) -> dict[str, list[float]]:
    """Return the same-sized downstream complexity search for either task.

    Five PLS-DA component counts for classification and five ridge penalties for
    regression keep the downstream tuning budget symmetric.
    """
    if task == "classification":
        return {"model__n_components": list(PLSDA_COMPONENT_VALUES)}
    return {"model__alpha": list(MODEL_REGULARIZATION_VALUES)}


def search_budget(
    *,
    method: str,
    selector: Any,
    selector_grid: dict[str, list[Any]],
    prediction_grid: dict[str, list[Any]],
    n_inner_splits: int,
    n_selector_seeds: int,
) -> dict[str, Any]:
    """Describe comparable outer tuning and intrinsic selector work.

    The intrinsic count is an upper bound on estimator CV fits inside one
    selector fit. Early stopping, invalid component counts, or GA mask-cache
    reuse can reduce the realized count.
    """
    joint_grid = selector_grid | prediction_grid
    outer_candidates = prod(len(values) for values in joint_grid.values())
    pipeline_fits = outer_candidates * n_inner_splits + 1
    intrinsic_cv_fits = 0
    intrinsic_limit = "none"
    if method in LENGTH_VALIDATED_METHODS:
        intrinsic_cv_fits = selector.max_features * n_inner_splits
        intrinsic_limit = "sequence prefixes"
    elif method == "cars":
        intrinsic_cv_fits = (
            selector.n_runs
            * selector.n_iterations
            * selector.max_components
            * n_inner_splits
        )
        intrinsic_limit = "candidate supports and component counts"
    elif method == "boss":
        intrinsic_cv_fits = (
            selector.n_runs
            * selector.max_iterations
            * selector.n_submodels
            * selector.max_components
            * n_inner_splits
        )
        intrinsic_limit = "worst case before shrinkage stopping"
    elif method == "genetic":
        intrinsic_cv_fits = (
            selector.population_size * selector.generations * n_inner_splits
        )
        intrinsic_limit = "distinct-mask upper bound before cache reuse"
    return {
        "selector_grid": selector_grid,
        "prediction_grid": prediction_grid,
        "joint_outer_candidates": outer_candidates,
        "inner_splits": n_inner_splits,
        "pipeline_fits_including_refit": pipeline_fits,
        "intrinsic_selector_cv_fit_upper_bound_per_selector_fit": intrinsic_cv_fits,
        "intrinsic_limit_interpretation": intrinsic_limit,
        "selector_seed_repeats": n_selector_seeds,
        "intrinsic_cv_fit_upper_bound_per_outer_fold_and_seed": (
            intrinsic_cv_fits * pipeline_fits
        ),
    }


def fit_matrix_fold(
    *,
    case: DatasetCase,
    method: str,
    fold: int,
    train: np.ndarray,
    test: np.ndarray,
    selector: Any,
    selector_grid: dict[str, list[Any]],
    prediction_grid: dict[str, list[Any]],
    model: Any,
    scoring: str,
    inner_cv: Any,
    n_jobs: int,
    selector_seed: int,
) -> dict[str, Any]:
    """Fit one cached dataset/method/fold pipeline and return held-out results."""
    pipeline = Pipeline(
        [("scale", StandardScaler()), ("select", selector), ("model", model)]
    )
    search = GridSearchCV(
        pipeline,
        selector_grid | prediction_grid,
        scoring=scoring,
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    fit_arguments = {}
    groups_train = None if case.groups is None else case.groups[train]
    if groups_train is not None:
        fit_arguments["groups"] = groups_train
        if method in WRAPPER_METHODS:
            fit_arguments["select__groups"] = groups_train
    already_tracing = tracemalloc.is_tracing()
    if already_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    started = perf_counter()
    cpu_started = process_time()
    try:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            search.fit(case.X[train], case.y[train], **fit_arguments)
        elapsed = perf_counter() - started
        prediction_started = perf_counter()
        prediction = search.predict(case.X[test])
        prediction_elapsed = perf_counter() - prediction_started
        cpu_elapsed = process_time() - cpu_started
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    fitted_selector = search.best_estimator_.named_steps["select"]
    if fitted_selector == "passthrough":
        mask = np.ones(case.X.shape[1], dtype=bool)
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
    record = {
        "dataset": case.name,
        "task": case.task,
        "method": method,
        "outer_fold": fold,
        "selector_seed": selector_seed,
        "n_train": len(train),
        "n_test": len(test),
        "n_selected": int(mask.sum()),
        "selection_time_seconds": selection_time,
        "nested_fit_time_seconds": elapsed,
        "prediction_time_seconds": prediction_elapsed,
        "main_process_cpu_time_seconds": cpu_elapsed,
        "peak_traced_memory_mebibytes": peak_traced_bytes / (1024**2),
        "selector_n_components": selector_n_components,
        "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        **predictive_metrics(case.task, case.y[test], prediction),
    }
    return {
        "record": record,
        "selected_indices": np.flatnonzero(mask),
        "selected_wavelengths": case.wavelengths[mask],
        "warnings": [
            {
                "category": warning.category.__name__,
                "message": str(warning.message),
            }
            for warning in captured_warnings
        ],
    }


def confidence_interval_95(values: np.ndarray) -> tuple[float, float]:
    """Return a two-sided 95% Student-t confidence interval for fold values."""
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, mean
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    half_width = float(student_t.ppf(0.975, values.size - 1) * standard_error)
    return mean - half_width, mean + half_width


def selection_jaccard(index_sets: list[list[int]]) -> float:
    """Return mean pairwise Jaccard similarity for selected-index sets."""
    if len(index_sets) < 2:
        return 1.0
    sets = [set(indices) for indices in index_sets]
    similarities = []
    for left_index, left in enumerate(sets[:-1]):
        for right in sets[left_index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(similarities))


def summarize(
    records: list[dict[str, Any]],
    selections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate predictive, resource, and support-stability diagnostics."""
    identifiers = {
        "dataset",
        "task",
        "method",
        "outer_fold",
        "selector_seed",
        "best_parameters",
        "selector_n_components",
    }
    output = []
    keys = sorted({(row["dataset"], row["task"], row["method"]) for row in records})
    for dataset, task, method in keys:
        rows = [
            row
            for row in records
            if (row["dataset"], row["task"], row["method"]) == (dataset, task, method)
        ]
        summary = {
            "dataset": dataset,
            "task": task,
            "method": method,
            "n_evaluations": len(rows),
            "n_outer_folds": len({row["outer_fold"] for row in rows}),
            "n_selector_seeds": len({row.get("selector_seed") for row in rows}),
        }
        for field in rows[0]:
            if field in identifiers:
                continue
            if not all(isinstance(row[field], (int, float)) for row in rows):
                continue
            values = np.asarray([row[field] for row in rows], dtype=float)
            summary[f"mean_{field}"] = float(np.mean(values))
            summary[f"std_{field}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            lower, upper = confidence_interval_95(values)
            summary[f"ci95_low_{field}"] = lower
            summary[f"ci95_high_{field}"] = upper

        primary_metric = "r2" if task == "regression" else "balanced_accuracy"
        baselines = {
            row["outer_fold"]: row[primary_metric]
            for row in records
            if row["dataset"] == dataset
            and row["task"] == task
            and row["method"] == "full_spectrum"
        }
        paired_deltas = np.asarray(
            [
                row[primary_metric] - baselines[row["outer_fold"]]
                for row in rows
                if row["outer_fold"] in baselines
            ],
            dtype=float,
        )
        if paired_deltas.size:
            summary["mean_primary_metric_delta_vs_full_spectrum"] = float(
                np.mean(paired_deltas)
            )
            lower, upper = confidence_interval_95(paired_deltas)
            summary["ci95_low_primary_metric_delta_vs_full_spectrum"] = lower
            summary["ci95_high_primary_metric_delta_vs_full_spectrum"] = upper
        if selections is not None:
            fold_indices = [
                selection["selected_indices"]
                for selection in selections
                if (
                    selection["dataset"],
                    selection["task"],
                    selection["method"],
                )
                == (dataset, task, method)
            ]
            if fold_indices:
                summary["mean_pairwise_selection_jaccard"] = selection_jaccard(
                    fold_indices
                )
                within_fold_seed_stability = []
                for outer_fold in sorted(
                    {
                        selection["outer_fold"]
                        for selection in selections
                        if (
                            selection["dataset"],
                            selection["task"],
                            selection["method"],
                        )
                        == (dataset, task, method)
                    }
                ):
                    fold_seed_indices = [
                        selection["selected_indices"]
                        for selection in selections
                        if (
                            selection["dataset"],
                            selection["task"],
                            selection["method"],
                            selection["outer_fold"],
                        )
                        == (dataset, task, method, outer_fold)
                    ]
                    if len(fold_seed_indices) > 1:
                        within_fold_seed_stability.append(
                            selection_jaccard(fold_seed_indices)
                        )
                if within_fold_seed_stability:
                    summary["mean_within_fold_seed_jaccard"] = float(
                        np.mean(within_fold_seed_stability)
                    )
        output.append(summary)

    for dataset, task in sorted({(row["dataset"], row["task"]) for row in output}):
        metric = "mean_r2" if task == "regression" else "mean_balanced_accuracy"
        group = [
            row for row in output if row["dataset"] == dataset and row["task"] == task
        ]
        ordered = sorted(group, key=lambda row: (-row[metric], row["method"]))
        for rank, row in enumerate(ordered, start=1):
            row["predictive_rank_within_dataset"] = rank
            row["predictive_rank_metric"] = metric.removeprefix("mean_")
    return output


def main() -> None:
    """Execute or resume the requested dataset-by-method matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", "--only-dataset", dest="datasets", default="all")
    parser.add_argument("--methods", "--only-method", dest="methods", default="all")
    parser.add_argument("--profile", choices=("smoke", "publication"), default="smoke")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/matrix")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("benchmarks/.cache/matrix")
    )
    parser.add_argument("--data-home", type=Path)
    parser.add_argument("--lettuce-path", type=Path)
    parser.add_argument(
        "--nirbench-path",
        type=Path,
        help="Local clone of github.com/DeepLightLab/NIRBENCH_DL_v0.1",
    )
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
    parser.add_argument(
        "--selector-seeds",
        help=(
            "Comma-separated independent seeds for stochastic selectors. "
            "Defaults to random-state for smoke and five consecutive seeds "
            "for the publication profile."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse exact successful run keys (default: enabled).",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()
    for option, value in (
        ("--outer-splits", args.outer_splits),
        ("--inner-splits", args.inner_splits),
    ):
        if value != -1 and value < 2:
            parser.error(f"{option} must be -1 for leave-one-out or at least 2")
    try:
        selector_seeds = parse_selector_seeds(
            args.selector_seeds,
            profile=args.profile,
            fallback=args.random_state,
        )
    except ValueError as error:
        parser.error(str(error))

    datasets = parse_requested(args.datasets, DATASET_NAMES, "datasets")
    methods = parse_requested(args.methods, METHOD_NAMES, "methods")
    args.output.mkdir(parents=True, exist_ok=True)
    store = BenchmarkResultStore(args.cache_dir)
    repository = Path(__file__).resolve().parents[1]
    code_fingerprint = source_tree_fingerprint(
        repository, ("src/spectralselector/**/*.py", "benchmarks/*.py")
    )
    provenance = git_state()
    runtime = runtime_environment()
    records: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    loaded_datasets: list[dict[str, Any]] = []
    cache_statistics = {"reused": 0, "executed": 0}

    for dataset_name in datasets:
        try:
            case = load_dataset_case(
                dataset_name,
                random_state=args.random_state,
                data_home=args.data_home,
                lettuce_path=args.lettuce_path,
                nirbench_path=args.nirbench_path,
            )
        except Exception as error:
            skipped.append(
                {
                    "dataset": dataset_name,
                    "method": "all",
                    "reason": f"dataset unavailable: {error}",
                }
            )
            continue

        dataset_fingerprint = arrays_fingerprint(
            case.X,
            case.y,
            case.wavelengths,
            np.asarray([]) if case.groups is None else case.groups,
        )
        provenance_record = dataset_provenance(
            case,
            content_fingerprint=dataset_fingerprint,
            runtime=runtime,
        )
        loaded_datasets.append(provenance_record)
        splits = outer_split_list(case, args.outer_splits, args.random_state)
        for fold, (train, test) in enumerate(splits, start=1):
            groups_train = None if case.groups is None else case.groups[train]
            inner_cv, smallest_training = inner_cv_for_fold(
                case.task,
                case.y[train],
                groups_train,
                args.inner_splits,
                args.random_state + fold,
            )
            sizes = candidate_sizes(case.X.shape[1], smallest_training)
            requested_scoring = (
                args.classification_scoring
                if case.task == "classification"
                else args.regression_scoring
            )
            model, scoring = prediction_configuration(
                case.task,
                requested_scoring,
            )
            prediction_grid = prediction_parameter_grid(case.task)
            n_inner_splits = inner_cv.get_n_splits(
                case.X[train], case.y[train], groups_train
            )
            for method in methods:
                seeds_for_method = method_seeds(
                    method,
                    selector_seeds,
                    args.random_state,
                )
                for selector_seed in seeds_for_method:
                    try:
                        selector, selector_grid = method_configuration(
                            name=method,
                            case=case,
                            sizes=sizes,
                            internal_cv=inner_cv,
                            profile=args.profile,
                            random_state=selector_seed,
                            n_jobs=args.n_jobs,
                            scoring=scoring,
                        )
                    except NotApplicable as error:
                        skipped.append(
                            {
                                "dataset": case.name,
                                "method": method,
                                "selector_seed": selector_seed,
                                "reason": str(error),
                            }
                        )
                        continue
                    manifest = {
                        "benchmark": "dataset_method_matrix",
                        "benchmark_schema_version": 4,
                        "spectralselector_version": spectralselector_version,
                        "code_fingerprint": code_fingerprint,
                        "git": provenance,
                        "runtime": runtime,
                        "dataset": provenance_record,
                        "task": case.task,
                        "method": method,
                        "profile": args.profile,
                        "outer_fold": fold,
                        "selector_seed": selector_seed,
                        "split_fingerprint": arrays_fingerprint(train, test),
                        "selector": selector,
                        "selector_grid": selector_grid,
                        "prediction_grid": prediction_grid,
                        "search_budget": search_budget(
                            method=method,
                            selector=selector,
                            selector_grid=selector_grid,
                            prediction_grid=prediction_grid,
                            n_inner_splits=n_inner_splits,
                            n_selector_seeds=len(seeds_for_method),
                        ),
                        "preprocessing": {
                            "spectral": "none",
                            "outlier_removal": "none",
                            "pipeline_scaler": StandardScaler(),
                        },
                        "prediction_model": model,
                        "plsda_implementation": PLSDA_IMPLEMENTATION,
                        "scoring": scoring,
                        "inner_cv": inner_cv,
                        "split_random_state": args.random_state,
                        "n_jobs": args.n_jobs,
                    }
                    try:
                        outcome = store.execute(
                            manifest,
                            partial(
                                fit_matrix_fold,
                                case=case,
                                method=method,
                                fold=fold,
                                train=train,
                                test=test,
                                selector=selector,
                                selector_grid=selector_grid,
                                prediction_grid=prediction_grid,
                                model=model,
                                scoring=scoring,
                                inner_cv=inner_cv,
                                n_jobs=args.n_jobs,
                                selector_seed=selector_seed,
                            ),
                            force=args.force or not args.resume,
                            rerun_failed=args.rerun_failed,
                        )
                    except Exception as error:
                        # The result store has already preserved the traceback.
                        # Keep traversing so one failed job cannot hide later jobs.
                        skipped.append(
                            {
                                "dataset": case.name,
                                "method": method,
                                "outer_fold": fold,
                                "selector_seed": selector_seed,
                                "reason": f"job failed: {error}",
                            }
                        )
                        continue
                    cache_statistics["reused" if outcome.cached else "executed"] += 1
                    payload = outcome.result
                    records.append(payload["record"])
                    selections.append(
                        {
                            "dataset": case.name,
                            "task": case.task,
                            "method": method,
                            "outer_fold": fold,
                            "selector_seed": selector_seed,
                            "selected_indices": payload["selected_indices"],
                            "selected_wavelengths": payload["selected_wavelengths"],
                            "warnings": payload["warnings"],
                        }
                    )

    write_csv(args.output / "fold_results.csv", records)
    write_csv(
        args.output / "summary.csv",
        summarize(records, selections) if records else [],
    )
    (args.output / "selections.json").write_text(
        json.dumps(selections, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Private data locations are recorded by folder name only, never as
    # absolute local paths (benchmarks/RESULTS_POLICY.md).
    private_paths = {"data_home", "lettuce_path", "nirbench_path"}
    configuration = {
        key: (value.name if key in private_paths else str(value))
        if isinstance(value, Path)
        else value
        for key, value in vars(args).items()
    }
    configuration["resolved_selector_seeds"] = selector_seeds
    matrix_manifest = {
        "schema_version": 4,
        "configuration": configuration,
        "datasets": datasets,
        "methods": methods,
        "profile": args.profile,
        "code_fingerprint": code_fingerprint,
        "git": provenance,
        "runtime": runtime,
        "dataset_provenance": loaded_datasets,
        "cache": cache_statistics | {"directory": str(args.cache_dir)},
        "skipped": skipped,
        "plsda_implementation": PLSDA_IMPLEMENTATION,
        "notes": {
            "classifier": (
                "Classification uses PLS-DA as the downstream model and inside "
                "VIP, SR, GA, and the length-validated selectors; CARS and BOSS "
                "build spectralselector.PLSDA internally."
            ),
            "selector_repeats": (
                "Only random_subset, CARS, BOSS, and genetic are repeated over "
                "selector seeds; deterministic methods are fitted once per fold."
            ),
            "fermentation_test": (
                "Excluded because spectra and HPLC reference rows are time-aligned, "
                "not one-to-one supervised samples."
            ),
        },
    }
    (args.output / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(records)} fold results; "
        f"cache {cache_statistics['reused']} reused/"
        f"{cache_statistics['executed']} executed; {len(skipped)} skipped"
    )


if __name__ == "__main__":
    main()
