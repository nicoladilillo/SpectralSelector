"""
Bootstrapping Soft Shrinkage (BOSS) wavelength selection.

Reference
---------
Deng, B.-C., Yun, Y.-H., Cao, D.-S., Yin, Y.-L., Wang, W.-T.,
Lu, H.-M., Luo, Q.-Y., & Liang, Y.-Z. (2016).
A bootstrapping soft shrinkage approach for variable selection in chemical
modeling. Analytica Chimica Acta, 908, 63-74.
https://doi.org/10.1016/j.aca.2016.01.001

Paper algorithm implemented here
--------------------------------
A single complete BOSS run is a population-based iterative procedure.

STEP 1 - Bootstrap sampling (BSS) on the variable space
    Start with equal variable weights. Generate ``n_submodels`` variable
    subsets by sampling variables WITH replacement. In the first iteration,
    the number of replacements equals the total number of variables P.
    Repeated sampled variables are collapsed to unique variables, therefore a
    first-generation subset contains approximately 0.632 * P variables.

STEP 2 - Model population analysis (MPA)
    Fit a PLS model for every generated variable subset and assess it by
    cross-validation. Retain the best ``top_fraction`` of sub-models.

STEP 3 - Update variable weights from regression coefficients
    For regression, the implementation follows the official MATLAB code:

        1. preserve the SIGNED PLS regression-coefficient vector;
        2. normalize each vector to Euclidean unit length;
        3. map each vector back to the complete original variable axis;
        4. sum the retained best-model coefficient vectors;
        5. take the absolute value of that sum;
        6. normalize the resulting weight vector by its Euclidean norm.

    For the PLSDA classification extension, where a single signed RC vector is
    not defined in the same way, L2 coefficient magnitude is used instead.

STEP 4 - Weighted bootstrap sampling (WBS)
    Generate the next population by weighted sampling WITH replacement. The
    number of replacements is determined from the average number of unique
    variables in the preceding population and the official ``speed`` factor::

        floor(mean_nVAR * (1 + 0.1 * speed))

    Consequently, the population shrinks softly rather than eliminating
    variables with a fixed hard threshold.

Steps 2-4 repeat until the generated subsets contain one variable (or
``min_features`` when the user explicitly requests a larger lower bound).
The best-scoring sub-model encountered over the complete trajectory is the
BOSS-selected subset. For regression with the default scorer, maximizing
``neg_root_mean_squared_error`` is exactly equivalent to minimizing RMSECV.

Repeated complete runs
----------------------
The original paper reports repeated executions to evaluate reproducibility and
stability. ``n_runs`` exposes that behavior directly. Every run starts from
uniform weights and an independent random seed.

The final support returned by this class is the optimal subset from the best
complete run, using:

    1. best CV score;
    2. fewer variables as a tie-breaker.

For diagnostics, ``selection_frequency_`` records how often each wavelength
appears in the optimal subset of the independent complete runs. This repeated-
run aggregation is a diagnostic extension; it does not replace the paper's
optimal-submodel rule.

Classification extension
------------------------
The publication defines BOSS with PLS regression. This implementation extends
it to classification with ``PLSDA``. For multi-response PLS coefficients, one
importance value per wavelength is obtained with the L2 norm across response
coefficient columns before the paper's unit-length normalization.

Scaling
-------
Default internal PLS scaling is disabled. For a workflow such as:

    SG1 -> SNV -> BOSS -> StandardScaler -> final model

pass the SG1+SNV spectra to BOSS and autoscale only after wavelength selection.

Computational cost
------------------
The paper gives K=1000 as an example population size. When component count is
also optimized by CV, a faithful BOSS run can therefore be expensive. Use a
smaller ``n_submodels`` for smoke tests, then increase it for final experiments.

Verbose levels
--------------
0 : silent
1 : overall start/final summary
2 : one summary line per complete BOSS run
3 : live sub-model progress bar inside each population iteration, followed by
    one compact iteration summary
4 : level 3 plus detailed subset/CV diagnostics
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y
from tqdm.auto import tqdm

from ._base import BaseWavelengthSelector
from ._wrapper import WrapperSelectorMixin
from .plsda import PLSDA

Task = Literal["auto", "regression", "classification"]


@contextmanager
def _tqdm_joblib(
    progress_bar: tqdm,
) -> Iterator[tqdm]:
    """Update a tqdm bar when joblib parallel batches actually complete.

    ``joblib.Parallel`` normally submits work in batches. Wrapping the input
    generator with tqdm would therefore measure task submission rather than
    completed sub-models. This context manager temporarily replaces Joblib's
    batch-completion callback so the progress bar advances only after workers
    report completed work.

    The original callback is always restored, including when evaluation raises
    an exception. This keeps the integration local to one BOSS population.
    """

    old_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_callback):
        """Joblib callback that forwards completed batch sizes to tqdm."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            progress_bar.update(n=self.batch_size)
            remaining = max(
                0,
                int(progress_bar.total or 0) - int(progress_bar.n),
            )
            progress_bar.set_postfix(
                remaining_models=remaining,
                refresh=False,
            )
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield progress_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        progress_bar.close()


class BOSSSelector(WrapperSelectorMixin, BaseWavelengthSelector):
    """Bootstrapping Soft Shrinkage variable selector.

    Parameters
    ----------
    estimator : PLSRegression or PLSDA, optional
        Optional configured PLS-family estimator. When omitted, the selector
        constructs ``PLSRegression(scale=False)`` for regression or ``PLSDA``
        with its internal scaling disabled for classification. The component
        count is always set internally during CV.

    task : {"auto", "regression", "classification"}, default="auto"
        Prediction task. With ``"auto"`` the target type is inferred from y,
        unless ``estimator`` unambiguously determines the task.

    scoring : str or callable or None, default=None
        Scikit-learn scorer used to rank sub-models. Defaults to
        ``"neg_root_mean_squared_error"`` for regression and ``"accuracy"``
        for classification. For imbalanced classification, explicitly use
        ``scoring="balanced_accuracy"``.

    cv : int, splitter, or -1, default=5
        Internal cross-validation. Integer values create shuffled KFold for
        regression and shuffled StratifiedKFold for classification. ``-1``
        requests LeaveOneOut.

    max_components : int, default=10
        Largest PLS latent-variable count evaluated for every candidate subset.
        The paper used a maximum of 10 latent variables in its experiments.

    n_submodels : int, default=1000
        Number K of bootstrap variable subsets / PLS sub-models in each BOSS
        population. The paper gives K=1000 as an example.

    top_fraction : float or None, default=None
        Fraction of best sub-models used to update the variable weights.

        If ``None``, reproduce the official MATLAB rule:

        - ``n_submodels >= 2000`` -> 5%
        - ``1000 <= n_submodels < 2000`` -> 10%
        - ``500 <= n_submodels < 1000`` -> 20%
        - ``n_submodels < 500`` -> invalid, matching the reference code.

        A custom value in ``(0, 1]`` overrides the MATLAB rule.

    speed : int, default=0
        Official MATLAB BOSS shrinkage-speed parameter. Allowed values are
        integers from 0 to 5. The replacement count for the next WBS
        population is calculated as::

            floor(mean_n_features * (1 + 0.1 * speed))

        Thus ``speed=0`` follows the basic soft-shrinkage schedule, whereas
        larger values retain more replacement draws and therefore shrink the
        variable space more slowly.

    n_runs : int, default=1
        Number of complete independent BOSS executions. The paper repeated
        methods for reproducibility analysis; one run is the core algorithm.

    min_features : int, default=1
        Smallest permitted candidate subset. ``1`` reproduces the paper's
        termination condition. Larger values are a practical extension.

    max_iterations : int or None, default=100
        Maximum number of BOSS population iterations. The official MATLAB
        implementation uses ``for j=1:100`` and terminates earlier when the
        number of retained replacements reaches one. ``None`` is accepted as
        an extension to disable the hard cap.

    random_state : int or None, default=None
        Master random seed.

    n_jobs : int, default=1
        Number of parallel workers used to evaluate population sub-models.
        Internal CV remains serial inside each worker to avoid nested parallel
        oversubscription. ``-1`` uses all available cores.

    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength values corresponding to X columns.

    verbose : int, default=0
        Verbosity level from 0 to 4.

    store_populations : bool, default=False
        If True, store every valid candidate subset from every population.
        This can consume substantial memory when ``n_submodels`` is large.

    Attributes
    ----------
    selected_indices_ : ndarray of int
        Final selected feature indices.

    selected_wavelengths_ : ndarray or None
        Final physical wavelength values when ``wavelengths`` is supplied.

    run_best_indices_ : tuple of ndarrays
        Optimal subset from every complete BOSS run.

    run_best_wavelengths_ : tuple of ndarrays or None
        Physical wavelength values for every run-best subset.

    run_best_scores_ : ndarray
        Best CV score achieved in every complete BOSS run.

    run_best_selection_path_ : ndarray of bool, shape (n_runs, n_features)
        One Boolean row per complete run containing that run's optimal subset.

    selection_frequency_ : ndarray of shape (n_features,)
        Fraction of complete BOSS runs in which each variable appears in the
        run-optimal subset. This is a stability diagnostic.

    weight_path_ : ndarray
        Variable-weight trajectory from the final selected complete run.
        Row 0 contains the initial uniform weights.

    iteration_best_scores_ : ndarray
        Best sub-model score in every population iteration of the selected run.

    iteration_mean_n_features_ : ndarray
        Average number of unique variables per population iteration.

    wavelength_frequency_table_ : pandas.DataFrame
        Per-variable table with occurrence count/frequency, final-run weight,
        and final-selection status.
    """

    def __init__(
        self,
        estimator: PLSRegression | PLSDA | None = None,
        *,
        task: Task = "auto",
        scoring: str | Any | None = None,
        cv: Any = 5,
        max_components: int = 10,
        n_submodels: int = 1000,
        top_fraction: float | None = None,
        speed: int = 0,
        n_runs: int = 1,
        min_features: int = 1,
        max_iterations: int | None = 100,
        random_state: int | None = None,
        n_jobs: int = 1,
        wavelengths: ArrayLike | None = None,
        verbose: int = 0,
        store_populations: bool = False,
    ):
        self.estimator = estimator
        self.task = task
        self.scoring = scoring
        self.cv = cv
        self.max_components = max_components
        self.n_submodels = n_submodels
        self.top_fraction = top_fraction
        self.speed = speed
        self.n_runs = n_runs
        self.min_features = min_features
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.wavelengths = wavelengths
        self.verbose = verbose
        self.store_populations = store_populations

    @property
    def n_bootstraps(self) -> int:
        """Compatibility alias for the population size K."""
        return int(self.n_submodels)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: ArrayLike | None = None,
    ) -> BOSSSelector:
        """Fit BOSS and learn a wavelength support mask."""
        X_array, y_array = check_X_y(
            X,
            y,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
            ensure_min_samples=2,
        )

        self._validate_X(X_array, reset=True)
        self._validate_parameters(X_array.shape)

        groups_array = None
        if groups is not None:
            groups_array = np.asarray(groups)
            if groups_array.shape[0] != X_array.shape[0]:
                raise ValueError("groups must contain one entry for every sample in X.")

        self.task_ = self._resolve_task(y_array)
        self.scoring_ = self._resolve_scoring()
        self.cv_ = self._resolve_cv()

        # Resolve the fraction of population models used for the coefficient
        # weight update. ``None`` reproduces the official MATLAB branching
        # based on num_bootstrap; a numeric value is an explicit extension.
        self.top_fraction_ = self._resolve_top_fraction()
        self.num_best_submodels_ = self._matlab_round_positive(
            self.n_submodels * self.top_fraction_
        )
        self.num_best_submodels_ = max(1, self.num_best_submodels_)

        n_samples, n_features = X_array.shape

        self._log(
            1,
            (
                "Starting BOSS | "
                f"task={self.task_} | samples={n_samples} | "
                f"features={n_features} | runs={self.n_runs} | "
                f"submodels/iteration={self.n_submodels} | "
                f"top_fraction={self.top_fraction_:.3f} | "
                f"num_best={self.num_best_submodels_} | "
                f"speed={self.speed} | "
                f"scoring={self.scoring_}"
            ),
        )

        master_rng = np.random.default_rng(self.random_state)
        run_seeds = master_rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=self.n_runs,
            dtype=np.uint32,
        )

        run_results: list[dict[str, Any]] = []

        # --------------------------------------------------------------
        # Independent complete BOSS executions.
        # --------------------------------------------------------------
        for run_index in range(self.n_runs):
            result = self._single_boss_run(
                X_array,
                y_array,
                groups_array,
                seed=int(run_seeds[run_index]),
                run_index=run_index,
            )
            run_results.append(result)

            self._log(
                2,
                (
                    f"[RUN {run_index + 1:03d}/{self.n_runs}] "
                    f"iterations={result['n_iterations']:2d} | "
                    f"best_iter={result['best_iteration'] + 1:2d} | "
                    f"features={result['best_subset'].size:4d} | "
                    f"LV={result['best_n_components']:2d} | "
                    f"score={result['best_score']:.6f}"
                ),
            )

        # ==================================================================
        # ONE OPTIMAL SUBSET FROM EVERY COMPLETE RUN
        # ==================================================================
        self.run_best_iterations_ = np.asarray(
            [r["best_iteration"] for r in run_results],
            dtype=np.int64,
        )
        self.run_best_scores_ = np.asarray(
            [r["best_score"] for r in run_results],
            dtype=np.float64,
        )
        self.run_best_n_components_ = np.asarray(
            [r["best_n_components"] for r in run_results],
            dtype=np.int64,
        )
        self.run_best_feature_counts_ = np.asarray(
            [r["best_subset"].size for r in run_results],
            dtype=np.int64,
        )
        self.run_best_indices_ = tuple(r["best_subset"].copy() for r in run_results)
        self.run_best_subsets_ = self.run_best_indices_

        if self.wavelengths is not None:
            wavelength_array = np.asarray(self.wavelengths, dtype=np.float64)
            self.run_best_wavelengths_ = tuple(
                wavelength_array[subset].copy() for subset in self.run_best_indices_
            )
        else:
            self.run_best_wavelengths_ = None

        # Boolean best-subset-per-run representation used for stability.
        self.run_best_selection_path_ = np.zeros(
            (self.n_runs, n_features),
            dtype=bool,
        )
        for run_index, subset in enumerate(self.run_best_indices_):
            self.run_best_selection_path_[run_index, subset] = True

        self.selection_path_ = self.run_best_selection_path_
        self.selection_counts_ = np.sum(
            self.run_best_selection_path_,
            axis=0,
            dtype=np.int64,
        )
        self.selection_frequency_ = self.selection_counts_.astype(np.float64) / float(
            self.n_runs
        )
        self.occurrence_counts_ = self.selection_counts_
        self.occurrence_frequency_ = self.selection_frequency_

        # ==================================================================
        # CHOOSE THE FINAL COMPLETE RUN
        # ==================================================================
        # This preserves the BOSS rule that the output is an optimal sub-model,
        # rather than converting the repeated-run frequency profile into a new
        # synthetic subset.
        overall_best = max(
            range(self.n_runs),
            key=lambda i: (
                self.run_best_scores_[i],
                -self.run_best_indices_[i].size,
            ),
        )

        self.overall_best_run_ = int(overall_best)
        self.best_iteration_ = int(self.run_best_iterations_[overall_best])
        self.best_score_ = float(self.run_best_scores_[overall_best])
        self.best_n_components_ = int(self.run_best_n_components_[overall_best])
        self.selected_indices_ = self.run_best_indices_[overall_best].copy()
        self.best_subset_ = self.selected_indices_.copy()

        if self.wavelengths is not None:
            self.selected_wavelengths_ = np.asarray(
                self.wavelengths,
                dtype=np.float64,
            )[self.selected_indices_].copy()
        else:
            self.selected_wavelengths_ = None

        # ==================================================================
        # EXPOSE THE FULL TRAJECTORY OF THE FINAL SELECTED RUN
        # ==================================================================
        selected_run = run_results[overall_best]

        self.weight_path_ = selected_run["weight_path"].copy()
        self.variable_weights_ = selected_run["final_weights"].copy()
        self.iteration_best_scores_ = selected_run["iteration_best_scores"].copy()
        self.score_history_ = self.iteration_best_scores_
        self.iteration_best_n_components_ = selected_run[
            "iteration_best_n_components"
        ].copy()
        self.component_history_ = self.iteration_best_n_components_
        self.iteration_mean_n_features_ = selected_run[
            "iteration_mean_n_features"
        ].copy()
        self.iteration_min_n_features_ = selected_run["iteration_min_n_features"].copy()
        self.iteration_max_n_features_ = selected_run["iteration_max_n_features"].copy()
        self.replacement_history_ = selected_run["replacement_history"].copy()
        self.retained_history_ = self.replacement_history_
        self.iteration_best_subsets_ = tuple(
            x.copy() for x in selected_run["iteration_best_subsets"]
        )
        self.iteration_best_coefficients_ = tuple(
            x.copy() for x in selected_run["iteration_best_coefficients"]
        )
        self.iteration_top_model_counts_ = selected_run[
            "iteration_top_model_counts"
        ].copy()
        self.n_completed_iterations_ = int(selected_run["n_iterations"])

        # Per-run histories are kept as tuples because different stochastic
        # trajectories can terminate after different numbers of iterations.
        self.run_weight_paths_ = tuple(r["weight_path"].copy() for r in run_results)
        self.run_iteration_best_scores_ = tuple(
            r["iteration_best_scores"].copy() for r in run_results
        )
        self.run_iteration_mean_n_features_ = tuple(
            r["iteration_mean_n_features"].copy() for r in run_results
        )
        self.run_replacement_histories_ = tuple(
            r["replacement_history"].copy() for r in run_results
        )

        if self.store_populations:
            self.run_populations_ = tuple(r["populations"] for r in run_results)
        else:
            self.run_populations_ = None

        # ==================================================================
        # FIT FINAL ESTIMATOR ON ALL SELECTION DATA
        # ==================================================================
        self.estimator_ = self._make_estimator(self.best_n_components_)
        self.estimator_.fit(
            X_array[:, self.selected_indices_],
            y_array,
        )

        local_coef = self._coefficient_matrix(
            self.estimator_,
            n_features=self.selected_indices_.size,
        )
        self.final_coefficients_ = np.zeros(
            (n_features, local_coef.shape[1]),
            dtype=np.float64,
        )
        self.final_coefficients_[self.selected_indices_, :] = local_coef
        self.final_importance_ = self._coefficient_importance(self.final_coefficients_)

        # Final ordering prioritizes selected-run weight, then final-model
        # coefficient importance, then lower feature index.
        order = np.lexsort(
            (
                self.selected_indices_,
                -self.final_importance_[self.selected_indices_],
                -self.variable_weights_[self.selected_indices_],
            )
        )
        selection_order = self.selected_indices_[order]

        support_mask = np.zeros(n_features, dtype=bool)
        support_mask[self.selected_indices_] = True

        self._finalize_selection(
            support_mask,
            wavelengths=self.wavelengths,
            selection_order=selection_order,
        )

        # ==================================================================
        # HUMAN-READABLE STABILITY / WEIGHT TABLE
        # ==================================================================
        if self.wavelengths is not None:
            wavelength_values = np.asarray(
                self.wavelengths,
                dtype=np.float64,
            )
        else:
            wavelength_values = np.full(
                n_features,
                np.nan,
                dtype=np.float64,
            )

        self.wavelength_frequency_table_ = pd.DataFrame(
            {
                "feature_index": np.arange(
                    n_features,
                    dtype=np.int64,
                ),
                "wavelength": wavelength_values,
                "selection_count": self.selection_counts_,
                "selection_frequency": self.selection_frequency_,
                "selected_run_final_weight": self.variable_weights_,
                "final_model_importance": self.final_importance_,
                "is_final_selected": support_mask,
            }
        )

        self._log(
            1,
            (
                "BOSS fit complete | "
                f"best_run={self.overall_best_run_ + 1} | "
                f"best_iteration={self.best_iteration_ + 1} | "
                f"final_features={self.selected_indices_.size} | "
                f"final_components={self.best_n_components_} | "
                f"CV_score={self.best_score_:.6f}"
            ),
        )

        if self.verbose >= 2:
            if self.selected_wavelengths_ is not None:
                self._log(
                    2,
                    "Selected wavelengths: "
                    + np.array2string(
                        self.selected_wavelengths_,
                        precision=4,
                        separator=", ",
                    ),
                )
            else:
                self._log(
                    2,
                    "Selected feature indices: "
                    + np.array2string(
                        self.selected_indices_,
                        separator=", ",
                    ),
                )

        return self

    # ======================================================================
    # ONE COMPLETE BOSS EXECUTION
    # ======================================================================

    def _single_boss_run(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        groups: NDArray[Any] | None,
        *,
        seed: int,
        run_index: int,
    ) -> dict[str, Any]:
        """Execute one complete BOSS soft-shrinkage trajectory."""
        rng = np.random.default_rng(seed)
        n_features = X.shape[1]

        # Step 1 starts with equal weights and P replacements.
        weights = np.full(
            n_features,
            1.0 / n_features,
            dtype=np.float64,
        )
        replacement_count = int(n_features)

        weight_history: list[NDArray[np.float64]] = [weights.copy()]
        iteration_best_scores: list[float] = []
        iteration_best_n_components: list[int] = []
        iteration_mean_n_features: list[float] = []
        iteration_min_n_features: list[int] = []
        iteration_max_n_features: list[int] = []
        replacement_history: list[int] = []
        iteration_best_subsets: list[NDArray[np.int64]] = []
        iteration_best_coefficients: list[NDArray[np.float64]] = []
        iteration_top_model_counts: list[int] = []
        stored_populations: list[tuple[NDArray[np.int64], ...]] = []

        best_score = -np.inf
        best_subset: NDArray[np.int64] | None = None
        best_n_components = 1
        best_iteration = -1

        iteration = 0

        while True:
            if self.max_iterations is not None and iteration >= self.max_iterations:
                self._log(
                    2,
                    (
                        f"[RUN {run_index + 1:03d}] reached "
                        f"max_iterations={self.max_iterations} before "
                        "the natural BOSS termination condition."
                    ),
                )
                break

            replacement_history.append(replacement_count)

            # --------------------------------------------------------------
            # STEP 1 / STEP 4: BSS or WBS on VARIABLE SPACE.
            # --------------------------------------------------------------
            # Iteration 1 uses equal weights -> ordinary bootstrap sampling.
            # Later iterations use the coefficient-derived weights -> WBS.
            subsets = tuple(
                self._draw_variable_subset(
                    n_features=n_features,
                    n_replacements=replacement_count,
                    weights=weights,
                    rng=rng,
                )
                for _ in range(self.n_submodels)
            )

            if self.store_populations:
                stored_populations.append(tuple(subset.copy() for subset in subsets))

            subset_sizes = np.fromiter(
                (subset.size for subset in subsets),
                dtype=np.int64,
                count=len(subsets),
            )

            mean_n_features = float(np.mean(subset_sizes))
            min_n_features = int(np.min(subset_sizes))
            max_n_features = int(np.max(subset_sizes))

            iteration_mean_n_features.append(mean_n_features)
            iteration_min_n_features.append(min_n_features)
            iteration_max_n_features.append(max_n_features)

            # --------------------------------------------------------------
            # STEP 2: build and cross-validate K PLS sub-models.
            # --------------------------------------------------------------
            #
            # At verbose >= 3 we display a LIVE progress bar for the current
            # model population. The bar reports models that have genuinely
            # completed, not merely models submitted to Joblib. The postfix
            # explicitly shows the number of sub-models still to evaluate.
            #
            # Example:
            #
            # Run 001 | Iter 03 | submodels:  62%|...| 620/1000
            #     remaining_models=380
            #
            # ``leave=False`` removes the completed bar and lets the compact
            # iteration summary printed below remain as the permanent record.
            progress_desc = (
                f"Run {run_index + 1:03d} | Iter {iteration + 1:02d} | submodels"
            )

            if self.n_jobs == 1:
                # ----------------------------------------------------------
                # SERIAL EVALUATION
                # ----------------------------------------------------------
                # tqdm is updated after each _evaluate_subset call returns,
                # therefore n is exactly the number of completed sub-models.
                progress = None
                if self.verbose >= 3:
                    progress = tqdm(
                        total=len(subsets),
                        desc=progress_desc,
                        unit="model",
                        leave=False,
                        dynamic_ncols=True,
                    )
                    progress.set_postfix(
                        remaining_models=len(subsets),
                        refresh=False,
                    )

                results = []
                try:
                    for model_index, subset in enumerate(subsets, start=1):
                        results.append(
                            self._evaluate_subset(
                                X,
                                y,
                                groups,
                                subset,
                            )
                        )

                        if progress is not None:
                            progress.update(1)
                            progress.set_postfix(
                                remaining_models=(len(subsets) - model_index),
                                refresh=False,
                            )
                finally:
                    if progress is not None:
                        progress.close()

            else:
                # ----------------------------------------------------------
                # PARALLEL EVALUATION
                # ----------------------------------------------------------
                # A normal ``tqdm(subsets)`` wrapper would advance while jobs
                # are submitted, which can make the bar reach 100% before the
                # workers actually finish. ``_tqdm_joblib`` hooks Joblib's
                # batch-completion callback, so each increment represents
                # completed model evaluations.
                with parallel_config(
                    backend="loky",
                    n_jobs=self.n_jobs,
                    inner_max_num_threads=1,
                ):
                    if self.verbose >= 3:
                        progress = tqdm(
                            total=len(subsets),
                            desc=progress_desc,
                            unit="model",
                            leave=False,
                            dynamic_ncols=True,
                        )
                        progress.set_postfix(
                            remaining_models=len(subsets),
                            refresh=False,
                        )

                        with _tqdm_joblib(progress):
                            results = Parallel()(
                                delayed(self._evaluate_subset)(
                                    X,
                                    y,
                                    groups,
                                    subset,
                                )
                                for subset in subsets
                            )
                    else:
                        results = Parallel()(
                            delayed(self._evaluate_subset)(
                                X,
                                y,
                                groups,
                                subset,
                            )
                            for subset in subsets
                        )

            candidates = [result for result in results if result is not None]

            if not candidates:
                raise ValueError(
                    "All BOSS population sub-models failed. Check the "
                    "estimator, CV splitter, class counts, and component "
                    "limits."
                )

            # Larger sklearn score is always better. For regression, the
            # default negative-RMSE scorer therefore reproduces minimum RMSECV.
            candidates.sort(
                key=lambda item: (
                    item["score"],
                    -item["subset"].size,
                ),
                reverse=True,
            )

            iteration_best = candidates[0]
            iteration_score = float(iteration_best["score"])
            iteration_subset = iteration_best["subset"].copy()
            iteration_components = int(iteration_best["n_components"])

            iteration_best_scores.append(iteration_score)
            iteration_best_n_components.append(iteration_components)
            iteration_best_subsets.append(iteration_subset.copy())
            iteration_best_coefficients.append(
                iteration_best["mapped_coefficients"].copy()
            )

            # Global optimal sub-model over the complete trajectory.
            if best_subset is None or (
                iteration_score,
                -iteration_subset.size,
            ) > (
                best_score,
                -best_subset.size,
            ):
                best_score = iteration_score
                best_subset = iteration_subset.copy()
                best_n_components = iteration_components
                best_iteration = int(iteration)

            # --------------------------------------------------------------
            # STEP 3: retain the best fraction and update variable weights.
            # --------------------------------------------------------------
            #
            # Official MATLAB logic:
            #
            #   [rmsecv_sort,index] = sort(rmsecv);
            #   w = abs(sum(B(:,index(1:num_best)),2));
            #   w = w / norm(w);
            #
            # ``num_best`` is computed once from num_bootstrap * ratio using
            # MATLAB-style rounding. Failed Python candidates are excluded,
            # so the count is capped at the number of valid models.
            top_count = min(
                len(candidates),
                self.num_best_submodels_,
            )
            top_count = max(1, int(top_count))
            iteration_top_model_counts.append(top_count)

            top_candidates = candidates[:top_count]

            if self.task_ == "regression":
                # For single-response PLS regression this follows the
                # official MATLAB code closely: preserve the sign of each
                # normalized regression-coefficient vector, SUM the vectors,
                # and only then take the absolute value. This differs subtly
                # from abs(coef) -> normalize -> sum when coefficient signs
                # vary between sub-models.
                summed = np.sum(
                    [
                        candidate["normalized_signed_coefficients"]
                        for candidate in top_candidates
                    ],
                    axis=0,
                    dtype=np.float64,
                )
                new_weights = np.abs(summed)
            else:
                # BOSS was published for regression. PLSDA may expose several
                # response coefficient columns, for which there is no single
                # MATLAB-equivalent signed RC vector. The classification
                # extension therefore uses the L2 coefficient magnitude per
                # wavelength, normalized model-by-model, before aggregation.
                new_weights = np.sum(
                    [
                        candidate["normalized_importance"]
                        for candidate in top_candidates
                    ],
                    axis=0,
                    dtype=np.float64,
                )

            # MATLAB normalizes the updated weight vector by its Euclidean
            # norm. ``randsample`` treats weights relatively; our sampler
            # converts this L2-normalized vector to probabilities only at the
            # moment of drawing.
            weight_norm = float(np.linalg.norm(new_weights, ord=2))

            if not np.isfinite(weight_norm) or weight_norm <= np.finfo(np.float64).eps:
                # Numerical safeguard not specified in the original code.
                # Preserve the preceding valid weight vector rather than
                # creating an undefined sampling distribution.
                new_weights = weights.copy()
            else:
                new_weights = new_weights / weight_norm

            weights = new_weights
            weight_history.append(weights.copy())

            self._log(
                3,
                (
                    f"  [run {run_index + 1:03d} | "
                    f"iter {iteration + 1:02d}] "
                    f"replacements={replacement_count:4d} | "
                    f"mean_nVAR={mean_n_features:8.3f} | "
                    f"range=[{min_n_features},{max_n_features}] | "
                    f"top={top_count:4d}/{len(candidates):4d} | "
                    f"best_nVAR={iteration_subset.size:4d} | "
                    f"LV={iteration_components:2d} | "
                    f"score={iteration_score:.6f}"
                    + (" *GLOBAL BEST*" if best_iteration == iteration else "")
                ),
            )

            # --------------------------------------------------------------
            # NATURAL TERMINATION.
            # --------------------------------------------------------------
            # With one replacement, every generated bootstrap subset contains
            # exactly one unique variable. For min_features > 1, that user-
            # requested lower bound replaces the paper's terminal value of 1.
            if replacement_count <= self.min_features:
                iteration += 1
                break

            if max_n_features <= self.min_features:
                iteration += 1
                break

            # --------------------------------------------------------------
            # STEP 4: replacement count for the next WBS population.
            # --------------------------------------------------------------
            # Official MATLAB code:
            #
            #   num_retained = mean_nVAR * (1 + 0.1 * speed);
            #   num_retained = floor(num_retained);
            #
            # speed=0 therefore uses floor(mean_nVAR), while speed=5 uses
            # floor(1.5 * mean_nVAR). Larger speed values slow the shrinkage.
            speed_factor = 1.0 + 0.1 * float(self.speed)
            next_replacements = int(np.floor(mean_n_features * speed_factor))

            # ``min_features`` is a Python extension. With the paper-faithful
            # default of 1 this reduces exactly to the MATLAB lower bound.
            replacement_count = max(
                self.min_features,
                next_replacements,
            )

            iteration += 1

        if best_subset is None:
            raise RuntimeError(
                "Complete BOSS run did not produce a valid optimal subset."
            )

        return {
            "best_iteration": int(best_iteration),
            "best_score": float(best_score),
            "best_subset": best_subset,
            "best_n_components": int(best_n_components),
            "weight_path": np.stack(weight_history, axis=0),
            "final_weights": weights.copy(),
            "iteration_best_scores": np.asarray(
                iteration_best_scores,
                dtype=np.float64,
            ),
            "iteration_best_n_components": np.asarray(
                iteration_best_n_components,
                dtype=np.int64,
            ),
            "iteration_mean_n_features": np.asarray(
                iteration_mean_n_features,
                dtype=np.float64,
            ),
            "iteration_min_n_features": np.asarray(
                iteration_min_n_features,
                dtype=np.int64,
            ),
            "iteration_max_n_features": np.asarray(
                iteration_max_n_features,
                dtype=np.int64,
            ),
            "replacement_history": np.asarray(
                replacement_history,
                dtype=np.int64,
            ),
            "iteration_best_subsets": tuple(iteration_best_subsets),
            "iteration_best_coefficients": tuple(iteration_best_coefficients),
            "iteration_top_model_counts": np.asarray(
                iteration_top_model_counts,
                dtype=np.int64,
            ),
            "n_iterations": len(iteration_best_scores),
            "populations": (
                tuple(stored_populations) if self.store_populations else None
            ),
        }

    # ======================================================================
    # VARIABLE-SPACE BOOTSTRAP SAMPLING
    # ======================================================================

    def _draw_variable_subset(
        self,
        *,
        n_features: int,
        n_replacements: int,
        weights: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Draw one BSS/WBS variable subset and keep unique variables."""
        # The official MATLAB code stores L2-normalized weights and passes
        # them to randsample as relative sampling weights. numpy.random.choice
        # requires probabilities that sum to one, so normalize only here.
        probability_sum = float(np.sum(weights))
        if (
            not np.isfinite(probability_sum)
            or probability_sum <= np.finfo(np.float64).eps
        ):
            probabilities = np.full(
                n_features,
                1.0 / n_features,
                dtype=np.float64,
            )
        else:
            probabilities = weights / probability_sum

        sampled = rng.choice(
            n_features,
            size=int(n_replacements),
            replace=True,
            p=probabilities,
        )

        subset = np.unique(sampled).astype(
            np.int64,
            copy=False,
        )

        # With the paper-faithful default min_features=1 this branch never
        # changes bootstrap behavior. It only supports an explicit user lower
        # bound larger than one.
        if subset.size < self.min_features:
            order = np.lexsort(
                (
                    np.arange(n_features),
                    -weights,
                )
            )
            supplement = order[: self.min_features]
            subset = np.unique(
                np.concatenate(
                    (
                        subset,
                        supplement,
                    )
                )
            ).astype(np.int64)

        return np.sort(subset)

    # ======================================================================
    # SUB-MODEL EVALUATION / COEFFICIENT WEIGHT UPDATE
    # ======================================================================

    def _evaluate_subset(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        groups: NDArray[Any] | None,
        subset: NDArray[np.int64],
    ) -> dict[str, Any] | None:
        """Evaluate one bootstrap subset and build its normalized RC vector."""
        try:
            score, n_components = self._score_subset(
                X,
                y,
                subset,
                groups,
            )

            estimator = self._make_estimator(n_components)
            estimator.fit(
                X[:, subset],
                y,
            )

            local_coef = self._coefficient_matrix(
                estimator,
                n_features=subset.size,
            )

            local_importance = self._coefficient_importance(local_coef)

            # --------------------------------------------------------------
            # Coefficient vectors used for BOSS weight updating.
            # --------------------------------------------------------------
            # Regression: match the official MATLAB implementation as closely
            # as possible. PLS.regcoef_pretreat is a SIGNED vector, normalized
            # to unit Euclidean norm before being mapped into the full feature
            # space. The absolute value is applied only AFTER summing the best
            # model vectors in _single_boss_run().
            mapped_signed_normalized = np.zeros(
                self.n_features_in_,
                dtype=np.float64,
            )

            if self.task_ == "regression" and local_coef.shape[1] == 1:
                signed_vector = local_coef[:, 0]
                signed_norm = float(np.linalg.norm(signed_vector, ord=2))
                if np.isfinite(signed_norm) and signed_norm > np.finfo(np.float64).eps:
                    mapped_signed_normalized[subset] = signed_vector / signed_norm

            # Classification extension: a multi-response PLSDA coefficient
            # matrix has no single signed coefficient vector equivalent to the
            # MATLAB regression implementation. We therefore preserve the
            # previously documented L2 magnitude extension for classification.
            importance_norm = float(
                np.linalg.norm(
                    local_importance,
                    ord=2,
                )
            )

            if (
                np.isfinite(importance_norm)
                and importance_norm > np.finfo(np.float64).eps
            ):
                local_normalized_importance = local_importance / importance_norm
            else:
                local_normalized_importance = np.zeros_like(
                    local_importance,
                    dtype=np.float64,
                )

            mapped_normalized_importance = np.zeros(
                self.n_features_in_,
                dtype=np.float64,
            )
            mapped_normalized_importance[subset] = local_normalized_importance

            mapped_coefficients = np.zeros(
                (
                    self.n_features_in_,
                    local_coef.shape[1],
                ),
                dtype=np.float64,
            )
            mapped_coefficients[subset, :] = local_coef

            return {
                "score": float(score),
                "subset": subset.copy(),
                "n_components": int(n_components),
                "normalized_signed_coefficients": mapped_signed_normalized,
                "normalized_importance": mapped_normalized_importance,
                "mapped_coefficients": mapped_coefficients,
            }

        except (
            ValueError,
            TypeError,
            AttributeError,
            FloatingPointError,
        ) as exc:
            self._log(
                4,
                (f"      candidate failed | nVAR={subset.size} | {exc}"),
            )
            return None

    # ======================================================================
    # TASK / SCORING / CV
    # ======================================================================

    def _resolve_top_fraction(self) -> float:
        """Resolve the model-retention fraction used by BOSS.

        A user-specified ``top_fraction`` is honored directly. Otherwise the
        branching reproduces the official MATLAB reference implementation.
        """
        if self.top_fraction is not None:
            return float(self.top_fraction)

        if self.n_submodels >= 2000:
            return 0.05
        if self.n_submodels >= 1000:
            return 0.10
        if self.n_submodels >= 500:
            return 0.20

        # Parameter validation normally catches this before fitting.
        raise ValueError(
            "Official MATLAB top-fraction selection requires n_submodels >= 500."
        )

    @staticmethod
    def _matlab_round_positive(value: float) -> int:
        """Round a non-negative scalar like MATLAB ``round``.

        NumPy uses bankers rounding for exact .5 ties. MATLAB rounds positive
        half ties away from zero, equivalent here to floor(x + 0.5).
        """
        if value < 0:
            raise ValueError("value must be non-negative.")
        return int(np.floor(float(value) + 0.5))

    def _resolve_task(
        self,
        y: NDArray[Any],
    ) -> Literal["regression", "classification"]:
        if self.task not in {
            "auto",
            "regression",
            "classification",
        }:
            raise ValueError("task must be 'auto', 'regression', or 'classification'.")

        if self.estimator is not None:
            if isinstance(self.estimator, PLSRegression):
                estimator_task = "regression"
            elif isinstance(self.estimator, PLSDA):
                estimator_task = "classification"
            else:
                raise TypeError("estimator must be PLSRegression, PLSDA, or None.")

            if self.task != "auto" and self.task != estimator_task:
                raise ValueError("task is incompatible with the supplied estimator.")

            if self.task == "auto":
                return estimator_task

        if self.task != "auto":
            return self.task

        target_type = type_of_target(y, raise_unknown=True)

        if target_type in {"binary", "multiclass"}:
            return "classification"

        if target_type == "continuous":
            return "regression"

        raise ValueError(f"Unsupported target type {target_type!r}.")

    def _resolve_cv(self) -> Any:
        if isinstance(self.cv, (int, np.integer)) and not isinstance(self.cv, bool):
            if self.cv == -1:
                return LeaveOneOut()

            if self.cv < 2:
                raise ValueError("cv must be >= 2, a splitter, or -1.")

            if self.task_ == "classification":
                return StratifiedKFold(
                    n_splits=int(self.cv),
                    shuffle=True,
                    random_state=self.random_state,
                )

            return KFold(
                n_splits=int(self.cv),
                shuffle=True,
                random_state=self.random_state,
            )

        return self.cv

    # ======================================================================
    # ESTIMATOR FACTORY
    # ======================================================================

    def _make_estimator(
        self,
        n_components: int,
    ) -> PLSRegression | PLSDA:
        n_components = int(n_components)

        if self.estimator is not None:
            estimator = clone(self.estimator)
            params = estimator.get_params(deep=False)

            if "n_components" in params:
                return estimator.set_params(n_components=n_components)

            if "n_pls_components" in params:
                return estimator.set_params(n_pls_components=n_components)

            raise TypeError(
                "The supplied estimator must expose 'n_components' or "
                "'n_pls_components'."
            )

        if self.task_ == "regression":
            return PLSRegression(
                n_components=n_components,
                scale=False,
            )

        estimator = PLSDA()
        params = estimator.get_params(deep=False)
        updates: dict[str, Any] = {}

        if "n_components" in params:
            updates["n_components"] = n_components
        elif "n_pls_components" in params:
            updates["n_pls_components"] = n_components
        else:
            raise TypeError(
                "PLSDA must expose either 'n_components' or 'n_pls_components'."
            )

        if "scale_pls" in params:
            updates["scale_pls"] = False
        if "scale_lda" in params:
            updates["scale_lda"] = False
        if "x_scaling" in params:
            updates["x_scaling"] = None

        return estimator.set_params(**updates)

    # ======================================================================
    # SUBSET CV / LATENT-VARIABLE OPTIMIZATION
    # ======================================================================

    def _score_subset(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        indices: NDArray[np.int64],
        groups: NDArray[Any] | None,
    ) -> tuple[float, int]:
        """Return the best CV score and PLS component count for a subset."""
        indices = np.asarray(indices, dtype=np.int64)

        if indices.size == 0:
            raise ValueError("Cannot score an empty wavelength subset.")

        max_components = min(
            int(self.max_components),
            int(indices.size),
            max(1, int(X.shape[0] - 1)),
        )

        best_score = -np.inf
        best_n_components: int | None = None

        for n_components in range(1, max_components + 1):
            estimator = self._make_estimator(n_components)

            try:
                fold_scores = cross_val_score(
                    estimator,
                    X[:, indices],
                    y,
                    groups=groups,
                    cv=self.cv_,
                    scoring=self.scoring_,
                    # Outer population parallelism is handled by joblib.
                    n_jobs=1,
                    error_score="raise",
                )
            except ValueError as exc:
                self._log(
                    4,
                    (f"      skip LV={n_components} | nVAR={indices.size} | {exc}"),
                )
                continue

            mean_score = float(np.mean(fold_scores))

            if (
                best_n_components is None
                or mean_score > best_score
                or (
                    np.isclose(mean_score, best_score)
                    and n_components < best_n_components
                )
            ):
                best_score = mean_score
                best_n_components = n_components

        if best_n_components is None:
            raise ValueError("No valid PLS component count could be evaluated.")

        return float(best_score), int(best_n_components)

    def _validate_parameters(
        self,
        shape: tuple[int, int],
    ) -> None:
        _, n_features = shape

        for name in (
            "max_components",
            "n_submodels",
            "n_runs",
            "min_features",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if self.min_features > n_features:
            raise ValueError("min_features cannot exceed input feature count.")

        if self.top_fraction is not None:
            if not np.isscalar(self.top_fraction) or not (
                0.0 < float(self.top_fraction) <= 1.0
            ):
                raise ValueError("top_fraction must lie in (0, 1] or be None.")
        elif self.n_submodels < 500:
            raise ValueError(
                "With top_fraction=None, the official MATLAB BOSS rule "
                "requires n_submodels >= 500. Supply a custom top_fraction "
                "to intentionally use a smaller population."
            )

        if (
            not isinstance(self.speed, (int, np.integer))
            or isinstance(self.speed, bool)
            or not 0 <= int(self.speed) <= 5
        ):
            raise ValueError("speed must be an integer from 0 to 5.")

        if self.max_iterations is not None and (
            not isinstance(
                self.max_iterations,
                (int, np.integer),
            )
            or isinstance(self.max_iterations, bool)
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer or None.")

        if (
            not isinstance(self.n_jobs, (int, np.integer))
            or isinstance(self.n_jobs, bool)
            or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be a non-zero integer.")

        if (
            not isinstance(self.verbose, (int, np.integer))
            or isinstance(self.verbose, bool)
            or self.verbose < 0
        ):
            raise ValueError("verbose must be a non-negative integer.")

        if not isinstance(self.store_populations, (bool, np.bool_)):
            raise ValueError("store_populations must be boolean.")

        if self.estimator is not None and not isinstance(
            self.estimator,
            (PLSRegression, PLSDA),
        ):
            raise TypeError("estimator must be PLSRegression, PLSDA, or None.")

        if self.wavelengths is not None:
            wavelength_array = np.asarray(self.wavelengths)

            if wavelength_array.ndim != 1:
                raise ValueError("wavelengths must be one-dimensional.")

            if wavelength_array.size != n_features:
                raise ValueError("wavelengths length must match X feature count.")
