"""
Repeated Competitive Adaptive Reweighted Sampling (CARS).

Reference
---------
Li, H., Liang, Y., Xu, Q., & Cao, D. (2009).
Key wavelengths screening using competitive adaptive reweighted sampling
method for multivariate calibration.
Analytica Chimica Acta, 648(1), 77-84.
https://doi.org/10.1016/j.aca.2009.06.046

Paper algorithm implemented here
--------------------------------
There are TWO nested levels.

OUTER LEVEL: ``n_runs`` complete independent CARS executions.

For each complete run:

    restart from all wavelengths

    for ``n_iterations`` sequential CARS iterations:

        select ``sample_fraction`` of ALL calibration samples
            ->
        fit PLS / PLSDA on the currently surviving wavelengths
            ->
        calculate coefficient-derived wavelength importance
            ->
        EDF forced reduction
            ->
        ARS weighted sampling with replacement
            ->
        cross-validate the resulting wavelength subset

    retain ONE best inner iteration:
        best CV metric;
        if tied, fewer wavelengths.

After all complete runs:

    best subset from run 1
    best subset from run 2
    ...
    best subset from run R
        ->
    count how often every wavelength occurs
        ->
    frequency = count / n_runs
        ->
    scipy.signal.find_peaks
        ->
    final peak wavelength(s)

By default ``peak_selection="best"`` retains only the single frequency peak
with the largest occurrence frequency.

Important terminology
---------------------
``selection_path_``
    Shape ``(n_runs, n_features)``.
    Each row is ONLY the best subset from one complete outer CARS run.
    This is the matrix used to calculate ``selection_frequency_``.

``iteration_selection_path_``
    Shape ``(n_runs, n_iterations, n_features)``.
    Full post-ARS survival history for every inner iteration.

``coefficients_path_``
    Shape ``(n_runs, n_iterations, n_features, n_responses)``.
    Full signed coefficient history from every step.

``run_best_indices_``
    Tuple of feature-index arrays. Entry ``r`` contains the wavelengths selected
    by the best inner iteration of complete outer run ``r``.

``run_best_wavelengths_``
    Tuple of physical-wavelength arrays corresponding to
    ``run_best_indices_``. Available when ``wavelengths`` is supplied.

``selected_indices_`` / ``selected_wavelengths_``
    Final feature indices and, when available, physical wavelength values after
    repeated-run frequency aggregation and peak selection.

``wavelength_frequency_table_``
    A pandas DataFrame with one row per original wavelength containing the
    repeated-run occurrence count/frequency, mean run-best importance, detected
    peak status, and final-selection status.

Original CARS vs extensions
---------------------------
The original CARS paper defines PLS-regression CARS and describes repeated
executions for stability/frequency analysis.

The following are extensions in this implementation:
- PLSDA classification;
- stratified Monte Carlo sampling for classification;
- multiclass coefficient importance using an L2 norm;
- scipy ``find_peaks`` post-processing of repeated-run frequencies.

Scaling
-------
Internal PLS scaling is disabled. If the experimental workflow is:

    SG1 -> SNV -> CARS -> StandardScaler -> final model

pass SG1+SNV spectra to this selector and autoscale only after selection.

Computational cost
------------------
Each complete run evaluates up to ``n_iterations`` candidate supports. Every
candidate searches the valid PLS component counts across all CV folds, so cost
grows approximately with ``n_runs * n_iterations * max_components * n_folds``
PLS fits, in addition to the coefficient fit used by each sampling iteration.

Verbose levels
--------------
0 : silent
1 : overall start/final summary
2 : one summary line per complete outer run
3 : one line per inner iteration
4 : detailed Monte Carlo / EDF / ARS / CV diagnostics
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.signal import find_peaks
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._wrapper import WrapperSelectorMixin
from .plsda import PLSDA

Task = Literal["auto", "regression", "classification"]
PeakSelection = Literal["best", "all"]


class CARSSelector(WrapperSelectorMixin, BaseWavelengthSelector):
    """Repeated CARS wavelength selector.

    Parameters
    ----------
    task : {"auto", "regression", "classification"}, default="auto"
        Prediction task.

    scoring : str or callable or None, default=None
        Candidate-subset CV scorer.

        Defaults:
        - regression: ``"neg_root_mean_squared_error"``
        - classification: ``"accuracy"``

    cv : int, splitter, or -1, default=5
        Internal CV used for candidate scoring and PLS-component optimization.
        ``-1`` requests leave-one-out validation.

    max_components : int, default=10
        Largest PLS component count considered.

    n_runs : int, default=50
        Number of complete independent CARS executions.

    n_iterations : int, default=100
        Sequential CARS iterations inside every complete run.

    sample_fraction : float, default=0.8
        Fraction of ALL calibration samples selected for coefficient
        estimation at every inner iteration.

    min_features : int, default=2
        Minimum number of wavelengths allowed during an inner trajectory.

    peak_selection : {"best", "all"}, default="best"
        ``"best"`` keeps only the highest-ranked frequency peak.
        ``"all"`` keeps all detected peaks, optionally limited by
        ``max_peaks``.

    peak_distance : int, default=1
        Minimum feature-index separation for ``scipy.signal.find_peaks``.

    peak_prominence : float or None, default=None
        Optional peak prominence.

    peak_min_frequency : float, default=0.0
        Minimum repeated-run selection frequency required for a peak.

    max_peaks : int or None, default=None
        Maximum retained peaks when ``peak_selection="all"``.

    random_state : int or None, default=None
        Master random seed.

    n_jobs : int, default=1
        Jobs used by internal ``cross_val_score``.

    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength metadata.

    verbose : int, default=0
        Verbosity level from 0 to 4.
    """

    def __init__(
        self,
        *,
        task: Task = "auto",
        scoring: str | Any | None = None,
        cv: Any = 5,
        max_components: int = 10,
        n_runs: int = 50,
        n_iterations: int = 100,
        sample_fraction: float = 0.8,
        min_features: int = 2,
        peak_selection: PeakSelection = "best",
        peak_distance: int = 1,
        peak_prominence: float | None = None,
        peak_min_frequency: float = 0.0,
        max_peaks: int | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        wavelengths: ArrayLike | None = None,
        verbose: int = 0,
    ):
        self.task = task
        self.scoring = scoring
        self.cv = cv
        self.max_components = max_components
        self.n_runs = n_runs
        self.n_iterations = n_iterations
        self.sample_fraction = sample_fraction
        self.min_features = min_features
        self.peak_selection = peak_selection
        self.peak_distance = peak_distance
        self.peak_prominence = peak_prominence
        self.peak_min_frequency = peak_min_frequency
        self.max_peaks = max_peaks
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.wavelengths = wavelengths
        self.verbose = verbose

    @property
    def n_sampling_runs(self) -> int:
        """Compatibility alias for code that previously used this name.

        In the repeated-CARS implementation this refers to the number of
        COMPLETE outer runs, i.e. ``n_runs``.
        """
        return int(self.n_runs)

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: ArrayLike | None = None,
    ) -> CARSSelector:
        """Fit repeated CARS."""
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

        n_samples, n_features = X_array.shape

        self._log(
            1,
            (
                "Starting repeated CARS | "
                f"task={self.task_} | "
                f"samples={n_samples} | "
                f"features={n_features} | "
                f"runs={self.n_runs} | "
                f"iterations/run={self.n_iterations} | "
                f"sample_fraction={self.sample_fraction:.3f} | "
                f"scoring={self.scoring_}"
            ),
        )

        # Independent deterministic seed for each COMPLETE CARS run.
        master_rng = np.random.default_rng(self.random_state)
        run_seeds = master_rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=self.n_runs,
            dtype=np.uint32,
        )

        run_results: list[dict[str, Any]] = []

        # ==============================================================
        # OUTER LOOP
        # ==============================================================
        for run_index in range(self.n_runs):
            result = self._single_cars_run(
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
                    f"best_iter={result['best_iteration'] + 1:03d} | "
                    f"features={result['best_subset'].size:4d} | "
                    f"LV={result['best_n_components']:2d} | "
                    f"score={result['best_score']:.6f}"
                ),
            )

        # ==============================================================
        # FULL INNER HISTORIES
        # ==============================================================

        # Shape:
        # (n_runs, n_iterations, n_features, n_responses)
        self.coefficients_path_ = np.stack(
            [r["coefficients"] for r in run_results],
            axis=0,
        )
        self.coefficient_path_ = self.coefficients_path_

        # Shape:
        # (n_runs, n_iterations, n_features)
        self.importance_path_ = np.stack(
            [r["importance"] for r in run_results],
            axis=0,
        )

        self.iteration_selection_path_ = np.stack(
            [r["selection_path"] for r in run_results],
            axis=0,
        )

        self.iteration_edf_selection_path_ = np.stack(
            [r["edf_selection_path"] for r in run_results],
            axis=0,
        )

        self.iteration_ars_probability_path_ = np.stack(
            [r["ars_probability_path"] for r in run_results],
            axis=0,
        )

        # Full inner-iteration EDF/ARS histories.  These aliases are kept for
        # compatibility with diagnostic/export code written for earlier
        # versions of CARSSelector.
        self.edf_selection_path_ = self.iteration_edf_selection_path_
        self.ars_probability_path_ = self.iteration_ars_probability_path_

        self.iteration_score_history_ = np.stack(
            [r["scores"] for r in run_results],
            axis=0,
        )

        self.iteration_n_features_history_ = np.stack(
            [r["n_features"] for r in run_results],
            axis=0,
        )

        self.iteration_n_components_history_ = np.stack(
            [r["n_components"] for r in run_results],
            axis=0,
        )

        self.coefficient_components_history_ = np.stack(
            [r["coefficient_components"] for r in run_results],
            axis=0,
        )

        self.run_subsets_ = tuple(r["subsets"] for r in run_results)
        self.run_edf_subsets_ = tuple(r["edf_subsets"] for r in run_results)
        self.run_sample_indices_ = tuple(r["sample_indices"] for r in run_results)

        # Compatibility-friendly average trajectories for plotting.
        self.score_history_ = np.mean(
            self.iteration_score_history_,
            axis=0,
        )
        self.n_features_history_ = np.mean(
            self.iteration_n_features_history_,
            axis=0,
        )

        if self.scoring_ == "neg_root_mean_squared_error":
            self.rmsecv_history_ = -self.score_history_
        else:
            self.rmsecv_history_ = None

        # ==============================================================
        # ONE BEST SUBSET FROM EACH COMPLETE OUTER RUN
        # ==============================================================
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

        self.run_best_subsets_ = tuple(r["best_subset"].copy() for r in run_results)

        # ``run_best_subsets_`` is retained for backwards compatibility.
        # ``run_best_indices_`` is the explicit public name: each tuple entry
        # contains the ORIGINAL feature indices selected by the best inner
        # iteration of one complete outer CARS run.
        self.run_best_indices_ = tuple(
            subset.copy() for subset in self.run_best_subsets_
        )

        # Store physical wavelength values directly as well, when wavelength
        # metadata was provided at construction time.  This means users can
        # inspect:
        #
        #     selector.run_best_indices_[r]
        #     selector.run_best_wavelengths_[r]
        #
        # without manually indexing the original wavelength vector.
        if self.wavelengths is not None:
            wavelength_array = np.asarray(
                self.wavelengths,
                dtype=np.float64,
            )
            self.run_best_wavelengths_ = tuple(
                wavelength_array[indices].copy() for indices in self.run_best_indices_
            )
        else:
            self.run_best_wavelengths_ = None

        self.run_best_selection_path_ = np.zeros(
            (self.n_runs, n_features),
            dtype=bool,
        )

        for run_index, subset in enumerate(self.run_best_indices_):
            self.run_best_selection_path_[
                run_index,
                subset,
            ] = True

        # IMPORTANT compatibility definition:
        # selection_path_ is now the BEST-SUBSET-PER-RUN matrix used for
        # occurrence frequency, not the inner trajectory matrix.
        self.selection_path_ = self.run_best_selection_path_

        # ==============================================================
        # REFIT EVERY RUN-BEST SUBSET ON ALL FITTING SAMPLES
        # ==============================================================
        best_coefficients_per_run: list[NDArray[np.float64]] = []
        best_importance_per_run: list[NDArray[np.float64]] = []

        for run_index, subset in enumerate(self.run_best_subsets_):
            estimator = self._make_estimator(
                int(self.run_best_n_components_[run_index])
            )

            estimator.fit(
                X_array[:, subset],
                y_array,
            )

            local_coef = self._coefficient_matrix(
                estimator,
                n_features=subset.size,
            )

            full_coef = np.zeros(
                (n_features, local_coef.shape[1]),
                dtype=np.float64,
            )
            full_coef[subset, :] = local_coef

            full_importance = self._importance_from_coefficients(full_coef)

            best_coefficients_per_run.append(full_coef)
            best_importance_per_run.append(full_importance)

        self.run_best_coefficients_ = np.stack(
            best_coefficients_per_run,
            axis=0,
        )

        self.run_best_importance_ = np.stack(
            best_importance_per_run,
            axis=0,
        )

        # ==============================================================
        # OCCURRENCE FREQUENCY:
        # ONLY BEST SUBSET OF EACH COMPLETE RUN CONTRIBUTES
        # ==============================================================
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
        self.peak_profile_ = self.selection_frequency_.copy()

        # Mean importance only over runs in which a wavelength was selected.
        importance_sum = np.sum(
            self.run_best_importance_,
            axis=0,
        )

        self.mean_selected_importance_ = np.divide(
            importance_sum,
            self.selection_counts_,
            out=np.zeros(
                n_features,
                dtype=np.float64,
            ),
            where=self.selection_counts_ > 0,
        )

        # ==============================================================
        # OVERALL BEST RUN - diagnostic only
        # ==============================================================
        overall_best = max(
            range(self.n_runs),
            key=lambda i: (
                self.run_best_scores_[i],
                -self.run_best_subsets_[i].size,
            ),
        )

        self.overall_best_run_ = int(overall_best)
        self.overall_best_subset_ = self.run_best_subsets_[overall_best].copy()
        self.overall_best_score_ = float(self.run_best_scores_[overall_best])

        # Compatibility attributes used by earlier quickstarts.
        self.best_iteration_ = int(self.run_best_iterations_[overall_best])
        self.cars_subset_ = self.overall_best_subset_.copy()
        self.trajectory_best_score_ = self.overall_best_score_
        self.trajectory_best_n_components_ = int(
            self.run_best_n_components_[overall_best]
        )
        self.best_coefficients_ = self.run_best_coefficients_[overall_best].copy()
        self.best_importance_ = self.run_best_importance_[overall_best].copy()

        # ==============================================================
        # SCIPY FREQUENCY-PEAK SELECTION
        # ==============================================================
        self.peak_indices_ = self._select_frequency_peaks(self.selection_frequency_)

        # Explicit final-selection names.  ``peak_indices_`` remains available
        # because it describes how the final subset was obtained, while
        # ``selected_indices_`` is clearer for downstream code.
        self.selected_indices_ = self.peak_indices_.copy()

        if self.wavelengths is not None:
            wavelength_array = np.asarray(
                self.wavelengths,
                dtype=np.float64,
            )
            self.selected_wavelengths_ = wavelength_array[self.selected_indices_].copy()
        else:
            self.selected_wavelengths_ = None

        # Highest-ranked detected peak, even if peak_selection="all".
        if self.detected_peak_indices_.size:
            ranked_detected = self._rank_peaks(self.detected_peak_indices_)
            self.best_peak_index_ = int(ranked_detected[0])
        else:
            self.best_peak_index_ = int(np.argmax(self.selection_frequency_))

        self.best_peak_frequency_ = float(
            self.selection_frequency_[self.best_peak_index_]
        )

        # --------------------------------------------------------------
        # Human-readable repeated-run frequency table.
        # --------------------------------------------------------------
        # This table is deliberately created AFTER peak detection so that it
        # can include both ``is_detected_peak`` and ``is_final_selected``.
        detected_peak_mask = np.zeros(
            n_features,
            dtype=bool,
        )
        detected_peak_mask[self.detected_peak_indices_] = True

        final_selection_mask = np.zeros(
            n_features,
            dtype=bool,
        )
        final_selection_mask[self.selected_indices_] = True

        if self.wavelengths is not None:
            table_wavelengths = np.asarray(
                self.wavelengths,
                dtype=np.float64,
            )
        else:
            # Preserve a stable table schema even when only feature indices
            # are known.
            table_wavelengths = np.full(
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
                "wavelength": table_wavelengths,
                "selection_count": self.selection_counts_,
                "selection_frequency": self.selection_frequency_,
                "mean_selected_importance": self.mean_selected_importance_,
                "is_detected_peak": detected_peak_mask,
                "is_final_selected": final_selection_mask,
            }
        )

        self._log(
            1,
            (
                "Repeated-run frequency aggregation | "
                f"detected_peaks={self.detected_peak_indices_.size} | "
                f"final_selected={self.peak_indices_.size} | "
                f"best_peak_index={self.best_peak_index_} | "
                f"best_peak_frequency={self.best_peak_frequency_:.4f}"
            ),
        )

        if self.verbose >= 2:
            if self.wavelengths is not None:
                w = np.asarray(self.wavelengths)
                text = np.array2string(
                    w[self.peak_indices_],
                    precision=4,
                    separator=", ",
                )
                self._log(
                    2,
                    "Final wavelength peak(s): " + text,
                )
            else:
                self._log(
                    2,
                    (
                        "Final feature peak(s): "
                        + np.array2string(
                            self.peak_indices_,
                            separator=", ",
                        )
                    ),
                )

        # ==============================================================
        # FINAL MODEL ON SELECTED FREQUENCY PEAK(S)
        # ==============================================================
        (
            self.best_score_,
            self.best_n_components_,
        ) = self._score_subset(
            X_array,
            y_array,
            self.selected_indices_,
            groups_array,
        )

        self.estimator_ = self._make_estimator(self.best_n_components_)

        self.estimator_.fit(
            X_array[:, self.selected_indices_],
            y_array,
        )

        local_final_coef = self._coefficient_matrix(
            self.estimator_,
            n_features=self.selected_indices_.size,
        )

        self.final_coefficients_ = np.zeros(
            (n_features, local_final_coef.shape[1]),
            dtype=np.float64,
        )
        self.final_coefficients_[
            self.selected_indices_,
            :,
        ] = local_final_coef

        self.final_importance_ = self._importance_from_coefficients(
            self.final_coefficients_
        )

        # Final selected-feature ranking.
        order = np.lexsort(
            (
                self.peak_indices_,
                -self.final_importance_[self.peak_indices_],
                -self.mean_selected_importance_[self.peak_indices_],
                -self.selection_frequency_[self.peak_indices_],
            )
        )

        selection_order = self.peak_indices_[order]

        mask = np.zeros(
            n_features,
            dtype=bool,
        )
        mask[self.peak_indices_] = True

        self._finalize_selection(
            mask,
            wavelengths=self.wavelengths,
            selection_order=selection_order,
        )

        self._log(
            1,
            (
                "CARS fit complete | "
                f"final_features={self.peak_indices_.size} | "
                f"final_components={self.best_n_components_} | "
                f"final_CV_score={self.best_score_:.6f}"
            ),
        )

        return self

    # ==================================================================
    # ONE COMPLETE OUTER CARS RUN
    # ==================================================================

    def _single_cars_run(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        groups: NDArray[Any] | None,
        *,
        seed: int,
        run_index: int,
    ) -> dict[str, Any]:
        """Perform one complete CARS trajectory."""
        rng = np.random.default_rng(seed)

        n_samples, n_features = X.shape

        # Every complete CARS run restarts from ALL wavelengths.
        current = np.arange(
            n_features,
            dtype=np.int64,
        )

        subsets: list[NDArray[np.int64]] = []
        edf_subsets: list[NDArray[np.int64]] = []
        sample_indices: list[NDArray[np.int64]] = []
        coefficient_history: list[NDArray[np.float64]] = []
        importance_history: list[NDArray[np.float64]] = []

        scores = np.full(
            self.n_iterations,
            -np.inf,
            dtype=np.float64,
        )

        n_features_history = np.zeros(
            self.n_iterations,
            dtype=np.int64,
        )

        n_components_history = np.zeros(
            self.n_iterations,
            dtype=np.int64,
        )

        coefficient_components = np.zeros(
            self.n_iterations,
            dtype=np.int64,
        )

        selection_path = np.zeros(
            (self.n_iterations, n_features),
            dtype=bool,
        )

        edf_selection_path = np.zeros(
            (self.n_iterations, n_features),
            dtype=bool,
        )

        ars_probability_path = np.zeros(
            (self.n_iterations, n_features),
            dtype=np.float64,
        )

        best_iteration = -1
        best_score = -np.inf
        best_subset: NDArray[np.int64] | None = None
        best_n_components = 1

        # ==============================================================
        # INNER CARS ITERATIONS
        # ==============================================================
        for iteration in range(self.n_iterations):
            active_before = current.size

            # 1. Determine PLS complexity on current wavelength space.
            (
                _,
                current_n_components,
            ) = self._score_subset(
                X,
                y,
                current,
                groups,
            )

            # 2. Sample sample_fraction of ALL samples.
            rows = self._sample_rows(
                X,
                y,
                rng,
            )
            sample_indices.append(rows.copy())

            mc_n_components = min(
                current_n_components,
                current.size,
                max(1, rows.size - 1),
            )

            coefficient_components[iteration] = int(mc_n_components)

            self._log(
                4,
                (
                    f"    run={run_index + 1:03d} "
                    f"iter={iteration + 1:03d} | "
                    f"MC={rows.size}/{n_samples} | "
                    f"active={active_before} | "
                    f"coef_LV={mc_n_components}"
                ),
            )

            # 3. Coefficient-estimation model.
            estimator = self._make_estimator(mc_n_components)

            estimator.fit(
                X[rows][:, current],
                y[rows],
            )

            local_coef = self._coefficient_matrix(
                estimator,
                n_features=current.size,
            )

            local_importance = self._importance_from_coefficients(local_coef)

            full_coef = np.zeros(
                (n_features, local_coef.shape[1]),
                dtype=np.float64,
            )
            full_coef[current, :] = local_coef

            full_importance = np.zeros(
                n_features,
                dtype=np.float64,
            )
            full_importance[current] = local_importance

            coefficient_history.append(full_coef)
            importance_history.append(full_importance)

            # 4. EDF.
            edf_keep = min(
                self._edf_keep_count(
                    iteration=iteration,
                    n_features=n_features,
                ),
                current.size,
            )

            local_order = np.lexsort(
                (
                    current,
                    -local_importance,
                )
            )

            edf_variables = np.sort(current[local_order[:edf_keep]])

            edf_importance = full_importance[edf_variables]

            edf_subsets.append(edf_variables.copy())
            edf_selection_path[
                iteration,
                edf_variables,
            ] = True

            # 5. ARS.
            probabilities = self._normalize_probabilities(edf_importance)

            ars_probability_path[
                iteration,
                edf_variables,
            ] = probabilities

            current = self._adaptive_reweighted_sample(
                edf_variables,
                probabilities,
                rng,
            )

            selection_path[
                iteration,
                current,
            ] = True

            # 6. CV score of this post-ARS subset.
            (
                score,
                subset_n_components,
            ) = self._score_subset(
                X,
                y,
                current,
                groups,
            )

            scores[iteration] = float(score)
            n_features_history[iteration] = int(current.size)
            n_components_history[iteration] = int(subset_n_components)

            subsets.append(current.copy())

            # Best metric; ties -> fewer variables.
            improved = best_subset is None or (
                score,
                -current.size,
            ) > (
                best_score,
                -best_subset.size,
            )

            if improved:
                best_iteration = int(iteration)
                best_score = float(score)
                best_subset = current.copy()
                best_n_components = int(subset_n_components)

            self._log(
                3,
                (
                    f"  [run {run_index + 1:03d} | "
                    f"iter {iteration + 1:03d}/{self.n_iterations}] "
                    f"active={active_before:4d} -> "
                    f"EDF={edf_variables.size:4d} -> "
                    f"ARS={current.size:4d} | "
                    f"LV={subset_n_components:2d} | "
                    f"score={score:.6f}" + (" *BEST*" if improved else "")
                ),
            )

        if best_subset is None:
            raise RuntimeError("Complete CARS run did not produce a valid subset.")

        return {
            "coefficients": np.stack(
                coefficient_history,
                axis=0,
            ),
            "importance": np.stack(
                importance_history,
                axis=0,
            ),
            "selection_path": selection_path,
            "edf_selection_path": edf_selection_path,
            "ars_probability_path": ars_probability_path,
            "scores": scores,
            "n_features": n_features_history,
            "n_components": n_components_history,
            "coefficient_components": coefficient_components,
            "subsets": tuple(subsets),
            "edf_subsets": tuple(edf_subsets),
            "sample_indices": tuple(sample_indices),
            "best_iteration": best_iteration,
            "best_score": best_score,
            "best_subset": best_subset,
            "best_n_components": best_n_components,
        }

    # ==================================================================
    # FREQUENCY PEAK SELECTION
    # ==================================================================

    def _select_frequency_peaks(
        self,
        frequency: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Detect final wavelength peak(s) from repeated-run frequency."""
        frequency = np.asarray(
            frequency,
            dtype=np.float64,
        ).reshape(-1)

        # Pad to allow first and last wavelengths to be candidate peaks.
        padded = np.concatenate(
            (
                [-np.inf],
                frequency,
                [-np.inf],
            )
        )

        kwargs: dict[str, Any] = {
            "distance": int(self.peak_distance),
            "height": float(self.peak_min_frequency),
        }

        if self.peak_prominence is not None:
            kwargs["prominence"] = float(self.peak_prominence)

        peaks, properties = find_peaks(
            padded,
            **kwargs,
        )

        peaks = peaks - 1
        peaks = peaks[(peaks >= 0) & (peaks < frequency.size)].astype(
            np.int64,
            copy=False,
        )

        self.detected_peak_indices_ = peaks
        self.peak_properties_ = properties

        # No local peak: use global frequency maximum.
        if peaks.size == 0:
            fallback = int(np.argmax(frequency))

            self._log(
                2,
                (
                    "No peak met the configured criteria. "
                    f"Using global frequency maximum at feature {fallback}."
                ),
            )

            return np.asarray(
                [fallback],
                dtype=np.int64,
            )

        ranked = self._rank_peaks(peaks)

        if self.peak_selection == "best":
            return np.asarray(
                [int(ranked[0])],
                dtype=np.int64,
            )

        if self.max_peaks is not None:
            ranked = ranked[: int(self.max_peaks)]

        return np.sort(
            ranked.astype(
                np.int64,
                copy=False,
            )
        )

    def _rank_peaks(
        self,
        peaks: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        """Rank peaks: frequency -> mean importance -> feature index."""
        peaks = np.asarray(
            peaks,
            dtype=np.int64,
        )

        order = np.lexsort(
            (
                peaks,
                -self.mean_selected_importance_[peaks],
                -self.selection_frequency_[peaks],
            )
        )

        return peaks[order]

    # ==================================================================
    # TASK / SCORING / CV
    # ==================================================================

    def _resolve_task(
        self,
        y: NDArray[Any],
    ) -> Literal[
        "regression",
        "classification",
    ]:
        if self.task not in {
            "auto",
            "regression",
            "classification",
        }:
            raise ValueError("task must be 'auto', 'regression', or 'classification'.")

        if self.task != "auto":
            return self.task

        target_type = type_of_target(y, raise_unknown=True)

        if target_type in {
            "binary",
            "multiclass",
        }:
            return "classification"

        if target_type == "continuous":
            return "regression"

        raise ValueError(f"Unsupported target type {target_type!r}.")

    def _resolve_cv(
        self,
    ) -> Any:
        if self.cv == -1:
            return LeaveOneOut()

        if isinstance(
            self.cv,
            (int, np.integer),
        ) and not isinstance(
            self.cv,
            bool,
        ):
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

    # ==================================================================
    # ESTIMATORS
    # ==================================================================

    def _make_estimator(
        self,
        n_components: int,
    ) -> PLSRegression | PLSDA:
        n_components = int(n_components)

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

    # ==================================================================
    # MONTE CARLO SAMPLE SELECTION
    # ==================================================================

    def _sample_rows(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        """Select sample_fraction of the TOTAL fitting dataset."""
        n_samples = X.shape[0]

        if self.sample_fraction >= 1.0:
            return np.arange(
                n_samples,
                dtype=np.int64,
            )

        sample_count = int(np.ceil(self.sample_fraction * n_samples))

        sample_count = max(
            2,
            min(
                sample_count,
                n_samples,
            ),
        )

        if self.task_ == "regression":
            return np.sort(
                rng.choice(
                    n_samples,
                    size=sample_count,
                    replace=False,
                ).astype(np.int64)
            )

        # Classification extension.
        child_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int32).max,
            )
        )

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=sample_count,
            random_state=child_seed,
        )

        try:
            rows, _ = next(
                splitter.split(
                    X,
                    y,
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Stratified Monte Carlo sampling failed. "
                "Check sample_fraction and class counts."
            ) from exc

        return np.sort(
            rows.astype(
                np.int64,
                copy=False,
            )
        )

    # ==================================================================
    # EDF
    # ==================================================================

    def _edf_keep_count(
        self,
        *,
        iteration: int,
        n_features: int,
    ) -> int:
        """Return EDF target count for one inner iteration."""
        final_ratio = self.min_features / n_features

        ratio = final_ratio ** (iteration / (self.n_iterations - 1))

        keep = int(np.ceil(ratio * n_features))

        return max(
            self.min_features,
            min(
                n_features,
                keep,
            ),
        )

    # ==================================================================
    # ARS
    # ==================================================================

    @staticmethod
    def _normalize_probabilities(
        importance: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weights = np.maximum(
            np.asarray(
                importance,
                dtype=np.float64,
            ),
            0.0,
        )

        if weights.size == 0:
            raise ValueError("Cannot normalize an empty importance vector.")

        total = float(np.sum(weights))

        if not np.isfinite(total) or total <= 0.0:
            return np.full(
                weights.size,
                1.0 / weights.size,
                dtype=np.float64,
            )

        return weights / total

    def _adaptive_reweighted_sample(
        self,
        variables: NDArray[np.int64],
        probabilities: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.int64]:
        variables = np.asarray(
            variables,
            dtype=np.int64,
        )
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if variables.size != probabilities.size:
            raise ValueError("variables and probabilities must have equal length.")

        if variables.size <= self.min_features:
            return np.sort(variables.copy())

        sampled = rng.choice(
            variables,
            size=variables.size,
            replace=True,
            p=probabilities,
        )

        survivors = np.unique(sampled).astype(
            np.int64,
            copy=False,
        )

        # Keep at least min_features.
        if survivors.size < self.min_features:
            order = np.lexsort(
                (
                    variables,
                    -probabilities,
                )
            )

            chosen = set(int(x) for x in survivors)

            supplemented = list(int(x) for x in survivors)

            for local_index in order:
                feature = int(variables[local_index])

                if feature not in chosen:
                    supplemented.append(feature)
                    chosen.add(feature)

                if len(supplemented) >= self.min_features:
                    break

            survivors = np.asarray(
                supplemented,
                dtype=np.int64,
            )

        return np.sort(survivors)

    # ==================================================================
    # SUBSET CV / PLS COMPONENT OPTIMIZATION
    # ==================================================================

    def _score_subset(
        self,
        X: NDArray[np.float64],
        y: NDArray[Any],
        indices: NDArray[np.int64],
        groups: NDArray[Any] | None,
    ) -> tuple[float, int]:
        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        if indices.size == 0:
            raise ValueError("Cannot score an empty wavelength subset.")

        max_components = min(
            int(self.max_components),
            int(indices.size),
            max(
                1,
                int(X.shape[0] - 1),
            ),
        )

        best_score = -np.inf
        best_n_components: int | None = None

        for n_components in range(
            1,
            max_components + 1,
        ):
            estimator = self._make_estimator(n_components)

            try:
                fold_scores = cross_val_score(
                    estimator,
                    X[:, indices],
                    y,
                    groups=groups,
                    cv=self.cv_,
                    scoring=self.scoring_,
                    n_jobs=self.n_jobs,
                    error_score="raise",
                )

            except ValueError as exc:
                self._log(
                    4,
                    (f"      skip LV={n_components} | features={indices.size} | {exc}"),
                )
                continue

            mean_score = float(np.mean(fold_scores))

            if (
                best_n_components is None
                or mean_score > best_score
                or (
                    np.isclose(
                        mean_score,
                        best_score,
                    )
                    and n_components < best_n_components
                )
            ):
                best_score = mean_score
                best_n_components = n_components

        if best_n_components is None:
            raise ValueError("No valid PLS component count could be evaluated.")

        return (
            float(best_score),
            int(best_n_components),
        )

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def _validate_parameters(
        self,
        shape: tuple[int, int],
    ) -> None:
        _, n_features = shape

        for name in (
            "max_components",
            "n_runs",
            "n_iterations",
            "min_features",
            "peak_distance",
        ):
            value = getattr(
                self,
                name,
            )

            if (
                not isinstance(
                    value,
                    (int, np.integer),
                )
                or isinstance(
                    value,
                    bool,
                )
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")

        if self.n_iterations < 2:
            raise ValueError("n_iterations must be at least 2.")

        if self.min_features > n_features:
            raise ValueError("min_features cannot exceed input feature count.")

        if not (0 < self.sample_fraction <= 1):
            raise ValueError("sample_fraction must lie in (0, 1].")

        if self.peak_selection not in {
            "best",
            "all",
        }:
            raise ValueError("peak_selection must be 'best' or 'all'.")

        if self.peak_prominence is not None and (
            not np.isscalar(self.peak_prominence) or self.peak_prominence < 0
        ):
            raise ValueError("peak_prominence must be non-negative or None.")

        if not (0.0 <= self.peak_min_frequency <= 1.0):
            raise ValueError("peak_min_frequency must be between 0 and 1.")

        if self.max_peaks is not None and (
            not isinstance(
                self.max_peaks,
                (int, np.integer),
            )
            or isinstance(
                self.max_peaks,
                bool,
            )
            or self.max_peaks < 1
        ):
            raise ValueError("max_peaks must be a positive integer or None.")

        if (
            not isinstance(
                self.n_jobs,
                (int, np.integer),
            )
            or isinstance(
                self.n_jobs,
                bool,
            )
            or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be a non-zero integer.")

        if (
            not isinstance(
                self.verbose,
                (int, np.integer),
            )
            or isinstance(
                self.verbose,
                bool,
            )
            or self.verbose < 0
        ):
            raise ValueError("verbose must be a non-negative integer.")

        if self.wavelengths is not None:
            wavelength_array = np.asarray(self.wavelengths)

            if wavelength_array.ndim != 1:
                raise ValueError("wavelengths must be one-dimensional.")

            if wavelength_array.size != n_features:
                raise ValueError("wavelengths length must match X feature count.")
