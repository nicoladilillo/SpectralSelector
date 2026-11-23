"""Interval Combination Optimization (ICO) wavelength selection.

Reference
---------
Song, X., Huang, Y., Yan, H., Xiong, Y., & Min, S. (2016).
A novel algorithm for spectral interval combination optimization.
Analytica Chimica Acta, 948, 19-29.
https://doi.org/10.1016/j.aca.2016.10.041

Paper algorithm implemented here
--------------------------------
ICO works under the model population analysis (MPA) framework.

STEP 1 - Split the spectrum into ``n_intervals`` (``N``) equal-width intervals.
STEP 2 - Weighted bootstrap sampling (WBS) generates ``n_combinations`` (``M``)
    random interval combinations. Each interval has a weight in [0, 1]; in
    every WBS draw an interval is chosen with probability
    ``p_i = w_i / sum(w)`` (roulette wheel, Eq. 1). The wheel is spun ``R``
    times with replacement and the intervals chosen at least once form the
    combination. ``R`` equals ``N`` in the first iteration and the average
    number of distinct intervals of the previous population afterwards, so the
    combinations shrink by about a factor of 0.632 per iteration.
    The initial weight of every interval is 1.
STEP 3 - Every combination is scored by cross-validation (the paper uses PLS
    with five-fold CV and reports RMSECV).
STEP 4 - The best ``best_fraction`` (``a``) of the combinations is extracted and
    the mean of their scores is recorded.
STEP 5 - New weights are the appearance frequencies of the intervals in those
    best combinations divided by their number (Eq. 2).
STEP 6 - The loop continues while the mean score of the best combinations
    improves; otherwise it stops.
STEP 7 - The best combination of the final iteration is the selection.
STEP 8 - Local search: the variables at the boundaries of every selected
    contiguous block are removed one by one while the CV score without them
    is better than with them.

Implementation notes
--------------------
* Scores follow scikit-learn's higher-is-better convention, so "RMSECV
  decreases" corresponds to the mean score of the best combinations
  increasing. With the default ``neg_root_mean_squared_error`` scorer this is
  exactly the RMSECV criterion of the paper.
* The paper does not state a numerical tolerance or an iteration cap; the
  strict improvement rule is used together with ``max_iterations`` as a
  safeguard against pathological scoring functions.
* The paper optimizes the number of PLS latent variables (maximum 10) by
  cross-validation for every combination. Here the model is the ``estimator``
  argument. To reproduce the paper, pass a model that tunes its components
  internally, for example ``GridSearchCV(PLSRegression(scale=False),
  {"n_components": range(1, 11)})``. The default is a one-component PLSR,
  which is only suitable for quick tests.
* If ``wavelengths`` is supplied and strictly monotonic, intervals have equal
  physical width on that axis (bins that contain no column are dropped).
  Otherwise intervals contain an equal number of columns.
* Classification targets are accepted as a documented extension: any
  scikit-learn classifier and scorer can be used.
* Step 8 is the removal-only boundary search described in the paper text.
  The paper's wording does not define behavior for ties; a variable is removed
  only when the score without it is strictly better.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._wrapper import WrapperSelectorMixin


class ICOSelector(WrapperSelectorMixin, BaseWavelengthSelector):
    """Select a combination of spectral intervals by ICO (Song et al., 2016).

    Parameters
    ----------
    estimator : estimator object, optional
        Cloneable predictive estimator evaluated by cross-validation on every
        interval combination. Defaults to one-component PLSR; pass a model
        that tunes its latent variables to follow the paper.
    task : {"auto", "regression", "classification"}, default="auto"
        Prediction task used for CV and scoring validation.
    scoring : str or callable, optional
        Scikit-learn scorer (higher is better). Defaults to negative RMSE for
        regression and accuracy for classification.
    cv : int, CV splitter, or -1, default=5
        Internal cross-validation. ``-1`` means leave-one-out.
    n_intervals : int, default=40
        Number of equal-width intervals ``N``. It is capped at the number of
        input features.
    n_combinations : int, default=1000
        Number of random interval combinations ``M`` generated per iteration.
    best_fraction : float, default=0.1
        Fraction ``a`` of best combinations used to update interval weights.
    local_search : bool, default=True
        Run the boundary-variable optimization (step 8).
    max_iterations : int, default=50
        Safeguard on the number of weight-update iterations.
    random_state : int, optional
        Seed controlling the weighted bootstrap sampling.
    n_jobs : int, default=1
        Worker processes used to evaluate combinations. ``-1`` uses all
        available CPU cores. Internal CV remains serial per worker.
    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength metadata. When strictly monotonic it defines
        equal-width physical intervals.

    Attributes
    ----------
    intervals_ : tuple of ndarrays
        Column indices of every interval.
    interval_weights_ : ndarray of shape (n_intervals,)
        Weights after the last update.
    selected_intervals_ : ndarray of int
        Intervals of the best combination of the final iteration.
    best_score_ : float
        Internal CV score of the final selection after local search.
    mean_score_history_ : ndarray
        Mean score of the best combinations in every iteration.
    n_iterations_ : int
        Number of iterations executed.
    n_evaluated_subsets_ : int
        Number of distinct subsets evaluated by cross-validation.
    estimator_ : estimator
        Estimator refitted on the selected columns.
    n_jobs_ : int
        Resolved number of worker processes.

    References
    ----------
    Song, X. et al. (2016). A novel algorithm for spectral interval combination
    optimization. *Analytica Chimica Acta*, 948, 19-29.
    https://doi.org/10.1016/j.aca.2016.10.041
    """

    def __init__(
        self,
        estimator: Any = None,
        *,
        task: str = "auto",
        scoring: str | Any | None = None,
        cv: Any = 5,
        n_intervals: int = 40,
        n_combinations: int = 1000,
        best_fraction: float = 0.1,
        local_search: bool = True,
        max_iterations: int = 50,
        random_state: int | None = None,
        n_jobs: int = 1,
        wavelengths: ArrayLike | None = None,
    ):
        """Store ICO configuration without evaluating any combination."""
        self.estimator = estimator
        self.task = task
        self.scoring = scoring
        self.cv = cv
        self.n_intervals = n_intervals
        self.n_combinations = n_combinations
        self.best_fraction = best_fraction
        self.local_search = local_search
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.wavelengths = wavelengths

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: ArrayLike | None = None,
    ) -> ICOSelector:
        """Run ICO and keep the best combination after local search.

        ``groups`` contains optional sample-group labels for a group-aware
        internal CV splitter supplied through ``cv``.
        """
        X_array, y_array = check_X_y(
            X,
            y,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
            ensure_min_samples=2,
        )
        self._validate_X(X_array, reset=True)
        axis = self._validate_wavelengths(self.wavelengths)
        self._validate_parameters()
        # STEP 1: split the spectrum into N equal-width intervals.
        intervals = self._make_intervals(axis)
        n_int = len(intervals)
        _, _, _, sample_groups = self._prepare_wrapper(y_array, groups)
        rng = np.random.default_rng(self.random_state)

        cache: dict[bytes, float] = {}
        # Every interval starts with weight 1; the first WBS spins the wheel N times.
        weights = np.ones(n_int, dtype=np.float64)
        n_draws = n_int
        n_best = max(1, int(np.ceil(float(self.best_fraction) * self.n_combinations)))
        history: list[float] = []
        combo_best: np.ndarray | None = None
        combo_best_score = -np.inf

        for _ in range(int(self.max_iterations)):
            # STEP 2: weighted bootstrap sampling of M interval combinations.
            combos = self._weighted_bootstrap(weights, n_draws, rng)
            # STEP 3: score every combination by cross-validation.
            scores = self._evaluate_combinations(
                combos, intervals, X_array, y_array, sample_groups, cache
            )
            order = sorted(
                range(len(combos)),
                key=lambda i: (scores[i], -combos[i].sum()),
                reverse=True,
            )
            # STEP 4: keep the best fraction a of the combinations, ties to fewer
            # intervals, and record the mean of their scores.
            best_ids = order[:n_best]
            best_scores = scores[best_ids]
            if not np.isfinite(best_scores).any():
                raise ValueError(
                    "Every ICO interval combination failed. Check estimator "
                    "requirements, scoring, and CV."
                )
            mean_score = float(np.mean(best_scores))
            previous = history[-1] if history else -np.inf
            history.append(mean_score)
            combo_best = combos[order[0]].copy()
            combo_best_score = float(scores[order[0]])
            if mean_score <= previous:
                # Step 6: the mean best score no longer improves; the best
                # combination of this final iteration is kept (step 7).
                break
            # STEP 5: new weight = frequency of each interval among the best
            # combinations (Eq. 2); the next WBS spins the wheel as many times as
            # the mean number of intervals per combination (about 0.632 shrink).
            weights = combos[best_ids].sum(axis=0) / float(len(best_ids))
            n_draws = max(1, round(float(np.mean(combos.sum(axis=1)))))

        assert combo_best is not None
        # STEP 7: the best combination of the final iteration is the selection.
        selected = self._decode(combo_best, intervals)
        # STEP 8: optionally trim boundary variables of each contiguous block.
        if self.local_search:
            selected, combo_best_score = self._local_search(
                selected,
                combo_best_score,
                X_array,
                y_array,
                sample_groups,
                cache,
            )

        self.intervals_ = tuple(group.copy() for group in intervals)
        self.interval_weights_ = weights
        self.selected_intervals_ = np.flatnonzero(combo_best)
        self.best_score_ = float(combo_best_score)
        self.mean_score_history_ = np.asarray(history, dtype=np.float64)
        self.n_iterations_ = len(history)
        self.n_evaluated_subsets_ = len(cache)
        self.estimator_ = clone(self.estimator_template_).fit(
            X_array[:, selected], y_array
        )
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True
        self._finalize_selection(mask, wavelengths=self.wavelengths)
        return self

    def _validate_parameters(self) -> None:
        """Validate ICO configuration."""
        for name in ("n_intervals", "n_combinations", "max_iterations"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < float(self.best_fraction) <= 1:
            raise ValueError("best_fraction must be in (0, 1].")

    def _make_intervals(self, axis: NDArray[np.float64] | None) -> list[np.ndarray]:
        """Split columns into equal-width (physical) or equal-count intervals."""
        n_features = self.n_features_in_
        n_int = min(int(self.n_intervals), n_features)
        if axis is not None:
            diffs = np.diff(axis)
            if axis.size > 1 and (np.all(diffs > 0) or np.all(diffs < 0)):
                edges = np.linspace(axis.min(), axis.max(), n_int + 1)
                bins = np.clip(np.digitize(axis, edges[1:-1]), 0, n_int - 1)
                groups = [np.flatnonzero(bins == k) for k in range(n_int)]
                groups = [g.astype(np.int64) for g in groups if g.size]
                if groups:
                    return groups
        return [
            g.astype(np.int64) for g in np.array_split(np.arange(n_features), n_int)
        ]

    def _weighted_bootstrap(
        self,
        weights: NDArray[np.float64],
        n_draws: int,
        rng: np.random.Generator,
    ) -> NDArray[np.int8]:
        """Return ``n_combinations`` WBS combinations as a binary matrix.

        Each row spins the roulette ``n_draws`` times with replacement using
        ``p_i = w_i / sum(w)`` (Eq. 1) and marks the intervals drawn at least
        once. A weight of 1 therefore does not guarantee inclusion.
        """
        probabilities = weights / weights.sum()
        draws = rng.choice(
            len(weights),
            size=(int(self.n_combinations), int(n_draws)),
            replace=True,
            p=probabilities,
        )
        combos = np.zeros((int(self.n_combinations), len(weights)), dtype=np.int8)
        rows = np.repeat(np.arange(draws.shape[0]), draws.shape[1])
        combos[rows, draws.ravel()] = 1
        return combos

    def _evaluate_combinations(
        self,
        combos: NDArray[np.int8],
        intervals: list[np.ndarray],
        X: NDArray[np.float64],
        y: NDArray[Any],
        sample_groups: NDArray[Any] | None,
        cache: dict[bytes, float],
    ) -> NDArray[np.float64]:
        """Evaluate uncached combinations in parallel and return ordered scores."""
        missing: dict[bytes, np.ndarray] = {}
        for combo in combos:
            key = combo.tobytes()
            if key not in cache and key not in missing:
                missing[key] = combo.copy()
        if missing:
            items = list(missing.items())
            with parallel_config(
                backend="loky", n_jobs=self.n_jobs_, inner_max_num_threads=1
            ):
                evaluated = Parallel()(
                    delayed(self._evaluate_indices)(
                        self._decode(combo, intervals), X, y, sample_groups
                    )
                    for _, combo in items
                )
            for (key, _), score in zip(items, evaluated, strict=True):
                cache[key] = score
        return np.asarray([cache[c.tobytes()] for c in combos], dtype=np.float64)

    def _evaluate_indices(
        self,
        indices: NDArray[np.int64],
        X: NDArray[np.float64],
        y: NDArray[Any],
        sample_groups: NDArray[Any] | None,
    ) -> float:
        """Return one CV score, or negative infinity for an invalid subset."""
        try:
            return self._score_indices(X, y, indices, sample_groups)
        except (ValueError, TypeError):
            return -np.inf

    @staticmethod
    def _decode(combo: np.ndarray, intervals: list[np.ndarray]) -> NDArray[np.int64]:
        """Expand active intervals into sorted original column indices."""
        return np.sort(
            np.concatenate(
                [g for flag, g in zip(combo, intervals, strict=True) if flag]
            )
        ).astype(np.int64)

    def _local_search(
        self,
        selected: NDArray[np.int64],
        score: float,
        X: NDArray[np.float64],
        y: NDArray[Any],
        sample_groups: NDArray[Any] | None,
        cache: dict[bytes, float],
    ) -> tuple[NDArray[np.int64], float]:
        """Trim boundary variables of each selected block (step 8)."""
        current = list(selected)
        best = score

        def score_of(indices: list[int]) -> float:
            key = b"idx" + np.asarray(indices, dtype=np.int64).tobytes()
            if key not in cache:
                cache[key] = self._evaluate_indices(
                    np.asarray(indices, dtype=np.int64), X, y, sample_groups
                )
            return cache[key]

        # Trim from the left edges of all blocks, then from the right edges.
        for side in ("left", "right"):
            changed = True
            while changed and len(current) > 1:
                changed = False
                arr = np.asarray(current)
                breaks = np.flatnonzero(np.diff(arr) != 1) + 1
                blocks = np.split(arr, breaks)
                # Visit blocks from the far end so removals keep positions valid.
                for block in blocks[::-1] if side == "right" else blocks:
                    while block.size > 1 and len(current) > 1:
                        edge = int(block[0] if side == "left" else block[-1])
                        trial = [i for i in current if i != edge]
                        trial_score = score_of(trial)
                        # Remove the edge only if the CV score strictly improves.
                        if trial_score > best:
                            current = trial
                            best = trial_score
                            block = block[1:] if side == "left" else block[:-1]
                            changed = True
                        else:
                            break
        return np.asarray(sorted(current), dtype=np.int64), float(best)
