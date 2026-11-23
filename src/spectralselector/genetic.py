"""Generic genetic optimization of wavelength or interval masks.

Reference status
----------------
This module does not claim reference equivalence with one named GA wavelength
selection paper. Song et al. (2016), DOI ``10.1016/j.aca.2016.10.041``, defines
Interval Combination Optimization (ICO) and gives GA-iPLS only as a comparison
whose detailed procedure is delegated to earlier sources: Goicoechea and
Olivieri (2003), Xiaobo et al. (2007), and especially Yun et al. (2014).

Algorithm implemented here
--------------------------
Binary chromosomes represent individual wavelengths or equal-column contiguous
groups. Mean internal-CV score is fitness. Tournament-selected parents undergo
single-point crossover and independent mutation; elitism preserves the best
chromosomes, constraint repair enforces active-group bounds, and repeated masks
are evaluated once through an in-memory cache.

One call to ``fit`` is one complete stochastic GA run. ``generations`` counts
population updates, not independent runs and not model fits. The actual number
of distinct cross-validated model fits is exposed as ``n_evaluated_subsets_``
and can be lower than ``population_size * generations`` because masks are
cached. Repeated independent runs must be orchestrated outside this estimator
with explicitly recorded seeds.

Computational cost
------------------
At most ``population_size * generations`` chromosomes are proposed in one run.
Each distinct repaired mask requires a complete CV evaluation of the configured
estimator; the fitted ``n_evaluated_subsets_`` value records the actual count
after duplicate-mask caching.

Scientific scope
----------------
``GeneticAlgorithmSelector`` is a generic scikit-learn wrapper. It is not ICO
and is not a validated GA-iPLS reproduction. A future named implementation must
define its canonical chromosome, initialization, genetic operators, PLS tuning,
stopping rule, repeated-run aggregation, and reference fixture separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed, parallel_config
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.utils.validation import check_X_y

from ._base import BaseWavelengthSelector
from ._wrapper import WrapperSelectorMixin


class GeneticAlgorithmSelector(WrapperSelectorMixin, BaseWavelengthSelector):
    """Optimize a variable mask using model cross-validation fitness.

    This is a generic genetic mask optimizer. It is not a validated GA-iPLS
    implementation unless its interval representation and search settings are
    independently matched to a specific GA-iPLS method.

    Set ``n_intervals=None`` to optimize individual variables. Set an integer to
    optimize contiguous groups created with ``numpy.array_split``.

    Parameters
    ----------
    estimator : estimator object, optional
        Cloneable predictive estimator. Unlike CARS/BOSS, it does not need to
        expose coefficients. Defaults to one-component PLSR.
    task : {"auto", "regression", "classification"}, default="auto"
        Prediction task used for CV and scoring validation.
    scoring : str or callable, optional
        Scikit-learn scorer used as chromosome fitness.
    cv : int, CV splitter, or -1, default=5
        Internal fitness CV. ``-1`` means leave-one-out.
    population_size : int, default=30
        Number of chromosomes in each generation.
    generations : int, default=30
        Number of population evaluations and evolutionary updates in one
        complete run. This is not a count of independent complete GA runs or
        distinct model fits.
    crossover_probability : float, default=0.8
        Probability of single-point crossover for each parent pair.
    mutation_probability : float, default=0.05
        Independent probability of flipping each chromosome bit.
    tournament_size : int, default=3
        Number of candidates used during parent selection.
    elitism : int, default=1
        Best chromosomes copied unchanged to the next generation.
    n_intervals : int, optional
        Number of contiguous column groups. ``None`` creates one gene per input
        variable.
    min_groups, max_groups : int, optional
        Bounds on active genes after chromosome repair.
    random_state : int, optional
        Seed controlling initialization, selection, crossover, and mutation.
    n_jobs : int, default=1
        Worker processes used to evaluate distinct chromosomes. ``-1`` uses all
        available CPU cores. Internal CV remains serial per worker.
    wavelengths : array-like of shape (n_features,), optional
        Physical wavelength metadata.

    Attributes
    ----------
    best_chromosome_ : ndarray of shape (n_groups,)
        Binary mask for the best group combination.
    best_score_history_ : ndarray of shape (generations,)
        Best fitness encountered in each generation.
    groups_ : tuple of ndarrays
        Mapping from genes to original feature indices.
    n_evaluated_subsets_ : int
        Number of distinct chromosomes evaluated by cross-validation after
        caching duplicates. This is the effective fitness-evaluation count.
    best_score_ : float
        Best internal CV fitness.
    n_jobs_ : int
        Resolved positive number of chromosome worker processes.

    Notes
    -----
    This class is intentionally a generic wrapper search. Song et al. (2016)
    introduce interval combination optimization (ICO) and describe GA-iPLS only
    as a comparison method whose detailed genetic procedure comes from earlier
    references. Therefore neither the ``GeneticAlgorithmSelector`` name nor its
    ``GASelector`` alias implies ICO or reference-equivalent GA-iPLS behavior.

    References
    ----------
    Song, X. et al. (2016). A novel algorithm for spectral interval combination
    optimization. *Analytica Chimica Acta*, 948, 19-29.
    https://doi.org/10.1016/j.aca.2016.10.041

    Goicoechea, H. C., & Olivieri, A. C. (2003). A new family of genetic
    algorithms for wavelength interval selection in multivariate analytical
    spectroscopy. *Journal of Chemometrics*, 17, 338-345.
    https://doi.org/10.1002/cem.812

    Yun, Y.-H. et al. (2014). A simple idea on applying large regression
    coefficient to improve the genetic algorithm-PLS for variable selection in
    multivariate calibration. *Chemometrics and Intelligent Laboratory
    Systems*, 130, 76-83.
    https://doi.org/10.1016/j.chemolab.2013.09.007
    """

    def __init__(
        self,
        estimator: Any = None,
        *,
        task: str = "auto",
        scoring: str | Any | None = None,
        cv: Any = 5,
        population_size: int = 30,
        generations: int = 30,
        crossover_probability: float = 0.8,
        mutation_probability: float = 0.05,
        tournament_size: int = 3,
        elitism: int = 1,
        n_intervals: int | None = None,
        min_groups: int = 2,
        max_groups: int | None = None,
        random_state: int | None = None,
        n_jobs: int = 1,
        wavelengths: ArrayLike | None = None,
    ):
        """Store genetic search configuration without evaluating any masks."""
        self.estimator = estimator
        self.task = task
        self.scoring = scoring
        self.cv = cv
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.n_intervals = n_intervals
        self.min_groups = min_groups
        self.max_groups = max_groups
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.wavelengths = wavelengths

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        groups: ArrayLike | None = None,
    ) -> GeneticAlgorithmSelector:
        """Evolve feature masks and retain the highest-fitness chromosome.

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
        # Step 1: one gene per wavelength, or per contiguous interval.
        feature_groups = self._make_groups()
        self._validate_parameters(len(feature_groups))
        _, _, _, sample_groups = self._prepare_wrapper(y_array, groups)
        rng = np.random.default_rng(self.random_state)
        # Step 2: random binary population; repair group-count constraint
        # violations before the first expensive CV evaluation.
        population = rng.integers(
            0, 2, size=(int(self.population_size), len(feature_groups)), dtype=np.int8
        )
        for chromosome in population:
            self._repair(chromosome, rng)

        # Chromosome caching avoids fitting the same model more than once.
        cache: dict[bytes, float] = {}
        best_chromosome: np.ndarray | None = None
        best_score = -np.inf
        history = []

        for _ in range(int(self.generations)):
            # Step 3: fitness is the mean internal CV score of the selected
            # variables (each distinct mask is evaluated once, then cached).
            scores = self._evaluate_population(
                population,
                feature_groups,
                X_array,
                y_array,
                sample_groups,
                cache,
            )
            order = sorted(
                range(len(population)),
                key=lambda index: (scores[index], -population[index].sum()),
                reverse=True,
            )
            # Step 4: rank by fitness (ties: fewer active genes) and keep the best
            # chromosome seen in any generation.
            generation_best = order[0]
            history.append(scores[generation_best])
            if (scores[generation_best], -population[generation_best].sum()) > (
                best_score,
                -(
                    len(feature_groups) + 1
                    if best_chromosome is None
                    else best_chromosome.sum()
                ),
            ):
                best_score = float(scores[generation_best])
                best_chromosome = population[generation_best].copy()

            # Step 5: elitism copies the strongest masks unchanged.
            # Step 6: tournament-selected parents fill the rest of the next
            # population through single-point crossover, mutation and repair.
            next_population = [
                population[index].copy() for index in order[: self.elitism]
            ]
            while len(next_population) < self.population_size:
                first = self._tournament(population, scores, rng)
                second = self._tournament(population, scores, rng)
                children = self._breed_pair(
                    first,
                    second,
                    rng,
                    n_children=min(2, self.population_size - len(next_population)),
                )
                for child in children:
                    next_population.append(child)
            population = np.asarray(next_population, dtype=np.int8)

        assert best_chromosome is not None
        if not np.isfinite(best_score):
            raise ValueError(
                "Every genetic-algorithm candidate failed. Check estimator "
                "requirements, scoring, CV, and group-count constraints."
            )
        # Step 7: after the last generation, the best chromosome is the selection
        # and the estimator is refitted on its wavelengths.
        selected = self._decode(best_chromosome, feature_groups)
        self.best_score_ = best_score
        self.best_chromosome_ = best_chromosome
        self.best_score_history_ = np.asarray(history, dtype=np.float64)
        self.groups_ = tuple(group.copy() for group in feature_groups)
        self.n_evaluated_subsets_ = len(cache)
        self.estimator_ = clone(self.estimator_template_).fit(
            X_array[:, selected], y_array
        )
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[selected] = True
        self._finalize_selection(
            mask, wavelengths=self.wavelengths, selection_order=selected
        )
        return self

    def _make_groups(self) -> list[np.ndarray]:
        """Map chromosome genes to variables or contiguous column intervals."""
        if self.n_intervals is None:
            return [
                np.asarray([index], dtype=np.int64)
                for index in range(self.n_features_in_)
            ]
        if (
            not isinstance(self.n_intervals, (int, np.integer))
            or isinstance(self.n_intervals, bool)
            or not 1 <= self.n_intervals <= self.n_features_in_
        ):
            raise ValueError(
                "n_intervals must be None or an integer between 1 and n_features."
            )
        return [
            group.astype(np.int64)
            for group in np.array_split(
                np.arange(self.n_features_in_), int(self.n_intervals)
            )
        ]

    def _validate_parameters(self, n_groups: int) -> None:
        """Validate population, probability, and active-group constraints."""
        for name in ("population_size", "generations", "tournament_size", "min_groups"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size.")
        if (
            not isinstance(self.elitism, (int, np.integer))
            or not 1 <= self.elitism < self.population_size
        ):
            raise ValueError("elitism must be between 1 and population_size - 1.")
        maximum = n_groups if self.max_groups is None else self.max_groups
        if (
            not isinstance(maximum, (int, np.integer))
            or not self.min_groups <= maximum <= n_groups
        ):
            raise ValueError(
                "max_groups must be between min_groups and the number of groups; "
                f"{n_groups} feature(s)/groups are available."
            )
        if self.min_groups > n_groups:
            raise ValueError("min_groups cannot exceed the number of groups.")
        for name in ("crossover_probability", "mutation_probability"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1].")

    def _repair(self, chromosome: np.ndarray, rng: np.random.Generator) -> None:
        """Mutate a chromosome in place until group-count constraints hold."""
        maximum = len(chromosome) if self.max_groups is None else int(self.max_groups)
        active = np.flatnonzero(chromosome)
        if active.size < self.min_groups:
            available = np.flatnonzero(chromosome == 0)
            chosen = rng.choice(
                available, size=self.min_groups - active.size, replace=False
            )
            chromosome[chosen] = 1
        active = np.flatnonzero(chromosome)
        if active.size > maximum:
            chosen = rng.choice(active, size=active.size - maximum, replace=False)
            chromosome[chosen] = 0

    def _breed_pair(
        self,
        first: np.ndarray,
        second: np.ndarray,
        rng: np.random.Generator,
        *,
        n_children: int = 2,
    ) -> tuple[np.ndarray, ...]:
        """Return repaired children after crossover and per-gene mutation."""
        if n_children not in (1, 2):
            raise ValueError("n_children must be one or two.")
        child_a, child_b = first.copy(), second.copy()
        # Single-point crossover: swap the tails after a random cut point.
        if len(child_a) > 1 and rng.random() < self.crossover_probability:
            point = int(rng.integers(1, len(child_a)))
            tail = child_a[point:].copy()
            child_a[point:] = child_b[point:]
            child_b[point:] = tail
        children = (child_a, child_b)[:n_children]
        for child in children:
            # Mutation flips each gene independently; repair restores the bounds.
            mutations = rng.random(len(child)) < self.mutation_probability
            child[mutations] = 1 - child[mutations]
            self._repair(child, rng)
        return children

    @staticmethod
    def _decode(chromosome: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
        """Expand active genes into sorted original feature indices."""
        return np.concatenate(
            [group for gene, group in zip(chromosome, groups, strict=True) if gene]
        ).astype(np.int64)

    def _evaluate_population(
        self,
        population: np.ndarray,
        groups: list[np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        sample_groups: np.ndarray | None,
        cache: dict[bytes, float],
    ) -> np.ndarray:
        """Evaluate uncached chromosomes in parallel and return ordered scores."""
        missing: dict[bytes, np.ndarray] = {}
        for chromosome in population:
            key = chromosome.tobytes()
            if key not in cache and key not in missing:
                missing[key] = chromosome.copy()

        if missing:
            items = list(missing.items())
            with parallel_config(
                backend="loky", n_jobs=self.n_jobs_, inner_max_num_threads=1
            ):
                evaluated = Parallel()(
                    delayed(self._evaluate_chromosome)(
                        chromosome, groups, X, y, sample_groups
                    )
                    for _, chromosome in items
                )
            for (key, _), score in zip(items, evaluated, strict=True):
                cache[key] = score

        return np.asarray(
            [cache[chromosome.tobytes()] for chromosome in population],
            dtype=np.float64,
        )

    def _evaluate_chromosome(
        self,
        chromosome: np.ndarray,
        groups: list[np.ndarray],
        X: np.ndarray,
        y: np.ndarray,
        sample_groups: np.ndarray | None,
    ) -> float:
        """Return one CV fitness, or negative infinity for an invalid mask."""
        try:
            return self._score_indices(
                X,
                y,
                self._decode(chromosome, groups),
                sample_groups,
            )
        except (ValueError, TypeError):
            return -np.inf

    def _tournament(
        self,
        population: np.ndarray,
        scores: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Choose one parent by fitness tournament with a sparsity tie-break."""
        candidates = rng.choice(
            len(population), size=int(self.tournament_size), replace=False
        )
        winner = max(
            candidates, key=lambda index: (scores[index], -population[index].sum())
        )
        return population[winner]


# Short public alias for interactive use and configuration files.
GASelector = GeneticAlgorithmSelector
