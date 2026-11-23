import numpy as np
import pytest
from sklearn.linear_model import Ridge

from spectralselector import GeneticAlgorithmSelector


class _ScriptedGeneticRNG:
    """Minimal deterministic RNG for isolating genetic operators."""

    def __init__(self, *, scalar_random, vector_random, crossover_point=2):
        self.scalar_random = scalar_random
        self.vector_random = vector_random
        self.crossover_point = crossover_point
        self.vector_calls = 0

    def random(self, size=None):
        if size is None:
            return self.scalar_random
        self.vector_calls += 1
        return np.full(size, self.vector_random, dtype=float)

    def integers(self, *args, **kwargs):
        return self.crossover_point

    def choice(self, *args, **kwargs):  # pragma: no cover - repair is not expected
        raise AssertionError("The operator fixture should not require repair.")


def test_genetic_equal_column_groups_ignore_irregular_wavelength_spacing():
    selector = GeneticAlgorithmSelector(
        n_intervals=3,
        wavelengths=np.asarray([400, 401, 405, 406, 450, 451, 600, 602, 900, 1100]),
    )
    selector.n_features_in_ = 10

    groups = selector._make_groups()
    decoded = selector._decode(np.asarray([1, 0, 1]), groups)

    assert [group.tolist() for group in groups] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    # Wavelengths are metadata for the generic GA; group boundaries follow
    # column positions even when the physical axis is irregularly spaced.
    np.testing.assert_array_equal(decoded, [0, 1, 2, 3, 7, 8, 9])


def test_genetic_repair_is_seeded_and_enforces_both_group_bounds():
    selector = GeneticAlgorithmSelector(min_groups=2, max_groups=3)
    lower_a = np.zeros(6, dtype=np.int8)
    lower_b = np.zeros(6, dtype=np.int8)
    upper_a = np.ones(6, dtype=np.int8)
    upper_b = np.ones(6, dtype=np.int8)

    selector._repair(lower_a, np.random.default_rng(12))
    selector._repair(lower_b, np.random.default_rng(12))
    selector._repair(upper_a, np.random.default_rng(13))
    selector._repair(upper_b, np.random.default_rng(13))

    np.testing.assert_array_equal(lower_a, lower_b)
    np.testing.assert_array_equal(upper_a, upper_b)
    assert lower_a.sum() == 2
    assert upper_a.sum() == 3


def test_genetic_single_point_crossover_isolated_from_mutation():
    selector = GeneticAlgorithmSelector(
        crossover_probability=1.0,
        mutation_probability=0.0,
        min_groups=1,
        max_groups=4,
    )
    rng = _ScriptedGeneticRNG(scalar_random=0.0, vector_random=1.0)

    first, second = selector._breed_pair(
        np.asarray([1, 0, 0, 1], dtype=np.int8),
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        rng,
    )

    np.testing.assert_array_equal(first, [1, 0, 1, 0])
    np.testing.assert_array_equal(second, [0, 1, 0, 1])


def test_genetic_per_gene_mutation_isolated_from_crossover():
    selector = GeneticAlgorithmSelector(
        crossover_probability=0.0,
        mutation_probability=1.0,
        min_groups=1,
        max_groups=4,
    )
    rng = _ScriptedGeneticRNG(scalar_random=1.0, vector_random=0.0)

    first, second = selector._breed_pair(
        np.asarray([1, 0, 0, 1], dtype=np.int8),
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        rng,
    )

    np.testing.assert_array_equal(first, [0, 1, 1, 0])
    np.testing.assert_array_equal(second, [1, 0, 0, 1])


def test_genetic_breeding_only_mutates_the_requested_child_count():
    selector = GeneticAlgorithmSelector(
        crossover_probability=0.0,
        mutation_probability=0.0,
        min_groups=1,
        max_groups=4,
    )
    rng = _ScriptedGeneticRNG(scalar_random=1.0, vector_random=1.0)

    children = selector._breed_pair(
        np.asarray([1, 0, 0, 1], dtype=np.int8),
        np.asarray([0, 1, 1, 0], dtype=np.int8),
        rng,
        n_children=1,
    )

    assert len(children) == 1
    assert rng.vector_calls == 1
    np.testing.assert_array_equal(children[0], [1, 0, 0, 1])


def test_genetic_tournament_uses_sparsity_as_deterministic_tie_break():
    selector = GeneticAlgorithmSelector(tournament_size=3)
    population = np.asarray([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)
    scores = np.asarray([0.8, 0.8, 0.7])

    winner = selector._tournament(population, scores, np.random.default_rng(5))

    np.testing.assert_array_equal(winner, population[1])


def test_genetic_population_cache_evaluates_each_distinct_mask_once(monkeypatch):
    selector = GeneticAlgorithmSelector(n_jobs=1)
    selector.n_jobs_ = 1
    population = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.int8)
    groups = [np.asarray([0]), np.asarray([1])]
    calls = []

    def evaluate(chromosome, *args):
        calls.append(chromosome.copy())
        return float(chromosome[0] + 2 * chromosome[1])

    monkeypatch.setattr(selector, "_evaluate_chromosome", evaluate)
    cache = {}
    first = selector._evaluate_population(
        population,
        groups,
        np.ones((4, 2)),
        np.arange(4.0),
        None,
        cache,
    )
    second = selector._evaluate_population(
        population,
        groups,
        np.ones((4, 2)),
        np.arange(4.0),
        None,
        cache,
    )

    np.testing.assert_array_equal(first, [1.0, 1.0, 2.0])
    np.testing.assert_array_equal(second, first)
    assert len(calls) == 2
    assert len(cache) == 2


def test_genetic_complete_run_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(91)
    X = rng.normal(size=(30, 8))
    y = 2.0 * X[:, 1] - X[:, 6] + rng.normal(scale=0.1, size=30)
    configuration = {
        "estimator": Ridge(alpha=0.1),
        "cv": 2,
        "population_size": 6,
        "generations": 3,
        "tournament_size": 2,
        "min_groups": 1,
        "max_groups": 4,
        "random_state": 17,
    }

    first = GeneticAlgorithmSelector(**configuration).fit(X, y)
    second = GeneticAlgorithmSelector(**configuration).fit(X, y)

    np.testing.assert_array_equal(first.best_chromosome_, second.best_chromosome_)
    np.testing.assert_array_equal(first.selected_indices_, second.selected_indices_)
    np.testing.assert_allclose(first.best_score_history_, second.best_score_history_)
    assert first.best_score_ == pytest.approx(second.best_score_)
    assert first.n_evaluated_subsets_ == second.n_evaluated_subsets_


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"generations": 0}, "generations must be a positive integer"),
        ({"population_size": 1, "tournament_size": 1}, "at least 2"),
        ({"tournament_size": 5}, "cannot exceed population_size"),
        ({"elitism": 0}, "elitism must be"),
        ({"max_groups": 9}, "max_groups must be"),
        ({"crossover_probability": 1.5}, "crossover_probability must be in"),
        ({"n_intervals": 0}, "n_intervals must be None"),
    ],
)
def test_genetic_rejects_invalid_parameters(params, match):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 8))
    selector = GeneticAlgorithmSelector(
        Ridge(),
        **{"population_size": 4, "generations": 1, "tournament_size": 2, **params},
    )
    with pytest.raises(ValueError, match=match):
        selector.fit(X, X[:, 0])
