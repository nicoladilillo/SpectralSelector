import pytest
from sklearn.utils.estimator_checks import check_estimator

from spectralselector import (
    PLSDA,
    BOSSSelector,
    CARSSelector,
    CovSelSelector,
    FCovSelSelector,
    GeneticAlgorithmSelector,
    ICOSelector,
    IndexSelector,
    MIVIFSelector,
    SelectivityRatioSelector,
    SPASelector,
    VIPSelector,
    WavelengthSelectKBest,
)


@pytest.mark.parametrize(
    "estimator",
    [
        IndexSelector([0]),
        WavelengthSelectKBest(k=2),
        CovSelSelector(n_features_to_select=1, task="regression"),
        FCovSelSelector(1, max_features=1, cv=2, random_state=0),
        VIPSelector(n_components=1, n_features_to_select=1),
        SelectivityRatioSelector(n_components=1, n_features_to_select=1),
        MIVIFSelector(task="regression", random_state=0),
        SPASelector(1, initial=0, max_features=1, cv=2, random_state=0),
        PLSDA(),
        CARSSelector(
            cv=2,
            max_components=1,
            n_runs=1,
            n_iterations=2,
            min_features=1,
            random_state=0,
        ),
        BOSSSelector(
            cv=2,
            max_components=1,
            n_submodels=4,
            top_fraction=0.5,
            n_runs=1,
            min_features=1,
            max_iterations=2,
            random_state=0,
        ),
        GeneticAlgorithmSelector(
            cv=2,
            population_size=4,
            generations=1,
            tournament_size=2,
            elitism=1,
            min_groups=1,
            max_groups=2,
            random_state=0,
        ),
        ICOSelector(
            cv=2,
            n_intervals=2,
            n_combinations=6,
            max_iterations=2,
            random_state=0,
        ),
    ],
    ids=[
        "fixed-index",
        "univariate",
        "covsel",
        "fcovsel",
        "vip",
        "selectivity-ratio",
        "mi-vif",
        "spa",
        "plsda",
        "cars",
        "boss",
        "genetic",
        "ico",
    ],
)
# The selectors are NumPy-only; scikit-learn skips its array-API check by design.
@pytest.mark.filterwarnings("ignore:Skipping check check_array_api_input")
def test_sklearn_estimator_contract(estimator):
    check_estimator(estimator)
