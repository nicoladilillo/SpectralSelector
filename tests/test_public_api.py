import inspect
from importlib.metadata import version

import spectralselector
from spectralselector import (
    BaseWavelengthSelector,
    GeneticAlgorithmSelector,
    SelectivityRatioSelector,
    VIPSelector,
)


def test_public_version_and_exports_are_consistent():
    assert spectralselector.__version__ == version("spectralselector")
    for name in spectralselector.__all__:
        assert hasattr(spectralselector, name)


def test_public_aliases_are_identity_preserving_and_scientifically_scoped():
    assert spectralselector.VIPSSelector is VIPSelector
    assert spectralselector.SRSelector is SelectivityRatioSelector
    assert spectralselector.GASelector is GeneticAlgorithmSelector
    assert not hasattr(spectralselector, "PCALDADiscriminantAnalysis")


def test_every_canonical_selector_is_in_the_shared_public_hierarchy():
    names = {
        "IndexSelector",
        "WavelengthSelectKBest",
        "CovSelSelector",
        "FCovSelSelector",
        "VIPSelector",
        "SelectivityRatioSelector",
        "MIVIFSelector",
        "SPASelector",
        "CARSSelector",
        "BOSSSelector",
        "GeneticAlgorithmSelector",
        "ICOSelector",
    }

    for name in names:
        assert issubclass(getattr(spectralselector, name), BaseWavelengthSelector)


def test_public_constructor_parameters_are_documented():
    canonical_names = {
        "PLSDA",
        "IndexSelector",
        "WavelengthSelectKBest",
        "CovSelSelector",
        "FCovSelSelector",
        "VIPSelector",
        "SelectivityRatioSelector",
        "MIVIFSelector",
        "SPASelector",
        "CARSSelector",
        "BOSSSelector",
        "GeneticAlgorithmSelector",
        "ICOSelector",
    }
    for name in canonical_names:
        public_class = getattr(spectralselector, name)
        signature_parameters = set(inspect.signature(public_class).parameters)
        docstring = inspect.getdoc(public_class)
        assert docstring is not None
        parameter_section = docstring.split("Parameters\n----------\n", maxsplit=1)[1]
        for parameter in signature_parameters:
            assert any(
                line.startswith(f"{parameter} :")
                or line.startswith(f"{parameter},")
                or f", {parameter} :" in line
                for line in parameter_section.splitlines()
            ), f"{name}.{parameter} is missing from its Parameters section"
