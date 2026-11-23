"""Wavelength selectors compatible with NumPy and scikit-learn pipelines."""

from ._base import BaseWavelengthSelector
from ._version import __version__
from .boss import BOSSSelector
from .cars import CARSSelector
from .covsel import CovSelSelector
from .fcovsel import FCovSelSelector
from .fixed import IndexSelector
from .genetic import GASelector, GeneticAlgorithmSelector
from .ico import ICOSelector
from .mivif import MIVIFSelector
from .model_based import (
    SelectivityRatioSelector,
    SRSelector,
    VIPSelector,
    VIPSSelector,
)
from .plsda import PLSDA
from .spa import SPASelector
from .univariate import WavelengthSelectKBest

__all__ = [
    "PLSDA",
    "BOSSSelector",
    "BaseWavelengthSelector",
    "CARSSelector",
    "CovSelSelector",
    "FCovSelSelector",
    "GASelector",
    "GeneticAlgorithmSelector",
    "ICOSelector",
    "IndexSelector",
    "MIVIFSelector",
    "SPASelector",
    "SRSelector",
    "SelectivityRatioSelector",
    "VIPSSelector",
    "VIPSelector",
    "WavelengthSelectKBest",
    "__version__",
]
