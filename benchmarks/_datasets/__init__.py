"""Spectral benchmark datasets for regression and classification."""

from ._base import get_data_home
from ._chemotools import load_coffee, load_fermentation_test, load_fermentation_train
from ._local import load_lettuce_water_stress
from ._remote import (
    fetch_corn,
    fetch_diesel,
    fetch_shootout,
    fetch_wang2025,
)

__all__ = [
    "fetch_corn",
    "fetch_diesel",
    "fetch_shootout",
    "fetch_wang2025",
    "get_data_home",
    "load_coffee",
    "load_fermentation_test",
    "load_fermentation_train",
    "load_lettuce_water_stress",
]
