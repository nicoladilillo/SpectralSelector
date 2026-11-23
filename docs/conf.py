"""Sphinx configuration for the spectralselector documentation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spectralselector import __version__  # noqa: E402

project = "SpectralSelector"
author = "Nicola Dilillo"
copyright = "2026, Nicola Dilillo"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}

# Wiley (Journal of Chemometrics DOIs) and MathWorks File Exchange answer
# automated checkers with HTTP 403 although the links resolve in a browser.
# Keep the citations, but exclude these endpoints from the availability check.
linkcheck_ignore = [
    r"https://doi\.org/10\.1002/cem\.",
    r"https://www\.mathworks\.com/matlabcentral/fileexchange/",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_title = f"SpectralSelector {release}"
html_logo = "../assets/branding/spectralselector-logo.svg"
html_favicon = "../assets/branding/spectralselector-mark.svg"
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
}
