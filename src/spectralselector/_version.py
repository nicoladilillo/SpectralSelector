"""Package version resolved from installed distribution metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spectralselector")
except PackageNotFoundError:  # pragma: no cover - only for an uninstalled source tree
    __version__ = "0+unknown"
