"""A collection of scripts for analysing trajectory files collected during US simulations"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("ustraj")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = ["__version__"]
