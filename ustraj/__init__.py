"""A collection of scripts for analysing trajectory files
collected during US simulations"""

import importlib.metadata
from .analysis import count_hbonds
from .analysis import calc_solvent_rdf
from .analysis import calc_RMSD
from .analysis import calc_closest_distance
from .analysis import calc_hbond_distance

try:
    __version__ = importlib.metadata.version("ustraj")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "count_hbonds",
    "calc_solvent_rdf",
    "calc_RMSD",
    "calc_closest_distance",
    "calc_hbond_distance"
]
