"""IEQ phase-field solver for two-component lipid vesicles (Wang & Du 2008)."""
from .params import Params
from .discretization.grid import SpectralGrid

__all__ = ["Params", "SpectralGrid"]
