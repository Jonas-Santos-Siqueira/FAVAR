"""Factor-Augmented Vector Autoregression (FAVAR)."""

from .model import FAVAR
from .order_selection import FAVAROrderSelection
from .results import FAVARResults

__all__ = ["FAVAR", "FAVAROrderSelection", "FAVARResults"]
__version__ = "0.1.3"
