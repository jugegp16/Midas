"""Strategy library with auto-registration."""

from midas.strategies.base import Strategy
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.profit_taking import ProfitTaking

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "MeanReversion": MeanReversion,
    "Momentum": Momentum,
    "ProfitTaking": ProfitTaking,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "MeanReversion",
    "Momentum",
    "ProfitTaking",
    "Strategy",
]
