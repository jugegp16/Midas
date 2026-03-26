"""Strategy library with auto-registration."""

from midas.strategies.base import Strategy
from midas.strategies.bollinger_band import BollingerBand
from midas.strategies.dca import DollarCostAveraging
from midas.strategies.gap_down_recovery import GapDownRecovery
from midas.strategies.ma_crossover import MovingAverageCrossover
from midas.strategies.macd_crossover import MACDCrossover
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.profit_taking import ProfitTaking
from midas.strategies.rsi_overbought import RSIOverbought
from midas.strategies.rsi_oversold import RSIOversold
from midas.strategies.stop_loss import StopLoss
from midas.strategies.trailing_stop import TrailingStop
from midas.strategies.vwap_reversion import VWAPReversion

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "MeanReversion": MeanReversion,
    "Momentum": Momentum,
    "ProfitTaking": ProfitTaking,
    "RSIOversold": RSIOversold,
    "RSIOverbought": RSIOverbought,
    "BollingerBand": BollingerBand,
    "MACDCrossover": MACDCrossover,
    "DollarCostAveraging": DollarCostAveraging,
    "GapDownRecovery": GapDownRecovery,
    "TrailingStop": TrailingStop,
    "StopLoss": StopLoss,
    "VWAPReversion": VWAPReversion,
    "MovingAverageCrossover": MovingAverageCrossover,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "BollingerBand",
    "DollarCostAveraging",
    "GapDownRecovery",
    "MACDCrossover",
    "MeanReversion",
    "Momentum",
    "MovingAverageCrossover",
    "ProfitTaking",
    "RSIOverbought",
    "RSIOversold",
    "StopLoss",
    "Strategy",
    "TrailingStop",
    "VWAPReversion",
]
