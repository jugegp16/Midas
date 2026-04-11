"""Strategy library with auto-registration."""

from midas.strategies.base import EntrySignal, ExitRule, Strategy
from midas.strategies.bollinger_band import BollingerBand
from midas.strategies.gap_down_recovery import GapDownRecovery
from midas.strategies.ma_crossover import MovingAverageCrossover
from midas.strategies.ma_crossover_exit import MovingAverageCrossoverExit
from midas.strategies.macd_crossover import MACDCrossover
from midas.strategies.macd_exit import MACDExit
from midas.strategies.mean_reversion import MeanReversion
from midas.strategies.momentum import Momentum
from midas.strategies.profit_taking import ProfitTaking
from midas.strategies.rsi_oversold import RSIOversold
from midas.strategies.stop_loss import StopLoss
from midas.strategies.trailing_stop import TrailingStop
from midas.strategies.vwap_reversion import VWAPReversion

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    # Entry signals
    "BollingerBand": BollingerBand,
    "GapDownRecovery": GapDownRecovery,
    "MACDCrossover": MACDCrossover,
    "MeanReversion": MeanReversion,
    "Momentum": Momentum,
    "MovingAverageCrossover": MovingAverageCrossover,
    "RSIOversold": RSIOversold,
    "VWAPReversion": VWAPReversion,
    # Exit rules
    "MACDExit": MACDExit,
    "MovingAverageCrossoverExit": MovingAverageCrossoverExit,
    "ProfitTaking": ProfitTaking,
    "StopLoss": StopLoss,
    "TrailingStop": TrailingStop,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "BollingerBand",
    "EntrySignal",
    "ExitRule",
    "GapDownRecovery",
    "MACDCrossover",
    "MACDExit",
    "MeanReversion",
    "Momentum",
    "MovingAverageCrossover",
    "MovingAverageCrossoverExit",
    "ProfitTaking",
    "RSIOversold",
    "StopLoss",
    "Strategy",
    "TrailingStop",
    "VWAPReversion",
]
