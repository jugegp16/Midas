"""Core data models for the Midas portfolio signal engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum

DEFAULT_MIN_CASH_PCT = 0.05
DEFAULT_REBALANCE_THRESHOLD = 0.02
DEFAULT_SIGMOID_STEEPNESS = 2.0
DEFAULT_MAX_POSITION_PCT = 0.25
DEFAULT_CONVICTION_WEIGHT = 1
DEFAULT_PROTECTIVE_VETO_THRESHOLD = -0.5


class Direction(Enum):
    BUY = "BUY"
    SELL = "SELL"


class HoldingPeriod(Enum):
    SHORT_TERM = "short-term"
    LONG_TERM = "long-term"


class AssetSuitability(Enum):
    BROAD_MARKET_ETF = "broad-market-etf"
    LARGE_CAP = "large-cap"
    INDIVIDUAL_EQUITY = "individual-equity"
    HIGH_VOLATILITY = "high-volatility"
    ALL = "all"


class StrategyTier(Enum):
    CONVICTION = "conviction"
    PROTECTIVE = "protective"
    MECHANICAL = "mechanical"


@dataclass
class Holding:
    ticker: str
    shares: float
    cost_basis: float | None = None


@dataclass
class CashInfusion:
    amount: float
    next_date: date
    frequency: str | None = None


@dataclass
class PortfolioConfig:
    holdings: list[Holding]
    available_cash: float
    cash_infusion: CashInfusion | None = None
    trading_restrictions: TradingRestrictions | None = None

    def __post_init__(self) -> None:
        self._by_ticker = {h.ticker: h for h in self.holdings}

    def get_holding(self, ticker: str) -> Holding | None:
        return self._by_ticker.get(ticker)


@dataclass(frozen=True)
class OrderContext:
    contributions: dict[str, float]
    blended_score: float
    target_weight: float
    current_weight: float
    reason: str
    source: str


@dataclass(frozen=True)
class Order:
    ticker: str
    direction: Direction
    shares: float
    price: float
    estimated_value: float
    context: OrderContext


@dataclass(frozen=True)
class MechanicalIntent:
    ticker: str
    direction: Direction
    target_value: float
    reason: str
    source: str


@dataclass(frozen=True)
class TradeRecord:
    date: date
    ticker: str
    direction: Direction
    shares: float
    price: float
    strategy_name: str
    holding_period: HoldingPeriod | None = None


@dataclass
class TradingRestrictions:
    round_trip_days: int = 0  # 0 = no restriction


@dataclass(frozen=True)
class AllocationConstraints:
    max_position_pct: float | None = None
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT
    rebalance_threshold: float = DEFAULT_REBALANCE_THRESHOLD
    sigmoid_steepness: float = DEFAULT_SIGMOID_STEEPNESS


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, float | int | str] = field(default_factory=dict)
    tickers: list[str] | None = None
    weight: float = DEFAULT_CONVICTION_WEIGHT
    veto_threshold: float = DEFAULT_PROTECTIVE_VETO_THRESHOLD
