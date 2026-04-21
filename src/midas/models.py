"""Core data models for the Midas portfolio signal engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Literal

DEFAULT_MIN_CASH_PCT = 0.05
DEFAULT_MIN_BUY_DELTA = 0.02
DEFAULT_SOFTMAX_TEMPERATURE = 0.5
DEFAULT_MAX_POSITION_PCT = 0.25
DEFAULT_ENTRY_WEIGHT = 1
DEFAULT_VOL_TARGET_ANNUALIZED = 0.20
DEFAULT_IDM_CAP = 2.5
DEFAULT_VOL_LOOKBACK_DAYS = 60
DEFAULT_CORR_LOOKBACK_DAYS = 252
DEFAULT_VOL_FLOOR = 0.02

FREQUENCY_DAYS: dict[str, int] = {
    "weekly": 7,
    "biweekly": 14,
    "monthly": 30,
}


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

    def advance(self) -> None:
        """Advance next_date by frequency. No-op if frequency is None."""
        if self.frequency is None:
            return
        days = FREQUENCY_DAYS.get(self.frequency)
        if days is None:
            msg = f"Unknown cash_infusion frequency: {self.frequency!r}"
            raise ValueError(msg)
        self.next_date += timedelta(days=days)


@dataclass
class PortfolioConfig:
    holdings: list[Holding]
    available_cash: float
    cash_infusion: CashInfusion | None = None
    trading_restrictions: TradingRestrictions | None = None

    def __post_init__(self) -> None:
        self._by_ticker = {holding.ticker: holding for holding in self.holdings}

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
class PositionLot:
    """A single tax lot for an open position.

    Used by the backtest engine for FIFO sell execution, cost-basis
    accounting, and holding-period classification. Each buy fill appends
    a new lot; each sell consumes lots first-in-first-out.
    """

    shares: float
    purchase_date: date | None
    cost_basis: float


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
    min_buy_delta: float = DEFAULT_MIN_BUY_DELTA
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, float | int | str] = field(default_factory=dict)
    tickers: list[str] | None = None
    weight: float = DEFAULT_ENTRY_WEIGHT


@dataclass(frozen=True)
class RiskConfig:
    """Portfolio-level risk-discipline parameters.

    See docs/superpowers/specs/2026-04-20-risk-discipline-design.md.
    """

    weighting: Literal["inverse_vol", "equal"] = "inverse_vol"
    vol_target_annualized: float = DEFAULT_VOL_TARGET_ANNUALIZED
    idm_cap: float = DEFAULT_IDM_CAP
    vol_lookback_days: int = DEFAULT_VOL_LOOKBACK_DAYS
    corr_lookback_days: int = DEFAULT_CORR_LOOKBACK_DAYS
    vol_floor: float = DEFAULT_VOL_FLOOR

    def __post_init__(self) -> None:
        if self.vol_target_annualized <= 0:
            msg = f"vol_target_annualized must be > 0, got {self.vol_target_annualized}"
            raise ValueError(msg)
        if self.idm_cap < 1.0:
            msg = f"idm_cap must be >= 1.0 (IDM is always >= 1), got {self.idm_cap}"
            raise ValueError(msg)
        if self.vol_lookback_days < 2:
            msg = f"vol_lookback_days must be >= 2, got {self.vol_lookback_days}"
            raise ValueError(msg)
        if self.corr_lookback_days < 2:
            msg = f"corr_lookback_days must be >= 2, got {self.corr_lookback_days}"
            raise ValueError(msg)
        if self.vol_floor < 0:
            msg = f"vol_floor must be >= 0, got {self.vol_floor}"
            raise ValueError(msg)
