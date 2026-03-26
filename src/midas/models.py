"""Core data models for the Midas portfolio signal engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


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


@dataclass(frozen=True)
class Signal:
    ticker: str
    direction: Direction
    strength: float
    reasoning: str
    timestamp: datetime
    price: float
    strategy_name: str


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

    def __post_init__(self) -> None:
        self._by_ticker = {h.ticker: h for h in self.holdings}

    def get_holding(self, ticker: str) -> Holding | None:
        return self._by_ticker.get(ticker)


@dataclass(frozen=True)
class Order:
    ticker: str
    direction: Direction
    shares: int
    estimated_value: float
    signal: Signal
    relies_on_pending_cash: bool = False


@dataclass(frozen=True)
class TradeRecord:
    date: date
    ticker: str
    direction: Direction
    shares: int
    price: float
    strategy_name: str
    holding_period: HoldingPeriod | None = None


@dataclass
class StrategyConfig:
    name: str
    params: dict[str, float | int | str] = field(default_factory=dict)
    tickers: list[str] | None = None  # None = all tickers
