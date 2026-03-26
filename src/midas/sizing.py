"""Order sizing engine — turns signals into concrete order suggestions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from midas.models import CashInfusion, Direction, Order, Signal


@dataclass
class SizingConfig:
    default_slippage: float = 0.0005  # 0.05% for large-cap / ETFs
    small_cap_slippage: float = 0.0015  # 0.15% for less liquid names
    circuit_breaker_pct: float = 0.25  # max 25% of cash per day
    infusion_lookahead_days: int = 3


class SizingEngine:
    def __init__(self, config: SizingConfig | None = None) -> None:
        self._config = config or SizingConfig()

    def size_order(
        self,
        signal: Signal,
        available_cash: float,
        position_shares: float,
        cash_infusion: CashInfusion | None = None,
        daily_deployed: float = 0.0,
        today: date | None = None,
        slippage: float | None = None,
    ) -> Order | None:
        """Convert a signal into a sized order. Returns None if filtered out."""
        if signal.direction == Direction.SELL:
            return self._size_sell(signal, position_shares, slippage)
        return self._size_buy(
            signal, available_cash, cash_infusion, daily_deployed, today, slippage
        )

    def _size_sell(
        self,
        signal: Signal,
        position_shares: float,
        slippage: float | None,
    ) -> Order | None:
        # Zero-position filtering
        if position_shares <= 0:
            return None

        slip = slippage if slippage is not None else self._config.default_slippage
        effective_price = signal.price * (1 - slip)
        # Sell whole shares only (floor of position)
        sell_shares = math.floor(position_shares)
        if sell_shares <= 0:
            return None
        estimated_proceeds = sell_shares * effective_price

        return Order(
            ticker=signal.ticker,
            direction=Direction.SELL,
            shares=sell_shares,
            estimated_value=round(estimated_proceeds, 2),
            signal=signal,
        )

    def _size_buy(
        self,
        signal: Signal,
        available_cash: float,
        cash_infusion: CashInfusion | None,
        daily_deployed: float,
        today: date | None,
        slippage: float | None,
    ) -> Order | None:
        today = today or date.today()
        slip = slippage if slippage is not None else self._config.default_slippage
        effective_price = signal.price * (1 + slip)

        if effective_price <= 0:
            return None

        # Determine pending cash
        pending_cash = 0.0
        relies_on_pending = False
        if cash_infusion and cash_infusion.next_date <= today + timedelta(
            days=self._config.infusion_lookahead_days
        ):
            pending_cash = cash_infusion.amount

        total_cash = available_cash + pending_cash

        # Circuit breaker: cap daily deployment
        max_today = total_cash * self._config.circuit_breaker_pct
        remaining_budget = max(0.0, max_today - daily_deployed)

        buyable_cash = min(total_cash, remaining_budget)
        shares = math.floor(buyable_cash / effective_price)

        if shares <= 0:
            return Order(
                ticker=signal.ticker,
                direction=Direction.BUY,
                shares=0,
                estimated_value=0.0,
                signal=signal,
                relies_on_pending_cash=pending_cash > 0,
            )

        cost = shares * effective_price

        # Check if we needed pending cash
        if cost > available_cash and pending_cash > 0:
            relies_on_pending = True

        return Order(
            ticker=signal.ticker,
            direction=Direction.BUY,
            shares=shares,
            estimated_value=round(cost, 2),
            signal=signal,
            relies_on_pending_cash=relies_on_pending,
        )
