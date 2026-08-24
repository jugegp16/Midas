"""Chandelier exit: k x ATR trailing stop off the rolling N-bar highest high."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule, true_range_series


class ChandelierStop(ExitRule):
    """Exit rule: sell when price drops a multiple of ATR below the rolling high."""

    def __init__(self, window: int = 22, multiplier: float = 3.0) -> None:
        self._window = window
        self._multiplier = multiplier
        # Per-ticker ATR recursion state: (bars_consumed, last_close_consumed,
        # atr_through_those_bars). The engine hands us a one-bar-longer history
        # every day; advancing the recursion from here is O(1) per day instead
        # of O(n) — the from-scratch loop was the hottest line in a backtest
        # profile. The final (possibly still-forming, in live mode) bar is
        # never folded into the stored state.
        self._atr_state: dict[str, tuple[int, float, float]] = {}

    @property
    def warmup_period(self) -> int:
        return self._window

    def _wilder_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """Wilder-smoothed ATR over the full available history.

        The seed ATR is the simple mean of the first ``window`` TRs;
        subsequent bars apply the recursive formula
        ``ATR_t = ((window - 1) * ATR_{t-1} + TR_t) / window``.

        With exactly ``window`` bars of history this degenerates to the
        simple mean (no iteration). With more, the exponential decay tail
        matches Wilder's original definition and most charting tools.
        """
        tr = true_range_series(high, low, close)
        window = self._window
        atr = float(tr[:window].mean())
        for idx in range(window, len(close)):
            atr = ((window - 1) * atr + float(tr[idx])) / window
        return atr

    @staticmethod
    def _true_range_at(high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int) -> float:
        """True range of bar *idx* (requires ``idx >= 1``)."""
        prev_close = float(close[idx - 1])
        return max(
            float(high[idx]) - float(low[idx]),
            abs(float(high[idx]) - prev_close),
            abs(float(low[idx]) - prev_close),
        )

    def _wilder_atr_incremental(
        self,
        ticker: str,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> float:
        """ATR equal to :meth:`_wilder_atr`, advancing cached state per ticker.

        The stored state covers bars ``[0, n-1)`` — everything except the
        final bar — so an intraday rewrite of the last bar (live mode) can
        never be served stale. The cache is trusted only when the incoming
        history is a superset of what was consumed: recursion point at or
        past the seed window, and the same last consumed close. Anything
        else (shorter history, a different series, a walk-forward window
        restart) recomputes from scratch via :meth:`_wilder_atr`.
        """
        n_bars = len(close)
        window = self._window
        consumed = n_bars - 1
        if consumed < window:
            # Too short to hold recursion state short of the last bar; the
            # reference loop over <= window+1 bars is effectively free.
            return self._wilder_atr(high, low, close)

        state = self._atr_state.get(ticker)
        if state is not None:
            state_consumed, state_last_close, atr = state
            if window <= state_consumed <= consumed and float(close[state_consumed - 1]) == state_last_close:
                for idx in range(state_consumed, consumed):
                    atr = ((window - 1) * atr + self._true_range_at(high, low, close, idx)) / window
            else:
                state = None
        if state is None:
            atr = self._wilder_atr(high[:consumed], low[:consumed], close[:consumed])

        self._atr_state[ticker] = (consumed, float(close[consumed - 1]), atr)
        return ((window - 1) * atr + self._true_range_at(high, low, close, consumed)) / window

    def _stop_level(self, ticker: str, price_history: PriceHistory) -> tuple[float, float, float] | None:
        """Return ``(stop, highest_high, atr)`` or None if not enough history."""
        close = price_history.close
        if len(close) < self._window:
            return None
        atr = self._wilder_atr_incremental(ticker, price_history.high, price_history.low, close)
        highest = float(price_history.high[-self._window :].max())
        return highest - self._multiplier * atr, highest, atr

    def clamp_target(
        self,
        ticker: str,
        proposed_target: float,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> float:
        if proposed_target <= 0:
            return proposed_target
        level = self._stop_level(ticker, price_history)
        if level is None:
            return proposed_target
        stop, _, _ = level
        current = float(price_history.close[-1])
        if current < stop:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        level = self._stop_level(ticker, price_history)
        current = float(price_history.close[-1])
        if level is None:
            return f"ChandelierStop: insufficient history for {ticker}"
        stop, highest, atr = level
        return (
            f"ChandelierStop: price ${current:.2f} below "
            f"${stop:.2f} (${highest:.2f} peak - "
            f"{self._multiplier:.1f} x ${atr:.2f} ATR{self._window})"
        )

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return (
            f"Sell when price falls {self._multiplier:.1f} x ATR below the "
            f"rolling {self._window}-bar high (volatility-adjusted trailing stop)"
        )
