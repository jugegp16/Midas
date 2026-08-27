"""Parabolic SAR exit: Wilder's self-accelerating trailing stop."""

from __future__ import annotations

import numpy as np

from midas.data.price_history import PriceHistory
from midas.models import AssetSuitability
from midas.strategies.base import ExitRule


class ParabolicSARExit(ExitRule):
    """Exit rule: liquidate when Wilder's Parabolic SAR flips to a downtrend."""

    def __init__(
        self,
        af_start: float = 0.02,
        af_step: float = 0.02,
        af_max: float = 0.20,
    ) -> None:
        self._af_start = af_start
        self._af_step = af_step
        self._af_max = af_max
        # Per-ticker recursion state: (bar_index, high_at_index, low_at_index,
        # sar, ep, af, uptrend) after processing bar ``bar_index``. The engine
        # hands us a one-bar-longer history every day, so advancing from here
        # is O(1) per day instead of replaying the whole series — this loop was
        # ~80% of a backtest trial for strategy sets using this rule. The final
        # bar is never folded into stored state, so a still-forming live bar
        # can never be served stale.
        self._state: dict[str, tuple[int, float, float, float, float, float, bool]] = {}

    @property
    def warmup_period(self) -> int:
        return 2

    def _seed_state(self, high: np.ndarray, low: np.ndarray) -> tuple[float, float, float, bool]:
        """Recursion state ``(sar, ep, af, uptrend)`` after bar 1.

        Wilder's spec uses HIGH for the uptrend extreme point and LOW for the
        downtrend extreme point.
        """
        h0, h1 = float(high[0]), float(high[1])
        l0, l1 = float(low[0]), float(low[1])
        uptrend = h1 >= h0
        if uptrend:
            return min(l0, l1), max(h0, h1), self._af_start, True
        return max(h0, h1), min(l0, l1), self._af_start, False

    def _advance(
        self,
        state: tuple[float, float, float, bool],
        hi: float,
        lo: float,
    ) -> tuple[float, float, float, bool]:
        """Apply one bar to the recursion state.

        The single source of truth for the SAR step: the trend flips when SAR
        crosses LOW (uptrend) or HIGH (downtrend), resetting the extreme point
        and acceleration factor.
        """
        sar, ep, af, uptrend = state
        sar = sar + af * (ep - sar)
        if uptrend:
            if hi > ep:
                ep = hi
                af = min(af + self._af_step, self._af_max)
            if sar > lo:
                return ep, lo, self._af_start, False
        else:
            if lo < ep:
                ep = lo
                af = min(af + self._af_step, self._af_max)
            if sar < hi:
                return ep, hi, self._af_start, True
        return sar, ep, af, uptrend

    def _compute(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> tuple[float, bool] | None:
        """Return (sar, uptrend) at the final bar, or None if too little data.

        The from-scratch reference: replays the whole series. Hot paths use
        :meth:`_compute_incremental`, which is pinned to this by tests.
        """
        num_bars = len(high)
        if num_bars < 2:
            return None
        state = self._seed_state(high, low)
        for i in range(2, num_bars):
            state = self._advance(state, float(high[i]), float(low[i]))
        sar, _ep, _af, uptrend = state
        return sar, uptrend

    def _compute_incremental(
        self,
        ticker: str,
        high: np.ndarray,
        low: np.ndarray,
    ) -> tuple[float, bool] | None:
        """Same result as :meth:`_compute`, advancing cached per-ticker state.

        State is stored through the second-to-last bar only, so the final
        (possibly still-forming) bar is always recomputed. The cache is
        trusted only when the incoming history is a superset of what was
        consumed — same bar index reachable, same high/low at that index;
        anything else (a shorter history, a different series, a walk-forward
        window restart) replays from the seed.
        """
        num_bars = len(high)
        if num_bars < 2:
            return None
        if num_bars == 2:
            sar, ep, af, uptrend = self._seed_state(high, low)
            return sar, uptrend

        target = num_bars - 2  # last bar folded into cached state
        cached = self._state.get(ticker)
        state: tuple[float, float, float, bool] | None = None
        start = 2
        if cached is not None:
            idx, cached_high, cached_low, sar, ep, af, uptrend = cached
            if 1 <= idx <= target and float(high[idx]) == cached_high and float(low[idx]) == cached_low:
                state = (sar, ep, af, uptrend)
                start = idx + 1
        if state is None:
            state = self._seed_state(high, low)
            start = 2
        for i in range(start, target + 1):
            state = self._advance(state, float(high[i]), float(low[i]))

        sar, ep, af, uptrend = state
        self._state[ticker] = (target, float(high[target]), float(low[target]), sar, ep, af, uptrend)

        final_sar, _ep, _af, final_uptrend = self._advance(state, float(high[-1]), float(low[-1]))
        return final_sar, final_uptrend

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
        result = self._compute_incremental(ticker, price_history.high, price_history.low)
        if result is None:
            return proposed_target
        _sar, uptrend = result
        if not uptrend:
            return 0.0
        return proposed_target

    def clamp_reason(
        self,
        ticker: str,
        price_history: PriceHistory,
        cost_basis: float,
        high_water_mark: float,
    ) -> str:
        close = price_history.close
        result = self._compute_incremental(ticker, price_history.high, price_history.low)
        current = float(close[-1]) if len(close) > 0 else 0.0
        if result is None:
            return f"ParabolicSARExit: flip on {ticker}"
        sar, _ = result
        return f"ParabolicSARExit: SAR ${sar:.2f} flipped above price ${current:.2f} (downtrend)"

    @property
    def suitability(self) -> list[AssetSuitability]:
        return [AssetSuitability.ALL]

    @property
    def description(self) -> str:
        return f"Sell when Wilder's Parabolic SAR (AF {self._af_start:.2f}->{self._af_max:.2f}) flips above price"
