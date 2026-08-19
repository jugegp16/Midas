"""One-shot generator for the golden-test price fixtures.

Run manually: ``uv run python tests/fixtures/golden/generate_fixtures.py``

The committed CSVs are the source of truth for the golden test — this
script exists so they are reproducible, not so they are regenerated
casually. Regenerating the fixtures invalidates every golden file and is
itself a golden-diff event that must be reviewed.

Ticker personalities (so every strategy family actually fires):
    AAA: steady uptrend with two sharp dip-and-recover windows
         (mean-reversion / RSI entries, stop-loss exits).
    BBB: choppy sideways mean-reverter.
    CCC: grinding down-year, then a strong sustained uptrend (momentum).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 76
TICKERS = ("AAA", "BBB", "CCC")
START = date(2023, 1, 3)
END = date(2025, 12, 30)
OUT_DIR = Path(__file__).parent


def _business_days(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _daily_drifts(n_days: int, ticker: str) -> np.ndarray:
    """Per-day deterministic drift component encoding the ticker personality."""
    drift = np.zeros(n_days)
    if ticker == "AAA":
        drift[:] = 0.0006  # steady uptrend
        # Two sharp dip-and-recover windows (~-12% over 8 days, +recovery).
        for dip_start in (200, 520):
            drift[dip_start : dip_start + 8] = -0.016
            drift[dip_start + 8 : dip_start + 30] = 0.005
    elif ticker == "BBB":
        # Sideways chop: alternating mild regimes every ~40 days.
        for i in range(0, n_days, 80):
            drift[i : i + 40] = 0.0012
            drift[i + 40 : i + 80] = -0.0012
    else:  # CCC
        third = n_days // 3
        drift[:third] = -0.0009  # grinding down-year
        drift[third:] = 0.0011  # strong sustained uptrend
    return drift


def generate() -> None:
    rng = np.random.default_rng(SEED)
    days = _business_days(START, END)
    n_days = len(days)

    for ticker, base in zip(TICKERS, (80.0, 45.0, 120.0), strict=True):
        drift = _daily_drifts(n_days, ticker)
        noise = rng.normal(0.0, 0.011, n_days)
        close = base * np.cumprod(1.0 + drift + noise)

        gap = rng.normal(0.0, 0.002, n_days)
        open_ = np.empty(n_days)
        open_[0] = base
        open_[1:] = close[:-1] * (1.0 + gap[1:])

        span_hi = np.abs(rng.normal(0.0, 0.006, n_days)) + 0.001
        span_lo = np.abs(rng.normal(0.0, 0.006, n_days)) + 0.001
        high = np.maximum(open_, close) * (1.0 + span_hi)
        low = np.minimum(open_, close) * (1.0 - span_lo)

        volume = np.round(rng.uniform(0.5e6, 2.5e6, n_days), 0)

        frame = pd.DataFrame(
            {
                "date": [d.isoformat() for d in days],
                "open": np.round(open_, 4),
                "high": np.round(high, 4),
                "low": np.round(low, 4),
                "close": np.round(close, 4),
                "volume": volume,
            }
        )
        out_path = OUT_DIR / f"{ticker}.csv"
        frame.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({n_days} rows, close {close[0]:.2f} -> {close[-1]:.2f})")


if __name__ == "__main__":
    existing = [f"{t}.csv" for t in TICKERS if (OUT_DIR / f"{t}.csv").exists()]
    if existing and "--force" not in sys.argv:
        raise SystemExit(
            f"fixture CSVs already exist ({', '.join(existing)}); regenerating "
            "invalidates every golden. Re-run with --force if that is the intent."
        )
    generate()
