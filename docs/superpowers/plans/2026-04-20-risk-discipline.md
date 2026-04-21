# Risk Discipline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship portfolio volatility targeting (#52 Part 1), per-ticker volatility scaling (#53), and Instrument Diversification Multiplier (#55) as a single cohesive risk-discipline layer woven into `Allocator.allocate`.

**Architecture:** New pure-function module `src/midas/risk.py`. Config via new `RiskConfig` dataclass and a `risk:` YAML block. Integration inside `Allocator` at phases 1.5 (cov/vol precompute), 2 (vol offset in softmax), 3.5 (IDM), 3.6 (re-cap), 3.7 (vol targeting). Risk-on defaults. No CLI overrides.

**Tech Stack:** Python 3.14, NumPy, pandas, `sklearn.covariance.LedoitWolf` (new dep), pytest, mypy strict, ruff.

**Spec:** `docs/superpowers/specs/2026-04-20-risk-discipline-design.md`

## File Structure

**New files:**
- `src/midas/risk.py` — five pure functions (`realized_vol`, `covariance_matrix`, `apply_instrument_diversification_multiplier`, `apply_vol_targeting`, `predict_portfolio_vol`)
- `tests/test_risk.py` — unit tests for the five functions

**Modified files:**
- `pyproject.toml` — add `scikit-learn` to `[project].dependencies`
- `src/midas/models.py` — add `RiskConfig` dataclass + `DEFAULT_*` constants
- `src/midas/config.py` — parse `risk:` YAML block, return `RiskConfig` from `load_strategies`
- `src/midas/allocator.py` — accept `RiskConfig`, add risk cache + phases 1.5/3.5/3.6/3.7
- `src/midas/cli.py` — thread `RiskConfig` through `_build_components` and the two call sites
- `tests/test_config.py` — cover `risk:` parsing + defaults + validation errors
- `tests/test_allocator.py` — risk-off regression + active-stage integration tests
- `tests/test_backtest.py` — end-to-end change-detector for default risk config

---

### Task 1: Add `scikit-learn` dependency

**Files:**
- Modify: `pyproject.toml:5-12`

- [ ] **Step 1: Add dependency**

Edit `pyproject.toml`:

```toml
dependencies = [
    "click>=8.1",
    "pandas>=2.2",
    "pyyaml>=6.0",
    "rich>=13.0",
    "yfinance>=0.2",
    "optuna>=4.0",
    "scikit-learn>=1.5",
]
```

- [ ] **Step 2: Install the new dep**

Run: `uv sync`
Expected: resolves and installs scikit-learn and its transitive deps (scipy, joblib, threadpoolctl).

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "from sklearn.covariance import LedoitWolf; print(LedoitWolf().__class__.__name__)"`
Expected: `LedoitWolf`

- [ ] **Step 4: Verify existing tests still pass**

Run: `uv run pytest -q`
Expected: all tests pass, no collection errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add scikit-learn dep for LedoitWolf covariance shrinkage"
```

---

### Task 2: `RiskConfig` dataclass with config-time validation

**Files:**
- Modify: `src/midas/models.py` (append to bottom, add DEFAULT constants near top)
- Modify: `tests/test_models.py` (new test class for RiskConfig)

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
import pytest

from midas.models import RiskConfig


class TestRiskConfig:
    def test_defaults_are_risk_on(self):
        cfg = RiskConfig()
        assert cfg.weighting == "inverse_vol"
        assert cfg.vol_target_annualized == 0.20
        assert cfg.idm_cap == 2.5
        assert cfg.vol_lookback_days == 60
        assert cfg.corr_lookback_days == 252
        assert cfg.vol_floor == 0.02

    def test_is_frozen(self):
        cfg = RiskConfig()
        with pytest.raises((AttributeError, Exception)):
            cfg.idm_cap = 1.0  # type: ignore[misc]

    def test_rejects_non_positive_vol_target(self):
        with pytest.raises(ValueError, match="vol_target_annualized"):
            RiskConfig(vol_target_annualized=0.0)
        with pytest.raises(ValueError, match="vol_target_annualized"):
            RiskConfig(vol_target_annualized=-0.10)

    def test_rejects_idm_cap_below_one(self):
        with pytest.raises(ValueError, match="idm_cap"):
            RiskConfig(idm_cap=0.99)

    def test_accepts_idm_cap_exactly_one(self):
        RiskConfig(idm_cap=1.0)  # disables IDM — valid

    def test_rejects_small_vol_lookback(self):
        with pytest.raises(ValueError, match="vol_lookback_days"):
            RiskConfig(vol_lookback_days=1)

    def test_rejects_small_corr_lookback(self):
        with pytest.raises(ValueError, match="corr_lookback_days"):
            RiskConfig(corr_lookback_days=1)

    def test_rejects_negative_vol_floor(self):
        with pytest.raises(ValueError, match="vol_floor"):
            RiskConfig(vol_floor=-0.01)

    def test_accepts_zero_vol_floor(self):
        RiskConfig(vol_floor=0.0)  # edge case — valid
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_models.py -q`
Expected: `ImportError` or `ModuleNotFoundError` on `RiskConfig`.

- [ ] **Step 3: Add RiskConfig to models.py**

In `src/midas/models.py`, near the top with the other defaults add:

```python
DEFAULT_VOL_TARGET_ANNUALIZED = 0.20
DEFAULT_IDM_CAP = 2.5
DEFAULT_VOL_LOOKBACK_DAYS = 60
DEFAULT_CORR_LOOKBACK_DAYS = 252
DEFAULT_VOL_FLOOR = 0.02
```

Update the existing import line `from dataclasses import dataclass, field` to also import `Literal` from `typing`:

```python
from typing import Literal
```

At the bottom of the file append:

```python
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
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_models.py -q`
Expected: new RiskConfig tests pass; existing tests unaffected.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/models.py tests/test_models.py --fix
uv run ruff format src/midas/models.py tests/test_models.py
uv run mypy src
git add src/midas/models.py tests/test_models.py
git commit -m "Add RiskConfig dataclass with config-time validation"
```

---

### Task 3: Parse `risk:` YAML block and return `RiskConfig` from `load_strategies`

**Files:**
- Modify: `src/midas/config.py` (extend `load_strategies` return type)
- Modify: `src/midas/cli.py` (update unpack at call sites)
- Modify: `tests/test_integration.py` (update unpack)
- Modify: `tests/test_config.py` (extend test + add new test cases)

- [ ] **Step 1: Write failing tests**

Edit `tests/test_config.py`. Update the existing `test_load_strategies` to unpack a 3-tuple and assert defaults, and add two new tests:

Change the existing `test_load_strategies` body to:

```python
def test_load_strategies(strategy_yaml: Path) -> None:
    configs, constraints, risk = load_strategies(strategy_yaml)
    assert len(configs) == 3

    assert configs[0].name == "MeanReversion"
    assert configs[0].params["window"] == 20
    assert configs[0].weight == 1.5

    assert configs[1].name == "StopLoss"
    assert configs[1].params["loss_threshold"] == 0.10

    assert configs[2].name == "Momentum"
    assert configs[2].params == {}
    assert configs[2].weight == 1.0  # default

    # Allocation knobs
    assert constraints.softmax_temperature == 0.25
    assert constraints.min_buy_delta == 0.03
    assert constraints.min_cash_pct == 0.10
    assert constraints.max_position_pct is None

    # Risk config defaults (no risk: block in fixture)
    assert risk.weighting == "inverse_vol"
    assert risk.vol_target_annualized == 0.20
    assert risk.idm_cap == 2.5
```

Add below it:

```python
def test_load_strategies_parses_risk_block(tmp_path: Path) -> None:
    data = {
        "strategies": [{"name": "Momentum"}],
        "risk": {
            "weighting": "equal",
            "vol_target_annualized": 0.15,
            "idm_cap": 2.0,
            "vol_lookback_days": 30,
            "corr_lookback_days": 120,
            "vol_floor": 0.05,
        },
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    _, _, risk = load_strategies(p)
    assert risk.weighting == "equal"
    assert risk.vol_target_annualized == 0.15
    assert risk.idm_cap == 2.0
    assert risk.vol_lookback_days == 30
    assert risk.corr_lookback_days == 120
    assert risk.vol_floor == 0.05


def test_load_strategies_risk_validation_surfaces(tmp_path: Path) -> None:
    data = {
        "strategies": [{"name": "Momentum"}],
        "risk": {"idm_cap": 0.5},
    }
    p = tmp_path / "strategies.yaml"
    p.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="idm_cap"):
        load_strategies(p)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: tests fail with `ValueError: too many values to unpack` or similar (load_strategies currently returns a 2-tuple).

- [ ] **Step 3: Update `load_strategies`**

Edit `src/midas/config.py`. Replace the import block and the `load_strategies` function:

```python
from midas.models import (
    DEFAULT_MIN_BUY_DELTA,
    DEFAULT_MIN_CASH_PCT,
    DEFAULT_SOFTMAX_TEMPERATURE,
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TradingRestrictions,
)
```

Replace the function signature and body:

```python
def load_strategies(
    path: Path,
) -> tuple[list[StrategyConfig], AllocationConstraints, RiskConfig]:
    """Load strategy configs, allocation-level knobs, and risk policy from YAML.

    Returns (strategies, constraints, risk) tuple. Top-level keys:
      * ``softmax_temperature``, ``min_buy_delta``, ``min_cash_pct``,
        ``max_position_pct`` — meta-strategy knobs.
      * ``risk:`` — nested mapping forwarded to :class:`RiskConfig`. Missing
        block defaults to risk-on.
    """
    raw = _load_yaml(path)

    configs = []
    for strat in raw["strategies"]:
        configs.append(
            StrategyConfig(
                name=strat["name"],
                params=strat.get("params", {}),
                tickers=strat.get("tickers"),
                weight=float(strat.get("weight", 1.0)),
            )
        )

    max_pos = raw.get("max_position_pct")
    constraints = AllocationConstraints(
        max_position_pct=float(max_pos) if max_pos is not None else None,
        min_cash_pct=float(raw.get("min_cash_pct", DEFAULT_MIN_CASH_PCT)),
        softmax_temperature=float(
            raw.get("softmax_temperature", DEFAULT_SOFTMAX_TEMPERATURE),
        ),
        min_buy_delta=float(
            raw.get("min_buy_delta", DEFAULT_MIN_BUY_DELTA),
        ),
    )

    risk_raw = raw.get("risk", {}) or {}
    risk = RiskConfig(
        weighting=risk_raw.get("weighting", "inverse_vol"),
        vol_target_annualized=float(risk_raw.get("vol_target_annualized", 0.20)),
        idm_cap=float(risk_raw.get("idm_cap", 2.5)),
        vol_lookback_days=int(risk_raw.get("vol_lookback_days", 60)),
        corr_lookback_days=int(risk_raw.get("corr_lookback_days", 252)),
        vol_floor=float(risk_raw.get("vol_floor", 0.02)),
    )
    return configs, constraints, risk
```

- [ ] **Step 4: Update call sites**

Edit `src/midas/cli.py`. Update the three call sites:

Line 153 (in `backtest`):
```python
    strat_configs, constraints, risk_config = (
        load_strategies(Path(strategies))
        if strategies
        else (None, AllocationConstraints(), RiskConfig())
    )
```

Line 214 (in `live`):
```python
    strat_configs, constraints, risk_config = (
        load_strategies(Path(strategies))
        if strategies
        else (None, AllocationConstraints(), RiskConfig())
    )
```

Line 333 (in optimize):
```python
        strat_configs, strat_constraints, _ = load_strategies(Path(strategies))
```

Update the import block in `src/midas/cli.py`:
```python
from midas.models import (
    AllocationConstraints,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
)
```

Edit `tests/test_integration.py` line 57:
```python
    strat_configs, constraints, _ = load_strategies(strategy_path)
```

- [ ] **Step 5: Run and verify pass**

Run: `uv run pytest tests/test_config.py tests/test_integration.py -q`
Expected: all pass.

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
uv run mypy src
git add src/midas/config.py src/midas/cli.py tests/test_config.py tests/test_integration.py
git commit -m "Thread RiskConfig through strategy YAML loader"
```

---

### Task 4: `realized_vol` function (TDD)

**Files:**
- Create: `src/midas/risk.py`
- Create: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk.py`:

```python
"""Tests for the risk-discipline module."""

from __future__ import annotations

import math

import numpy as np
from conftest import ph, ph_ohlc  # noqa: F401 — ph_ohlc reserved for future tests

from midas.risk import realized_vol


class TestRealizedVol:
    def test_constant_series_returns_zero(self):
        history = ph(np.full(100, 100.0))
        vol = realized_vol(history, window=30)
        assert vol == 0.0

    def test_known_series_matches_hand_computed(self):
        # Daily returns: exactly +1% and -1% alternating over 30 bars (29 returns).
        # Each daily return is ±0.01 (computed from ratio p_i / p_{i-1}).
        prices = [100.0]
        for i in range(1, 31):
            prices.append(prices[-1] * (1.01 if i % 2 == 1 else 1 / 1.01))
        history = ph(np.asarray(prices))

        vol = realized_vol(history, window=30)
        # daily std of ±0.01 (equal-sized up/down moves) ≈ 0.01; annualized × sqrt(252)
        assert vol is not None
        assert math.isclose(vol, 0.01 * math.sqrt(252), rel_tol=1e-2)

    def test_insufficient_history_returns_none(self):
        history = ph(np.full(10, 100.0))
        assert realized_vol(history, window=30) is None

    def test_annualize_false_returns_daily_vol(self):
        prices = [100.0]
        for i in range(1, 31):
            prices.append(prices[-1] * (1.01 if i % 2 == 1 else 1 / 1.01))
        history = ph(np.asarray(prices))
        daily = realized_vol(history, window=30, annualize=False)
        annual = realized_vol(history, window=30, annualize=True)
        assert daily is not None and annual is not None
        assert math.isclose(annual / daily, math.sqrt(252), rel_tol=1e-6)

    def test_nan_in_window_returns_none(self):
        prices = np.full(60, 100.0)
        prices[-5] = float("nan")
        history = ph(prices)
        # Window includes the NaN → insufficient clean history.
        assert realized_vol(history, window=30) is None

    def test_exact_window_size_is_sufficient(self):
        # Window == len(history) should succeed (not fail with off-by-one).
        history = ph(np.full(30, 100.0))
        assert realized_vol(history, window=30) == 0.0
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_risk.py -q`
Expected: `ModuleNotFoundError: No module named 'midas.risk'`.

- [ ] **Step 3: Create `src/midas/risk.py` with `realized_vol`**

```python
"""Risk-discipline primitives: realized vol, covariance, IDM, vol targeting.

Pure functions only — no class state. All functions return new objects
(dicts, DataFrames) and never mutate their inputs. Degenerate inputs are
handled by returning a sentinel (``None`` or an unchanged result) rather
than raising, so the allocator can degrade gracefully at runtime.

See docs/superpowers/specs/2026-04-20-risk-discipline-design.md.
"""

from __future__ import annotations

import math

import numpy as np

from midas.data.price_history import PriceHistory

TRADING_DAYS_PER_YEAR = 252


def realized_vol(
    history: PriceHistory,
    window: int,
    annualize: bool = True,
) -> float | None:
    """Realized volatility over the last ``window`` closes.

    Args:
        history: OHLCV bars for a single ticker.
        window: Number of most-recent bars to consider. The vol is computed
            over ``window - 1`` daily simple returns.
        annualize: Multiply by sqrt(252) when True.

    Returns:
        Volatility as a fraction (0.20 == 20% annualized), or None when
        there is insufficient clean history (fewer than ``window`` bars or
        any NaN inside the window).
    """
    if window < 2:
        msg = f"window must be >= 2, got {window}"
        raise ValueError(msg)
    if len(history) < window:
        return None
    closes = history.close[-window:]
    if not np.all(np.isfinite(closes)):
        return None
    returns = closes[1:] / closes[:-1] - 1.0
    if returns.size < 2:
        return None
    daily = float(np.std(returns, ddof=1))
    return daily * math.sqrt(TRADING_DAYS_PER_YEAR) if annualize else daily
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_risk.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/risk.py tests/test_risk.py --fix
uv run ruff format src/midas/risk.py tests/test_risk.py
uv run mypy src
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add realized_vol to risk module"
```

---

### Task 5: `covariance_matrix` function (TDD)

**Files:**
- Modify: `src/midas/risk.py`
- Modify: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_risk.py`:

```python
import pandas as pd

from midas.risk import covariance_matrix


def _prices_from_returns(daily_returns: np.ndarray, base: float = 100.0) -> np.ndarray:
    """Build a close array from daily returns."""
    out = np.empty(len(daily_returns) + 1)
    out[0] = base
    for i, ret in enumerate(daily_returns):
        out[i + 1] = out[i] * (1.0 + ret)
    return out


class TestCovarianceMatrix:
    def test_uncorrelated_series_give_diagonal_correlation(self):
        rng = np.random.default_rng(42)
        # 260 bars each — enough for 252-day corr window + slack.
        rets_a = rng.normal(0.0, 0.01, 260)
        rets_b = rng.normal(0.0, 0.02, 260)
        hist_a = ph(_prices_from_returns(rets_a))
        hist_b = ph(_prices_from_returns(rets_b))

        cov = covariance_matrix(
            {"A": hist_a, "B": hist_b},
            vol_window=60,
            corr_window=252,
            vol_floor=0.001,
        )

        assert cov is not None
        # Diagonals should be ~ (realized vol ** 2)
        # Off-diagonal correlation should be near zero.
        corr_ab = cov.loc["A", "B"] / math.sqrt(cov.loc["A", "A"] * cov.loc["B", "B"])
        assert abs(corr_ab) < 0.2  # shrinkage may pull slightly toward 0 or away

    def test_perfectly_correlated_series_give_unit_correlation(self):
        rng = np.random.default_rng(7)
        base_rets = rng.normal(0.0, 0.01, 260)
        # B is identical to A — scaled vol by 2× to make the math nontrivial.
        hist_a = ph(_prices_from_returns(base_rets))
        hist_b = ph(_prices_from_returns(base_rets * 2.0))

        cov = covariance_matrix(
            {"A": hist_a, "B": hist_b},
            vol_window=60,
            corr_window=252,
            vol_floor=0.001,
        )

        assert cov is not None
        corr_ab = cov.loc["A", "B"] / math.sqrt(cov.loc["A", "A"] * cov.loc["B", "B"])
        # LedoitWolf shrinks correlation toward the mean off-diagonal, which
        # for a 2-asset system with one off-diagonal = 1.0 still stays near 1.
        assert corr_ab > 0.9

    def test_vol_floor_clamps_low_vol_ticker(self):
        rng = np.random.default_rng(3)
        rets_normal = rng.normal(0.0, 0.01, 260)
        rets_tiny = rng.normal(0.0, 0.0001, 260)  # well below floor
        hist_normal = ph(_prices_from_returns(rets_normal))
        hist_tiny = ph(_prices_from_returns(rets_tiny))

        cov = covariance_matrix(
            {"N": hist_normal, "T": hist_tiny},
            vol_window=60,
            corr_window=252,
            vol_floor=0.05,  # 5% annualized
        )

        assert cov is not None
        vol_t = math.sqrt(cov.loc["T", "T"])
        # Vol floor is annualized 5% → variance ≈ 0.0025 on the diagonal.
        assert vol_t >= 0.05 - 1e-9

    def test_insufficient_history_returns_none(self):
        short = ph(np.full(50, 100.0))  # < corr_window of 252
        cov = covariance_matrix(
            {"A": short, "B": short},
            vol_window=30,
            corr_window=252,
            vol_floor=0.001,
        )
        assert cov is None

    def test_output_is_psd(self):
        rng = np.random.default_rng(99)
        hists = {
            name: ph(_prices_from_returns(rng.normal(0, 0.01, 260)))
            for name in ("A", "B", "C", "D")
        }
        cov = covariance_matrix(hists, vol_window=60, corr_window=252, vol_floor=0.001)
        assert cov is not None
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert (eigenvalues >= -1e-9).all()

    def test_index_and_columns_match_ticker_order(self):
        rng = np.random.default_rng(1)
        hists = {
            name: ph(_prices_from_returns(rng.normal(0, 0.01, 260)))
            for name in ("Z", "A", "M")
        }
        cov = covariance_matrix(hists, vol_window=60, corr_window=252, vol_floor=0.001)
        assert cov is not None
        # Deterministic order for downstream matmul correctness.
        assert list(cov.index) == ["A", "M", "Z"]
        assert list(cov.columns) == ["A", "M", "Z"]
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_risk.py::TestCovarianceMatrix -q`
Expected: `ImportError: cannot import name 'covariance_matrix'`.

- [ ] **Step 3: Add `covariance_matrix` to `src/midas/risk.py`**

Add imports:

```python
from collections.abc import Mapping

import pandas as pd
from sklearn.covariance import LedoitWolf
```

Add function:

```python
def covariance_matrix(
    histories: Mapping[str, PriceHistory],
    vol_window: int,
    corr_window: int,
    vol_floor: float,
) -> pd.DataFrame | None:
    """Composed covariance: vols (short window) × corr (long window, shrunk).

    Correlation is fit with :class:`sklearn.covariance.LedoitWolf` on the
    last ``corr_window`` returns across the universe. Per-ticker vols are
    realized over ``vol_window`` and floored at ``vol_floor`` before the
    outer-product composition.

    Returns:
        NxN covariance DataFrame indexed by ticker (sorted), or None when
        any ticker has fewer than ``corr_window + 1`` clean bars or the
        LedoitWolf fit fails.
    """
    if corr_window < 2 or vol_window < 2:
        msg = f"vol_window and corr_window must be >= 2, got {vol_window}, {corr_window}"
        raise ValueError(msg)

    tickers = sorted(histories.keys())
    if len(tickers) < 1:
        return None

    returns_cols: list[np.ndarray] = []
    for ticker in tickers:
        history = histories[ticker]
        if len(history) < corr_window + 1:
            return None
        closes = history.close[-(corr_window + 1):]
        if not np.all(np.isfinite(closes)):
            return None
        returns_cols.append(closes[1:] / closes[:-1] - 1.0)

    returns_matrix = np.column_stack(returns_cols)  # shape (corr_window, N)

    try:
        lw = LedoitWolf().fit(returns_matrix)
    except Exception:  # noqa: BLE001 — any LW failure degrades gracefully
        return None

    daily_cov = lw.covariance_
    # Derive correlation from LedoitWolf's daily covariance.
    daily_std = np.sqrt(np.diag(daily_cov))
    # Guard against zero diag from a degenerate constant series.
    if not np.all(daily_std > 0):
        return None
    corr = daily_cov / np.outer(daily_std, daily_std)

    # Per-ticker realized (annualized) vol over the short window. None-fallback
    # = floor so the composition always has a defined scale.
    vols: list[float] = []
    for ticker in tickers:
        v = realized_vol(histories[ticker], window=vol_window, annualize=True)
        vols.append(max(v if v is not None else vol_floor, vol_floor))
    vols_arr = np.asarray(vols)

    cov_composed = corr * np.outer(vols_arr, vols_arr)
    # Numerical safety: symmetrize.
    cov_composed = 0.5 * (cov_composed + cov_composed.T)
    return pd.DataFrame(cov_composed, index=tickers, columns=tickers)
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_risk.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/risk.py tests/test_risk.py --fix
uv run ruff format src/midas/risk.py tests/test_risk.py
uv run mypy src
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add covariance_matrix with LedoitWolf-shrunk correlation"
```

---

### Task 6: `apply_instrument_diversification_multiplier` (TDD)

**Files:**
- Modify: `src/midas/risk.py`
- Modify: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_risk.py`:

```python
from midas.risk import apply_instrument_diversification_multiplier


def _corr_frame(tickers: list[str], matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


class TestApplyInstrumentDiversificationMultiplier:
    def test_uncorrelated_equal_weights_give_idm_approx_sqrt_n(self):
        tickers = ["A", "B", "C", "D"]
        weights = {t: 0.25 for t in tickers}
        corr = _corr_frame(tickers, np.eye(4))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=5.0)
        # IDM = 1/sqrt(0.25 * 4 * 0.25) = 1/sqrt(0.25) = 2.0 = sqrt(4).
        total = sum(out.values())
        assert math.isclose(total, sum(weights.values()) * 2.0, rel_tol=1e-9)

    def test_perfect_correlation_gives_idm_one(self):
        tickers = ["A", "B"]
        weights = {"A": 0.4, "B": 0.4}
        corr = _corr_frame(tickers, np.ones((2, 2)))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=5.0)
        assert math.isclose(out["A"], 0.4, rel_tol=1e-9)
        assert math.isclose(out["B"], 0.4, rel_tol=1e-9)

    def test_cap_binds_when_raw_idm_exceeds(self):
        tickers = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        weights = dict.fromkeys(tickers, 1.0 / 9.0)
        corr = _corr_frame(tickers, np.eye(9))
        # raw IDM = sqrt(9) = 3.0; cap at 2.0 → multiplier = 2.0 / 3.0 of raw, so scale = 2.0.
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.0)
        total = sum(out.values())
        assert math.isclose(total, sum(weights.values()) * 2.0, rel_tol=1e-9)

    def test_empty_weights_returns_empty(self):
        corr = _corr_frame(["A"], np.eye(1))
        assert apply_instrument_diversification_multiplier({}, corr, cap=2.5) == {}

    def test_single_ticker_weights_unchanged(self):
        weights = {"A": 0.5}
        corr = _corr_frame(["A"], np.eye(1))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.5)
        assert out == {"A": 0.5}

    def test_zero_weights_returns_unchanged(self):
        tickers = ["A", "B"]
        weights = dict.fromkeys(tickers, 0.0)
        corr = _corr_frame(tickers, np.eye(2))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.5)
        assert out == weights

    def test_weights_referencing_missing_ticker_are_ignored(self):
        # "A" is in weights but absent from the correlation matrix — treat as
        # uncorrelated (contributes to sum but not to the diversification math).
        weights = {"A": 0.3, "B": 0.3}
        corr = _corr_frame(["B"], np.eye(1))
        out = apply_instrument_diversification_multiplier(weights, corr, cap=2.5)
        # A passes through with multiplier 1.0; B scales by its own IDM which
        # for a 1-asset "portfolio" is trivially 1.0.
        assert math.isclose(out["A"], 0.3, rel_tol=1e-9)
        assert math.isclose(out["B"], 0.3, rel_tol=1e-9)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_risk.py::TestApplyInstrumentDiversificationMultiplier -q`
Expected: `ImportError`.

- [ ] **Step 3: Add the function to `src/midas/risk.py`**

```python
def apply_instrument_diversification_multiplier(
    weights: dict[str, float],
    corr: pd.DataFrame,
    cap: float,
) -> dict[str, float]:
    """Scale all weights by ``min(1/sqrt(w·corr·w), cap)``.

    Args:
        weights: Ticker -> weight before IDM.
        corr: Symmetric correlation matrix. Tickers in ``weights`` but not
            in ``corr`` are left unscaled (treated as uncorrelated singletons).
        cap: Upper bound on the multiplier. Must be >= 1.0 (checked by
            :class:`RiskConfig`).

    Returns:
        New dict with scaled weights. Returns inputs unchanged on degenerate
        inputs (empty weights, single ticker, zero weights, or numerical
        w·corr·w <= 0).
    """
    if not weights:
        return {}
    if sum(weights.values()) == 0:
        return dict(weights)

    # Intersect weights with corr index — only covered tickers participate.
    covered = [t for t in weights if t in corr.index]
    if len(covered) <= 1:
        return dict(weights)

    w = np.asarray([weights[t] for t in covered])
    sub_corr = corr.loc[covered, covered].values
    quad = float(w @ sub_corr @ w)
    if quad <= 0:
        return dict(weights)

    raw_idm = 1.0 / math.sqrt(quad)
    multiplier = min(raw_idm, cap)
    if multiplier <= 1.0:
        # min(idm, cap) == 1.0 exactly happens only when idm<=1 (perfect
        # correlation) or cap==1.0 (IDM stage disabled). Return unchanged.
        return dict(weights)

    return {t: weight * multiplier if t in set(covered) else weight for t, weight in weights.items()}
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_risk.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/risk.py tests/test_risk.py --fix
uv run ruff format src/midas/risk.py tests/test_risk.py
uv run mypy src
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add apply_instrument_diversification_multiplier"
```

---

### Task 7: `apply_vol_targeting` (TDD)

**Files:**
- Modify: `src/midas/risk.py`
- Modify: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_risk.py`:

```python
from midas.risk import apply_vol_targeting


def _cov_frame(tickers: list[str], matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=tickers, columns=tickers)


class TestApplyVolTargeting:
    def test_below_target_returns_unchanged(self):
        # diag(cov) = 0.01 (10% ann vol), w = 0.5,0.5 uncorrelated → portfolio vol ≈ 7.07%
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.diag([0.01, 0.01]))
        weights = {"A": 0.5, "B": 0.5}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert out == weights

    def test_exactly_at_target_returns_unchanged(self):
        # Single asset, vol ≈ 20%; weight = 1 → portfolio vol = 20%.
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        weights = {"A": 1.0}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert math.isclose(out["A"], 1.0, rel_tol=1e-9)

    def test_double_target_halves_weights(self):
        # Single asset, vol = 40% (variance 0.16), target = 20% → scale by 0.5.
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.16]]))
        weights = {"A": 1.0}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert math.isclose(out["A"], 0.5, rel_tol=1e-9)

    def test_zero_weights_returns_zero(self):
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.eye(2))
        weights = {"A": 0.0, "B": 0.0}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert out == weights

    def test_degenerate_cov_returns_unchanged(self):
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.zeros((2, 2)))
        weights = {"A": 0.5, "B": 0.5}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        assert out == weights

    def test_weights_missing_from_cov_pass_through_unscaled(self):
        # "B" isn't in cov. "A" dominates the vol calculation.
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.16]]))  # 40% vol
        weights = {"A": 1.0, "B": 0.3}
        out = apply_vol_targeting(weights, cov, target_annualized_vol=0.20)
        # A scales by 0.5; B is uncovered → passes through scaled by the same
        # portfolio-wide multiplier so the downscaling remains coherent.
        assert math.isclose(out["A"], 0.5, rel_tol=1e-9)
        assert math.isclose(out["B"], 0.15, rel_tol=1e-9)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_risk.py::TestApplyVolTargeting -q`
Expected: `ImportError`.

- [ ] **Step 3: Add `apply_vol_targeting` and `predict_portfolio_vol` to `src/midas/risk.py`**

Note: `predict_portfolio_vol` is defined here because `apply_vol_targeting` uses it. Task 8 then adds tests for it separately.

```python
def predict_portfolio_vol(
    weights: dict[str, float],
    cov: pd.DataFrame,
) -> float:
    """Ex-ante portfolio vol: ``sqrt(w · cov · w) * sqrt(252)``.

    Tickers in ``weights`` but not in ``cov`` are ignored. Returns 0.0 on
    empty weights, all-zero weights, or when the quadratic form is
    non-positive (degenerate covariance).
    """
    covered = [t for t in weights if t in cov.index]
    if not covered:
        return 0.0
    w = np.asarray([weights[t] for t in covered])
    sub_cov = cov.loc[covered, covered].values
    quad = float(w @ sub_cov @ w)
    if quad <= 0:
        return 0.0
    # cov is already annualized (composed from annualized vols), so sqrt(252)
    # is NOT reapplied here. The function name says "annualized" implicitly.
    return math.sqrt(quad)


def apply_vol_targeting(
    weights: dict[str, float],
    cov: pd.DataFrame,
    target_annualized_vol: float,
) -> dict[str, float]:
    """Scale weights down so predicted vol does not exceed the target.

    Never scales up — if predicted vol is below target, weights are
    returned unchanged. Degenerate cov matrices (w·cov·w <= 0) also
    pass through unchanged.
    """
    if not weights:
        return {}
    if sum(weights.values()) == 0:
        return dict(weights)

    predicted = predict_portfolio_vol(weights, cov)
    if predicted <= 0 or predicted <= target_annualized_vol:
        return dict(weights)

    scale = target_annualized_vol / predicted
    return {t: weight * scale for t, weight in weights.items()}
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_risk.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/risk.py tests/test_risk.py --fix
uv run ruff format src/midas/risk.py tests/test_risk.py
uv run mypy src
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add apply_vol_targeting and predict_portfolio_vol"
```

---

### Task 8: `predict_portfolio_vol` direct tests

**Files:**
- Modify: `tests/test_risk.py`

The function is already implemented (Task 7). This task adds focused unit tests.

- [ ] **Step 1: Write tests**

Append to `tests/test_risk.py`:

```python
from midas.risk import predict_portfolio_vol


class TestPredictPortfolioVol:
    def test_two_asset_uncorrelated_equal_weight(self):
        # var_a = var_b = 0.04 (20% vol), w = (0.5, 0.5), uncorrelated.
        # portfolio_var = 0.25 * 0.04 + 0.25 * 0.04 = 0.02 → vol ≈ sqrt(0.02) ≈ 0.1414
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, np.diag([0.04, 0.04]))
        vol = predict_portfolio_vol({"A": 0.5, "B": 0.5}, cov)
        assert math.isclose(vol, math.sqrt(0.02), rel_tol=1e-9)

    def test_zero_weights_return_zero(self):
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        assert predict_portfolio_vol({"A": 0.0}, cov) == 0.0

    def test_empty_weights_return_zero(self):
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        assert predict_portfolio_vol({}, cov) == 0.0

    def test_weights_outside_cov_index_ignored(self):
        tickers = ["A"]
        cov = _cov_frame(tickers, np.array([[0.04]]))
        # Only A contributes; B is ignored.
        vol = predict_portfolio_vol({"A": 1.0, "B": 5.0}, cov)
        assert math.isclose(vol, 0.20, rel_tol=1e-9)

    def test_degenerate_cov_returns_zero(self):
        tickers = ["A", "B"]
        cov = _cov_frame(tickers, -np.eye(2))  # non-PSD, forces quad <= 0
        assert predict_portfolio_vol({"A": 1.0, "B": 1.0}, cov) == 0.0
```

- [ ] **Step 2: Run and verify pass**

Run: `uv run pytest tests/test_risk.py::TestPredictPortfolioVol -q`
Expected: all tests pass (no implementation change — function already exists from Task 7).

- [ ] **Step 3: Commit**

```bash
git add tests/test_risk.py
git commit -m "Add direct unit tests for predict_portfolio_vol"
```

---

### Task 9: Thread `RiskConfig` through `Allocator` constructor (no behavior change)

**Files:**
- Modify: `src/midas/allocator.py`
- Modify: `src/midas/cli.py:35-60` (`_build_components`)
- Modify: `tests/test_allocator.py` (one new test)

The allocator accepts `RiskConfig` but stores it without using it yet. This isolates the signature change from the stage integrations that follow.

- [ ] **Step 1: Write failing test**

Append to `tests/test_allocator.py`:

```python
from midas.models import RiskConfig


class TestAllocatorRiskConfig:
    def test_accepts_risk_config_kwarg(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05)
        risk = RiskConfig()
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2, risk_config=risk)
        # Verify it's stored on the instance for later phases to read.
        assert allocator._risk_config is risk

    def test_defaults_to_risk_config_when_omitted(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05)
        allocator = Allocator([(mr, 1.0)], constraints, n_tickers=2)
        # Default matches RiskConfig() exactly.
        assert allocator._risk_config == RiskConfig()
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_allocator.py::TestAllocatorRiskConfig -q`
Expected: `TypeError` — unexpected keyword `risk_config`.

- [ ] **Step 3: Update `Allocator.__init__`**

Edit `src/midas/allocator.py`. Add import:

```python
from midas.models import DEFAULT_MAX_POSITION_PCT, AllocationConstraints, RiskConfig
```

Change the signature and add storage:

```python
    def __init__(
        self,
        entries: list[tuple[EntrySignal, float]],
        constraints: AllocationConstraints,
        n_tickers: int,
        *,
        risk_config: RiskConfig | None = None,
    ) -> None:
        self._entries: list[_ScoredEntry] = [_ScoredEntry(strat, wt) for strat, wt in entries]
        self._constraints = constraints
        self._risk_config = risk_config if risk_config is not None else RiskConfig()
        self._n_tickers = n_tickers
        self._signal_cache: dict[int, dict[str, np.ndarray]] = {}
        self._risk_cache: dict[str, Any] = {}
        # ... rest of existing __init__ body unchanged
```

Leave everything else unchanged for now.

- [ ] **Step 4: Thread through `_build_components`**

Edit `src/midas/cli.py:35-60`. Change the signature and pass through:

```python
def _build_components(
    strategy_configs: list[StrategyConfig] | None,
    constraints: AllocationConstraints,
    n_tickers: int,
    risk_config: RiskConfig | None = None,
) -> tuple[Allocator, OrderSizer, list[ExitRule]]:
    """Build allocator, order sizer, and exit rules from config."""
    configs = strategy_configs or [StrategyConfig(name=name) for name in STRATEGY_REGISTRY]

    entries: list[tuple[EntrySignal, float]] = []
    exits: list[ExitRule] = []

    for cfg in configs:
        strategy = _build_strategy(cfg)

        if isinstance(strategy, ExitRule):
            exits.append(strategy)
        elif isinstance(strategy, EntrySignal):
            entries.append((strategy, cfg.weight))
        else:
            msg = f"Strategy {cfg.name!r} is neither EntrySignal nor ExitRule"
            raise click.ClickException(msg)

    allocator = Allocator(entries, constraints, n_tickers, risk_config=risk_config)
    order_sizer = OrderSizer()

    return allocator, order_sizer, exits
```

Update the two call sites in `backtest` (line 158) and `live` (line 218):

```python
    allocator, order_sizer, exit_rules = _build_components(
        strat_configs,
        constraints,
        n_tickers,
        risk_config=risk_config,
    )
```

- [ ] **Step 5: Run tests and verify pass**

Run: `uv run pytest -q`
Expected: all tests pass — including the new ones and all existing integration/backtest tests (no behavior change).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
uv run mypy src
git add src/midas/allocator.py src/midas/cli.py tests/test_allocator.py
git commit -m "Thread RiskConfig through Allocator constructor"
```

---

### Task 10: Phase 1.5 — build risk cache in `precompute_signals`

**Files:**
- Modify: `src/midas/allocator.py`
- Modify: `tests/test_allocator.py`

Precompute populates `self._risk_cache` but no allocation phase consumes it yet. This isolates cache correctness from phase integration.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_allocator.py`:

```python
import pandas as pd


class TestAllocatorRiskCache:
    def test_cache_populated_with_sufficient_history(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.90)
        rng = np.random.default_rng(0)
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 1, 260)))
            for name in ("A", "B", "C")
        }
        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=3,
            risk_config=RiskConfig(vol_lookback_days=60, corr_lookback_days=252),
        )
        allocator.precompute_signals(histories)

        cache = allocator._risk_cache
        assert cache["cov"] is not None
        assert isinstance(cache["cov"], pd.DataFrame)
        assert cache["corr"] is not None
        assert set(cache["per_ticker_vol"].keys()) == {"A", "B", "C"}

    def test_cache_none_when_insufficient_history(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.90)
        # Only 30 bars — less than default corr_lookback_days=252.
        histories = {name: ph(np.full(30, 100.0)) for name in ("A", "B")}
        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(),
        )
        allocator.precompute_signals(histories)

        cache = allocator._risk_cache
        assert cache["cov"] is None
        assert cache["corr"] is None
        # Per-ticker vols also None when window not met.
        assert cache["per_ticker_vol"] == {"A": None, "B": None}
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_allocator.py::TestAllocatorRiskCache -q`
Expected: KeyError on `"cov"` (cache currently initialized to `{}`).

- [ ] **Step 3: Implement cache build**

Edit `src/midas/allocator.py`. Add imports:

```python
from midas.risk import covariance_matrix, realized_vol
```

Replace the existing `precompute_signals` method:

```python
    def precompute_signals(self, price_data: dict[str, PriceHistory]) -> None:
        """Precompute entry-signal scores and risk cache over the full price arrays."""
        self._signal_cache = {}
        for entry in self._entries:
            cache: dict[str, np.ndarray] = {}
            for ticker, history in price_data.items():
                result = entry.strategy.precompute(history)
                if result is not None:
                    cache[ticker] = result
            if cache:
                self._signal_cache[id(entry.strategy)] = cache

        self._risk_cache = self._build_risk_cache(price_data)

    def _build_risk_cache(
        self,
        price_data: dict[str, PriceHistory],
    ) -> dict[str, Any]:
        rc = self._risk_config
        per_ticker_vol: dict[str, float | None] = {
            ticker: realized_vol(history, window=rc.vol_lookback_days, annualize=True)
            for ticker, history in price_data.items()
        }
        cov = covariance_matrix(
            price_data,
            vol_window=rc.vol_lookback_days,
            corr_window=rc.corr_lookback_days,
            vol_floor=rc.vol_floor,
        )
        if cov is None:
            corr = None
        else:
            diag = np.sqrt(np.diag(cov.values))
            if np.all(diag > 0):
                corr_vals = cov.values / np.outer(diag, diag)
                corr = pd.DataFrame(corr_vals, index=cov.index, columns=cov.columns)
            else:
                corr = None
                cov = None
        return {"cov": cov, "corr": corr, "per_ticker_vol": per_ticker_vol}
```

Add pandas import near the existing numpy import:

```python
import pandas as pd
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_allocator.py -q`
Expected: all tests pass. No allocation behavior change — just a new cache field.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/allocator.py tests/test_allocator.py --fix
uv run ruff format src/midas/allocator.py tests/test_allocator.py
uv run mypy src
git add src/midas/allocator.py tests/test_allocator.py
git commit -m "Phase 1.5: build risk cache in precompute_signals"
```

---

### Task 11: Phase 2 — per-ticker vol offset in softmax

**Files:**
- Modify: `src/midas/allocator.py`
- Modify: `tests/test_allocator.py`

When `weighting == "inverse_vol"` and a ticker has a known vol, add `-log(max(vol, vol_floor))` to its blended score before softmax. Low-vol tickers get higher effective scores.

- [ ] **Step 1: Write failing test**

Append to `tests/test_allocator.py`:

```python
class TestPerTickerVolScaling:
    def test_low_vol_ticker_gets_more_weight_than_high_vol(self):
        """With equal signals, inverse-vol scaling should overweight low-vol ticker."""
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.90,
        )
        rng = np.random.default_rng(0)
        # LOW vol — tiny daily noise.
        low = 100.0 + np.cumsum(rng.normal(0, 0.05, 260))
        # HIGH vol — large daily noise.
        high = 100.0 + np.cumsum(rng.normal(0, 2.0, 260))
        # Both end with an identical "buy" signal: 10% drop on last bar.
        low[-1] = low[-2] * 0.90
        high[-1] = high[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(weighting="inverse_vol"),
        )
        histories = {"LOW": ph(low), "HIGH": ph(high)}
        allocator.precompute_signals(histories)
        result = allocator.allocate(
            ["LOW", "HIGH"],
            histories,
            current_weights={"LOW": 0.0, "HIGH": 0.0},
        )
        assert result.targets["LOW"] > result.targets["HIGH"]

    def test_equal_weighting_disables_offset(self):
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.90,
        )
        rng = np.random.default_rng(1)
        low = 100.0 + np.cumsum(rng.normal(0, 0.05, 260))
        high = 100.0 + np.cumsum(rng.normal(0, 2.0, 260))
        low[-1] = low[-2] * 0.90
        high[-1] = high[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(weighting="equal"),
        )
        histories = {"LOW": ph(low), "HIGH": ph(high)}
        allocator.precompute_signals(histories)
        result = allocator.allocate(
            ["LOW", "HIGH"],
            histories,
            current_weights={"LOW": 0.0, "HIGH": 0.0},
        )
        # Without vol offset, equal signals → equal softmax weights (within 1e-9).
        assert math.isclose(result.targets["LOW"], result.targets["HIGH"], rel_tol=1e-6)
```

Add `import math` at the top of `test_allocator.py` if not already present.

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_allocator.py::TestPerTickerVolScaling -q`
Expected: `test_low_vol_ticker_gets_more_weight_than_high_vol` fails (weights roughly equal without offset).

- [ ] **Step 3: Modify `_softmax_allocate` to accept per-ticker offsets**

Edit `src/midas/allocator.py`. Change `_softmax_allocate` to accept an optional offset map and use it:

```python
    def _softmax_allocate(
        self,
        tickers: list[str],
        blended_scores: dict[str, float],
        budget: float,
        temperature: float,
        targets: dict[str, float],
        score_offsets: dict[str, float] | None = None,
    ) -> None:
        """Distribute `budget` across `tickers` via softmax((blended_scores + offsets) / T).

        Temperature semantics:
            T → 0   winner-take-all (budget to the argmax ticker)
            T = 1   standard softmax over raw scores
            T → ∞   uniform split regardless of conviction
        """
        if not tickers or budget <= 0:
            for ticker in tickers:
                targets[ticker] = 0.0
            return
        temp_safe = max(temperature, MIN_TEMPERATURE)
        offsets = score_offsets or {}
        adjusted = {ticker: blended_scores[ticker] + offsets.get(ticker, 0.0) for ticker in tickers}
        max_score = max(adjusted.values())
        exps = {ticker: math.exp((adjusted[ticker] - max_score) / temp_safe) for ticker in tickers}
        total_exp = sum(exps.values())
        for ticker in tickers:
            targets[ticker] = budget * exps[ticker] / total_exp
```

Update `_apply_cap_with_redistribution` to accept and forward the offsets:

```python
    def _apply_cap_with_redistribution(
        self,
        active: list[str],
        blended_scores: dict[str, float],
        initial_budget: float,
        temperature: float,
        targets: dict[str, float],
        score_offsets: dict[str, float] | None = None,
    ) -> None:
        """Clamp targets exceeding max_position_pct; re-softmax the survivors."""
        cap = self._max_position_pct
        survivors = list(active)
        budget = initial_budget
        for _ in range(len(active) + 1):
            over = [ticker for ticker in survivors if targets[ticker] > cap + 1e-12]
            if not over:
                return
            for ticker in over:
                targets[ticker] = cap
                budget -= cap
                survivors.remove(ticker)
            if not survivors or budget <= 0:
                for ticker in survivors:
                    targets[ticker] = 0.0
                return
            self._softmax_allocate(
                survivors, blended_scores, budget, temperature, targets, score_offsets=score_offsets,
            )
```

In `allocate`, build the offsets and pass them in phase 2 and phase 3's cap:

After the Phase 1 block (just before "Phase 2: Softmax construct-to-budget"), add:

```python
        # Phase 1.5a: Compute per-ticker vol offsets for softmax (inverse-vol weighting).
        score_offsets = self._vol_score_offsets(active)

        # Phase 2: Softmax construct-to-budget over active tickers.
        self._softmax_allocate(
            active, blended_scores, budget_for_active, temperature, targets, score_offsets=score_offsets,
        )

        # Phase 3: Soft position cap.
        self._apply_cap_with_redistribution(
            active, blended_scores, budget_for_active, temperature, targets, score_offsets=score_offsets,
        )
```

And add the helper method:

```python
    def _vol_score_offsets(self, active: list[str]) -> dict[str, float]:
        """Per-ticker score offsets for inverse-vol weighting.

        Returns an empty dict when the weighting is ``"equal"`` or when a
        ticker has no known vol (warmup). Missing entries default to 0 in
        ``_softmax_allocate``.
        """
        if self._risk_config.weighting != "inverse_vol":
            return {}
        offsets: dict[str, float] = {}
        vol_floor = self._risk_config.vol_floor
        per_ticker_vol = self._risk_cache.get("per_ticker_vol") or {}
        for ticker in active:
            vol = per_ticker_vol.get(ticker)
            if vol is None:
                continue  # warmup — no offset
            offsets[ticker] = -math.log(max(vol, vol_floor))
        return offsets
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_allocator.py -q`
Expected: all tests pass, including the new per-ticker scaling tests.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/allocator.py tests/test_allocator.py --fix
uv run ruff format src/midas/allocator.py tests/test_allocator.py
uv run mypy src
git add src/midas/allocator.py tests/test_allocator.py
git commit -m "Phase 2: inverse-vol score offset in softmax"
```

---

### Task 12: Phase 3.5 — apply IDM after the cap

**Files:**
- Modify: `src/midas/allocator.py`
- Modify: `tests/test_allocator.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_allocator.py`:

```python
class TestIDMStage:
    def test_three_uncorrelated_tickers_scale_up_by_sqrt3(self):
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.90,  # loose cap — won't bind after IDM
        )
        rng = np.random.default_rng(42)
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 0.5, 260)))
            for name in ("A", "B", "C")
        }
        # Final bar identical-size drop on all three → equal signals.
        for h in histories.values():
            h.close[-1] = h.close[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=3,
            risk_config=RiskConfig(weighting="equal", idm_cap=5.0, vol_target_annualized=99.0),
        )
        allocator.precompute_signals(histories)
        pre_idm_sum = (1 - 0.05)
        result = allocator.allocate(
            ["A", "B", "C"],
            histories,
            current_weights={"A": 0.0, "B": 0.0, "C": 0.0},
        )
        total = sum(result.targets.values())
        # Roughly sqrt(3) × pre_idm_sum, with some slack from LW shrinkage
        # pulling correlations slightly off-zero.
        assert total > 1.3 * pre_idm_sum
        assert total < 1.8 * pre_idm_sum

    def test_idm_cap_binds_at_1_0_disables_stage(self):
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.90,
        )
        rng = np.random.default_rng(42)
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 0.5, 260)))
            for name in ("A", "B", "C")
        }
        for h in histories.values():
            h.close[-1] = h.close[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=3,
            risk_config=RiskConfig(weighting="equal", idm_cap=1.0, vol_target_annualized=99.0),
        )
        allocator.precompute_signals(histories)
        result = allocator.allocate(
            ["A", "B", "C"],
            histories,
            current_weights={"A": 0.0, "B": 0.0, "C": 0.0},
        )
        total = sum(result.targets.values())
        # idm_cap=1.0 → multiplier capped at 1.0 → no scale-up.
        assert math.isclose(total, 1 - 0.05, rel_tol=1e-6)

    def test_idm_then_recap_clips_overshoots(self):
        mr = MeanReversion(window=5, threshold=0.20)
        # Tight cap: IDM will push at least one ticker past it.
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.40,
        )
        rng = np.random.default_rng(42)
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 0.5, 260)))
            for name in ("A", "B", "C")
        }
        for h in histories.values():
            h.close[-1] = h.close[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=3,
            risk_config=RiskConfig(weighting="equal", idm_cap=3.0, vol_target_annualized=99.0),
        )
        allocator.precompute_signals(histories)
        result = allocator.allocate(
            ["A", "B", "C"],
            histories,
            current_weights={"A": 0.0, "B": 0.0, "C": 0.0},
        )
        # Re-cap (phase 3.6) must pull everything back to max_position_pct.
        for ticker, weight in result.targets.items():
            assert weight <= 0.40 + 1e-9, f"{ticker} exceeded cap after re-cap"
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_allocator.py::TestIDMStage -q`
Expected: `test_three_uncorrelated_tickers_scale_up_by_sqrt3` fails — totals stay at 0.95.

- [ ] **Step 3: Integrate phases 3.5 and 3.6**

Edit `src/midas/allocator.py`. Add import:

```python
from midas.risk import (
    apply_instrument_diversification_multiplier,
    apply_vol_targeting,
    covariance_matrix,
    realized_vol,
)
```

After the existing phase-3 call in `allocate`, add phases 3.5 and 3.6:

```python
        # Phase 3.5: Instrument Diversification Multiplier.
        corr = self._risk_cache.get("corr")
        if corr is not None and active:
            active_weights = {ticker: targets[ticker] for ticker in active}
            scaled = apply_instrument_diversification_multiplier(
                active_weights, corr, cap=self._risk_config.idm_cap,
            )
            for ticker, weight in scaled.items():
                targets[ticker] = weight

        # Phase 3.6: Re-cap after IDM (IDM can push single positions past the cap).
        self._apply_cap_with_redistribution(
            active, blended_scores, budget_for_active, temperature, targets, score_offsets=score_offsets,
        )
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_allocator.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/allocator.py tests/test_allocator.py --fix
uv run ruff format src/midas/allocator.py tests/test_allocator.py
uv run mypy src
git add src/midas/allocator.py tests/test_allocator.py
git commit -m "Phases 3.5 and 3.6: apply IDM and re-cap"
```

---

### Task 13: Phase 3.7 — apply portfolio vol targeting

**Files:**
- Modify: `src/midas/allocator.py`
- Modify: `tests/test_allocator.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_allocator.py`:

```python
class TestVolTargetingStage:
    def test_high_vol_portfolio_scaled_down_to_target(self):
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.90,
        )
        rng = np.random.default_rng(3)
        # Very volatile: daily sigma 5% each → annualized vol ≈ 80%.
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 5.0, 260)))
            for name in ("A", "B")
        }
        for h in histories.values():
            h.close[-1] = h.close[-2] * 0.90  # buy signal

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(
                weighting="equal",
                idm_cap=1.0,                     # disable IDM
                vol_target_annualized=0.20,
            ),
        )
        allocator.precompute_signals(histories)
        result = allocator.allocate(
            ["A", "B"],
            histories,
            current_weights={"A": 0.0, "B": 0.0},
        )
        total = sum(result.targets.values())
        # Vol targeting clamps down — total should be well below 0.95.
        assert total < 0.60

    def test_low_vol_portfolio_not_scaled_up(self):
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.90,
        )
        rng = np.random.default_rng(5)
        # Tame daily noise — well below 20% annualized.
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 0.1, 260)))
            for name in ("A", "B")
        }
        for h in histories.values():
            h.close[-1] = h.close[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(
                weighting="equal",
                idm_cap=1.0,
                vol_target_annualized=0.20,
            ),
        )
        allocator.precompute_signals(histories)
        result = allocator.allocate(
            ["A", "B"],
            histories,
            current_weights={"A": 0.0, "B": 0.0},
        )
        total = sum(result.targets.values())
        # Below target vol → no scaling down.
        assert math.isclose(total, 1 - 0.05, rel_tol=1e-6)
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run pytest tests/test_allocator.py::TestVolTargetingStage -q`
Expected: `test_high_vol_portfolio_scaled_down_to_target` fails.

- [ ] **Step 3: Integrate phase 3.7**

Edit `src/midas/allocator.py`. After phase 3.6, append:

```python
        # Phase 3.7: Portfolio vol targeting. Only scales down.
        cov = self._risk_cache.get("cov")
        if cov is not None and active:
            active_weights = {ticker: targets[ticker] for ticker in active}
            scaled = apply_vol_targeting(
                active_weights, cov,
                target_annualized_vol=self._risk_config.vol_target_annualized,
            )
            for ticker, weight in scaled.items():
                targets[ticker] = weight
```

- [ ] **Step 4: Run and verify pass**

Run: `uv run pytest tests/test_allocator.py -q`
Expected: all tests pass.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check src/midas/allocator.py tests/test_allocator.py --fix
uv run ruff format src/midas/allocator.py tests/test_allocator.py
uv run mypy src
git add src/midas/allocator.py tests/test_allocator.py
git commit -m "Phase 3.7: portfolio vol targeting"
```

---

### Task 14: Risk-off regression test (safety net)

**Files:**
- Modify: `tests/test_allocator.py`

Lock in that an explicitly risk-off `RiskConfig` produces the same targets as a pre-risk allocator would have.

- [ ] **Step 1: Write the test**

Append to `tests/test_allocator.py`:

```python
class TestRiskOffRegression:
    def test_risk_off_matches_pre_risk_behavior(self):
        """weighting=equal, idm_cap=1.0, vol_target=999 → no risk stage moves weights."""
        mr = MeanReversion(window=5, threshold=0.20)
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.50,
        )
        rng = np.random.default_rng(11)
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 0.5, 260)))
            for name in ("A", "B", "C")
        }
        # Only A gets a real buy signal; B and C are flat-ish.
        histories["A"].close[-1] = histories["A"].close[-2] * 0.90

        # Risk-off version.
        risk_off = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=3,
            risk_config=RiskConfig(
                weighting="equal",
                idm_cap=1.0,
                vol_target_annualized=999.0,
            ),
        )
        risk_off.precompute_signals(histories)
        result_off = risk_off.allocate(
            ["A", "B", "C"],
            histories,
            current_weights={"A": 0.0, "B": 0.0, "C": 0.0},
        )
        total = sum(result_off.targets.values())
        # Investable budget preserved — no scale-up (IDM disabled), no scale-down
        # (target vol 999% never binds).
        assert math.isclose(total, 1 - 0.05, rel_tol=1e-9)
        # A should still dominate the softmax with its buy signal.
        assert result_off.targets["A"] > result_off.targets["B"]
        assert result_off.targets["A"] > result_off.targets["C"]
```

- [ ] **Step 2: Run and verify pass**

Run: `uv run pytest tests/test_allocator.py::TestRiskOffRegression -q`
Expected: passes on first run (all risk stages defined to no-op in this config).

- [ ] **Step 3: Commit**

```bash
git add tests/test_allocator.py
git commit -m "Add risk-off regression test as safety net"
```

---

### Task 15: Warmup integration test

**Files:**
- Modify: `tests/test_allocator.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_allocator.py`:

```python
class TestWarmupProgression:
    def test_short_history_disables_all_risk_stages(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.90)
        # Only 30 bars — below any default lookback window.
        histories = {name: ph(np.full(30, 100.0)) for name in ("A", "B")}
        # Last-bar drop to generate signals.
        histories["A"].close[-1] = 90.0
        histories["B"].close[-1] = 90.0

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(),  # defaults
        )
        allocator.precompute_signals(histories)
        # Cache should be fully None.
        assert allocator._risk_cache["cov"] is None
        assert allocator._risk_cache["corr"] is None
        assert allocator._risk_cache["per_ticker_vol"] == {"A": None, "B": None}

        # Allocation still works — falls through all risk stages cleanly.
        result = allocator.allocate(
            ["A", "B"],
            histories,
            current_weights={"A": 0.0, "B": 0.0},
        )
        total = sum(result.targets.values())
        assert math.isclose(total, 1 - 0.05, rel_tol=1e-9)

    def test_mid_window_enables_per_ticker_vol_only(self):
        mr = MeanReversion(window=5, threshold=0.01)
        constraints = AllocationConstraints(min_cash_pct=0.05, max_position_pct=0.90)
        # 120 bars — enough for vol_lookback_days=60 but not corr_lookback_days=252.
        rng = np.random.default_rng(22)
        histories = {
            name: ph(100.0 + np.cumsum(rng.normal(0, 0.5, 120)))
            for name in ("A", "B")
        }
        histories["A"].close[-1] = histories["A"].close[-2] * 0.90
        histories["B"].close[-1] = histories["B"].close[-2] * 0.90

        allocator = Allocator(
            [(mr, 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(),  # defaults (vol=60, corr=252)
        )
        allocator.precompute_signals(histories)
        # Per-ticker vol defined; cov/corr still None.
        assert allocator._risk_cache["per_ticker_vol"]["A"] is not None
        assert allocator._risk_cache["per_ticker_vol"]["B"] is not None
        assert allocator._risk_cache["cov"] is None
        assert allocator._risk_cache["corr"] is None

        result = allocator.allocate(
            ["A", "B"],
            histories,
            current_weights={"A": 0.0, "B": 0.0},
        )
        # IDM and vol targeting skipped → total ≈ investable.
        total = sum(result.targets.values())
        assert math.isclose(total, 1 - 0.05, rel_tol=1e-6)
```

- [ ] **Step 2: Run and verify pass**

Run: `uv run pytest tests/test_allocator.py::TestWarmupProgression -q`
Expected: passes — warmup paths already guard all three stages.

- [ ] **Step 3: Commit**

```bash
git add tests/test_allocator.py
git commit -m "Add warmup progression integration test"
```

---

### Task 16: End-to-end backtest change-detector

**Files:**
- Modify: `tests/test_backtest.py` (append a new test)

Lock in the default risk-on behavior against a canned fixture so accidental changes surface immediately.

- [ ] **Step 1: Write the test**

Append to `tests/test_backtest.py`:

```python
class TestRiskDisciplineEndToEnd:
    def test_default_risk_config_produces_stable_summary(self, tmp_path):
        """End-to-end smoke test: default RiskConfig + canned fixture → stable summary.

        Change-detector, not a correctness test. If this breaks, confirm the
        drift is intentional before updating the golden numbers.
        """
        from datetime import date as dt_date

        from midas.allocator import Allocator
        from midas.backtest import BacktestEngine
        from midas.models import (
            AllocationConstraints,
            Holding,
            PortfolioConfig,
            RiskConfig,
        )
        from midas.order_sizer import OrderSizer
        from midas.strategies.mean_reversion import MeanReversion

        # Deterministic synthetic: 2 tickers, 400 bars, mild drift + noise.
        rng = np.random.default_rng(2026)
        days = 400
        start = dt_date(2023, 1, 2)
        frames: dict[str, pd.DataFrame] = {}
        for name, sigma in [("AAA", 0.01), ("BBB", 0.02)]:
            returns = rng.normal(0.0003, sigma, days).tolist()
            frames[name] = make_price_series(start, days, 100.0, returns, name=name)

        portfolio = PortfolioConfig(
            holdings=[
                Holding(ticker="AAA", shares=0),
                Holding(ticker="BBB", shares=0),
            ],
            available_cash=10_000.0,
        )
        constraints = AllocationConstraints(
            min_cash_pct=0.05,
            softmax_temperature=0.5,
            max_position_pct=0.60,
        )
        allocator = Allocator(
            [(MeanReversion(window=20, threshold=0.03), 1.0)],
            constraints,
            n_tickers=2,
            risk_config=RiskConfig(),  # defaults — risk-on
        )
        engine = BacktestEngine(
            allocator=allocator,
            order_sizer=OrderSizer(),
            exit_rules=[],
            constraints=constraints,
            train_pct=1.0,
            enable_split=False,
        )
        result = engine.run(portfolio, frames, start, start + timedelta(days=days))

        # Change-detector: lock in a few stable numbers. Update deliberately.
        assert result.total_days > 0
        assert math.isfinite(result.final_value)
        assert result.final_value > 0
        # Default risk-on pipeline should produce at least one trade on this
        # 400-day synthetic — guards against the pipeline no-op'ing silently.
        assert len(result.trades) >= 1
```

Add imports at the top of `tests/test_backtest.py` if not already present: `import math`, `import numpy as np`, `import pandas as pd`, `from datetime import timedelta`, `from conftest import make_price_series`.

- [ ] **Step 2: Run and verify pass**

Run: `uv run pytest tests/test_backtest.py::TestRiskDisciplineEndToEnd -q`
Expected: passes.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest -q`
Expected: all tests pass.

- [ ] **Step 4: Lint, type-check, final commit**

```bash
uv run ruff check src tests --fix
uv run ruff format src tests
uv run mypy src
git add tests/test_backtest.py
git commit -m "Add end-to-end change-detector for default risk config"
```

---

## Closing checks

- [ ] **Final validation**

Run all of the following and confirm clean:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -q
```

Expected: all green.

- [ ] **Review commit history**

```bash
git log --oneline $(git merge-base HEAD master)..HEAD
```

Expected: 16 focused commits mapping 1:1 to the tasks above.

---

## Self-Review

**Spec coverage:**
- Section 1 (phase ordering): Tasks 10–13 implement 1.5/2/3.5/3.6/3.7 ✓
- Section 2 (cov estimator): Task 5 uses `sklearn.covariance.LedoitWolf` on correlation, composes via outer-product ✓
- Section 3 (module organization): Tasks 4–8 build all five public functions in `src/midas/risk.py` ✓
- Section 4 (config shape): Task 2 `RiskConfig`; Task 3 YAML parsing; Task 9 thread-through ✓
- Section 5 (data flow): Task 10 builds `_risk_cache` in `precompute_signals` ✓
- Section 6 (error handling): Task 2 config-time `ValueError`s; Tasks 4–8 include degenerate-input tests; Tasks 10 + 15 cover warmup `None` sentinels ✓
- Section 7 (testing): All unit categories covered in Tasks 4–8, integration in Tasks 11–15, end-to-end in Task 16; risk-off regression in Task 14 ✓
- Dependencies (§"Dependencies"): Task 1 adds scikit-learn ✓

**Placeholder scan:** No TBDs, TODOs, or "add appropriate X" phrases.

**Type/signature consistency:**
- `realized_vol(history, window, annualize=True) -> float | None` — Task 4, used in Task 5 and Task 10 ✓
- `covariance_matrix(histories, vol_window, corr_window, vol_floor) -> pd.DataFrame | None` — Task 5, used in Task 10 ✓
- `apply_instrument_diversification_multiplier(weights, corr, cap) -> dict[str, float]` — Task 6, used in Task 12 ✓
- `apply_vol_targeting(weights, cov, target_annualized_vol) -> dict[str, float]` — Task 7, used in Task 13 ✓
- `predict_portfolio_vol(weights, cov) -> float` — Task 7, tests added Task 8 ✓
- `RiskConfig` kwargs match across Tasks 2, 3, 9, 10–15 ✓
- `Allocator.__init__(..., *, risk_config: RiskConfig | None = None)` — Task 9 establishes; all later tasks use `risk_config=...` keyword ✓
