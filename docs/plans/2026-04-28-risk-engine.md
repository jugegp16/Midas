# Risk Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPPI drawdown overlay, per-ticker inverse-vol scaling, portfolio vol target, and read-only telemetry to the Midas allocator. No solver dep, no MVO. Risk policy lives in YAML; the optimizer doesn't search it.

**Architecture:** Pure-function `risk` module (sliced history per bar, no caches). New `RiskConfig` threaded through `Allocator`. Phase 0 (CPPI) before softmax; modified Phase 2 (inverse-vol score offset *outside* the `/T` divider — fixes PR #63's `(1/vol)^(1/T)` regression); Phase 4 (vol target) after cap. Backtest tracks running peak. Live mode warns CPPI is inert (peak persistence is v2). Read-only `RiskMetrics` surfaced through existing output layer.

**Tech Stack:** Python 3.14, numpy, scikit-learn (`LedoitWolf`), pytest, optuna.

**Spec:** `docs/specs/2026-04-27-risk-engine-design.md`

---

## File Structure

**New files:**

- `src/midas/risk.py` — pure functions: `realized_vol`, `covariance_matrix`, `predict_portfolio_vol`, `apply_drawdown_overlay`, `inverse_vol_offset`. No state, no caching. Inputs are sliced numpy arrays.
- `src/midas/risk_metrics.py` — `RiskMetrics` dataclass + `compute_risk_metrics(equity_curve, returns_per_strategy)`.
- `tests/test_risk.py` — unit tests for `risk.py` functions.
- `tests/test_risk_metrics.py` — unit tests for telemetry math.
- `tests/test_allocator_risk.py` — allocator integration (CPPI, inverse-vol, vol target, T-independence, composition).
- `tests/test_lookahead_regression.py` — sliced-vs-full backtest equality at bar T.
- `tests/test_optimizer_risk_propagation.py` — `risk:` survives the trial loop.

**Modified files:**

- `src/midas/models.py` — add `RiskConfig` and risk default constants.
- `src/midas/config.py` — parse `risk:` block; `load_strategies` returns 3-tuple.
- `src/midas/allocator.py` — accept `RiskConfig`, optional `current_drawdown`; new Phase 0 and Phase 4; inverse-vol offset in softmax.
- `src/midas/cli.py` — pass `risk_config` to `Allocator`.
- `src/midas/optimizer.py` — pass `risk_config` to `Allocator`; expose for trial-loop test.
- `src/midas/backtest.py` — add `peak_value` to `_SimState`, compute `current_drawdown`, populate `RiskMetrics`, per-strategy attribution dict per position.
- `src/midas/live.py` — warn at startup if CPPI configured; populate `RiskMetrics` per tick.
- `src/midas/results.py` and `src/midas/output.py` — surface `RiskMetrics` in summary.
- `docs/architecture.md` — document Phase 0, Phase 4, inverse-vol offset, telemetry.
- `docs/strategies.md` — document `risk:` YAML block.
- `pyproject.toml` — add `scikit-learn`.

---

## Task 1: Add scikit-learn dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add the dep**

Edit `pyproject.toml` `[project] dependencies` list, alphabetical order kept:

```toml
dependencies = [
    "click>=8.1",
    "optuna>=4.0",
    "pandas>=2.2",
    "pyyaml>=6.0",
    "rich>=13.0",
    "scikit-learn>=1.5",
    "yfinance>=0.2",
]
```

- [ ] **Step 2: Sync and verify**

Run: `uv sync`
Run: `uv run python -c "from sklearn.covariance import LedoitWolf; print(LedoitWolf().__class__.__name__)"`
Expected: prints `LedoitWolf` with no errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add scikit-learn dep for Ledoit-Wolf covariance shrinkage (#64)"
```

---

## Task 2: Add RiskConfig dataclass

**Files:**
- Modify: `src/midas/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_models.py`:

```python
import pytest

from midas.models import RiskConfig


class TestRiskConfig:
    def test_defaults_disable_everything(self) -> None:
        cfg = RiskConfig()
        assert cfg.weighting == "equal"
        assert cfg.vol_lookback_days == 60
        assert cfg.vol_target is None
        assert cfg.drawdown_penalty is None
        assert cfg.drawdown_floor is None

    def test_drawdown_both_or_neither_neither(self) -> None:
        RiskConfig(drawdown_penalty=None, drawdown_floor=None)  # ok

    def test_drawdown_both_or_neither_both(self) -> None:
        RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)  # ok

    def test_drawdown_penalty_without_floor_raises(self) -> None:
        with pytest.raises(ValueError, match="drawdown_floor"):
            RiskConfig(drawdown_penalty=1.5)

    def test_drawdown_floor_without_penalty_raises(self) -> None:
        with pytest.raises(ValueError, match="drawdown_penalty"):
            RiskConfig(drawdown_floor=0.5)

    def test_weighting_validation(self) -> None:
        with pytest.raises(ValueError, match="weighting"):
            RiskConfig(weighting="bogus")
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_models.py::TestRiskConfig -v`
Expected: ImportError or NameError on `RiskConfig`.

- [ ] **Step 3: Implement RiskConfig**

Edit `src/midas/models.py`. Add constants near the existing defaults block (after `DEFAULT_ENTRY_WEIGHT`):

```python
DEFAULT_VOL_LOOKBACK_DAYS = 60
DEFAULT_VOL_FLOOR = 0.005
WEIGHTING_OPTIONS = frozenset({"equal", "inverse_vol"})
```

Append the dataclass at the end of the file:

```python
@dataclass(frozen=True)
class RiskConfig:
    """Optional risk-discipline policy. Defaults reduce the engine to current behavior.

    weighting:        "equal" (current softmax) or "inverse_vol" (score offset of -log(vol)).
    vol_lookback_days: rolling window for vol and covariance estimates.
    vol_target:       annualized portfolio vol cap; None disables Phase 4 vol scaling.
    drawdown_penalty/floor: CPPI overlay; both required, both must be set or both None.
    """

    weighting: str = "equal"
    vol_lookback_days: int = DEFAULT_VOL_LOOKBACK_DAYS
    vol_target: float | None = None
    drawdown_penalty: float | None = None
    drawdown_floor: float | None = None

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTING_OPTIONS:
            msg = f"weighting must be one of {sorted(WEIGHTING_OPTIONS)}, got {self.weighting!r}"
            raise ValueError(msg)
        if (self.drawdown_penalty is None) != (self.drawdown_floor is None):
            missing = "drawdown_floor" if self.drawdown_penalty is not None else "drawdown_penalty"
            msg = f"drawdown_penalty and drawdown_floor must both be set or both omitted; missing {missing}"
            raise ValueError(msg)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_models.py::TestRiskConfig -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/models.py tests/test_models.py
git commit -m "Add RiskConfig dataclass with both-or-neither CPPI validation (#64)"
```

---

## Task 3: Parse `risk:` block from YAML

**Files:**
- Modify: `src/midas/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config.py`:

```python
import textwrap

from midas.config import load_strategies
from midas.models import RiskConfig


def _write(tmp_path, body: str):
    path = tmp_path / "s.yaml"
    path.write_text(textwrap.dedent(body))
    return path


class TestLoadStrategiesRisk:
    def test_no_risk_block_returns_default(self, tmp_path) -> None:
        path = _write(tmp_path, """
            strategies:
              - name: BollingerBand
                params: {window: 20}
        """)
        _configs, _constraints, risk = load_strategies(path)
        assert risk == RiskConfig()

    def test_full_risk_block(self, tmp_path) -> None:
        path = _write(tmp_path, """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              weighting: inverse_vol
              vol_lookback_days: 90
              vol_target: 0.20
              drawdown_penalty: 1.5
              drawdown_floor: 0.5
        """)
        _, _, risk = load_strategies(path)
        assert risk.weighting == "inverse_vol"
        assert risk.vol_lookback_days == 90
        assert risk.vol_target == 0.20
        assert risk.drawdown_penalty == 1.5
        assert risk.drawdown_floor == 0.5

    def test_partial_risk_block_only_vol_target(self, tmp_path) -> None:
        path = _write(tmp_path, """
            strategies:
              - name: BollingerBand
                params: {window: 20}
            risk:
              vol_target: 0.18
        """)
        _, _, risk = load_strategies(path)
        assert risk.vol_target == 0.18
        assert risk.weighting == "equal"
        assert risk.drawdown_penalty is None
        assert risk.drawdown_floor is None
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_config.py::TestLoadStrategiesRisk -v`
Expected: TypeError or ValueError because `load_strategies` still returns a 2-tuple.

- [ ] **Step 3: Update `load_strategies`**

Edit `src/midas/config.py`. Add `RiskConfig` to the imports:

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

Change the function signature and body. Replace the existing `load_strategies` (lines 77-110) with:

```python
def load_strategies(
    path: Path,
) -> tuple[list[StrategyConfig], AllocationConstraints, RiskConfig]:
    """Load strategy configs, allocation knobs, and optional risk policy from YAML.

    Returns (strategies, constraints, risk_config). The risk block is optional —
    omitting it yields a default ``RiskConfig`` (all features off, current behavior).
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

    risk_raw = raw.get("risk") or {}
    risk = RiskConfig(
        weighting=str(risk_raw.get("weighting", "equal")),
        vol_lookback_days=int(risk_raw.get("vol_lookback_days", 60)),
        vol_target=float(risk_raw["vol_target"]) if risk_raw.get("vol_target") is not None else None,
        drawdown_penalty=(
            float(risk_raw["drawdown_penalty"]) if risk_raw.get("drawdown_penalty") is not None else None
        ),
        drawdown_floor=(
            float(risk_raw["drawdown_floor"]) if risk_raw.get("drawdown_floor") is not None else None
        ),
    )

    return configs, constraints, risk
```

- [ ] **Step 4: Update existing call sites**

Search for `load_strategies(` and update unpacking. Run:

```bash
grep -rn "load_strategies(" src/ tests/
```

Expected matches: `src/midas/cli.py`, `src/midas/optimizer.py`, possibly tests. For each, change `configs, constraints = load_strategies(path)` to `configs, constraints, risk_config = load_strategies(path)`.

After each call site update, store `risk_config` in the local scope (it'll be used by Task 9). For now in `cli.py` and `optimizer.py`, add a stub variable next to the existing unpack — the wiring comes in Task 9.

- [ ] **Step 5: Run all tests, verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: existing config tests still pass; new TestLoadStrategiesRisk tests pass.

Run: `uv run pytest`
Expected: all existing tests pass (call site updates didn't break anything).

- [ ] **Step 6: Commit**

```bash
git add src/midas/config.py src/midas/cli.py src/midas/optimizer.py tests/test_config.py
git commit -m "Parse optional risk: block in load_strategies (#64)"
```

---

## Task 4: `realized_vol` pure function

**Files:**
- Create: `src/midas/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_risk.py`:

```python
"""Unit tests for the pure risk module. No allocator involvement here."""

import math

import numpy as np
import pytest

from midas.risk import realized_vol


class TestRealizedVol:
    def test_constant_prices_yields_zero_vol(self) -> None:
        prices = np.full(100, 100.0)
        assert realized_vol(prices, lookback=60) == 0.0

    def test_known_synthetic_series(self) -> None:
        # Build a series with known daily log-return stdev = 0.01.
        np.random.seed(0)
        daily_log_returns = np.random.normal(loc=0.0, scale=0.01, size=10_000)
        prices = 100.0 * np.exp(np.cumsum(daily_log_returns))
        # Annualized: 0.01 * sqrt(252) ≈ 0.1587
        vol = realized_vol(prices, lookback=10_000)
        assert math.isclose(vol, 0.01 * math.sqrt(252), rel_tol=0.05)

    def test_lookback_truncates(self) -> None:
        prices = np.array([100.0, 101.0, 100.0, 99.0, 100.0, 101.0, 102.0, 103.0])
        # With lookback=3, only the last 3 returns participate.
        vol_3 = realized_vol(prices, lookback=3)
        vol_full = realized_vol(prices, lookback=len(prices))
        assert vol_3 != vol_full

    def test_insufficient_history_returns_zero(self) -> None:
        prices = np.array([100.0, 101.0])
        # Lookback bigger than available history → 0 (caller must check).
        assert realized_vol(prices, lookback=60) == 0.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_risk.py::TestRealizedVol -v`
Expected: ImportError on `midas.risk`.

- [ ] **Step 3: Implement**

Create `src/midas/risk.py`:

```python
"""Pure risk-discipline functions. No state, no caches.

Inputs are sliced numpy arrays sized at the caller's discretion. Each function
must produce identical output regardless of whether the caller passes a slice
of a larger array or a fresh array — this is the no-lookahead invariant.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252


def realized_vol(prices: np.ndarray, lookback: int) -> float:
    """Annualized stdev of close-to-close log returns over the last ``lookback`` bars.

    Returns 0.0 if fewer than ``lookback + 1`` prices are available, or if the
    log-return series is constant (zero stdev). Callers must treat 0.0 as
    "insufficient signal" and fall back accordingly.
    """
    if prices.size < lookback + 1:
        return 0.0
    window = prices[-(lookback + 1):]
    log_returns = np.diff(np.log(window))
    return float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_risk.py::TestRealizedVol -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add realized_vol to risk module (#64)"
```

---

## Task 5: `covariance_matrix` (Ledoit-Wolf)

**Files:**
- Modify: `src/midas/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_risk.py`:

```python
from midas.risk import covariance_matrix


class TestCovarianceMatrix:
    def test_shape_and_symmetry(self) -> None:
        np.random.seed(0)
        # 3 tickers, 252 bars
        log_returns = np.random.normal(0, 0.01, size=(252, 3))
        cov = covariance_matrix(log_returns)
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_positive_semidefinite(self) -> None:
        np.random.seed(0)
        log_returns = np.random.normal(0, 0.01, size=(252, 4))
        cov = covariance_matrix(log_returns)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert (eigenvalues >= -1e-12).all()

    def test_independent_series_diagonal_dominant(self) -> None:
        np.random.seed(0)
        # Three uncorrelated series with very different vols.
        n = 5_000
        log_returns = np.column_stack([
            np.random.normal(0, 0.005, n),
            np.random.normal(0, 0.01, n),
            np.random.normal(0, 0.02, n),
        ])
        cov = covariance_matrix(log_returns)
        # Diagonal should approximate the per-series variance; off-diagonal small.
        assert cov[0, 0] < cov[1, 1] < cov[2, 2]
        assert abs(cov[0, 1]) < cov[0, 0]
        assert abs(cov[0, 2]) < cov[0, 0]
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_risk.py::TestCovarianceMatrix -v`
Expected: ImportError on `covariance_matrix`.

- [ ] **Step 3: Implement**

Append to `src/midas/risk.py`:

```python
from sklearn.covariance import LedoitWolf


def covariance_matrix(log_returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf-shrunk covariance of daily log returns.

    Args:
        log_returns: shape (n_bars, n_tickers). Each column is one ticker's
            daily log-return series over a common window.

    Returns:
        (n_tickers, n_tickers) shrunk covariance matrix in daily units.
        Annualize at the call site if needed.
    """
    estimator = LedoitWolf().fit(log_returns)
    return np.asarray(estimator.covariance_)
```

Move the import to the top of the file with the other imports.

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_risk.py::TestCovarianceMatrix -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add covariance_matrix with Ledoit-Wolf shrinkage (#64)"
```

---

## Task 6: `predict_portfolio_vol`

**Files:**
- Modify: `src/midas/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_risk.py`:

```python
from midas.risk import predict_portfolio_vol


class TestPredictPortfolioVol:
    def test_uncorrelated_equal_weight_diversification(self) -> None:
        # Three independent series, daily stdev 0.01 each → annualized ≈ 0.1587.
        # Equal weight across 3 → predicted ≈ 0.1587 / sqrt(3) ≈ 0.0916.
        n = 20_000
        np.random.seed(0)
        log_returns = np.column_stack([
            np.random.normal(0, 0.01, n),
            np.random.normal(0, 0.01, n),
            np.random.normal(0, 0.01, n),
        ])
        weights = np.array([1 / 3, 1 / 3, 1 / 3])
        predicted = predict_portfolio_vol(weights, log_returns)
        assert math.isclose(predicted, 0.01 * math.sqrt(252) / math.sqrt(3), rel_tol=0.10)

    def test_perfectly_correlated_no_diversification(self) -> None:
        np.random.seed(0)
        # Two identical series.
        x = np.random.normal(0, 0.01, 5_000)
        log_returns = np.column_stack([x, x])
        weights = np.array([0.5, 0.5])
        predicted = predict_portfolio_vol(weights, log_returns)
        # Same as either ticker's vol — no diversification.
        assert math.isclose(predicted, 0.01 * math.sqrt(252), rel_tol=0.05)

    def test_zero_weights_yields_zero(self) -> None:
        np.random.seed(0)
        log_returns = np.random.normal(0, 0.01, size=(100, 3))
        weights = np.array([0.0, 0.0, 0.0])
        assert predict_portfolio_vol(weights, log_returns) == 0.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_risk.py::TestPredictPortfolioVol -v`
Expected: ImportError on `predict_portfolio_vol`.

- [ ] **Step 3: Implement**

Append to `src/midas/risk.py`:

```python
def predict_portfolio_vol(weights: np.ndarray, log_returns: np.ndarray) -> float:
    """Annualized predicted portfolio volatility from weights and log-return history.

    Args:
        weights: shape (n_tickers,). Same column order as ``log_returns``.
        log_returns: shape (n_bars, n_tickers). Daily log returns over a common window.

    Returns:
        Annualized stdev of the portfolio's daily returns under the given weights.
    """
    cov = covariance_matrix(log_returns)
    daily_var = float(weights @ cov @ weights)
    if daily_var <= 0:
        return 0.0
    return float(np.sqrt(daily_var) * np.sqrt(TRADING_DAYS_PER_YEAR))
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_risk.py::TestPredictPortfolioVol -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add predict_portfolio_vol to risk module (#64)"
```

---

## Task 7: `apply_drawdown_overlay` (CPPI)

**Files:**
- Modify: `src/midas/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_risk.py`:

```python
from midas.risk import apply_drawdown_overlay


class TestApplyDrawdownOverlay:
    def test_no_drawdown_no_change(self) -> None:
        scale = apply_drawdown_overlay(current_drawdown=0.0, penalty=1.5, floor=0.5)
        assert scale == 1.0

    def test_moderate_drawdown(self) -> None:
        # 20% DD * 1.5 penalty = 0.30 reduction → 0.70 exposure.
        scale = apply_drawdown_overlay(current_drawdown=0.20, penalty=1.5, floor=0.5)
        assert math.isclose(scale, 0.70)

    def test_floor_binds_at_deep_drawdown(self) -> None:
        # 50% DD * 1.5 penalty = 0.75 reduction → 0.25 raw, but floor=0.5.
        scale = apply_drawdown_overlay(current_drawdown=0.50, penalty=1.5, floor=0.5)
        assert scale == 0.5

    def test_penalty_zero_disables(self) -> None:
        scale = apply_drawdown_overlay(current_drawdown=0.30, penalty=0.0, floor=0.5)
        assert scale == 1.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_risk.py::TestApplyDrawdownOverlay -v`
Expected: ImportError on `apply_drawdown_overlay`.

- [ ] **Step 3: Implement**

Append to `src/midas/risk.py`:

```python
def apply_drawdown_overlay(current_drawdown: float, penalty: float, floor: float) -> float:
    """CPPI-style exposure scaler: max(1 - penalty * dd, floor).

    Args:
        current_drawdown: positive fraction (e.g. 0.20 for 20% drawdown from peak).
        penalty: how aggressively to de-risk per unit of drawdown.
        floor: minimum exposure scale; never reduce below this.

    Returns:
        Scalar in [floor, 1.0] to multiply the gross investable budget by.
    """
    return max(1.0 - penalty * current_drawdown, floor)
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_risk.py::TestApplyDrawdownOverlay -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add CPPI-style apply_drawdown_overlay (#64)"
```

---

## Task 8: `inverse_vol_offset`

**Files:**
- Modify: `src/midas/risk.py`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_risk.py`:

```python
from midas.models import DEFAULT_VOL_FLOOR
from midas.risk import inverse_vol_offset


class TestInverseVolOffset:
    def test_offset_is_negative_log_vol(self) -> None:
        assert math.isclose(inverse_vol_offset(0.20, vol_floor=DEFAULT_VOL_FLOOR), -math.log(0.20))
        assert math.isclose(inverse_vol_offset(0.50, vol_floor=DEFAULT_VOL_FLOOR), -math.log(0.50))

    def test_floor_clamps_low_vols(self) -> None:
        # Vol below floor uses floor.
        assert math.isclose(
            inverse_vol_offset(0.001, vol_floor=DEFAULT_VOL_FLOOR),
            -math.log(DEFAULT_VOL_FLOOR),
        )

    def test_zero_vol_returns_nan(self) -> None:
        # Caller must check; zero indicates insufficient signal.
        result = inverse_vol_offset(0.0, vol_floor=DEFAULT_VOL_FLOOR)
        assert math.isnan(result)
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_risk.py::TestInverseVolOffset -v`
Expected: ImportError on `inverse_vol_offset`.

- [ ] **Step 3: Implement**

Append to `src/midas/risk.py`:

```python
def inverse_vol_offset(vol: float, vol_floor: float) -> float:
    """Score offset for inverse-vol weighting: ``-log(max(vol, vol_floor))``.

    Returns NaN when ``vol == 0``: the caller must treat this as "insufficient
    signal" and fall back (typically Option A: hold current weight). The floor
    handles low-but-nonzero vols; literal zero indicates no valid signal.
    """
    if vol == 0.0:
        return float("nan")
    return -float(np.log(max(vol, vol_floor)))
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_risk.py::TestInverseVolOffset -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/risk.py tests/test_risk.py
git commit -m "Add inverse_vol_offset with floor + NaN signal (#64)"
```

---

## Task 9: Thread `RiskConfig` through `Allocator`

**Files:**
- Modify: `src/midas/allocator.py`
- Modify: `src/midas/cli.py`
- Modify: `src/midas/optimizer.py`
- Test: `tests/test_allocator.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_allocator.py`:

```python
from midas.models import RiskConfig


def test_allocator_accepts_risk_config(make_allocator) -> None:
    alloc = make_allocator(risk_config=RiskConfig())
    assert alloc.risk_config == RiskConfig()


def test_allocator_default_risk_config_is_none_safe(make_allocator) -> None:
    alloc = make_allocator()  # no risk_config kwarg
    # Either None or default RiskConfig acceptable; allocate() must not crash.
    assert alloc.risk_config is None or isinstance(alloc.risk_config, RiskConfig)
```

If `make_allocator` doesn't exist as a fixture, add a minimal fixture to `tests/conftest.py` that constructs `Allocator` with sensible defaults — check existing tests for the construction pattern and lift it.

Also add a regression test:

```python
def test_default_risk_matches_legacy_behavior(allocator_two_tickers, simple_price_data) -> None:
    """No risk_config (or default) must produce identical output to the legacy allocator."""
    legacy = allocator_two_tickers
    risk_off = legacy.__class__(  # same construction, with risk_config=None or RiskConfig()
        legacy._entries[0:1],  # adapt to existing test pattern
        legacy._constraints,
        n_tickers=2,
        risk_config=None,
    )
    r1 = legacy.allocate(["A", "B"], simple_price_data)
    r2 = risk_off.allocate(["A", "B"], simple_price_data)
    assert r1.targets == r2.targets
```

(Adjust to match the existing `tests/test_allocator.py` style if it diverges.)

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_allocator.py -k risk -v`
Expected: TypeError on the new `risk_config=` kwarg.

- [ ] **Step 3: Implement constructor change**

Edit `src/midas/allocator.py`. Add to imports:

```python
from midas.models import DEFAULT_MAX_POSITION_PCT, AllocationConstraints, RiskConfig
```

Update `Allocator.__init__` (around line 59):

```python
def __init__(
    self,
    entries: list[tuple[EntrySignal, float]],
    constraints: AllocationConstraints,
    n_tickers: int,
    risk_config: RiskConfig | None = None,
) -> None:
    self._entries: list[_ScoredEntry] = [_ScoredEntry(strat, wt) for strat, wt in entries]
    self._constraints = constraints
    self._n_tickers = n_tickers
    self._risk_config = risk_config
    self._signal_cache: dict[int, dict[str, np.ndarray]] = {}
    # ... rest of existing __init__ unchanged ...
```

Add a property:

```python
@property
def risk_config(self) -> RiskConfig | None:
    return self._risk_config
```

Update `allocate()` signature to accept `current_drawdown`:

```python
def allocate(
    self,
    tickers: list[str],
    price_data: dict[str, PriceHistory],
    context: dict[str, dict[str, Any]] | None = None,
    current_weights: dict[str, float] | None = None,
    current_drawdown: float = 0.0,
) -> AllocationResult:
```

(Body unchanged for now — Phase 0 wiring lands in Task 10.)

- [ ] **Step 4: Update call sites**

`src/midas/cli.py:57`:

```python
allocator = Allocator(entries, constraints, n_tickers, risk_config=risk_config)
```

(`risk_config` came from `load_strategies` in Task 3.)

`src/midas/optimizer.py:276`:

```python
allocator = Allocator(entries, constraints, n_tickers, risk_config=risk_config)
```

(Pass through from the trial-level YAML load. The full propagation regression is in Task 18.)

- [ ] **Step 5: Run all tests, verify pass**

Run: `uv run pytest`
Expected: existing tests pass; new risk-related allocator tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/midas/allocator.py src/midas/cli.py src/midas/optimizer.py tests/test_allocator.py tests/conftest.py
git commit -m "Thread RiskConfig through Allocator constructor (#64)"
```

---

## Task 10: Phase 0 — CPPI overlay in `allocate()`

**Files:**
- Modify: `src/midas/allocator.py`
- Test: `tests/test_allocator_risk.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_allocator_risk.py`:

```python
"""Allocator integration tests for the risk engine (CPPI, inverse-vol, vol target)."""

from __future__ import annotations

import math

import numpy as np

from midas.allocator import Allocator
from midas.data.price_history import PriceHistory
from midas.models import AllocationConstraints, RiskConfig


def _flat_history(n_bars: int, price: float = 100.0) -> PriceHistory:
    arr = np.full(n_bars, price)
    return PriceHistory(open=arr, high=arr, low=arr, close=arr, volume=arr)


def _diverging_history(n_bars: int, daily_vol: float, seed: int) -> PriceHistory:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0.0, daily_vol, n_bars)
    closes = 100.0 * np.exp(np.cumsum(log_returns))
    return PriceHistory(open=closes, high=closes, low=closes, close=closes, volume=closes)


class _ScoreOnce:
    """Test-only entry signal returning a fixed score per ticker."""

    name = "ScoreOnce"
    warmup_period = 1
    suitability = None
    description = ""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores

    def score(self, history, **ticker_ctx) -> float | None:
        ticker = ticker_ctx.get("ticker")
        return self._scores.get(ticker)

    def precompute(self, history) -> np.ndarray | None:
        return None


def test_phase_0_no_drawdown_matches_no_risk(make_allocator):
    """With current_drawdown=0, CPPI is a no-op."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = make_allocator(risk_config=risk)
    tickers = ["A", "B"]
    prices = {t: _flat_history(100) for t in tickers}
    result = alloc.allocate(tickers, prices, current_drawdown=0.0)
    # Sum of targets should equal investable budget (1 - min_cash_pct).
    investable = 1.0 - alloc._constraints.min_cash_pct
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-9)


def test_phase_0_drawdown_shrinks_investable(make_allocator):
    """20% DD with penalty 1.5 → exposure scaled to 0.70."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = make_allocator(risk_config=risk)
    tickers = ["A", "B"]
    prices = {t: _flat_history(100) for t in tickers}
    result = alloc.allocate(tickers, prices, current_drawdown=0.20)
    investable = (1.0 - alloc._constraints.min_cash_pct) * 0.70
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_0_floor_binds(make_allocator):
    """50% DD with floor=0.5 → exposure floored at 0.5, not 0.25."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    alloc = make_allocator(risk_config=risk)
    tickers = ["A", "B"]
    prices = {t: _flat_history(100) for t in tickers}
    result = alloc.allocate(tickers, prices, current_drawdown=0.50)
    investable = (1.0 - alloc._constraints.min_cash_pct) * 0.50
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_0_disabled_when_penalty_none(make_allocator):
    """No drawdown_penalty → ignore current_drawdown entirely."""
    alloc = make_allocator(risk_config=RiskConfig())
    tickers = ["A", "B"]
    prices = {t: _flat_history(100) for t in tickers}
    result = alloc.allocate(tickers, prices, current_drawdown=0.50)
    investable = 1.0 - alloc._constraints.min_cash_pct
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-9)
```

Add a `make_allocator` fixture to `tests/conftest.py` that constructs an Allocator with a single `_ScoreOnce` entry signal scoring all tickers at 1.0 — adapt to existing fixtures if `make_allocator` already covers it.

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_allocator_risk.py -v -k phase_0`
Expected: failures because Phase 0 isn't applied yet (sum(targets) won't match the scaled investable).

- [ ] **Step 3: Implement Phase 0**

Edit `src/midas/allocator.py`. At the top of `allocate()` (after `if num_tickers == 0` early-return), insert:

```python
        # Phase 0: CPPI drawdown overlay (no-op if disabled).
        exposure_scale = 1.0
        if (
            self._risk_config is not None
            and self._risk_config.drawdown_penalty is not None
            and self._risk_config.drawdown_floor is not None
        ):
            from midas.risk import apply_drawdown_overlay
            exposure_scale = apply_drawdown_overlay(
                current_drawdown=current_drawdown,
                penalty=self._risk_config.drawdown_penalty,
                floor=self._risk_config.drawdown_floor,
            )

        investable = (1.0 - self._constraints.min_cash_pct) * exposure_scale
```

(Replace the existing line `investable = 1.0 - self._constraints.min_cash_pct` with the block above.)

Move the `from midas.risk import` to the module-level imports.

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_allocator_risk.py -v -k phase_0`
Expected: 4 tests pass.

Run: `uv run pytest`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/allocator.py tests/test_allocator_risk.py tests/conftest.py
git commit -m "Phase 0: CPPI drawdown overlay in allocator (#64)"
```

---

## Task 11: Phase 2 — inverse-vol score offset (with T-independence regression)

**Files:**
- Modify: `src/midas/allocator.py`
- Test: `tests/test_allocator_risk.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_allocator_risk.py`:

```python
def test_inverse_vol_high_vol_gets_less_weight(make_allocator):
    """Two tickers, identical scores, 10x vol gap → low-vol ticker gets ~10x more weight."""
    risk = RiskConfig(weighting="inverse_vol", vol_lookback_days=60)
    alloc = make_allocator(risk_config=risk)
    tickers = ["LO", "HI"]
    prices = {
        "LO": _diverging_history(120, daily_vol=0.005, seed=1),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result = alloc.allocate(tickers, prices)
    # 10x vol → 10x weight ratio (within ±15% tolerance for noise).
    ratio = result.targets["LO"] / max(result.targets["HI"], 1e-12)
    assert 7.0 < ratio < 13.0


def test_inverse_vol_t_independence(make_allocator):
    """T-independence: the offset is added *outside* /T, so the inverse-vol weight
    ratio at fixed scores must be invariant to softmax_temperature.

    Regression test for PR #63: that bug used (1/vol)^(1/T), which would produce
    wildly different ratios at T=0.2 vs T=2.0 (~32:1 vs ~1.4:1 instead of ~10:1).
    """
    tickers = ["LO", "HI"]
    prices = {
        "LO": _diverging_history(120, daily_vol=0.005, seed=1),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }

    def ratio_at_T(temperature: float) -> float:
        constraints = AllocationConstraints(softmax_temperature=temperature)
        alloc = make_allocator(
            risk_config=RiskConfig(weighting="inverse_vol", vol_lookback_days=60),
            constraints=constraints,
        )
        result = alloc.allocate(tickers, prices)
        return result.targets["LO"] / max(result.targets["HI"], 1e-12)

    r_cold = ratio_at_T(0.2)
    r_hot = ratio_at_T(2.0)
    # Identical scores → score term cancels at any T → ratio comes from offset alone.
    # Ratio should be ~10x at both temperatures, with ~5% relative tolerance.
    assert math.isclose(r_cold, r_hot, rel_tol=0.10)


def test_equal_weighting_unaffected_by_vol(make_allocator):
    """Default weighting='equal' should ignore vol entirely."""
    risk = RiskConfig()  # weighting=equal by default
    alloc = make_allocator(risk_config=risk)
    tickers = ["LO", "HI"]
    prices = {
        "LO": _diverging_history(120, daily_vol=0.005, seed=1),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result = alloc.allocate(tickers, prices)
    # Equal scores + equal weighting → equal targets.
    assert math.isclose(result.targets["LO"], result.targets["HI"], rel_tol=1e-9)


def test_inverse_vol_falls_back_when_zero_vol(make_allocator):
    """A constant-price ticker (zero realized vol) falls back to Option A — held."""
    risk = RiskConfig(weighting="inverse_vol", vol_lookback_days=60)
    alloc = make_allocator(risk_config=risk)
    tickers = ["FLAT", "HI"]
    prices = {
        "FLAT": _flat_history(120),
        "HI": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    current = {"FLAT": 0.10, "HI": 0.0}
    result = alloc.allocate(tickers, prices, current_weights=current)
    # FLAT held at its current weight; HI takes the active budget.
    assert math.isclose(result.targets["FLAT"], 0.10, abs_tol=1e-9)
    investable = 1.0 - alloc._constraints.min_cash_pct
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_allocator_risk.py -v -k "inverse_vol or equal_weighting"`
Expected: failures because the offset isn't applied.

- [ ] **Step 3: Implement inverse-vol offset**

Edit `src/midas/allocator.py`. Add to imports:

```python
from midas.risk import apply_drawdown_overlay, inverse_vol_offset, realized_vol
```

Modify `_softmax_allocate` to accept optional offsets (around line 218):

```python
def _softmax_allocate(
    self,
    tickers: list[str],
    blended_scores: dict[str, float],
    budget: float,
    temperature: float,
    targets: dict[str, float],
    offsets: dict[str, float] | None = None,
) -> None:
    """Distribute ``budget`` across ``tickers`` via softmax with optional offsets.

    With offsets, weight_i ∝ exp(blended_i / T + offset_i). The offset is applied
    *outside* the /T divider so its relative impact is invariant to ``temperature``
    — fixes the PR #63 (1/vol)^(1/T) regression where inverse-vol intensity
    coupled to softmax temperature.
    """
    if not tickers or budget <= 0:
        for ticker in tickers:
            targets[ticker] = 0.0
        return
    temp_safe = max(temperature, MIN_TEMPERATURE)
    offsets = offsets or {}
    # Compute exponents = blended/T + offset; subtract max for stability.
    exponents = {
        ticker: (blended_scores[ticker] / temp_safe) + offsets.get(ticker, 0.0)
        for ticker in tickers
    }
    max_exp = max(exponents.values())
    exps = {ticker: math.exp(exponents[ticker] - max_exp) for ticker in tickers}
    total_exp = sum(exps.values())
    for ticker in tickers:
        targets[ticker] = budget * exps[ticker] / total_exp
```

Update `_apply_cap_with_redistribution` to pass `offsets` through. Add a parameter:

```python
def _apply_cap_with_redistribution(
    self,
    active: list[str],
    blended_scores: dict[str, float],
    initial_budget: float,
    temperature: float,
    targets: dict[str, float],
    offsets: dict[str, float] | None = None,
) -> None:
```

And in its body, the recursive `_softmax_allocate` call passes `offsets=offsets`.

In `allocate()`, before calling `_softmax_allocate` for Phase 2, compute offsets and check for zero-vol fallback:

```python
        # Phase 2 prep: compute inverse-vol offsets and reclassify zero-vol tickers as held.
        offsets: dict[str, float] = {}
        if self._risk_config is not None and self._risk_config.weighting == "inverse_vol":
            from midas.models import DEFAULT_VOL_FLOOR
            still_active: list[str] = []
            for ticker in active:
                history = price_data.get(ticker)
                if history is None or len(history) < self._risk_config.vol_lookback_days + 1:
                    held.append(ticker)
                    targets[ticker] = _held_target(ticker)
                    continue
                vol = realized_vol(np.asarray(history.close), self._risk_config.vol_lookback_days)
                offset = inverse_vol_offset(vol, vol_floor=DEFAULT_VOL_FLOOR)
                if math.isnan(offset):
                    held.append(ticker)
                    targets[ticker] = _held_target(ticker)
                    continue
                offsets[ticker] = offset
                still_active.append(ticker)
            active = still_active
            # Recompute held_total + budget after reclassification.
            held_total = sum(targets[t] for t in held)
            budget_for_active = max(investable - held_total, 0.0)
```

Insert this block between the existing held-bookkeeping and the `_softmax_allocate` call. Pass `offsets=offsets` to `_softmax_allocate` and `_apply_cap_with_redistribution`.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_allocator_risk.py -v`
Expected: all phase_0 + inverse_vol + equal_weighting + zero_vol tests pass.

Run: `uv run pytest`
Expected: all existing tests pass — the offset path is gated by `weighting == "inverse_vol"`.

- [ ] **Step 5: Commit**

```bash
git add src/midas/allocator.py tests/test_allocator_risk.py
git commit -m "Phase 2: inverse-vol score offset (T-independent), with zero-vol fallback (#64)"
```

---

## Task 12: Phase 4 — portfolio vol target

**Files:**
- Modify: `src/midas/allocator.py`
- Test: `tests/test_allocator_risk.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_allocator_risk.py`:

```python
def test_phase_4_vol_target_scales_when_predicted_exceeds(make_allocator):
    """High-vol portfolio with tight target → all weights scale by target/predicted."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.10)
    alloc = make_allocator(risk_config=risk)
    tickers = ["HI1", "HI2"]
    # ~80% annualized vol per ticker, equal weight → ~57% predicted vs 10% target.
    prices = {
        "HI1": _diverging_history(120, daily_vol=0.05, seed=1),
        "HI2": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result = alloc.allocate(tickers, prices)
    investable = 1.0 - alloc._constraints.min_cash_pct
    # Sum dropped well below investable: most went to cash.
    assert sum(result.targets.values()) < investable * 0.5


def test_phase_4_vol_target_skips_when_predicted_below(make_allocator):
    """Low-vol portfolio under target → no scaling."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.50)
    alloc = make_allocator(risk_config=risk)
    tickers = ["LO1", "LO2"]
    prices = {
        "LO1": _diverging_history(120, daily_vol=0.005, seed=1),
        "LO2": _diverging_history(120, daily_vol=0.005, seed=2),
    }
    result = alloc.allocate(tickers, prices)
    investable = 1.0 - alloc._constraints.min_cash_pct
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_4_skips_on_insufficient_history(make_allocator):
    """Any ticker with too little history → entire Phase 4 step skipped."""
    risk = RiskConfig(weighting="equal", vol_lookback_days=60, vol_target=0.10)
    alloc = make_allocator(risk_config=risk)
    tickers = ["FULL", "SHORT"]
    prices = {
        "FULL": _diverging_history(120, daily_vol=0.05, seed=1),
        "SHORT": _diverging_history(30, daily_vol=0.05, seed=2),  # < lookback+1
    }
    result = alloc.allocate(tickers, prices)
    # No vol scaling → targets sum to investable.
    investable = 1.0 - alloc._constraints.min_cash_pct
    # SHORT has < lookback bars so falls back to held — it doesn't get a softmax weight,
    # but the *active* budget is still consumed by FULL. Sum across all tickers is investable.
    assert math.isclose(sum(result.targets.values()), investable, abs_tol=1e-6)


def test_phase_0_phase_4_compose_multiplicatively(make_allocator):
    """20% DD shrinks gross to 0.7×; if vol target binds on top, total is 0.7 × target/predicted."""
    risk = RiskConfig(
        weighting="equal",
        vol_lookback_days=60,
        vol_target=0.10,
        drawdown_penalty=1.5,
        drawdown_floor=0.5,
    )
    alloc = make_allocator(risk_config=risk)
    tickers = ["HI1", "HI2"]
    prices = {
        "HI1": _diverging_history(120, daily_vol=0.05, seed=1),
        "HI2": _diverging_history(120, daily_vol=0.05, seed=2),
    }
    result_no_dd = alloc.allocate(tickers, prices, current_drawdown=0.0)
    result_dd = alloc.allocate(tickers, prices, current_drawdown=0.20)
    # Same vol-targeting math; only difference is the 0.7× CPPI multiplier.
    sum_no_dd = sum(result_no_dd.targets.values())
    sum_dd = sum(result_dd.targets.values())
    assert math.isclose(sum_dd, sum_no_dd * 0.70, rel_tol=0.02)
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_allocator_risk.py -v -k "phase_4 or compose"`
Expected: failures — Phase 4 not implemented.

- [ ] **Step 3: Implement Phase 4**

In `src/midas/allocator.py`, add to imports:

```python
from midas.risk import (
    apply_drawdown_overlay,
    inverse_vol_offset,
    predict_portfolio_vol,
    realized_vol,
)
```

After `_apply_cap_with_redistribution` returns inside `allocate()`, append:

```python
        # Phase 4: portfolio vol target. Scale the entire weight vector down if
        # predicted annualized vol exceeds the configured target. Skip if any
        # active ticker has insufficient or degenerate history.
        if (
            self._risk_config is not None
            and self._risk_config.vol_target is not None
            and active
        ):
            lookback = self._risk_config.vol_lookback_days
            log_returns_per_ticker = []
            ok = True
            for ticker in active:
                history = price_data.get(ticker)
                if history is None or len(history) < lookback + 1:
                    ok = False
                    break
                window = np.asarray(history.close[-(lookback + 1):])
                if np.any(window <= 0):
                    ok = False
                    break
                log_returns_per_ticker.append(np.diff(np.log(window)))
            if ok and log_returns_per_ticker:
                log_returns = np.column_stack(log_returns_per_ticker)
                # Skip if any ticker's series has zero variance (singular row in Σ).
                if not np.any(np.std(log_returns, axis=0, ddof=1) == 0.0):
                    weights = np.array([targets[t] for t in active])
                    predicted = predict_portfolio_vol(weights, log_returns)
                    if predicted > self._risk_config.vol_target > 0.0:
                        scale = self._risk_config.vol_target / predicted
                        for ticker in active:
                            targets[ticker] *= scale
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_allocator_risk.py -v`
Expected: all phase_4 and compose tests pass.

Run: `uv run pytest`
Expected: all existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/allocator.py tests/test_allocator_risk.py
git commit -m "Phase 4: portfolio vol target + multiplicative composition with Phase 0 (#64)"
```

---

## Task 13: Track running peak in backtest, pass `current_drawdown` to allocator

**Files:**
- Modify: `src/midas/backtest.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_backtest.py`:

```python
from midas.models import RiskConfig


class TestBacktestDrawdownThreading:
    def test_cppi_reduces_exposure_during_drawdown(self, fixture_for_backtest_with_risk):
        """End-to-end: configuring CPPI shrinks gross exposure during a synthetic crash."""
        risk_off = fixture_for_backtest_with_risk(risk_config=RiskConfig())
        risk_on = fixture_for_backtest_with_risk(
            risk_config=RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5),
        )
        # During the crash window, gross exposure should be lower under CPPI.
        # Compare cash-as-fraction-of-portfolio at the trough.
        assert risk_on.cash_at_trough > risk_off.cash_at_trough
```

(The fixture `fixture_for_backtest_with_risk` constructs a small synthetic backtest where prices crash 30% mid-window. Adapt to existing test infrastructure — the existing `tests/test_backtest.py` shows the pattern; lift the closest fixture and add a `risk_config` kwarg.)

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_backtest.py::TestBacktestDrawdownThreading -v`
Expected: failure — `current_drawdown` isn't computed yet, so allocator gets 0.0.

- [ ] **Step 3: Add peak tracking to `_SimState`**

Edit `src/midas/backtest.py`. In the `_SimState` dataclass (around line 130), add:

```python
    # Running peak portfolio value, updated each bar after equity_curve append.
    # Used by the allocator's CPPI overlay to compute current drawdown.
    peak_value: float = 0.0
```

Update the equity-curve update site (line 451) to also update peak:

```python
            value = state.portfolio_value(_close_prices(current_data))
            state.equity_curve.append((day, value))
            state.peak_value = max(state.peak_value, value)
```

- [ ] **Step 4: Pass `current_drawdown` to `allocate()`**

Find every call to `self._allocator.allocate(` in `backtest.py`. Compute current drawdown before each call:

```python
            current_drawdown = (
                (state.peak_value - current_value) / state.peak_value
                if state.peak_value > 0 else 0.0
            )
            allocation = self._allocator.allocate(
                tickers,
                current_data,
                context=context,
                current_weights=current_weights,
                current_drawdown=current_drawdown,
            )
```

Where `current_value = state.portfolio_value(_close_prices(current_data))`. Reuse if already computed in scope.

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_backtest.py -v -k drawdown`
Expected: drawdown threading test passes.

Run: `uv run pytest`
Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/midas/backtest.py tests/test_backtest.py
git commit -m "Track running peak; thread current_drawdown to allocator (#64)"
```

---

## Task 14: `RiskMetrics` dataclass + per-bar computation

**Files:**
- Create: `src/midas/risk_metrics.py`
- Test: `tests/test_risk_metrics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_risk_metrics.py`:

```python
"""Unit tests for read-only risk telemetry."""

import math
from datetime import date, timedelta

import numpy as np

from midas.risk_metrics import RiskMetrics, compute_risk_metrics


def _equity_curve(values: list[float]) -> list[tuple[date, float]]:
    start = date(2020, 1, 1)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


class TestComputeRiskMetrics:
    def test_drawdown_from_peak(self) -> None:
        # Peak 110, current 99 → 10% drawdown.
        curve = _equity_curve([100, 105, 110, 99])
        m = compute_risk_metrics(curve, vol_target=None, per_strategy_pnl={})
        assert math.isclose(m.drawdown_from_peak, 0.10, abs_tol=1e-9)

    def test_no_drawdown_at_new_peak(self) -> None:
        curve = _equity_curve([100, 110, 120])
        m = compute_risk_metrics(curve, vol_target=None, per_strategy_pnl={})
        assert m.drawdown_from_peak == 0.0

    def test_realized_vol_60d(self) -> None:
        np.random.seed(0)
        # 80 days of synthetic returns with daily stdev 0.01.
        returns = np.random.normal(0.0, 0.01, 80)
        values = [100.0]
        for r in returns:
            values.append(values[-1] * (1 + r))
        curve = _equity_curve(values)
        m = compute_risk_metrics(curve, vol_target=None, per_strategy_pnl={})
        # Annualized: 0.01 * sqrt(252) ≈ 0.1587, with ±20% sample noise.
        assert math.isclose(m.realized_vol_60d, 0.01 * math.sqrt(252), rel_tol=0.20)

    def test_per_strategy_pnl_passthrough(self) -> None:
        curve = _equity_curve([100, 110])
        m = compute_risk_metrics(
            curve, vol_target=0.20, per_strategy_pnl={"BollingerBand": 5.0, "RSIOversold": 5.0}
        )
        assert m.per_strategy_pnl == {"BollingerBand": 5.0, "RSIOversold": 5.0}
        assert m.vol_target == 0.20
```

- [ ] **Step 2: Run tests, verify failure**

Run: `uv run pytest tests/test_risk_metrics.py -v`
Expected: ImportError on `midas.risk_metrics`.

- [ ] **Step 3: Implement**

Create `src/midas/risk_metrics.py`:

```python
"""Read-only risk telemetry. No feedback into construction.

Computed each bar in backtest, each tick in live. Surfaced through the
existing output layer; never modifies allocator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from midas.risk import TRADING_DAYS_PER_YEAR

VOL_LOOKBACK_BARS = 60
SHARPE_LOOKBACK_BARS = 252


@dataclass(frozen=True)
class RiskMetrics:
    realized_vol_60d: float
    vol_target: float | None
    drawdown_from_peak: float
    rolling_sharpe_252d: float
    per_strategy_pnl: dict[str, float] = field(default_factory=dict)
    per_ticker_vol_contribution: dict[str, float] = field(default_factory=dict)


def compute_risk_metrics(
    equity_curve: list[tuple[date, float]],
    vol_target: float | None,
    per_strategy_pnl: dict[str, float],
    per_ticker_vol_contribution: dict[str, float] | None = None,
) -> RiskMetrics:
    """Compute current risk metrics from equity curve and pre-aggregated attribution.

    Args:
        equity_curve: list of (date, portfolio_value), oldest first.
        vol_target: configured vol target for context (None if unset).
        per_strategy_pnl: cumulative attributed P&L per strategy name.
        per_ticker_vol_contribution: per-ticker share of portfolio vol (optional).

    Returns:
        RiskMetrics snapshot at the latest equity-curve point.
    """
    if not equity_curve:
        return RiskMetrics(
            realized_vol_60d=0.0,
            vol_target=vol_target,
            drawdown_from_peak=0.0,
            rolling_sharpe_252d=0.0,
            per_strategy_pnl=dict(per_strategy_pnl),
            per_ticker_vol_contribution=per_ticker_vol_contribution or {},
        )

    values = np.array([v for _, v in equity_curve], dtype=float)

    peak = float(np.maximum.accumulate(values)[-1])
    current = float(values[-1])
    drawdown = (peak - current) / peak if peak > 0 else 0.0

    realized_vol = _annualized_vol(values, VOL_LOOKBACK_BARS)
    sharpe = _rolling_sharpe(values, SHARPE_LOOKBACK_BARS)

    return RiskMetrics(
        realized_vol_60d=realized_vol,
        vol_target=vol_target,
        drawdown_from_peak=drawdown,
        rolling_sharpe_252d=sharpe,
        per_strategy_pnl=dict(per_strategy_pnl),
        per_ticker_vol_contribution=per_ticker_vol_contribution or {},
    )


def _annualized_vol(values: np.ndarray, lookback: int) -> float:
    if values.size < 2:
        return 0.0
    window = values[-(lookback + 1):] if values.size > lookback else values
    log_returns = np.diff(np.log(window[window > 0]))
    if log_returns.size < 2:
        return 0.0
    return float(np.std(log_returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _rolling_sharpe(values: np.ndarray, lookback: int) -> float:
    if values.size < 2:
        return 0.0
    window = values[-(lookback + 1):] if values.size > lookback else values
    log_returns = np.diff(np.log(window[window > 0]))
    if log_returns.size < 2:
        return 0.0
    mean = float(np.mean(log_returns))
    stdev = float(np.std(log_returns, ddof=1))
    if stdev == 0.0:
        return 0.0
    return mean / stdev * float(np.sqrt(TRADING_DAYS_PER_YEAR))
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_risk_metrics.py -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/risk_metrics.py tests/test_risk_metrics.py
git commit -m "Add RiskMetrics dataclass + compute_risk_metrics (#64)"
```

---

## Task 15: Per-strategy attribution mechanic in backtest

**Files:**
- Modify: `src/midas/backtest.py`
- Test: `tests/test_backtest.py` (or `tests/test_risk_metrics.py`)

- [ ] **Step 1: Write failing test**

Append to `tests/test_backtest.py`:

```python
class TestPerStrategyAttribution:
    def test_two_strategy_split(self, two_strategy_backtest_fixture):
        """Synthetic 2-strategy backtest: each strategy contributes 0.5 to every buy.
        After a +10% portfolio gain, attribution should split P&L 50/50.
        """
        result = two_strategy_backtest_fixture.run()
        attribution = result.risk_metrics.per_strategy_pnl
        assert "BollingerBand" in attribution
        assert "RSIOversold" in attribution
        total = attribution["BollingerBand"] + attribution["RSIOversold"]
        assert math.isclose(attribution["BollingerBand"], total / 2, rel_tol=0.05)

    def test_single_strategy_entry_full_attribution(self, single_strategy_fixture):
        """The typical case: only one strategy contributed to entry → all P&L attributed to it."""
        result = single_strategy_fixture.run()
        attribution = result.risk_metrics.per_strategy_pnl
        non_zero = [k for k, v in attribution.items() if v != 0.0]
        assert non_zero == ["BollingerBand"]
```

(Build the fixture `two_strategy_backtest_fixture` to wrap an existing minimal backtest with two strategies that score 1.0 / 1.0 on a single ticker. Lift from existing `tests/test_backtest.py` patterns.)

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_backtest.py::TestPerStrategyAttribution -v`
Expected: failure — `result.risk_metrics` doesn't exist yet, or `per_strategy_pnl` is empty.

- [ ] **Step 3: Implement attribution in backtest**

Edit `src/midas/backtest.py`. Add to `_SimState`:

```python
    # Per-ticker cost-basis-weighted attribution dict: {ticker: {strategy_name: share}}.
    # Shares sum to 1.0 per ticker. Updated on buys; consumed on sells/MTM for P&L split.
    attribution: dict[str, dict[str, float]] = field(default_factory=dict)
    # Aggregate cost basis per ticker (sum of buy_size across all open lots).
    basis: dict[str, float] = field(default_factory=dict)
    # Cumulative attributed P&L per strategy.
    cumulative_strategy_pnl: dict[str, float] = field(default_factory=dict)
```

In the buy-execution code path (where lots are appended), update attribution. Find the location where a buy is recorded (search for `state.lots[ticker].append`):

```python
            buy_size = order.shares * exec_price
            new_basis = state.basis.get(ticker, 0.0) + buy_size
            old_attr = state.attribution.get(ticker, {})
            new_attr: dict[str, float] = {}
            for strat, share in old_attr.items():
                new_attr[strat] = share * (state.basis.get(ticker, 0.0) / new_basis)
            for strat, contrib in order.context.contributions.items():
                new_attr[strat] = new_attr.get(strat, 0.0) + contrib * (buy_size / new_basis)
            # Renormalize to sum to 1.0 (contributions may not be normalized).
            total = sum(new_attr.values())
            if total > 0:
                new_attr = {k: v / total for k, v in new_attr.items()}
            state.attribution[ticker] = new_attr
            state.basis[ticker] = new_basis
```

In the sell-execution code path (where lots are consumed), split realized P&L:

```python
            realized_pnl = (exec_price - lot.cost_basis) * shares_consumed
            attr = state.attribution.get(ticker, {})
            for strat, share in attr.items():
                state.cumulative_strategy_pnl[strat] = (
                    state.cumulative_strategy_pnl.get(strat, 0.0) + realized_pnl * share
                )
            # Reduce basis proportionally; attribution shares unchanged.
            sold_value = exec_price * shares_consumed  # use exec, not basis, for size
            state.basis[ticker] = max(state.basis.get(ticker, 0.0) - sold_value, 0.0)
```

(The exact site depends on the existing FIFO-consumption block in `backtest.py`. Place the attribution split next to where `realized_pnl` is computed today.)

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_backtest.py::TestPerStrategyAttribution -v`
Expected: both tests pass.

Run: `uv run pytest`
Expected: all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/backtest.py tests/test_backtest.py
git commit -m "Per-strategy P&L attribution via cost-basis-weighted blend (#64)"
```

---

## Task 16: Surface `RiskMetrics` in `BacktestResult` and output

**Files:**
- Modify: `src/midas/results.py`
- Modify: `src/midas/backtest.py`
- Modify: `src/midas/output.py`
- Test: `tests/test_backtest.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_backtest.py`:

```python
def test_backtest_result_has_risk_metrics(simple_backtest_fixture):
    result = simple_backtest_fixture.run()
    assert result.risk_metrics is not None
    assert result.risk_metrics.drawdown_from_peak >= 0.0
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_backtest.py -v -k "risk_metrics"`
Expected: AttributeError on `result.risk_metrics`.

- [ ] **Step 3: Add field + populate**

Edit `src/midas/results.py`. Add to `BacktestResult` (frozen dataclass):

```python
from midas.risk_metrics import RiskMetrics

# Inside BacktestResult dataclass:
    risk_metrics: RiskMetrics | None = None
```

Edit `src/midas/backtest.py`. At the end of `BacktestEngine.run()` where `BacktestResult` is constructed, compute and attach:

```python
        from midas.risk_metrics import compute_risk_metrics

        risk_metrics = compute_risk_metrics(
            equity_curve=state.equity_curve,
            vol_target=(self._allocator.risk_config.vol_target if self._allocator.risk_config else None),
            per_strategy_pnl=state.cumulative_strategy_pnl,
        )
        # ... pass risk_metrics= to BacktestResult(...)
```

- [ ] **Step 4: Render in output**

Edit `src/midas/output.py`. Find the function that renders the backtest summary (search for `final_value` or `total_return`). Append a Risk Telemetry block:

```python
    if result.risk_metrics is not None:
        m = result.risk_metrics
        lines.append("")
        lines.append("Risk Telemetry")
        lines.append(f"  Realized vol (60d):     {m.realized_vol_60d:.2%}")
        if m.vol_target is not None:
            lines.append(f"  Vol target:             {m.vol_target:.2%}")
        lines.append(f"  Drawdown from peak:     {m.drawdown_from_peak:.2%}")
        lines.append(f"  Rolling Sharpe (252d):  {m.rolling_sharpe_252d:.2f}")
        if m.per_strategy_pnl:
            lines.append("  Per-strategy P&L:")
            for strat, pnl in sorted(m.per_strategy_pnl.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {strat:24s} {pnl:>+12.2f}")
```

(Adapt to the existing rendering style — look at how Train/Test summaries are formatted.)

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_backtest.py -v -k "risk_metrics"`
Expected: pass.

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/midas/results.py src/midas/backtest.py src/midas/output.py tests/test_backtest.py
git commit -m "Surface RiskMetrics in BacktestResult and summary output (#64)"
```

---

## Task 17: Live mode warning + telemetry

**Files:**
- Modify: `src/midas/live.py`
- Test: `tests/test_live.py` (create if not present)

- [ ] **Step 1: Write failing test**

Add to `tests/test_live.py` (or create it):

```python
import logging

from midas.live import LiveEngine
from midas.models import RiskConfig


def test_live_warns_when_cppi_configured(make_live_engine, caplog):
    """LiveEngine must warn if drawdown_penalty is set — CPPI is inert in live v1."""
    risk = RiskConfig(drawdown_penalty=1.5, drawdown_floor=0.5)
    with caplog.at_level(logging.WARNING):
        _engine = make_live_engine(risk_config=risk)
    assert any("drawdown_penalty" in r.message and "live" in r.message.lower() for r in caplog.records)


def test_live_silent_when_cppi_off(make_live_engine, caplog):
    risk = RiskConfig(weighting="inverse_vol", vol_target=0.20)
    with caplog.at_level(logging.WARNING):
        _engine = make_live_engine(risk_config=risk)
    assert not any("drawdown_penalty" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_live.py -v -k cppi`
Expected: failure — no warning emitted.

- [ ] **Step 3: Implement warning**

Edit `src/midas/live.py`. Following the existing TrailingStop warning pattern at lines 73–77, add inside `LiveEngine.__init__` after the `TrailingStop` block:

```python
        risk_config = getattr(self._allocator, "risk_config", None)
        if risk_config is not None and risk_config.drawdown_penalty is not None:
            logger.warning(
                "drawdown_penalty is configured but live mode does not yet "
                "track a persistent peak across runs; the CPPI overlay is "
                "inert in live and behavior will diverge from backtest. "
                "Peak persistence is tracked for v2.",
            )
```

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_live.py -v -k cppi`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/live.py tests/test_live.py
git commit -m "Live mode: warn that CPPI overlay is inert (#64)"
```

---

## Task 18: Optimizer risk-config propagation regression test

**Files:**
- Create: `tests/test_optimizer_risk_propagation.py`
- Modify (only if test fails): `src/midas/optimizer.py`

- [ ] **Step 1: Write the regression test**

Create `tests/test_optimizer_risk_propagation.py`:

```python
"""Regression: PR #63's bug 2 — optimizer silently dropped the risk: block.

This test asserts that a configured RiskConfig survives the optimizer's
trial-construction loop and reaches every trial's Allocator instance.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

from midas.config import load_strategies
from midas.models import RiskConfig
from midas.optimizer import _build_allocator_for_trial  # adapt name if different


def test_risk_config_survives_trial_construction(tmp_path: Path) -> None:
    yaml_path = tmp_path / "strategies.yaml"
    yaml_path.write_text(textwrap.dedent("""
        strategies:
          - name: BollingerBand
            params: {window: 20}
        risk:
          weighting: inverse_vol
          vol_lookback_days: 90
          vol_target: 0.18
          drawdown_penalty: 1.7
          drawdown_floor: 0.4
    """))

    _, _, risk_config = load_strategies(yaml_path)
    assert risk_config == RiskConfig(
        weighting="inverse_vol",
        vol_lookback_days=90,
        vol_target=0.18,
        drawdown_penalty=1.7,
        drawdown_floor=0.4,
    )

    # Whatever the optimizer's per-trial Allocator-construction helper is, it
    # must propagate this RiskConfig unchanged.
    captured: list[RiskConfig | None] = []

    def fake_allocator(_entries, _constraints, _n, *, risk_config=None):
        captured.append(risk_config)

        class _Stub:
            risk_config = None
        _Stub.risk_config = risk_config
        return _Stub()

    with patch("midas.optimizer.Allocator", fake_allocator):
        # Construct one trial allocator using the helper used inside the trial loop.
        # Replace this with the actual entry point — see optimizer.py:276 region.
        _build_allocator_for_trial(
            strategies_yaml_path=yaml_path,
            trial_params={},  # default
        )

    assert len(captured) == 1
    assert captured[0] == risk_config
```

(If the optimizer doesn't already factor trial-allocator construction into a function, refactor `optimizer.py` so that the helper exists and is testable. Keep the refactor minimal — extract just the `Allocator(entries, constraints, n_tickers, risk_config=...)` call into a named function.)

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_optimizer_risk_propagation.py -v`
Expected: AttributeError on `_build_allocator_for_trial`, or assertion failure if `risk_config` arrives as `None`.

- [ ] **Step 3: Refactor `optimizer.py` if needed**

If `_build_allocator_for_trial` doesn't exist, extract one. Find the trial loop in `optimizer.py` (around line 276) and lift the `Allocator(...)` construction into a function:

```python
def _build_allocator_for_trial(
    strategies_yaml_path: Path,
    trial_params: dict[str, Any],
) -> Allocator:
    configs, constraints, risk_config = load_strategies(strategies_yaml_path)
    # ... apply trial_params overrides to constraints and strategy params ...
    return Allocator(entries, constraints, n_tickers, risk_config=risk_config)
```

The trial loop now calls `_build_allocator_for_trial(...)` instead of constructing inline.

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_optimizer_risk_propagation.py -v`
Expected: passes.

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/midas/optimizer.py tests/test_optimizer_risk_propagation.py
git commit -m "Optimizer: factor trial allocator construction; assert RiskConfig propagates (#64)"
```

---

## Task 19: Lookahead regression — sliced-vs-full equality at bar T

**Files:**
- Create: `tests/test_lookahead_regression.py`

- [ ] **Step 1: Write the test**

Create `tests/test_lookahead_regression.py`:

```python
"""Regression for PR #63 bug 1: risk math must use sliced history per bar.

This test runs the same backtest twice — once truncated at bar T, once with
the full history — and asserts the risk metrics at bar T are identical. If
any risk computation depends on prices[T+1:], the truncated run differs and
the test fails.
"""

from __future__ import annotations

import numpy as np

from midas.models import RiskConfig


def test_risk_metrics_at_bar_t_invariant_to_future_bars(make_synthetic_backtest):
    """Two backtests with identical [0:T] history must produce identical RiskMetrics[T].

    The full-history run extends past T but the truncated run stops at T. If the
    truncated run's terminal RiskMetrics matches the full run's RiskMetrics at T,
    no risk math has consulted prices[T+1:].
    """
    risk_config = RiskConfig(
        weighting="inverse_vol",
        vol_lookback_days=30,
        vol_target=0.20,
        drawdown_penalty=1.5,
        drawdown_floor=0.5,
    )
    full = make_synthetic_backtest(risk_config=risk_config, n_bars=120)
    truncated = make_synthetic_backtest(risk_config=risk_config, n_bars=80)

    # equity_curve length differs; align at bar 80.
    full_value_at_80 = full.equity_curve[79][1]
    trunc_value_at_80 = truncated.equity_curve[79][1]
    assert np.isclose(full_value_at_80, trunc_value_at_80, rtol=1e-9)

    # Compare the full run's mid-window RiskMetrics to the truncated run's terminal.
    # (Requires the backtest to optionally snapshot RiskMetrics per bar; if not
    # available, compare cumulative_strategy_pnl as a proxy that depends on the
    # same quantities.)
    assert np.isclose(
        full.attribution_at_bar(79),
        truncated.attribution_at_bar(79),
        rtol=1e-9,
    )
```

(The fixture `make_synthetic_backtest` must produce a small synthetic 2-ticker backtest and expose `equity_curve` and a way to query attribution at a given bar. If the backtest doesn't currently snapshot RiskMetrics per bar, add minimal hooks: a `risk_metrics_per_bar: list[RiskMetrics]` field on `_SimState`, populated alongside `equity_curve`.)

- [ ] **Step 2: Run test, verify failure**

Run: `uv run pytest tests/test_lookahead_regression.py -v`
Expected: passes if the implementation is honest, fails if any function uses non-sliced history. (At this point in the plan it should pass — the implementation has been TDD-disciplined to avoid lookahead. This test is a *guard* for future regressions.)

- [ ] **Step 3: Add per-bar RiskMetrics snapshotting if needed**

If the test exposes a missing hook, add to `_SimState`:

```python
    risk_metrics_per_bar: list["RiskMetrics"] = field(default_factory=list)
```

In the bar loop after `equity_curve.append`, call:

```python
            from midas.risk_metrics import compute_risk_metrics
            state.risk_metrics_per_bar.append(
                compute_risk_metrics(
                    equity_curve=state.equity_curve,
                    vol_target=(
                        self._allocator.risk_config.vol_target
                        if self._allocator.risk_config
                        else None
                    ),
                    per_strategy_pnl=state.cumulative_strategy_pnl,
                )
            )
```

(Optional: gate behind a flag if the per-bar telemetry is too expensive for production. The lookahead regression test only needs it on for synthetic runs.)

- [ ] **Step 4: Run test, verify pass**

Run: `uv run pytest tests/test_lookahead_regression.py -v`
Expected: passes.

Run: `uv run pytest`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_lookahead_regression.py src/midas/backtest.py
git commit -m "Lookahead regression: truncated-vs-full bar-T equality test (#64)"
```

---

## Task 20: Documentation — `architecture.md` and `strategies.md`

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/strategies.md`

- [ ] **Step 1: Update `docs/architecture.md`**

Edit `docs/architecture.md`. In the **Allocator** section (around lines 46–78), add a new subsection after Phase 3 ("Soft Position Cap"):

```markdown
#### Phase 0: CPPI Drawdown Overlay (optional)

When `risk.drawdown_penalty` and `risk.drawdown_floor` are configured, the allocator
multiplies the gross investable budget by `max(1 - penalty * current_drawdown, floor)`
*before* the softmax runs. The freed budget becomes deliberate cash reserve and
recovers automatically as the portfolio heals. The driver (`backtest.py`) tracks
the running peak portfolio value; live mode warns at startup and treats the overlay
as inert pending peak persistence (v2).

#### Phase 4: Portfolio Vol Target (optional)

After the cap converges, if `risk.vol_target` is configured, the allocator computes
predicted annualized portfolio vol from a Ledoit-Wolf-shrunk covariance matrix and
scales the entire weight vector down by `vol_target / predicted` if the predicted
exceeds the target. Slack flows to cash. The cap is *not* re-applied after scaling —
scaling can only shrink, so an upper cap remains satisfied.

Phase 0 and Phase 4 stack multiplicatively. A 20% drawdown with `drawdown_penalty: 1.5`
shrinks gross to 70%; if the resulting portfolio's predicted vol still exceeds target,
Phase 4 scales further. Both are reduce-only; with aggressive settings during deep
drawdowns the gross can drop well below `drawdown_floor`. This is intentional.

#### Inverse-Vol Weighting

When `risk.weighting: inverse_vol`, the softmax exponent gains a per-ticker offset
of `-log(max(vol_i, vol_floor))` *added outside the `/T` divider*:

```
weight_i ∝ exp(blended_i / T + offset_i)
```

The form keeps the offset's contribution invariant to softmax temperature — a 10×
vol gap is always a 10× weight gap from vol alone, regardless of how concentrated
conviction is. (PR #63 used `(1/vol)^(1/T)` and was rejected for this coupling.)

Tickers with insufficient history or zero realized vol fall back to Option A
(held at current weight, excluded from the softmax).

#### Risk Telemetry

`RiskMetrics` (rolling 60-day vol, drawdown from peak, rolling Sharpe,
per-strategy P&L attribution, per-ticker vol contribution) is computed each bar
and surfaced through `output.py`. Strictly observational — never feeds back into
construction.
```

- [ ] **Step 2: Update `docs/strategies.md`**

Edit `docs/strategies.md`. Add a new section near the existing top-level configuration block:

```markdown
## Risk Discipline

Optional risk policy lives under a top-level `risk:` block in the strategies YAML.
Omit it for current behavior bit-for-bit. All vol quantities are **annualized**.

```yaml
risk:
  weighting: inverse_vol      # equal | inverse_vol; default equal
  vol_lookback_days: 60       # rolling window for vol and covariance estimates

  vol_target: 0.20            # annualized; null/omit disables Phase 4

  drawdown_penalty: 1.5       # exposure = max(1 - penalty * dd, floor)
  drawdown_floor: 0.5         # both required, both must be set or both omitted
```

The optimizer **does not** search risk knobs — risk is policy, easy to overfit,
and changing it is a deliberate user act. To experiment, edit the YAML and rerun.
For A/B comparisons keep two strategies files in version control.

CPPI (`drawdown_penalty`/`drawdown_floor`) is currently inert in `live` mode —
peak persistence requires a state file that is tracked for v2. Live runs will
log a warning at startup if CPPI is configured.
```

- [ ] **Step 3: Verify docs build cleanly**

Run: `uv run mypy src` (sanity check; docs changes shouldn't affect this).
Run: `uv run ruff check . && uv run ruff format .` (no-op on docs).
Run: `uv run pytest` (final all-tests sanity check).
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md docs/strategies.md
git commit -m "Document risk: YAML block, Phases 0/4, inverse-vol offset, telemetry (#64)"
```

---

## Final Verification

- [ ] **Step 1: Full test suite**

Run: `uv run pytest`
Expected: all tests pass, including:
- Existing tests (no regressions)
- `test_risk.py` (5 pure-function tests)
- `test_risk_metrics.py` (telemetry math)
- `test_allocator_risk.py` (CPPI, inverse-vol, T-independence, vol target, composition)
- `test_lookahead_regression.py` (sliced-vs-full equality)
- `test_optimizer_risk_propagation.py` (config survives trial loop)
- `test_models.py` (RiskConfig validation)
- `test_config.py` (YAML parsing)
- `test_live.py` (CPPI warning)

- [ ] **Step 2: Lint, format, type-check**

Run: `uv run ruff check . --fix && uv run ruff format . && uv run mypy src`
Expected: all clean. If anything fails, fix and recommit.

- [ ] **Step 3: Smoke test on a sample portfolio**

Run: `uv run midas backtest -p sample-portfolios/<some.yaml> -s example-strategies/balanced-growth.yaml --start 2020-01-01 --end 2024-01-01`
Expected: runs to completion; output includes a "Risk Telemetry" section showing zero/baseline values (no `risk:` block in the example YAML).

- [ ] **Step 4: Smoke test with risk enabled**

Add a `risk:` block to a copy of `balanced-growth.yaml` (vol_target + drawdown overlay), run the same backtest. Verify:
- Output shows non-zero drawdown / realized vol in the telemetry block
- Final value differs from the no-risk run (CPPI / vol target are doing something)

- [ ] **Step 5: Open the PR**

Push the branch and open a PR linking to #64. Title: `Risk engine: CPPI overlay, inverse-vol scaling, vol target, telemetry (#64)`.
