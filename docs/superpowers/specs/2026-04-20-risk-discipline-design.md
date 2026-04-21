# Risk Discipline Design

**Date:** 2026-04-20
**Scope:** Issues #52 (portfolio volatility targeting), #53 (per-ticker volatility scaling), #55 (Instrument Diversification Multiplier)
**Status:** Design approved, ready for implementation plan

## Motivation

Carver's `pysystemtrade` philosophy treats risk discipline as **policy at design time**, not as an optimization target. The three issues above compose into a single cohesive layer that sits between signal scoring and order generation:

- **#53 (per-ticker vol scaling)** reshapes weights so low-vol assets can absorb larger notional shares.
- **#55 (IDM)** scales the whole portfolio up when holdings are genuinely diversified, recapturing capital that would otherwise sit idle.
- **#52 (portfolio vol targeting)** enforces a ceiling on ex-ante portfolio volatility, scaling down when the forecast exceeds the target.

Shipping these together — with risk-on defaults and no way to "search over" risk parameters via the optimizer — makes midas opinionated where existing tools are permissive. This document specifies the architecture for that cohesive addition.

## Non-goals

- CPPI / drawdown-triggered de-risking (listed in #52 as "Part 2", explicitly deferred).
- Per-regime risk policies (crisis detection, volatility forecasting models beyond realized).
- Exposing risk parameters as optimizer search dimensions. Risk is policy, not a dial.
- CLI overrides for risk parameters. Strategy-level YAML only.

## Design Decisions

### 1. Phase ordering inside `Allocator.allocate`

Risk stages interleave with the existing 3-phase allocator:

```
Phase 1    Score entry signals                         (existing)
Phase 1.5  Precompute per-ticker vol + cov/corr        (NEW)
Phase 2    Softmax allocate with optional vol offset   (MODIFIED: consumes per-ticker vol)
Phase 3    Cap + redistribute                          (existing)
Phase 3.5  Apply Instrument Diversification Multiplier (NEW)
Phase 3.6  Re-cap + redistribute                       (existing helper, 2nd call)
Phase 3.7  Apply portfolio vol targeting               (NEW)
```

Rationale:

- Per-ticker vol scaling belongs inside softmax (phase 2) as an offset `-log(vol_i)` on the score exponent, so the existing normalization still produces a valid weight simplex.
- IDM scales weights up; it can push single positions over `max_position_pct`, so phase 3.6 reapplies the per-position cap.
- Vol targeting runs last because it only ever scales down — no cap interaction possible.

### 2. Covariance estimator

`sklearn.covariance.LedoitWolf` applied to the **correlation matrix** over `corr_lookback_days`. The covariance matrix is then composed:

```
cov = vols[:, None] * corr * vols[None, :]
```

where `vols` is the realized annualized vol per ticker over `vol_lookback_days`. This separation (Engle-Sheppard 2001, DCC-GARCH family) lets volatility react faster than correlation, which matches empirical behavior: vol spikes on news, correlation shifts regimes.

Defaults: `vol_lookback_days=60`, `corr_lookback_days=252`.

### 3. Module organization

New module `src/midas/risk.py` with five pure functions. No class state. All functions operate on already-computed frames rather than raw history (where applicable), keeping the hot path cheap:

```python
def realized_vol(
    history: PriceHistory,
    window: int,
    annualize: bool = True,
) -> float | None

def covariance_matrix(
    histories: Mapping[str, PriceHistory],
    vol_window: int,
    corr_window: int,
    vol_floor: float,
) -> pd.DataFrame

def apply_instrument_diversification_multiplier(
    weights: dict[str, float],
    corr: pd.DataFrame,
    cap: float,
) -> dict[str, float]

def apply_vol_targeting(
    weights: dict[str, float],
    cov: pd.DataFrame,
    target_annualized_vol: float,
) -> dict[str, float]

def predict_portfolio_vol(
    weights: dict[str, float],
    cov: pd.DataFrame,
) -> float
```

All functions return new dicts/frames (no mutation), and short-circuit on degenerate inputs.

### 4. Config shape

New `RiskConfig` dataclass colocated with `AllocationConstraints` in `src/midas/models.py`:

```python
@dataclass(frozen=True)
class RiskConfig:
    weighting: Literal["inverse_vol", "equal"] = "inverse_vol"
    vol_target_annualized: float = 0.20
    idm_cap: float = 2.5
    vol_lookback_days: int = 60
    corr_lookback_days: int = 252
    vol_floor: float = 0.02
```

`StrategyConfig` gains `risk: RiskConfig = field(default_factory=RiskConfig)`. Missing `risk:` block in YAML → all defaults (risk-on baseline).

YAML surface:

```yaml
risk:
  weighting: inverse_vol          # or "equal"
  vol_target_annualized: 0.20
  idm_cap: 2.5
  vol_lookback_days: 60
  corr_lookback_days: 252
  vol_floor: 0.02
```

No on/off flags. To disable a stage, set its knob to the neutral value (`weighting: equal`, `idm_cap: 1.0`, `vol_target_annualized: 999.0`). Keeps the pipeline uniform.

Note: `weighting: equal` disables the phase-2 per-ticker offset only. The correlation and covariance matrices are still computed each bar and remain available for IDM (phase 3.5) and vol targeting (phase 3.7).

### 5. Data flow and caching

Risk precompute piggybacks on the existing `Allocator.precompute_signals` method. Per bar:

1. Build per-ticker realized vol map.
2. Build correlation matrix (LedoitWolf-shrunk).
3. Compose covariance matrix.
4. Stash on `self._risk_cache` as `{"cov": DataFrame | None, "corr": DataFrame | None, "per_ticker_vol": dict[str, float]}`.

Cache is read-only inside allocation phases. Rebuilt every bar — LedoitWolf on 252×N (N ≤ ~50) is sub-millisecond, so staleness is not worth optimizing around.

### 6. Error handling

**Config-time (raises):**
- `vol_target_annualized <= 0` → `ValueError`
- `idm_cap < 1.0` → `ValueError` (IDM is always ≥ 1; cap < 1 is incoherent)
- `vol_lookback_days < 2` or `corr_lookback_days < 2` → `ValueError`
- `vol_floor < 0` → `ValueError`
- Unknown `weighting` value → caught by `Literal` + config parser

**Runtime (degrades silently, logs reason):**

| Condition | Behavior |
|---|---|
| `sum(weights) == 0` (nothing selected) | All phases no-op |
| Single ticker in weights | IDM trivially 1.0, vol targeting still applies |
| Ticker has NaN in window | Treated as insufficient history for that ticker |
| Ticker's realized vol < `vol_floor` | Clamped to `vol_floor` |
| `w·cov·w <= 0` (numerical zero) | Skip vol targeting |
| `w·corr·w <= 0` (near-singular) | Skip IDM (treat as 1.0) |
| LedoitWolf fails to converge | Catch, set `corr`/`cov` to `None`, skip stages |

**Logging:**
- Warmup → active transition: one-time info via `print_status`.
- Runtime degradation (LW failure): `print_status` with reason.
- Routine scaling events (cap hit, targeting fired): DEBUG only.

**Warmup:**
- `bars_available < vol_lookback_days` for a ticker → per-ticker vol is `None`; per-ticker offset is 0 for that ticker.
- `bars_available < corr_lookback_days` portfolio-wide → `corr`/`cov` both `None`; phases 3.5 and 3.7 no-op for that bar.

### 7. Testing strategy

Bottom-up, TDD-driven.

**Unit (`tests/test_risk.py`, new):**
- `realized_vol`: constant series → 0, known series → hand-computed value, short history → None, NaN → insufficient.
- `covariance_matrix`: uncorrelated synthetic → diagonal cov; perfectly correlated → corr all 1; vol floor clamps low-vol ticker; short history → None; output PSD.
- `apply_instrument_diversification_multiplier`: uncorrelated + equal weights → IDM ≈ √N; perfect correlation → IDM = 1; cap binding → scale exactly `cap / raw_idm`; empty weights → empty; single ticker → unchanged; degenerate `w·corr·w` → unchanged.
- `apply_vol_targeting`: below target → unchanged; 2× target → halved; at target → unchanged; zero weights → zero; degenerate `w·cov·w` → unchanged.
- `predict_portfolio_vol`: hand-computed 2-asset case; zero weights → 0.

**Integration (`tests/test_allocator.py`, extend):**
- **Risk-off regression:** `weighting="equal"`, `idm_cap=1.0`, `vol_target=999.0` → bit-identical to current allocator output. Safety net for the refactor.
- Per-ticker vol scaling active: high-vol ticker gets less weight than equal-signal low-vol ticker.
- IDM active: three uncorrelated tickers, weights scale up; sum matches `original * min(√N, cap)`.
- Vol targeting active: synthetic high-vol portfolio scaled down to hit target.
- Phase ordering: IDM pushes position past cap → phase 3.6 pulls it back.
- Warmup: `<60` bars → equal weighting; `60–252` → per-ticker active, IDM/targeting off; `≥253` → all active.

**End-to-end (`tests/test_backtest.py`, extend):**
- Canned-fixture run with default risk config → asserts stable `summary.json` (change-detector).

**Out of scope:**
- Ledoit-Wolf internals (sklearn's responsibility).
- Performance microbenchmarks.
- Cross-platform numerical parity beyond `abs_tol=1e-9`.

## Dependencies

New runtime dependency: `scikit-learn` (for `sklearn.covariance.LedoitWolf`). Added to `pyproject.toml` under the main dep group.

## Implementation order

The writing-plans skill will produce a task-level plan. Suggested ordering:

1. Add `scikit-learn` dependency.
2. `RiskConfig` dataclass + YAML parsing + validation (config-time errors).
3. `src/midas/risk.py` pure functions (TDD, one function at a time).
4. Risk cache build in `Allocator.precompute_signals`.
5. Phase 2 modification: per-ticker offset in softmax.
6. Phase 3.5 and 3.7 integration.
7. Phase 3.6 reuse of existing cap helper.
8. Risk-off regression test (must pass before any default change).
9. Flip defaults to risk-on.
10. End-to-end backtest smoke test.
