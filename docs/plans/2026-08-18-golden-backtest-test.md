# Golden End-to-End Backtest Test — Plan (#76)

**Spec:** issue #76 (approved 2026-08-17). Goal: pin the full pipeline
(config → engine → allocator → risk → tax → results writers) against
committed golden outputs, with a blessed `UPDATE_GOLDEN=1` regeneration path.

## Files

- Create `tests/fixtures/golden/generate_fixtures.py` — seeded generator
  (numpy `default_rng(76)`), run manually via
  `uv run python tests/fixtures/golden/generate_fixtures.py`; writes the
  three price CSVs below. Committed so fixtures are reproducible, but the
  CSVs are the source of truth (regenerating is itself a golden-diff event).
- Create `tests/fixtures/golden/{AAA,BBB,CCC}.csv` — ~2023-01-03..2025-12-30
  business days, lowercase `date,open,high,low,close,volume`, real H/L
  spreads (not degenerate). Personalities: AAA uptrend with two sharp dip
  windows (fires mean-reversion/RSI/stops), BBB choppy sideways, CCC
  down-year then strong uptrend (fires momentum).
- Create `tests/fixtures/golden/portfolio.yaml` — 3 holdings with cost
  bases, `available_cash`, biweekly `cash_infusion`, `trading_restrictions`
  (round_trip_days 10), full `tax:` block.
- Create `tests/fixtures/golden/strategies.yaml` — entries (Momentum,
  MeanReversion, RSIOversold) + exits (StopLoss, ProfitTaking) with pinned
  params, allocation knobs, full `risk:` block (inverse_vol, vol_target,
  drawdown pair).
- Create `tests/fixtures/golden/expected/` — blessed `summary.json`,
  `trades.csv`, `equity_curve.csv`, `strategy_breakdown.csv`.
- Create `tests/test_golden.py`.

## Test design

1. Load CSVs → DataFrames (parse `date` index); load fixture YAMLs via
   `load_portfolio` / `load_strategies`; build components via
   `midas.cli._build_components` (production wiring, not a re-implementation).
2. `BacktestEngine(..., execution_mode="next_open", tax_config=port.tax_config)`
   with CLI-default split; `run(port, price_data, 2024-01-02, 2025-12-30)`
   (2023 serves as warmup history in the frames).
3. `write_backtest_results(result, tmp_path)`; then either:
   - `UPDATE_GOLDEN=1` → copy outputs over `expected/`, fail-safe message; or
   - compare each output to `expected/` — JSON recursively, CSVs row/cell —
     numeric cells via `math.isclose(rel_tol=1e-6, abs_tol=1e-9)`, strings exact.
     Mismatches report path/row/col, both values.
4. `test_golden_detects_change`: monkeypatch `order_sizer.DEFAULT_SLIPPAGE`
   ×10, re-run, assert the comparator REPORTS mismatches — proves the net
   catches fish (DoD item).

## Steps

- [ ] Generator script + run it; eyeball CSVs (dips/trends present)
- [ ] Fixture YAMLs
- [ ] `test_golden.py` harness with comparator + UPDATE_GOLDEN path
- [ ] Bless goldens (UPDATE_GOLDEN=1), re-run plain → PASS; run twice → deterministic
- [ ] Perturbation meta-test passes
- [ ] Assert outputs are non-trivial (trades occurred in both ST/LT, tax_summary non-empty) so the scenario can't silently degrade to "no activity"
- [ ] `uv run pytest`, ruff, mypy; commit
