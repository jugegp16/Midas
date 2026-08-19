"""Golden end-to-end backtest test (issue #76).

Pins the full pipeline — config → engine → allocator → risk overlays →
tax → results writers — against committed golden outputs. Any change that
shifts results fails here with a readable diff; intended shifts are
blessed with:

    UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py

after which the git diff of tests/fixtures/golden/expected/ is the review
artifact. Golden tests detect change, not correctness — they complement
the hand-computed unit tests, never replace them.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from midas.backtest import DEFAULT_TRAIN_PCT, BacktestEngine
from midas.cli import _build_components
from midas.config import load_portfolio, load_strategies
from midas.models import Direction, HoldingPeriod
from midas.order_sizer import OrderSizer
from midas.results import BacktestResult, write_backtest_results

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "golden"
EXPECTED_DIR = FIXTURE_DIR / "expected"
TICKERS = ("AAA", "BBB", "CCC")
SIM_START = date(2024, 1, 2)
SIM_END = date(2025, 12, 30)
OUTPUT_FILES = ("summary.json", "trades.csv", "equity_curve.csv", "strategy_breakdown.csv")
REL_TOL = 1e-6
ABS_TOL = 1e-9


def _load_price_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        frame = pd.read_csv(FIXTURE_DIR / f"{ticker}.csv", index_col="date", parse_dates=True)
        # Mirror CachedYFinanceProvider's normalization: plain date objects.
        frame.index = pd.to_datetime(frame.index).date
        frame.index.name = "date"
        data[ticker] = frame
    return data


def _run_scenario(out_dir: Path, order_sizer: OrderSizer | None = None) -> BacktestResult:
    """Run the golden scenario exactly as the CLI wires it."""
    port = load_portfolio(FIXTURE_DIR / "portfolio.yaml")
    strat_configs, constraints, risk_config = load_strategies(FIXTURE_DIR / "strategies.yaml")
    n_tickers = sum(1 for holding in port.holdings if holding.shares > 0)
    allocator, default_sizer, exit_rules = _build_components(
        strat_configs,
        constraints,
        n_tickers,
        risk_config=risk_config,
    )
    engine = BacktestEngine(
        allocator=allocator,
        order_sizer=order_sizer or default_sizer,
        exit_rules=exit_rules,
        constraints=constraints,
        train_pct=DEFAULT_TRAIN_PCT,
        enable_split=True,
        execution_mode="next_open",
        tax_config=port.tax_config,
    )
    result = engine.run(port, _load_price_data(), SIM_START, SIM_END)
    write_backtest_results(result, out_dir)
    return result


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _compare_values(expected: Any, actual: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                mismatches.append(f"{path}.{key}: present in only one side")
                continue
            _compare_values(expected[key], actual[key], f"{path}.{key}", mismatches)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            mismatches.append(f"{path}: length {len(expected)} != {len(actual)}")
            return
        for idx, (exp_item, act_item) in enumerate(zip(expected, actual, strict=True)):
            _compare_values(exp_item, act_item, f"{path}[{idx}]", mismatches)
    elif isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            mismatches.append(f"{path}: {expected!r} != {actual!r}")
    elif isinstance(expected, int | float) and isinstance(actual, int | float):
        if not math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            mismatches.append(f"{path}: {expected!r} != {actual!r}")
    elif expected != actual:
        mismatches.append(f"{path}: {expected!r} != {actual!r}")


def _compare_json(expected_path: Path, actual_path: Path) -> list[str]:
    with open(expected_path) as handle:
        expected = json.load(handle)
    with open(actual_path) as handle:
        actual = json.load(handle)
    mismatches: list[str] = []
    _compare_values(expected, actual, expected_path.name, mismatches)
    return mismatches


def _compare_csv(expected_path: Path, actual_path: Path) -> list[str]:
    with open(expected_path, newline="") as handle:
        expected_rows = list(csv.reader(handle))
    with open(actual_path, newline="") as handle:
        actual_rows = list(csv.reader(handle))
    mismatches: list[str] = []
    name = expected_path.name
    if len(expected_rows) != len(actual_rows):
        mismatches.append(f"{name}: row count {len(expected_rows)} != {len(actual_rows)}")
        return mismatches
    for row_idx, (exp_row, act_row) in enumerate(zip(expected_rows, actual_rows, strict=True)):
        if len(exp_row) != len(act_row):
            mismatches.append(f"{name} row {row_idx}: field count {len(exp_row)} != {len(act_row)}")
            continue
        for col_idx, (exp_cell, act_cell) in enumerate(zip(exp_row, act_row, strict=True)):
            if _is_number(exp_cell) and _is_number(act_cell):
                if not math.isclose(float(exp_cell), float(act_cell), rel_tol=REL_TOL, abs_tol=ABS_TOL):
                    mismatches.append(f"{name} row {row_idx} col {col_idx}: {exp_cell} != {act_cell}")
            elif exp_cell != act_cell:
                mismatches.append(f"{name} row {row_idx} col {col_idx}: {exp_cell!r} != {act_cell!r}")
    return mismatches


def _compare_outputs(actual_dir: Path) -> list[str]:
    mismatches: list[str] = []
    for filename in OUTPUT_FILES:
        expected_path = EXPECTED_DIR / filename
        actual_path = actual_dir / filename
        if not expected_path.exists():
            mismatches.append(f"{filename}: no golden file — run UPDATE_GOLDEN=1 to bless")
            continue
        if not actual_path.exists():
            mismatches.append(f"{filename}: not produced by this run")
            continue
        if filename.endswith(".json"):
            mismatches.extend(_compare_json(expected_path, actual_path))
        else:
            mismatches.extend(_compare_csv(expected_path, actual_path))
    return mismatches


def test_golden_backtest(tmp_path: Path) -> None:
    result = _run_scenario(tmp_path)

    # Guard against the scenario silently degrading into "no activity":
    # the goldens are only meaningful if the run exercises the full surface.
    # (Carryforward is deliberately NOT exercised here — the 2024 loss sits
    # under the deductible cap; the tax unit tests own that branch.)
    sells = [t for t in result.trades if t.direction == Direction.SELL]
    assert len(result.trades) > 10, "scenario produced too few trades to pin anything"
    periods = {t.holding_period for t in sells}
    assert HoldingPeriod.SHORT_TERM in periods, "no short-term sells — scenario too quiet"
    assert HoldingPeriod.LONG_TERM in periods, "no long-term sells — scenario too quiet"
    assert result.tax_summary, "tax summary empty — after-tax path not exercised"
    assert result.risk_history is not None
    assert any(s != 1.0 for s in result.risk_history.cppi_scale), "CPPI overlay never engaged"
    assert any(s != 1.0 for s in result.risk_history.vol_target_scale), "vol-target overlay never engaged"
    # Every file the writer emits must be pinned — a fifth output file added
    # later has to show up in OUTPUT_FILES (and get a golden) to pass.
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(OUTPUT_FILES), (
        "write_backtest_results emitted different files than OUTPUT_FILES pins"
    )

    if os.environ.get("UPDATE_GOLDEN"):
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        for filename in OUTPUT_FILES:
            shutil.copy(tmp_path / filename, EXPECTED_DIR / filename)
        pytest.fail(
            "Goldens regenerated from this run. Review the git diff of "
            "tests/fixtures/golden/expected/ and re-run WITHOUT UPDATE_GOLDEN "
            "to confirm. (Failing on purpose so a blessing run is never "
            "mistaken for a passing one.)"
        )

    mismatches = _compare_outputs(tmp_path)
    assert not mismatches, "golden mismatch (bless intended changes with UPDATE_GOLDEN=1):\n" + "\n".join(
        mismatches[:40]
    )


def test_golden_detects_change(tmp_path: Path) -> None:
    """Meta-test: the comparator must actually catch a behavior change.

    Runs the scenario with 10x slippage — if the comparator still reports
    no mismatches, the golden net has a hole in it.
    """
    if os.environ.get("UPDATE_GOLDEN"):
        pytest.skip("blessing run")
    _run_scenario(tmp_path, order_sizer=OrderSizer(default_slippage=0.005))
    mismatches = _compare_outputs(tmp_path)
    assert mismatches, "10x slippage produced byte-equivalent outputs — comparator is not detecting change"
