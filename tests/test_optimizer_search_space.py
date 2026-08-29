"""Tests for the entry-only search space (policy-engine Step 0)."""

from datetime import date

import numpy as np
import pytest
from conftest import make_price_series

from midas.models import AllocationConstraints, Holding, PortfolioConfig
from midas.optimizer import ALLOCATION_KEY, PARAM_RANGES, _prepare_names_and_ranges, _run_trial, entry_search_names
from midas.strategies import STRATEGY_REGISTRY
from midas.strategies.base import EntrySignal


def test_entry_search_names_are_exactly_the_entry_signals() -> None:
    expected = {
        name for name in PARAM_RANGES if name != ALLOCATION_KEY and issubclass(STRATEGY_REGISTRY[name], EntrySignal)
    }
    assert set(entry_search_names()) == expected
    assert "ChandelierStop" not in entry_search_names()
    assert "MeanReversion" in entry_search_names()


def test_default_search_space_excludes_exits_and_globals() -> None:
    names, ranges = _prepare_names_and_ranges(None, min_cash_pct=0.05, n_tickers=3)
    assert ALLOCATION_KEY not in names
    assert ALLOCATION_KEY not in ranges
    assert "StopLoss" not in names and "ChandelierStop" not in names
    assert set(names) == set(entry_search_names())


def test_requesting_an_exit_by_name_without_search_globals_is_rejected() -> None:
    with pytest.raises(ValueError, match="policy-owned"):
        _prepare_names_and_ranges(["MeanReversion", "ChandelierStop"], min_cash_pct=0.05, n_tickers=3)


def test_search_globals_true_restores_the_old_space() -> None:
    names, ranges = _prepare_names_and_ranges(
        ["MeanReversion", "ChandelierStop"], min_cash_pct=0.05, n_tickers=3, search_globals=True
    )
    assert ALLOCATION_KEY in names
    assert "ChandelierStop" in names
    assert "max_position_pct" in ranges[ALLOCATION_KEY]


# ---------------------------------------------------------------------------
# Trials with policy-owned exits and constraints.
# ---------------------------------------------------------------------------


def _data() -> tuple[PortfolioConfig, dict, date, date]:
    rng = np.random.default_rng(5)
    prices = make_price_series(date(2023, 1, 2), 240, 100.0, list(rng.normal(0.0005, 0.012, 240)), name="TEST")
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=2000.0)
    return portfolio, {"TEST": prices}, min(prices.index), max(prices.index)


def test_run_trial_uses_policy_owned_exits_and_constraints() -> None:
    portfolio, price_data, start, end = _data()
    constraints = AllocationConstraints(max_position_pct=0.5, softmax_temperature=0.3, min_buy_delta=0.03)
    entries_only = {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}}
    exits = {"StopLoss": {"loss_threshold": 0.08}}
    _metrics, result = _run_trial(
        entries_only,
        portfolio,
        price_data,
        start,
        end,
        enable_split=False,
        exit_params=exits,
        constraints=constraints,
    )
    assert result.starting_value > 0
    sell_sources = {t.strategy_name for t in result.trades if t.direction.value == "SELL"}
    assert sell_sources <= {"StopLoss"}


def test_run_trial_policy_constraints_change_the_outcome() -> None:
    """Different policy-owned max_position_pct must change sizing — proves the
    constraints argument is honored rather than silently rebuilt from params."""
    portfolio, price_data, start, end = _data()
    entries_only = {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}}
    loose = AllocationConstraints(max_position_pct=0.95, min_buy_delta=0.01)
    tight = AllocationConstraints(max_position_pct=0.10, min_buy_delta=0.01)
    _, r_loose = _run_trial(
        entries_only, portfolio, price_data, start, end, enable_split=False, exit_params={}, constraints=loose
    )
    _, r_tight = _run_trial(
        entries_only, portfolio, price_data, start, end, enable_split=False, exit_params={}, constraints=tight
    )
    assert r_loose.equity_curve != r_tight.equity_curve


def test_optimize_default_searches_entries_only() -> None:
    portfolio, price_data, start, end = _data()
    from midas.optimizer import optimize

    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=4,
        objective="gross",
        exit_params={"StopLoss": {"loss_threshold": 0.08}},
        constraints=AllocationConstraints(max_position_pct=0.5),
    )
    assert set(result.best_params) == {"MeanReversion"}
    assert ALLOCATION_KEY not in result.best_params
    # The final re-run honored the fixed exits too (coherence with trials).
    assert result.best_result is not None
    sell_sources = {t.strategy_name for t in result.best_result.trades if t.direction.value == "SELL"}
    assert sell_sources <= {"StopLoss"}


def test_optimize_search_globals_matches_legacy_space() -> None:
    portfolio, price_data, start, end = _data()
    from midas.optimizer import optimize

    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=4,
        objective="gross",
        search_globals=True,
    )
    assert ALLOCATION_KEY in result.best_params


def test_search_globals_rejects_fixed_exits_and_constraints() -> None:
    portfolio, price_data, start, end = _data()
    from midas.optimizer import optimize

    with pytest.raises(ValueError, match="search_globals"):
        optimize(
            portfolio=portfolio,
            price_data=price_data,
            start=start,
            end=end,
            strategy_names=["MeanReversion"],
            n_trials=2,
            objective="gross",
            search_globals=True,
            exit_params={"StopLoss": {"loss_threshold": 0.08}},
        )


def test_enqueued_params_run_first_and_flatten_round_trips() -> None:
    from midas.optimizer import flatten_params, optimize

    portfolio, price_data, start, end = _data()
    fixed = {"MeanReversion": {"window": 20.0, "threshold": 0.05, "_weight": 1.0}}
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=3,
        objective="gross",
        exit_params={},
        constraints=AllocationConstraints(),
        enqueue=[flatten_params(fixed)],
        record_top=3,
    )
    # The enqueued combination must appear among the recorded trials.
    assert any(params == fixed for _score, params in result.top_trials)


def test_record_top_caps_and_orders_best_first() -> None:
    from midas.optimizer import optimize

    portfolio, price_data, start, end = _data()
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=6,
        objective="gross",
        exit_params={},
        constraints=AllocationConstraints(),
        record_top=4,
    )
    scores = [score for score, _ in result.top_trials]
    assert len(result.top_trials) == 4
    assert scores == sorted(scores, reverse=True)
    assert result.top_trials[0][0] == result.best_objective_value
