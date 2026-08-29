"""Tests for the entry-only search space (policy-engine Step 0)."""

import pytest

from midas.optimizer import ALLOCATION_KEY, PARAM_RANGES, _prepare_names_and_ranges, entry_search_names
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
