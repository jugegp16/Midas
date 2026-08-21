"""Tests for the optimizer's selectable objective."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml
from conftest import make_price_series

from midas.models import DEFAULT_OBJECTIVE, Holding, PortfolioConfig, TaxConfig, objective_error
from midas.optimizer import (
    TrialMetrics,
    _run_trial,
    optimize,
    score_trial,
    train_window_end,
    walk_forward_optimize,
    write_strategies_yaml,
)


def _metrics(*, twr: float = 0.25, sharpe: float = 1.5, after_tax_twr: float | None = 0.18) -> TrialMetrics:
    return TrialMetrics(twr=twr, bh_return=0.1, sharpe_ratio=sharpe, after_tax_twr=after_tax_twr)


def test_default_objective_is_sharpe() -> None:
    assert DEFAULT_OBJECTIVE == "sharpe"


def test_gross_scores_raw_twr() -> None:
    assert score_trial(_metrics(), "gross") == 0.25


def test_sharpe_scores_sharpe_ratio() -> None:
    assert score_trial(_metrics(), "sharpe") == 1.5


def test_net_scores_after_tax_twr() -> None:
    assert score_trial(_metrics(), "net") == 0.18


def test_net_without_tax_accounting_raises() -> None:
    with pytest.raises(ValueError, match="tax"):
        score_trial(_metrics(after_tax_twr=None), "net")


def test_unknown_objective_raises_with_allowed_values() -> None:
    with pytest.raises(ValueError, match="'gross', 'sharpe', 'net'"):
        score_trial(_metrics(), "calmar")  # type: ignore[arg-type]


def test_objective_error_lists_allowed_values() -> None:
    assert "'gross', 'sharpe', 'net'" in objective_error("calmar")


# ---------------------------------------------------------------------------
# Trial plumbing: every objective is scored on a train-only window.
# ---------------------------------------------------------------------------


def _make_data(*, tax_config: TaxConfig | None = None) -> tuple[PortfolioConfig, dict[str, pd.DataFrame], date, date]:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)],
        available_cash=2000.0,
        tax_config=tax_config,
    )
    returns = [0.005] * 100 + [-0.003] * 100 + [0.004] * 100 + [0.002] * 100
    prices = make_price_series(date(2023, 1, 2), 400, 100.0, returns, name="TEST")
    return portfolio, {"TEST": prices}, min(prices.index), max(prices.index)


def test_run_trial_returns_metrics_without_after_tax_when_untaxed() -> None:
    portfolio, price_data, start, end = _make_data()
    metrics, result = _run_trial(
        {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}},
        portfolio,
        price_data,
        start,
        end,
        enable_split=False,
    )
    assert metrics.twr == result.twr
    assert metrics.sharpe_ratio == result.sharpe_ratio
    assert metrics.after_tax_twr is None


def test_run_trial_reports_after_tax_twr_when_taxed() -> None:
    portfolio, price_data, start, end = _make_data(tax_config=TaxConfig())
    metrics, result = _run_trial(
        {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}},
        portfolio,
        price_data,
        start,
        end,
        enable_split=False,
        tax_config=portfolio.tax_config,
    )
    assert metrics.after_tax_twr == result.after_tax_twr
    assert metrics.after_tax_twr is not None


def test_train_window_end_matches_engine_split() -> None:
    days = [date(2024, 1, 1) + pd.Timedelta(days=i).to_pytimedelta() for i in range(10)]
    # Engine: split_date = days[int(10 * 0.7)] = days[7]; train = days < split_date.
    assert train_window_end(days, 0.7) == days[6]
    assert train_window_end(days, 1.0) == days[-1]


def test_optimize_scores_best_trial_on_train_window_only() -> None:
    """No leakage: best value must equal the sharpe of a direct train-only run."""
    portfolio, price_data, start, end = _make_data()
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=4,
        objective="sharpe",
    )
    assert result.objective == "sharpe"
    train_end = train_window_end(sorted(price_data["TEST"].index), 0.7)
    metrics, _ = _run_trial(result.best_params, portfolio, price_data, start, train_end, enable_split=False)
    assert result.best_objective_value == pytest.approx(metrics.sharpe_ratio, abs=1e-4)
    # train_return stays a return, independent of the objective.
    assert result.best_train_return == pytest.approx(metrics.twr, abs=1e-4)


def test_optimize_gross_objective_value_is_train_return() -> None:
    portfolio, price_data, start, end = _make_data()
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=4,
        objective="gross",
    )
    assert result.best_objective_value == pytest.approx(result.best_train_return, abs=1e-4)


def test_optimize_net_requires_tax_config() -> None:
    portfolio, price_data, start, end = _make_data()
    with pytest.raises(ValueError, match="tax"):
        optimize(
            portfolio=portfolio,
            price_data=price_data,
            start=start,
            end=end,
            strategy_names=["MeanReversion"],
            n_trials=2,
            objective="net",
        )


def test_optimize_net_scores_after_tax_return() -> None:
    portfolio, price_data, start, end = _make_data(tax_config=TaxConfig())
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=4,
        objective="net",
        tax_config=portfolio.tax_config,
    )
    train_end = train_window_end(sorted(price_data["TEST"].index), 0.7)
    metrics, _ = _run_trial(
        result.best_params, portfolio, price_data, start, train_end, enable_split=False, tax_config=portfolio.tax_config
    )
    assert result.best_objective_value == pytest.approx(metrics.after_tax_twr, abs=1e-4)


def test_walk_forward_fold_keeps_train_return_as_return_under_sharpe() -> None:
    portfolio, price_data, start, end = _make_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
        objective="sharpe",
    )
    assert result.objective == "sharpe"
    fold = result.folds[0]
    metrics, _ = _run_trial(
        fold.best_params, portfolio, price_data, fold.train_start, fold.train_end, enable_split=False
    )
    assert fold.train_return_raw == pytest.approx(metrics.twr, abs=1e-4)
    assert fold.objective_value == pytest.approx(metrics.sharpe_ratio, abs=1e-4)


def test_walk_forward_net_requires_tax_config() -> None:
    portfolio, price_data, start, end = _make_data()
    with pytest.raises(ValueError, match="tax"):
        walk_forward_optimize(
            portfolio=portfolio,
            price_data=price_data,
            start=start,
            end=end,
            strategy_names=["MeanReversion"],
            n_trials=10,
            objective="net",
        )


def test_write_strategies_yaml_records_objective(tmp_path: Path) -> None:
    """The optimized YAML says which objective produced it, as a comment, not a key."""
    out = tmp_path / "strats.yaml"
    write_strategies_yaml({"MeanReversion": {"window": 20.0, "threshold": 0.1}}, str(out), objective="net")
    text = out.read_text()
    assert text.splitlines()[0] == "# optimized with --objective net"
    assert "objective" not in yaml.safe_load(text)


def test_train_window_end_rejects_a_split_with_no_training_days() -> None:
    """A tiny train_pct would wrap to trading_days[-1] and score trials on the full range."""
    days = [date(2024, 1, 1) + pd.Timedelta(days=i).to_pytimedelta() for i in range(10)]
    with pytest.raises(ValueError, match="no training days"):
        train_window_end(days, 0.01)


def test_train_window_end_rejects_empty_trading_days() -> None:
    with pytest.raises(ValueError, match="trading days"):
        train_window_end([], 0.7)


def test_degenerate_taxed_trial_scores_zero_under_net_instead_of_aborting() -> None:
    """A trial with nothing to invest must score 0.0 like gross/sharpe do, not kill the study."""
    portfolio, price_data, start, end = _make_data(tax_config=TaxConfig())
    empty = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=0, cost_basis=100.0)],
        available_cash=0.0,
        tax_config=portfolio.tax_config,
    )
    metrics, _ = _run_trial(
        {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}},
        empty,
        price_data,
        start,
        end,
        enable_split=False,
        tax_config=empty.tax_config,
    )
    assert score_trial(metrics, "net") == 0.0
