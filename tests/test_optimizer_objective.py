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


def _metrics(
    *,
    twr: float = 0.25,
    sharpe: float = 1.5,
    after_tax_twr: float | None = 0.18,
    max_drawdown: float = 0.10,
    ulcer_index: float = 0.05,
    block_returns: tuple[float, ...] = (0.02,) * 8,
) -> TrialMetrics:
    return TrialMetrics(
        twr=twr,
        bh_return=0.1,
        sharpe_ratio=sharpe,
        after_tax_twr=after_tax_twr,
        max_drawdown=max_drawdown,
        ulcer_index=ulcer_index,
        block_returns=block_returns,
    )


def test_default_objective_is_calmar() -> None:
    assert DEFAULT_OBJECTIVE == "calmar"


def test_gross_scores_raw_twr() -> None:
    assert score_trial(_metrics(), "gross") == 0.25


def test_sharpe_scores_sharpe_ratio() -> None:
    assert score_trial(_metrics(), "sharpe") == 1.5


def test_net_scores_after_tax_twr() -> None:
    assert score_trial(_metrics(), "net") == 0.18


def test_net_without_tax_accounting_raises() -> None:
    with pytest.raises(ValueError, match="tax"):
        score_trial(_metrics(after_tax_twr=None), "net")


def test_calmar_scores_return_over_drawdown() -> None:
    assert score_trial(_metrics(twr=0.30, max_drawdown=0.20), "calmar") == pytest.approx(1.5)


def test_calmar_floors_drawdown_so_flat_runs_stay_finite() -> None:
    # A monotonic train window has ~zero drawdown; an unfloored ratio would be
    # inf and dominate every real candidate.
    assert score_trial(_metrics(twr=0.30, max_drawdown=0.0), "calmar") == pytest.approx(0.30 / 0.01)


def test_calmar_of_degenerate_trial_is_zero() -> None:
    assert score_trial(_metrics(twr=0.0, max_drawdown=0.0), "calmar") == 0.0


def test_unknown_objective_raises_with_allowed_values() -> None:
    with pytest.raises(ValueError, match="'gross', 'sharpe', 'net', 'calmar', 'ulcer', 'robust'"):
        score_trial(_metrics(), "sortino")  # type: ignore[arg-type]


def test_objective_error_lists_allowed_values() -> None:
    assert "'gross', 'sharpe', 'net', 'calmar', 'ulcer', 'robust'" in objective_error("sortino")


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


# ---------------------------------------------------------------------------
# Walk-forward carries after-tax OOS figures so `net` can be judged on its own terms.
# ---------------------------------------------------------------------------


def test_walk_forward_reports_after_tax_oos_per_fold_when_taxed() -> None:
    portfolio, price_data, start, end = _make_data(tax_config=TaxConfig())
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
        objective="gross",
        tax_config=portfolio.tax_config,
    )
    assert result.after_tax_annualized_return is not None
    for fold in result.folds:
        assert fold.after_tax_test_return is not None
        _, test_result = _run_trial(
            fold.best_params,
            portfolio,
            price_data,
            fold.test_start,
            fold.test_end,
            enable_split=False,
            tax_config=portfolio.tax_config,
        )
        assert fold.after_tax_test_return_raw == pytest.approx(test_result.after_tax_twr, abs=1e-4)


def test_walk_forward_after_tax_oos_is_absent_without_tax() -> None:
    portfolio, price_data, start, end = _make_data()
    result = walk_forward_optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=10,
    )
    assert result.after_tax_annualized_return is None
    assert all(fold.after_tax_test_return is None for fold in result.folds)


def test_run_trial_metrics_carry_max_drawdown() -> None:
    portfolio, price_data, start, end = _make_data()
    metrics, result = _run_trial(
        {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}},
        portfolio,
        price_data,
        start,
        end,
        enable_split=False,
    )
    assert metrics.max_drawdown == result.max_drawdown


# ---------------------------------------------------------------------------
# Ulcer (Martin ratio) and block-robustness objectives.
# ---------------------------------------------------------------------------


def test_ulcer_scores_return_over_ulcer_index() -> None:
    assert score_trial(_metrics(twr=0.30, ulcer_index=0.05), "ulcer") == pytest.approx(6.0)


def test_ulcer_floors_index_so_smooth_runs_stay_finite() -> None:
    assert score_trial(_metrics(twr=0.30, ulcer_index=0.0), "ulcer") == pytest.approx(0.30 / 0.005)


def test_robust_scores_lower_quartile_of_block_returns() -> None:
    blocks = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
    import statistics

    expected = statistics.quantiles(blocks, n=4, method="inclusive")[0]
    assert score_trial(_metrics(block_returns=blocks), "robust") == pytest.approx(expected)


def test_robust_quartile_never_extrapolates_below_worst_block() -> None:
    assert score_trial(_metrics(block_returns=(0.10, 0.20)), "robust") >= 0.10


def test_robust_with_no_blocks_scores_zero() -> None:
    assert score_trial(_metrics(block_returns=()), "robust") == 0.0


def test_run_trial_metrics_carry_ulcer_and_blocks() -> None:
    portfolio, price_data, start, end = _make_data()
    metrics, result = _run_trial(
        {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}},
        portfolio,
        price_data,
        start,
        end,
        enable_split=False,
    )
    assert metrics.ulcer_index == result.ulcer_index
    assert metrics.block_returns == tuple(result.block_returns)
    assert len(result.block_returns) == 8


# ---------------------------------------------------------------------------
# Optimizer trials skip report-only vol attribution; the final re-run keeps it.
# ---------------------------------------------------------------------------


def test_run_trial_can_skip_vol_attribution() -> None:
    from midas.models import RiskConfig

    portfolio, price_data, start, end = _make_data()
    args = ({"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}}, portfolio, price_data, start, end)
    kwargs = dict(enable_split=False, risk_config=RiskConfig())
    metrics_off, result_off = _run_trial(*args, track_vol_contribution=False, **kwargs)
    metrics_on, result_on = _run_trial(*args, **kwargs)
    assert result_off.risk_metrics is not None
    assert result_off.risk_metrics.per_ticker_vol_contribution == {}
    assert result_on.risk_metrics is not None and result_on.risk_metrics.per_ticker_vol_contribution
    assert metrics_off.twr == metrics_on.twr


def test_optimize_final_rerun_keeps_vol_attribution() -> None:
    from midas.models import RiskConfig

    portfolio, price_data, start, end = _make_data()
    result = optimize(
        portfolio=portfolio,
        price_data=price_data,
        start=start,
        end=end,
        strategy_names=["MeanReversion"],
        n_trials=2,
        objective="gross",
        risk_config=RiskConfig(),
    )
    assert result.best_result is not None
    assert result.best_result.risk_metrics is not None
    assert result.best_result.risk_metrics.per_ticker_vol_contribution
