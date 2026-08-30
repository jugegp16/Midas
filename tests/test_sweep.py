"""Tests for sweep statistics."""

import numpy as np

from midas.sweep import deflated_sharpe_ratio, stationary_bootstrap


def test_bootstrap_is_deterministic_and_centered() -> None:
    rng = np.random.default_rng(1)
    returns = list(rng.normal(0.0004, 0.01, 500))
    a = stationary_bootstrap(returns, n_resamples=200, seed=7)
    b = stationary_bootstrap(returns, n_resamples=200, seed=7)
    assert a == b
    assert len(a) == 200
    point = (np.prod([1 + r for r in returns]) ** (252 / len(returns))) - 1
    lo, hi = np.percentile(a, 2.5), np.percentile(a, 97.5)
    assert lo < point < hi  # the CI contains the point estimate


def test_bootstrap_degenerate_inputs() -> None:
    assert stationary_bootstrap([]) == []
    assert stationary_bootstrap([0.01]) == []


def test_dsr_decreases_with_trial_count() -> None:
    common = dict(sharpe_variance=0.25, n_obs=504, skew=0.0, kurtosis=3.0)
    few = deflated_sharpe_ratio(1.0, n_trials=10, **common)
    many = deflated_sharpe_ratio(1.0, n_trials=10_000, **common)
    assert 0.0 <= many < few <= 1.0


def test_dsr_zero_sharpe_is_below_half_once_trials_exist() -> None:
    dsr = deflated_sharpe_ratio(0.0, n_trials=100, sharpe_variance=0.25, n_obs=504)
    assert dsr < 0.5


# ---------------------------------------------------------------------------
# Variants and orchestration.
# ---------------------------------------------------------------------------


def test_make_variants_parses_typed_fields() -> None:
    from midas.policy import Policy
    from midas.sweep import make_variants

    base = Policy(objective="gross", budget=4, restarts=1)
    variants = make_variants(base, "objective", ["calmar", "sharpe"])
    assert [name for name, _ in variants] == ["calmar", "sharpe"]
    assert variants[0][1].objective == "calmar" and variants[0][1].budget == 4

    sizes = make_variants(Policy(deployment="ensemble", restarts=1), "ensemble_size", ["4", "8"])
    assert sizes[0][1].ensemble_size == 4

    assert make_variants(base, "objective", []) == [("base", base)]


def test_make_variants_rejects_unknown_field() -> None:
    import pytest

    from midas.policy import Policy
    from midas.sweep import make_variants

    with pytest.raises(ValueError, match="objective"):
        make_variants(Policy(), "not_a_field", ["x"])


def _stub_validate_factory(cagr_by_key):
    """A validate_policy stand-in fabricating Baselines per (objective, base_seed)."""
    from datetime import date

    from midas.baselines import Baselines, FoldRecord

    def stub(portfolio, price_data, start, end, policy, **kwargs):
        cagr = cagr_by_key[(policy.objective, policy.base_seed)]
        fold = FoldRecord(
            fold=1,
            test_start=date(2024, 1, 2),
            test_end=date(2024, 3, 28),
            daily_returns=[cagr / 252] * 60,
            return_raw=cagr / 4,
            return_annualized=cagr,
            drawdown=0.05,
            after_tax_return_raw=None,
            adopted=True,
        )
        return Baselines(
            folds=[fold],
            aggregate_oos_cagr=cagr,
            policy_hash="p" * 64,
            input_hash="i" * 64,
            validation_start=start,
            validation_end=end,
            cadence_days=policy.cadence_days,
        )

    return stub


def test_run_sweep_replicates_and_qualifies() -> None:
    from datetime import date

    from midas.models import AllocationConstraints, Holding, PortfolioConfig
    from midas.policy import Policy
    from midas.sweep import run_sweep

    base = Policy(objective="gross", budget=4, restarts=1, base_seed=42)
    # Seed spread (0.02) dwarfs the variant difference (0.005): not resolvable.
    cagr_by_key = {
        ("gross", 42): 0.10,
        ("gross", 43): 0.12,
        ("calmar", 42): 0.105,
        ("calmar", 43): 0.125,
    }
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=1, cost_basis=1.0)], available_cash=1.0)
    report = run_sweep(
        {"etf": portfolio},
        {"etf": {}},
        date(2016, 1, 4),
        date(2024, 6, 28),
        base,
        "objective",
        ["gross", "calmar"],
        seeds=2,
        constraints=AllocationConstraints(),
        exit_params={},
        validate=_stub_validate_factory(cagr_by_key),
    )
    assert len(report.cells) == 4  # 2 variants x 1 portfolio x 2 seeds
    assert {c.seed_base for c in report.cells} == {42, 43}
    text = "\n".join(report.summary_lines())
    assert "not resolvable" in text
    assert "(on these windows)" in text  # single portfolio -> qualified


def _report_cells(paths_by_seed: dict[int, list[float]], sharpe: float, n_folds: int = 8):
    from midas.sweep import SweepCell, SweepReport

    cells = [
        SweepCell(
            variant="base",
            portfolio="p",
            seed_base=42 + i,
            aggregate_oos_cagr=0.1,
            oos_sharpe=sharpe,
            daily_returns=path,
            n_folds=n_folds,
        )
        for i, path in paths_by_seed.items()
    ]
    return SweepReport(cells=cells, holdout_trimmed_to=None)


def test_dsr_uses_per_period_units() -> None:
    """The DSR must be a probability, not a saturated 0-or-1 indicator.

    Feeding an annualized SR with daily n_obs inflates |z| by ~sqrt(252),
    so every comparison saturates (this scenario lands at ~1e-5). In daily
    units an annualized SR of 0.2 sits just under the expected max of 600
    noise trials at the variance floor, putting the DSR near 0.4 — a
    moderate, meaningful value only the per-period computation produces.
    """
    import re

    rng = np.random.default_rng(3)
    paths = {i: list(rng.normal(0.0, 0.01, 500)) for i in range(3)}
    report = _report_cells(paths, sharpe=0.2, n_folds=200)
    line = report.summary_lines()[0]
    dsr = float(re.search(r"DSR~([0-9.]+)", line).group(1))
    assert 0.25 < dsr < 0.75


def test_bootstrap_ci_respects_seed_disagreement() -> None:
    """The CI must cover what the seeds actually produced, not their blend.

    Concatenating seed paths into one series before resampling narrows the
    interval by sqrt(seeds) and centers it between basins. Resampling each
    seed's own path and pooling the resampled CAGRs keeps both modes.
    """
    import re

    up = [0.002] * 250
    down = [-0.002] * 250
    report = _report_cells({0: up, 1: down}, sharpe=1.0)
    line = report.summary_lines()[0]
    lo, hi = (float(x) / 100 for x in re.search(r"bootstrap95% \[([+-][0-9.]+)%, ([+-][0-9.]+)%\]", line).groups())
    assert hi > 0.60  # the up-seed's ~+65% annualized survives
    assert lo < -0.30  # the down-seed's ~-40% annualized survives


def test_make_variants_can_freeze_the_trigger() -> None:
    """Experiment 1 (hold vs re-fit) needs adopt_trigger as a sweep axis."""
    from midas.policy import Policy
    from midas.sweep import make_variants

    base = Policy(objective="sharpe")
    variants = make_variants(base, "adopt_trigger", ["none", "default"])
    assert [name for name, _p in variants] == ["none", "default"]
    frozen, default = variants[0][1], variants[1][1]
    assert frozen.adopt_trigger.disabled
    assert not default.adopt_trigger.disabled
