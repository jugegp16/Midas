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


def test_run_sweep_uses_each_portfolios_own_tax_config() -> None:
    """Multi-portfolio sweeps must tax each basket by its own rules.

    Passing None for every portfolio silently reports pre-tax numbers for
    a comparison whose pre-registered criterion is after-tax.
    """
    from datetime import date

    from midas.models import AllocationConstraints, Holding, PortfolioConfig, TaxConfig
    from midas.policy import Policy
    from midas.sweep import run_sweep

    tax_a = TaxConfig(short_term_rate=0.35, long_term_rate=0.15)
    tax_b = TaxConfig(short_term_rate=0.24, long_term_rate=0.15)
    port_a = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=1, cost_basis=1.0)], available_cash=1.0, tax_config=tax_a
    )
    port_b = PortfolioConfig(
        holdings=[Holding(ticker="TEST", shares=1, cost_basis=1.0)], available_cash=1.0, tax_config=tax_b
    )
    seen: dict[str, object] = {}

    def spy_validate(portfolio, price_data, start, end, policy, **kwargs):
        name = next(k for k, p in {"a": port_a, "b": port_b}.items() if p is portfolio)
        seen[name] = kwargs.get("tax_config")
        return _stub_validate_factory({("gross", 42): 0.1})(portfolio, price_data, start, end, policy, **kwargs)

    run_sweep(
        {"a": port_a, "b": port_b},
        {"a": {}, "b": {}},
        date(2016, 1, 4),
        date(2024, 6, 28),
        Policy(objective="gross"),
        "objective",
        [],
        seeds=1,
        constraints=AllocationConstraints(),
        exit_params={},
        validate=spy_validate,
    )
    assert seen == {"a": tax_a, "b": tax_b}


def test_summary_reports_after_tax_when_available() -> None:
    from midas.sweep import SweepCell, SweepReport

    cell = SweepCell(
        variant="base",
        portfolio="p",
        seed_base=42,
        aggregate_oos_cagr=0.1,
        oos_sharpe=1.0,
        daily_returns=[0.001] * 100,
        n_folds=4,
        after_tax_mean=0.08,
    )
    text = "\n".join(SweepReport(cells=[cell], holdout_trimmed_to=None).summary_lines())
    assert "after-tax" in text
    assert "+8.00%" in text


def test_run_sweep_threads_progress_logging_through_validate() -> None:
    """A 24-hour sweep must not be silent between cells.

    The optimizer already emits per-trial progress and the validator
    per-fold lines — but only if run_sweep hands its log_fn down. The
    cell header carries a counter and, after the first cell, a rough
    remaining-time estimate.
    """
    from datetime import date

    from midas.models import AllocationConstraints, Holding, PortfolioConfig
    from midas.policy import Policy
    from midas.sweep import run_sweep

    seen_log_fns: list[object] = []

    def spy_validate(portfolio, price_data, start, end, policy, **kwargs):
        seen_log_fns.append(kwargs.get("log_fn"))
        return _stub_validate_factory({("gross", 42): 0.1, ("gross", 43): 0.1})(
            portfolio, price_data, start, end, policy, **kwargs
        )

    lines: list[str] = []
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=1, cost_basis=1.0)], available_cash=1.0)
    run_sweep(
        {"etf": portfolio},
        {"etf": {}},
        date(2016, 1, 4),
        date(2024, 6, 28),
        Policy(objective="gross"),
        "objective",
        [],
        seeds=2,
        constraints=AllocationConstraints(),
        exit_params={},
        validate=spy_validate,
        log_fn=lines.append,
    )
    assert all(fn is not None for fn in seen_log_fns), "log_fn not passed to validate"
    text = "\n".join(lines)
    assert "cell 1/2" in text
    assert "cell 2/2" in text
    assert "remaining" in text  # estimate appears once a cell has completed


def test_run_sweep_checkpoints_each_cell() -> None:
    """A crash at 95% of a multi-hour sweep must not lose finished cells."""
    from datetime import date

    from midas.models import AllocationConstraints, Holding, PortfolioConfig
    from midas.policy import Policy
    from midas.sweep import run_sweep

    checkpoints: list[dict] = []
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=1, cost_basis=1.0)], available_cash=1.0)
    run_sweep(
        {"etf": portfolio},
        {"etf": {}},
        date(2016, 1, 4),
        date(2024, 6, 28),
        Policy(objective="gross"),
        "objective",
        [],
        seeds=2,
        constraints=AllocationConstraints(),
        exit_params={},
        validate=_stub_validate_factory({("gross", 42): 0.11, ("gross", 43): 0.12}),
        checkpoint_fn=checkpoints.append,
    )
    assert len(checkpoints) == 2
    assert checkpoints[0]["variant"] == "base"
    assert checkpoints[0]["seed_base"] == 42
    assert checkpoints[0]["aggregate_oos_cagr"] == 0.11
    assert "daily_returns" not in checkpoints[0]  # summary numbers, not bulk paths


def test_summary_reports_paired_differences() -> None:
    """Same-seed variants share search trials, so pairing removes search
    luck — the unpaired spread verdict wildly understates power for
    two-variant sweeps and must not be the only readout."""
    from midas.sweep import SweepCell, SweepReport

    cells = []
    for variant, cagrs, taxes in (
        ("best", [0.085, 0.095, 0.114], [0.0186, 0.0213, 0.0247]),
        ("ensemble", [0.082, 0.091, 0.100], [0.0197, 0.0204, 0.0218]),
    ):
        for i, (cagr, tax) in enumerate(zip(cagrs, taxes, strict=True)):
            cells.append(
                SweepCell(
                    variant=variant,
                    portfolio="etf",
                    seed_base=42 + i,
                    aggregate_oos_cagr=cagr,
                    oos_sharpe=1.0,
                    daily_returns=[0.001] * 50,
                    n_folds=13,
                    after_tax_mean=tax,
                )
            )
    text = "\n".join(SweepReport(cells=cells, holdout_trimmed_to=None).summary_lines())
    assert "paired" in text
    assert "3/3" in text  # best wins every same-seed pair on CAGR
    assert "+0.70%" in text  # mean paired CAGR difference (0.3+0.4+1.4)/3 pts


def test_no_paired_line_for_single_variant() -> None:
    from midas.sweep import SweepCell, SweepReport

    cells = [
        SweepCell(
            variant="base",
            portfolio="p",
            seed_base=42,
            aggregate_oos_cagr=0.1,
            oos_sharpe=1.0,
            daily_returns=[0.001] * 50,
            n_folds=13,
        )
    ]
    text = "\n".join(SweepReport(cells=cells, holdout_trimmed_to=None).summary_lines())
    assert "paired" not in text
