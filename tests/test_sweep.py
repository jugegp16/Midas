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


def _decision_report(best_tax: list[float], other_tax: list[float], vary: str = "objective"):
    from midas.policy import Policy
    from midas.sweep import SweepCell, SweepReport

    base = Policy(objective="sharpe", budget=1000, restarts=4)
    variants = [("sharpe", base), ("net", Policy(objective="net", budget=1000, restarts=4))]
    cells = []
    for (name, _p), taxes in zip(variants, (best_tax, other_tax), strict=True):
        for i, tax in enumerate(taxes):
            cells.append(
                SweepCell(
                    variant=name,
                    portfolio="etf",
                    seed_base=42 + i,
                    aggregate_oos_cagr=0.10 + tax,
                    oos_sharpe=1.0,
                    daily_returns=[0.001] * 50,
                    n_folds=13,
                    after_tax_mean=tax,
                )
            )
    return SweepReport(cells=cells, holdout_trimmed_to=None, variants=variants, vary=vary)


def test_decision_recommends_resolved_winner_with_paste_block() -> None:
    # net beats sharpe by ~0.5pt with tiny per-seed scatter: resolved.
    sharpe_tax = [0.020 + 0.001 * (i % 3) for i in range(8)]
    net_tax = [t + 0.005 for t in sharpe_tax]
    report = _decision_report(sharpe_tax, net_tax)
    text = "\n".join(report.decision_lines())
    assert "RESOLVED" in text
    assert "objective: net" in text
    assert "changed: objective: sharpe -> net" in text


def test_decision_refuses_to_crown_a_noise_winner() -> None:
    # Differences well inside per-seed scatter: no fake winner.
    a = [0.020, 0.030, 0.010, 0.025, 0.015, 0.028, 0.012, 0.022]
    b = [0.021, 0.028, 0.012, 0.023, 0.017, 0.026, 0.014, 0.020]
    report = _decision_report(a, b)
    text = "\n".join(report.decision_lines())
    assert "NOT RESOLVABLE" in text
    assert "keep" in text.lower()
    assert "changed:" not in text  # nothing pretends there is a winner


def test_decision_absent_for_single_variant() -> None:
    from midas.policy import Policy
    from midas.sweep import SweepCell, SweepReport

    base = Policy(objective="sharpe")
    report = SweepReport(
        cells=[
            SweepCell(
                variant="base",
                portfolio="p",
                seed_base=42,
                aggregate_oos_cagr=0.1,
                oos_sharpe=1.0,
                daily_returns=[0.001] * 50,
                n_folds=13,
            )
        ],
        holdout_trimmed_to=None,
        variants=[("base", base)],
        vary="objective",
    )
    assert report.decision_lines() == []


STRATEGIES_TEXT = """# my tuned strategies — do not touch
softmax_temperature: 1.0
min_cash_pct: 0.05
strategies:
- name: MeanReversion  # the workhorse
  weight: 1.0
  params:
    window: 10
policy:
  objective: sharpe
  budget: 1000
  restarts: 4
  base_seed: 42
  deployment: ensemble
  cadence_days: 63
  adopt_trigger: none
"""


def test_render_recommended_yaml_is_surgical() -> None:
    """Everything except the changed knob survives byte-for-byte."""
    from midas.policy import Policy
    from midas.sweep import render_recommended_yaml

    winner = Policy(objective="sharpe", budget=1000, restarts=4, deployment="best")
    out = render_recommended_yaml(
        STRATEGIES_TEXT,
        vary="deployment",
        winner=winner,
        winner_name="best",
        verdict="RESOLVED (+0.22% ± 0.09%)",
    )
    assert out.startswith("# recommended by midas sweep")
    assert "deployment: ensemble -> best" in out
    assert "RESOLVED" in out
    body = "\n".join(out.splitlines()[2:]) + "\n"  # strip the two provenance lines
    assert body == STRATEGIES_TEXT.replace("  deployment: ensemble", "  deployment: best")
    assert "# my tuned strategies — do not touch" in out
    assert "# the workhorse" in out


def test_render_recommended_yaml_refuses_missing_knob() -> None:
    import pytest

    from midas.policy import Policy
    from midas.sweep import render_recommended_yaml

    text = STRATEGIES_TEXT.replace("  deployment: ensemble\n", "")
    with pytest.raises(ValueError, match="deployment"):
        render_recommended_yaml(text, vary="deployment", winner=Policy(), winner_name="best", verdict="RESOLVED")


def test_recommendation_object_none_on_noise() -> None:
    a = [0.020, 0.030, 0.010, 0.025, 0.015, 0.028, 0.012, 0.022]
    b = [0.021, 0.028, 0.012, 0.023, 0.017, 0.026, 0.014, 0.020]
    assert _decision_report(a, b).recommendation() is None


def test_recommendation_object_on_resolved() -> None:
    sharpe_tax = [0.020 + 0.001 * (i % 3) for i in range(8)]
    net_tax = [t + 0.005 for t in sharpe_tax]
    rec = _decision_report(sharpe_tax, net_tax).recommendation()
    assert rec is not None
    assert rec.vary == "objective"
    assert rec.winner_name == "net"
    assert rec.old_name == "sharpe"
    assert rec.confidence == "RESOLVED"


def test_three_variants_get_descriptive_verdict_not_noise_claim() -> None:
    """With 3+ variants the paired test never ran — the verdict must say
    that instead of claiming 'within seed noise'."""
    from midas.policy import Policy
    from midas.sweep import SweepCell, SweepReport

    variants = [(n, Policy(objective="sharpe")) for n in ("a", "b", "c")]
    cells = [
        SweepCell(
            variant=name,
            portfolio="p",
            seed_base=42 + i,
            aggregate_oos_cagr=cagr,
            oos_sharpe=1.0,
            daily_returns=[0.001] * 30,
            n_folds=4,
            after_tax_mean=cagr / 4,
        )
        for name, cagr in (("a", 0.13), ("b", 0.10), ("c", 0.09))
        for i in range(3)
    ]
    report = SweepReport(cells=cells, holdout_trimmed_to=None, variants=variants, vary="objective")
    text = "\n".join(report.decision_lines())
    assert "paired test unavailable" in text
    assert "seed noise" not in text
    assert report.recommendation() is None  # no promotion file without a test


def test_resolved_requires_enough_pairs() -> None:
    """Two seeds cannot be RESOLVED — mean > 2·SE is asymptotic (t at n=2
    is 12.7) and zero observed variance gives a 0.00% SE."""
    rec = _decision_report([0.020, 0.020], [0.025, 0.025]).recommendation()
    assert rec is None


def test_pretax_scoreboard_shows_cagr_not_na() -> None:
    from midas.policy import Policy
    from midas.sweep import SweepCell, SweepReport

    variants = [("a", Policy(objective="gross")), ("b", Policy(objective="sharpe"))]
    cells = [
        SweepCell(
            variant=name,
            portfolio="p",
            seed_base=42 + i,
            aggregate_oos_cagr=cagr,
            oos_sharpe=1.0,
            daily_returns=[0.001] * 30,
            n_folds=4,
        )
        for name, cagr in (("a", 0.13), ("b", 0.10))
        for i in range(3)
    ]
    report = SweepReport(cells=cells, holdout_trimmed_to=None, variants=variants, vary="objective")
    text = "\n".join(report.decision_lines())
    assert "n/a" not in text
    assert "+13.00%" in text


def test_render_header_reports_the_files_actual_value() -> None:
    """The 'was X' claim must come from the file, not the runner-up."""
    from midas.policy import Policy
    from midas.sweep import render_recommended_yaml

    text = STRATEGIES_TEXT.replace("  objective: sharpe", "  objective: calmar")
    out = render_recommended_yaml(
        text,
        vary="objective",
        winner=Policy(objective="net", budget=1000, restarts=4),
        winner_name="net",
        verdict="RESOLVED",
    )
    assert "objective: calmar -> net" in out  # the file said calmar, whatever the sweep's runner-up was


def test_lean_verdict_still_recommends() -> None:
    """LEAN (consistent direction, under the t bar) names a winner too."""

    base = [0.020, 0.021, 0.019, 0.022, 0.020, 0.021, 0.018, 0.023]
    winner = [t + (0.002 if i % 2 else 0.0005) for i, t in enumerate(base)]
    rec = _decision_report(base, winner).recommendation()
    assert rec is not None and rec.confidence in ("LEAN", "RESOLVED")


def test_render_swaps_scalar_trigger_for_nested_block() -> None:
    from midas.policy import AdoptTrigger, Policy
    from midas.sweep import render_recommended_yaml

    winner = Policy(objective="sharpe", adopt_trigger=AdoptTrigger(oos_percentile_below=10.0, drawdown_breach=True))
    out = render_recommended_yaml(
        STRATEGIES_TEXT, vary="adopt_trigger", winner=winner, winner_name="default", verdict="RESOLVED"
    )
    assert "adopt_trigger:\n" in out
    assert "oos_percentile_below: 10" in out
    assert "drawdown_breach: true" in out
    assert "adopt_trigger: none" not in out.split("verdict")[1]  # header may name the old value


def test_render_swaps_nested_trigger_for_none() -> None:
    from midas.policy import AdoptTrigger, Policy
    from midas.sweep import render_recommended_yaml

    nested = STRATEGIES_TEXT.replace(
        "  adopt_trigger: none\n",
        "  adopt_trigger:\n    oos_percentile_below: 10\n    drawdown_breach: true\n",
    )
    winner = Policy(objective="sharpe", adopt_trigger=AdoptTrigger.never())
    out = render_recommended_yaml(nested, vary="adopt_trigger", winner=winner, winner_name="none", verdict="RESOLVED")
    assert "adopt_trigger: none" in out
    assert "oos_percentile_below" not in out.split("verdict")[1]  # old nested lines swallowed
