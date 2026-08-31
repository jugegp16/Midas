# Phase 2: Ensemble Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The allocator accepts K entry-parameter members, averages their blended per-ticker scores (flat mean), and runs one policy-level sizing/risk/exit pass on the blend — with an ensemble of one byte-identical to today's engine.

**Architecture:** `Allocator` gains a `members` construction path; `_score_tickers` becomes a per-member `_score_member` plus a blend step that is skipped entirely for K=1 (identity by construction). A new `midas.ensemble` module holds the pure `behavioral_key` fingerprint used later by the fitter's dedupe. A second golden scenario pins the K=2 blend path end to end.

**Tech Stack:** Python 3.14, numpy, pytest, existing golden-test machinery.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (Component 1; Resolved decisions 1 and 4)

## Global Constraints

- Same as phase 1 plan (`2026-08-29-step0-search-space-split.md`): uv commands, Google style, no ticket refs in code, trailers on commits, hooks must pass.
- **Identity property is non-negotiable:** the existing golden (`tests/fixtures/golden/expected/`) must pass untouched at every commit of this phase.
- Blend semantics (spec + this plan's decision): a member that abstains on a ticker contributes **0** to the flat mean when at least one member scored it (consensus requires breadth); a ticker no member scores stays on the hold path exactly as today. `active` = ensemble mean > 0.

---

### Task 1: `behavioral_key` in a new `midas.ensemble` module

**Files:**
- Create: `src/midas/ensemble.py`
- Test: `tests/test_ensemble.py` (create)

**Interfaces:**
- Produces: `behavioral_key(target_series: Sequence[Mapping[str, float]]) -> tuple[tuple[tuple[str, float], ...], ...]` — a hashable fingerprint of a per-bar target-weight series with weights rounded to 4 decimals; equal for series differing only below 5e-5, different at 1e-3.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for ensemble helpers."""

from midas.ensemble import behavioral_key


def test_behavioral_key_collapses_float_jitter() -> None:
    a = [{"AAA": 0.30000001, "BBB": 0.1}, {"AAA": 0.25}]
    b = [{"AAA": 0.30000002, "BBB": 0.1}, {"AAA": 0.25}]
    assert behavioral_key(a) == behavioral_key(b)


def test_behavioral_key_distinguishes_real_differences() -> None:
    a = [{"AAA": 0.300, "BBB": 0.1}]
    b = [{"AAA": 0.301, "BBB": 0.1}]
    assert behavioral_key(a) != behavioral_key(b)


def test_behavioral_key_is_order_insensitive_within_a_bar() -> None:
    assert behavioral_key([{"AAA": 0.3, "BBB": 0.1}]) == behavioral_key([{"BBB": 0.1, "AAA": 0.3}])


def test_behavioral_key_is_hashable_and_bar_order_sensitive() -> None:
    a = [{"AAA": 0.3}, {"AAA": 0.2}]
    b = [{"AAA": 0.2}, {"AAA": 0.3}]
    assert hash(behavioral_key(a)) != hash(behavioral_key(b))
```

- [ ] **Step 2: Run to verify RED** — `uv run pytest tests/test_ensemble.py -q --no-cov`; expect ImportError.

- [ ] **Step 3: Implement**

```python
"""Ensemble deployment helpers.

An ensemble is K entry-parameter members whose blended scores are averaged
before one policy-level sizing/risk/exit pass. This module holds the pure
helpers; the allocator owns the blending and the fitter owns member
selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

BEHAVIORAL_DECIMALS = 4


def behavioral_key(
    target_series: Sequence[Mapping[str, float]],
) -> tuple[tuple[tuple[str, float], ...], ...]:
    """Hashable fingerprint of a per-bar target-weight series.

    Two parameter tuples that differ only in a never-binding parameter
    produce identical targets and must count as one member — dedupe is
    behavioral, not textual. Weights are rounded so float jitter below
    the rounding threshold cannot defeat the collapse.
    """
    return tuple(
        tuple(sorted((ticker, round(weight, BEHAVIORAL_DECIMALS)) for ticker, weight in bar.items()))
        for bar in target_series
    )
```

- [ ] **Step 4: GREEN** — `uv run pytest tests/test_ensemble.py -q --no-cov`; 4 passed.
- [ ] **Step 5: Commit** (message: "behavioral_key: behavioral fingerprint for ensemble member dedupe", with trailers).

---

### Task 2: Allocator members path with K=1 identity

**Files:**
- Modify: `src/midas/allocator.py` (`__init__`, `strategies`, `precompute_signals`, `_score_tickers` → `_score_member` + blend)
- Test: `tests/test_allocator.py` (extend)

**Interfaces:**
- Produces: `Allocator(entries, constraints, n_tickers, risk_config=None, members: Sequence[Sequence[tuple[EntrySignal, float]]] | None = None)`. Rules: `members=None` ⇒ one member built from `entries` (today's behavior, bit for bit); `members` given ⇒ `entries` must be `[]` (ValueError otherwise); `strategies` property returns all members' strategies flattened (feeds warmup and precompute); K=1 via `members=[entries]` is identical to the `entries` path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_allocator.py` (it already defines `FixedScore(EntrySignal)` and helpers for constructing price histories — reuse them; if the existing helper names differ, adapt the construction lines, not the assertions):

```python
class TestEnsembleMembers:
    def test_members_and_entries_are_mutually_exclusive(self) -> None:
        entry = (FixedScore(0.5), 1.0)
        with pytest.raises(ValueError, match="entries"):
            Allocator([entry], AllocationConstraints(), 1, members=[[entry]])

    def test_single_member_is_identical_to_entries_path(self, price_data_two_tickers) -> None:
        entries = [(FixedScore(0.8), 1.0), (FixedScore(0.3), 2.0)]
        solo = Allocator(entries, AllocationConstraints(), 2)
        one_member = Allocator([], AllocationConstraints(), 2, members=[entries])
        tickers = list(price_data_two_tickers)
        a = solo.allocate(tickers, price_data_two_tickers)
        b = one_member.allocate(tickers, price_data_two_tickers)
        assert a.target_weights == b.target_weights
        assert a.blended_scores == b.blended_scores
        assert a.contributions == b.contributions

    def test_two_members_average_blended_scores(self, price_data_two_tickers) -> None:
        """Hand-computed: member A scores 0.8, member B scores 0.4 on every
        ticker -> ensemble blend is 0.6 on every ticker (flat mean)."""
        member_a = [(FixedScore(0.8), 1.0)]
        member_b = [(FixedScore(0.4), 1.0)]
        ens = Allocator(
            [], AllocationConstraints(forecast_scaling="none"), 2, members=[member_a, member_b]
        )
        tickers = list(price_data_two_tickers)
        result = ens.allocate(tickers, price_data_two_tickers)
        for ticker in tickers:
            assert result.blended_scores[ticker] == pytest.approx(0.6)

    def test_abstaining_member_counts_as_zero_when_another_scores(self, price_data_two_tickers) -> None:
        """Member A scores 0.8, member B abstains -> blend 0.4, ticker active."""
        member_a = [(FixedScore(0.8), 1.0)]
        member_b = [(FixedScore(None), 1.0)]
        ens = Allocator(
            [], AllocationConstraints(forecast_scaling="none"), 2, members=[member_a, member_b]
        )
        tickers = list(price_data_two_tickers)
        result = ens.allocate(tickers, price_data_two_tickers)
        for ticker in tickers:
            assert result.blended_scores[ticker] == pytest.approx(0.4)
            assert result.target_weights[ticker] > 0.0

    def test_all_members_abstaining_holds_exactly_like_today(self, price_data_two_tickers) -> None:
        member_a = [(FixedScore(None), 1.0)]
        member_b = [(FixedScore(None), 1.0)]
        ens = Allocator(
            [], AllocationConstraints(forecast_scaling="none"), 2, members=[member_a, member_b]
        )
        tickers = list(price_data_two_tickers)
        result = ens.allocate(tickers, price_data_two_tickers, current_weights={t: 0.2 for t in tickers})
        for ticker in tickers:
            assert result.target_weights[ticker] == pytest.approx(0.2)  # Option A hold
```

(If `tests/test_allocator.py` lacks a `price_data_two_tickers` fixture, add one at module level building two tickers × 40 bars of flat-ish prices via the file's existing history constructor, and make `FixedScore` accept `None` to mean abstain if it doesn't already — check its `score` return.)

- [ ] **Step 2: RED** — `uv run pytest tests/test_allocator.py -q --no-cov -k Ensemble`; expect TypeError (`members` unexpected).

- [ ] **Step 3: Implement**

In `Allocator.__init__`:

```python
    def __init__(
        self,
        entries: list[tuple[EntrySignal, float]],
        constraints: AllocationConstraints,
        n_tickers: int,
        risk_config: RiskConfig | None = None,
        members: Sequence[Sequence[tuple[EntrySignal, float]]] | None = None,
    ) -> None:
        if members is not None and entries:
            msg = "pass entries or members, not both (entries is the single-member convenience path)"
            raise ValueError(msg)
        member_lists = [entries] if members is None else [list(m) for m in members]
        if not member_lists or not any(member_lists):
            member_lists = [entries]  # preserve today's empty-entries behavior
        self._members: list[list[_ScoredEntry]] = [
            [_ScoredEntry(strat, wt) for strat, wt in member] for member in member_lists
        ]
        self._entries = self._members[0]  # single-member fast path & legacy attribute
        ...
```

- `strategies` property: `[e.strategy for member in self._members for e in member]`.
- `precompute_signals`: iterate the flattened list (instances are distinct objects per member; the id-keyed cache already isolates them).
- Rename the body of `_score_tickers` to `_score_member(self, member: list[_ScoredEntry], tickers, price_data, ctx)` (same body, `self._entries` → `member`); new `_score_tickers` :

```python
    def _score_tickers(self, tickers, price_data, ctx):
        results = [self._score_member(member, tickers, price_data, ctx) for member in self._members]
        if len(results) == 1:
            return results[0]  # identity: K=1 is byte-for-byte today's path
        return self._blend_members(tickers, results)

    def _blend_members(self, tickers, results):
        """Flat mean of member blends; abstention counts as zero once any member scores.

        A ticker no member scores keeps the hold path. Contributions average
        per strategy name across the members that produced one (display
        attribution only — the blend uses member blended scores).
        """
        k = len(results)
        contributions: dict[str, dict[str, float]] = {}
        blended: dict[str, float] = {}
        active: list[str] = []
        held: list[str] = []
        for ticker in tickers:
            mean_score = sum(r[1].get(ticker, 0.0) for r in results) / k
            per_name: dict[str, list[float]] = {}
            for r in results:
                for name, val in r[0].get(ticker, {}).items():
                    per_name.setdefault(name, []).append(val)
            contributions[ticker] = {name: sum(vals) / len(vals) for name, vals in per_name.items()}
            blended[ticker] = mean_score
            if mean_score > 0:
                active.append(ticker)
            else:
                held.append(ticker)
        return contributions, blended, active, held
```

- [ ] **Step 4: GREEN + identity sweep** — `uv run pytest tests/test_allocator.py tests/test_golden.py -q --no-cov`; all pass, golden untouched.
- [ ] **Step 5: Full verify + commit** ("Allocator accepts ensemble members; K=1 is the identity", trailers).

---

### Task 3: Engine-level identity and blend tests

**Files:**
- Test: `tests/test_backtest.py` (extend)

**Interfaces:** consumes Task 2's `members` kwarg only.

- [ ] **Step 1: Write the failing test** (fails only if Task 2 broke something — this is the end-to-end pin):

```python
def test_ensemble_of_one_backtest_is_byte_identical() -> None:
    """The identity property, end to end: same trades, same curve."""
    from midas.strategies.mean_reversion import MeanReversion

    prices = make_price_series(date(2024, 1, 2), 200, 100.0, list(np.random.default_rng(3).normal(0, 0.015, 200)))
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=5_000.0)
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)

    def run(allocator: Allocator):
        engine = BacktestEngine(
            allocator=allocator, order_sizer=OrderSizer(), exit_rules=[StopLoss(loss_threshold=0.1)],
            constraints=constraints, enable_split=False,
        )
        return engine.run(portfolio, {"TEST": prices}, min(prices.index), max(prices.index))

    entries = [(MeanReversion(window=20, threshold=0.05), 1.0)]
    solo = run(Allocator(entries, constraints, 1))
    ensemble_entries = [(MeanReversion(window=20, threshold=0.05), 1.0)]
    ens = run(Allocator([], constraints, 1, members=[ensemble_entries]))
    assert solo.equity_curve == ens.equity_curve
    assert [(t.date, t.ticker, t.shares, t.price) for t in solo.trades] == [
        (t.date, t.ticker, t.shares, t.price) for t in ens.trades
    ]


def test_two_member_backtest_differs_from_either_member_alone() -> None:
    """The blend is a real third thing, not a passthrough of one member."""
    from midas.strategies.mean_reversion import MeanReversion
    from midas.strategies.momentum import Momentum

    rng = np.random.default_rng(9)
    prices = make_price_series(date(2024, 1, 2), 200, 100.0, list(rng.normal(0.0005, 0.02, 200)))
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=5_000.0)
    constraints = AllocationConstraints(min_buy_delta=0.01, max_position_pct=0.95)

    member_a = lambda: [(MeanReversion(window=15, threshold=0.03), 1.0)]  # noqa: E731
    member_b = lambda: [(Momentum(window=25, momentum_scale=0.05), 1.0)]  # noqa: E731

    def run(allocator: Allocator):
        engine = BacktestEngine(
            allocator=allocator, order_sizer=OrderSizer(), exit_rules=[],
            constraints=constraints, enable_split=False,
        )
        return engine.run(portfolio, {"TEST": prices}, min(prices.index), max(prices.index))

    blend = run(Allocator([], constraints, 1, members=[member_a(), member_b()]))
    only_a = run(Allocator(member_a(), constraints, 1))
    only_b = run(Allocator(member_b(), constraints, 1))
    assert blend.equity_curve != only_a.equity_curve
    assert blend.equity_curve != only_b.equity_curve
```

- [ ] **Step 2: Run** — `uv run pytest tests/test_backtest.py -q --no-cov -k "ensemble_of_one or two_member_backtest"`; expect PASS if Task 2 is correct (these are pins, written after the unit RED/GREEN cycle of Task 2 — their value is regression, and Step 3 of this task proves they can fail).
- [ ] **Step 3: Prove the pin bites** — temporarily change the blend divisor `/ k` to `/ (k + 1)` in `_blend_members`, run the two tests: the two-member test must FAIL. Revert.
- [ ] **Step 4: Full verify + commit** ("Engine-level ensemble pins: K=1 identity and K=2 blend reality", trailers).

---

### Task 4: K=2 golden scenario

**Files:**
- Modify: `tests/test_golden.py` (parameterize expected dir; add `_run_scenario_k2`, `test_golden_backtest_k2`)
- Create: `tests/fixtures/golden/expected_k2/` (blessed outputs)

**Interfaces:** consumes Task 2. Member A = the fixture's entry configs verbatim; member B = same strategies with `window` +5 bars each and `momentum_scale` 0.05→0.03 (a real, behaviorally distinct second member). Exits and constraints stay the fixture's — policy-owned.

- [ ] **Step 1: Parameterize the comparators**

`_compare_outputs(actual_dir, expected_dir=EXPECTED_DIR)` and thread `expected_dir` through the missing-golden messages. Add `EXPECTED_K2_DIR = FIXTURE_DIR / "expected_k2"`.

- [ ] **Step 2: Add the K2 scenario runner + test**

```python
def _k2_members(strat_configs):
    """Member A: fixture entries verbatim. Member B: same entries, jittered."""
    from midas.cli import _build_strategy
    from midas.strategies.base import EntrySignal

    entry_cfgs = [c for c in strat_configs if isinstance(_build_strategy(c), EntrySignal)]
    member_a = [(_build_strategy(c), c.weight) for c in entry_cfgs]
    member_b = []
    for c in entry_cfgs:
        params = dict(c.params)
        if "window" in params:
            params["window"] = int(params["window"]) + 5
        if "momentum_scale" in params:
            params["momentum_scale"] = 0.03
        jittered = StrategyConfig(name=c.name, params=params, weight=c.weight)
        member_b.append((_build_strategy(jittered), c.weight))
    return [member_a, member_b]


def _run_scenario_k2(out_dir: Path) -> BacktestResult:
    """The golden scenario deployed as a two-member ensemble."""
    port = load_portfolio(FIXTURE_DIR / "portfolio.yaml")
    strat_configs, constraints, risk_config = load_strategies(FIXTURE_DIR / "strategies.yaml")
    _allocator, sizer, exit_rules = _build_components(
        strat_configs, constraints, port.active_ticker_count(), risk_config=risk_config
    )
    allocator = Allocator([], constraints, port.active_ticker_count(), risk_config=risk_config,
                          members=_k2_members(strat_configs))
    engine = BacktestEngine(
        allocator=allocator, order_sizer=sizer, exit_rules=exit_rules, constraints=constraints,
        train_pct=DEFAULT_TRAIN_PCT, enable_split=True, execution_mode="next_open",
        tax_config=port.tax_config,
    )
    result = engine.run(port, _load_price_data(), SIM_START, SIM_END)
    write_backtest_results(result, out_dir)
    return result


def test_golden_backtest_k2(tmp_path: Path, update_golden: bool) -> None:
    result = _run_scenario_k2(tmp_path)
    assert len(result.trades) > 10, "K2 scenario produced too few trades to pin anything"
    if update_golden:
        EXPECTED_K2_DIR.mkdir(parents=True, exist_ok=True)
        for filename in OUTPUT_FILES:
            shutil.copy(tmp_path / filename, EXPECTED_K2_DIR / filename)
        pytest.fail("K2 goldens regenerated; review the diff and re-run without --update-golden.")
    mismatches = _compare_outputs(tmp_path, expected_dir=EXPECTED_K2_DIR)
    assert not mismatches, "K2 golden mismatch:\n" + "\n".join(mismatches[:40])
```

(Imports to add at top: `Allocator`, `StrategyConfig`. The K2 result must also *differ* from the K=1 golden — add `assert _compare_outputs(tmp_path) != []` before the update block, proving the blend changed behavior; if it ever reports no differences the second member has become inert.)

- [ ] **Step 3: Bless** — `uv run pytest tests/test_golden.py --update-golden -q --no-cov` (fails on purpose), then `uv run pytest tests/test_golden.py -q --no-cov` → all pass. Inspect `git diff --stat tests/fixtures/golden/` — ONLY `expected_k2/` files are new; `expected/` untouched (identity property).
- [ ] **Step 4: Full verify + commit** ("K=2 golden scenario pins the ensemble blend path", include expected_k2/ files, note in message that expected/ is untouched, trailers).

---

### Task 5: Architecture doc note

**Files:** `docs/architecture.md`

- [ ] **Step 1:** After the allocator section's Phase 1.5 paragraph, add:

```markdown
**Ensemble members** -- The allocator can hold K entry-parameter sets
("members"). Each member's signals are scored and quantile-normalized
independently; the ensemble's blended score is the flat mean across
members (a member with no opinion contributes zero once any member
scores a ticker). Softmax, position caps, the risk overlay, and exit
rules then run exactly once on the blended scores — members differ only
in forecasts, never in sizing or exits. An ensemble of one is
byte-identical to the single-set path.
```

- [ ] **Step 2:** `uv run pytest -q` full green; commit ("Document ensemble members in the allocator", trailers) and push the branch.

---

## Self-review record

- Spec coverage: Component 1 blend rule (Task 2), abstention semantics decided and pinned (Task 2), identity property (Tasks 2–4), K=2 golden (Task 4), behavioral dedupe helper with 4-decimal rounding (Task 1, spec Resolved decision 4), flat mean (Resolved decision 1). Membership *selection* (stratified top-K/R) is phase 3 (fitter) by the spec's build order.
- Types: `members: Sequence[Sequence[tuple[EntrySignal, float]]] | None` named identically in Tasks 2–4.
- The K=2-differs-from-K=1 assertion in Task 4 guards against the second member silently becoming inert — the failure mode of a blend that averages a member with itself.
