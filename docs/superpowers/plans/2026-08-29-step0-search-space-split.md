# Step 0: Search-Space Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The optimizer searches only entry-signal parameters and weights; allocator globals, exit-rule parameters, risk config, `min_cash_pct`, and `forecast_scaling` become policy-owned inputs it never samples.

**Architecture:** `_prepare_names_and_ranges` gains a `search_globals` flag (default `False`) that filters the search space to `EntrySignal` strategies and drops `_global`. `_run_trial` accepts fixed exit parameters and a pre-built `AllocationConstraints` instead of reading them from sampled params. The CLI threads exits and constraints from the strategies file through both optimizer modes, and `write_strategies_yaml` round-trips the policy-owned parts verbatim. The old behavior survives behind `search_globals: true` for experiments.

**Tech Stack:** Python 3.14, Optuna (existing ask/tell loop), pytest, uv.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (section "Step 0 — shrink the search space")

## Global Constraints

- Run tests with `uv run pytest`, lint `uv run ruff check . --fix`, format `uv run ruff format .`, types `uv run mypy src` — all clean before every commit.
- Google Python Style per CLAUDE.md: full type annotations, Google docstrings, no bare `except`, `X | None` unions, module constants in `CAPS_SNAKE_CASE` **without** leading underscore.
- Never put issue/PR numbers in code comments or docstrings — comments state rationale only.
- The golden test (`tests/test_golden.py`) must stay green untouched: the golden scenario runs `backtest`, not `optimize`, and nothing in this plan changes backtest behavior.
- Commits end with the Co-Authored-By and Claude-Session trailers used on this branch (see `git log`).
- Pre-commit hooks run ruff/format/mypy; if ruff-format modifies files during commit, re-stage and re-commit.

---

### Task 1: Search-space filter in `_prepare_names_and_ranges`

**Files:**
- Modify: `src/midas/optimizer.py:635-660` (`_prepare_names_and_ranges`)
- Test: `tests/test_optimizer_search_space.py` (create)

**Interfaces:**
- Consumes: `PARAM_RANGES`, `ALLOCATION_KEY`, `STRATEGY_REGISTRY`, `EntrySignal` (all existing).
- Produces: `_prepare_names_and_ranges(strategy_names, min_cash_pct, n_tickers, search_globals: bool = False)` — with `search_globals=False`, the returned `names` contain only `EntrySignal` strategy names (no exits, no `ALLOCATION_KEY`) and `ranges` has no `ALLOCATION_KEY` entry; with `True`, behavior is exactly today's. Also produces module helper `entry_search_names() -> list[str]` returning the entry-strategy names present in `PARAM_RANGES`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for the entry-only search space (policy-engine Step 0)."""

import pytest

from midas.optimizer import ALLOCATION_KEY, PARAM_RANGES, _prepare_names_and_ranges, entry_search_names
from midas.strategies import STRATEGY_REGISTRY
from midas.strategies.base import EntrySignal


def test_entry_search_names_are_exactly_the_entry_signals() -> None:
    expected = {
        name
        for name in PARAM_RANGES
        if name != ALLOCATION_KEY and issubclass(STRATEGY_REGISTRY[name], EntrySignal)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_optimizer_search_space.py -q --no-cov`
Expected: FAIL / collection error with `ImportError: cannot import name 'entry_search_names'`.

- [ ] **Step 3: Implement**

In `src/midas/optimizer.py`, add near `PARAM_RANGES` (import `EntrySignal` is already available via `midas.strategies.base` import of `ExitRule`; extend that import):

```python
def entry_search_names() -> list[str]:
    """Entry-signal strategies the optimizer may search.

    Exit rules and allocator globals are policy-owned: the operator sets
    them and the search never samples them, so ensembles differ only in
    forecasts and the engine has exactly one sizing/exit configuration.
    """
    return [
        name
        for name in PARAM_RANGES
        if name != ALLOCATION_KEY and issubclass(STRATEGY_REGISTRY[name], EntrySignal)
    ]
```

Rework `_prepare_names_and_ranges`:

```python
def _prepare_names_and_ranges(
    strategy_names: list[str] | None,
    min_cash_pct: float,
    n_tickers: int,
    search_globals: bool = False,
) -> tuple[list[str], dict[str, dict[str, tuple[float, float, float]]]]:
    """Resolve searched strategies and their ranges (shared by optimize/walk-forward).

    By default only entry signals are searched; exit rules and allocator
    globals are policy-owned. ``search_globals=True`` restores the full
    legacy space for experiments.

    Raises:
        ValueError: If no searchable strategies remain, or a policy-owned
            name is requested without ``search_globals``.
    """
    searchable = set(PARAM_RANGES) if search_globals else set(entry_search_names())
    names = strategy_names or sorted(searchable - {ALLOCATION_KEY})
    rejected = [n for n in names if n in PARAM_RANGES and n not in searchable]
    if rejected:
        msg = f"{sorted(rejected)} are policy-owned (exit rules); set search_globals to search them"
        raise ValueError(msg)
    names = [name for name in names if name in searchable]

    if not names:
        msg = "No optimizable strategies found"
        raise ValueError(msg)

    ranges = {name: dict(PARAM_RANGES[name]) for name in names}

    if search_globals:
        names.append(ALLOCATION_KEY)
        equal_weight = (1.0 - min_cash_pct) / max(n_tickers, 1)
        lo = max(round(1.5 * equal_weight, 2), 0.10)
        hi = min(round(5.0 * equal_weight, 2), 0.80)
        if lo >= hi:
            lo, hi = 0.10, 0.80
        step = round((hi - lo) / 8, 2) or 0.01
        ranges.setdefault(ALLOCATION_KEY, dict(PARAM_RANGES[ALLOCATION_KEY]))
        ranges[ALLOCATION_KEY]["max_position_pct"] = (lo, hi, step)

    return names, ranges
```

- [ ] **Step 4: Run the new tests**

Run: `uv run pytest tests/test_optimizer_search_space.py -q --no-cov`
Expected: 4 passed.

- [ ] **Step 5: Run the optimizer suites to find fallout, fix expectations**

Run: `uv run pytest tests/test_optimizer.py tests/test_optimizer_objective.py tests/test_optimizer_forecast_scaling.py tests/test_optimizer_risk_propagation.py -q --no-cov`

Expected failures to fix in this task (they encode the old default):
- Any test asserting `_global`/`max_position_pct` appears in `result.best_params` from a default run: update to assert it does **not** (or pass `search_globals=True` where the test is specifically about globals).
- Tests passing `strategy_names=["MeanReversion"]` keep passing unchanged.

Do not change production code to satisfy an old expectation — update the expectation and note why in the test docstring.

- [ ] **Step 6: Full verification and commit**

Run: `uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest -q`
Expected: all clean, all pass.

```bash
git add src/midas/optimizer.py tests/
git commit -m "Search space defaults to entry signals only

Exit rules and allocator globals become policy-owned: the operator sets
them, the search never samples them. search_globals=True restores the
legacy space for experiments. Roughly triples search density at fixed
budget and is the precondition for well-defined ensemble membership.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013yJkQDopaTBz9S1TvztVnU"
```

---

### Task 2: `_run_trial` takes policy-owned exits and constraints

**Files:**
- Modify: `src/midas/optimizer.py` (`_run_trial`, `_instantiate_strategies`, `_init_worker`, `_trial_worker`, `_wf_init_worker`, `_wf_trial_worker`)
- Test: `tests/test_optimizer_search_space.py` (extend)

**Interfaces:**
- Consumes: Task 1's filtered search space.
- Produces: `_run_trial(strategy_params, portfolio, price_data, start, end, *, min_cash_pct=..., train_pct=..., enable_split=..., risk_config=None, forecast_scaling="none", tax_config=None, track_vol_contribution=True, exit_params: dict[str, dict[str, float]] | None = None, constraints: AllocationConstraints | None = None)`. Semantics: when `constraints` is given it is used as-is (policy-owned) instead of being built from `strategy_params[ALLOCATION_KEY]`; when `exit_params` is given, those exit rules are instantiated with exactly those params and `strategy_params` must contain entries only. When both are `None`, legacy behavior (globals/exits read from `strategy_params`) is unchanged — that is the `search_globals` path and the path all existing tests use.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_optimizer_search_space.py`:

```python
from datetime import date

from conftest import make_price_series

from midas.models import AllocationConstraints, Holding, PortfolioConfig
from midas.optimizer import _run_trial


def _data() -> tuple[PortfolioConfig, dict, date, date]:
    import numpy as np

    rng = np.random.default_rng(5)
    prices = make_price_series(date(2023, 1, 2), 240, 100.0, list(rng.normal(0.0005, 0.012, 240)), name="TEST")
    portfolio = PortfolioConfig(holdings=[Holding(ticker="TEST", shares=10, cost_basis=100.0)], available_cash=2000.0)
    return portfolio, {"TEST": prices}, min(prices.index), max(prices.index)


def test_run_trial_uses_policy_owned_exits_and_constraints() -> None:
    portfolio, price_data, start, end = _data()
    constraints = AllocationConstraints(max_position_pct=0.5, softmax_temperature=0.3, min_buy_delta=0.03)
    entries_only = {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}}
    exits = {"StopLoss": {"loss_threshold": 0.08}}
    metrics, result = _run_trial(
        entries_only, portfolio, price_data, start, end,
        enable_split=False, exit_params=exits, constraints=constraints,
    )
    assert result.starting_value > 0
    # The exit rule ran with the fixed param: any SELL rows attribute to StopLoss, never a searched exit.
    sell_sources = {t.strategy_name for t in result.trades if t.direction.value == "SELL"}
    assert sell_sources <= {"StopLoss"}


def test_run_trial_policy_constraints_change_the_outcome() -> None:
    """Different policy-owned max_position_pct must change sizing — proves the
    constraints argument is honored rather than silently rebuilt from params."""
    portfolio, price_data, start, end = _data()
    entries_only = {"MeanReversion": {"window": 20, "threshold": 0.05, "_weight": 1.0}}
    loose = AllocationConstraints(max_position_pct=0.95, min_buy_delta=0.01)
    tight = AllocationConstraints(max_position_pct=0.10, min_buy_delta=0.01)
    _, r_loose = _run_trial(entries_only, portfolio, price_data, start, end,
                            enable_split=False, exit_params={}, constraints=loose)
    _, r_tight = _run_trial(entries_only, portfolio, price_data, start, end,
                            enable_split=False, exit_params={}, constraints=tight)
    assert r_loose.equity_curve != r_tight.equity_curve
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_optimizer_search_space.py -q --no-cov -k "policy_owned or policy_constraints"`
Expected: FAIL with `TypeError: _run_trial() got an unexpected keyword argument 'exit_params'`.

- [ ] **Step 3: Implement**

In `_run_trial`, replace the constraints/instantiation block:

```python
    global_params = strategy_params.get(ALLOCATION_KEY, {})
    if constraints is None:
        constraints = AllocationConstraints(
            max_position_pct=global_params.get("max_position_pct"),
            min_cash_pct=min_cash_pct,
            softmax_temperature=global_params.get("softmax_temperature", 0.5),
            min_buy_delta=global_params.get("min_buy_delta", 0.02),
            forecast_scaling=forecast_scaling,
        )
    entries, exits = _instantiate_strategies(strategy_params)
    if exit_params is not None:
        if exits:
            msg = "strategy_params contains exit rules but exit_params is also given"
            raise ValueError(msg)
        _, exits = _instantiate_strategies(dict(exit_params))
```

(`_instantiate_strategies` already splits by `EntrySignal`/`ExitRule`; reusing it for the fixed exits keeps one construction path. `dict(exit_params)` guards the caller's mapping from the `_weight` pop inside.)

Thread the two new kwargs through `_init_worker`/`_wf_init_worker` `worker_state` and both workers, mirroring exactly how `tax_config` is threaded today (same three call sites each).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_optimizer_search_space.py -q --no-cov`
Expected: all pass (including Task 1's).

- [ ] **Step 5: Full verification and commit**

Run: `uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest -q`

```bash
git add src/midas/optimizer.py tests/test_optimizer_search_space.py
git commit -m "Trials accept policy-owned exits and constraints

When given, AllocationConstraints is used as-is and exit rules are
instantiated from fixed params; sampled params then carry entries only.
Legacy behavior (globals and exits read from sampled params) is unchanged
when the new arguments are omitted — that is the search_globals path.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013yJkQDopaTBz9S1TvztVnU"
```

---

### Task 3: `optimize`/`walk_forward_optimize` run the split space end to end

**Files:**
- Modify: `src/midas/optimizer.py` (`optimize`, `walk_forward_optimize` signatures and their `_run_trial`/pool wiring)
- Test: `tests/test_optimizer_search_space.py` (extend)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: both functions gain `search_globals: bool = False`, `exit_params: dict[str, dict[str, float]] | None = None`, `constraints: AllocationConstraints | None = None`. With defaults, best_params contain only entry strategies; the final re-run and fold test evaluations use the same fixed exits/constraints as the trials (coherence). With `search_globals=True` and the other two `None`, behavior is exactly pre-plan.

- [ ] **Step 1: Write the failing tests**

```python
def test_optimize_default_searches_entries_only() -> None:
    portfolio, price_data, start, end = _data()
    from midas.optimizer import optimize

    result = optimize(
        portfolio=portfolio, price_data=price_data, start=start, end=end,
        strategy_names=["MeanReversion"], n_trials=4, objective="gross",
        exit_params={"StopLoss": {"loss_threshold": 0.08}},
        constraints=AllocationConstraints(max_position_pct=0.5),
    )
    assert set(result.best_params) == {"MeanReversion"}
    assert ALLOCATION_KEY not in result.best_params


def test_optimize_search_globals_matches_legacy_space() -> None:
    portfolio, price_data, start, end = _data()
    from midas.optimizer import optimize

    result = optimize(
        portfolio=portfolio, price_data=price_data, start=start, end=end,
        strategy_names=["MeanReversion"], n_trials=4, objective="gross",
        search_globals=True,
    )
    assert ALLOCATION_KEY in result.best_params
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_optimizer_search_space.py -q --no-cov -k "entries_only or legacy_space"`
Expected: FAIL with `TypeError` (unexpected keyword `exit_params` / `search_globals`).

- [ ] **Step 3: Implement**

- Add the three kwargs to `optimize` and `walk_forward_optimize`; pass `search_globals` to `_prepare_names_and_ranges`; add `exit_params`/`constraints` to both pools' `initargs` and to every direct `_run_trial` call (final re-run in `optimize`, fold-test evaluation in `walk_forward_optimize`).
- Guard: `search_globals=True` together with non-None `exit_params`/`constraints` raises `ValueError("search_globals searches what exit_params/constraints would fix")`.
- The final-re-run/fold-test calls must pass the same `exit_params`/`constraints` the trials used — a test in Step 1 already covers trials; add one assertion to `test_optimize_default_searches_entries_only`: `assert result.best_result is not None` and that its trades' SELL sources ⊆ {"StopLoss"} (the re-run honored the fixed exits).

- [ ] **Step 4: Run tests, then the full optimizer suites**

Run: `uv run pytest tests/test_optimizer_search_space.py tests/test_optimizer.py tests/test_optimizer_objective.py -q --no-cov`
Fix any remaining old-default expectations as in Task 1 Step 5.

- [ ] **Step 5: Full verification and commit**

Run: `uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest -q`

```bash
git add src/midas/optimizer.py tests/
git commit -m "Optimizer modes thread the policy-owned split end to end

Both optimize and walk_forward_optimize accept fixed exits and
constraints, pass them to every trial, the final re-run, and fold test
evaluations, and search entries only by default. search_globals=True is
mutually exclusive with fixing them.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013yJkQDopaTBz9S1TvztVnU"
```

---

### Task 4: CLI wiring and YAML round-trip

**Files:**
- Modify: `src/midas/cli.py` (the `optimize` command and `_run_optimizer_and_write`)
- Modify: `src/midas/optimizer.py` (`write_strategies_yaml`)
- Test: `tests/test_cli_optimize.py` (extend)

**Interfaces:**
- Consumes: Task 3's kwargs.
- Produces: `midas optimize` splits the loaded strategies file into entry configs (their names → `strategy_names`) and exit configs (→ `exit_params`), builds `constraints` from the file's policy-owned knobs, and passes all three. New flag `--search-globals` (default off) restores the legacy space and forbids `-s` files containing exits (error). `write_strategies_yaml(params, path, *, min_cash_pct, risk_config, forecast_scaling, objective, exit_configs: list[StrategyConfig] | None = None, constraints: AllocationConstraints | None = None)` writes optimized entries plus the policy-owned block (globals from `constraints`, exits appended to `strategies:` verbatim from `exit_configs`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_optimize.py` (reuses its `fake_prices` fixture and `_invoke` helper; extend `_invoke` to accept a `strategies_text` kwarg that writes a `-s` file when given):

```python
STRATS_WITH_EXITS = """\
softmax_temperature: 0.4
min_buy_delta: 0.03
max_position_pct: 0.5
min_cash_pct: 0.05
strategies:
  - name: MeanReversion
    weight: 1.0
    params: {window: 20, threshold: 0.05}
  - name: StopLoss
    params: {loss_threshold: 0.08}
"""


def test_optimize_keeps_exits_and_globals_policy_owned(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross",
                          strategies_text=STRATS_WITH_EXITS)
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out.read_text())
    names = [s["name"] for s in data["strategies"]]
    assert "MeanReversion" in names and "StopLoss" in names
    stop = next(s for s in data["strategies"] if s["name"] == "StopLoss")
    assert stop["params"] == {"loss_threshold": 0.08}  # verbatim, not optimized
    assert data["softmax_temperature"] == 0.4  # policy-owned, round-tripped
    assert data["max_position_pct"] == 0.5


def test_search_globals_flag_restores_legacy_output(tmp_path: Path, fake_prices: None) -> None:
    result, out = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross", "--search-globals")
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(out.read_text())
    assert "softmax_temperature" in data  # searched value emitted, as before


def test_search_globals_rejects_strategy_file_with_exits(tmp_path: Path, fake_prices: None) -> None:
    result, _ = _invoke(tmp_path, PORTFOLIO_NO_TAX, "--objective", "gross", "--search-globals",
                        strategies_text=STRATS_WITH_EXITS)
    assert result.exit_code != 0
    assert "search-globals" in result.output
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli_optimize.py -q --no-cov -k "policy_owned or legacy_output or rejects_strategy"`
Expected: FAIL (`_invoke` has no `strategies_text`; then flag/behavior missing).

- [ ] **Step 3: Implement**

- `_invoke` helper: `strategies_text` kwarg writes `tmp_path / "strats.yaml"` and appends `-s <path>`.
- CLI `optimize`: after `load_strategies`, split `strat_configs` via `STRATEGY_REGISTRY` + `issubclass(..., EntrySignal)` into `entry_names` and `exit_configs`; build `exit_params = {c.name: dict(c.params) for c in exit_configs}`; `constraints = strat_constraints` (already loaded — it carries the file's policy-owned knobs; when no `-s` file, `AllocationConstraints()` defaults and `exit_params = {}` so no exit rules run — the historical no-file default searched exits, the new no-file default runs without them, stated in `--help`). Pass through `_run_optimizer_and_write` to both modes and to `write_strategies_yaml`.
- `--search-globals` flag: passes `search_globals=True`, requires `exit_params` empty (else `click.UsageError`), passes `constraints=None`.
- `write_strategies_yaml`: when `constraints` given, emit its `softmax_temperature`/`min_buy_delta`/`max_position_pct` (skip `None`) instead of values from `params[ALLOCATION_KEY]`; append `exit_configs` to the `strategies:` list verbatim after the optimized entries.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli_optimize.py -q --no-cov`
Expected: all pass, including the pre-existing objective/seed tests.

- [ ] **Step 5: Full verification and commit**

Run: `uv run ruff check . --fix && uv run ruff format . && uv run mypy src && uv run pytest -q`

```bash
git add src/midas/cli.py src/midas/optimizer.py tests/test_cli_optimize.py
git commit -m "CLI keeps exits and allocator globals policy-owned through optimize

The strategies file's exits and globals pass through the search untouched
and round-trip verbatim into the optimized YAML; only entry parameters
and weights are searched. --search-globals restores the legacy space and
refuses files that pin exits.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013yJkQDopaTBz9S1TvztVnU"
```

---

### Task 5: Loud unknown keys in `load_strategies`

**Files:**
- Modify: `src/midas/config.py` (`load_strategies`)
- Test: `tests/test_config.py` (extend)

**Interfaces:**
- Consumes: nothing new.
- Produces: `load_strategies` raises `ValueError` naming the offending keys for any top-level key outside the known set `{"strategies", "min_cash_pct", "softmax_temperature", "min_buy_delta", "max_position_pct", "forecast_scaling", "risk"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_load_strategies_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("strategies:\n  - name: MeanReversion\nsoftmax_temp: 0.4\n")  # typo'd key
    with pytest.raises(ValueError, match="softmax_temp"):
        load_strategies(path)


def test_load_strategies_accepts_all_known_keys(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(
        "strategies:\n  - name: MeanReversion\n"
        "min_cash_pct: 0.05\nsoftmax_temperature: 0.4\nmin_buy_delta: 0.02\n"
        "max_position_pct: 0.5\nforecast_scaling: quantile\nrisk:\n  weighting: equal\n"
    )
    configs, constraints, risk = load_strategies(path)
    assert constraints.softmax_temperature == 0.4
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q --no-cov -k "unknown_top_level or all_known_keys"`
Expected: the unknown-keys test FAILS (`DID NOT RAISE`); the known-keys test passes already.

- [ ] **Step 3: Implement**

At the top of `load_strategies`, after `raw = _load_yaml(path)`:

```python
    known_keys = {
        "strategies", "min_cash_pct", "softmax_temperature", "min_buy_delta",
        "max_position_pct", "forecast_scaling", "risk",
    }
    unknown = set(raw) - known_keys
    if unknown:
        msg = f"strategies file has unrecognized keys {sorted(unknown)}; known keys: {sorted(known_keys)}"
        raise ValueError(msg)
```

Then check nothing else in the repo writes files with extra keys: `grep -rn "objective" example-strategies/ sample-portfolios/ *.yaml` — the `# optimized with --objective` line is a comment, not a key, so it passes; any `*optimized*.yaml` in the repo root is gitignored scratch and may legitimately fail to load afterward (acceptable, single-user).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_config.py -q --no-cov`
Expected: all pass. Also run `uv run pytest -q` — golden and example-strategy loads must stay green (they use only known keys).

- [ ] **Step 5: Full verification and commit**

```bash
git add src/midas/config.py tests/test_config.py
git commit -m "Strategies loader rejects unknown top-level keys

A typo'd policy-owned knob previously fell back to its default silently;
with the search no longer covering these knobs, a silent default is a
silently different engine. Loud errors, per the project's single-user
config philosophy.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013yJkQDopaTBz9S1TvztVnU"
```

---

### Task 6: Docs and example-strategy alignment

**Files:**
- Modify: `docs/cli.md` (optimize section), `docs/architecture.md` (optimizer section)
- Verify (no change expected): `example-strategies/*.yaml`
- Test: none (docs); `uv run pytest -q` as regression gate.

**Interfaces:** none.

- [ ] **Step 1: Update `docs/cli.md`**

In the optimize section, add under the options table:

```markdown
| `--search-globals` | off | Legacy full search space: exit-rule parameters and allocator globals become searchable again. Incompatible with a strategies file that pins exits. |

### What gets searched

By default the optimizer searches **entry-signal parameters and weights
only**. Exit rules, `softmax_temperature`, `min_buy_delta`,
`max_position_pct`, `min_cash_pct`, `forecast_scaling`, and `risk:` are
policy-owned: the strategies file sets them, every trial runs them
unchanged, and the optimized YAML round-trips them verbatim. This
roughly triples search density at a fixed trial budget and keeps exactly
one sizing and exit configuration in play. Without `-s`, all entry
signals are searched and no exit rules run — pin exits in a strategies
file for any run you intend to act on.
```

- [ ] **Step 2: Update `docs/architecture.md`**

Amend the Standard Mode paragraph: replace "Each trial suggests a parameter combination" with "Each trial suggests entry-signal parameters and weights (exit rules and allocator globals are policy-owned and fixed for every trial)".

- [ ] **Step 3: Verify example strategies still load and run**

Run: `for f in example-strategies/*.yaml; do uv run python -c "from pathlib import Path; from midas.config import load_strategies; load_strategies(Path('$f'))" || echo "FAIL $f"; done`
Expected: no FAIL lines (they use only known keys).

- [ ] **Step 4: Full verification and commit**

Run: `uv run pytest -q && uv run ruff check .`

```bash
git add docs/cli.md docs/architecture.md
git commit -m "Document the entry-only search space and --search-globals

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013yJkQDopaTBz9S1TvztVnU"
```

---

## Self-review record

- Spec coverage: Step 0's four requirements — entry-only search (Tasks 1–3), policy-owned pass-through and round-trip (Task 4), loud unknown keys (Task 5), `search_globals` escape hatch incompatible-with-ensemble note (enforced later when `deployment` exists; the flag itself lands in Tasks 1/3/4) — all mapped. The spec's "escape hatch incompatible with `deployment: ensemble`" cannot be enforced until the policy block exists (Plan 3); recorded here so Plan 3 picks it up.
- Types: `exit_params: dict[str, dict[str, float]] | None` and `constraints: AllocationConstraints | None` are named identically in Tasks 2, 3, and 4.
- No placeholders; every code step is concrete.
