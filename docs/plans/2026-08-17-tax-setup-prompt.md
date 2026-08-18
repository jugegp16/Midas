# Interactive Tax Setup Prompt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `backtest`, `live`, or `tax-report` runs interactively and the portfolio YAML has no `tax:` key, ask the user setup questions (including deriving rates from filing status + income) and append the result — or an explicit `tax: off` opt-out — to the portfolio file.

**Architecture:** A pure rate-derivation function with 2026 federal bracket constants lives in `tax.py`; `load_portfolio` gains tri-state `tax:` parsing (`absent` / `off` / mapping) exposed via a new `PortfolioConfig.tax_declared` flag; all prompting and file write-back lives in one `cli.py` helper `_ensure_tax_config` called by the three commands. The tax engine is untouched.

**Tech Stack:** Python 3.12, click prompts, pytest + CliRunner (with `_stdin_is_interactive` monkeypatched), uv.

**Spec:** `docs/specs/2026-08-17-tax-setup-prompt-design.md`

## Global Constraints

- Google Python Style Guide (see CLAUDE.md): full type annotations, Google docstrings, `X | None`, no mutable defaults, specific exceptions.
- Module-level constants: `CAPS_SNAKE_CASE`, **no leading underscore**.
- Verify with `uv run pytest`, `uv run ruff check . --fix`, `uv run ruff format .`, `uv run mypy src` before each commit.
- Write-back must APPEND to the portfolio YAML — never rewrite/`yaml.dump` the file (preserves user comments/formatting).
- Only derived rates are written to the file; the income figure is never stored.
- No changes to `compute_tax_summary` or any engine code.

---

### Task 1: Rate derivation from federal brackets

**Files:**
- Modify: `src/midas/tax.py` (constants + function after the imports/docstring, before `AnnualTaxSummary`)
- Test: `tests/test_tax.py` (append a new test class)

**Interfaces:**
- Produces: `derive_tax_rates(filing_status: str, taxable_income: float) -> tuple[float, float]` returning `(short_term_rate, long_term_rate)`; constants `TAX_BRACKET_YEAR` (int), `ORDINARY_BRACKETS`, `LTCG_BRACKETS`, `NIIT_RATE`, `NIIT_THRESHOLD`. Task 3 imports `derive_tax_rates` and `TAX_BRACKET_YEAR` from `midas.tax`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_tax.py`:

```python
class TestDeriveTaxRates:
    """derive_tax_rates maps filing status + income to flat federal rates."""

    def test_low_income_single(self) -> None:
        assert derive_tax_rates("single", 40_000.0) == (0.12, 0.0)

    def test_mid_income_single(self) -> None:
        assert derive_tax_rates("single", 60_000.0) == (0.22, 0.15)

    def test_mid_income_married(self) -> None:
        assert derive_tax_rates("married", 150_000.0) == (0.22, 0.15)

    def test_niit_added_above_threshold(self) -> None:
        # 250k single: 32% ordinary bracket + 3.8% NIIT; LT 15% + 3.8%.
        assert derive_tax_rates("single", 250_000.0) == (0.358, 0.188)

    def test_niit_threshold_is_exclusive(self) -> None:
        assert derive_tax_rates("single", 200_000.0) == (0.24, 0.15)
        assert derive_tax_rates("single", 200_001.0) == (0.278, 0.188)

    def test_lt_clamped_to_st_in_crossover_band(self) -> None:
        # $50K single: ordinary marginal is 12% but LTCG bracket is 15%;
        # TaxConfig rejects lt > st, so lt clamps down to st.
        assert derive_tax_rates("single", 50_000.0) == (0.12, 0.12)

    def test_top_brackets(self) -> None:
        assert derive_tax_rates("married", 1_000_000.0) == (0.408, 0.238)

    def test_unknown_filing_status_raises(self) -> None:
        with pytest.raises(ValueError, match="filing_status"):
            derive_tax_rates("head_of_household", 50_000.0)

    def test_negative_income_raises(self) -> None:
        with pytest.raises(ValueError, match="taxable_income"):
            derive_tax_rates("single", -1.0)
```

Add to the imports at the top of `tests/test_tax.py`: `import pytest` (if absent) and extend the existing `from midas.tax import ...` with `derive_tax_rates`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tax.py::TestDeriveTaxRates -v`
Expected: FAIL — `ImportError: cannot import name 'derive_tax_rates'`

- [ ] **Step 3: Implement** — in `src/midas/tax.py`, after the existing imports:

```python
TAX_BRACKET_YEAR = 2026

# (upper_bound, marginal_rate) pairs, ascending; the last upper bound is
# infinity. Figures from Rev. Proc. 2025-32 (tax year 2026, federal only).
# Update yearly together with TAX_BRACKET_YEAR.
ORDINARY_BRACKETS: dict[str, tuple[tuple[float, float], ...]] = {
    "single": (
        (12_400.0, 0.10),
        (50_400.0, 0.12),
        (105_700.0, 0.22),
        (201_775.0, 0.24),
        (256_225.0, 0.32),
        (640_600.0, 0.35),
        (float("inf"), 0.37),
    ),
    "married": (
        (24_800.0, 0.10),
        (100_800.0, 0.12),
        (211_400.0, 0.22),
        (403_550.0, 0.24),
        (512_450.0, 0.32),
        (768_700.0, 0.35),
        (float("inf"), 0.37),
    ),
}

LTCG_BRACKETS: dict[str, tuple[tuple[float, float], ...]] = {
    "single": ((49_450.0, 0.0), (545_500.0, 0.15), (float("inf"), 0.20)),
    "married": ((98_900.0, 0.0), (613_700.0, 0.15), (float("inf"), 0.20)),
}

NIIT_RATE = 0.038
NIIT_THRESHOLD: dict[str, float] = {"single": 200_000.0, "married": 250_000.0}


def _marginal_rate(brackets: tuple[tuple[float, float], ...], income: float) -> float:
    for upper, rate in brackets:
        if income <= upper:
            return rate
    return brackets[-1][1]


def derive_tax_rates(filing_status: str, taxable_income: float) -> tuple[float, float]:
    """Derive flat (short_term_rate, long_term_rate) from federal brackets.

    Uses the ``TAX_BRACKET_YEAR`` ordinary-income and LTCG bracket tables
    plus NIIT. The short-term rate is the marginal ordinary bracket at
    *taxable_income* (trading gains stack on top of ordinary income); the
    long-term rate is the LTCG 0/15/20 bracket. Both gain ``NIIT_RATE``
    above the NIIT threshold. Federal only — state taxes are on the user.

    The long-term rate is clamped to the short-term rate: in the narrow
    band where the LTCG 15% floor sits below the top of the 12% ordinary
    bracket, true LT > ST, which ``TaxConfig`` rejects as a transposition.

    Args:
        filing_status: ``"single"`` or ``"married"`` (filing jointly).
        taxable_income: Approximate annual taxable income in USD.

    Returns:
        ``(short_term_rate, long_term_rate)`` as decimal fractions,
        rounded to 4 places.

    Raises:
        ValueError: On unknown filing status or negative income.
    """
    if filing_status not in ORDINARY_BRACKETS:
        msg = f"filing_status must be one of {sorted(ORDINARY_BRACKETS)}, got {filing_status!r}"
        raise ValueError(msg)
    if taxable_income < 0:
        msg = f"taxable_income must be >= 0, got {taxable_income}"
        raise ValueError(msg)
    short_term = _marginal_rate(ORDINARY_BRACKETS[filing_status], taxable_income)
    long_term = _marginal_rate(LTCG_BRACKETS[filing_status], taxable_income)
    if taxable_income > NIIT_THRESHOLD[filing_status]:
        short_term += NIIT_RATE
        long_term += NIIT_RATE
    long_term = min(long_term, short_term)
    return round(short_term, 4), round(long_term, 4)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tax.py -v`
Expected: all PASS (existing tax tests included)

- [ ] **Step 5: Lint, typecheck, commit**

```bash
uv run ruff check . --fix && uv run ruff format . && uv run mypy src
git add src/midas/tax.py tests/test_tax.py
git commit -m "Add derive_tax_rates: flat rates from 2026 federal brackets"
```

---

### Task 2: Tri-state `tax:` parsing in load_portfolio

**Files:**
- Modify: `src/midas/models.py` (add field to `PortfolioConfig`)
- Modify: `src/midas/config.py` (the `tax_raw` block inside `load_portfolio`)
- Test: `tests/test_config.py` (extend the tax-block section)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `PortfolioConfig.tax_declared: bool` — `True` whenever the YAML contains a `tax:` key (mapping or `off`), `False` when absent. `tax_config` stays `TaxConfig | None` exactly as today. Task 3 reads both fields.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_config.py` after `test_load_portfolio_without_tax_block`:

```python
def test_load_portfolio_tax_off_declares_without_config(tmp_path: Path) -> None:
    """`tax: off` (YAML false) means: asked and declined — never prompt again."""
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\ntax: off\n")
    portfolio = load_portfolio(path)
    assert portfolio.tax_config is None
    assert portfolio.tax_declared is True


def test_load_portfolio_absent_tax_is_undeclared(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n")
    assert load_portfolio(path).tax_declared is False


def test_load_portfolio_tax_mapping_is_declared(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\n"
        "tax:\n  short_term_rate: 0.30\n  long_term_rate: 0.15\n"
    )
    portfolio = load_portfolio(path)
    assert portfolio.tax_declared is True
    assert portfolio.tax_config is not None


def test_load_portfolio_tax_invalid_scalar_raises(tmp_path: Path) -> None:
    path = tmp_path / "portfolio.yaml"
    path.write_text("portfolio:\n  - ticker: AAPL\n    shares: 100\navailable_cash: 1000\ntax: on\n")
    with pytest.raises(ValueError, match="tax:"):
        load_portfolio(path)
```

(`pytest` is already imported in this file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -k "tax" -v`
Expected: the four new tests FAIL (`tax_declared` attribute missing / `tax: on` currently crashes with `AttributeError`); the two pre-existing tax tests PASS.

- [ ] **Step 3: Implement** — in `src/midas/models.py`, add the field:

```python
    state_file: Path | None = None
    tax_config: TaxConfig | None = None
    tax_declared: bool = False
```

In `src/midas/config.py`, replace the existing `tax_raw` block inside `load_portfolio`:

```python
    tax_raw = raw.get("tax")
    tax_declared = "tax" in raw
    tax: TaxConfig | None = None
    if isinstance(tax_raw, dict):
        tax = TaxConfig(
            short_term_rate=float(tax_raw.get("short_term_rate", 0.37)),
            long_term_rate=float(tax_raw.get("long_term_rate", 0.20)),
            deductible_loss_cap=float(tax_raw.get("deductible_loss_cap", 3000.0)),
            payment_lag_days=int(tax_raw.get("payment_lag_days", 105)),
        )
    elif tax_raw:
        msg = f"tax: must be a mapping of rates or `off`, got {tax_raw!r}"
        raise ValueError(msg)
```

and pass both to the constructor:

```python
        state_file=state_file,
        tax_config=tax,
        tax_declared=tax_declared,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Lint, typecheck, commit**

```bash
uv run ruff check . --fix && uv run ruff format . && uv run mypy src
git add src/midas/models.py src/midas/config.py tests/test_config.py
git commit -m "Tri-state tax: key in portfolio YAML (absent / off / mapping)"
```

---

### Task 3: CLI prompt flow, wiring, and docs

**Files:**
- Modify: `src/midas/cli.py` (imports; new helpers `_stdin_is_interactive`, `_prompt_tax_rates`, `_ensure_tax_config` next to `_resolve_state_path`; call sites in `backtest`, `live`, `tax_report`)
- Modify: `docs/tax-reporting.md` (new "Interactive setup" subsection + caveats)
- Test: Create `tests/test_cli_tax_setup.py`

**Interfaces:**
- Consumes: `derive_tax_rates`, `TAX_BRACKET_YEAR` from `midas.tax` (Task 1); `PortfolioConfig.tax_declared` (Task 2).
- Produces: `_ensure_tax_config(port: PortfolioConfig, portfolio_path: Path) -> TaxConfig | None` — the only entry point commands use; `_stdin_is_interactive() -> bool` — the seam tests monkeypatch.

- [ ] **Step 1: Write the failing tests** — create `tests/test_cli_tax_setup.py`:

```python
"""Tests for the interactive tax-setup prompt (spec 2026-08-17)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

import midas.cli
from midas.cli import cli
from midas.config import load_portfolio
from midas.models import Direction, HoldingPeriod, TradeRecord
from midas.trade_log import append_trade

PORTFOLIO_NO_TAX = "portfolio:\n  - ticker: AAPL\n    shares: 10\navailable_cash: 1000.0\n"


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(midas.cli, "_stdin_is_interactive", lambda: True)


def _seed_log(log: Path) -> None:
    append_trade(
        log,
        TradeRecord(
            date=date(2026, 6, 1),
            ticker="AAPL",
            direction=Direction.SELL,
            shares=10.0,
            price=30.0,
            strategy_name="StopLoss",
            holding_period=HoldingPeriod.SHORT_TERM,
            purchase_date=date(2026, 1, 1),
        ),
        cost_basis=20.0,
        purchase_date=date(2026, 1, 1),
    )


def _invoke_tax_report(tmp_path: Path, portfolio: Path, user_input: str):
    log = tmp_path / "trades.csv"
    if not log.exists():
        _seed_log(log)
    return CliRunner().invoke(
        cli,
        [
            "tax-report",
            "--portfolio",
            str(portfolio),
            "--from-trades",
            str(log),
            "--year",
            "2026",
            "--output",
            str(tmp_path / "out.csv"),
        ],
        input=user_input,
    )


def test_accept_and_derive_from_income(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    # confirm=y, mode=default (income), status=single, income, cap/lag defaults
    result = _invoke_tax_report(tmp_path, portfolio, "y\n\nsingle\n60000\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Derived rates" in result.output

    reloaded = load_portfolio(portfolio)
    assert reloaded.tax_config is not None
    assert reloaded.tax_config.short_term_rate == 0.22
    assert reloaded.tax_config.long_term_rate == 0.15
    assert "2026 federal brackets" in portfolio.read_text()


def test_accept_with_direct_rates(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    result = _invoke_tax_report(tmp_path, portfolio, "y\nrates\n0.30\n0.15\n\n\n")
    assert result.exit_code == 0, result.output

    reloaded = load_portfolio(portfolio)
    assert reloaded.tax_config is not None
    assert reloaded.tax_config.short_term_rate == 0.30
    assert reloaded.tax_config.long_term_rate == 0.15


def test_invalid_rates_reprompted(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    # First attempt transposes the rates (lt > st) -> TaxConfig rejects -> re-asked.
    result = _invoke_tax_report(tmp_path, portfolio, "y\nrates\n0.10\n0.20\n\n\nrates\n0.30\n0.15\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Invalid tax config" in result.stderr
    reloaded = load_portfolio(portfolio)
    assert reloaded.tax_config is not None
    assert reloaded.tax_config.short_term_rate == 0.30


def test_decline_writes_tax_off_and_never_asks_again(tmp_path: Path, interactive: None) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    first = _invoke_tax_report(tmp_path, portfolio, "n\n")
    assert first.exit_code != 0  # tax-report cannot run without rates
    assert "tax: off" in portfolio.read_text()

    second = _invoke_tax_report(tmp_path, portfolio, "")
    assert second.exit_code != 0
    assert "set up after-tax accounting" not in second.output


def test_non_interactive_never_prompts(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.yaml"
    portfolio.write_text(PORTFOLIO_NO_TAX)

    result = _invoke_tax_report(tmp_path, portfolio, "")
    assert result.exit_code != 0
    assert "set up after-tax accounting" not in result.output
    assert portfolio.read_text() == PORTFOLIO_NO_TAX
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_tax_setup.py -v`
Expected: FAIL — `AttributeError: module 'midas.cli' has no attribute '_stdin_is_interactive'` (fixture) / prompts never happen.

- [ ] **Step 3: Implement the helpers** — in `src/midas/cli.py`:

Add `import sys` to the stdlib imports. Extend the models import with `TaxConfig` and the tax import with `TAX_BRACKET_YEAR, derive_tax_rates` (keeping alphabetical order):

```python
from midas.tax import TAX_BRACKET_YEAR, AnnualTaxSummary, compute_tax_summary, derive_tax_rates
```

After `_resolve_state_path`, add:

```python
def _stdin_is_interactive() -> bool:
    """Whether both stdin and stdout are TTYs, i.e. it is safe to prompt."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_tax_rates() -> tuple[float, float, str]:
    """Ask for capital-gains rates, directly or derived from income.

    Returns:
        ``(short_term_rate, long_term_rate, provenance)`` where provenance
        is the YAML comment describing where the numbers came from.
    """
    mode = click.prompt(
        "Enter rates directly, or derive them from your income",
        type=click.Choice(["rates", "income"]),
        default="income",
    )
    if mode == "rates":
        short_term = click.prompt("Short-term capital-gains rate (decimal fraction)", type=float, default=0.37)
        long_term = click.prompt("Long-term capital-gains rate (decimal fraction)", type=float, default=0.20)
        return short_term, long_term, "configured interactively"
    status = click.prompt("Filing status", type=click.Choice(["single", "married"]), default="single")
    income = click.prompt("Approximate annual taxable income (USD)", type=float)
    short_term, long_term = derive_tax_rates(status, income)
    click.echo(
        f"Derived rates: short-term {short_term:.1%}, long-term {long_term:.1%} "
        f"({TAX_BRACKET_YEAR} federal brackets; add state taxes to both if applicable)"
    )
    return short_term, long_term, f"derived from ~${income:,.0f} {status}, {TAX_BRACKET_YEAR} federal brackets"


def _ensure_tax_config(port: PortfolioConfig, portfolio_path: Path) -> TaxConfig | None:
    """Return the portfolio's tax config, offering one-time interactive setup.

    When the portfolio file has no ``tax:`` key at all and the session is
    interactive, ask whether to set up after-tax accounting and append the
    outcome to the file: a ``tax:`` block on yes, ``tax: off`` on no. A
    file that declares ``tax:`` either way is never prompted about. The
    income figure used for derivation is never written to the file.
    """
    if port.tax_declared:
        return port.tax_config
    if not _stdin_is_interactive():
        return None
    if not click.confirm(
        f"No tax config in {portfolio_path} — set up after-tax accounting now?",
        default=False,
    ):
        with open(portfolio_path, "a", encoding="utf-8") as handle:
            handle.write("\ntax: off  # after-tax accounting disabled; delete this line to be asked again\n")
        click.echo(f"Wrote `tax: off` to {portfolio_path}.")
        return None
    while True:
        short_term, long_term, provenance = _prompt_tax_rates()
        cap = click.prompt("Deductible loss cap (USD/year)", type=float, default=3000.0)
        lag = click.prompt("Payment lag (days from year-end to payment)", type=int, default=105)
        try:
            tax_config = TaxConfig(
                short_term_rate=short_term,
                long_term_rate=long_term,
                deductible_loss_cap=cap,
                payment_lag_days=lag,
            )
            break
        except ValueError as exc:
            click.echo(f"Invalid tax config: {exc}. Let's try again.", err=True)
    block = (
        f"\ntax:  # {provenance}\n"
        f"  short_term_rate: {tax_config.short_term_rate}\n"
        f"  long_term_rate: {tax_config.long_term_rate}\n"
        f"  deductible_loss_cap: {tax_config.deductible_loss_cap}\n"
        f"  payment_lag_days: {tax_config.payment_lag_days}\n"
    )
    with open(portfolio_path, "a", encoding="utf-8") as handle:
        handle.write(block)
    click.echo(f"Wrote tax config to {portfolio_path}.")
    return tax_config
```

- [ ] **Step 4: Wire the three commands**

`backtest` — the command already binds `port = load_portfolio(Path(portfolio))`; change the engine kwarg:

```python
        tax_config=_ensure_tax_config(port, Path(portfolio)),
```

`live` — after `state_path = _resolve_state_path(port, portfolio_path)`:

```python
    # Setup opportunity: live's trade log is what tax-report consumes later.
    _ensure_tax_config(port, portfolio_path)
```

`tax_report` — replace the `tax_config = port.tax_config` line with:

```python
    tax_config = _ensure_tax_config(port, portfolio_path)
```

and extend the error message so non-interactive users know setup exists:

```python
        msg = (
            "portfolio file has no `tax:` block; tax-report requires configured rates "
            "(short_term_rate, long_term_rate). Run interactively to be walked through "
            "setup, or see docs/tax-reporting.md."
        )
```

Note `tax_report` currently loads the portfolio *after* the window validation and binds `portfolio_path` — keep that; only the tax-config lines change.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_tax_setup.py tests/test_cli_tax_report.py -v`
Expected: all PASS (the existing `test_tax_report_requires_tax_block` message assertion still matches — the text still contains ``no `tax:` block``).

- [ ] **Step 6: Update docs** — in `docs/tax-reporting.md`, after the "## Configuring rates" YAML example, add:

```markdown
### Interactive setup

If the portfolio file has no `tax:` key at all, `midas backtest`,
`midas live`, and `midas tax-report` offer a one-time setup when run in a
terminal: answer a few questions and the block is appended to the file.
You can enter the two rates directly, or give filing status
(single / married filing jointly) and approximate taxable income to have
flat rates derived from the current-year federal brackets (NIIT
included). Declining writes `tax: off`, which silences the question;
delete that line to be asked again. Non-interactive runs (pipes, CI)
never prompt. Only the derived rates are written — income is not stored.
```

and append to the Caveats list:

```markdown
- **Derived rates are federal-only, for one tax year.** The bracket table
  (`TAX_BRACKET_YEAR` in `tax.py`) is federal; add your state's rate to
  both fields by hand if applicable, and expect the table to be updated
  yearly.
```

- [ ] **Step 7: Full verification and commit**

```bash
uv run pytest
uv run ruff check . --fix && uv run ruff format . && uv run mypy src
git add src/midas/cli.py docs/tax-reporting.md tests/test_cli_tax_setup.py
git commit -m "Interactive tax setup prompt in backtest/live/tax-report"
```

Expected: full suite passes (no existing test regressions — `CliRunner` is not a TTY, so the real `_stdin_is_interactive` gate keeps every pre-existing CLI test on the non-prompting path).
