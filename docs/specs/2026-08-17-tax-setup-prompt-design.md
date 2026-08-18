# Interactive tax setup prompt — design

**Date:** 2026-08-17
**Status:** Approved

## Problem

The `tax:` block in the portfolio YAML is opt-in. Users who never heard of
it silently get no after-tax accounting in `backtest`, no usable
`tax-report`, and only discover the gap at tax time. Worse, the natural
questionnaire question — "what is your short-term capital-gains rate?" —
is one most users cannot answer, though "filing status + rough taxable
income" is easy.

## Decision summary

When `backtest`, `live`, or `tax-report` runs interactively and the
portfolio file has **no `tax:` key at all**, offer a one-time setup
questionnaire. The answers (or an explicit opt-out) are appended to the
portfolio YAML, so the question is asked at most once per portfolio file.
Users who cannot name their rates can instead give filing status and
approximate taxable income; flat rates are derived from a current-year
federal bracket table. The engine stays flat-rate; nothing in
`compute_tax_summary` changes.

## Tri-state `tax:` key

| Portfolio YAML          | `tax_config` | `tax_declared` | Prompt eligible |
|-------------------------|--------------|----------------|-----------------|
| key absent              | `None`       | `False`        | yes             |
| `tax: off` (YAML false) | `None`       | `True`         | no              |
| `tax: {…}` mapping      | `TaxConfig`  | `True`         | no              |

- `PortfolioConfig` gains `tax_declared: bool = False`.
- `load_portfolio` maps a false-y `tax:` value (`off`/`false`/`no`/null)
  to `tax_config=None, tax_declared=True`.
- Any other non-mapping value (e.g. `tax: on`) raises `ValueError` — there
  is no meaningful "enabled with no rates" state.

## Prompt flow (cli.py)

New helper `ensure_tax_config(port, portfolio_path) -> TaxConfig | None`
called by `backtest`, `live`, and `tax-report` in place of reading
`port.tax_config` directly:

1. If `tax_declared` or `tax_config` is set → return `tax_config` (no IO).
2. If not interactive (`sys.stdin.isatty() and sys.stdout.isatty()`,
   wrapped in a patchable `stdin_is_interactive()` helper) → return `None`.
3. `click.confirm("No tax config in <file> — set up after-tax accounting
   now?", default=False)`:
   - **No** → append `tax: off  # after-tax accounting disabled; delete
     this line to be asked again` to the portfolio file, echo what was
     written, return `None`.
   - **Yes** → ask "Enter rates directly, or derive from income?"
     (`click.Choice(["rates", "income"])`, default `income`):
     - **rates**: prompt for `short_term_rate` and `long_term_rate`
       (decimal fractions, defaults 0.37 / 0.20).
     - **income**: prompt for filing status (`single`/`married`) and
       approximate annual taxable income; derive both rates (below).
   - Prompt for `deductible_loss_cap` (default 3000.0) and
     `payment_lag_days` (default 105).
   - Construct `TaxConfig`; on `ValueError` (bad rates, transposed values)
     echo the message and re-ask the rate questions.
   - Append the `tax:` block to the portfolio file with a provenance
     comment; echo; return the config.

Per-command behavior after the helper returns `None`:

- `backtest` / `live`: proceed with after-tax accounting disabled
  (today's behavior).
- `tax-report`: the existing `UsageError` ("no \`tax:\` block") — a
  command that requires rates cannot proceed without them. The error text
  gains a hint that running interactively offers setup.

## Rate derivation (tax.py)

Pure function + constants, unit-testable, no engine changes:

```python
TAX_BRACKET_YEAR = 2026  # update yearly; Rev. Proc. 2025-32
ORDINARY_BRACKETS: dict[str, tuple[tuple[float, float], ...]]  # (upper, rate)
LTCG_BRACKETS: dict[str, tuple[tuple[float, float], ...]]
NIIT_RATE = 0.038
NIIT_THRESHOLD: dict[str, float]  # 200_000 single / 250_000 married

def derive_tax_rates(filing_status: str, taxable_income: float) -> tuple[float, float]
```

- Short-term rate = marginal ordinary bracket at `taxable_income`.
- Long-term rate = LTCG 0/15/20 bracket at `taxable_income`.
- Both get `NIIT_RATE` added when income exceeds the NIIT threshold.
- Clamp `lt = min(lt, st)`: in the narrow band where the LTCG 15% floor
  sits below the top of the 12% ordinary bracket ($49,450–$50,400 single,
  $98,900–$100,800 MFJ), true LT > ST, which `TaxConfig` rejects as a
  transposition. Clamping keeps the dominant ST rate exact and understates
  LT by ≤3 points in a ~$1–2K income sliver.
- 2026 figures (verified against Rev. Proc. 2025-32 / Tax Foundation):
  - Single ordinary: 10% ≤ 12,400; 12% ≤ 50,400; 22% ≤ 105,700;
    24% ≤ 201,775; 32% ≤ 256,225; 35% ≤ 640,600; 37% above.
  - MFJ ordinary: 10% ≤ 24,800; 12% ≤ 100,800; 22% ≤ 211,400;
    24% ≤ 403,550; 32% ≤ 512,450; 35% ≤ 768,700; 37% above.
  - LTCG single: 0% ≤ 49,450; 15% ≤ 545,500; 20% above.
  - LTCG MFJ: 0% ≤ 98,900; 15% ≤ 613,700; 20% above.
- `filing_status` outside the table raises `ValueError`.

Deliberately out of scope: head-of-household / MFS, state taxes,
historical bracket years, storing the income figure (only derived rates
land in the file — no salary in a possibly-committed config).

## Write-back

Append-only, preserving the user's formatting and comments — never
`yaml.dump` the whole file:

```yaml

tax:  # derived from ~$150,000 married, 2026 federal brackets
  short_term_rate: 0.22
  long_term_rate: 0.15
  deductible_loss_cap: 3000.0
  payment_lag_days: 105
```

Direct-entry path uses `# configured interactively` instead of the
derivation comment.

## Testing

- `derive_tax_rates`: bracket edges, NIIT threshold crossing, the LT>ST
  clamp band, unknown filing status.
- `load_portfolio` tri-state: absent / `off` / mapping / invalid scalar.
- CLI flows via `CliRunner` with `input=` and `stdin_is_interactive`
  monkeypatched to `True`: accept+derive (file gains parseable block with
  expected rates), accept+direct-rates, decline (`tax: off` appended; a
  second run does not prompt), invalid rate re-asked, `tax-report` decline
  → `UsageError`, non-interactive (patch `False`) never prompts.
- Existing tests need no changes: `CliRunner` is not a TTY, so the real
  `stdin_is_interactive` returns `False` throughout.

## Docs

`docs/tax-reporting.md`: new "Interactive setup" subsection; caveats gain
"federal brackets only — add your state rate to both fields if
applicable" and a note that the bracket table is for `TAX_BRACKET_YEAR`.
