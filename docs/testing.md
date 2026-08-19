# Testing

```
uv run pytest                    # full suite
uv run pytest tests/test_x.py    # one module
```

The suite is offline — no test may hit the network. Provider-shaped data
comes from `tests/conftest.py` helpers or committed fixtures.

## Golden end-to-end backtest test

`tests/test_golden.py` pins the entire pipeline (config → engine →
allocator → risk overlays → tax → results writers) against committed
outputs in `tests/fixtures/golden/expected/`. Any change that shifts
backtest results fails this test with per-cell mismatch lines:

```
summary.json.final_value: 72502.75 != 71980.11
trades.csv row 12 col 4: 30.12 != 30.09
```

Golden tests detect **change**, not correctness — a stable bug would be
enshrined. Correctness is owned by the unit tests; the golden test makes
every result shift a deliberate, reviewed event.

### Run

```
uv run pytest tests/test_golden.py
```

(Also runs as part of the plain suite.)

### When it fails

- **You didn't expect results to change** → the test did its job. Either
  your change wasn't behavior-neutral or you found a bug. Investigate;
  do not bless.
- **You expected results to change** (new indicator math, allocator
  changes, objective work) → bless:

```
UPDATE_GOLDEN=1 uv run pytest tests/test_golden.py
```

This rewrites `tests/fixtures/golden/expected/` and then **fails on
purpose**, so a blessing run is never mistaken for a passing one. Then:

1. Review `git diff tests/fixtures/golden/expected/` — the diff IS the
   review artifact. Confirm the numbers moved the way the change intended.
2. Re-run without the env var; it must pass.
3. Commit the golden diff together with the code change, with a sentence
   in the commit/PR saying why results moved.

### Fixtures

Input prices are three committed CSVs (`AAA/BBB/CCC.csv`) produced once by
the seeded generator:

```
uv run python tests/fixtures/golden/generate_fixtures.py
```

Regenerating them (or editing `portfolio.yaml`/`strategies.yaml` in the
fixture dir) invalidates every golden and requires a bless + diff review.
Only do this to deliberately change the *scenario* — e.g. so a new
strategy family or config knob is exercised. The scenario must keep the
health guards in `test_golden.py` passing: enough trades, both ST and LT
sells, non-empty tax summary, CPPI and vol-target overlays actually
engaging.
