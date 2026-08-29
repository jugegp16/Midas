# Phase 6: Live Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The close-of-day report gains monitor lines comparing the live trailing fold-window against the validation baselines — day-offset comparison on TWR, drawdown vs the validated worst fold, input-hash gated, edge-triggered warnings.

**Architecture:** `src/midas/monitor.py` is pure (offset-bootstrap distribution + line rendering + warning flag); `LiveState` grows the persisted window (anchor date, daily returns, warned flag, expected cash); the live engine captures a TWR daily return at report time — detecting manual deposits as unexplained cash deltas so they never read as return — and feeds the lines into the existing `DailyReport`. The CLI wires baselines + input hash + the members sidecar's `fitted_as_of` (the window anchor, regardless of adoption).

**Tech Stack:** Python 3.14, existing live/report machinery, pytest.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (Component 4)

## Global Constraints

Same as prior phases. Additions:

- **The window anchor is the members sidecar's `fitted_as_of`** — a declined proposal still closed a fold-analog window. No members sidecar or no baselines ⇒ no monitor lines (silent absence, not an error).
- **Input-hash mismatch prints one explicit stale line and refuses percentages** — never compares against a different engine.
- **No annualization of partial windows**: the day-offset line reports cumulative TWR ranked against a bootstrap distribution built at the same offset.
- **Manual deposits**: live cash changes not attributable to booked fills are treated as inflows and excluded from that day's return (the TWR rule, applied to live).

---

### Task 1: Pure monitor — offset distribution and lines

**Files:** Create `src/midas/monitor.py`; test `tests/test_monitor.py` (create).

**Interfaces:**
- `offset_cumulative_distribution(fold_returns_concat: Sequence[float], offset: int, *, n_resamples: int = 500, mean_block: int = 21, seed: int = 0) -> list[float]` — stationary-bootstrap resamples of length `offset + 1`, one cumulative return each; `[]` when inputs are too short.
- `monitor_lines(baselines: Baselines, window_returns: Sequence[float], *, current_input_hash: str) -> tuple[list[str], bool]` — `(lines, drawdown_breached)`:
  - Hash gate first: mismatch ⇒ `["monitor: baselines stale (input mismatch) — comparison refused"]`, `False`.
  - Empty window ⇒ `([], False)`.
  - Line 1: `fold-window day {d}/{cadence}: {cum:+.2%} TWR (pre-tax) — p{rank} of validation distribution` with rank from the offset bootstrap (falls back to "distribution unavailable" when the bootstrap is empty).
  - Line 2: `drawdown (window peak): {dd:.2%} vs validated worst-fold {worst:.2%} — {normal|BREACH}`; `drawdown_breached` True on breach.

- [ ] **Step 1: failing tests** — hash-gate refusal; empty window; a healthy window ranks mid-distribution and reports `normal`; a crash window reports `BREACH` and `True`; day count printed as `day n/cadence`; no `%` annualization words appear. Build `Baselines` fixtures inline (two folds, constant returns) as in `tests/test_baselines.py`.
- [ ] **Step 2: RED.** **Step 3: implement** (reuse `stationary_bootstrap`'s block logic via a local generator or import-and-adapt; drawdown via the validator's `_max_drawdown_along` — import it or duplicate the 8-line helper into monitor as `_window_drawdown` with a comment). **Step 4: GREEN.** **Step 5: commit** ("Monitor lines: offset-bootstrap ranking, hash gate, breach flag").

---

### Task 2: LiveState monitor fields

**Files:** Modify `src/midas/live_state.py`; test `tests/test_live_state.py` (extend).

**Interfaces:** `LiveState` gains `monitor_anchor: date | None = None`, `monitor_returns: list[float] = []`, `monitor_dd_warned: bool = False`, `expected_cash: float | None = None` — all persisted through the YAML round-trip like existing fields, defaulting correctly when absent from old state files (loud-config philosophy applies to unknown keys, not missing new ones).

- [ ] **Step 1: failing tests** — round-trip with the fields set; loading a state file written before the fields existed yields the defaults.
- [ ] **Step 2: RED.** **Step 3: implement** following the exact (de)serialization pattern of the nearest existing field (`last_report_date` / cooldowns). **Step 4: GREEN.** **Step 5: commit** ("LiveState carries the monitor window").

---

### Task 3: Live engine + CLI integration

**Files:** Modify `src/midas/live.py`, `src/midas/alerts.py` (`DailyReport.monitor_lines: tuple[str, ...] = ()` + embed rendering), `src/midas/cli.py` (live command wiring); test `tests/test_live_engine.py` (extend).

**Interfaces:**
- `LiveEngine.__init__(..., baselines: Baselines | None = None, monitor_input_hash: str | None = None, monitor_anchor: date | None = None)`.
- At report build (`_maybe_send_report` path): compute the session's TWR return — `inflow = state.cash - expected_cash` when `expected_cash` is known (booked fills update `expected_cash` at booking; anything else is a manual deposit), `ret = (equity - max(inflow, 0)) / previous_equity - 1`; append to `state.monitor_returns`. If the sidecar anchor advanced past `state.monitor_anchor`, reset the window first (new fit ⇒ new window, adoption or not). Then `lines, breached = monitor_lines(...)`; **edge-trigger**: warning text appended only when `breached and not state.monitor_dd_warned`; `state.monitor_dd_warned = breached` (recovery resets). Lines go into `DailyReport.monitor_lines` and the terminal print.
- CLI `live`: when the strategies file has a `policy:` block, read `read_baselines` + `read_members`; compute the input fingerprint from the loaded configuration; pass all three. Absent any of them ⇒ monitor disabled, engine unchanged.

- [ ] **Step 1: failing tests** (follow `tests/test_live_engine.py`'s harness conventions — fake provider, fixed clock):
  - report carries monitor lines when baselines+anchor provided and window has a return;
  - manual cash injection between ticks does not inflate the monitor return (set state cash += 1000 externally; day return excludes it);
  - anchor advance resets the window;
  - breach warns once and not on the following report while still breached; recovery re-arms.
- [ ] **Step 2: RED.** **Step 3: implement.** **Step 4: GREEN + full suite.** **Step 5: commit** ("Close report gains validation-anchored monitor lines").

---

### Task 4: Docs

**Files:** `docs/cli.md` (live section: monitor lines paragraph), `docs/architecture.md` (one paragraph).

- [ ] Update, full suite, commit + push.

---

## Self-review record

- Spec Component 4 coverage: fold-window day-offset TWR comparison (T1/T3), bootstrap not 16-fold percentile (T1), fit-date anchor regardless of adoption (T3 reset rule), fold-window drawdown anchor (T1 uses window peak), edge-triggered warning with persisted flag (T2/T3), input-hash refusal (T1), pre-tax stated (T1 line text), notify-don't-act (lines only; no engine action).
- Deposits-as-inflow handling implements the TWR requirement for live rather than documenting it away.
- Types consistent: `monitor_lines(...) -> tuple[list[str], bool]` in T1 and T3.
