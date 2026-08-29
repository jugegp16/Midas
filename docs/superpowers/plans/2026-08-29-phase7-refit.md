# Phase 7: Scheduled Re-fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `midas refit` fits at cadence in its own process, records to history, and proposes adoption only on degradation evidence; live posts the proposal for ✅ confirmation (terminal tier never auto-adopts), reloads members on adoption, and spawns the re-fit overnight when `--refit` is on.

**Architecture:** The artifacts module splits `write_fit` into `record_fit` (history only) + `deploy_fit` (members sidecar) — `midas fit` keeps doing both (explicit operator command), `refit` records always and deploys only through adoption. A proposal artifact (`<strategies>.proposal.yaml`) carries the pending adoption between the refit process and live; live watches the members sidecar's `fitted_as_of` each tick and swaps its allocator when it changes, which makes adoption-by-any-path (✅ in Discord, `midas refit --adopt` in a terminal, even while live runs) converge on one reload mechanism. The confirmer grows one generic `post_message`; `poll_decision` is already message-id-shaped and reused as-is.

**Tech Stack:** Python 3.14, existing live/confirmer machinery, subprocess spawning behind an injected seam.

**Spec:** `docs/specs/2026-08-29-policy-engine-design.md` (Component 5; live tiers; kill criteria)

## Global Constraints

Same as prior phases. Additions:

- **`--refit` is opt-in (default off)** per the spec's gate; flag polarity flips only after landing experiment 1 and #54.
- **Live never auto-adopts in any tier.** Confirm tier adopts on ✅; terminal tier prints the proposal (with the `midas refit --adopt` instruction) and lets it expire; dry-run neither fits nor proposes.
- **Proposals expire after `pending_ttl_hours`** and are re-raised only by the next cadence fit.
- **Refit reads live state read-only** (no exclusive lock — live holds it); adoption never edits the human YAML.

---

### Task 1: Artifacts split — `record_fit` / `deploy_fit`

**Files:** `src/midas/artifacts.py`; `tests/test_artifacts.py`.

**Interfaces:** `record_fit(strategies_path, fit) -> Path` (history entry only); `deploy_fit(strategies_path, fit) -> None` (members sidecar only); `write_fit` = record + deploy (unchanged behavior, now composed). New: proposal artifact — `proposal_path(strategies_path)`, `write_proposal(strategies_path, fit_as_of: date, input_hash: str, evidence: list[str], created_at: datetime) -> None`, `read_proposal(...) -> Proposal | None`, `clear_proposal(...)`, with `@dataclass(frozen=True) Proposal(fit_as_of, input_hash, evidence, created_at, status: Literal["pending"])`.

- [ ] Tests: `record_fit` leaves the members sidecar untouched; `deploy_fit` writes it without adding history; `write_fit` still equals both (existing tests keep passing); proposal round-trip + clear. RED → implement → GREEN → commit ("Artifacts: record/deploy split and the adoption proposal").

---

### Task 2: `midas refit`

**Files:** `src/midas/cli.py`; `tests/test_cli_refit.py` (create; reuse `fake_prices` + `STRATS_WITH_POLICY` patterns from `tests/test_cli_fit.py`).

**Interfaces:** `midas refit -p PORTFOLIO -s STRATEGIES --start DATE [--as-of DATE] [--state PATH] [--adopt]`:
- Default mode: `fit_as_of` (incumbent = deployed members) → `record_fit`. Then evaluate `trigger_fired(baselines.folds, state.monitor_returns, policy.adopt_trigger)` — state loaded read-only from `--state` (default derived like live does), baselines from the sidecar; missing state/baselines ⇒ no trigger, message says so. Trigger fired ⇒ `write_proposal` with evidence lines (the monitor lines at evaluation time) and echo "proposal written; confirm in Discord or run midas refit --adopt".
- `--adopt`: deploys the latest recorded fit (`deploy_fit`), clears any proposal, echoes what was deployed. Refuses when history is empty.
- Input-hash gate: refuses to fit or adopt when the deployed members' `input_hash` differs from the current configuration's fingerprint ("no valid incumbent — run midas fit for a fresh deployment").

- [ ] Tests: refit records history without touching members; degraded state (crash returns seeded in a state file + tiny-worst-dd baselines) ⇒ proposal file written; healthy state ⇒ no proposal; `--adopt` deploys latest and clears proposal; hash mismatch refuses. RED → implement → GREEN → commit ("midas refit: cadence fitting with degradation-gated proposals").

---

### Task 3: Live proposal flow + member reload

**Files:** `src/midas/live.py`, `src/midas/alerts.py` (confirmer `post_message`), `src/midas/live_state.py` (`pending_proposal_message_id: str | None`, `proposal_fit_as_of: date | None`, `proposal_posted_at: datetime | None` persisted); `tests/test_live_engine.py`.

**Interfaces:**
- `OrderConfirmer` protocol + `DiscordConfirmer` + tests' `FakeConfirmer` gain `post_message(embed: dict) -> str | None` (plain embed, no seeding).
- `LiveEngine.__init__(..., strategies_path: Path | None = None)` — enables sidecar watching and proposal handling when given.
- Per tick (before order handling): (a) **member reload** — if the members sidecar's `fitted_as_of` differs from the last one loaded, rebuild the allocator from the members (via `_instantiate_strategies`, `Allocator(members=...)`) and log it; (b) **proposal pickup** — a `pending` proposal artifact not yet posted: confirm tier posts an embed (old vs new member count/scores, evidence lines, "trades implied are estimates at proposal-time prices") and persists the message id; terminal tier prints it once with the `--adopt` instruction; (c) **proposal poll** — confirmed ⇒ `deploy_fit`(the referenced fit from history) + `clear_proposal` (+ reload happens via (a) next tick); declined ⇒ `clear_proposal`; older than `pending_ttl_hours` ⇒ expire + `clear_proposal`, log "re-raised at next cadence fit".

- [ ] Tests (harness patterns from phase 6): reload swaps the allocator when the sidecar changes; proposal posts once in confirm tier and prints once in terminal tier; ✅ deploys and clears; ❌ clears without deploying; TTL expiry clears; dry-run ignores proposals. RED → implement → GREEN → commit ("Live: adoption proposals with confirm-tier ✅ and sidecar-watching member reload").

---

### Task 4: Cadence spawning + `--refit` flag

**Files:** `src/midas/live.py`, `src/midas/cli.py`; `tests/test_live_engine.py`.

**Interfaces:** `LiveEngine.__init__(..., refit_spawner: Callable[[], None] | None = None, cadence_days: int | None = None)` — after the close-of-day report, when `cadence_days` is set and the members sidecar's `fitted_as_of` is ≥ `cadence_days` trading days old (approximate with calendar `cadence_days * 7 / 5`) and no spawn happened this session, call the spawner. CLI `live --refit` builds the default spawner: `subprocess.Popen([sys.executable, "-m", "midas", "refit", ...])` detached, and passes `cadence_days` from the policy; without the flag, no spawner, engine unchanged.

- [ ] Tests: spawner called exactly once after the report when the sidecar is stale; not called when fresh, when flag off, or in dry-run. RED → implement → GREEN → commit ("Live spawns midas refit at cadence, opt-in").

---

### Task 5: Docs + spec closeout

**Files:** `docs/cli.md` (refit section + live `--refit` row), `docs/architecture.md`, spec deviations (rollback now also reachable via `refit --adopt` path note; sidecar-watch reload as the single adoption mechanism).

- [ ] Update, full suite, commit + push.

---

## Self-review record

- Spec Component 5: separate process (T2/T4 spawner), fit-always/propose-on-trigger (T2), Discord proposal with evidence + estimate marking + TTL (T3), terminal tier never auto-adopts (T3), opt-in flag (T4), input-hash gate with fresh-fit path (T2), adoption writes sidecar never YAML (T1/T3), reload machinery (T3 — sidecar watch, covering ✅, `--adopt`, and offline adoptions uniformly), history + rollback already landed in phase 3.
- The operator-veto coherence gap stays disclosed (validator assumes trigger ⇒ adoption; live may decline/expire).
- Types consistent: `Proposal`, `record_fit`/`deploy_fit`, `post_message` across tasks.
