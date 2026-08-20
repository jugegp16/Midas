"""Integration tests for LiveEngine driving LiveState across ticks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from midas.allocator import Allocator
from midas.live import LiveEngine
from midas.live_state import LiveState, StateFileError, aggregate_cost_basis, load_state, save_atomic
from midas.models import (
    AllocationConstraints,
    CashInfusion,
    Direction,
    Holding,
    Order,
    OrderContext,
    PortfolioConfig,
    PositionLot,
    TradingRestrictions,
)
from midas.order_sizer import OrderSizer

ProviderFactory = Callable[[dict[str, list[float]], list[date]], MagicMock]


@pytest.fixture
def basic_portfolio() -> PortfolioConfig:
    return PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=150.0)],
        available_cash=1000.0,
    )


def test_live_engine_seeds_state_on_first_construction(
    basic_portfolio: PortfolioConfig, tmp_path: Path, make_provider: ProviderFactory
) -> None:
    state_path = tmp_path / "portfolio.state.yaml"
    assert not state_path.exists()

    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    sizer = OrderSizer()
    provider = make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])

    LiveEngine(
        portfolio=basic_portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
    )

    assert state_path.exists()
    state = load_state(state_path)
    assert state.available_cash == 1000.0
    assert "AAPL" in state.lots


def test_tick_advances_in_memory_hwm(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """First tick should advance per-ticker HWM in self._state to the latest close."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    sizer = OrderSizer()
    provider = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    assert engine._state.high_water_marks["AAPL"] == 200.0


def test_tick_uses_weighted_avg_basis_from_state(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """If the operator has two lots at different prices, the exit-rule loop
    should see the share-weighted average basis, not the YAML's static value."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=20.0, cost_basis=999.0)],  # YAML lies
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    # Pre-populate state with two real lots so the engine sees the right basis.
    seeded = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        lots={
            "AAPL": [
                PositionLot(shares=10.0, purchase_date=None, cost_basis=100.0),
                PositionLot(shares=10.0, purchase_date=None, cost_basis=200.0),
            ]
        },
    )
    save_atomic(seeded, state_path)

    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    sizer = OrderSizer()
    provider = make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=sizer,
        provider=provider,
        state_path=state_path,
    )

    # Verify positions/basis as the engine would compute them inside _tick.
    assert aggregate_cost_basis(engine._state.lots["AAPL"]) == 150.0
    # And after a tick, HWM should advance to 150.0 (the close).
    engine._tick(["AAPL"])
    assert engine._state.high_water_marks["AAPL"] == 150.0


def test_peak_equity_advances_and_persists(tmp_path: Path, make_provider: ProviderFactory) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.peak_equity == pytest.approx(10 * 200.0)


def test_state_is_persisted_to_disk_each_tick(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Even on a no-op tick (no orders), HWM and peak advance get saved."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [150.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    on_disk = load_state(state_path)
    assert on_disk.high_water_marks["AAPL"] == 150.0
    assert on_disk.peak_equity == pytest.approx(10 * 150.0)


def test_cash_infusion_advances_when_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """If today >= cash_infusion.next_date, cash increases by amount and
    next_date advances."""
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=500.0,
        cash_infusion=CashInfusion(amount=1500.0, next_date=date(2026, 5, 1), frequency="biweekly"),
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Patch date.today() to 2026-05-07 (past next_date 2026-05-01).
    class _FakeDate:
        @staticmethod
        def today() -> date:
            return date(2026, 5, 7)

    monkeypatch.setattr("midas.live.date", _FakeDate)
    engine._tick(["AAPL"])

    state = load_state(state_path)
    assert state.available_cash == pytest.approx(500.0 + 1500.0)
    assert state.cash_infusion_next_date == date(2026, 5, 15)  # +14 days for biweekly


def test_alert_cash_display_does_not_double_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """``apply_buy``/``apply_sell`` mutate ``state.available_cash`` in place,
    so the per-alert running subtotal must seed from the *pre-fill* baseline.
    Otherwise a single $100 BUY against $1000 cash would print "$800" instead
    of "$900" — both cash mutation and the printed delta apply.
    """
    portfolio = PortfolioConfig(
        holdings=[],  # no held positions; only a buy will fire
        available_cash=1000.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    # Inject a synthetic $100 BUY directly into the order sizer's buy pass; this
    # bypasses allocator/strategy plumbing while exercising the real apply-fills
    # and alert-emission paths in ``_tick``.
    fake_buy = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=1.0,
        price=100.0,
        estimated_value=100.0,
        context=OrderContext(
            contributions={"fake": 1.0},
            blended_score=1.0,
            target_weight=1.0,
            current_weight=0.0,
            reason="test",
            source="fake",
        ),
    )
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [fake_buy])

    captured: list[float] = []

    def capture_alert(order: Order, remaining_cash: float, *_args: object, **_kw: object) -> None:
        captured.append(remaining_cash)

    monkeypatch.setattr("midas.live.print_alert", capture_alert)

    engine._tick(["AAPL"])

    # Pre-fill cash $1000 minus $100 buy = $900. If the bug returns,
    # state.available_cash (already $900 after apply_buy) would be debited again
    # to $800.
    assert captured == [pytest.approx(900.0)]
    assert engine._state.available_cash == pytest.approx(900.0)


def test_tick_passes_current_drawdown_to_allocator(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """``_tick`` must compute ``(peak_equity - total_value) / peak_equity`` from
    persisted state and pass it as ``current_drawdown`` to the allocator. Without
    this, the CPPI overlay is inert in live regardless of how far below peak the
    portfolio sits — the headline scope-1 fix this PR claims to deliver. (Earlier
    iterations let it default to 0.0, leaving CPPI a no-op even when peak_equity
    persistence was present.)
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    # Pre-seed state with peak_equity=$2000 so the current $800 portfolio is
    # 60% below peak — non-trivial drawdown that should flow through.
    seeded = LiveState(
        available_cash=0.0,
        cash_infusion_next_date=None,
        peak_equity=2000.0,
        lots={"AAPL": [PositionLot(shares=10.0, purchase_date=None, cost_basis=100.0)]},
    )
    save_atomic(seeded, state_path)

    provider = make_provider({"AAPL": [80.0]}, [date(2026, 5, 7)])
    captured_kwargs: dict[str, object] = {}
    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)

    original_allocate = allocator.allocate

    def capturing_allocate(*args: object, **kwargs: object) -> object:
        captured_kwargs.update(kwargs)
        return original_allocate(*args, **kwargs)  # type: ignore[arg-type]

    allocator.allocate = capturing_allocate  # type: ignore[method-assign]

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    # total_value = $0 cash + 10 shares * $80 = $800. peak = $2000.
    # current_drawdown = (2000 - 800) / 2000 = 0.6
    assert captured_kwargs["current_drawdown"] == pytest.approx(0.6)


def test_peak_equity_survives_engine_restart_and_drives_drawdown(
    tmp_path: Path, make_provider: ProviderFactory
) -> None:
    """Tick A advances peak via the engine's own logic, then a fresh engine
    constructed from the same state path reads peak from disk and the next
    tick's current_drawdown reflects the persisted value.

    Without persistence, the second engine would re-seed peak from the YAML's
    seed_equity and compute drawdown=0.0 on the same total_value that was a
    real drawdown vs the persisted peak. Same dead-code-shape as the original
    C1 finding (peak_equity wired but never restored across processes).
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"

    # Tick A: price at 200 → equity at $2000, peak ratchets to $2000.
    provider_a = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine_a = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider_a,
        state_path=state_path,
    )
    engine_a._tick(["AAPL"])
    persisted_peak = engine_a._state.peak_equity
    engine_a.close()

    # On disk the peak should now be $2000 (state was written by tick A).
    on_disk = load_state(state_path)
    assert on_disk.peak_equity == pytest.approx(2000.0)
    assert persisted_peak == pytest.approx(2000.0)

    # Tick B: fresh engine, lower price → current_drawdown reflects the persisted peak.
    captured: dict[str, float] = {}
    allocator_b = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    original_allocate = allocator_b.allocate

    def capturing_allocate(*args: object, current_drawdown: float = 0.0, **kwargs: object) -> object:
        captured["current_drawdown"] = current_drawdown
        return original_allocate(*args, current_drawdown=current_drawdown, **kwargs)  # type: ignore[arg-type]

    allocator_b.allocate = capturing_allocate  # type: ignore[method-assign]

    provider_b = make_provider({"AAPL": [80.0]}, [date(2026, 5, 8)])
    engine_b = LiveEngine(
        portfolio=portfolio,
        allocator=allocator_b,
        order_sizer=OrderSizer(),
        provider=provider_b,
        state_path=state_path,
    )
    engine_b._tick(["AAPL"])
    engine_b.close()

    # total_value on tick B = 0 cash + 10 * 80 = $800. peak (persisted) = $2000.
    # current_drawdown = (2000 - 800) / 2000 = 0.6
    assert captured["current_drawdown"] == pytest.approx(0.6)


def test_tick_does_not_advance_hwm_for_unheld_tickers(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """HWM must only ratchet for tickers we actually hold. Otherwise a ticker
    that ran up before the strategy first bought it would seed ``TrailingStop``
    against a pre-purchase peak — silently more conservative than backtest,
    which only tracks HWM on positions with shares > 0.
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=0.0, cost_basis=None)],
        available_cash=10000.0,
    )
    state_path = tmp_path / "state.yaml"
    seeded = LiveState(
        available_cash=10000.0,
        cash_infusion_next_date=None,
        # No lots → not held. AAPL is on the watchlist but has zero shares.
        lots={},
    )
    save_atomic(seeded, state_path)

    provider = make_provider({"AAPL": [200.0]}, [date(2026, 5, 7)])
    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    engine._tick(["AAPL"])

    assert "AAPL" not in engine._state.high_water_marks, (
        "HWM should not be set for an unheld ticker; otherwise TrailingStop seeds against pre-purchase peaks"
    )


def test_lockfile_prevents_concurrent_engines(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Two ``LiveEngine``s against the same state file would otherwise both
    load, both compute, both write — ``os.replace`` would pick a winner and
    silently drop the loser's fills. The advisory lock turns this into a clear
    "another midas live is already running" error.
    """
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    state_path = tmp_path / "state.yaml"
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])

    first = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    try:
        with pytest.raises(RuntimeError, match="another midas live process"):
            LiveEngine(
                portfolio=portfolio,
                allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
                order_sizer=OrderSizer(),
                provider=provider,
                state_path=state_path,
            )
    finally:
        first.close()

    # After close(), a fresh engine should acquire the lock cleanly.
    second = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    second.close()


def test_lock_released_when_init_fails_after_acquisition(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """If load_or_seed raises after the lock is acquired (e.g., corrupt state file),
    the lock fd must be released. Otherwise a retry in the same process hits a
    bogus 'another midas live process' RuntimeError.
    """
    # Pre-write a corrupt state file so load_or_seed raises StateFileError.
    state_path = tmp_path / "state.yaml"
    state_path.write_text("schema_version: 99\navailable_cash: 0\n")  # unsupported

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=0.0,
    )
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])

    # First attempt: load_or_seed raises on the unsupported schema_version.
    with pytest.raises(StateFileError):
        LiveEngine(
            portfolio=portfolio,
            allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
            order_sizer=OrderSizer(),
            provider=provider,
            state_path=state_path,
        )

    # Fix the state file and retry — should succeed without "another midas live"
    # RuntimeError because the lock was released on the first failure.
    state_path.write_text("schema_version: 1\navailable_cash: 0.0\n")
    second = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )
    second.close()


def test_engine_creates_trade_log_alongside_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """Smoke: a tick that produces a fill writes ``<state>.trades.csv`` with a BUY row."""
    from midas.trade_log import read_trades

    portfolio = PortfolioConfig(
        holdings=[],
        available_cash=1000.0,
    )
    state_path = tmp_path / "portfolio.state.yaml"
    expected_log = state_path.with_suffix(state_path.suffix + ".trades.csv")
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 5, 7)])

    engine = LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=state_path,
    )

    fake_buy = Order(
        ticker="AAPL",
        direction=Direction.BUY,
        shares=1.0,
        price=100.0,
        estimated_value=100.0,
        context=OrderContext(
            contributions={"fake": 1.0},
            blended_score=1.0,
            target_weight=1.0,
            current_weight=0.0,
            reason="test",
            source="fake",
        ),
    )
    monkeypatch.setattr(engine._order_sizer, "size_buys", lambda *a, **kw: [fake_buy])

    try:
        engine._tick(["AAPL"])
    finally:
        engine.close()

    assert expected_log.exists(), "trade log should be created next to the state file"
    trades = read_trades(expected_log)
    buys = [t for t in trades if t.direction == Direction.BUY]
    assert buys, "first tick should record at least one BUY row"


def test_drawdown_overlay_produces_smaller_exposure_under_real_drawdown() -> None:
    """End-to-end check that current_drawdown actually affects exposure scaling.

    The Task 9 fix wires current_drawdown from state.peak_equity through to
    Allocator.allocate. Paired with ``test_tick_passes_current_drawdown_to_allocator``
    (which pins the kwarg arriving), this test pins the downstream formula —
    ``apply_drawdown_overlay`` reduces exposure when drawdown is non-zero, with
    the expected formula and clamping. Together they establish the chain works
    end-to-end: kwarg arrives + overlay produces expected scale = behavior is
    correct (and not silently dead code).
    """
    from midas.risk import apply_drawdown_overlay

    # No drawdown: exposure scale stays at 1.0.
    no_dd = apply_drawdown_overlay(current_drawdown=0.0, penalty=2.0, floor=0.5)
    assert no_dd == 1.0

    # 20% drawdown with penalty=2.0: exposure = max(1 - 2*0.2, 0.5) = 0.6.
    moderate = apply_drawdown_overlay(current_drawdown=0.2, penalty=2.0, floor=0.5)
    assert moderate == pytest.approx(0.6)

    # 60% drawdown (matches the live-engine kwarg-arrival test): raw =
    # 1 - 1.2 = -0.2, clamped to floor 0.5.
    deep = apply_drawdown_overlay(current_drawdown=0.6, penalty=2.0, floor=0.5)
    assert deep == pytest.approx(0.5)

    # Confirm the chain: deeper drawdown produces smaller-or-equal exposure.
    assert deep <= moderate <= no_dd


class FakeConfirmer:
    """OrderConfirmer test double with scriptable decisions."""

    def __init__(self) -> None:
        self.posted: list[Order] = []
        self.decisions: dict[str, str | None] = {}
        self.booked: list[tuple[str, str]] = []
        self.declined: list[str] = []
        self.expired: list[str] = []
        self.superseded: list[str] = []
        self.unfillable: list[tuple[str, str]] = []
        self.fail_posts = False
        self._counter = 0

    def post_order(self, order: Order, timestamp: datetime) -> str | None:
        self.posted.append(order)
        if self.fail_posts:
            return None
        self._counter += 1
        return f"msg-{self._counter}"

    def poll_decision(self, message_id: str) -> str | None:
        return self.decisions.get(message_id)

    def mark_booked(self, message_id: str, note: str) -> None:
        self.booked.append((message_id, note))

    def mark_declined(self, message_id: str) -> None:
        self.declined.append(message_id)

    def mark_expired(self, message_id: str) -> None:
        self.expired.append(message_id)

    def mark_superseded(self, message_id: str) -> None:
        self.superseded.append(message_id)

    def mark_unfillable(self, message_id: str, note: str) -> None:
        self.unfillable.append((message_id, note))


def _sized_order(ticker: str, direction: Direction, shares: float, price: float = 100.0) -> Order:
    return Order(
        ticker=ticker,
        direction=direction,
        shares=shares,
        price=price,
        estimated_value=shares * price,
        context=OrderContext(
            contributions={},
            blended_score=0.5,
            target_weight=0.2,
            current_weight=0.1,
            reason="test",
            source="Momentum",
        ),
    )


def _make_confirm_engine(
    tmp_path: Path,
    make_provider: ProviderFactory,
    confirmer: FakeConfirmer,
    *,
    pending_ttl_hours: float = 8.0,
    restrictions: TradingRestrictions | None = None,
) -> LiveEngine:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=10_000.0,
        trading_restrictions=restrictions,
    )
    allocator = Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1)
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 8, 19)])
    return LiveEngine(
        portfolio=portfolio,
        allocator=allocator,
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=tmp_path / "state.yaml",
        confirmer=confirmer,
        pending_ttl_hours=pending_ttl_hours,
    )


NOW = datetime(2026, 8, 19, 15, 0, 0, tzinfo=UTC)
TODAY = date(2026, 8, 19)


def test_post_pending_creates_pending_order_not_fill(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        cash_before = engine._state.available_cash
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, cash_before)

        assert len(confirmer.posted) == 1
        assert len(engine._state.pending_orders) == 1
        pending = engine._state.pending_orders[0]
        assert (pending.ticker, pending.direction, pending.shares) == ("AAPL", Direction.BUY, 5.0)
        assert pending.message_id == "msg-1"
        # Nothing filled: cash and lots untouched.
        assert engine._state.available_cash == cash_before
        assert sum(lot.shares for lot in engine._state.lots["AAPL"]) == 10.0


def test_post_failure_leaves_no_pending(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    confirmer.fail_posts = True
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)

        assert engine._state.pending_orders == []


def test_pending_suppresses_same_intent(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        # Recomputed within tolerance (10%): suppressed, no second post.
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.2)], NOW, 10_000.0)

        assert len(confirmer.posted) == 1
        assert len(engine._state.pending_orders) == 1


def test_material_resize_replaces_pending_only_after_cooldown(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Size wobble within the cooldown must NOT churn expire-and-replace.

    Live prices move every tick, so recomputed share counts routinely
    cross the resize tolerance; replacing on every recompute buried the
    operator in "Expired" spam minutes apart.
    """
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)

        # Materially different size, but the pending alert is minutes old:
        # keep the original alert, no churn.
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 9.0)], NOW + timedelta(minutes=5), 10_000.0)
        assert confirmer.superseded == []
        assert [p.message_id for p in engine._state.pending_orders] == ["msg-1"]

        # Past the re-alert cooldown (1h default): now supersede.
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 9.0)], NOW + timedelta(hours=2), 10_000.0)
        assert confirmer.superseded == ["msg-1"]
        assert [p.message_id for p in engine._state.pending_orders] == ["msg-2"]
        assert engine._state.pending_orders[0].shares == 9.0


def test_direction_flip_replaces_pending(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW, 10_000.0)

        assert confirmer.superseded == ["msg-1"]
        assert [p.direction for p in engine._state.pending_orders] == [Direction.SELL]


def test_confirmed_buy_books_fill_and_restriction(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(
        tmp_path, make_provider, confirmer, restrictions=TradingRestrictions(round_trip_days=30)
    ) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "confirmed"

        engine._resolve_pending(TODAY, NOW)

        assert engine._state.pending_orders == []
        assert engine._state.available_cash == 10_000.0 - 500.0
        assert sum(lot.shares for lot in engine._state.lots["AAPL"]) == 15.0
        assert confirmer.booked and confirmer.booked[0][0] == "msg-1"
        # Round-trip clock started at confirm: an immediate SELL is blocked.
        assert engine._restriction_tracker is not None
        assert engine._restriction_tracker.is_blocked("AAPL", Direction.SELL, TODAY)
        # Trade log row written.
        log = (tmp_path / "state.yaml.trades.csv").read_text()
        assert "AAPL" in log and "BUY" in log


def test_declined_removes_without_fill(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "declined"

        engine._resolve_pending(TODAY, NOW)

        assert engine._state.pending_orders == []
        assert engine._state.available_cash == 10_000.0
        assert confirmer.declined == ["msg-1"]


def test_unresolved_pending_expires_after_ttl(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer, pending_ttl_hours=2.0) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)

        engine._resolve_pending(TODAY, NOW + timedelta(hours=1))
        assert len(engine._state.pending_orders) == 1  # still within TTL

        engine._resolve_pending(TODAY, NOW + timedelta(hours=3))
        assert engine._state.pending_orders == []
        assert confirmer.expired == ["msg-1"]
        assert engine._state.available_cash == 10_000.0


def test_confirmed_sell_books_buckets_and_clamps(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        # Ask to sell more than held: books the held 10, not the pending 12.
        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 12.0, price=110.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "confirmed"

        engine._resolve_pending(TODAY, NOW)

        assert engine._state.pending_orders == []
        assert "AAPL" not in engine._state.lots
        assert engine._state.available_cash == 10_000.0 + 10.0 * 110.0
        log = (tmp_path / "state.yaml.trades.csv").read_text()
        assert "SELL" in log


def test_dry_run_disables_confirm_mode(tmp_path: Path, make_provider: ProviderFactory) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=1_000.0,
    )
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 8, 19)])
    with LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=tmp_path / "state.yaml",
        dry_run=True,
        confirmer=FakeConfirmer(),
    ) as engine:
        assert engine._confirmer is None


def test_confirmed_fill_is_durable_before_side_effects(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """The fill and pending removal must hit disk inside _resolve_pending.

    The operator's reaction is durable external state that gets re-polled
    after a restart; if the booking weren't persisted before the trade-log
    append and Discord edits, a crash mid-tick would replay the
    confirmation and double-book the fill.
    """
    from midas.live_state import load_state

    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], NOW, 10_000.0)
        save_atomic(engine._state, tmp_path / "state.yaml")
        confirmer.decisions["msg-1"] = "confirmed"

        engine._resolve_pending(TODAY, NOW)

    on_disk = load_state(tmp_path / "state.yaml")
    assert on_disk.pending_orders == []
    assert on_disk.available_cash == 10_000.0 - 500.0
    assert sum(lot.shares for lot in on_disk.lots["AAPL"]) == 15.0


def test_sticky_reaction_is_not_replayed(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Discord never clears the operator's reaction — a resolved order must not re-book."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "confirmed"

        engine._resolve_pending(TODAY, NOW)
        engine._resolve_pending(TODAY, NOW)  # reaction still "present"

        assert engine._state.available_cash == 10_000.0 - 500.0
        assert len(confirmer.booked) == 1


def test_declined_intent_suppresses_reposting(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """A declined trade must not come back on the next poll."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "declined"
        engine._resolve_pending(TODAY, NOW)

        # Same intent recomputed on later ticks — even at a different size.
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 9.0)], NOW, 10_000.0)

        assert len(confirmer.posted) == 1
        assert engine._state.pending_orders == []


def test_declined_intent_expires_after_ttl(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer, pending_ttl_hours=2.0) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "declined"
        engine._resolve_pending(TODAY, NOW)

        # Within the window: suppressed. Past it: re-alerted.
        engine._resolve_pending(TODAY, NOW + timedelta(hours=1))
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW + timedelta(hours=1), 10_000.0)
        assert len(confirmer.posted) == 1

        engine._resolve_pending(TODAY, NOW + timedelta(hours=3))
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW + timedelta(hours=3), 10_000.0)
        assert len(confirmer.posted) == 2


def test_declined_buy_does_not_suppress_sell(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "declined"
        engine._resolve_pending(TODAY, NOW)

        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW, 10_000.0)

        assert len(confirmer.posted) == 2
        assert [p.direction for p in engine._state.pending_orders] == [Direction.SELL]


def test_declined_intent_survives_restart(tmp_path: Path, make_provider: ProviderFactory) -> None:
    from midas.live_state import load_state

    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "declined"
        engine._resolve_pending(TODAY, NOW)

    on_disk = load_state(tmp_path / "state.yaml")
    assert [(c.ticker, c.direction) for c in on_disk.intent_cooldowns] == [("AAPL", Direction.BUY)]


def test_confirmed_fill_cools_down_realerts(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """After a ✅, the same intent must not fire again on the next tick."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "confirmed"
        engine._resolve_pending(TODAY, NOW)

        # Allocator still wants more AAPL a tick later: suppressed.
        engine._resolve_pending(TODAY, NOW + timedelta(minutes=1))
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 3.0)], NOW + timedelta(minutes=1), 9_500.0)
        assert len(confirmer.posted) == 1

        # After the cooldown (1h default), a genuine scale-in may re-alert.
        engine._resolve_pending(TODAY, NOW + timedelta(hours=2))
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 3.0)], NOW + timedelta(hours=2), 9_500.0)
        assert len(confirmer.posted) == 2


def test_direction_flip_supersedes_within_cooldown(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """An exit signal replaces a fresh pending BUY immediately — no waiting."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW + timedelta(minutes=1), 10_000.0)

        assert confirmer.superseded == ["msg-1"]
        assert [p.direction for p in engine._state.pending_orders] == [Direction.SELL]


def test_unavailable_poll_never_expires_pending(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """A failing poll past TTL must NOT expire — the operator may have confirmed.

    Expiring on an outage/revoked-token span drops a fill the operator
    executed at the broker: exactly the phantom divergence confirm mode exists to
    eliminate.
    """
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer, pending_ttl_hours=1.0) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "unavailable"

        engine._resolve_pending(TODAY, NOW + timedelta(hours=10))
        assert len(engine._state.pending_orders) == 1
        assert confirmer.expired == []

        # Discord answers again — and the operator had confirmed.
        confirmer.decisions["msg-1"] = "confirmed"
        engine._resolve_pending(TODAY, NOW + timedelta(hours=10))
        assert engine._state.available_cash == 10_000.0 - 500.0


def test_posted_pending_is_durable_immediately(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """The Discord message is durable the moment it posts — the pending
    order must be too, or a crash before tick-end orphans the message."""
    from midas.live_state import load_state

    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)

        # No tick-end save has run — the pending must already be on disk.
        on_disk = load_state(tmp_path / "state.yaml")
        assert [p.message_id for p in on_disk.pending_orders] == ["msg-1"]


def test_confirmed_nothing_held_marks_unfillable(tmp_path: Path, make_provider: ProviderFactory) -> None:
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._state.lots.pop("AAPL", None)
        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "confirmed"

        engine._resolve_pending(TODAY, NOW)

        assert confirmer.booked == []
        assert len(confirmer.unfillable) == 1
        assert engine._state.available_cash == 10_000.0


def test_tick_confirm_mode_reserves_pending_buy_cash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """Cash promised to an unconfirmed BUY must be excluded from this
    tick's buy sizing, and the confirm branch must not assume-fill."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        post_time = datetime.now(tz=UTC)
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], post_time, 10_000.0)

        sizing_cash: list[float] = []

        def capture_size_buys(*args: object, **_kw: object) -> list[Order]:
            sizing_cash.append(float(args[3]))  # type: ignore[arg-type]
            return []

        monkeypatch.setattr(engine._order_sizer, "size_buys", capture_size_buys)
        engine._tick(["AAPL"])

        # $10,000 cash minus the $500 reserved by the pending BUY.
        assert sizing_cash == [pytest.approx(9_500.0)]
        # Confirm branch: no assumed fills, pending and cash untouched.
        assert engine._state.available_cash == 10_000.0
        assert [p.message_id for p in engine._state.pending_orders] == ["msg-1"]
        assert sum(lot.shares for lot in engine._state.lots["AAPL"]) == 10.0


def test_tick_confirm_mode_resolves_fill_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_provider: ProviderFactory
) -> None:
    """A confirmed fill must update positions/cash BEFORE this tick's
    allocation sees them, and the tick-end save must persist everything."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        post_time = datetime.now(tz=UTC)
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], post_time, 10_000.0)
        confirmer.decisions["msg-1"] = "confirmed"

        seen_weights: list[dict[str, float] | None] = []
        original_allocate = engine._allocator.allocate

        def spy_allocate(*args: object, **kwargs: object) -> object:
            seen_weights.append(kwargs.get("current_weights"))  # type: ignore[arg-type]
            return original_allocate(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(engine._allocator, "allocate", spy_allocate)
        engine._tick(["AAPL"])

        # Allocation saw the post-fill position: 15 sh x $100 of an $11,000
        # portfolio — not the pre-fill 10 sh (1000/11000).
        assert seen_weights == [{"AAPL": pytest.approx(1_500.0 / 11_000.0)}]
        assert engine._state.available_cash == pytest.approx(9_500.0)
        assert engine._state.pending_orders == []
        assert confirmer.booked and confirmer.booked[0][0] == "msg-1"

        # Tick-end save: the fill and peak equity are on disk.
        on_disk = load_state(tmp_path / "state.yaml")
        assert on_disk.available_cash == pytest.approx(9_500.0)
        assert sum(lot.shares for lot in on_disk.lots["AAPL"]) == 15.0
        assert on_disk.peak_equity == pytest.approx(11_000.0)


def test_stranded_pendings_warn_when_confirm_mode_off(
    tmp_path: Path, make_provider: ProviderFactory, caplog: pytest.LogCaptureFixture
) -> None:
    """Pending orders left by a confirm-mode run must be called out loudly
    when the engine restarts without a confirmer (token unset or --dry-run):
    they will never be polled, and terminal mode's assumed fills can
    double-book the same intent when confirm mode returns."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)

    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=10_000.0,
    )
    with caplog.at_level("WARNING", logger="midas.live"):
        engine_off = LiveEngine(
            portfolio=portfolio,
            allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
            order_sizer=OrderSizer(),
            provider=make_provider({"AAPL": [100.0]}, [date(2026, 8, 19)]),
            state_path=tmp_path / "state.yaml",
        )
        engine_off.close()

    warnings = [r.message for r in caplog.records if "pending order" in r.message]
    assert len(warnings) == 1
    assert "BUY AAPL" in warnings[0]
    assert "msg-1" in warnings[0]


def test_no_stranded_warning_without_pendings(
    tmp_path: Path, make_provider: ProviderFactory, caplog: pytest.LogCaptureFixture
) -> None:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=10_000.0,
    )
    with caplog.at_level("WARNING", logger="midas.live"):
        engine = LiveEngine(
            portfolio=portfolio,
            allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
            order_sizer=OrderSizer(),
            provider=make_provider({"AAPL": [100.0]}, [date(2026, 8, 19)]),
            state_path=tmp_path / "state.yaml",
        )
        engine.close()

    assert [r for r in caplog.records if "pending order" in r.message] == []


def test_supersede_repolls_and_keeps_reacted_pending(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """A reaction landing between the top-of-tick poll and the supersede
    check must not be dropped — the operator may already have executed.

    The supersede path re-polls immediately before replacing; any decision
    (or an unanswerable poll) keeps the pending for _resolve_pending.
    """
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0, price=100.0)], NOW, 10_000.0)
        # Operator confirms in the window after this tick's _resolve_pending
        # already polled (and saw nothing).
        confirmer.decisions["msg-1"] = "confirmed"

        # Direction flip would normally supersede immediately.
        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW + timedelta(minutes=1), 10_000.0)

        assert confirmer.superseded == []
        assert [p.message_id for p in engine._state.pending_orders] == ["msg-1"]
        # Next tick's resolve books the confirmed fill normally.
        engine._resolve_pending(TODAY, NOW + timedelta(minutes=2))
        assert engine._state.available_cash == 10_000.0 - 500.0


def test_supersede_defers_when_repoll_unavailable(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """If the pre-supersede re-poll cannot answer, keep the pending —
    superseding blind could drop a fill the operator executed."""
    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "unavailable"

        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW + timedelta(minutes=1), 10_000.0)

        assert confirmer.superseded == []
        assert [p.message_id for p in engine._state.pending_orders] == ["msg-1"]


def test_superseded_removal_is_durable_before_annotation(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """The supersede removal must hit disk before the Discord annotation,
    matching the persist-before-side-effects rule everywhere else."""
    from midas.live_state import load_state

    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)

        on_disk_at_annotation: list[list[str]] = []
        original_mark = confirmer.mark_superseded

        def spy_mark(message_id: str) -> None:
            on_disk_at_annotation.append([p.message_id for p in load_state(tmp_path / "state.yaml").pending_orders])
            original_mark(message_id)

        confirmer.mark_superseded = spy_mark  # type: ignore[method-assign]
        engine._post_pending_alerts([_sized_order("AAPL", Direction.SELL, 5.0)], NOW + timedelta(minutes=1), 10_000.0)

        assert confirmer.superseded == ["msg-1"]
        # When mark_superseded ran, msg-1 was already gone from disk.
        assert on_disk_at_annotation == [[]]


class FakeReporter:
    """DiscordReporter test double recording posted reports."""

    def __init__(self, deliver: bool = True) -> None:
        self.reports: list[object] = []
        self.deliver = deliver

    def post_report(self, report: object) -> bool:
        self.reports.append(report)
        return self.deliver


def _make_report_engine(tmp_path: Path, make_provider: ProviderFactory, reporter: FakeReporter) -> LiveEngine:
    portfolio = PortfolioConfig(
        holdings=[Holding(ticker="AAPL", shares=10.0, cost_basis=100.0)],
        available_cash=1_000.0,
    )
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 8, 18)])
    return LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=tmp_path / "state.yaml",
        reporter=reporter,  # type: ignore[arg-type]
    )


# Tue 2026-08-18: EDT, so 16:00 ET close == 20:00 UTC.
NEAR_CLOSE = datetime(2026, 8, 18, 19, 56, tzinfo=UTC)
MID_DAY = datetime(2026, 8, 18, 15, 0, tzinfo=UTC)


def test_report_fires_once_near_close(tmp_path: Path, make_provider: ProviderFactory) -> None:
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 18)

        engine._maybe_send_report(MID_DAY)
        assert reporter.reports == []  # too early in the session

        engine._maybe_send_report(NEAR_CLOSE)
        assert len(reporter.reports) == 1

        engine._maybe_send_report(NEAR_CLOSE + timedelta(minutes=1))
        assert len(reporter.reports) == 1  # once per session

        assert engine._state.last_report_date == date(2026, 8, 18)
        assert engine._state.last_report_equity == 2_000.0


def test_report_day_delta_uses_previous_anchor(tmp_path: Path, make_provider: ProviderFactory) -> None:
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 18)
        engine._maybe_send_report(NEAR_CLOSE)
        first = reporter.reports[0]
        assert first.previous_equity is None  # type: ignore[attr-defined]

        engine._last_equity = 2_100.0
        engine._last_equity_session = date(2026, 8, 19)
        engine._maybe_send_report(NEAR_CLOSE + timedelta(days=1))
        second = reporter.reports[1]
        assert second.previous_equity == 2_000.0  # type: ignore[attr-defined]
        assert second.equity == 2_100.0  # type: ignore[attr-defined]


def test_report_anchor_persists_before_delivery_failure(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """A Discord failure must not re-arm the report (no double-post)."""
    from midas.live_state import load_state

    reporter = FakeReporter(deliver=False)
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 18)
        engine._maybe_send_report(NEAR_CLOSE)

        assert len(reporter.reports) == 1
        engine._maybe_send_report(NEAR_CLOSE + timedelta(minutes=2))
        assert len(reporter.reports) == 1

    on_disk = load_state(tmp_path / "state.yaml")
    assert on_disk.last_report_date == date(2026, 8, 18)
    assert on_disk.last_report_equity == 2_000.0


def test_report_counters_reset_after_send(tmp_path: Path, make_provider: ProviderFactory) -> None:
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 18)
        engine._state.tally_posted = 3
        engine._state.tally_confirmed = 2
        engine._state.tally_declined = 1
        engine._maybe_send_report(NEAR_CLOSE)

        report = reporter.reports[0]
        assert (report.alerts_posted, report.confirmed, report.declined) == (3, 2, 1)  # type: ignore[attr-defined]
        assert engine._state.tally_posted == engine._state.tally_confirmed == engine._state.tally_declined == 0


def test_no_report_when_market_hours_disabled(tmp_path: Path, make_provider: ProviderFactory) -> None:
    reporter = FakeReporter()
    portfolio = PortfolioConfig(holdings=[Holding(ticker="AAPL", shares=10.0)], available_cash=1_000.0)
    provider = make_provider({"AAPL": [100.0]}, [date(2026, 8, 18)])
    with LiveEngine(
        portfolio=portfolio,
        allocator=Allocator(entries=[], constraints=AllocationConstraints(), n_tickers=1),
        order_sizer=OrderSizer(),
        provider=provider,
        state_path=tmp_path / "state.yaml",
        reporter=reporter,  # type: ignore[arg-type]
        market_hours_only=False,
    ) as engine:
        engine._last_equity = 2_000.0
        engine._maybe_send_report(NEAR_CLOSE)

    # 24/7 mode has no session concept — no close-of-day report.
    assert reporter.reports == []


def test_report_catches_up_after_close(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """A long poll interval can step from 15:5x straight past 16:00 — the
    closed-branch catch-up must still emit the report before sleeping."""
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 18)
        after_close = datetime(2026, 8, 18, 20, 3, tzinfo=UTC)  # 16:03 ET

        engine._maybe_send_report(after_close)

        assert len(reporter.reports) == 1


def test_no_catchup_report_without_equity_observation(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """An engine started in the evening never saw the session — no report."""
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        assert engine._last_equity is None
        engine._maybe_send_report(datetime(2026, 8, 18, 20, 3, tzinfo=UTC))

        assert reporter.reports == []


def test_no_report_on_weekend(tmp_path: Path, make_provider: ProviderFactory) -> None:
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 22)
        engine._maybe_send_report(datetime(2026, 8, 22, 19, 56, tzinfo=UTC))  # Saturday

        assert reporter.reports == []


def test_no_report_from_another_sessions_equity(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """A machine suspended since Monday must not stamp Monday's equity onto
    Wednesday's close-of-day report."""
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        engine._last_equity = 2_000.0
        engine._last_equity_session = date(2026, 8, 17)  # Monday's observation

        engine._maybe_send_report(datetime(2026, 8, 19, 20, 30, tzinfo=UTC))  # Wednesday evening

        assert reporter.reports == []
        assert engine._state.last_report_date is None


def test_pre_close_report_requires_equity_observation(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """An engine started at 15:56 whose fetches all failed must not report
    cash-only equity — and must not poison the persisted anchor with it."""
    from midas.live_state import load_state

    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        assert engine._last_equity is None
        engine._maybe_send_report(NEAR_CLOSE)

        assert reporter.reports == []
        assert engine._state.last_report_equity is None
    assert load_state(tmp_path / "state.yaml").last_report_equity is None


def test_alert_tally_survives_restart(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """The day's counts are durable state, not an in-memory counter."""
    from midas.live_state import load_state

    confirmer = FakeConfirmer()
    with _make_confirm_engine(tmp_path, make_provider, confirmer) as engine:
        engine._post_pending_alerts([_sized_order("AAPL", Direction.BUY, 5.0)], NOW, 10_000.0)
        confirmer.decisions["msg-1"] = "declined"
        engine._resolve_pending(TODAY, NOW)

    on_disk = load_state(tmp_path / "state.yaml")
    assert on_disk.tally_posted == 1
    assert on_disk.tally_declined == 1


def test_run_loop_sleeps_weekend_to_monday_open(tmp_path: Path, make_provider: ProviderFactory) -> None:
    """Saturday startup sleeps to Monday 9:30 ET, then ticks — driven
    through the clock seams in virtual time."""
    reporter = FakeReporter()
    with _make_report_engine(tmp_path, make_provider, reporter) as engine:
        clock = {"now": datetime(2026, 8, 22, 16, 0, tzinfo=UTC)}  # Saturday noon ET
        slept: list[float] = []
        ticks: list[datetime] = []

        def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock["now"] += timedelta(seconds=seconds)

        def fake_tick(tickers: list[str]) -> None:
            ticks.append(clock["now"])
            raise KeyboardInterrupt  # stop the loop after the first tick

        engine._now = lambda: clock["now"]
        engine._sleep = fake_sleep
        engine._tick = fake_tick  # type: ignore[method-assign]

        engine.run()

        assert len(ticks) == 1
        first_tick_et = ticks[0].astimezone(ZoneInfo("America/New_York"))
        assert first_tick_et.date() == date(2026, 8, 24)  # Monday
        assert (first_tick_et.hour, first_tick_et.minute) == (9, 30)
        assert all(chunk <= 3600.0 for chunk in slept)  # chunked, never one giant sleep
