"""YAML config loading for portfolio and strategy definitions."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from midas.models import (
    DEFAULT_ALERT_TIMEOUT_SECONDS,
    DEFAULT_MIN_BUY_DELTA,
    DEFAULT_MIN_CASH_PCT,
    DEFAULT_PENDING_TTL_HOURS,
    DEFAULT_REALERT_HOURS,
    DEFAULT_SOFTMAX_TEMPERATURE,
    DEFAULT_VOL_LOOKBACK_DAYS,
    AlertsConfig,
    AllocationConstraints,
    CashInfusion,
    Holding,
    PortfolioConfig,
    RiskConfig,
    StrategyConfig,
    TaxConfig,
    TradingRestrictions,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, requiring a top-level mapping.

    Raises:
        ValueError: If the file's top-level value is not a mapping.
    """
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path}"
        raise ValueError(msg)
    return data


def _parse_cash_infusion(raw: dict[str, Any]) -> CashInfusion | None:
    """Build the optional ``cash_infusion:`` block, coercing its date field."""
    if "cash_infusion" not in raw:
        return None
    ci = raw["cash_infusion"]
    next_date = ci["next_date"]
    if isinstance(next_date, str):
        next_date = date.fromisoformat(next_date)
    elif isinstance(next_date, datetime):
        next_date = next_date.date()
    return CashInfusion(
        amount=float(ci["amount"]),
        next_date=next_date,
        frequency=ci.get("frequency"),
    )


def _parse_tax_config(tax_raw: Any) -> TaxConfig | None:
    """Build the optional ``tax:`` block.

    Raises:
        ValueError: If the block has unrecognized keys or is neither a
            mapping nor ``off``.
    """
    if isinstance(tax_raw, dict):
        known_keys = {"short_term_rate", "long_term_rate", "deductible_loss_cap", "payment_lag_days"}
        unknown_keys = set(tax_raw) - known_keys
        if unknown_keys:
            # A typo'd rate key would otherwise silently fall back to the
            # default and skew every after-tax figure.
            msg = f"tax: has unrecognized keys {sorted(unknown_keys)}; known keys: {sorted(known_keys)}"
            raise ValueError(msg)
        return TaxConfig(
            short_term_rate=float(tax_raw.get("short_term_rate", 0.37)),
            long_term_rate=float(tax_raw.get("long_term_rate", 0.20)),
            deductible_loss_cap=float(tax_raw.get("deductible_loss_cap", 3000.0)),
            payment_lag_days=int(tax_raw.get("payment_lag_days", 105)),
        )
    if tax_raw is not None and tax_raw is not False:
        msg = f"tax: must be a mapping of rates or `off`, got {tax_raw!r}"
        raise ValueError(msg)
    return None


def _parse_alerts_number(alerts_raw: dict[str, Any], key: str, default: float) -> float:
    """Coerce an ``alerts:`` numeric field with a clear error."""
    raw_value = alerts_raw.get(key, default)
    # bool is an int subclass, so float(True) would silently become 1.0.
    if isinstance(raw_value, bool):
        msg = f"alerts: {key} must be a number, got {raw_value!r}"
        raise ValueError(msg)
    try:
        return float(raw_value)
    except TypeError, ValueError:
        msg = f"alerts: {key} must be a number, got {raw_value!r}"
        raise ValueError(msg) from None


def _parse_alerts_config(alerts_raw: Any) -> AlertsConfig | None:
    """Build the optional ``alerts:`` block.

    Raises:
        ValueError: If the block has unrecognized keys, is not a mapping,
            or still uses the removed webhook key.
    """
    if alerts_raw is None:
        return None
    if not isinstance(alerts_raw, dict):
        msg = f"alerts: must be a mapping, got {alerts_raw!r}"
        raise ValueError(msg)
    if "discord_webhook_url" in alerts_raw:
        # Removed when bot delivery replaced the webhook — silently ignoring
        # it would leave the operator believing alerts still flow through
        # the (dead) webhook.
        msg = (
            "alerts: discord_webhook_url was replaced by bot delivery — "
            "set discord_channel_id here and export MIDAS_DISCORD_BOT_TOKEN instead "
            "(see docs/cli.md, Discord push notifications)"
        )
        raise ValueError(msg)
    known_keys = {"discord_channel_id", "timeout_seconds", "pending_ttl_hours", "realert_hours"}
    unknown_keys = set(alerts_raw) - known_keys
    if unknown_keys:
        # A typo'd key would otherwise silently disable push notifications.
        msg = f"alerts: has unrecognized keys {sorted(unknown_keys)}; known keys: {sorted(known_keys)}"
        raise ValueError(msg)
    # Channel ids are snowflakes — huge integers that YAML happily parses
    # as int. Accept int or str, normalize to a digits-only string.
    channel_raw = alerts_raw.get("discord_channel_id")
    channel_id = str(channel_raw).strip() if isinstance(channel_raw, int | str) else None
    if channel_raw is None:
        channel_id = ""
    if channel_id is None or (channel_id and not channel_id.isdigit()):
        msg = f"alerts: discord_channel_id must be a numeric Discord channel id, got {channel_raw!r}"
        raise ValueError(msg)
    return AlertsConfig(
        discord_channel_id=channel_id,
        timeout_seconds=_parse_alerts_number(alerts_raw, "timeout_seconds", DEFAULT_ALERT_TIMEOUT_SECONDS),
        pending_ttl_hours=_parse_alerts_number(alerts_raw, "pending_ttl_hours", DEFAULT_PENDING_TTL_HOURS),
        realert_hours=_parse_alerts_number(alerts_raw, "realert_hours", DEFAULT_REALERT_HOURS),
    )


def load_portfolio(path: Path) -> PortfolioConfig:
    """Load portfolio config from YAML."""
    raw = _load_yaml(path)

    holdings = [
        Holding(
            ticker=entry["ticker"],
            shares=float(entry["shares"]),
            cost_basis=float(entry["cost_basis"]) if "cost_basis" in entry else None,
        )
        for entry in raw["portfolio"]
    ]

    infusion = _parse_cash_infusion(raw)

    restrictions = None
    if "trading_restrictions" in raw:
        tr = raw["trading_restrictions"]
        restrictions = TradingRestrictions(
            round_trip_days=int(tr.get("round_trip_days", 0)),
        )

    state_file_raw = raw.get("state_file")
    tax = _parse_tax_config(raw.get("tax"))
    alerts = _parse_alerts_config(raw.get("alerts"))

    return PortfolioConfig(
        holdings=holdings,
        available_cash=float(raw["available_cash"]),
        cash_infusion=infusion,
        trading_restrictions=restrictions,
        state_file=Path(state_file_raw) if state_file_raw is not None else None,
        tax_config=tax,
        tax_declared="tax" in raw,
        alerts_config=alerts,
    )


def load_strategies(
    path: Path,
) -> tuple[list[StrategyConfig], AllocationConstraints, RiskConfig]:
    """Load strategy configs, allocation knobs, and optional risk policy.

    Returns (strategies, constraints, risk_config). The risk block is
    optional; omitting it yields a default ``RiskConfig``.
    """
    raw = _load_yaml(path)

    configs = [
        StrategyConfig(
            name=strat["name"],
            params=strat.get("params", {}),
            tickers=strat.get("tickers"),
            weight=float(strat.get("weight", 1.0)),
        )
        for strat in raw["strategies"]
    ]

    max_pos = raw.get("max_position_pct")
    constraints = AllocationConstraints(
        max_position_pct=float(max_pos) if max_pos is not None else None,
        min_cash_pct=float(raw.get("min_cash_pct", DEFAULT_MIN_CASH_PCT)),
        softmax_temperature=float(
            raw.get("softmax_temperature", DEFAULT_SOFTMAX_TEMPERATURE),
        ),
        min_buy_delta=float(
            raw.get("min_buy_delta", DEFAULT_MIN_BUY_DELTA),
        ),
        forecast_scaling=str(raw.get("forecast_scaling", "none")),
    )

    risk_raw = raw.get("risk") or {}
    risk = RiskConfig(
        weighting=str(risk_raw.get("weighting", "equal")),
        vol_lookback_days=int(risk_raw.get("vol_lookback_days", DEFAULT_VOL_LOOKBACK_DAYS)),
        vol_target=float(risk_raw["vol_target"]) if risk_raw.get("vol_target") is not None else None,
        drawdown_penalty=(
            float(risk_raw["drawdown_penalty"]) if risk_raw.get("drawdown_penalty") is not None else None
        ),
        drawdown_floor=(float(risk_raw["drawdown_floor"]) if risk_raw.get("drawdown_floor") is not None else None),
    )

    return configs, constraints, risk
