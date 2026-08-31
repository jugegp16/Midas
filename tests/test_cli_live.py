"""End-to-end smoke test for the live CLI command's state-path resolution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from midas.cli import cli


def test_live_command_uses_sidecar_state_path_by_default(tmp_path: Path) -> None:
    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 10\n    cost_basis: 150\navailable_cash: 1000\n"
    )

    captured: dict[str, Path] = {}

    class _StubEngine:
        def __init__(self, *args: object, state_path: Path, **kwargs: object) -> None:
            captured["state_path"] = state_path

        def __enter__(self) -> _StubEngine:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        result = runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured["state_path"] == portfolio_yaml.with_suffix(".state.yaml")


def test_live_command_honors_state_file_field(tmp_path: Path) -> None:
    state_target = tmp_path / "run" / "explicit.state.yaml"
    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        f"state_file: {state_target}\n"
        "portfolio:\n"
        "  - ticker: AAPL\n"
        "    shares: 10\n"
        "    cost_basis: 150\n"
        "available_cash: 1000\n"
    )

    captured: dict[str, Path] = {}

    class _StubEngine:
        def __init__(self, *args: object, state_path: Path, **kwargs: object) -> None:
            captured["state_path"] = state_path

        def __enter__(self) -> _StubEngine:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        result = runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "--dry-run"])
        assert result.exit_code == 0, result.output
    assert captured["state_path"] == state_target


STRATS_POLICY = """\
min_cash_pct: 0.05
strategies:
- name: MeanReversion
  weight: 1.0
  params:
    window: 10
    threshold: 0.05
policy:
  objective: sharpe
  budget: 4
  restarts: 1
  base_seed: 42
  deployment: best
  cadence_days: 40
"""


def _policy_live(tmp_path: Path, *, members_hash: str | None):
    from datetime import date

    from midas.fitter import FitResult

    portfolio_yaml = tmp_path / "portfolio.yaml"
    portfolio_yaml.write_text(
        "portfolio:\n  - ticker: AAPL\n    shares: 10\n    cost_basis: 150\navailable_cash: 1000\n"
    )
    strategies_yaml = tmp_path / "strats.yaml"
    strategies_yaml.write_text(STRATS_POLICY)
    if members_hash is not None:
        from midas.artifacts import write_fit

        write_fit(
            strategies_yaml,
            FitResult(
                members=[{"MeanReversion": {"window": 10.0, "threshold": 0.05, "_weight": 1.0}}],
                member_scores=[1.0],
                restart_bests=[1.0],
                as_of=date(2026, 5, 1),
                policy_hash="p" * 64,
                input_hash=members_hash,
            ),
        )

    class _StubEngine:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def run(self) -> None:
            return

    with patch("midas.live.LiveEngine", _StubEngine):
        runner = CliRunner()
        return runner.invoke(cli, ["live", "-p", str(portfolio_yaml), "-s", str(strategies_yaml), "--dry-run"])


def test_live_refuses_policy_without_deployed_members(tmp_path: Path) -> None:
    """A policy-driven session must never fall back to hand-written params."""
    result = _policy_live(tmp_path, members_hash=None)
    assert result.exit_code != 0
    assert "midas fit" in result.output


def test_live_refuses_mismatched_members(tmp_path: Path) -> None:
    result = _policy_live(tmp_path, members_hash="e" * 64)
    assert result.exit_code != 0
    assert "different configuration" in result.output
