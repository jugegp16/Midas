"""Seed x budget grid across walk-forward folds (recorded experiment, 2026-08-24).

Q1: does 'seed dominates budget' hold across folds?
Q2: does the best training score give the best OOS?

Deterministic given the seeds below; rerun with:
    uv run python docs/experiments/2026-08-24-seed-budget-grid/fold_grid.py
"""

import json
from datetime import date
from pathlib import Path

OUT = Path(__file__).parent / "results.json"
SEEDS = (42, 7, 123)
FOLDS_WANTED = (1, 5, 10, 15)
TRIALS = (666, 2000)


def main() -> None:
    from midas.cli import _fetch_prices
    from midas.config import load_portfolio, load_strategies
    from midas.optimizer import WF_MIN_TEST_DAYS, WF_MIN_TRAIN_PCT, _fold_boundaries, _run_trial, optimize

    port = load_portfolio(Path("sample-portfolios/test.yaml"))
    cfgs, constraints, risk = load_strategies(Path("example-strategies/balanced-growth.yaml"))
    names = [c.name for c in cfgs]
    start, end = date(2016, 4, 20), date(2026, 4, 20)
    price_data = _fetch_prices(port, start, end, warmup_bars=100)

    all_dates: set[date] = set()
    for df in price_data.values():
        all_dates.update(d for d in df.index if start <= d <= end)
    days = sorted(all_dates)
    bounds = _fold_boundaries(len(days), WF_MIN_TRAIN_PCT, WF_MIN_TEST_DAYS)

    rows = []
    for fold in FOLDS_WANTED:
        i = fold - 1
        if i + 1 >= len(bounds):
            continue
        tr_s, tr_e = days[0], days[bounds[i] - 1]
        te_s, te_e = days[bounds[i]], days[bounds[i + 1] - 1]
        print(f"=== fold {fold}: train {tr_s}..{tr_e} | test {te_s}..{te_e}", flush=True)
        for seed in SEEDS:
            for trials in TRIALS:
                res = optimize(
                    portfolio=port,
                    price_data=price_data,
                    start=tr_s,
                    end=tr_e,
                    strategy_names=names,
                    n_trials=trials,
                    train_pct=1.0,
                    objective="gross",
                    risk_config=risk,
                    forecast_scaling=constraints.forecast_scaling,
                    seed=seed,
                )
                metrics, _ = _run_trial(
                    res.best_params,
                    port,
                    price_data,
                    te_s,
                    te_e,
                    enable_split=False,
                    risk_config=risk,
                    forecast_scaling=constraints.forecast_scaling,
                )
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "trials": trials,
                        "train": res.best_objective_value,
                        "oos": metrics.twr,
                        "best_params": res.best_params,
                    }
                )
                print(
                    f"  seed {seed:>3} n={trials:<5} train {res.best_objective_value:>9.2%}  oos {metrics.twr:>8.2%}",
                    flush=True,
                )
                OUT.write_text(json.dumps(rows, indent=1, default=str))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
