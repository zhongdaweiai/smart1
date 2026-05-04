#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance metrics and option-expression proxy for runner state-machine output.

Metrics use daily end-of-day equity, including no-trade days as flat days.
The option section is intentionally a proxy, not a real option-chain backtest:
it maps each underlying signed return to a premium return with assumed
premium-to-underlying ratio, delta, premium allocation, and option costs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "510300_breadth_regime"
    / "hs300_phase1_5m_90d_movegate_fixed_quality80_v1"
    / "runner_state_machine_v1"
)


@dataclass
class Config:
    run_dir: str = str(DEFAULT_RUN_DIR)
    eval_panel_file: str = str(
        REPO_ROOT
        / "results"
        / "510300_breadth_regime"
        / "hs300_phase1_5m_90d_strict_v1"
        / "eval_panel.parquet"
    )
    output_dir: str = str(DEFAULT_RUN_DIR / "performance_summary")
    initial_capital: float = 500000.0
    trading_days_per_year: int = 252


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Summarize runner performance and option proxy")
    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument("--eval-panel-file", default=Config.eval_panel_file)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--initial-capital", type=float, default=Config.initial_capital)
    p.add_argument("--trading-days-per-year", type=int, default=Config.trading_days_per_year)
    return Config(**vars(p.parse_args()))


def read_curve(run_dir: Path, filename: str) -> pd.DataFrame:
    curve = pd.read_csv(run_dir / filename)
    curve["entry_datetime"] = pd.to_datetime(curve["entry_datetime"])
    curve["exit_datetime"] = pd.to_datetime(curve["exit_datetime"])
    curve["date"] = curve["exit_datetime"].dt.normalize()
    return curve.sort_values("exit_datetime").reset_index(drop=True)


def read_trading_dates(eval_panel_file: str) -> pd.DatetimeIndex:
    panel = pd.read_parquet(eval_panel_file, columns=["date"])
    dates = pd.to_datetime(panel["date"]).dt.normalize().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def read_panel_prices(eval_panel_file: str) -> pd.DataFrame:
    panel = pd.read_parquet(eval_panel_file, columns=["date", "minute_idx", "etf_close"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.drop_duplicates(["date", "minute_idx"], keep="last")


def daily_equity_from_curve(curve: pd.DataFrame, trading_dates: pd.DatetimeIndex, initial_capital: float) -> pd.DataFrame:
    equity = initial_capital
    by_day: Dict[pd.Timestamp, float] = {}
    for _, row in curve.iterrows():
        equity = float(row["equity"])
        by_day[pd.Timestamp(row["date"])] = equity
    rows = []
    prev = initial_capital
    for date in trading_dates:
        prev = by_day.get(pd.Timestamp(date), prev)
        rows.append({"date": date, "equity": prev})
    out = pd.DataFrame(rows)
    out["daily_ret"] = out["equity"].pct_change().fillna(out["equity"] / initial_capital - 1.0)
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0
    return out


def max_drawdown_from_equity(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0


def perf_stats(daily: pd.DataFrame, initial_capital: float, trading_days_per_year: int) -> dict:
    if daily.empty:
        return {}
    ret = daily["daily_ret"].astype(float)
    eq = daily["equity"].astype(float)
    n_days = len(daily)
    ending = float(eq.iloc[-1])
    total_return = ending / initial_capital - 1.0
    ann_return = (ending / initial_capital) ** (trading_days_per_year / n_days) - 1.0 if n_days > 0 else np.nan
    vol = float(ret.std(ddof=1)) if len(ret) > 1 else np.nan
    sharpe = float(ret.mean() / vol * np.sqrt(trading_days_per_year)) if vol and np.isfinite(vol) and vol > 0 else np.nan
    downside = ret[ret < 0]
    downside_vol = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = (
        float(ret.mean() / downside_vol * np.sqrt(trading_days_per_year))
        if downside_vol and np.isfinite(downside_vol) and downside_vol > 0
        else np.nan
    )
    max_dd = max_drawdown_from_equity(eq)
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan
    return {
        "n_days": int(n_days),
        "ending_equity": ending,
        "net_profit": ending - initial_capital,
        "total_return": total_return,
        "annual_return": ann_return,
        "daily_mean_return": float(ret.mean()),
        "daily_vol_annualized": float(vol * np.sqrt(trading_days_per_year)) if np.isfinite(vol) else np.nan,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "positive_day_rate": float((ret > 0).mean()),
        "trade_day_rate": float((ret.abs() > 1e-12).mean()),
    }


def option_proxy_scenarios() -> List[dict]:
    return [
        {
            "scenario": "spread_conservative_alloc20_eff12_cost2_cap100",
            "premium_alloc": 0.20,
            "effective_leverage_on_premium": 12.0,
            "premium_cost_rate": 0.02,
            "max_premium_return": 1.00,
        },
        {
            "scenario": "atm_balanced_alloc25_eff25_cost5_uncapped",
            "premium_alloc": 0.25,
            "effective_leverage_on_premium": 25.0,
            "premium_cost_rate": 0.05,
            "max_premium_return": np.inf,
        },
        {
            "scenario": "otm_gamma_alloc15_eff35_cost8_uncapped",
            "premium_alloc": 0.15,
            "effective_leverage_on_premium": 35.0,
            "premium_cost_rate": 0.08,
            "max_premium_return": np.inf,
        },
        {
            "scenario": "aggressive_alloc30_eff35_cost8_uncapped",
            "premium_alloc": 0.30,
            "effective_leverage_on_premium": 35.0,
            "premium_cost_rate": 0.08,
            "max_premium_return": np.inf,
        },
    ]


def simulate_option_proxy(trades: pd.DataFrame, scenario: dict, initial_capital: float) -> pd.DataFrame:
    equity = initial_capital
    rows = []
    for _, trade in trades.sort_values("entry_datetime").iterrows():
        premium = equity * scenario["premium_alloc"]
        premium_ret = (
            scenario["effective_leverage_on_premium"] * float(trade["signed_ret"])
            - scenario["premium_cost_rate"]
        )
        premium_ret = max(premium_ret, -1.0)
        cap = scenario["max_premium_return"]
        if np.isfinite(cap):
            premium_ret = min(premium_ret, float(cap))
        pnl = premium * premium_ret
        start = equity
        equity += pnl
        rows.append(
            {
                "entry_datetime": trade["entry_datetime"],
                "exit_datetime": trade["exit_datetime"],
                "date": trade["date"],
                "underlying_signed_bps": float(trade["signed_bps"]),
                "premium_alloc": scenario["premium_alloc"],
                "effective_leverage_on_premium": scenario["effective_leverage_on_premium"],
                "premium_cost_rate": scenario["premium_cost_rate"],
                "premium_return": premium_ret,
                "start_equity": start,
                "premium": premium,
                "pnl": pnl,
                "equity": equity,
            }
        )
    return pd.DataFrame(rows)


def simulate_runner_only_option_proxy(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    scenario: dict,
    initial_capital: float,
) -> pd.DataFrame:
    """Buy option only after PROBE confirms, so failed probes do not burn premium."""
    equity = initial_capital
    rows = []
    px = prices.set_index(["date", "minute_idx"])["etf_close"].to_dict()
    for _, trade in trades.sort_values("entry_datetime").iterrows():
        if not bool(trade.get("confirmed_runner", False)):
            continue
        date = pd.Timestamp(trade["date"]).normalize()
        confirm_bar = int(trade.get("param_confirm_bar", 5))
        confirm_minute = int(trade["entry_minute_idx"]) + confirm_bar
        confirm_close = px.get((date, confirm_minute))
        if confirm_close is None or confirm_close <= 0:
            continue
        exit_close = float(trade["exit_close"])
        side = int(trade["side"])
        runner_signed_ret = side * float(np.log(exit_close / confirm_close))
        premium = equity * scenario["premium_alloc"]
        premium_ret = (
            scenario["effective_leverage_on_premium"] * runner_signed_ret
            - scenario["premium_cost_rate"]
        )
        premium_ret = max(premium_ret, -1.0)
        cap = scenario["max_premium_return"]
        if np.isfinite(cap):
            premium_ret = min(premium_ret, float(cap))
        pnl = premium * premium_ret
        start = equity
        equity += pnl
        rows.append(
            {
                "entry_datetime": pd.to_datetime(trade["entry_datetime"]) + pd.Timedelta(minutes=confirm_bar),
                "exit_datetime": trade["exit_datetime"],
                "date": date,
                "underlying_signed_bps": runner_signed_ret * 10000.0,
                "original_trade_signed_bps": float(trade["signed_bps"]),
                "confirm_close": confirm_close,
                "exit_close": exit_close,
                "premium_alloc": scenario["premium_alloc"],
                "effective_leverage_on_premium": scenario["effective_leverage_on_premium"],
                "premium_cost_rate": scenario["premium_cost_rate"],
                "premium_return": premium_ret,
                "start_equity": start,
                "premium": premium,
                "pnl": pnl,
                "equity": equity,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    run_dir = Path(cfg.run_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trading_dates = read_trading_dates(cfg.eval_panel_file)
    prices = read_panel_prices(cfg.eval_panel_file)

    curve_files = [
        "selected_IF_min_margin_8pct_util80_curve.csv",
        "selected_IF_conservative_12pct_util70_curve.csv",
    ]
    perf_rows = []
    daily_artifacts = {}
    for curve_file in curve_files:
        curve_path = run_dir / curve_file
        if not curve_path.exists():
            continue
        curve = read_curve(run_dir, curve_file)
        daily = daily_equity_from_curve(curve, trading_dates, cfg.initial_capital)
        scenario = curve_file.replace("_curve.csv", "")
        daily_path = out_dir / f"{scenario}_daily_equity.csv"
        daily.to_csv(daily_path, index=False)
        daily_artifacts[scenario] = str(daily_path.resolve())
        perf_rows.append({"scenario": scenario, **perf_stats(daily, cfg.initial_capital, cfg.trading_days_per_year)})

    trades = pd.read_csv(run_dir / "state_machine_trades.csv")
    trades["entry_datetime"] = pd.to_datetime(trades["entry_datetime"])
    trades["exit_datetime"] = pd.to_datetime(trades["exit_datetime"])
    trades["date"] = trades["exit_datetime"].dt.normalize()

    option_rows = []
    for sc in option_proxy_scenarios():
        curve = simulate_option_proxy(trades, sc, cfg.initial_capital)
        curve_path = out_dir / f"{sc['scenario']}_curve.csv"
        curve.to_csv(curve_path, index=False)
        daily = daily_equity_from_curve(curve, trading_dates, cfg.initial_capital)
        daily.to_csv(out_dir / f"{sc['scenario']}_daily_equity.csv", index=False)
        stats = perf_stats(daily, cfg.initial_capital, cfg.trading_days_per_year)
        trade_win = float((curve["pnl"] > 0).mean()) if not curve.empty else np.nan
        avg_premium_ret = float(curve["premium_return"].mean()) if not curve.empty else np.nan
        option_rows.append(
            {
                **sc,
                **stats,
                "trade_win_rate_after_proxy_cost": trade_win,
                "avg_premium_return_per_trade": avg_premium_ret,
                "curve": str(curve_path.resolve()),
            }
        )

        runner_curve = simulate_runner_only_option_proxy(trades, prices, sc, cfg.initial_capital)
        runner_curve_path = out_dir / f"runner_only_{sc['scenario']}_curve.csv"
        runner_curve.to_csv(runner_curve_path, index=False)
        runner_daily = daily_equity_from_curve(runner_curve, trading_dates, cfg.initial_capital)
        runner_daily.to_csv(out_dir / f"runner_only_{sc['scenario']}_daily_equity.csv", index=False)
        runner_stats = perf_stats(runner_daily, cfg.initial_capital, cfg.trading_days_per_year)
        runner_trade_win = float((runner_curve["pnl"] > 0).mean()) if not runner_curve.empty else np.nan
        runner_avg_premium_ret = float(runner_curve["premium_return"].mean()) if not runner_curve.empty else np.nan
        option_rows.append(
            {
                **sc,
                "scenario": "runner_only_" + sc["scenario"],
                **runner_stats,
                "trade_win_rate_after_proxy_cost": runner_trade_win,
                "avg_premium_return_per_trade": runner_avg_premium_ret,
                "curve": str(runner_curve_path.resolve()),
            }
        )

    perf = pd.DataFrame(perf_rows)
    opt = pd.DataFrame(option_rows)
    perf.to_csv(out_dir / "if_performance_metrics.csv", index=False)
    opt.to_csv(out_dir / "option_proxy_metrics.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "if_performance": perf.to_dict("records"),
        "option_proxy": opt.to_dict("records"),
        "artifacts": {
            "if_performance_metrics": str((out_dir / "if_performance_metrics.csv").resolve()),
            "option_proxy_metrics": str((out_dir / "option_proxy_metrics.csv").resolve()),
            **daily_artifacts,
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
