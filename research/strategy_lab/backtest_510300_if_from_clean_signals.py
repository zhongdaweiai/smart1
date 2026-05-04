#!/usr/bin/env python3
"""
Map the clean 510300 breadth-regime strategy onto IF futures.

This script does not assume a separate IF minute dataset exists locally.
Instead, it reconstructs the ETF entry/exit closes from real 510300 minute bars
and maps them to HS300 index points with:

    HS300 points ~= 510300 close * 1000

Then it computes one-contract IF PnL using:
    IF contract multiplier = 300 CNY / point

Outputs:
  - trades_if.csv
  - daily_if.csv
  - summary.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/daweizhong/Documents/projects")
REPO_ROOT = ROOT / "smart1"
ETF_DIR = ROOT / "ETF data core7"
TRADE_PATH = REPO_ROOT / "results" / "510300_breadth_regime" / "fixedthreshold_v1" / "walkforward_trades.csv"
OUT_DIR = REPO_ROOT / "results" / "510300_breadth_regime" / "if_backtest_2y_v1"


@dataclass(frozen=True)
class Config:
    trade_path: Path = TRADE_PATH
    etf_dir: Path = ETF_DIR
    out_dir: Path = OUT_DIR
    etf_code: str = "510300.XSHG"
    index_scale: float = 1000.0
    if_multiplier: float = 300.0
    round_trip_cost_bps: float = 4.5
    margin_rates: tuple[float, ...] = (0.08, 0.12)
    end_date: str = "2026-03-05"
    start_date: str = "2024-03-05"


def load_trade_signals(cfg: Config) -> pd.DataFrame:
    trades = pd.read_csv(cfg.trade_path)
    trades["date"] = pd.to_datetime(trades["date"]).dt.normalize()
    trades["datetime"] = pd.to_datetime(trades["datetime"])
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    trades = trades[(trades["date"] >= start) & (trades["date"] <= end)].copy()
    trades = trades.sort_values(["date", "datetime"]).reset_index(drop=True)
    return trades


def load_day_etf(etf_dir: Path, date_str: str, code: str) -> pd.DataFrame:
    fp = etf_dir / f"{date_str}.parquet"
    df = pd.read_parquet(fp, columns=["code", "datetime", "open", "close"])
    df = df[df["code"] == code].copy()
    if df.empty:
        raise FileNotFoundError(f"{code} not found in {fp}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return df


def build_trade_price_table(signal_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    if signal_df.empty:
        return pd.DataFrame()

    for date, day_signals in signal_df.groupby("date", sort=True):
        date_str = date.strftime("%Y-%m-%d")
        day_etf = load_day_etf(cfg.etf_dir, date_str, cfg.etf_code)
        day_lookup = day_etf.set_index("datetime")

        for row in day_signals.itertuples(index=False):
            entry_dt = row.datetime
            exit_dt = entry_dt + pd.Timedelta(minutes=int(row.horizon_min))
            if entry_dt not in day_lookup.index or exit_dt not in day_lookup.index:
                continue

            entry_close = float(day_lookup.at[entry_dt, "close"])
            exit_close = float(day_lookup.at[exit_dt, "close"])
            direction = 1.0 if float(row.pred_sign) >= 0 else -1.0

            entry_points = entry_close * cfg.index_scale
            exit_points = exit_close * cfg.index_scale
            point_move = direction * (exit_points - entry_points)

            notional = entry_points * cfg.if_multiplier
            gross_yuan = point_move * cfg.if_multiplier
            cost_yuan = notional * cfg.round_trip_cost_bps / 10000.0
            net_yuan = gross_yuan - cost_yuan
            gross_ret = direction * (exit_points / entry_points - 1.0)
            net_bps = gross_ret * 10000.0 - cfg.round_trip_cost_bps

            out = row._asdict()
            out.update(
                {
                    "entry_datetime": entry_dt,
                    "exit_datetime": exit_dt,
                    "entry_etf_close": entry_close,
                    "exit_etf_close": exit_close,
                    "entry_if_points_est": entry_points,
                    "exit_if_points_est": exit_points,
                    "if_point_move": point_move,
                    "if_notional_yuan": notional,
                    "if_gross_yuan": gross_yuan,
                    "if_cost_yuan": cost_yuan,
                    "if_net_yuan": net_yuan,
                    "if_gross_bps": gross_ret * 10000.0,
                    "if_net_bps": net_bps,
                }
            )

            for margin_rate in cfg.margin_rates:
                label = f"{int(round(margin_rate * 100)):02d}"
                margin_yuan = notional * margin_rate
                out[f"margin_{label}_yuan"] = margin_yuan
                out[f"net_ret_on_margin_{label}"] = net_yuan / margin_yuan

            rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["date", "entry_datetime"]).reset_index(drop=True)


def list_trading_dates(etf_dir: Path, start_date: str, end_date: str) -> list[pd.Timestamp]:
    dates = []
    for fp in sorted(etf_dir.glob("*.parquet")):
        date_str = fp.stem
        if start_date <= date_str <= end_date:
            dates.append(pd.Timestamp(date_str))
    return dates


def build_daily_curve(trades_if: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    all_dates = pd.DataFrame({"date": list_trading_dates(cfg.etf_dir, cfg.start_date, cfg.end_date)})
    if all_dates.empty:
        return all_dates

    if trades_if.empty:
        daily = all_dates.copy()
        daily["day_net_yuan"] = 0.0
        daily["day_gross_yuan"] = 0.0
        daily["trade_count"] = 0
        return daily

    grouped = (
        trades_if.groupby("date", as_index=False)
        .agg(
            day_net_yuan=("if_net_yuan", "sum"),
            day_gross_yuan=("if_gross_yuan", "sum"),
            trade_count=("if_net_yuan", "size"),
        )
    )
    daily = all_dates.merge(grouped, on="date", how="left").fillna(
        {"day_net_yuan": 0.0, "day_gross_yuan": 0.0, "trade_count": 0}
    )
    daily["trade_count"] = daily["trade_count"].astype(int)
    return daily.sort_values("date").reset_index(drop=True)


def summarize(trades_if: pd.DataFrame, daily: pd.DataFrame, cfg: Config) -> dict:
    summary: dict[str, object] = {
        "strategy_source": str(cfg.trade_path),
        "date_window": {"start": cfg.start_date, "end": cfg.end_date},
        "etf_code": cfg.etf_code,
        "mapping_assumption": f"HS300 points ~= {cfg.etf_code} close * {cfg.index_scale:.0f}",
        "if_multiplier": cfg.if_multiplier,
        "round_trip_cost_bps": cfg.round_trip_cost_bps,
        "margin_rates": list(cfg.margin_rates),
        "n_window_trading_days": int(len(daily)),
        "n_signal_days": int(trades_if["date"].nunique()) if not trades_if.empty else 0,
        "n_trades": int(len(trades_if)),
    }

    if trades_if.empty:
        return summary

    summary.update(
        {
            "first_trade_date": str(pd.Timestamp(trades_if["date"].min()).date()),
            "last_trade_date": str(pd.Timestamp(trades_if["date"].max()).date()),
            "total_net_yuan_per_contract": float(trades_if["if_net_yuan"].sum()),
            "total_gross_yuan_per_contract": float(trades_if["if_gross_yuan"].sum()),
            "avg_net_yuan_per_trade": float(trades_if["if_net_yuan"].mean()),
            "median_net_yuan_per_trade": float(trades_if["if_net_yuan"].median()),
            "win_rate": float((trades_if["if_net_yuan"] > 0).mean()),
            "gross_win_rate": float((trades_if["if_gross_yuan"] > 0).mean()),
            "avg_if_net_bps": float(trades_if["if_net_bps"].mean()),
            "median_if_net_bps": float(trades_if["if_net_bps"].median()),
            "best_trade_net_yuan": float(trades_if["if_net_yuan"].max()),
            "worst_trade_net_yuan": float(trades_if["if_net_yuan"].min()),
        }
    )

    best_trade = trades_if.loc[trades_if["if_net_yuan"].idxmax()].to_dict()
    worst_trade = trades_if.loc[trades_if["if_net_yuan"].idxmin()].to_dict()
    summary["best_trade"] = {
        "date": str(pd.Timestamp(best_trade["date"]).date()),
        "entry_datetime": str(best_trade["entry_datetime"]),
        "exit_datetime": str(best_trade["exit_datetime"]),
        "entry_if_points_est": float(best_trade["entry_if_points_est"]),
        "exit_if_points_est": float(best_trade["exit_if_points_est"]),
        "if_net_yuan": float(best_trade["if_net_yuan"]),
        "if_net_bps": float(best_trade["if_net_bps"]),
    }
    summary["worst_trade"] = {
        "date": str(pd.Timestamp(worst_trade["date"]).date()),
        "entry_datetime": str(worst_trade["entry_datetime"]),
        "exit_datetime": str(worst_trade["exit_datetime"]),
        "entry_if_points_est": float(worst_trade["entry_if_points_est"]),
        "exit_if_points_est": float(worst_trade["exit_if_points_est"]),
        "if_net_yuan": float(worst_trade["if_net_yuan"]),
        "if_net_bps": float(worst_trade["if_net_bps"]),
    }

    for margin_rate in cfg.margin_rates:
        label = f"{int(round(margin_rate * 100)):02d}"
        margin_col = f"margin_{label}_yuan"
        start_capital = float(trades_if[margin_col].max())
        equity = start_capital + daily["day_net_yuan"].cumsum()
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        ending_equity = float(equity.iloc[-1])
        total_return = ending_equity / start_capital - 1.0
        annualized = (ending_equity / start_capital) ** (242.0 / len(daily)) - 1.0
        sharpe = 0.0
        if daily["day_net_yuan"].std(ddof=0) > 0:
            daily_ret = daily["day_net_yuan"] / start_capital
            sharpe = float(daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(242.0))

        summary[f"margin_{label}"] = {
            "start_capital_yuan": start_capital,
            "ending_equity_yuan": ending_equity,
            "period_return": float(total_return),
            "annualized_return": float(annualized),
            "max_drawdown": float(drawdown.min()),
            "sharpe_daily": sharpe,
        }

    return summary


def main() -> None:
    cfg = Config()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    signals = load_trade_signals(cfg)
    trades_if = build_trade_price_table(signals, cfg)
    daily = build_daily_curve(trades_if, cfg)
    summary = summarize(trades_if, daily, cfg)

    trades_if.to_csv(cfg.out_dir / "trades_if.csv", index=False)
    daily.to_csv(cfg.out_dir / "daily_if.csv", index=False)
    with open(cfg.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
