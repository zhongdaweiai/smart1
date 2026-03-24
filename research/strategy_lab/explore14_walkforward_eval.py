#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class WFConfig:
    scored_panel_path: str
    output_dir: str
    train_days: int = 120
    test_days: int = 20
    min_train_trades: int = 8
    round_trip_cost_bps: float = 6.0


def parse_args() -> WFConfig:
    p = argparse.ArgumentParser(description="Walk-forward evaluation for explore14 candidate rules")
    p.add_argument("--scored-panel-path", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--train-days", type=int, default=120)
    p.add_argument("--test-days", type=int, default=20)
    p.add_argument("--min-train-trades", type=int, default=8)
    p.add_argument("--round-trip-cost-bps", type=float, default=6.0)
    return WFConfig(**vars(p.parse_args()))


def build_candidate_grid() -> pd.DataFrame:
    rows: List[dict] = []
    for side in ("LONG", "SHORT"):
        state = "CONFIRMED_UP" if side == "LONG" else "CONFIRMED_DOWN"
        for hold in (20, 30, 45):
            for dir_th in (1.0, 1.2, 1.5, 1.8):
                for trend_th in (0.8, 1.0, 1.2, 1.5):
                    for exh_max in (0.2, 0.3, 0.4):
                        for market_abs in (0.0, 0.2, 0.5, 0.8):
                            rows.append(
                                {
                                    "side": side,
                                    "state": state,
                                    "hold_bars": hold,
                                    "dir_th": dir_th,
                                    "trend_th": trend_th,
                                    "exh_max": exh_max,
                                    "market_abs": market_abs,
                                }
                            )
    return pd.DataFrame(rows)


def simulate_candidate(
    day: pd.DataFrame,
    candidate: dict,
    cost_bps: float,
    max_per_day: int = 2,
    cooldown_bars: int = 20,
) -> List[dict]:
    side = 1 if candidate["side"] == "LONG" else -1
    trades: List[dict] = []
    next_ok = -1
    trades_today = 0
    hold = int(candidate["hold_bars"])

    for bar in range(len(day) - hold - 1):
        minute_idx = int(day.loc[bar, "minute_idx"])
        if minute_idx < next_ok or trades_today >= max_per_day:
            continue
        if day.loc[bar, "state"] != candidate["state"]:
            continue
        dir10 = float(day.loc[bar, "DirScore_10"])
        trend10 = float(day.loc[bar, "TrendSupport_10"])
        exh10 = float(day.loc[bar, "Exhaustion_10"])
        market10 = float(day.loc[bar, "MarketPressure_10"])

        ok = trend10 >= candidate["trend_th"] and exh10 <= candidate["exh_max"]
        if side > 0:
            ok = ok and dir10 >= candidate["dir_th"] and market10 >= candidate["market_abs"]
        else:
            ok = ok and dir10 <= -candidate["dir_th"] and market10 <= -candidate["market_abs"]
        if not ok:
            continue

        entry_bar = bar + 1
        exit_bar = bar + hold + 1
        entry_px = float(day.loc[entry_bar, "etf_open"])
        exit_px = float(day.loc[exit_bar, "etf_open"])
        gross = side * (exit_px / entry_px - 1.0)
        net_bps = gross * 10000.0 - cost_bps
        trades.append(
            {
                "date": pd.Timestamp(day.loc[bar, "date"]).normalize(),
                "entry_time": day.loc[entry_bar, "datetime"],
                "exit_time": day.loc[exit_bar, "datetime"],
                "direction": candidate["side"],
                "state": candidate["state"],
                "hold_bars": hold,
                "dir_th": candidate["dir_th"],
                "trend_th": candidate["trend_th"],
                "exh_max": candidate["exh_max"],
                "market_abs": candidate["market_abs"],
                "entry_px": entry_px,
                "exit_px": exit_px,
                "gross_bps": gross * 10000.0,
                "net_bps": net_bps,
                "signal_dir10": dir10,
                "signal_trend10": trend10,
                "signal_exh10": exh10,
                "signal_market10": market10,
            }
        )
        next_ok = minute_idx + cooldown_bars
        trades_today += 1
    return trades


def simulate_period(panel: pd.DataFrame, candidate: dict, cfg: WFConfig) -> pd.DataFrame:
    trades: List[dict] = []
    for _, day in panel.groupby("date", sort=True):
        trades.extend(simulate_candidate(day.sort_values("datetime").reset_index(drop=True), candidate, cfg.round_trip_cost_bps))
    return pd.DataFrame(trades)


def summarize_panel(panel: pd.DataFrame, trades: pd.DataFrame) -> dict:
    unique_dates = sorted(pd.to_datetime(panel["date"]).dt.normalize().unique())
    date_index = pd.Index(unique_dates, name="date")
    if trades.empty:
        daily = pd.Series(0.0, index=date_index, name="day_pnl_bps")
    else:
        daily = trades.groupby("date")["net_bps"].sum().reindex(date_index, fill_value=0.0)
    pnl = daily.to_numpy(dtype=float) / 10000.0
    equity = (1.0 + pd.Series(pnl, index=date_index)).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    sharpe = float(pnl.mean() / pnl.std(ddof=0) * np.sqrt(242)) if pnl.std(ddof=0) > 0 else 0.0
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    return {
        "n_days": int(len(date_index)),
        "n_trade_days": int((daily != 0.0).sum()),
        "n_trades": int(len(trades)),
        "win_rate": float((trades["net_bps"] > 0).mean()) if len(trades) else 0.0,
        "avg_trade_bps": float(trades["net_bps"].mean()) if len(trades) else 0.0,
        "median_trade_bps": float(trades["net_bps"].median()) if len(trades) else 0.0,
        "avg_day_bps": float(daily.mean()) if len(daily) else 0.0,
        "total_return": total_return,
        "annualized_return": float((1.0 + total_return) ** (242 / max(len(date_index), 1)) - 1.0) if len(date_index) else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def candidate_score(summary: dict, min_train_trades: int) -> float:
    n = summary["n_trades"]
    if n < min_train_trades:
        return -1e9
    return summary["avg_trade_bps"] * np.sqrt(n) * max(summary["win_rate"], 0.01) + 50.0 * summary["total_return"] - 20.0 * abs(summary["max_drawdown"])


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(cfg.scored_panel_path).sort_values(["date", "datetime"]).reset_index(drop=True)
    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    all_dates = sorted(scored["date"].unique())
    grid = build_candidate_grid()

    fold_rows: List[dict] = []
    wf_trades: List[pd.DataFrame] = []

    for train_end in range(cfg.train_days, len(all_dates) - cfg.test_days + 1, cfg.test_days):
        train_dates = all_dates[train_end - cfg.train_days : train_end]
        test_dates = all_dates[train_end : train_end + cfg.test_days]
        train_panel = scored[scored["date"].isin(train_dates)]
        test_panel = scored[scored["date"].isin(test_dates)]
        if train_panel.empty or test_panel.empty:
            continue

        best_score = -1e18
        best_candidate: dict | None = None
        best_train_summary: dict | None = None
        for candidate in grid.to_dict("records"):
            train_trades = simulate_period(train_panel, candidate, cfg)
            train_summary = summarize_panel(train_panel, train_trades)
            score = candidate_score(train_summary, cfg.min_train_trades)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_train_summary = train_summary

        if best_candidate is None or best_train_summary is None:
            continue

        test_trades = simulate_period(test_panel, best_candidate, cfg)
        test_summary = summarize_panel(test_panel, test_trades)
        if not test_trades.empty:
            test_trades = test_trades.copy()
            test_trades["fold_train_start"] = pd.Timestamp(train_dates[0])
            test_trades["fold_train_end"] = pd.Timestamp(train_dates[-1])
            test_trades["fold_test_start"] = pd.Timestamp(test_dates[0])
            test_trades["fold_test_end"] = pd.Timestamp(test_dates[-1])
            wf_trades.append(test_trades)

        fold_rows.append(
            {
                "train_start": pd.Timestamp(train_dates[0]),
                "train_end": pd.Timestamp(train_dates[-1]),
                "test_start": pd.Timestamp(test_dates[0]),
                "test_end": pd.Timestamp(test_dates[-1]),
                "selected_side": best_candidate["side"],
                "selected_state": best_candidate["state"],
                "selected_hold_bars": best_candidate["hold_bars"],
                "selected_dir_th": best_candidate["dir_th"],
                "selected_trend_th": best_candidate["trend_th"],
                "selected_exh_max": best_candidate["exh_max"],
                "selected_market_abs": best_candidate["market_abs"],
                "train_n_trades": best_train_summary["n_trades"],
                "train_avg_trade_bps": best_train_summary["avg_trade_bps"],
                "train_total_return": best_train_summary["total_return"],
                "train_sharpe": best_train_summary["sharpe"],
                "test_n_trades": test_summary["n_trades"],
                "test_avg_trade_bps": test_summary["avg_trade_bps"],
                "test_total_return": test_summary["total_return"],
                "test_sharpe": test_summary["sharpe"],
                "test_max_drawdown": test_summary["max_drawdown"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    wf_trades_df = pd.concat(wf_trades, ignore_index=True) if wf_trades else pd.DataFrame()
    test_start = fold_df["test_start"].min() if not fold_df.empty else None
    test_end = fold_df["test_end"].max() if not fold_df.empty else None
    wf_panel = scored[scored["date"].between(test_start, test_end)] if test_start is not None else scored.iloc[0:0]
    summary = summarize_panel(wf_panel, wf_trades_df)
    summary["folds"] = int(len(fold_df))

    folds_path = out_dir / "folds.csv"
    trades_path = out_dir / "wf_trades.csv"
    summary_path = out_dir / "wf_summary.json"
    fold_df.to_csv(folds_path, index=False)
    wf_trades_df.to_csv(trades_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "summary": summary,
                "artifacts": {
                    "folds": str(folds_path.resolve()),
                    "wf_trades": str(trades_path.resolve()),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps({"config": asdict(cfg), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
