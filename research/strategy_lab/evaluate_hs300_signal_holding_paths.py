#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate post-signal holding paths for HS300 Phase 1 / MoveGate signals.

The direction model decides whether to be long or short. This script answers
the next question:
    "After a high-quality signal appears, how long should we try to hold it?"

It reports fixed-bar holding results, path MFE/MAE diagnostics, a trailing
exit grid, and approximate IF futures compounding from 500k CNY.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "510300_breadth_regime"
    / "hs300_phase1_5m_60d_movegate_fixed_quality85_v1"
)


@dataclass
class Config:
    run_dir: str = str(DEFAULT_RUN_DIR)
    signal_file: str = "signals_5m_throttled.csv"
    eval_panel_file: str = str(
        REPO_ROOT
        / "results"
        / "510300_breadth_regime"
        / "hs300_phase1_5m_60d_strict_v1"
        / "eval_panel.parquet"
    )
    output_dir: str = str(DEFAULT_RUN_DIR / "holding_path_eval")
    fixed_horizons: str = "5,6,7,8,10,12,15,20,25,30"
    dynamic_max_horizons: str = "10,15,20,30"
    dynamic_min_holds: str = "3,5"
    dynamic_stop_loss_bps: str = "15,25,40"
    dynamic_activation_bps: str = "20,30,40"
    dynamic_trail_bps: str = "10,20,30"
    runner_check_bars: str = "5"
    runner_trigger_bps: str = "10,20,30,40"
    runner_final_horizons: str = "15,20,25,30"
    initial_capital: float = 500000.0
    if_multiplier: float = 300.0
    etf_to_index_scale: float = 1000.0
    cost_bps: float = 4.5


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Evaluate HS300 signal holding horizons and trailing exits")
    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument("--signal-file", default=Config.signal_file)
    p.add_argument("--eval-panel-file", default=Config.eval_panel_file)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--fixed-horizons", default=Config.fixed_horizons)
    p.add_argument("--dynamic-max-horizons", default=Config.dynamic_max_horizons)
    p.add_argument("--dynamic-min-holds", default=Config.dynamic_min_holds)
    p.add_argument("--dynamic-stop-loss-bps", default=Config.dynamic_stop_loss_bps)
    p.add_argument("--dynamic-activation-bps", default=Config.dynamic_activation_bps)
    p.add_argument("--dynamic-trail-bps", default=Config.dynamic_trail_bps)
    p.add_argument("--runner-check-bars", default=Config.runner_check_bars)
    p.add_argument("--runner-trigger-bps", default=Config.runner_trigger_bps)
    p.add_argument("--runner-final-horizons", default=Config.runner_final_horizons)
    p.add_argument("--initial-capital", type=float, default=Config.initial_capital)
    p.add_argument("--if-multiplier", type=float, default=Config.if_multiplier)
    p.add_argument("--etf-to-index-scale", type=float, default=Config.etf_to_index_scale)
    p.add_argument("--cost-bps", type=float, default=Config.cost_bps)
    return Config(**vars(p.parse_args()))


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def resolve_child(base: Path, maybe_path: str) -> Path:
    path = Path(maybe_path)
    return path if path.is_absolute() else base / path


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(cfg.run_dir)
    signals = pd.read_csv(run_dir / cfg.signal_file)
    signals["datetime"] = pd.to_datetime(signals["datetime"])
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    signals = signals.sort_values(["date", "minute_idx", "datetime"]).reset_index(drop=True)

    panel_path = resolve_child(run_dir, cfg.eval_panel_file)
    panel = pd.read_parquet(panel_path, columns=["date", "datetime", "minute_idx", "etf_close"])
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["date", "minute_idx", "datetime"]).reset_index(drop=True)
    return signals, panel


def summarize_returns(trades: pd.DataFrame, ret_col: str = "signed_ret") -> dict:
    if trades.empty:
        return {
            "n": 0,
            "nonzero_n": 0,
            "zero_rate": np.nan,
            "hit_rate": np.nan,
            "nonzero_hit_rate": np.nan,
            "mean_bps": np.nan,
            "net_mean_bps": np.nan,
            "nonzero_mean_bps": np.nan,
            "median_bps": np.nan,
            "p75_bps": np.nan,
            "p90_bps": np.nan,
        }
    signed = trades[ret_col].astype(float)
    bps = signed * 10000.0
    nonzero = bps[bps.abs() > 1e-9]
    return {
        "n": int(len(trades)),
        "nonzero_n": int(len(nonzero)),
        "zero_rate": float(1.0 - len(nonzero) / len(trades)),
        "hit_rate": float((bps > 0).mean()),
        "nonzero_hit_rate": float((nonzero > 0).mean()) if len(nonzero) else np.nan,
        "mean_bps": float(bps.mean()),
        "net_mean_bps": float((bps - trades.get("cost_bps", 0.0)).mean()),
        "nonzero_mean_bps": float(nonzero.mean()) if len(nonzero) else np.nan,
        "median_bps": float(bps.median()),
        "p75_bps": float(bps.quantile(0.75)),
        "p90_bps": float(bps.quantile(0.90)),
    }


def build_paths(signals: pd.DataFrame, panel: pd.DataFrame, max_horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows = []
    diag_rows = []
    by_day = {date: day.reset_index(drop=True) for date, day in panel.groupby("date", sort=False)}

    for sig_idx, sig in signals.iterrows():
        day = by_day.get(sig["date"])
        if day is None or day.empty:
            continue
        hit = day.index[day["datetime"].eq(sig["datetime"])]
        if len(hit) == 0:
            continue
        entry_pos = int(hit[0])
        entry_close = float(day.loc[entry_pos, "etf_close"])
        side = int(sig["signal_side_5"])
        if entry_close <= 0:
            continue
        end_pos = min(entry_pos + max_horizon, len(day) - 1)
        if end_pos <= entry_pos:
            continue
        future = day.loc[entry_pos + 1 : end_pos].copy()
        future["bar"] = np.arange(1, len(future) + 1)
        future["signal_row"] = int(sig_idx)
        future["entry_datetime"] = sig["datetime"]
        future["entry_minute_idx"] = int(sig["minute_idx"])
        future["side"] = side
        future["entry_close"] = entry_close
        future["signed_ret"] = side * np.log(future["etf_close"].astype(float) / entry_close)
        future["signed_bps"] = future["signed_ret"] * 10000.0
        path_rows.append(
            future[
                [
                    "signal_row",
                    "date",
                    "entry_datetime",
                    "datetime",
                    "entry_minute_idx",
                    "minute_idx",
                    "bar",
                    "side",
                    "entry_close",
                    "etf_close",
                    "signed_ret",
                    "signed_bps",
                ]
            ]
        )
        path = future["signed_bps"]
        diag_rows.append(
            {
                "signal_row": int(sig_idx),
                "date": sig["date"],
                "entry_datetime": sig["datetime"],
                "side": side,
                "entry_close": entry_close,
                "max_horizon_available": int(len(future)),
                "mfe_bps": float(path.max()),
                "mae_bps": float(path.min()),
                "best_bar": int(future.loc[path.idxmax(), "bar"]),
                "worst_bar": int(future.loc[path.idxmin(), "bar"]),
                "end30_bps": float(path.iloc[-1]) if len(path) else np.nan,
            }
        )
    paths = pd.concat(path_rows, ignore_index=True) if path_rows else pd.DataFrame()
    diag = pd.DataFrame(diag_rows)
    return paths, diag


def fixed_horizon_trades(signals: pd.DataFrame, paths: pd.DataFrame, horizons: Iterable[int], cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summary_rows = []
    signal_meta = signals.reset_index().rename(columns={"index": "signal_row"})
    meta_cols = [
        "signal_row",
        "date",
        "datetime",
        "minute_idx",
        "signal_name_5",
        "signal_side_5",
        "GateScore_5",
        "DirScore_5",
        "MoveScore_quality",
        "MoveScore_core",
    ]
    meta_cols = [c for c in meta_cols if c in signal_meta.columns]
    meta = signal_meta[meta_cols].copy()

    for horizon in horizons:
        selected = paths[paths["bar"].eq(horizon)].copy()
        if selected.empty:
            summary_rows.append({"exit_rule": f"fixed_{horizon}", "horizon": horizon, **summarize_returns(selected)})
            continue
        trades = selected.merge(meta, on=["signal_row", "date"], how="left", suffixes=("", "_signal"))
        trades["exit_rule"] = f"fixed_{horizon}"
        trades["horizon"] = int(horizon)
        trades["exit_bar"] = int(horizon)
        trades["exit_datetime"] = trades["datetime"]
        trades["cost_bps"] = float(cost_bps)
        trades["net_signed_ret"] = trades["signed_ret"] - cost_bps / 10000.0
        rows.append(trades)
        summary_rows.append({"exit_rule": f"fixed_{horizon}", "horizon": horizon, **summarize_returns(trades)})
    all_trades = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return all_trades, summary


def dynamic_exit_one(path: pd.DataFrame, max_horizon: int, min_hold: int, stop_loss_bps: float, activation_bps: float, trail_bps: float) -> pd.Series:
    path = path[path["bar"].le(max_horizon)].sort_values("bar")
    peak = -np.inf
    exit_row = path.iloc[-1]
    reason = "time"
    for _, row in path.iterrows():
        bps = float(row["signed_bps"])
        peak = max(peak, bps)
        if int(row["bar"]) < min_hold:
            continue
        if bps <= -stop_loss_bps:
            exit_row = row
            reason = "stop_loss"
            break
        if peak >= activation_bps and bps <= peak - trail_bps:
            exit_row = row
            reason = "trailing_stop"
            break
    out = exit_row.copy()
    out["exit_bar"] = int(exit_row["bar"])
    out["exit_datetime"] = exit_row["datetime"]
    out["exit_reason"] = reason
    out["path_peak_bps_to_exit"] = float(path[path["bar"].le(out["exit_bar"])]["signed_bps"].max())
    out["path_trough_bps_to_exit"] = float(path[path["bar"].le(out["exit_bar"])]["signed_bps"].min())
    return out


def dynamic_exit_grid(paths: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_horizons = parse_int_list(cfg.dynamic_max_horizons)
    min_holds = parse_int_list(cfg.dynamic_min_holds)
    stop_losses = parse_float_list(cfg.dynamic_stop_loss_bps)
    activations = parse_float_list(cfg.dynamic_activation_bps)
    trails = parse_float_list(cfg.dynamic_trail_bps)

    grid_rows = []
    trade_frames = []
    grouped = list(paths.groupby("signal_row", sort=True))
    for max_h in max_horizons:
        for min_hold in min_holds:
            if min_hold > max_h:
                continue
            for stop_loss in stop_losses:
                for activation in activations:
                    for trail in trails:
                        rows = []
                        for _, path in grouped:
                            if path["bar"].max() < min_hold:
                                continue
                            rows.append(dynamic_exit_one(path, max_h, min_hold, stop_loss, activation, trail))
                        trades = pd.DataFrame(rows)
                        if trades.empty:
                            continue
                        trades["exit_rule"] = (
                            f"trail_max{max_h}_min{min_hold}_sl{stop_loss:g}_act{activation:g}_tr{trail:g}"
                        )
                        trades["max_horizon"] = max_h
                        trades["min_hold"] = min_hold
                        trades["stop_loss_bps"] = stop_loss
                        trades["activation_bps"] = activation
                        trades["trail_bps"] = trail
                        trades["cost_bps"] = cfg.cost_bps
                        trades["net_signed_ret"] = trades["signed_ret"] - cfg.cost_bps / 10000.0
                        stats = summarize_returns(trades)
                        stats["avg_exit_bar"] = float(trades["exit_bar"].mean())
                        stats["time_exit_rate"] = float((trades["exit_reason"] == "time").mean())
                        stats["trail_exit_rate"] = float((trades["exit_reason"] == "trailing_stop").mean())
                        stats["stop_exit_rate"] = float((trades["exit_reason"] == "stop_loss").mean())
                        stats["score"] = stats["net_mean_bps"] * np.sqrt(stats["n"]) + 20.0 * (
                            (stats["nonzero_hit_rate"] if np.isfinite(stats["nonzero_hit_rate"]) else 0.5) - 0.5
                        )
                        grid_rows.append(
                            {
                                "exit_rule": trades["exit_rule"].iloc[0],
                                "max_horizon": max_h,
                                "min_hold": min_hold,
                                "stop_loss_bps": stop_loss,
                                "activation_bps": activation,
                                "trail_bps": trail,
                                **stats,
                            }
                        )
                        trade_frames.append(trades)
    grid = pd.DataFrame(grid_rows).sort_values(["score", "net_mean_bps"], ascending=False).reset_index(drop=True)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return all_trades, grid


def runner_exit_grid(paths: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-stage runner: check early profit; only winners are allowed to run."""
    check_bars = parse_int_list(cfg.runner_check_bars)
    triggers = parse_float_list(cfg.runner_trigger_bps)
    finals = parse_int_list(cfg.runner_final_horizons)

    grid_rows = []
    trade_frames = []
    grouped = list(paths.groupby("signal_row", sort=True))
    for check_bar in check_bars:
        for trigger in triggers:
            for final_horizon in finals:
                if final_horizon <= check_bar:
                    continue
                rows = []
                for _, path in grouped:
                    path = path.sort_values("bar")
                    checkpoint = path[path["bar"].eq(check_bar)]
                    if checkpoint.empty:
                        continue
                    checkpoint = checkpoint.iloc[0]
                    if float(checkpoint["signed_bps"]) >= trigger and path["bar"].max() >= final_horizon:
                        exit_row = path[path["bar"].eq(final_horizon)].iloc[0].copy()
                        reason = "runner"
                    else:
                        exit_row = checkpoint.copy()
                        reason = "base_exit"
                    exit_row["exit_bar"] = int(exit_row["bar"])
                    exit_row["exit_datetime"] = exit_row["datetime"]
                    exit_row["exit_reason"] = reason
                    exit_row["check_bar"] = check_bar
                    exit_row["trigger_bps"] = trigger
                    exit_row["final_horizon"] = final_horizon
                    rows.append(exit_row)
                trades = pd.DataFrame(rows)
                if trades.empty:
                    continue
                trades["exit_rule"] = f"runner_check{check_bar}_trig{trigger:g}_final{final_horizon}"
                trades["cost_bps"] = cfg.cost_bps
                trades["net_signed_ret"] = trades["signed_ret"] - cfg.cost_bps / 10000.0
                stats = summarize_returns(trades)
                stats["runner_rate"] = float((trades["exit_reason"] == "runner").mean())
                stats["avg_exit_bar"] = float(trades["exit_bar"].mean())
                stats["score"] = stats["net_mean_bps"] * np.sqrt(stats["n"]) + 20.0 * (
                    (stats["nonzero_hit_rate"] if np.isfinite(stats["nonzero_hit_rate"]) else 0.5) - 0.5
                )
                grid_rows.append(
                    {
                        "exit_rule": trades["exit_rule"].iloc[0],
                        "check_bar": check_bar,
                        "trigger_bps": trigger,
                        "final_horizon": final_horizon,
                        **stats,
                    }
                )
                trade_frames.append(trades)
    grid = pd.DataFrame(grid_rows).sort_values(["score", "net_mean_bps"], ascending=False).reset_index(drop=True)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    return all_trades, grid


def simulate_integer_if(trades: pd.DataFrame, cfg: Config, margin_rate: float, margin_utilization: float) -> pd.DataFrame:
    equity = cfg.initial_capital
    rows = []
    for _, trade in trades.sort_values("entry_datetime").iterrows():
        notional = float(trade["entry_close"]) * cfg.etf_to_index_scale * cfg.if_multiplier
        margin_per_lot = notional * margin_rate
        lots = int(np.floor(equity * margin_utilization / margin_per_lot)) if margin_per_lot > 0 else 0
        exposure = lots * notional
        gross = exposure * float(trade["signed_ret"])
        cost = exposure * cfg.cost_bps / 10000.0
        pnl = gross - cost
        start = equity
        equity += pnl
        rows.append(
            {
                "entry_datetime": trade["entry_datetime"],
                "exit_datetime": trade["exit_datetime"],
                "exit_bar": int(trade["exit_bar"]),
                "exit_reason": trade.get("exit_reason", trade.get("exit_rule", "")),
                "side": int(trade["side"]),
                "underlying_signed_bps": float(trade["signed_ret"]) * 10000.0,
                "lots": lots,
                "effective_leverage": exposure / start if start > 0 else 0.0,
                "start_equity": start,
                "gross_pnl": gross,
                "cost": cost,
                "pnl": pnl,
                "equity": equity,
            }
        )
    return pd.DataFrame(rows)


def summarize_curve(curve: pd.DataFrame, initial_capital: float) -> dict:
    if curve.empty:
        return {"ending_equity": initial_capital, "net_profit": 0.0, "max_drawdown": 0.0, "n_trades": 0}
    eq = curve["equity"].astype(float)
    return {
        "ending_equity": float(eq.iloc[-1]),
        "net_profit": float(eq.iloc[-1] - initial_capital),
        "total_return": float(eq.iloc[-1] / initial_capital - 1.0),
        "max_drawdown": float((eq / eq.cummax() - 1.0).min()),
        "n_trades": int(len(curve)),
        "win_rate_after_cost": float((curve["pnl"] > 0).mean()),
        "avg_effective_leverage": float(curve["effective_leverage"].mean()),
        "median_lots": float(curve["lots"].median()),
        "max_lots": int(curve["lots"].max()),
    }


def write_if_summary_for_rule(trades: pd.DataFrame, cfg: Config, out_dir: Path, prefix: str) -> pd.DataFrame:
    scenarios = [
        ("IF_min_margin_8pct_util80", 0.08, 0.80),
        ("IF_conservative_12pct_util70", 0.12, 0.70),
    ]
    rows = []
    for name, margin, util in scenarios:
        curve = simulate_integer_if(trades, cfg, margin, util)
        curve.to_csv(out_dir / f"{prefix}_{name}_curve.csv", index=False)
        rows.append({"scenario": f"{prefix}_{name}", **summarize_curve(curve, cfg.initial_capital)})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals, panel = load_inputs(cfg)
    max_needed = max(parse_int_list(cfg.fixed_horizons) + parse_int_list(cfg.dynamic_max_horizons))
    paths, path_diag = build_paths(signals, panel, max_needed)
    path_diag.to_csv(out_dir / "path_mfe_mae.csv", index=False)
    paths.to_csv(out_dir / "minute_paths.csv", index=False)

    fixed_trades, fixed_summary = fixed_horizon_trades(signals, paths, parse_int_list(cfg.fixed_horizons), cfg.cost_bps)
    fixed_trades.to_csv(out_dir / "fixed_horizon_trades.csv", index=False)
    fixed_summary.to_csv(out_dir / "fixed_horizon_summary.csv", index=False)

    dynamic_trades, dynamic_grid = dynamic_exit_grid(paths, cfg)
    dynamic_trades.to_csv(out_dir / "dynamic_exit_all_trades.csv", index=False)
    dynamic_grid.to_csv(out_dir / "dynamic_exit_grid.csv", index=False)

    runner_trades, runner_grid = runner_exit_grid(paths, cfg)
    runner_trades.to_csv(out_dir / "runner_exit_all_trades.csv", index=False)
    runner_grid.to_csv(out_dir / "runner_exit_grid.csv", index=False)

    if not dynamic_grid.empty:
        best_rule = str(dynamic_grid.iloc[0]["exit_rule"])
        best_dynamic = dynamic_trades[dynamic_trades["exit_rule"].eq(best_rule)].copy()
    else:
        best_rule = ""
        best_dynamic = pd.DataFrame()
    best_dynamic.to_csv(out_dir / "best_dynamic_trades.csv", index=False)

    if not runner_grid.empty:
        best_runner_rule = str(runner_grid.iloc[0]["exit_rule"])
        best_runner = runner_trades[runner_trades["exit_rule"].eq(best_runner_rule)].copy()
    else:
        best_runner_rule = ""
        best_runner = pd.DataFrame()
    best_runner.to_csv(out_dir / "best_runner_trades.csv", index=False)

    if_rows = []
    best_fixed_row = fixed_summary.sort_values(["net_mean_bps", "mean_bps"], ascending=False).head(1)
    if not best_fixed_row.empty:
        best_fixed_rule = str(best_fixed_row.iloc[0]["exit_rule"])
        best_fixed = fixed_trades[fixed_trades["exit_rule"].eq(best_fixed_rule)].copy()
        if_rows.append(write_if_summary_for_rule(best_fixed, cfg, out_dir, best_fixed_rule))
    else:
        best_fixed_rule = ""
    if not best_dynamic.empty:
        if_rows.append(write_if_summary_for_rule(best_dynamic, cfg, out_dir, "best_dynamic"))
    if not best_runner.empty:
        if_rows.append(write_if_summary_for_rule(best_runner, cfg, out_dir, "best_runner"))
    if_summary = pd.concat(if_rows, ignore_index=True) if if_rows else pd.DataFrame()
    if_summary.to_csv(out_dir / "if_summary.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "signals_n": int(len(signals)),
        "path_diag": summarize_returns(path_diag.rename(columns={"end30_bps": "signed_ret"}).assign(signed_ret=path_diag["end30_bps"] / 10000.0))
        if not path_diag.empty
        else {},
        "best_fixed_rule": best_fixed_rule,
        "best_fixed": fixed_summary[fixed_summary["exit_rule"].eq(best_fixed_rule)].to_dict("records"),
        "best_dynamic_rule": best_rule,
        "best_dynamic": dynamic_grid.head(1).to_dict("records"),
        "best_runner_rule": best_runner_rule,
        "best_runner": runner_grid.head(1).to_dict("records"),
        "if_summary": if_summary.to_dict("records"),
        "artifacts": {
            "path_mfe_mae": str((out_dir / "path_mfe_mae.csv").resolve()),
            "fixed_horizon_summary": str((out_dir / "fixed_horizon_summary.csv").resolve()),
            "dynamic_exit_grid": str((out_dir / "dynamic_exit_grid.csv").resolve()),
            "best_dynamic_trades": str((out_dir / "best_dynamic_trades.csv").resolve()),
            "runner_exit_grid": str((out_dir / "runner_exit_grid.csv").resolve()),
            "best_runner_trades": str((out_dir / "best_runner_trades.csv").resolve()),
            "if_summary": str((out_dir / "if_summary.csv").resolve()),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
