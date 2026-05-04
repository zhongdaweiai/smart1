#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner state machine for HS300 MoveGate signals.

State idea:
- FLAT: no position.
- PROBE: enter on a high-quality MoveGate signal and give it a short window
  to prove that price is actually moving.
- RUNNER: only trades with enough early profit are allowed to keep running.
- EXIT: close on failed confirmation, time exit, optional hard stop, or
  optional trailing stop.

The default rule is intentionally simple and live-feasible:
    enter -> check after 5 minutes -> if signed PnL >= 10 bps, hold to 30
    minutes from entry; otherwise exit at 5 minutes.

It writes transition logs, trade rows, parameter grid summaries, and approximate
IF futures compounding from 500k CNY.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    REPO_ROOT
    / "results"
    / "510300_breadth_regime"
    / "hs300_phase1_5m_90d_movegate_fixed_quality80_v1"
)


@dataclass
class Config:
    run_dir: str = str(DEFAULT_RUN_DIR)
    signal_file: str = "signals_5m_throttled.csv"
    eval_panel_file: str = str(
        REPO_ROOT
        / "results"
        / "510300_breadth_regime"
        / "hs300_phase1_5m_90d_strict_v1"
        / "eval_panel.parquet"
    )
    output_dir: str = str(DEFAULT_RUN_DIR / "runner_state_machine_v1")
    confirm_bar: int = 5
    confirm_bps: float = 10.0
    max_hold_bars: int = 30
    probe_stop_bps: float = 0.0
    runner_stop_bps: float = 0.0
    trail_activation_bps: float = 0.0
    trail_bps: float = 0.0
    allow_overlap: bool = False
    cost_bps: float = 4.5
    initial_capital: float = 500000.0
    if_multiplier: float = 300.0
    etf_to_index_scale: float = 1000.0
    grid_confirm_bps: str = "0,10,20,30"
    grid_max_hold_bars: str = "15,20,25,30"
    grid_probe_stop_bps: str = "0,25,40"
    grid_runner_stop_bps: str = "0,40"
    grid_trail_bps: str = "0,20,30"
    grid_trail_activation_bps: str = "30"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run HS300 runner state machine")
    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument("--signal-file", default=Config.signal_file)
    p.add_argument("--eval-panel-file", default=Config.eval_panel_file)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--confirm-bar", type=int, default=Config.confirm_bar)
    p.add_argument("--confirm-bps", type=float, default=Config.confirm_bps)
    p.add_argument("--max-hold-bars", type=int, default=Config.max_hold_bars)
    p.add_argument("--probe-stop-bps", type=float, default=Config.probe_stop_bps)
    p.add_argument("--runner-stop-bps", type=float, default=Config.runner_stop_bps)
    p.add_argument("--trail-activation-bps", type=float, default=Config.trail_activation_bps)
    p.add_argument("--trail-bps", type=float, default=Config.trail_bps)
    p.add_argument("--allow-overlap", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    p.add_argument("--cost-bps", type=float, default=Config.cost_bps)
    p.add_argument("--initial-capital", type=float, default=Config.initial_capital)
    p.add_argument("--if-multiplier", type=float, default=Config.if_multiplier)
    p.add_argument("--etf-to-index-scale", type=float, default=Config.etf_to_index_scale)
    p.add_argument("--grid-confirm-bps", default=Config.grid_confirm_bps)
    p.add_argument("--grid-max-hold-bars", default=Config.grid_max_hold_bars)
    p.add_argument("--grid-probe-stop-bps", default=Config.grid_probe_stop_bps)
    p.add_argument("--grid-runner-stop-bps", default=Config.grid_runner_stop_bps)
    p.add_argument("--grid-trail-bps", default=Config.grid_trail_bps)
    p.add_argument("--grid-trail-activation-bps", default=Config.grid_trail_activation_bps)
    return Config(**vars(p.parse_args()))


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def child_path(base: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else base / p


def load_inputs(cfg: Config) -> tuple[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]:
    run_dir = Path(cfg.run_dir)
    signals = pd.read_csv(run_dir / cfg.signal_file)
    signals["datetime"] = pd.to_datetime(signals["datetime"])
    signals["date"] = pd.to_datetime(signals["date"]).dt.normalize()
    signals = signals.sort_values(["date", "minute_idx", "datetime"]).reset_index(drop=True)

    panel = pd.read_parquet(child_path(run_dir, cfg.eval_panel_file), columns=["date", "datetime", "minute_idx", "etf_close"])
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["date", "minute_idx", "datetime"]).reset_index(drop=True)
    days = {date: day.reset_index(drop=True) for date, day in panel.groupby("date", sort=False)}
    return signals, days


def signed_bps(side: int, entry_close: float, px: float) -> float:
    if entry_close <= 0 or px <= 0:
        return np.nan
    return float(side) * float(np.log(px / entry_close) * 10000.0)


def add_event(events: list, signal_row: int, dt, bar: int, state_from: str, state_to: str, reason: str, pnl_bps: float) -> None:
    events.append(
        {
            "signal_row": signal_row,
            "datetime": dt,
            "bar": int(bar),
            "from_state": state_from,
            "to_state": state_to,
            "reason": reason,
            "signed_bps": float(pnl_bps),
        }
    )


def run_one_trade(signal_row: int, sig: pd.Series, day: pd.DataFrame, params: dict) -> tuple[dict | None, list]:
    hit = day.index[day["datetime"].eq(sig["datetime"])]
    if len(hit) == 0:
        return None, []
    entry_pos = int(hit[0])
    entry_close = float(day.loc[entry_pos, "etf_close"])
    side = int(sig["signal_side_5"])
    max_pos = min(entry_pos + int(params["max_hold_bars"]), len(day) - 1)
    if max_pos <= entry_pos:
        return None, []

    events = []
    add_event(events, signal_row, sig["datetime"], 0, "FLAT", "PROBE", "entry_signal", 0.0)
    state = "PROBE"
    peak_bps = -np.inf
    trough_bps = np.inf
    exit_row = None
    exit_reason = "end_of_data"
    confirmed = False

    for pos in range(entry_pos + 1, max_pos + 1):
        row = day.loc[pos]
        bar = int(pos - entry_pos)
        pnl_bps = signed_bps(side, entry_close, float(row["etf_close"]))
        peak_bps = max(peak_bps, pnl_bps)
        trough_bps = min(trough_bps, pnl_bps)

        if state == "PROBE":
            if params["probe_stop_bps"] > 0 and pnl_bps <= -params["probe_stop_bps"]:
                exit_row = row
                exit_reason = "probe_stop"
                add_event(events, signal_row, row["datetime"], bar, "PROBE", "EXIT", exit_reason, pnl_bps)
                break
            if bar >= params["confirm_bar"]:
                if pnl_bps >= params["confirm_bps"]:
                    state = "RUNNER"
                    confirmed = True
                    add_event(events, signal_row, row["datetime"], bar, "PROBE", "RUNNER", "confirm_pass", pnl_bps)
                else:
                    exit_row = row
                    exit_reason = "confirm_fail"
                    add_event(events, signal_row, row["datetime"], bar, "PROBE", "EXIT", exit_reason, pnl_bps)
                    break

        if state == "RUNNER":
            if params["runner_stop_bps"] > 0 and pnl_bps <= -params["runner_stop_bps"]:
                exit_row = row
                exit_reason = "runner_stop"
                add_event(events, signal_row, row["datetime"], bar, "RUNNER", "EXIT", exit_reason, pnl_bps)
                break
            if (
                params["trail_bps"] > 0
                and params["trail_activation_bps"] > 0
                and peak_bps >= params["trail_activation_bps"]
                and pnl_bps <= peak_bps - params["trail_bps"]
            ):
                exit_row = row
                exit_reason = "trailing_stop"
                add_event(events, signal_row, row["datetime"], bar, "RUNNER", "EXIT", exit_reason, pnl_bps)
                break
            if bar >= params["max_hold_bars"]:
                exit_row = row
                exit_reason = "time_exit"
                add_event(events, signal_row, row["datetime"], bar, "RUNNER", "EXIT", exit_reason, pnl_bps)
                break

    if exit_row is None:
        exit_row = day.loc[max_pos]
        bar = int(max_pos - entry_pos)
        pnl_bps = signed_bps(side, entry_close, float(exit_row["etf_close"]))
        add_event(events, signal_row, exit_row["datetime"], bar, state, "EXIT", exit_reason, pnl_bps)

    signed_ret = side * float(np.log(float(exit_row["etf_close"]) / entry_close))
    trade = {
        "signal_row": int(signal_row),
        "date": sig["date"],
        "entry_datetime": sig["datetime"],
        "exit_datetime": exit_row["datetime"],
        "entry_minute_idx": int(sig["minute_idx"]),
        "exit_minute_idx": int(exit_row["minute_idx"]),
        "entry_close": entry_close,
        "exit_close": float(exit_row["etf_close"]),
        "side": side,
        "signal_name_5": sig.get("signal_name_5", "LONG" if side > 0 else "SHORT"),
        "exit_bar": int(exit_row["minute_idx"] - sig["minute_idx"]),
        "confirmed_runner": bool(confirmed),
        "exit_reason": exit_reason,
        "signed_ret": signed_ret,
        "signed_bps": signed_ret * 10000.0,
        "net_bps": signed_ret * 10000.0 - params["cost_bps"],
        "path_peak_bps": float(peak_bps),
        "path_trough_bps": float(trough_bps),
        **{f"param_{k}": v for k, v in params.items()},
    }
    for col in ["GateScore_5", "DirScore_5", "MoveScore_quality", "MoveScore_core", "state"]:
        if col in sig.index:
            trade[col] = sig[col]
    return trade, events


def summarize_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    bps = trades["signed_bps"].astype(float)
    nz = bps[bps.abs() > 1e-9]
    return {
        "n": int(len(trades)),
        "runner_n": int(trades["confirmed_runner"].sum()),
        "runner_rate": float(trades["confirmed_runner"].mean()),
        "nonzero_n": int(len(nz)),
        "zero_rate": float(1.0 - len(nz) / len(trades)),
        "hit_rate": float((bps > 0).mean()),
        "nonzero_hit_rate": float((nz > 0).mean()) if len(nz) else np.nan,
        "mean_bps": float(bps.mean()),
        "net_mean_bps": float(trades["net_bps"].astype(float).mean()),
        "nonzero_mean_bps": float(nz.mean()) if len(nz) else np.nan,
        "median_bps": float(bps.median()),
        "p75_bps": float(bps.quantile(0.75)),
        "p90_bps": float(bps.quantile(0.90)),
        "avg_exit_bar": float(trades["exit_bar"].mean()),
    }


def run_state_machine(signals: pd.DataFrame, days: Dict[pd.Timestamp, pd.DataFrame], params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trades = []
    events = []
    skipped = []
    busy_until: Dict[pd.Timestamp, int] = {}

    for signal_row, sig in signals.iterrows():
        date = sig["date"]
        if date not in days:
            skipped.append({"signal_row": int(signal_row), "date": date, "reason": "missing_day"})
            continue
        if not params["allow_overlap"]:
            next_ok = busy_until.get(date, -1)
            if int(sig["minute_idx"]) < next_ok:
                skipped.append({"signal_row": int(signal_row), "date": date, "reason": "position_active"})
                continue
        trade, ev = run_one_trade(int(signal_row), sig, days[date], params)
        if trade is None:
            skipped.append({"signal_row": int(signal_row), "date": date, "reason": "no_trade_path"})
            continue
        trades.append(trade)
        events.extend(ev)
        if not params["allow_overlap"]:
            busy_until[date] = max(busy_until.get(date, -1), int(trade["exit_minute_idx"]) + 1)

    return pd.DataFrame(trades), pd.DataFrame(events), pd.DataFrame(skipped)


def simulate_if(trades: pd.DataFrame, cfg: Config, margin_rate: float, margin_utilization: float) -> pd.DataFrame:
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
                "exit_reason": trade["exit_reason"],
                "confirmed_runner": bool(trade["confirmed_runner"]),
                "side": int(trade["side"]),
                "underlying_signed_bps": float(trade["signed_bps"]),
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
        return {"ending_equity": initial_capital, "net_profit": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "n_trades": 0}
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


def if_summaries(trades: pd.DataFrame, cfg: Config, out_dir: Path, prefix: str) -> pd.DataFrame:
    scenarios = [
        ("IF_min_margin_8pct_util80", 0.08, 0.80),
        ("IF_conservative_12pct_util70", 0.12, 0.70),
    ]
    rows = []
    for name, margin, util in scenarios:
        curve = simulate_if(trades, cfg, margin, util)
        curve.to_csv(out_dir / f"{prefix}_{name}_curve.csv", index=False)
        rows.append({"scenario": f"{prefix}_{name}", **summarize_curve(curve, cfg.initial_capital)})
    return pd.DataFrame(rows)


def grid_params(cfg: Config) -> Iterable[dict]:
    for confirm_bps in parse_float_list(cfg.grid_confirm_bps):
        for max_hold in parse_int_list(cfg.grid_max_hold_bars):
            if max_hold <= cfg.confirm_bar:
                continue
            for probe_stop in parse_float_list(cfg.grid_probe_stop_bps):
                for runner_stop in parse_float_list(cfg.grid_runner_stop_bps):
                    for trail_bps in parse_float_list(cfg.grid_trail_bps):
                        activations = [0.0] if trail_bps <= 0 else parse_float_list(cfg.grid_trail_activation_bps)
                        for activation in activations:
                            yield {
                                "confirm_bar": cfg.confirm_bar,
                                "confirm_bps": confirm_bps,
                                "max_hold_bars": max_hold,
                                "probe_stop_bps": probe_stop,
                                "runner_stop_bps": runner_stop,
                                "trail_activation_bps": activation,
                                "trail_bps": trail_bps,
                                "allow_overlap": cfg.allow_overlap,
                                "cost_bps": cfg.cost_bps,
                            }


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signals, days = load_inputs(cfg)

    selected_params = {
        "confirm_bar": cfg.confirm_bar,
        "confirm_bps": cfg.confirm_bps,
        "max_hold_bars": cfg.max_hold_bars,
        "probe_stop_bps": cfg.probe_stop_bps,
        "runner_stop_bps": cfg.runner_stop_bps,
        "trail_activation_bps": cfg.trail_activation_bps,
        "trail_bps": cfg.trail_bps,
        "allow_overlap": cfg.allow_overlap,
        "cost_bps": cfg.cost_bps,
    }
    trades, events, skipped = run_state_machine(signals, days, selected_params)
    trades.to_csv(out_dir / "state_machine_trades.csv", index=False)
    events.to_csv(out_dir / "state_transitions.csv", index=False)
    skipped.to_csv(out_dir / "skipped_signals.csv", index=False)
    if_summary = if_summaries(trades, cfg, out_dir, "selected")
    if_summary.to_csv(out_dir / "if_summary.csv", index=False)

    grid_rows = []
    best_trades = pd.DataFrame()
    best_score = -np.inf
    best_params = None
    for params in grid_params(cfg):
        g_trades, _, _ = run_state_machine(signals, days, params)
        stats = summarize_trades(g_trades)
        if_curve = simulate_if(g_trades, cfg, 0.08, 0.80)
        curve_stats = summarize_curve(if_curve, cfg.initial_capital)
        score = curve_stats["net_profit"] / 10000.0 + curve_stats["max_drawdown"] * 100.0
        row = {
            **params,
            **{f"trade_{k}": v for k, v in stats.items()},
            **{f"if_{k}": v for k, v in curve_stats.items()},
            "score": float(score),
        }
        grid_rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params
            best_trades = g_trades
    grid = pd.DataFrame(grid_rows).sort_values(["score", "if_net_profit"], ascending=False).reset_index(drop=True)
    grid.to_csv(out_dir / "parameter_grid.csv", index=False)

    best_if = pd.DataFrame()
    if best_params is not None:
        best_trades.to_csv(out_dir / "best_grid_trades.csv", index=False)
        best_if = if_summaries(best_trades, cfg, out_dir, "best_grid")
        best_if.to_csv(out_dir / "best_grid_if_summary.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "selected_params": selected_params,
        "selected_trade_summary": summarize_trades(trades),
        "selected_if_summary": if_summary.to_dict("records"),
        "best_grid_params": best_params,
        "best_grid_trade_summary": summarize_trades(best_trades) if not best_trades.empty else {},
        "best_grid_if_summary": best_if.to_dict("records") if not best_if.empty else [],
        "artifacts": {
            "trades": str((out_dir / "state_machine_trades.csv").resolve()),
            "transitions": str((out_dir / "state_transitions.csv").resolve()),
            "parameter_grid": str((out_dir / "parameter_grid.csv").resolve()),
            "if_summary": str((out_dir / "if_summary.csv").resolve()),
            "best_grid_trades": str((out_dir / "best_grid_trades.csv").resolve()),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
