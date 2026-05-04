#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map HS300 Phase 1 5-minute signals to approximate IF futures compounding.

Inputs are the throttled signal files produced by run_hs300_phase1_5m_eval.py.
The script reports:
- continuous target-leverage compounding,
- integer IF contract compounding using ETF price as an index-level proxy,
- an oracle nonzero-only upper bound for diagnostics only.

Important: nonzero-only uses future realized returns and is not tradable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "510300_breadth_regime" / "hs300_phase1_5m_90d_strict_v1"


@dataclass
class Config:
    run_dir: str = str(DEFAULT_RUN_DIR)
    signal_file: str = "signals_5m_throttled.csv"
    eval_panel_file: str = "eval_panel.parquet"
    output_dir: str = str(DEFAULT_RUN_DIR / "if_compound_500k")
    initial_capital: float = 500000.0
    if_multiplier: float = 300.0
    etf_to_index_scale: float = 1000.0
    continuous_leverages: str = "1,3,5,8,10,12.5"
    cost_bps_list: str = "0,2,4.5,6"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Approximate IF futures compounding from HS300 Phase 1 signals")
    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument("--signal-file", default=Config.signal_file)
    p.add_argument("--eval-panel-file", default=Config.eval_panel_file)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--initial-capital", type=float, default=Config.initial_capital)
    p.add_argument("--if-multiplier", type=float, default=Config.if_multiplier)
    p.add_argument("--etf-to-index-scale", type=float, default=Config.etf_to_index_scale)
    p.add_argument("--continuous-leverages", default=Config.continuous_leverages)
    p.add_argument("--cost-bps-list", default=Config.cost_bps_list)
    return Config(**vars(p.parse_args()))


def load_signals(cfg: Config) -> pd.DataFrame:
    run_dir = Path(cfg.run_dir)
    sig = pd.read_csv(run_dir / cfg.signal_file)
    sig["datetime"] = pd.to_datetime(sig["datetime"])
    sig["date"] = pd.to_datetime(sig["date"]).dt.normalize()

    panel = pd.read_parquet(run_dir / cfg.eval_panel_file, columns=["date", "datetime", "etf_close"])
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    sig = sig.merge(panel, on=["date", "datetime"], how="left")
    sig = sig.sort_values("datetime").reset_index(drop=True)
    sig["if_index_px_proxy"] = sig["etf_close"] * cfg.etf_to_index_scale
    sig["if_notional_proxy"] = sig["if_index_px_proxy"] * cfg.if_multiplier
    return sig


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def summarize_equity(curve: pd.DataFrame, initial_capital: float) -> dict:
    if curve.empty:
        return {}
    eq = curve["equity"].astype(float)
    total_return = float(eq.iloc[-1] / initial_capital - 1.0)
    max_dd = float((eq / eq.cummax() - 1.0).min())
    pnl = curve["equity"].diff().fillna(curve["equity"] - initial_capital)
    return {
        "ending_equity": float(eq.iloc[-1]),
        "net_profit": float(eq.iloc[-1] - initial_capital),
        "total_return": total_return,
        "max_drawdown": max_dd,
        "n_trades": int(len(curve)),
        "win_rate_after_cost": float((pnl > 0).mean()) if len(pnl) else None,
    }


def simulate_continuous(signals: pd.DataFrame, initial_capital: float, leverage: float, cost_bps: float) -> pd.DataFrame:
    equity = initial_capital
    rows = []
    for _, r in signals.iterrows():
        signed_ret = float(r["signed_fwd_ret_5"])
        net_underlying_ret = signed_ret - cost_bps / 10000.0
        trade_ret = leverage * net_underlying_ret
        start = equity
        equity = equity * (1.0 + trade_ret)
        rows.append(
            {
                "datetime": r["datetime"],
                "date": r["date"],
                "side": r["signal_name_5"],
                "underlying_signed_ret": signed_ret,
                "cost_bps": cost_bps,
                "leverage": leverage,
                "start_equity": start,
                "trade_ret": trade_ret,
                "pnl": equity - start,
                "equity": equity,
            }
        )
    return pd.DataFrame(rows)


def simulate_integer_if(
    signals: pd.DataFrame,
    initial_capital: float,
    margin_rate: float,
    margin_utilization: float,
    cost_bps: float,
) -> pd.DataFrame:
    equity = initial_capital
    rows = []
    for _, r in signals.iterrows():
        notional = float(r["if_notional_proxy"])
        margin_per_lot = notional * margin_rate
        lots = int(np.floor(equity * margin_utilization / margin_per_lot)) if margin_per_lot > 0 else 0
        if lots <= 0:
            pnl = 0.0
            cost = 0.0
            gross = 0.0
            exposure = 0.0
            eff_lev = 0.0
        else:
            exposure = lots * notional
            eff_lev = exposure / equity
            gross = exposure * float(r["signed_fwd_ret_5"])
            cost = exposure * cost_bps / 10000.0
            pnl = gross - cost
        start = equity
        equity = equity + pnl
        rows.append(
            {
                "datetime": r["datetime"],
                "date": r["date"],
                "side": r["signal_name_5"],
                "underlying_signed_ret": float(r["signed_fwd_ret_5"]),
                "if_notional_proxy": notional,
                "lots": lots,
                "effective_leverage": eff_lev,
                "margin_rate": margin_rate,
                "margin_utilization": margin_utilization,
                "cost_bps": cost_bps,
                "start_equity": start,
                "gross_pnl": gross,
                "cost": cost,
                "pnl": pnl,
                "equity": equity,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    signals = load_signals(cfg)
    signals_nonzero = signals[signals["signed_fwd_ret_5"].abs() > 1e-12].copy()

    continuous_rows = []
    continuous_curves = {}
    for cost_bps in parse_float_list(cfg.cost_bps_list):
        for lev in parse_float_list(cfg.continuous_leverages):
            curve = simulate_continuous(signals, cfg.initial_capital, lev, cost_bps)
            key = f"continuous_L{lev:g}_cost{cost_bps:g}"
            continuous_curves[key] = curve
            continuous_rows.append({"scenario": key, **summarize_equity(curve, cfg.initial_capital)})

            oracle_curve = simulate_continuous(signals_nonzero, cfg.initial_capital, lev, cost_bps)
            oracle_key = f"ORACLE_nonzero_continuous_L{lev:g}_cost{cost_bps:g}"
            continuous_rows.append({"scenario": oracle_key, **summarize_equity(oracle_curve, cfg.initial_capital)})

    integer_scenarios = [
        {"name": "IF_min_margin_8pct_util80_cost2bps", "margin_rate": 0.08, "margin_utilization": 0.80, "cost_bps": 2.0},
        {"name": "IF_min_margin_8pct_util80_cost4p5bps", "margin_rate": 0.08, "margin_utilization": 0.80, "cost_bps": 4.5},
        {"name": "IF_conservative_12pct_util70_cost2bps", "margin_rate": 0.12, "margin_utilization": 0.70, "cost_bps": 2.0},
        {"name": "IF_conservative_12pct_util70_cost4p5bps", "margin_rate": 0.12, "margin_utilization": 0.70, "cost_bps": 4.5},
    ]
    integer_rows = []
    for sc in integer_scenarios:
        curve = simulate_integer_if(
            signals,
            cfg.initial_capital,
            margin_rate=sc["margin_rate"],
            margin_utilization=sc["margin_utilization"],
            cost_bps=sc["cost_bps"],
        )
        curve.to_csv(out_dir / f"{sc['name']}_curve.csv", index=False)
        integer_rows.append(
            {
                "scenario": sc["name"],
                **summarize_equity(curve, cfg.initial_capital),
                "avg_effective_leverage": float(curve["effective_leverage"].mean()) if not curve.empty else None,
                "median_lots": float(curve["lots"].median()) if not curve.empty else None,
                "max_lots": int(curve["lots"].max()) if not curve.empty else 0,
            }
        )

        oracle_curve = simulate_integer_if(
            signals_nonzero,
            cfg.initial_capital,
            margin_rate=sc["margin_rate"],
            margin_utilization=sc["margin_utilization"],
            cost_bps=sc["cost_bps"],
        )
        integer_rows.append(
            {
                "scenario": "ORACLE_nonzero_" + sc["name"],
                **summarize_equity(oracle_curve, cfg.initial_capital),
                "avg_effective_leverage": float(oracle_curve["effective_leverage"].mean()) if not oracle_curve.empty else None,
                "median_lots": float(oracle_curve["lots"].median()) if not oracle_curve.empty else None,
                "max_lots": int(oracle_curve["lots"].max()) if not oracle_curve.empty else 0,
            }
        )

    continuous_summary = pd.DataFrame(continuous_rows)
    integer_summary = pd.DataFrame(integer_rows)
    continuous_summary.to_csv(out_dir / "continuous_leverage_summary.csv", index=False)
    integer_summary.to_csv(out_dir / "integer_if_summary.csv", index=False)
    signals.to_csv(out_dir / "input_signals_with_notional.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "input_signal_count": int(len(signals)),
                "input_nonzero_count": int(len(signals_nonzero)),
                "continuous_summary": continuous_summary.to_dict("records"),
                "integer_if_summary": integer_summary.to_dict("records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Continuous leverage summary")
    print(continuous_summary.to_string(index=False))
    print("\nInteger IF summary")
    print(integer_summary.to_string(index=False))
    print(f"\nSaved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

