#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward MoveGate filter for HS300 Phase 1 5-minute signals.

MoveGate answers a separate question from direction:
    "Is the next 5-minute move likely large enough to justify trading?"

This script uses only trailing days to select:
- one MoveScore formula,
- one threshold quantile,
then applies it to the next day. It can also fit once on the non-evaluation
warmup window and apply that fixed MoveGate to the evaluation window.
It produces signal files compatible with simulate_hs300_phase1_if_compound.py.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "510300_breadth_regime" / "hs300_phase1_5m_60d_strict_v1"


@dataclass
class Config:
    run_dir: str = str(DEFAULT_RUN_DIR)
    eval_panel_file: str = "eval_panel.parquet"
    output_dir: str = str(DEFAULT_RUN_DIR / "movegate_wf_cost4p5_v1")
    mode: str = "walkforward"
    train_days: int = 25
    min_hist_signals: int = 10
    min_hist_selected: int = 5
    optimize_cost_bps: float = 4.5
    dir_abs_threshold: float = 1.6
    long_dir_threshold: float = -1.0
    short_dir_threshold: float = -1.0
    cooldown_bars: int = 5
    allow_no_trade_when_no_positive_hist: bool = True
    fixed_move_score_col: str = "MoveScore_quality"
    fixed_move_quantile: float = 0.85


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Apply walk-forward MoveGate to HS300 Phase 1 signals")
    p.add_argument("--run-dir", default=Config.run_dir)
    p.add_argument("--eval-panel-file", default=Config.eval_panel_file)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--mode", choices=["walkforward", "fixed_warmup", "warmup_select"], default=Config.mode)
    p.add_argument("--train-days", type=int, default=Config.train_days)
    p.add_argument("--min-hist-signals", type=int, default=Config.min_hist_signals)
    p.add_argument("--min-hist-selected", type=int, default=Config.min_hist_selected)
    p.add_argument("--optimize-cost-bps", type=float, default=Config.optimize_cost_bps)
    p.add_argument("--dir-abs-threshold", type=float, default=Config.dir_abs_threshold)
    p.add_argument("--long-dir-threshold", type=float, default=Config.long_dir_threshold)
    p.add_argument("--short-dir-threshold", type=float, default=Config.short_dir_threshold)
    p.add_argument("--cooldown-bars", type=int, default=Config.cooldown_bars)
    p.add_argument(
        "--allow-no-trade-when-no-positive-hist",
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    p.add_argument("--fixed-move-score-col", default=Config.fixed_move_score_col)
    p.add_argument("--fixed-move-quantile", type=float, default=Config.fixed_move_quantile)
    return Config(**vars(p.parse_args()))


def regular_signal_mask(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["datetime"])
    hhmm = dt.dt.hour * 100 + dt.dt.minute
    return (hhmm >= 935) & ~((hhmm >= 1125) & (hhmm <= 1305)) & (hhmm < 1450)


def add_move_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Core "enough energy to move" score: structural gate + directional pressure
    # + price path + cross-sectional cohesion, penalizing concentration.
    out["MoveScore_core"] = (
        0.35 * out["GateScore_5"].fillna(0.0)
        + 0.20 * out["DirScore_5"].abs().fillna(0.0)
        + 0.15 * out["IPG_5"].abs().fillna(0.0)
        + 0.10 * out["z_Dir_5_raw"].fillna(0.0)
        + 0.10 * out["z_Corr_5_raw"].fillna(0.0)
        + 0.10 * out["z_EP_5_raw"].fillna(0.0)
        - 0.10 * out["z_TopK_5_raw"].fillna(0.0)
    )
    # Breakout-flavored score: stronger bias toward price path and broad pressure.
    out["MoveScore_breakout"] = (
        0.30 * out["GateScore_5"].fillna(0.0)
        + 0.25 * out["DirScore_5"].abs().fillna(0.0)
        + 0.20 * out["IPG_5"].abs().fillna(0.0)
        + 0.15 * out["z_Dir_5_raw"].fillna(0.0)
        + 0.10 * out["z_B_5_raw"].fillna(0.0)
    )
    # Quality score: prefers cleaner breadth and sector diffusion, less top-heavy.
    out["MoveScore_quality"] = (
        0.25 * out["GateScore_5"].fillna(0.0)
        + 0.20 * out["z_Dir_5_raw"].fillna(0.0)
        + 0.20 * out["z_Corr_5_raw"].fillna(0.0)
        + 0.15 * out["z_EP_5_raw"].fillna(0.0)
        + 0.10 * out["z_SE_5_raw"].fillna(0.0)
        - 0.10 * out["z_TopK_5_raw"].fillna(0.0)
    )
    # Volatility/impulse score: intentionally aggressive.
    out["MoveScore_impulse"] = (
        0.30 * out["DirScore_5"].abs().fillna(0.0)
        + 0.25 * out["IPG_5"].abs().fillna(0.0)
        + 0.20 * out["z_Dir_5_raw"].abs().fillna(0.0)
        + 0.15 * out["z_ETFRet_5_raw"].abs().fillna(0.0)
        + 0.10 * out["GateScore_5"].fillna(0.0)
    )
    return out


def base_signal_mask(df: pd.DataFrame, cfg: Config) -> pd.Series:
    long_th = cfg.dir_abs_threshold if cfg.long_dir_threshold < 0 else cfg.long_dir_threshold
    short_th = cfg.dir_abs_threshold if cfg.short_dir_threshold < 0 else cfg.short_dir_threshold
    direction_ok = (df["DirScore_5"] >= long_th) | (df["DirScore_5"] <= -short_th)
    return (
        regular_signal_mask(df)
        & df["fwd_ret_5"].notna()
        & (df["Gate_5"])
        & (df["Exhaustion_5"] < 0.80)
        & direction_ok
        & ((df["DirScore_5"] * df["DirScore_10"]) >= -0.10)
    )


def prepare_base(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    work = add_move_scores(df)
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    base = work[base_signal_mask(work, cfg)].copy()
    base["candidate_side"] = np.sign(base["DirScore_5"]).astype(int)
    base["signal_side_5"] = base["candidate_side"]
    base["signal_name_5"] = np.where(base["candidate_side"] > 0, "LONG", "SHORT")
    base["signed_fwd_ret_5"] = base["candidate_side"] * base["fwd_ret_5"]
    base["signal_strength_5"] = base["DirScore_5"].abs()
    return base


def evaluate_selected(df: pd.DataFrame, cost_bps: float) -> dict:
    if df.empty:
        return {
            "n": 0,
            "nonzero_n": 0,
            "mean_net_bps": np.nan,
            "mean_signed_bps": np.nan,
            "nonzero_hit_rate": np.nan,
            "score": -1e18,
        }
    signed = df["candidate_side"] * df["fwd_ret_5"]
    net_bps = signed * 10000.0 - cost_bps
    nonzero = signed.abs() > 1e-12
    mean_net = float(net_bps.mean())
    mean_signed = float(signed.mean() * 10000.0)
    nonzero_hit = float((signed[nonzero] > 0).mean()) if nonzero.any() else np.nan
    # Favor positive mean after cost, enough samples, and nonzero hit rate.
    score = mean_net * np.sqrt(len(df)) + 3.0 * max(float(nonzero.sum()), 0.0)
    if np.isfinite(nonzero_hit):
        score += 20.0 * (nonzero_hit - 0.50)
    return {
        "n": int(len(df)),
        "nonzero_n": int(nonzero.sum()),
        "mean_net_bps": mean_net,
        "mean_signed_bps": mean_signed,
        "nonzero_hit_rate": nonzero_hit,
        "score": float(score),
    }


def candidate_grid() -> List[dict]:
    rows = []
    for score_col in ["MoveScore_core", "MoveScore_breakout", "MoveScore_quality", "MoveScore_impulse"]:
        for q in [0.00, 0.30, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90]:
            rows.append({"move_score_col": score_col, "move_quantile": q})
    return rows


def pick_candidate(hist: pd.DataFrame, cfg: Config) -> dict | None:
    if len(hist) < cfg.min_hist_signals:
        return None
    best = None
    best_stats = None
    for cand in candidate_grid():
        score_col = cand["move_score_col"]
        q = cand["move_quantile"]
        vals = hist[score_col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        threshold = float(vals.quantile(q))
        selected = hist[hist[score_col] >= threshold].copy()
        if len(selected) < cfg.min_hist_selected:
            continue
        stats = evaluate_selected(selected, cfg.optimize_cost_bps)
        if best_stats is None or stats["score"] > best_stats["score"]:
            best_stats = stats
            best = {
                **cand,
                "move_threshold": threshold,
                **{f"hist_{k}": v for k, v in stats.items() if k != "score"},
                "hist_score": stats["score"],
            }
    if best is None:
        return None
    if cfg.allow_no_trade_when_no_positive_hist and best.get("hist_mean_net_bps", -1e9) <= 0:
        return None
    return best


def throttle_signals(signals: pd.DataFrame, cooldown_bars: int) -> pd.DataFrame:
    rows = []
    for _, day in signals.sort_values(["date", "minute_idx"]).groupby("date", sort=True):
        next_ok = -1
        for idx, row in day.iterrows():
            minute_idx = int(row["minute_idx"])
            if minute_idx < next_ok:
                continue
            rows.append(idx)
            next_ok = minute_idx + cooldown_bars
    return signals.loc[rows].copy() if rows else signals.iloc[0:0].copy()


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0}
    signed = df["signed_fwd_ret_5"]
    nz = df[signed.abs() > 1e-12]
    return {
        "n": int(len(df)),
        "days": int(df["date"].nunique()),
        "nonzero_n": int(len(nz)),
        "zero_rate": float(1.0 - len(nz) / len(df)) if len(df) else None,
        "hit_rate": float((signed > 0).mean()),
        "nonzero_hit_rate": float((nz["signed_fwd_ret_5"] > 0).mean()) if len(nz) else None,
        "mean_signed_bps": float(signed.mean() * 10000.0),
        "nonzero_mean_signed_bps": float(nz["signed_fwd_ret_5"].mean() * 10000.0) if len(nz) else None,
        "median_signed_bps": float(signed.median() * 10000.0),
        "p75_signed_bps": float(signed.quantile(0.75) * 10000.0),
    }


def apply_walkforward(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prepare_base(df, cfg)

    all_dates = sorted(base["date"].unique())
    selected_parts = []
    decision_rows = []
    for i, date in enumerate(all_dates):
        hist_dates = all_dates[max(0, i - cfg.train_days) : i]
        if not hist_dates:
            continue
        hist = base[base["date"].isin(hist_dates)].copy()
        today = base[base["date"] == date].copy()
        if today.empty:
            continue
        cand = pick_candidate(hist, cfg)
        if cand is None:
            decision_rows.append({"date": date, "selected": False, "today_base_signals": int(len(today))})
            continue
        score_col = cand["move_score_col"]
        threshold = cand["move_threshold"]
        selected = today[today[score_col] >= threshold].copy()
        for k, v in cand.items():
            selected[k] = v
        selected["movegate_pass"] = True
        selected_parts.append(selected)
        decision_rows.append(
            {
                "date": date,
                "selected": True,
                "today_base_signals": int(len(today)),
                "today_selected_signals": int(len(selected)),
                **cand,
            }
        )
    signals = pd.concat(selected_parts, ignore_index=True) if selected_parts else base.iloc[0:0].copy()
    decisions = pd.DataFrame(decision_rows)
    return signals, decisions


def apply_fixed_warmup(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prepare_base(df, cfg)

    if "is_eval" not in base.columns:
        dates = sorted(base["date"].unique())
        cutoff = dates[max(0, len(dates) - cfg.train_days)]
        base["is_eval"] = base["date"] >= cutoff

    warmup = base[~base["is_eval"]].copy()
    test = base[base["is_eval"]].copy()
    score_col = cfg.fixed_move_score_col
    vals = warmup[score_col].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        threshold = float(test[score_col].quantile(cfg.fixed_move_quantile))
    else:
        threshold = float(vals.quantile(cfg.fixed_move_quantile))
    selected = test[test[score_col] >= threshold].copy()
    selected["move_score_col"] = score_col
    selected["move_quantile"] = cfg.fixed_move_quantile
    selected["move_threshold"] = threshold
    selected["movegate_pass"] = True
    hist_stats = evaluate_selected(warmup[warmup[score_col] >= threshold].copy(), cfg.optimize_cost_bps)
    decisions = pd.DataFrame(
        [
            {
                "mode": "fixed_warmup",
                "move_score_col": score_col,
                "move_quantile": cfg.fixed_move_quantile,
                "move_threshold": threshold,
                "warmup_base_signals": int(len(warmup)),
                "eval_base_signals": int(len(test)),
                "eval_selected_signals": int(len(selected)),
                **{f"warmup_{k}": v for k, v in hist_stats.items()},
            }
        ]
    )
    return selected, decisions


def apply_warmup_select(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prepare_base(df, cfg)
    if "is_eval" not in base.columns:
        dates = sorted(base["date"].unique())
        cutoff = dates[max(0, len(dates) - cfg.train_days)]
        base["is_eval"] = base["date"] >= cutoff

    warmup = base[~base["is_eval"]].copy()
    test = base[base["is_eval"]].copy()
    cand = pick_candidate(warmup, cfg)
    if cand is None:
        decisions = pd.DataFrame(
            [
                {
                    "mode": "warmup_select",
                    "selected": False,
                    "warmup_base_signals": int(len(warmup)),
                    "eval_base_signals": int(len(test)),
                    "eval_selected_signals": 0,
                }
            ]
        )
        return test.iloc[0:0].copy(), decisions

    score_col = cand["move_score_col"]
    threshold = cand["move_threshold"]
    selected = test[test[score_col] >= threshold].copy()
    for k, v in cand.items():
        selected[k] = v
    selected["movegate_pass"] = True
    decisions = pd.DataFrame(
        [
            {
                "mode": "warmup_select",
                "selected": True,
                "warmup_base_signals": int(len(warmup)),
                "eval_base_signals": int(len(test)),
                "eval_selected_signals": int(len(selected)),
                **cand,
            }
        ]
    )
    return selected, decisions


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = Path(cfg.run_dir) / cfg.eval_panel_file
    panel = pd.read_parquet(panel_path)

    if cfg.mode == "fixed_warmup":
        signals, decisions = apply_fixed_warmup(panel, cfg)
    elif cfg.mode == "warmup_select":
        signals, decisions = apply_warmup_select(panel, cfg)
    else:
        signals, decisions = apply_walkforward(panel, cfg)
    throttled = throttle_signals(signals, cfg.cooldown_bars)

    signal_cols = [
        "date",
        "datetime",
        "minute_idx",
        "signal_name_5",
        "signal_side_5",
        "state",
        "signal_strength_5",
        "GateScore_5",
        "DirScore_5",
        "DirScore_10",
        "Exhaustion_5",
        "IPG_5",
        "MoveScore_core",
        "MoveScore_breakout",
        "MoveScore_quality",
        "MoveScore_impulse",
        "move_score_col",
        "move_quantile",
        "move_threshold",
        "fwd_ret_5",
        "signed_fwd_ret_5",
    ]
    signal_cols = [c for c in signal_cols if c in signals.columns]

    signals.to_csv(out_dir / "signals_5m.csv", index=False)
    throttled[signal_cols].to_csv(out_dir / "signals_5m_throttled.csv", index=False)
    decisions.to_csv(out_dir / "movegate_daily_decisions.csv", index=False)
    summary = {
        "config": asdict(cfg),
        "raw_movegate_signal": summarize(signals),
        "throttled_signal": summarize(throttled),
        "artifacts": {
            "signals_5m": str((out_dir / "signals_5m.csv").resolve()),
            "signals_5m_throttled": str((out_dir / "signals_5m_throttled.csv").resolve()),
            "movegate_daily_decisions": str((out_dir / "movegate_daily_decisions.csv").resolve()),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
