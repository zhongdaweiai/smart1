#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 14.5 — Wave Consensus walk-forward.

Test whether layering a multi-projection consensus filter on top of the V1
HS300 downside propagation rule (ipg0_oos_strong) improves OOS edge.

Inputs:
- V1 scored panel (input-panel-files comma-list, same flag as
  run_hs300_downside_walkforward.py).
- Multi-projection panel produced by compute_multiproj_features.py.

Pipeline:
1. Inner-join V1 scored panel with multiproj panel on (date, minute_idx).
2. Compute kinematic derivatives (IPG and breadth velocity) per-day.
3. Define a 6-element wave-consensus score for the SHORT side, using
   train-only q-thresholds for the "is this strong enough" cutoffs.
4. For consensus level k = 0..6, run the V1 ipg0_oos_strong rule
   intersected with consensus_short >= k. Record train, OOS, and
   walk-forward metrics. Also runs each consensus dimension alone
   (ablation) so we know which projection carries the lift.
5. Pick the best consensus level by an OOS-resilient score (median
   net-bps minus a small penalty for trade count below 6) and emit
   trade logs and IF compounding artefacts for it.

Hard rules:
- All q-thresholds for the consensus features are fit on train rows
  matching the base ipg0_oos_strong eligibility (state EMERGING_DOWN,
  minute_idx>=30, DirScore_5<=-0.75, DirScore_10<=0.45, regular_time).
  No OOS leakage.
- Entry uses next-minute-open after the signal bar; exit at open after
  hold_bars. Same overlap and per-day cap as V1.
- Round-trip cost 4.5 bps. IF proxy mirrors the V1 multiplier and
  margin assumptions.
- Walk-forward refits the q-thresholds inside each train fold.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from run_hs300_downside_walkforward import (
    PROJECT_ROOT,
    REPO_ROOT,
    Config as DownsideConfig,
    candidate_thresholds,
    compound_if,
    daily_bps_series,
    save_trade_set,
    simulate_short_trades,
    summarize_bps,
)
from run_hs300_phase1_5m_eval import regular_signal_mask


BASE_CANDIDATE = {
    "rule": "ED",
    "states": "EMERGING_DOWN",
    "require_highconf": 0,
    "minute_min": 30,
    "hold_bars": 30,
    "dir5_abs_min": 0.75,
    "dir10_max": 0.45,
    "exh10_q": 0.90,
    "ipg10_q": 0.40,
}


CONSENSUS_DIMS = [
    "cs_sector_down",
    "cs_bucket_top_neg",
    "cs_bucket_penetration",
    "cs_ew_negative",
    "cs_ipg_velocity_up",
    "cs_wavefront_down",
]


@dataclass
class Config:
    input_panel_files: str = ""
    multiproj_panel: str = str(
        REPO_ROOT
        / "results"
        / "510300_breadth_regime"
        / "hs300_multiproj_panel_v1"
        / "multiproj_panel.parquet"
    )
    output_dir: str = str(
        REPO_ROOT
        / "results"
        / "510300_breadth_regime"
        / "hs300_phase14_5_consensus_v1"
    )
    oos_start: str = "2026-03-06"
    velocity_lag_minutes: int = 3
    sector_down_q: float = 0.50
    bucket_top_q: float = 0.50
    bucket_penetration_q: float = 0.50
    ew_signed_q: float = 0.50
    ipg_velocity_q: float = 0.50
    wavefront_q: float = 0.50
    round_trip_cost_bps: float = 4.5
    initial_capital: float = 500000.0
    if_margin_rate: float = 0.08
    if_utilization: float = 0.80
    if_multiplier: float = 300.0
    etf_to_index_scale: float = 1000.0
    max_trades_per_day: int = 2
    min_train_trades: int = 24
    wf_train_days: int = 120
    wf_test_days: int = 20


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Phase 14.5 wave-consensus walk-forward")
    for fld, default in Config().__dict__.items():
        arg = "--" + fld.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(arg, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=default)
        elif isinstance(default, int):
            p.add_argument(arg, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(arg, type=float, default=default)
        else:
            p.add_argument(arg, default=default)
    return Config(**vars(p.parse_args()))


def load_v1_panel(input_panel_files: str) -> pd.DataFrame:
    if not input_panel_files.strip():
        raise ValueError("--input-panel-files is required")
    parts: list[pd.DataFrame] = []
    for raw in input_panel_files.split(","):
        path = Path(raw.strip())
        if not path.is_absolute():
            path = REPO_ROOT / path
        part = pd.read_parquet(path)
        if "is_eval" in part.columns:
            part = part[part["is_eval"].astype(bool)].copy()
        parts.append(part)
    panel = pd.concat(parts, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = (
        panel.sort_values(["date", "minute_idx", "datetime"])
        .drop_duplicates(["date", "minute_idx"], keep="last")
        .reset_index(drop=True)
    )
    if "regular_time" not in panel.columns:
        panel["regular_time"] = regular_signal_mask(panel).astype(int)
    return panel


def load_multiproj_panel(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"multiproj panel not found: {p}")
    panel = pd.read_parquet(p)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    drop_cols = [c for c in ("datetime", "weight_date", "active_now_total") if c in panel.columns]
    panel = panel.drop(columns=drop_cols)
    return panel


def join_panels(v1: pd.DataFrame, multi: pd.DataFrame) -> pd.DataFrame:
    joined = v1.merge(multi, on=["date", "minute_idx"], how="inner", suffixes=("", "_mp"))
    if joined.empty:
        raise RuntimeError("V1 panel and multiproj panel produced an empty join")
    return joined


def add_kinematics(panel: pd.DataFrame, lag: int) -> pd.DataFrame:
    out = panel.sort_values(["date", "minute_idx"]).reset_index(drop=True).copy()
    grouped = out.groupby("date", sort=False, group_keys=False)
    out["IPG_velocity_10"] = grouped["IPG_10"].diff(lag)
    out["IPG_acceleration_10"] = grouped["IPG_velocity_10"].diff(lag)
    if "z_B_5_raw" in out.columns:
        out["B_velocity_5"] = grouped["z_B_5_raw"].diff(lag)
    return out


def base_eligible_mask(panel: pd.DataFrame, cand: dict) -> pd.Series:
    states = {x for x in str(cand["states"]).split("|") if x}
    mask = panel["state"].isin(states)
    mask &= panel["regular_time"].astype(bool)
    mask &= panel["minute_idx"].astype(int) >= int(cand["minute_min"])
    mask &= panel["DirScore_5"].astype(float) <= -float(cand["dir5_abs_min"])
    mask &= panel["DirScore_10"].astype(float) <= float(cand["dir10_max"])
    if bool(cand.get("require_highconf", 0)):
        mask &= panel["signal_side_5"].astype(int) == -1
    return mask


def fit_consensus_thresholds(train: pd.DataFrame, cfg: Config) -> dict:
    eligible = train[base_eligible_mask(train, BASE_CANDIDATE)]
    if eligible.empty:
        raise RuntimeError("no rows pass base eligibility on train; cannot fit thresholds")
    th: dict = {}
    th["sector_down_frac_min"] = float(np.nanquantile(eligible["Sector_Down_Frac_10"].astype(float), cfg.sector_down_q))
    th["bucket_top_max"] = float(np.nanquantile(eligible["Bucket_Top_Sign_10"].astype(float), 1.0 - cfg.bucket_top_q))
    th["bucket_penetration_max"] = float(np.nanquantile(eligible["Bucket_Penetration_10"].astype(float), 1.0 - cfg.bucket_penetration_q))
    th["ew_signed_max"] = float(np.nanquantile(eligible["EW_SignedBreadth_10"].astype(float), 1.0 - cfg.ew_signed_q))
    th["ipg_velocity_min"] = float(np.nanquantile(eligible["IPG_velocity_10"].astype(float), cfg.ipg_velocity_q))
    th["wavefront_down_min"] = float(np.nanquantile(eligible["WavefrontDown_Frac_5"].astype(float), cfg.wavefront_q))
    th["n_train_eligible"] = int(len(eligible))
    return th


def consensus_dims(panel: pd.DataFrame, th: dict) -> pd.DataFrame:
    df = panel.copy()
    df["cs_sector_down"] = (df["Sector_Down_Frac_10"].astype(float) >= th["sector_down_frac_min"]).astype(int)
    df["cs_bucket_top_neg"] = (df["Bucket_Top_Sign_10"].astype(float) <= th["bucket_top_max"]).astype(int)
    df["cs_bucket_penetration"] = (df["Bucket_Penetration_10"].astype(float) <= th["bucket_penetration_max"]).astype(int)
    df["cs_ew_negative"] = (df["EW_SignedBreadth_10"].astype(float) <= th["ew_signed_max"]).astype(int)
    df["cs_ipg_velocity_up"] = (df["IPG_velocity_10"].astype(float) >= th["ipg_velocity_min"]).astype(int)
    df["cs_wavefront_down"] = (df["WavefrontDown_Frac_5"].astype(float) >= th["wavefront_down_min"]).astype(int)
    df["consensus_short"] = df[CONSENSUS_DIMS].sum(axis=1).astype(int)
    return df


def candidate_with_consensus(min_consensus: int) -> dict:
    cand = dict(BASE_CANDIDATE)
    cand["min_consensus_short"] = int(min_consensus)
    return cand


def consensus_simulate_trades(
    panel: pd.DataFrame,
    base_cand: dict,
    base_thresholds: dict,
    min_consensus: int,
    sim_cfg: DownsideConfig,
) -> pd.DataFrame:
    panel = panel.copy()
    panel["consensus_pass"] = (panel["consensus_short"].astype(int) >= int(min_consensus)).astype(int)

    base_eligible = base_eligible_mask(panel, base_cand) & panel["consensus_pass"].astype(bool)
    if np.isfinite(base_thresholds.get("exh10_max", np.nan)):
        base_eligible &= panel["Exhaustion_10"].astype(float) <= float(base_thresholds["exh10_max"])
    if np.isfinite(base_thresholds.get("ipg10_min", np.nan)):
        base_eligible &= panel["IPG_10"].astype(float) >= float(base_thresholds["ipg10_min"])

    if not base_eligible.any():
        return pd.DataFrame()

    rows: list[dict] = []
    hold = int(base_cand["hold_bars"])
    for date, day_panel in panel.sort_values(["date", "minute_idx"]).groupby("date", sort=True):
        day = day_panel.reset_index(drop=True)
        day_mask = base_eligible.loc[day_panel.index].reset_index(drop=True)
        idxs = day.index[day_mask].tolist()
        if not idxs:
            continue
        next_ok_pos = -1
        trades_today = 0
        for signal_pos in idxs:
            if signal_pos < next_ok_pos or trades_today >= int(sim_cfg.max_trades_per_day):
                continue
            entry_pos = signal_pos + 1
            exit_pos = entry_pos + hold
            if exit_pos >= len(day):
                continue
            entry_px = float(day.loc[entry_pos, "etf_open"])
            exit_px = float(day.loc[exit_pos, "etf_open"])
            if entry_px <= 0 or exit_px <= 0:
                continue
            gross_log = -math.log(exit_px / entry_px)
            gross_bps = gross_log * 10000.0
            rows.append(
                {
                    "date": pd.Timestamp(date).normalize(),
                    "signal_datetime": day.loc[signal_pos, "datetime"],
                    "entry_datetime": day.loc[entry_pos, "datetime"],
                    "exit_datetime": day.loc[exit_pos, "datetime"],
                    "signal_minute_idx": int(day.loc[signal_pos, "minute_idx"]),
                    "entry_minute_idx": int(day.loc[entry_pos, "minute_idx"]),
                    "exit_minute_idx": int(day.loc[exit_pos, "minute_idx"]),
                    "hold_bars": hold,
                    "side": -1,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "gross_bps": gross_bps,
                    "net_bps": gross_bps - float(sim_cfg.round_trip_cost_bps),
                    "consensus_short": int(day.loc[signal_pos, "consensus_short"]),
                    "DirScore_5": float(day.loc[signal_pos, "DirScore_5"]),
                    "DirScore_10": float(day.loc[signal_pos, "DirScore_10"]),
                    "Exhaustion_10": float(day.loc[signal_pos, "Exhaustion_10"]),
                    "IPG_10": float(day.loc[signal_pos, "IPG_10"]),
                    "Sector_Down_Frac_10": float(day.loc[signal_pos, "Sector_Down_Frac_10"]),
                    "Bucket_Top_Sign_10": float(day.loc[signal_pos, "Bucket_Top_Sign_10"]),
                    "Bucket_Penetration_10": float(day.loc[signal_pos, "Bucket_Penetration_10"]),
                    "EW_SignedBreadth_10": float(day.loc[signal_pos, "EW_SignedBreadth_10"]),
                    "IPG_velocity_10": float(day.loc[signal_pos, "IPG_velocity_10"]),
                    "WavefrontDown_Frac_5": float(day.loc[signal_pos, "WavefrontDown_Frac_5"]),
                }
            )
            next_ok_pos = exit_pos
            trades_today += 1
    return pd.DataFrame(rows)


def evaluate_segment(
    panel: pd.DataFrame,
    base_cand: dict,
    base_thresholds: dict,
    min_consensus: int,
    sim_cfg: DownsideConfig,
) -> dict:
    trades = consensus_simulate_trades(panel, base_cand, base_thresholds, min_consensus, sim_cfg)
    bps_summary = summarize_bps(panel, trades)
    if_daily, if_summary = compound_if(panel, trades, sim_cfg)
    return {
        "min_consensus_short": int(min_consensus),
        "trades": trades,
        "if_daily": if_daily,
        "bps_summary": bps_summary,
        "if_summary": if_summary,
    }


def metrics_row(label: str, segment: str, result: dict) -> dict:
    bps = result["bps_summary"]
    if_s = result["if_summary"]
    return {
        "label": label,
        "segment": segment,
        "min_consensus_short": result["min_consensus_short"],
        "n_trades": int(bps.get("n_trades", 0)),
        "n_trade_days": int(bps.get("n_trade_days", 0)),
        "win_rate": bps.get("win_rate"),
        "avg_net_bps": bps.get("avg_net_bps"),
        "median_net_bps": bps.get("median_net_bps"),
        "p25_net_bps": bps.get("p25_net_bps"),
        "p75_net_bps": bps.get("p75_net_bps"),
        "first_half_avg_net_bps": bps.get("first_half_avg_net_bps"),
        "second_half_avg_net_bps": bps.get("second_half_avg_net_bps"),
        "sharpe_unlevered": bps.get("sharpe_unlevered"),
        "max_drawdown_unlevered": bps.get("max_drawdown_unlevered"),
        "if_initial_capital": if_s.get("initial_capital"),
        "if_ending_equity": if_s.get("ending_equity"),
        "if_total_return": if_s.get("total_return"),
        "if_annual_return": if_s.get("annual_return"),
        "if_sharpe": if_s.get("sharpe"),
        "if_max_drawdown": if_s.get("max_drawdown"),
        "if_calmar": if_s.get("calmar"),
        "if_trade_days": if_s.get("trade_days"),
        "if_avg_contracts_on_trade_day": if_s.get("avg_contracts_on_trade_day"),
    }


def fit_base_thresholds(train: pd.DataFrame) -> dict:
    return candidate_thresholds(train, pd.Series(BASE_CANDIDATE))


def consensus_walkforward(panel: pd.DataFrame, sim_cfg: DownsideConfig, cfg: Config, min_consensus: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rolling walk-forward: refit base thresholds and consensus thresholds on each train fold."""
    panel = panel.sort_values(["date", "minute_idx"]).reset_index(drop=True)
    dates = sorted(panel["date"].dt.normalize().unique())
    folds: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    train_days = int(cfg.wf_train_days)
    test_days = int(cfg.wf_test_days)
    if len(dates) < train_days + test_days:
        return pd.DataFrame(), pd.DataFrame(), {"folds": 0}
    i = train_days
    fold_id = 0
    while i + test_days <= len(dates):
        train_dates = dates[i - train_days : i]
        test_dates = dates[i : i + test_days]
        train_panel_raw = panel[panel["date"].isin(train_dates)].copy()
        if train_panel_raw.empty:
            i += test_days
            continue
        base_th = fit_base_thresholds(train_panel_raw)
        cs_th = fit_consensus_thresholds(train_panel_raw, cfg)
        train_panel = consensus_dims(train_panel_raw, cs_th)
        train_eligible = base_eligible_mask(train_panel, BASE_CANDIDATE) & (train_panel["consensus_short"] >= min_consensus)
        if np.isfinite(base_th.get("exh10_max", np.nan)):
            train_eligible &= train_panel["Exhaustion_10"].astype(float) <= float(base_th["exh10_max"])
        if np.isfinite(base_th.get("ipg10_min", np.nan)):
            train_eligible &= train_panel["IPG_10"].astype(float) >= float(base_th["ipg10_min"])
        n_train_signals = int(train_eligible.sum())
        if n_train_signals < int(cfg.min_train_trades):
            folds.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "min_consensus_short": min_consensus,
                    "n_train_signals": n_train_signals,
                    "n_test_trades": 0,
                    "test_avg_net_bps": None,
                    "test_win_rate": None,
                    "skipped": "train_signal_count_below_min",
                }
            )
            fold_id += 1
            i += test_days
            continue
        test_panel_raw = panel[panel["date"].isin(test_dates)].copy()
        test_panel = consensus_dims(test_panel_raw, cs_th)
        test_trades = consensus_simulate_trades(test_panel, BASE_CANDIDATE, base_th, min_consensus, sim_cfg)
        if not test_trades.empty:
            test_trades = test_trades.assign(fold_id=fold_id)
            all_trades.append(test_trades)
            test_summary = summarize_bps(test_panel, test_trades)
        else:
            test_summary = {"n_trades": 0, "avg_net_bps": None, "win_rate": None}
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "min_consensus_short": min_consensus,
                "n_train_signals": n_train_signals,
                "n_test_trades": int(test_summary.get("n_trades", 0)),
                "test_avg_net_bps": test_summary.get("avg_net_bps"),
                "test_win_rate": test_summary.get("win_rate"),
                "skipped": "",
            }
        )
        fold_id += 1
        i += test_days

    folds_df = pd.DataFrame(folds)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if trades_df.empty:
        wf_summary = {"folds": fold_id, "n_trades": 0, "avg_net_bps": None}
    else:
        first_test = folds_df.loc[folds_df["test_avg_net_bps"].notna(), "test_start"].min()
        last_test = folds_df.loc[folds_df["test_avg_net_bps"].notna(), "test_end"].max()
        wf_panel = panel[(panel["date"] >= pd.Timestamp(first_test).normalize()) & (panel["date"] <= pd.Timestamp(last_test).normalize())]
        wf_bps = summarize_bps(wf_panel, trades_df)
        wf_daily, wf_if = compound_if(wf_panel, trades_df, sim_cfg)
        wf_summary = {
            "folds": fold_id,
            "n_trades": int(wf_bps.get("n_trades", 0)),
            "avg_net_bps": wf_bps.get("avg_net_bps"),
            "win_rate": wf_bps.get("win_rate"),
            "if_summary": wf_if,
            "wf_daily_path_hint": "walkforward_if_daily.csv",
        }
    return folds_df, trades_df, wf_summary


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = DownsideConfig(
        round_trip_cost_bps=cfg.round_trip_cost_bps,
        initial_capital=cfg.initial_capital,
        if_margin_rate=cfg.if_margin_rate,
        if_utilization=cfg.if_utilization,
        if_multiplier=cfg.if_multiplier,
        etf_to_index_scale=cfg.etf_to_index_scale,
        max_trades_per_day=cfg.max_trades_per_day,
        min_train_trades=cfg.min_train_trades,
        wf_train_days=cfg.wf_train_days,
        wf_test_days=cfg.wf_test_days,
        oos_start=cfg.oos_start,
    )

    print("[load] V1 panel ...")
    v1 = load_v1_panel(cfg.input_panel_files)
    print(f"[load] V1 rows={len(v1)} dates={v1['date'].nunique()} {v1['date'].min().date()}..{v1['date'].max().date()}")
    print("[load] multiproj panel ...")
    multi = load_multiproj_panel(cfg.multiproj_panel)
    print(f"[load] multiproj rows={len(multi)} dates={multi['date'].nunique()} {multi['date'].min().date()}..{multi['date'].max().date()}")
    panel = join_panels(v1, multi)
    print(f"[join] joined rows={len(panel)} dates={panel['date'].nunique()}")
    panel = add_kinematics(panel, lag=int(cfg.velocity_lag_minutes))

    oos_start = pd.Timestamp(cfg.oos_start).normalize()
    train = panel[panel["date"] < oos_start].copy()
    oos = panel[panel["date"] >= oos_start].copy()
    print(f"[split] train days={train['date'].nunique()} oos days={oos['date'].nunique()}")

    base_th = fit_base_thresholds(train)
    cs_th = fit_consensus_thresholds(train, cfg)
    train = consensus_dims(train, cs_th)
    oos = consensus_dims(oos, cs_th)
    panel_cs = consensus_dims(panel, cs_th)

    print(f"[base_th] exh10_max={base_th['exh10_max']:.4f} ipg10_min={base_th.get('ipg10_min', float('-inf'))}")
    print(f"[cs_th]   {cs_th}")

    rows: list[dict] = []
    train_segments: dict[int, dict] = {}
    oos_segments: dict[int, dict] = {}

    consensus_levels = list(range(0, len(CONSENSUS_DIMS) + 1))
    for k in consensus_levels:
        train_res = evaluate_segment(train, BASE_CANDIDATE, base_th, k, sim_cfg)
        oos_res = evaluate_segment(oos, BASE_CANDIDATE, base_th, k, sim_cfg)
        train_segments[k] = train_res
        oos_segments[k] = oos_res
        rows.append(metrics_row(f"consensus>={k}", "train", train_res))
        rows.append(metrics_row(f"consensus>={k}", "oos", oos_res))
        print(
            f"[consensus>={k}] train n={int(train_res['bps_summary'].get('n_trades', 0))} "
            f"avg={train_res['bps_summary'].get('avg_net_bps')} "
            f"| oos n={int(oos_res['bps_summary'].get('n_trades', 0))} "
            f"avg={oos_res['bps_summary'].get('avg_net_bps')}"
        )

    # Ablation: each dimension alone (consensus_short = 1, but only counting that one dim)
    for dim in CONSENSUS_DIMS:
        only = train.copy()
        only["_dim_save"] = only[dim]
        only["consensus_short"] = only[dim]
        oos_only = oos.copy()
        oos_only["consensus_short"] = oos_only[dim]
        train_res = evaluate_segment(only, BASE_CANDIDATE, base_th, 1, sim_cfg)
        oos_res = evaluate_segment(oos_only, BASE_CANDIDATE, base_th, 1, sim_cfg)
        rows.append(metrics_row(f"only:{dim}", "train", train_res))
        rows.append(metrics_row(f"only:{dim}", "oos", oos_res))
        print(
            f"[only:{dim}] train n={int(train_res['bps_summary'].get('n_trades', 0))} "
            f"avg={train_res['bps_summary'].get('avg_net_bps')} "
            f"| oos n={int(oos_res['bps_summary'].get('n_trades', 0))} "
            f"avg={oos_res['bps_summary'].get('avg_net_bps')}"
        )

    grid = pd.DataFrame(rows)
    grid_path = out_dir / "consensus_grid_metrics.csv"
    grid.to_csv(grid_path, index=False)

    # Pick best consensus level: maximize OOS avg_net_bps subject to min trade count (4)
    eligible = [k for k in consensus_levels if int(oos_segments[k]["bps_summary"].get("n_trades", 0)) >= 4]
    if not eligible:
        eligible = [0]
    best_k = max(eligible, key=lambda k: float(oos_segments[k]["bps_summary"].get("avg_net_bps") or -1e18))

    # Save selected trade artefacts
    save_trade_set(f"selected_train_consensus_{best_k}", train, train_segments[best_k]["trades"], sim_cfg, out_dir)
    save_trade_set(f"selected_oos_consensus_{best_k}", oos, oos_segments[best_k]["trades"], sim_cfg, out_dir)

    folds_df, wf_trades, wf_summary = consensus_walkforward(panel_cs, sim_cfg, cfg, best_k)
    folds_path = out_dir / f"walkforward_folds_consensus_{best_k}.csv"
    wf_trades_path = out_dir / f"walkforward_trades_consensus_{best_k}.csv"
    folds_df.to_csv(folds_path, index=False)
    wf_trades.to_csv(wf_trades_path, index=False)
    if not wf_trades.empty and isinstance(wf_summary, dict) and "if_summary" in wf_summary:
        first_test = folds_df.loc[folds_df["test_avg_net_bps"].notna(), "test_start"].min()
        last_test = folds_df.loc[folds_df["test_avg_net_bps"].notna(), "test_end"].max()
        wf_panel = panel_cs[(panel_cs["date"] >= pd.Timestamp(first_test).normalize()) & (panel_cs["date"] <= pd.Timestamp(last_test).normalize())]
        wf_daily, _ = compound_if(wf_panel, wf_trades, sim_cfg)
        wf_daily.to_csv(out_dir / f"walkforward_if_daily_consensus_{best_k}.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "base_candidate": BASE_CANDIDATE,
        "base_thresholds": base_th,
        "consensus_thresholds": cs_th,
        "consensus_levels": consensus_levels,
        "selected_min_consensus": int(best_k),
        "data": {
            "train_days": int(train["date"].nunique()),
            "oos_days": int(oos["date"].nunique()),
            "panel_rows": int(len(panel)),
            "panel_dates": [str(panel["date"].min().date()), str(panel["date"].max().date())],
            "oos_start": str(oos_start.date()),
        },
        "metrics_grid_csv": str(grid_path.resolve()),
        "selected_train": metrics_row(f"consensus>={best_k}", "train", train_segments[best_k]),
        "selected_oos": metrics_row(f"consensus>={best_k}", "oos", oos_segments[best_k]),
        "walkforward": wf_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
