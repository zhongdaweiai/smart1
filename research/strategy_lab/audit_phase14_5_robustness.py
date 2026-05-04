#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 14.5 robustness audit.

Three jobs:

1. Apply the Bucket_Penetration single-dim filter on the FIVE V1
   shortlist candidates (not just ipg0_oos_strong) and on the
   auto-selected score_selected. Tests whether the filter is a
   genuine edge or a cherry-pick amplifier.

2. Run a single-dim walk-forward (Bucket_Penetration only) with
   relaxed min_train_trades. This is the cleanest "real money"
   number because the dim was selected on TRAIN merit only.

3. Held-out chunks inside the train window: split 2025-06-09..
   2026-03-05 into three sequential 60-day blocks. Treat each as
   a sub-OOS using the prior block as train. This is true forward
   testing inside the original train window.

For each job, report n_trades, win_rate, avg_net_bps, t-stat,
binomial 95% CI on win rate, and a sanity-stripped IF compounding
proxy.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make sibling scripts importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_hs300_downside_walkforward import (
    Config as DownsideConfig,
    REPO_ROOT,
    candidate_thresholds,
    compound_if,
    summarize_bps,
)
from run_hs300_phase1_5m_eval import regular_signal_mask
from run_phase14_5_consensus_walkforward import (
    BASE_CANDIDATE,
    add_kinematics,
    base_eligible_mask,
    consensus_dims,
    consensus_simulate_trades,
    fit_consensus_thresholds,
    join_panels,
    load_multiproj_panel,
    load_v1_panel,
    Config as Phase145Config,
)


CANDIDATES = {
    "score_selected": {
        "rule": "ED",
        "states": "EMERGING_DOWN",
        "require_highconf": 1,
        "minute_min": 30,
        "hold_bars": 30,
        "dir5_abs_min": 0.75,
        "dir10_max": 0.45,
        "exh10_q": 0.80,
        "ipg10_q": -1.0,
    },
    "ipg0_oos_strong": {
        "rule": "ED",
        "states": "EMERGING_DOWN",
        "require_highconf": 0,
        "minute_min": 30,
        "hold_bars": 30,
        "dir5_abs_min": 0.75,
        "dir10_max": 0.45,
        "exh10_q": 0.90,
        "ipg10_q": 0.40,
    },
    "ipg_m008_balanced": {
        "rule": "ED",
        "states": "EMERGING_DOWN",
        "require_highconf": 0,
        "minute_min": 30,
        "hold_bars": 30,
        "dir5_abs_min": 0.75,
        "dir10_max": 0.0,
        "exh10_q": 0.90,
        "ipg10_q": 0.40,
    },
    "ipg_q25_balanced": {
        "rule": "ED",
        "states": "EMERGING_DOWN",
        "require_highconf": 1,
        "minute_min": 30,
        "hold_bars": 30,
        "dir5_abs_min": 0.75,
        "dir10_max": 0.0,
        "exh10_q": 0.90,
        "ipg10_q": 0.25,
    },
    "base_exh90": {
        "rule": "ED",
        "states": "EMERGING_DOWN",
        "require_highconf": 1,
        "minute_min": 30,
        "hold_bars": 30,
        "dir5_abs_min": 0.75,
        "dir10_max": 0.45,
        "exh10_q": 0.90,
        "ipg10_q": -1.0,
    },
}


def cand_eligible_mask(panel: pd.DataFrame, cand: dict, thresholds: dict) -> pd.Series:
    states = {x for x in str(cand["states"]).split("|") if x}
    mask = panel["state"].isin(states)
    mask &= panel["regular_time"].astype(bool)
    mask &= panel["minute_idx"].astype(int) >= int(cand["minute_min"])
    mask &= panel["DirScore_5"].astype(float) <= -float(cand["dir5_abs_min"])
    mask &= panel["DirScore_10"].astype(float) <= float(cand["dir10_max"])
    if bool(cand.get("require_highconf", 0)):
        mask &= panel["signal_side_5"].astype(int) == -1
    if np.isfinite(thresholds.get("exh10_max", np.nan)):
        mask &= panel["Exhaustion_10"].astype(float) <= float(thresholds["exh10_max"])
    if np.isfinite(thresholds.get("ipg10_min", np.nan)):
        mask &= panel["IPG_10"].astype(float) >= float(thresholds["ipg10_min"])
    return mask


def simulate_with_optional_filter(
    panel: pd.DataFrame,
    cand: dict,
    thresholds: dict,
    extra_filter: pd.Series | None,
    cfg: DownsideConfig,
) -> pd.DataFrame:
    base = cand_eligible_mask(panel, cand, thresholds)
    if extra_filter is not None:
        extra_filter = extra_filter.reindex(panel.index).fillna(False).astype(bool)
        base &= extra_filter
    if not base.any():
        return pd.DataFrame()
    rows: list[dict] = []
    hold = int(cand["hold_bars"])
    for date, day_panel in panel.sort_values(["date", "minute_idx"]).groupby("date", sort=True):
        day = day_panel.reset_index(drop=True)
        day_mask = base.loc[day_panel.index].reset_index(drop=True)
        idxs = day.index[day_mask].tolist()
        if not idxs:
            continue
        next_ok_pos = -1
        trades_today = 0
        for signal_pos in idxs:
            if signal_pos < next_ok_pos or trades_today >= int(cfg.max_trades_per_day):
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
                    "entry_minute_idx": int(day.loc[entry_pos, "minute_idx"]),
                    "exit_minute_idx": int(day.loc[exit_pos, "minute_idx"]),
                    "side": -1,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "gross_bps": gross_bps,
                    "net_bps": gross_bps - float(cfg.round_trip_cost_bps),
                }
            )
            next_ok_pos = exit_pos
            trades_today += 1
    return pd.DataFrame(rows)


def perf_row(label: str, segment: str, panel: pd.DataFrame, trades: pd.DataFrame, cfg: DownsideConfig) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "label": label, "segment": segment, "n_trades": 0,
            "win_rate": None, "avg_net_bps": None, "median_net_bps": None,
            "t_stat": None, "win_rate_ci_low": None, "win_rate_ci_high": None,
            "if_total_return": None, "if_max_drawdown": None,
            "if_sharpe": None, "if_calmar": None,
        }
    s = trades["net_bps"].astype(float).values
    win_rate = float((s > 0).mean())
    avg = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    t_stat = avg / (std / math.sqrt(n)) if std > 0 else float("nan")
    # Wilson 95% CI for win rate
    z = 1.96
    p = win_rate
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    if_daily, if_summary = compound_if(panel, trades, cfg)
    return {
        "label": label, "segment": segment, "n_trades": n,
        "win_rate": round(win_rate, 4),
        "avg_net_bps": round(avg, 3),
        "median_net_bps": round(float(np.median(s)), 3),
        "std_net_bps": round(std, 3),
        "t_stat": round(t_stat, 3) if math.isfinite(t_stat) else None,
        "win_rate_ci_low": round(max(0.0, center - radius), 3),
        "win_rate_ci_high": round(min(1.0, center + radius), 3),
        "if_total_return": round(float(if_summary.get("total_return") or 0.0), 4),
        "if_max_drawdown": round(float(if_summary.get("max_drawdown") or 0.0), 4),
        "if_sharpe": round(float(if_summary.get("sharpe") or 0.0), 3),
        "if_calmar": round(float(if_summary.get("calmar") or 0.0), 2),
    }


def run_audit():
    repo = Path(REPO_ROOT)
    panel_files = ",".join(
        [
            str(repo / "results/510300_breadth_regime/hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet"),
            str(repo / "results/510300_breadth_regime/hs300_phase1_5m_oos_20260306_precise_static_v1/scored_panel.parquet"),
        ]
    )
    multiproj_path = str(repo / "results/510300_breadth_regime/hs300_multiproj_panel_v1/multiproj_panel.parquet")
    out_dir = repo / "results/510300_breadth_regime/hs300_phase14_5_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] V1 + multiproj ...")
    v1 = load_v1_panel(panel_files)
    multi = load_multiproj_panel(multiproj_path)
    panel = join_panels(v1, multi)
    panel = add_kinematics(panel, lag=3)

    sim_cfg = DownsideConfig(
        round_trip_cost_bps=4.5,
        initial_capital=500_000.0,
        if_margin_rate=0.08,
        if_utilization=0.80,
        if_multiplier=300.0,
        etf_to_index_scale=1000.0,
        max_trades_per_day=2,
    )
    cfg145 = Phase145Config()

    oos_start = pd.Timestamp("2026-03-06").normalize()
    train = panel[panel["date"] < oos_start].copy()
    oos = panel[panel["date"] >= oos_start].copy()

    cs_th = fit_consensus_thresholds(train, cfg145)
    print(f"[cs_thresholds (train q50)] {json.dumps(cs_th, indent=2, default=str)}")
    train = consensus_dims(train, cs_th)
    oos = consensus_dims(oos, cs_th)

    bp_train = (train["cs_bucket_penetration"] == 1)
    bp_oos = (oos["cs_bucket_penetration"] == 1)

    rows: list[dict] = []

    print("\n=== JOB 1: Bucket_Penetration filter applied to ALL 5 V1 candidates ===")
    for name, cand in CANDIDATES.items():
        # Per-candidate train-fit thresholds
        thresholds = candidate_thresholds(train, pd.Series(cand))
        # Without filter
        tr = simulate_with_optional_filter(train, cand, thresholds, None, sim_cfg)
        oo = simulate_with_optional_filter(oos, cand, thresholds, None, sim_cfg)
        rows.append(perf_row(f"{name} | base", "train", train, tr, sim_cfg))
        rows.append(perf_row(f"{name} | base", "oos", oos, oo, sim_cfg))
        # With Bucket_Penetration filter
        tr_bp = simulate_with_optional_filter(train, cand, thresholds, bp_train, sim_cfg)
        oo_bp = simulate_with_optional_filter(oos, cand, thresholds, bp_oos, sim_cfg)
        rows.append(perf_row(f"{name} | + bucket_penetration", "train", train, tr_bp, sim_cfg))
        rows.append(perf_row(f"{name} | + bucket_penetration", "oos", oos, oo_bp, sim_cfg))
        # Print delta
        b_train_avg = rows[-4]["avg_net_bps"]
        f_train_avg = rows[-2]["avg_net_bps"]
        b_oos_avg = rows[-3]["avg_net_bps"]
        f_oos_avg = rows[-1]["avg_net_bps"]
        b_train_n = rows[-4]["n_trades"]
        f_train_n = rows[-2]["n_trades"]
        b_oos_n = rows[-3]["n_trades"]
        f_oos_n = rows[-1]["n_trades"]
        print(
            f"  {name:>22s}  "
            f"train base ({b_train_n:>3d}, {b_train_avg}) -> + BP ({f_train_n:>3d}, {f_train_avg})  "
            f"| oos base ({b_oos_n:>2d}, {b_oos_avg}) -> + BP ({f_oos_n:>2d}, {f_oos_avg})"
        )

    print("\n=== JOB 2: Bucket_Penetration single-dim walk-forward ===")
    # 4 folds of 120-train / 20-test, use only Bucket_Penetration as filter
    panel_full = consensus_dims(panel, cs_th)
    panel_full = panel_full.sort_values(["date", "minute_idx"]).reset_index(drop=True)
    bp_full = panel_full["cs_bucket_penetration"] == 1
    train_days = 120
    test_days = 20
    min_train_trades = 14
    cand = CANDIDATES["ipg0_oos_strong"]  # state filter is the same across all candidates
    dates = sorted(panel_full["date"].dt.normalize().unique())
    fold_records: list[dict] = []
    all_wf_trades: list[pd.DataFrame] = []
    i = train_days
    fold_id = 0
    while i + test_days <= len(dates):
        train_dates = dates[i - train_days : i]
        test_dates = dates[i : i + test_days]
        train_p = panel_full[panel_full["date"].isin(train_dates)].copy()
        # Fit base thresholds and consensus thresholds on this fold's train
        cs_th_fold = fit_consensus_thresholds(train_p, cfg145)
        train_p = consensus_dims(train_p, cs_th_fold)
        thresholds = candidate_thresholds(train_p, pd.Series(cand))
        bp_train_p = train_p["cs_bucket_penetration"] == 1
        train_signals_count = (cand_eligible_mask(train_p, cand, thresholds) & bp_train_p).sum()
        if train_signals_count < min_train_trades:
            fold_records.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "n_train_signals": int(train_signals_count),
                    "n_test_trades": 0,
                    "test_avg_net_bps": None,
                    "test_win_rate": None,
                    "skipped": "below_min_train_trades",
                }
            )
        else:
            test_p = panel_full[panel_full["date"].isin(test_dates)].copy()
            test_p = consensus_dims(test_p, cs_th_fold)
            bp_test_p = test_p["cs_bucket_penetration"] == 1
            tt = simulate_with_optional_filter(test_p, cand, thresholds, bp_test_p, sim_cfg)
            if not tt.empty:
                tt = tt.assign(fold_id=fold_id)
                all_wf_trades.append(tt)
            test_n = len(tt)
            test_avg = float(tt["net_bps"].mean()) if test_n else None
            test_win = float((tt["net_bps"] > 0).mean()) if test_n else None
            fold_records.append(
                {
                    "fold_id": fold_id,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "n_train_signals": int(train_signals_count),
                    "n_test_trades": test_n,
                    "test_avg_net_bps": test_avg,
                    "test_win_rate": test_win,
                    "skipped": "",
                }
            )
        fold_id += 1
        i += test_days
    folds_df = pd.DataFrame(fold_records)
    folds_df.to_csv(out_dir / "bp_only_walkforward_folds.csv", index=False)
    if all_wf_trades:
        wf_trades = pd.concat(all_wf_trades, ignore_index=True)
        wf_trades.to_csv(out_dir / "bp_only_walkforward_trades.csv", index=False)
        first = folds_df.loc[folds_df["test_avg_net_bps"].notna(), "test_start"].min()
        last = folds_df.loc[folds_df["test_avg_net_bps"].notna(), "test_end"].max()
        wf_panel = panel_full[(panel_full["date"] >= pd.Timestamp(first).normalize()) & (panel_full["date"] <= pd.Timestamp(last).normalize())]
        rows.append(perf_row("BP-only walkforward", "wf", wf_panel, wf_trades, sim_cfg))
        wf_daily, wf_if = compound_if(wf_panel, wf_trades, sim_cfg)
        wf_daily.to_csv(out_dir / "bp_only_walkforward_if_daily.csv", index=False)
        print(
            f"  BP-only WF: folds={fold_id} trades={len(wf_trades)} "
            f"win_rate={(wf_trades['net_bps']>0).mean():.3f} avg_bps={wf_trades['net_bps'].mean():.2f} "
            f"if_total={wf_if.get('total_return'):.4f} if_max_dd={wf_if.get('max_drawdown'):.4f} "
            f"if_sharpe={wf_if.get('sharpe'):.2f} if_calmar={wf_if.get('calmar'):.2f}"
        )
    else:
        print("  BP-only WF: no trades produced")

    print("\n=== JOB 3: In-train held-out chunks (3 sequential 60-day blocks) ===")
    # Use ipg0_oos_strong as primary candidate for symmetry with V1 handoff
    cand = CANDIDATES["ipg0_oos_strong"]
    # Split train period into 3 chunks of about equal size
    train_dates_sorted = sorted(train["date"].dt.normalize().unique())
    n_train = len(train_dates_sorted)
    chunk = n_train // 3
    chunks = [
        ("chunk_A", train_dates_sorted[:chunk]),
        ("chunk_B", train_dates_sorted[chunk : 2 * chunk]),
        ("chunk_C", train_dates_sorted[2 * chunk :]),
    ]
    print(f"  in-train chunk sizes: {[len(c[1]) for c in chunks]} days")
    # For each chunk, treat prior chunks as train, this chunk as held-out
    for idx, (name, chunk_dates) in enumerate(chunks):
        if idx == 0:
            print(f"  {name}: skipped (no prior train block)")
            continue
        prior_dates = []
        for prev_idx in range(idx):
            prior_dates.extend(chunks[prev_idx][1])
        train_chunk = panel[panel["date"].isin(prior_dates)].copy()
        test_chunk = panel[panel["date"].isin(chunk_dates)].copy()
        cs_th_chunk = fit_consensus_thresholds(train_chunk, cfg145)
        train_chunk = consensus_dims(train_chunk, cs_th_chunk)
        test_chunk = consensus_dims(test_chunk, cs_th_chunk)
        thresholds_chunk = candidate_thresholds(train_chunk, pd.Series(cand))
        # Plain (no filter)
        tt_plain = simulate_with_optional_filter(test_chunk, cand, thresholds_chunk, None, sim_cfg)
        # With BP filter
        bp_chunk = test_chunk["cs_bucket_penetration"] == 1
        tt_bp = simulate_with_optional_filter(test_chunk, cand, thresholds_chunk, bp_chunk, sim_cfg)
        rows.append(perf_row(f"{name} | ipg0 base", f"hold_out", test_chunk, tt_plain, sim_cfg))
        rows.append(perf_row(f"{name} | ipg0 + bucket_penetration", f"hold_out", test_chunk, tt_bp, sim_cfg))
        plain_avg = rows[-2]["avg_net_bps"]
        bp_avg = rows[-1]["avg_net_bps"]
        plain_n = rows[-2]["n_trades"]
        bp_n = rows[-1]["n_trades"]
        plain_win = rows[-2]["win_rate"]
        bp_win = rows[-1]["win_rate"]
        print(
            f"  {name}: ipg0 base ({plain_n}, win={plain_win}, avg={plain_avg}) "
            f"-> + BP ({bp_n}, win={bp_win}, avg={bp_avg})"
        )

    grid = pd.DataFrame(rows)
    grid_path = out_dir / "audit_grid.csv"
    grid.to_csv(grid_path, index=False)
    print(f"\nSaved grid to: {grid_path}")
    print(f"Saved fold records to: {out_dir / 'bp_only_walkforward_folds.csv'}")
    return grid


if __name__ == "__main__":
    run_audit()
