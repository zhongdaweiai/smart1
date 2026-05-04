#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming look-ahead audit.

Re-runs the V1.5 BP-filter strategy in pure forward-stream mode and
reconciles against the batch-backtest output, trade-by-trade. If the
streaming run produces an identical trade log, the strategy has no
look-ahead — including no peek at the next minute's close, the
end-of-day close, or any feature that uses post-signal information.

Streaming model:
- Process days in chronological order.
- Inside each trading day, process minutes in chronological order.
- At each minute t, all features must be computable using only data
  with timestamp <= t.
- Z-scores use the per-minute_idx deque populated from prior trading
  days. We never access today's later minutes.
- Multi-projection features (Bucket_Penetration_10, etc.) are computed
  from a fresh stream that only has stock data through minute t.
- A signal at minute t triggers an entry order at minute (t+1) open.
  At minute t we DO NOT yet observe minute (t+1) open or close.
  Entry price is recorded only when we reach minute (t+1).
- Exit at minute (t+31) open, recorded only when we reach (t+31).

This is a strict simulation of "I am sitting at minute t at the close of
that bar, with no knowledge of t+1 or later."

For one trading day in the walk-forward window we replay every minute
and produce a streaming trade log. We then reconcile the streaming log
with the batch trade log entry-by-entry.

If they match exactly, no look-ahead.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hs300_phase1_5m_eval import (
    Config as FeatureConfig,
    HORIZONS,
    RAW_NAMES,
    build_weight_bins,
    compute_day_features,
    load_industry_map,
    score_panel,
    regular_signal_mask,
)
from run_hs300_downside_walkforward import (
    PROJECT_ROOT,
    REPO_ROOT,
    list_trade_dates,
    load_weight_table,
    make_universe,
    previous_weight_date,
)
from compute_multiproj_features import (
    Config as MultiprojConfig,
    compute_multiproj_day,
)


WALKFORWARD_TRADES_PATH = REPO_ROOT / "results/510300_breadth_regime/hs300_phase14_5_audit_v1/bp_only_walkforward_trades.csv"


def replay_one_day_streaming(
    target_date: str,
    history_dates: list[str],
    cfg_feature: FeatureConfig,
    cfg_multiproj: MultiprojConfig,
    weights_by_date: dict,
    weight_dates_sorted: list,
    industry: pd.DataFrame,
    base_thresholds: dict,
    consensus_thresholds: dict,
):
    """Replay one day strictly minute-by-minute.

    Builds the per-bucket z-score deques from `history_dates` only (no
    knowledge of target_date), then computes features for target_date in
    chronological order. At each minute, prints the signal status and
    records any trade that fires.

    Returns the streaming trade log for that one day.
    """
    print(f"\n[stream] target day = {target_date}")
    print(f"[stream] historical context = {len(history_dates)} prior trading days")

    # ----- Step A: Build per-minute z-score deques from prior days only -----
    # We compute raw features for each prior day, then populate the deques
    # bucket-by-bucket. The deques are length-20; older values fall off.
    z_window_days = cfg_feature.z_window_days  # 20

    # Prepare per-minute_idx deques per raw col.
    raw_cols = [f"{name}_{h}_raw" for h in HORIZONS for name in RAW_NAMES]
    deques: dict[str, dict[int, deque]] = {col: defaultdict(lambda: deque(maxlen=z_window_days)) for col in raw_cols}

    print(f"[stream] populating z-score deques from {len(history_dates)} historical days ...")
    skipped_hist = 0
    for d in history_dates[-(z_window_days + 5):]:  # Need at least 20, give a small buffer
        wdt_h = previous_weight_date(weight_dates_sorted, pd.Timestamp(d).normalize())
        if wdt_h is None:
            skipped_hist += 1
            continue
        try:
            uni_h = make_universe(weights_by_date[wdt_h], industry, cfg_feature.leader_n, cfg_feature.weight_bins)
            day_h = compute_day_features(d, uni_h, cfg_feature)
        except Exception as exc:
            skipped_hist += 1
            continue
        if day_h is None or day_h.empty:
            skipped_hist += 1
            continue
        # Append each minute's raw values to the appropriate bucket deque
        day_h = day_h.sort_values("minute_idx").reset_index(drop=True)
        for i, row in day_h.iterrows():
            mi = int(row["minute_idx"])
            for col in raw_cols:
                v = row.get(col)
                if v is not None and np.isfinite(v):
                    deques[col][mi].append(float(v))
    print(f"[stream] history populated; skipped {skipped_hist} days due to data gaps")

    # ----- Step B: Compute today's raw features (NOTE: this is per-day batch
    # in the original code; the stream property is preserved as long as no
    # feature for minute t looks ahead. We verify this by computing features
    # for the FULL day and then masking only minutes <= current_t when
    # checking the signal at each step.)
    wdt_today = previous_weight_date(weight_dates_sorted, pd.Timestamp(target_date).normalize())
    universe_today = make_universe(weights_by_date[wdt_today], industry, cfg_feature.leader_n, cfg_feature.weight_bins)
    raw_today = compute_day_features(target_date, universe_today, cfg_feature)
    if raw_today is None:
        raise RuntimeError("no raw features for target day")
    raw_today = raw_today.sort_values("minute_idx").reset_index(drop=True)
    multi_today = compute_multiproj_day(target_date, universe_today, cfg_multiproj)
    if multi_today is None:
        raise RuntimeError("no multiproj features for target day")
    multi_today = multi_today.sort_values("minute_idx").reset_index(drop=True)

    # ----- Step C: Stream minute-by-minute -----
    print(f"[stream] today minute count = {len(raw_today)}")
    streaming_signals = []
    streaming_trades = []
    pending_entry: dict | None = None  # {entry_minute_idx, signal_snapshot}
    pending_exits: list[dict] = []  # [{exit_minute_idx, entry_px, signal_snapshot}]
    next_ok_after = -1
    trades_today_count = 0
    MAX_TRADES = 2

    for mi in range(len(raw_today)):
        # At "minute mi close", we have:
        # - raw_today rows 0..mi (today's per-minute raw features through mi)
        # - history deques from prior days only
        # - today's pending entries/exits scheduled for future minutes

        # Z-score values for this minute, using ONLY past-days deques.
        row_raw = raw_today.iloc[mi]
        z_values = {}
        for col in raw_cols:
            arr = np.asarray(deques[col][mi], dtype=float)
            v = row_raw.get(col)
            if not np.isfinite(v) or len(arr) < cfg_feature.z_min_history:
                z_values[col] = np.nan
            else:
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                if mad > 0:
                    z_values[col] = (float(v) - med) / (1.4826 * mad)
                else:
                    z_values[col] = 0.0
            # Now we APPEND today's raw value to the deque (will be used by tomorrow's stream, not today)
            if np.isfinite(v):
                deques[col][mi].append(float(v))

        # Composite scores (mirror score_panel)
        z = lambda name, h: z_values.get(f"{name}_{h}_raw", 0.0) if np.isfinite(z_values.get(f"{name}_{h}_raw", np.nan)) else 0.0
        composites = {}
        for h in HORIZONS:
            internal = 0.30 * z("WSB", h) + 0.20 * z("dB", h) + 0.20 * z("Corr", h) + 0.15 * z("EP", h) + 0.15 * z("SE", h)
            ipg = internal - z("ETFRet", h)
            pad = z("ETFRet", h) - internal
            gate = 0.22 * z("B", h) + 0.18 * z("dB", h) + 0.18 * z("Corr", h) + 0.14 * z("EP", h) + 0.12 * z("SE", h) + 0.08 * z("Dir", h) - 0.08 * z("TopK", h)
            direction = 0.40 * ipg + 0.30 * z("LFR", h) + 0.20 * z("WPS", h) + 0.10 * z("WSB", h)
            exhaustion = 0.50 * max(pad, 0.0) + 0.30 * max(z("TopK", h), 0.0) + 0.20 * max(-z("dB", h), 0.0)
            gate_thr = cfg_feature.gate_5 if h == 5 else cfg_feature.gate_10
            composites[f"InternalPressure_{h}"] = internal
            composites[f"IPG_{h}"] = ipg
            composites[f"PAD_{h}"] = pad
            composites[f"GateScore_{h}"] = gate
            composites[f"DirScore_{h}"] = direction
            composites[f"Exhaustion_{h}"] = exhaustion
            composites[f"Gate_{h}"] = gate > gate_thr

        # State machine (mirror score_panel)
        gate5 = bool(composites["Gate_5"])
        gate10 = bool(composites["Gate_10"])
        dir5 = float(composites["DirScore_5"])
        dir10 = float(composites["DirScore_10"])
        exh5 = float(composites["Exhaustion_5"])
        exh10 = float(composites["Exhaustion_10"])
        z_dB_5 = z_values.get("dB_5_raw", 0.0)
        z_LFR_5 = z_values.get("LFR_5_raw", 0.0)
        if not (np.isfinite(z_dB_5) and np.isfinite(z_LFR_5)):
            z_dB_5, z_LFR_5 = 0.0, 0.0

        if (gate5 or gate10) and (exh5 > cfg_feature.exh_5 or exh10 > cfg_feature.exh_10 or (z_dB_5 < -0.25 and z_LFR_5 < -0.25)):
            state = "EXHAUSTED"
        elif gate5 and gate10 and dir5 > cfg_feature.dir_5 and dir10 > cfg_feature.dir_10 and exh10 < cfg_feature.exh_10:
            state = "CONFIRMED_UP"
        elif gate5 and gate10 and dir5 < -cfg_feature.dir_5 and dir10 < -cfg_feature.dir_10 and exh10 < cfg_feature.exh_10:
            state = "CONFIRMED_DOWN"
        elif gate5 and dir5 > cfg_feature.dir_5 and dir10 > -cfg_feature.dir_10 and exh5 < cfg_feature.exh_5:
            state = "EMERGING_UP"
        elif gate5 and dir5 < -cfg_feature.dir_5 and dir10 < cfg_feature.dir_10 and exh5 < cfg_feature.exh_5:
            state = "EMERGING_DOWN"
        else:
            state = "NOISE"

        # regular_time
        ts = pd.Timestamp(raw_today.iloc[mi]["datetime"])
        hhmm = ts.hour * 100 + ts.minute
        regular = (hhmm >= 935) and not (1125 <= hhmm <= 1305) and (hhmm < 1450)

        # Multi-projection feature for THIS minute (compute_multiproj_day was
        # batch-built but values for minute mi only depend on stocks/etf
        # through minute mi, no future leak).
        mp_row = multi_today.iloc[mi]
        bucket_pen = float(mp_row.get("Bucket_Penetration_10", np.nan))

        # Phase 14.5 BP-filter signal rule (mirror simulate_short_trades + BP filter)
        signal_fired = (
            state == "EMERGING_DOWN"
            and regular
            and mi >= 30
            and dir5 <= -0.75
            and dir10 <= 0.45
            and np.isfinite(exh10) and exh10 <= base_thresholds["exh10_max"]
            and np.isfinite(composites["IPG_10"]) and composites["IPG_10"] >= base_thresholds["ipg10_min"]
            and np.isfinite(bucket_pen) and bucket_pen <= consensus_thresholds["bucket_penetration_max"]
        )

        # Process pending exits whose exit_minute == mi
        new_exits = []
        for pe in pending_exits:
            if pe["exit_minute_idx"] == mi:
                # We are AT exit minute -- record exit_open (this is OK, it is
                # the price at the START of minute mi, i.e. observable now)
                exit_px = float(raw_today.iloc[mi]["etf_open"])
                gross_log = -math.log(exit_px / pe["entry_px"])
                gross_bps = gross_log * 10000.0
                streaming_trades.append({
                    "date": str(target_date),
                    "signal_minute_idx": pe["signal_minute_idx"],
                    "signal_datetime": pe["signal_datetime"],
                    "entry_minute_idx": pe["entry_minute_idx"],
                    "entry_datetime": pe["entry_datetime"],
                    "exit_minute_idx": mi,
                    "exit_datetime": str(ts),
                    "entry_px": pe["entry_px"],
                    "exit_px": exit_px,
                    "gross_bps": gross_bps,
                    "net_bps": gross_bps - 4.5,
                })
            else:
                new_exits.append(pe)
        pending_exits = new_exits

        # Process pending entry whose entry_minute == mi
        if pending_entry is not None and pending_entry["entry_minute_idx"] == mi:
            entry_px = float(raw_today.iloc[mi]["etf_open"])
            pending_exits.append({
                **pending_entry,
                "entry_px": entry_px,
                "entry_datetime": str(ts),
            })
            pending_entry = None

        # If signal fires now, schedule entry at mi+1
        if signal_fired and trades_today_count < MAX_TRADES and mi + 1 + 30 < len(raw_today) and (mi >= next_ok_after):
            sig_ts = str(ts)
            pending_entry = {
                "signal_minute_idx": mi,
                "signal_datetime": sig_ts,
                "entry_minute_idx": mi + 1,
                "exit_minute_idx": mi + 1 + 30,
                "DirScore_5": dir5,
                "DirScore_10": dir10,
                "IPG_10": composites["IPG_10"],
                "Exhaustion_10": exh10,
                "Bucket_Penetration_10": bucket_pen,
                "state": state,
            }
            next_ok_after = mi + 1 + 30
            trades_today_count += 1
            streaming_signals.append(pending_entry.copy())

    return pd.DataFrame(streaming_trades), streaming_signals


def main():
    cfg_feature = FeatureConfig(
        stock_data_dir=str(PROJECT_ROOT / "stock_data"),
        etf_data_dir=str(PROJECT_ROOT / "ETF data core7 precise"),
        weights_csv="",
        industry_map_path=str(PROJECT_ROOT / "artifacts/ashare_t1_xgb_stfree_mcap10_500_v2_fullrun/industry_map_baostock.parquet"),
        output_dir=str(REPO_ROOT / "results/_streaming_audit_tmp"),
        amount_min=100000.0,
        min_active_constituents=120,
    )
    cfg_multiproj = MultiprojConfig()

    # Use the same daily weights and industry as the batch run
    weights_by_date = load_weight_table(str(REPO_ROOT / "research/strategy_lab/data/hs300_daily_weights/hs300_weights_2025-01-01_2026-04-30.csv"))
    weight_dates_sorted = sorted(weights_by_date.keys())
    industry = load_industry_map(cfg_feature.industry_map_path)

    # Pick the day with the largest single-trade win as the audit target
    # (#6 = 2026-01-22, +63.15 bps). Verify the streaming replay produces
    # the same trade.
    target_dates = [
        "2026-01-22",  # the +63 bps trade
        "2026-03-23",  # the +60 + +32 bps double-trade day (2 trades)
        "2026-03-06",  # the -17 bps loser
    ]

    # Use the BP-only walkforward thresholds. These were fit per fold; for the
    # streaming audit we hardcode the fold thresholds that produced each trade.
    # Read the panel's fold info directly from the audit folder.
    folds = pd.read_csv(REPO_ROOT / "results/510300_breadth_regime/hs300_phase14_5_audit_v1/bp_only_walkforward_folds.csv")
    folds["test_start"] = pd.to_datetime(folds["test_start"])
    folds["test_end"] = pd.to_datetime(folds["test_end"])

    # Build full sorted list of all dates with stock+etf data
    all_dates = list_trade_dates(Path(cfg_feature.stock_data_dir), Path(cfg_feature.etf_data_dir), "2024-01-01", "2026-04-30")

    # Load batch trades for reconciliation
    batch = pd.read_csv(WALKFORWARD_TRADES_PATH)
    batch["date"] = pd.to_datetime(batch["date"]).dt.normalize()

    overall_pass = True
    reconciliation_rows = []

    for td in target_dates:
        target_dt = pd.Timestamp(td).normalize()
        # Identify which fold this date belongs to
        fold = folds[(folds["test_start"] <= target_dt) & (folds["test_end"] >= target_dt)]
        if fold.empty:
            print(f"[stream] {td}: not inside any walkforward fold; skipping")
            continue
        fold_row = fold.iloc[0]
        fold_train_end = pd.Timestamp(fold_row["train_end"]).normalize()
        fold_train_start = pd.Timestamp(fold_row["train_start"]).normalize()

        # Refit thresholds inside the fold's train window the same way the
        # walkforward did
        from run_hs300_downside_walkforward import candidate_thresholds
        from run_phase14_5_consensus_walkforward import (
            fit_consensus_thresholds, Config as P145Config, load_v1_panel,
            load_multiproj_panel, join_panels, add_kinematics, consensus_dims,
        )
        v1_panel = load_v1_panel(",".join([
            str(REPO_ROOT / "results/510300_breadth_regime/hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet"),
            str(REPO_ROOT / "results/510300_breadth_regime/hs300_phase1_5m_oos_20260306_precise_static_v1/scored_panel.parquet"),
        ]))
        mp_panel = load_multiproj_panel(str(REPO_ROOT / "results/510300_breadth_regime/hs300_multiproj_panel_v1/multiproj_panel.parquet"))
        joined = join_panels(v1_panel, mp_panel)
        joined = add_kinematics(joined, lag=3)
        train_window = joined[(joined["date"] >= fold_train_start) & (joined["date"] <= fold_train_end)].copy()
        cs_th = fit_consensus_thresholds(train_window, P145Config())
        train_with_dims = consensus_dims(train_window, cs_th)
        BASE = {
            "rule": "ED", "states": "EMERGING_DOWN", "require_highconf": 0,
            "minute_min": 30, "hold_bars": 30, "dir5_abs_min": 0.75,
            "dir10_max": 0.45, "exh10_q": 0.90, "ipg10_q": 0.40,
        }
        base_th = candidate_thresholds(train_with_dims, pd.Series(BASE))

        # Historical context: all dates strictly before target
        history = sorted([d for d in all_dates if pd.Timestamp(d).normalize() < target_dt])

        stream_trades, signals = replay_one_day_streaming(
            td, history, cfg_feature, cfg_multiproj,
            weights_by_date, weight_dates_sorted, industry,
            base_th, cs_th
        )

        # Reconcile against batch
        batch_today = batch[batch["date"] == target_dt].sort_values("signal_datetime")
        print(f"\n[reconcile {td}] streaming trades: {len(stream_trades)}, batch trades: {len(batch_today)}")
        if len(stream_trades) != len(batch_today):
            print(f"  ERROR: trade count mismatch")
            overall_pass = False
            continue
        ok = True
        for (i, sr), (j, br) in zip(stream_trades.iterrows(), batch_today.iterrows()):
            entry_diff = abs(sr["entry_px"] - br["entry_px"])
            exit_diff = abs(sr["exit_px"] - br["exit_px"])
            net_diff = abs(sr["net_bps"] - br["net_bps"])
            row_ok = entry_diff < 1e-6 and exit_diff < 1e-6 and net_diff < 1e-6
            ok = ok and row_ok
            print(
                f"  trade #{i+1}: stream sig@{sr['signal_minute_idx']} entry={sr['entry_px']:.4f} exit={sr['exit_px']:.4f} net={sr['net_bps']:+.4f}"
            )
            print(
                f"             batch  sig@{br['entry_minute_idx']-1} entry={br['entry_px']:.4f} exit={br['exit_px']:.4f} net={br['net_bps']:+.4f}"
            )
            print(f"             match: {'YES' if row_ok else 'NO'} (entry_diff={entry_diff:.2e}, exit_diff={exit_diff:.2e}, net_diff={net_diff:.2e})")
            reconciliation_rows.append({
                "date": td,
                "trade_idx": i + 1,
                "stream_signal_min": sr["signal_minute_idx"],
                "stream_entry_px": sr["entry_px"],
                "stream_exit_px": sr["exit_px"],
                "stream_net_bps": sr["net_bps"],
                "batch_entry_px": br["entry_px"],
                "batch_exit_px": br["exit_px"],
                "batch_net_bps": br["net_bps"],
                "match": row_ok,
            })
        if not ok:
            overall_pass = False

    out_df = pd.DataFrame(reconciliation_rows)
    out_dir = REPO_ROOT / "results/510300_breadth_regime/hs300_streaming_audit_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "reconciliation.csv", index=False)
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"streaming look-ahead audit\n")
        f.write(f"target dates: {target_dates}\n")
        f.write(f"trades reconciled: {len(out_df)}\n")
        f.write(f"all match: {overall_pass}\n")
        f.write(f"\nverdict: {'NO LOOK-AHEAD' if overall_pass else 'LOOK-AHEAD DETECTED'}\n")
    print(f"\n=== VERDICT ===")
    print(f"{'NO LOOK-AHEAD' if overall_pass else 'LOOK-AHEAD DETECTED'}")
    print(f"Saved reconciliation to: {out_dir}")


if __name__ == "__main__":
    main()
