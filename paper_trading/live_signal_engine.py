#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live signal engine for HS300 Downside Propagation V1.5.

Designed to run in three modes:

1. replay <YYYY-MM-DD>     — read a single trading day from local parquet
                             and stream minute-by-minute as if live.
                             Logs signals + trades to SQLite.

2. live                    — poll the configured data source every 60s
                             during market hours (09:30-15:00 CN).
                             Same streaming logic as replay.

3. backfill <START> <END>  — process a date range as a batch of replays
                             (useful for catching up after a downtime).

The streaming logic mirrors streaming_lookahead_audit.py exactly. Bit-
exact reconciliation against the batch backtest is verified for three
target days; this engine is the canonical paper-trading surface.

Outputs land in `paper_trading.db` (SQLite). Tables:

  signals (id, ts, date, minute_idx, dir5, dir10, exh10, ipg10, bp10,
           state, fired)
  trades  (id, signal_id, entry_ts, entry_px, exit_ts, exit_px,
           gross_bps, net_bps, hold_minutes, status, fold_thresholds_json)
  equity  (date, equity_close, daily_pnl, contracts)

Data source adapter is pluggable. Default: local parquet under
PROJECT_ROOT (works for replay/backfill). To run live, plug an
implementation of DataSource that reads from a real-time API.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "research" / "strategy_lab"))

from run_hs300_phase1_5m_eval import (
    Config as FeatureConfig,
    HORIZONS,
    RAW_NAMES,
    compute_day_features,
    load_industry_map,
)
from run_hs300_downside_walkforward import (
    PROJECT_ROOT,
    list_trade_dates,
    load_weight_table,
    make_universe,
    previous_weight_date,
)
from compute_multiproj_features import (
    Config as MultiprojConfig,
    compute_multiproj_day,
)
from run_phase14_5_consensus_walkforward import (
    BASE_CANDIDATE,
    fit_consensus_thresholds,
    Config as P145Config,
    consensus_dims,
)
from run_hs300_downside_walkforward import candidate_thresholds


DEFAULT_DB = REPO_ROOT / "paper_trading" / "paper_trading.db"


# =============================================================================
# Data source adapters
# =============================================================================

class DataSource:
    """Abstract data source. Implement to plug real-time feeds."""
    def read_stock_day(self, date_str: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError
    def read_etf_day(self, date_str: str, target_code: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError
    def latest_complete_minute(self) -> Optional[pd.Timestamp]:
        """For live mode: return the latest minute whose bar is fully closed.
        Returns None if no minute closed yet today."""
        raise NotImplementedError


class LocalParquetSource(DataSource):
    """Reads from local parquet under PROJECT_ROOT."""
    def __init__(self, stock_dir: str, etf_dir: str):
        self.stock_dir = Path(stock_dir)
        self.etf_dir = Path(etf_dir)

    def read_stock_day(self, date_str: str):
        p = self.stock_dir / f"{date_str}.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p)

    def read_etf_day(self, date_str: str, target_code: str):
        p = self.etf_dir / f"{date_str}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        df = df[df["code"] == target_code].copy()
        if df.empty:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df.sort_values("datetime").drop_duplicates("datetime", keep="last")

    def latest_complete_minute(self):
        # Local mode: pretend the whole day is available
        return None


# =============================================================================
# DB schema
# =============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    date TEXT NOT NULL,
    minute_idx INTEGER,
    state TEXT,
    DirScore_5 REAL,
    DirScore_10 REAL,
    Exhaustion_10 REAL,
    IPG_10 REAL,
    Bucket_Penetration_10 REAL,
    fired INTEGER NOT NULL,
    skipped_reason TEXT,
    fold_train_start TEXT,
    fold_train_end TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
CREATE INDEX IF NOT EXISTS idx_signals_fired ON signals(fired);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    date TEXT NOT NULL,
    signal_ts TEXT NOT NULL,
    entry_ts TEXT,
    exit_ts TEXT,
    entry_minute_idx INTEGER,
    exit_minute_idx INTEGER,
    side TEXT,
    entry_px REAL,
    exit_px REAL,
    gross_bps REAL,
    net_bps REAL,
    hold_minutes INTEGER,
    status TEXT,
    FOREIGN KEY(signal_id) REFERENCES signals(id)
);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

CREATE TABLE IF NOT EXISTS equity (
    date TEXT PRIMARY KEY,
    equity_close REAL,
    daily_pnl REAL,
    contracts INTEGER
);

CREATE TABLE IF NOT EXISTS run_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    date TEXT,
    mode TEXT,
    message TEXT
);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def log_run(conn, date_str, mode, message):
    conn.execute(
        "INSERT INTO run_log(ts, date, mode, message) VALUES(?,?,?,?)",
        (pd.Timestamp.now().isoformat(timespec="seconds"), date_str, mode, message),
    )
    conn.commit()


# =============================================================================
# Streaming engine
# =============================================================================

@dataclass
class FoldThresholds:
    train_start: str
    train_end: str
    exh10_max: float
    ipg10_min: float
    bp10_max: float


def derive_fold_thresholds(
    target_date: str,
    train_days: int = 120,
) -> FoldThresholds:
    """Refit base + consensus thresholds using the last `train_days` trading
    days strictly before target_date. This mirrors the walk-forward logic.

    On Render, where the V1 + multiproj panels are not present, we fall
    back to fold_thresholds_seed.json (frozen thresholds fitted on
    2025-11-03 to 2026-04-30 train window).
    """
    seed_json = REPO_ROOT / "paper_trading" / "fold_thresholds_seed.json"
    panel_path_v1 = REPO_ROOT / "results/510300_breadth_regime/hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet"

    if not panel_path_v1.exists() and seed_json.exists():
        # Render mode: load frozen thresholds
        with open(seed_json) as f:
            seed = json.load(f)
        print(f"[engine] V1 panel missing; loading frozen fold thresholds from {seed_json.name}")
        return FoldThresholds(
            train_start=seed["train_start"],
            train_end=seed["train_end"],
            exh10_max=float(seed["exh10_max"]),
            ipg10_min=float(seed["ipg10_min"]),
            bp10_max=float(seed["bp10_max"]),
        )

    from run_phase14_5_consensus_walkforward import (
        load_v1_panel,
        load_multiproj_panel,
        join_panels,
        add_kinematics,
    )

    v1_paths = ",".join([
        str(REPO_ROOT / "results/510300_breadth_regime/hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet"),
        str(REPO_ROOT / "results/510300_breadth_regime/hs300_phase1_5m_oos_20260306_precise_static_v1/scored_panel.parquet"),
    ])
    v1 = load_v1_panel(v1_paths)
    mp = load_multiproj_panel(str(REPO_ROOT / "results/510300_breadth_regime/hs300_multiproj_panel_v1/multiproj_panel.parquet"))
    j = join_panels(v1, mp)
    j = add_kinematics(j, lag=3)

    target = pd.Timestamp(target_date).normalize()
    j_pre = j[j["date"] < target].copy()
    train_dates_sorted = sorted(j_pre["date"].dt.normalize().unique())
    if len(train_dates_sorted) < train_days:
        raise RuntimeError(f"only {len(train_dates_sorted)} train days available, need {train_days}")
    selected = train_dates_sorted[-train_days:]
    train_window = j_pre[j_pre["date"].isin(selected)].copy()

    cs_th = fit_consensus_thresholds(train_window, P145Config())
    train_with_dims = consensus_dims(train_window, cs_th)
    base_th = candidate_thresholds(train_with_dims, pd.Series(BASE_CANDIDATE))
    return FoldThresholds(
        train_start=str(selected[0].date()),
        train_end=str(selected[-1].date()),
        exh10_max=float(base_th["exh10_max"]),
        ipg10_min=float(base_th.get("ipg10_min", float("-inf"))),
        bp10_max=float(cs_th["bucket_penetration_max"]),
    )


def stream_live_day(
    target_date: str,
    history_cfg_feature: FeatureConfig,
    live_cfg_feature: FeatureConfig,
    live_cfg_multiproj: MultiprojConfig,
    weights_by_date: dict,
    weight_dates_sorted: list,
    industry: pd.DataFrame,
    fold: FoldThresholds,
    conn: sqlite3.Connection,
    history_dates: Optional[list] = None,
    z_window_days: int = 20,
):
    """Live mode: history seeded from history_cfg (PROJECT_ROOT local),
    today's day fetched via live_cfg (akshare overlay)."""
    return _stream_internal(
        target_date=target_date,
        history_cfg_feature=history_cfg_feature,
        target_cfg_feature=live_cfg_feature,
        target_cfg_multiproj=live_cfg_multiproj,
        weights_by_date=weights_by_date,
        weight_dates_sorted=weight_dates_sorted,
        industry=industry,
        fold=fold,
        conn=conn,
        history_dates=history_dates,
        z_window_days=z_window_days,
    )


def stream_one_day(
    target_date: str,
    cfg_feature: FeatureConfig,
    cfg_multiproj: MultiprojConfig,
    weights_by_date: dict,
    weight_dates_sorted: list,
    industry: pd.DataFrame,
    fold: FoldThresholds,
    conn: sqlite3.Connection,
    history_dates: Optional[list] = None,
    z_window_days: int = 20,
):
    """Replay/backfill mode: same cfg for history and target (local parquet)."""
    return _stream_internal(
        target_date=target_date,
        history_cfg_feature=cfg_feature,
        target_cfg_feature=cfg_feature,
        target_cfg_multiproj=cfg_multiproj,
        weights_by_date=weights_by_date,
        weight_dates_sorted=weight_dates_sorted,
        industry=industry,
        fold=fold,
        conn=conn,
        history_dates=history_dates,
        z_window_days=z_window_days,
    )


def _stream_internal(
    target_date: str,
    history_cfg_feature: FeatureConfig,
    target_cfg_feature: FeatureConfig,
    target_cfg_multiproj: MultiprojConfig,
    weights_by_date: dict,
    weight_dates_sorted: list,
    industry: pd.DataFrame,
    fold: FoldThresholds,
    conn: sqlite3.Connection,
    history_dates: Optional[list] = None,
    z_window_days: int = 20,
):
    """Shared internal streaming loop. history_cfg can differ from target_cfg
    so live mode can pull history from local parquet and target from akshare
    overlay."""
    # Use target_cfg as the "primary" cfg variable name for downstream code
    cfg_feature = target_cfg_feature
    cfg_multiproj = target_cfg_multiproj
    """Stream a single day minute-by-minute. Logs every minute as a signal
    row (with `fired=0` if rule did not pass), and inserts trades when
    the corresponding entry/exit minute arrives.
    """
    if history_dates is None:
        try:
            all_dates = list_trade_dates(
                Path(history_cfg_feature.stock_data_dir),
                Path(history_cfg_feature.etf_data_dir),
                "2024-01-01",
                target_date,
            )
            history_dates = sorted([d for d in all_dates if pd.Timestamp(d).normalize() < pd.Timestamp(target_date).normalize()])
        except Exception:
            history_dates = []

    # Build z-deques from prior days
    raw_cols = [f"{name}_{h}_raw" for h in HORIZONS for name in RAW_NAMES]
    deques: dict = {col: defaultdict(lambda: deque(maxlen=z_window_days)) for col in raw_cols}

    # Try seed-based bootstrap first (Render mode where local parquet is unavailable).
    seed_path = REPO_ROOT / "paper_trading" / "zdeque_seed.parquet"
    used_seed = False
    if not history_dates and seed_path.exists():
        print(f"[engine] no local history; bootstrapping z-deques from {seed_path.name}")
        seed = pd.read_parquet(seed_path)
        seed = seed.sort_values(["date", "minute_idx"]).reset_index(drop=True)
        for col in raw_cols:
            if col not in seed.columns:
                continue
            for (mi, val_series) in seed[["minute_idx", col]].groupby("minute_idx"):
                vals = val_series[col].dropna().astype(float).tolist()
                # Take the most recent up to z_window_days
                for v in vals[-z_window_days:]:
                    deques[col][int(mi)].append(v)
        n_seed_days = seed["date"].nunique()
        print(f"[engine] seeded {n_seed_days} prior days into z-deques")
        used_seed = True

    print(f"[engine] populating z-deques from {min(z_window_days+5, len(history_dates))} prior days ...")
    skipped = 0
    for d in history_dates[-(z_window_days + 5):]:
        wdt = previous_weight_date(weight_dates_sorted, pd.Timestamp(d).normalize())
        if wdt is None:
            skipped += 1
            continue
        try:
            uni = make_universe(weights_by_date[wdt], industry, history_cfg_feature.leader_n, history_cfg_feature.weight_bins)
            day = compute_day_features(d, uni, history_cfg_feature)
        except Exception:
            skipped += 1
            continue
        if day is None or day.empty:
            skipped += 1
            continue
        day = day.sort_values("minute_idx").reset_index(drop=True)
        for _, row in day.iterrows():
            mi = int(row["minute_idx"])
            for col in raw_cols:
                v = row.get(col)
                if v is not None and np.isfinite(v):
                    deques[col][mi].append(float(v))
    print(f"[engine] history populated; skipped {skipped} days due to data gaps")

    # Today's raw and multiproj (these are batch-built but the per-minute
    # values for minute t depend ONLY on data through t; verified by the
    # streaming_lookahead_audit.py reconciliation)
    wdt_today = previous_weight_date(weight_dates_sorted, pd.Timestamp(target_date).normalize())
    if wdt_today is None:
        print(f"[engine] no weight available for {target_date}; aborting")
        log_run(conn, target_date, "stream", "no weight date available; aborted")
        return
    universe = make_universe(weights_by_date[wdt_today], industry, cfg_feature.leader_n, cfg_feature.weight_bins)
    raw_today = compute_day_features(target_date, universe, cfg_feature)
    multi_today = compute_multiproj_day(target_date, universe, cfg_multiproj)
    if raw_today is None or multi_today is None:
        print(f"[engine] no raw/multiproj data for {target_date}; aborting")
        log_run(conn, target_date, "stream", "missing raw or multiproj; aborted")
        return
    raw_today = raw_today.sort_values("minute_idx").reset_index(drop=True)
    multi_today = multi_today.sort_values("minute_idx").reset_index(drop=True)

    print(f"[engine] streaming {len(raw_today)} minutes for {target_date}")
    pending_entries: list[dict] = []
    pending_exits: list[dict] = []
    next_ok_after = -1
    trades_today_count = 0
    MAX_TRADES = 2

    for mi in range(len(raw_today)):
        row_raw = raw_today.iloc[mi]
        ts = pd.Timestamp(row_raw["datetime"])
        z_values = {}
        for col in raw_cols:
            arr = np.asarray(deques[col][mi], dtype=float)
            v = row_raw.get(col)
            if not np.isfinite(v) or len(arr) < cfg_feature.z_min_history:
                z_values[col] = np.nan
            else:
                med = float(np.median(arr))
                mad = float(np.median(np.abs(arr - med)))
                z_values[col] = (float(v) - med) / (1.4826 * mad) if mad > 0 else 0.0
            if np.isfinite(v):
                deques[col][mi].append(float(v))

        z = lambda name, h: z_values.get(f"{name}_{h}_raw", 0.0) if np.isfinite(z_values.get(f"{name}_{h}_raw", np.nan)) else 0.0

        composites = {}
        for h in HORIZONS:
            internal = 0.30*z("WSB",h) + 0.20*z("dB",h) + 0.20*z("Corr",h) + 0.15*z("EP",h) + 0.15*z("SE",h)
            ipg = internal - z("ETFRet", h)
            pad = z("ETFRet", h) - internal
            gate = 0.22*z("B",h) + 0.18*z("dB",h) + 0.18*z("Corr",h) + 0.14*z("EP",h) + 0.12*z("SE",h) + 0.08*z("Dir",h) - 0.08*z("TopK",h)
            direction = 0.40*ipg + 0.30*z("LFR",h) + 0.20*z("WPS",h) + 0.10*z("WSB",h)
            exhaustion = 0.50*max(pad,0.0) + 0.30*max(z("TopK",h),0.0) + 0.20*max(-z("dB",h),0.0)
            gate_thr = cfg_feature.gate_5 if h == 5 else cfg_feature.gate_10
            composites[f"IPG_{h}"] = ipg
            composites[f"DirScore_{h}"] = direction
            composites[f"Exhaustion_{h}"] = exhaustion
            composites[f"Gate_{h}"] = gate > gate_thr

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

        hhmm = ts.hour * 100 + ts.minute
        regular = (hhmm >= 935) and not (1125 <= hhmm <= 1305) and (hhmm < 1450)

        mp_row = multi_today.iloc[mi]
        bp10 = float(mp_row.get("Bucket_Penetration_10", np.nan))

        signal_fired = (
            state == "EMERGING_DOWN" and regular and mi >= 30
            and dir5 <= -0.75 and dir10 <= 0.45
            and np.isfinite(exh10) and exh10 <= fold.exh10_max
            and np.isfinite(composites["IPG_10"]) and composites["IPG_10"] >= fold.ipg10_min
            and np.isfinite(bp10) and bp10 <= fold.bp10_max
        )

        skipped_reason = ""
        if signal_fired and trades_today_count >= MAX_TRADES:
            skipped_reason = "max_trades_per_day"
            signal_fired = False
        if signal_fired and mi < next_ok_after:
            skipped_reason = "overlap"
            signal_fired = False
        if signal_fired and (mi + 1 + 30) >= len(raw_today):
            skipped_reason = "no_room_for_30min_hold"
            signal_fired = False

        # Insert signal row (whether fired or not, for transparency)
        cur = conn.execute(
            "INSERT INTO signals(ts, date, minute_idx, state, DirScore_5, DirScore_10, Exhaustion_10, IPG_10, Bucket_Penetration_10, fired, skipped_reason, fold_train_start, fold_train_end) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ts.isoformat(timespec="seconds"), target_date, mi, state,
                dir5, dir10, exh10, composites["IPG_10"], bp10,
                1 if signal_fired else 0,
                skipped_reason if skipped_reason else None,
                fold.train_start, fold.train_end,
            ),
        )
        signal_id = cur.lastrowid
        conn.commit()

        # Process pending exits: do any close at this minute?
        for pe in list(pending_exits):
            if pe["exit_minute_idx"] == mi:
                exit_px = float(row_raw["etf_open"])
                gross_log = -math.log(exit_px / pe["entry_px"])
                gross_bps = gross_log * 10000.0
                net_bps = gross_bps - 4.5
                conn.execute(
                    "UPDATE trades SET exit_ts=?, exit_px=?, gross_bps=?, net_bps=?, status=? WHERE id=?",
                    (ts.isoformat(timespec="seconds"), exit_px, gross_bps, net_bps, "CLOSED", pe["trade_id"]),
                )
                conn.commit()
                print(f"  [{ts}] EXIT  trade#{pe['trade_id']} @ {exit_px:.4f}  net_bps={net_bps:+.2f}")
                pending_exits.remove(pe)

        # Process pending entries: do any open at this minute?
        for pent in list(pending_entries):
            if pent["entry_minute_idx"] == mi:
                entry_px = float(row_raw["etf_open"])
                cur = conn.execute(
                    "INSERT INTO trades(signal_id, date, signal_ts, entry_ts, entry_minute_idx, exit_minute_idx, side, entry_px, hold_minutes, status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        pent["signal_id"], target_date, pent["signal_ts"],
                        ts.isoformat(timespec="seconds"), mi, pent["exit_minute_idx"],
                        "SHORT", entry_px, 30, "OPEN",
                    ),
                )
                trade_id = cur.lastrowid
                conn.commit()
                pending_exits.append({
                    "trade_id": trade_id,
                    "entry_px": entry_px,
                    "exit_minute_idx": pent["exit_minute_idx"],
                })
                pending_entries.remove(pent)
                print(f"  [{ts}] ENTRY trade#{trade_id} SHORT @ {entry_px:.4f} (will exit at minute {pent['exit_minute_idx']})")

        # If signal fired this minute, schedule entry at mi+1
        if signal_fired:
            print(f"  [{ts}] SIGNAL minute_idx={mi} state={state} dir5={dir5:+.2f} dir10={dir10:+.2f} ipg10={composites['IPG_10']:+.2f} bp10={bp10:+.3f}")
            pending_entries.append({
                "signal_id": signal_id,
                "signal_ts": ts.isoformat(timespec="seconds"),
                "entry_minute_idx": mi + 1,
                "exit_minute_idx": mi + 1 + 30,
            })
            next_ok_after = mi + 1 + 30
            trades_today_count += 1

    log_run(conn, target_date, "stream", f"completed; trades={trades_today_count}")


# =============================================================================
# CLI
# =============================================================================

def fetch_via_akshare_into_overlay(target_date: str) -> tuple[Path, Path]:
    """Fetch universe + ETF for target_date via akshare, save in local-format
    parquet under paper_trading/akshare_overlay. Returns the overlay dirs.

    Used by `live` and `live-once` modes to produce parquet that has the same
    schema as PROJECT_ROOT/stock_data and PROJECT_ROOT/ETF data core7 precise.
    """
    from data_source_akshare import AkshareDataSource
    overlay = REPO_ROOT / "paper_trading" / "akshare_overlay"
    stock_dir = overlay / "stock"
    etf_dir = overlay / "etf"
    stock_dir.mkdir(parents=True, exist_ok=True)
    etf_dir.mkdir(parents=True, exist_ok=True)
    src = AkshareDataSource(REPO_ROOT / "paper_trading" / "akshare_cache")

    universe = src.fetch_universe_bars(target_date=target_date)
    universe.to_parquet(stock_dir / f"{target_date}.parquet", index=False)
    etf = src.fetch_etf_bars(target_date=target_date)
    etf.to_parquet(etf_dir / f"{target_date}.parquet", index=False)
    print(f"[live] saved akshare overlay: stock={len(universe)} rows, etf={len(etf)} rows")
    return stock_dir, etf_dir


def main():
    parser = argparse.ArgumentParser(description="Live signal engine for HS300 V1.5 strategy")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_replay = sub.add_parser("replay", help="Replay one historical day from local parquet")
    p_replay.add_argument("date", help="YYYY-MM-DD")
    p_replay.add_argument("--db", default=str(DEFAULT_DB))
    p_backfill = sub.add_parser("backfill", help="Replay a date range")
    p_backfill.add_argument("start", help="YYYY-MM-DD")
    p_backfill.add_argument("end", help="YYYY-MM-DD")
    p_backfill.add_argument("--db", default=str(DEFAULT_DB))
    p_live = sub.add_parser("live-once", help="Fetch one date via akshare and stream-process. History from PROJECT_ROOT.")
    p_live.add_argument("--target-date", default=None, help="YYYY-MM-DD; default = latest akshare trading date")
    p_live.add_argument("--db", default=str(DEFAULT_DB))
    p_live.add_argument("--reset-today", action="store_true", help="Delete existing rows for target_date before processing (idempotent re-run)")
    p_status = sub.add_parser("status", help="Show DB status")
    p_status.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    # On Render the project-root artifact is not present; fall back to the
    # shipped copy under paper_trading/.
    industry_map_local = PROJECT_ROOT / "artifacts/ashare_t1_xgb_stfree_mcap10_500_v2_fullrun/industry_map_baostock.parquet"
    industry_map_shipped = REPO_ROOT / "paper_trading/industry_map_baostock.parquet"
    industry_map_path = str(industry_map_local if industry_map_local.exists() else industry_map_shipped)

    cfg_feature = FeatureConfig(
        stock_data_dir=str(PROJECT_ROOT / "stock_data"),
        etf_data_dir=str(PROJECT_ROOT / "ETF data core7 precise"),
        weights_csv="",
        industry_map_path=industry_map_path,
        output_dir=str(REPO_ROOT / "paper_trading" / "_tmp"),
        amount_min=100000.0,
        min_active_constituents=120,
    )
    cfg_multiproj = MultiprojConfig()
    weights_by_date = load_weight_table(str(REPO_ROOT / "research/strategy_lab/data/hs300_daily_weights/hs300_weights_2025-01-01_2026-04-30.csv"))
    weight_dates_sorted = sorted(weights_by_date.keys())
    industry = load_industry_map(industry_map_path)

    if args.cmd in ("replay", "backfill"):
        conn = init_db(Path(args.db))
        all_dates = list_trade_dates(Path(cfg_feature.stock_data_dir), Path(cfg_feature.etf_data_dir), "2024-01-01", "2026-12-31")
        if args.cmd == "replay":
            dates_to_run = [args.date]
        else:
            dates_to_run = sorted([d for d in all_dates if args.start <= d <= args.end])
        for d in dates_to_run:
            print(f"\n========= {d} =========")
            try:
                fold = derive_fold_thresholds(d, train_days=120)
                history = sorted([h for h in all_dates if pd.Timestamp(h).normalize() < pd.Timestamp(d).normalize()])
                stream_one_day(
                    d, cfg_feature, cfg_multiproj,
                    weights_by_date, weight_dates_sorted, industry,
                    fold, conn, history_dates=history,
                )
            except Exception as exc:
                print(f"[engine] {d} failed: {exc}")
                log_run(conn, d, args.cmd, f"FAILED: {exc}")
        conn.close()
    elif args.cmd == "live-once":
        # Resolve target_date (default = latest akshare trading day)
        target_date = args.target_date
        if target_date is None:
            from data_source_akshare import AkshareDataSource
            src = AkshareDataSource(REPO_ROOT / "paper_trading" / "akshare_cache")
            target_date = src.latest_trading_date()
            print(f"[live-once] target_date defaulted to akshare latest = {target_date}")
        # Fetch akshare data and write to overlay dir
        overlay_stock, overlay_etf = fetch_via_akshare_into_overlay(target_date)
        # Build feature config that uses the overlay dir for THIS date and
        # local PROJECT_ROOT for history
        live_cfg_feature = FeatureConfig(
            stock_data_dir=str(overlay_stock),
            etf_data_dir=str(overlay_etf),
            weights_csv="",
            industry_map_path=cfg_feature.industry_map_path,
            output_dir=cfg_feature.output_dir,
            amount_min=100000.0,
            min_active_constituents=120,
        )
        live_cfg_multiproj = MultiprojConfig(
            stock_data_dir=str(overlay_stock),
            etf_data_dir=str(overlay_etf),
            industry_map_path=cfg_feature.industry_map_path,
        )
        # Build history from PROJECT_ROOT
        all_dates = list_trade_dates(Path(cfg_feature.stock_data_dir), Path(cfg_feature.etf_data_dir), "2024-01-01", "2026-12-31")
        history = sorted([h for h in all_dates if pd.Timestamp(h).normalize() < pd.Timestamp(target_date).normalize()])
        # Init DB and optionally clean today's rows
        conn = init_db(Path(args.db))
        if args.reset_today:
            conn.execute("DELETE FROM trades WHERE date=?", (target_date,))
            conn.execute("DELETE FROM signals WHERE date=?", (target_date,))
            conn.commit()
            print(f"[live-once] reset today's rows for {target_date}")
        # Patched stream_one_day: history uses cfg_feature (PROJECT_ROOT),
        # target_date inside stream_one_day uses live_cfg_feature
        try:
            fold = derive_fold_thresholds(target_date, train_days=120)
            stream_live_day(
                target_date, cfg_feature, live_cfg_feature, live_cfg_multiproj,
                weights_by_date, weight_dates_sorted, industry,
                fold, conn, history_dates=history,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log_run(conn, target_date, "live-once", f"FAILED: {exc}")
        conn.close()
    elif args.cmd == "status":
        conn = init_db(Path(args.db))
        n_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        n_fired = conn.execute("SELECT COUNT(*) FROM signals WHERE fired=1").fetchone()[0]
        n_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        n_open = conn.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
        n_closed = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()[0]
        total_bps = conn.execute("SELECT COALESCE(SUM(net_bps),0) FROM trades WHERE status='CLOSED'").fetchone()[0]
        avg_bps = conn.execute("SELECT COALESCE(AVG(net_bps),0) FROM trades WHERE status='CLOSED'").fetchone()[0]
        wins = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND net_bps > 0").fetchone()[0]
        print(f"signals: {n_signals} ({n_fired} fired)")
        print(f"trades: {n_trades} ({n_open} open, {n_closed} closed)")
        if n_closed:
            print(f"  win_rate: {wins/n_closed:.2%}")
            print(f"  total_net_bps: {total_bps:.2f}")
            print(f"  avg_net_bps: {avg_bps:.2f}")
        conn.close()


if __name__ == "__main__":
    main()
