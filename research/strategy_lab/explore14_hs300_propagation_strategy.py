#!/usr/bin/env python3
"""
Explore 14: HS300 Resonance Propagation Strategy
================================================

First-pass implementation of the three-layer intraday framework:
1) Gate: is the next 5m / 10m interval worth trading?
2) Direction: is internal stock pressure likely to release up or down?
3) Exhaustion: has ETF price already moved too far relative to internals?

Data assumptions for this v1:
- Constituents/weights: static HS300 snapshot from hs300_weights.csv
- Sector map: static industry map from baostock cache
- Signal source: HS300 stock minute data from stock_data/
- Trade target: real ETF minute data from ETF data/, default 510300.XSHG

Known limitation:
- Static 2026-style HS300 weights introduce survivorship bias on older samples.
  This is acceptable for a first-pass signal-engine build but should be replaced
  by historical constituent snapshots before trusting long-range production stats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[2]
STOCK_DATA_DIR = ROOT_DIR / "stock_data"
ETF_DATA_DIR = ROOT_DIR / "ETF data"
DATA_DIR = Path(__file__).resolve().parent / "data"
RESULT_DIR = Path(__file__).resolve().parent / "results" / "explore14"

HORIZONS = (5, 10)
DIAG_HORIZONS = (5, 10, 15, 20, 30, 45, 60, 90, 120)


@dataclass
class StrategyConfig:
    stock_data_dir: str = str(STOCK_DATA_DIR)
    etf_data_dir: str = str(ETF_DATA_DIR)
    weights_csv: str = str(DATA_DIR / "hs300_weights.csv")
    universe_mode: str = "dynamic_mcap_proxy"
    industry_map_path: str = str(
        ROOT_DIR / "artifacts" / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun" / "industry_map_baostock.parquet"
    )
    daily_mcap_panel_path: str = str(
        ROOT_DIR / "artifacts" / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun" / "feature_table_stfree_mcap_advanced.parquet"
    )
    output_dir: str = str(RESULT_DIR)
    target_etf_code: str = "510300.XSHG"
    start_date: str = "2015-01-01"
    end_date: str = "2021-08-17"
    max_days: int = 0
    amount_min: float = 100_000.0
    min_active_constituents: int = 120
    min_market_active: int = 1500
    z_window_days: int = 20
    z_min_history: int = 5
    tick_size_stock: float = 0.01
    eps_floor_5: float = 0.00020
    eps_floor_10: float = 0.00030
    topk_k: int = 10
    leader_n: int = 30
    weight_bins: int = 5
    proxy_top_n: int = 320
    market_large_n: int = 300
    market_limit_band: float = 0.002
    round_trip_cost_bps: float = 6.0
    gate_enter_5: float = 0.60
    gate_enter_10: float = 0.55
    gate_keep_5: float = 0.40
    gate_keep_10: float = 0.35
    dir_th_5: float = 0.50
    dir_th_10: float = 0.45
    exh_th_5: float = 0.80
    exh_th_10: float = 0.90
    trade_confirmed_only: bool = True
    trade_enable_long: bool = True
    trade_enable_short: bool = True
    trade_dir_long_10: float = 1.20
    trade_dir_short_10: float = -1.20
    trade_exhaustion_max_10: float = 0.40
    trade_followthrough_long_10: float = 0.20
    trade_trend_cont_long_10: float = 0.40
    trade_day_cont_long_10: float = 0.50
    trade_trend_day_long_10: float = 0.80
    trade_day_cont_short_10: float = 0.50
    trade_trend_day_short_10: float = 0.80
    trade_min_hold_10: int = 5
    trade_max_hold_10: int = 10
    trade_hold_long_confirmed: int = 15
    trade_hold_short_confirmed: int = 30
    trade_hold_long_strong: int = 30
    trade_hold_short_strong: int = 45
    trade_hold_long_combo: int = 45
    trade_hold_long_trendday: int = 90
    trade_hold_short_trendday: int = 75
    trade_min_hold_long_confirmed: int = 8
    trade_min_hold_short_confirmed: int = 12
    trade_min_hold_long_strong: int = 12
    trade_min_hold_short_strong: int = 18
    trade_min_hold_long_combo: int = 15
    trade_min_hold_long_trendday: int = 30
    trade_min_hold_short_trendday: int = 25
    trade_entry_trend_support_10: float = 0.80
    trade_strong_trend_support_10: float = 1.60
    trade_keep_trend_support_10: float = 0.20
    trade_max_per_day: int = 3
    trade_cooldown_bars: int = 20
    preview_days: int = 5


def parse_args() -> StrategyConfig:
    p = argparse.ArgumentParser(description="Explore 14 HS300 resonance propagation strategy")
    p.add_argument("--stock-data-dir", type=str, default=str(STOCK_DATA_DIR))
    p.add_argument("--etf-data-dir", type=str, default=str(ETF_DATA_DIR))
    p.add_argument("--weights-csv", type=str, default=str(DATA_DIR / "hs300_weights.csv"))
    p.add_argument("--universe-mode", type=str, default="dynamic_mcap_proxy", choices=["static", "dynamic_mcap_proxy"])
    p.add_argument(
        "--industry-map-path",
        type=str,
        default=str(ROOT_DIR / "artifacts" / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun" / "industry_map_baostock.parquet"),
    )
    p.add_argument(
        "--daily-mcap-panel-path",
        type=str,
        default=str(ROOT_DIR / "artifacts" / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun" / "feature_table_stfree_mcap_advanced.parquet"),
    )
    p.add_argument("--output-dir", type=str, default=str(RESULT_DIR))
    p.add_argument("--target-etf-code", type=str, default="510300.XSHG")
    p.add_argument("--start-date", type=str, default="2015-01-01")
    p.add_argument("--end-date", type=str, default="2021-08-17")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--amount-min", type=float, default=100_000.0)
    p.add_argument("--min-active-constituents", type=int, default=120)
    p.add_argument("--min-market-active", type=int, default=1500)
    p.add_argument("--z-window-days", type=int, default=20)
    p.add_argument("--z-min-history", type=int, default=5)
    p.add_argument("--proxy-top-n", type=int, default=320)
    p.add_argument("--market-large-n", type=int, default=300)
    p.add_argument("--market-limit-band", type=float, default=0.002)
    p.add_argument("--round-trip-cost-bps", type=float, default=6.0)
    p.add_argument(
        "--trade-confirmed-only",
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    p.add_argument(
        "--trade-enable-long",
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    p.add_argument(
        "--trade-enable-short",
        type=lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    p.add_argument("--trade-dir-long-10", type=float, default=1.20)
    p.add_argument("--trade-dir-short-10", type=float, default=-1.20)
    p.add_argument("--trade-exhaustion-max-10", type=float, default=0.40)
    p.add_argument("--trade-followthrough-long-10", type=float, default=0.20)
    p.add_argument("--trade-trend-cont-long-10", type=float, default=0.40)
    p.add_argument("--trade-day-cont-long-10", type=float, default=0.50)
    p.add_argument("--trade-trend-day-long-10", type=float, default=0.80)
    p.add_argument("--trade-day-cont-short-10", type=float, default=0.50)
    p.add_argument("--trade-trend-day-short-10", type=float, default=0.80)
    p.add_argument("--trade-min-hold-10", type=int, default=5)
    p.add_argument("--trade-max-hold-10", type=int, default=10)
    p.add_argument("--trade-hold-long-confirmed", type=int, default=15)
    p.add_argument("--trade-hold-short-confirmed", type=int, default=30)
    p.add_argument("--trade-hold-long-strong", type=int, default=30)
    p.add_argument("--trade-hold-short-strong", type=int, default=45)
    p.add_argument("--trade-hold-long-combo", type=int, default=45)
    p.add_argument("--trade-hold-long-trendday", type=int, default=90)
    p.add_argument("--trade-hold-short-trendday", type=int, default=75)
    p.add_argument("--trade-min-hold-long-confirmed", type=int, default=8)
    p.add_argument("--trade-min-hold-short-confirmed", type=int, default=12)
    p.add_argument("--trade-min-hold-long-strong", type=int, default=12)
    p.add_argument("--trade-min-hold-short-strong", type=int, default=18)
    p.add_argument("--trade-min-hold-long-combo", type=int, default=15)
    p.add_argument("--trade-min-hold-long-trendday", type=int, default=30)
    p.add_argument("--trade-min-hold-short-trendday", type=int, default=25)
    p.add_argument("--trade-entry-trend-support-10", type=float, default=0.80)
    p.add_argument("--trade-strong-trend-support-10", type=float, default=1.60)
    p.add_argument("--trade-keep-trend-support-10", type=float, default=0.20)
    p.add_argument("--trade-max-per-day", type=int, default=3)
    p.add_argument("--trade-cooldown-bars", type=int, default=20)
    args = p.parse_args()
    return StrategyConfig(**vars(args))


def shift_matrix(arr: np.ndarray, steps: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if steps <= 0:
        out[:] = arr
        return out
    if steps < len(arr):
        out[steps:] = arr[:-steps]
    return out


def shift_vector(arr: np.ndarray, steps: int) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if steps <= 0:
        out[:] = arr
        return out
    if steps < len(arr):
        out[steps:] = arr[:-steps]
    return out


def normalize_rows(weights: np.ndarray) -> np.ndarray:
    row_sum = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, row_sum, out=np.zeros_like(weights, dtype=float), where=row_sum > 0)


def rolling_sum_matrix(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.where(np.isfinite(arr), arr, 0.0)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    csum = np.cumsum(clean, axis=0, dtype=float)
    out = csum.copy()
    if window < len(arr):
        out[window:] = csum[window:] - csum[:-window]
    return out


def rolling_mean_matrix(arr: np.ndarray, window: int) -> np.ndarray:
    clean = np.where(np.isfinite(arr), arr, 0.0)
    count = rolling_sum_matrix(np.isfinite(arr).astype(float), window)
    total = rolling_sum_matrix(clean, window)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def load_industry_map(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        ind = pd.read_parquet(path)[["code", "industry"]].copy()
        ind["code"] = ind["code"].astype(str).str.strip()
        ind["industry"] = ind["industry"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
        ind = ind.drop_duplicates("code", keep="last")
        return ind
    return pd.DataFrame(columns=["code", "industry"])


def load_static_universe(cfg: StrategyConfig) -> pd.DataFrame:
    w = pd.read_csv(cfg.weights_csv)
    if "code" not in w.columns or "weight_pct" not in w.columns:
        raise ValueError("weights csv must contain code, weight_pct")
    w = w[["code", "weight_pct"]].copy()
    w["code"] = w["code"].astype(str).str.strip()
    w["weight"] = w["weight_pct"].astype(float) / 100.0
    w = w[w["weight"] > 0].copy()
    w["weight"] = w["weight"] / w["weight"].sum()

    ind = load_industry_map(cfg.industry_map_path)
    if not ind.empty:
        w = w.merge(ind, on="code", how="left")
    else:
        w["industry"] = "Unknown"
    w["industry"] = w["industry"].fillna("Unknown").astype(str)

    w = w.sort_values("weight", ascending=False).reset_index(drop=True)
    w["leader_flag"] = 0
    w.loc[: max(cfg.leader_n - 1, 0), "leader_flag"] = 1
    w["weight_bin"] = pd.qcut(
        w["weight"].rank(method="first", ascending=True),
        q=min(cfg.weight_bins, len(w)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    return w


def load_mcap_panel(path: str, start_date: str = "", end_date: str = "") -> pd.DataFrame:
    cols = ["date", "code", "float_mcap_lag"]
    df = pd.read_parquet(path, columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = df["code"].astype(str).str.strip()
    df["float_mcap_lag"] = pd.to_numeric(df["float_mcap_lag"], errors="coerce")
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df = df.dropna(subset=["float_mcap_lag"])
    return df


def build_dynamic_proxy_universe(
    date_str: str,
    mcap_panel: pd.DataFrame,
    industry_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    d = pd.to_datetime(date_str).normalize()
    day = mcap_panel[mcap_panel["date"] == d][["code", "float_mcap_lag"]].copy()
    if day.empty:
        return pd.DataFrame(columns=["code", "weight", "industry", "leader_flag", "weight_bin"])
    day = day.sort_values("float_mcap_lag", ascending=False).head(cfg.proxy_top_n).copy()
    day = day.rename(columns={"float_mcap_lag": "weight"})
    day = day[day["weight"] > 0].copy()
    if day.empty:
        return pd.DataFrame(columns=["code", "weight", "industry", "leader_flag", "weight_bin"])
    day["weight"] = day["weight"] / day["weight"].sum()
    if not industry_df.empty:
        day = day.merge(industry_df, on="code", how="left")
    day["industry"] = day.get("industry", pd.Series("Unknown", index=day.index)).fillna("Unknown").astype(str)
    day = day.sort_values("weight", ascending=False).reset_index(drop=True)
    day["leader_flag"] = 0
    day.loc[: max(cfg.leader_n - 1, 0), "leader_flag"] = 1
    q = min(cfg.weight_bins, len(day))
    day["weight_bin"] = pd.qcut(
        day["weight"].rank(method="first", ascending=True),
        q=q,
        labels=False,
        duplicates="drop",
    ).astype(int)
    return day


def build_market_meta_day(
    date_str: str,
    codes: Iterable[str],
    mcap_panel: pd.DataFrame,
    industry_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> pd.DataFrame:
    meta = pd.DataFrame({"code": list(codes)})
    if not mcap_panel.empty:
        d = pd.to_datetime(date_str).normalize()
        day = mcap_panel[mcap_panel["date"] == d][["code", "float_mcap_lag"]].copy()
        meta = meta.merge(day, on="code", how="left")
    else:
        meta["float_mcap_lag"] = np.nan
    if not industry_df.empty:
        meta = meta.merge(industry_df, on="code", how="left")
    meta["industry"] = meta.get("industry", pd.Series("Unknown", index=meta.index)).fillna("Unknown").astype(str)
    meta["float_mcap_lag"] = pd.to_numeric(meta["float_mcap_lag"], errors="coerce")
    meta["mcap_weight"] = meta["float_mcap_lag"].fillna(0.0)
    total_mcap = float(meta.loc[meta["mcap_weight"] > 0, "mcap_weight"].sum())
    if total_mcap > 0:
        meta["mcap_weight"] = meta["mcap_weight"] / total_mcap
    else:
        meta["mcap_weight"] = 0.0
    meta["large_flag"] = 0
    known = meta["float_mcap_lag"].notna()
    if known.any():
        top_idx = meta.loc[known].sort_values("float_mcap_lag", ascending=False).head(cfg.market_large_n).index
        meta.loc[top_idx, "large_flag"] = 1
    return meta


def read_stock_day(path: Path, codes: Iterable[str] | None = None, columns: Iterable[str] | None = None) -> pd.DataFrame:
    read_columns = list(columns) if columns is not None else ["code", "datetime", "close", "volume", "money", "paused"]
    kwargs = {"columns": read_columns}
    if codes is not None:
        kwargs["filters"] = [("code", "in", list(codes))]
    table = pq.read_table(path, **kwargs)
    df = table.to_pandas()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    if "paused" in df.columns:
        df["paused"] = df["paused"].fillna(0.0).astype(float)
    return df


def read_etf_day(path: Path, target_code: str) -> pd.DataFrame:
    table = pq.read_table(
        path,
        columns=["code", "datetime", "open", "close", "volume", "money", "paused"],
        filters=[("code", "==", target_code)],
    )
    df = table.to_pandas()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["paused"] = df["paused"].fillna(0.0).astype(float)
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return df


def select_dates(cfg: StrategyConfig) -> List[str]:
    stock_dates = {p.stem for p in Path(cfg.stock_data_dir).glob("*.parquet")}
    etf_dates = {p.stem for p in Path(cfg.etf_data_dir).glob("*.parquet")}
    dates = sorted(stock_dates & etf_dates)
    if cfg.start_date:
        dates = [d for d in dates if d >= cfg.start_date]
    if cfg.end_date:
        dates = [d for d in dates if d <= cfg.end_date]
    if cfg.max_days and cfg.max_days > 0:
        dates = dates[-cfg.max_days :]
    if not dates:
        raise ValueError("No overlapping stock/ETF dates after filtering.")
    return dates


def compute_corr_fast(ret1_df: pd.DataFrame, active_df: pd.DataFrame, current_active_count: np.ndarray, window: int) -> np.ndarray:
    min_periods = max(3, window // 2)
    valid_ret = ret1_df.where(active_df)
    ew = valid_ret.mean(axis=1)
    var_ew = ew.rolling(window, min_periods=min_periods).var(ddof=0)
    var_i = valid_ret.rolling(window, min_periods=min_periods).var(ddof=0).mean(axis=1)
    n_eff = pd.Series(current_active_count, index=ret1_df.index, dtype=float).clip(lower=2.0)
    corr = ((n_eff * var_ew / var_i) - 1.0) / (n_eff - 1.0)
    corr = corr.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
    return corr.to_numpy(dtype=float)


def compute_state(row: pd.Series, cfg: StrategyConfig) -> Tuple[str, int]:
    gate5 = bool(row["Gate_5"])
    gate10 = bool(row["Gate_10"])
    dir5 = float(row["DirScore_5"])
    dir10 = float(row["DirScore_10"])
    exh5 = float(row["Exhaustion_5"])
    exh10 = float(row["Exhaustion_10"])
    db5 = float(row.get("z_dB_5_raw", 0.0))
    lfr5 = float(row.get("z_LFR_5_raw", 0.0))
    trend5 = float(row.get("TrendSupport_5", 0.0))
    trend10 = float(row.get("TrendSupport_10", 0.0))
    market10 = float(row.get("MarketPressure_10", 0.0))

    if (gate5 or gate10) and (exh5 > cfg.exh_th_5 or exh10 > cfg.exh_th_10 or (db5 < -0.25 and lfr5 < -0.25)):
        return "EXHAUSTED", 0
    if gate5 and gate10 and dir5 > cfg.dir_th_5 and dir10 > cfg.dir_th_10 and trend10 > 0.0 and market10 > -0.35 and exh10 < cfg.exh_th_10:
        return "CONFIRMED_UP", 10
    if gate5 and gate10 and dir5 < -cfg.dir_th_5 and dir10 < -cfg.dir_th_10 and trend10 > 0.0 and market10 < 0.35 and exh10 < cfg.exh_th_10:
        return "CONFIRMED_DOWN", 10
    if gate5 and dir5 > cfg.dir_th_5 and dir10 > -0.10 and trend5 > -0.10 and exh5 < cfg.exh_th_5:
        return "EMERGING_UP", 5
    if gate5 and dir5 < -cfg.dir_th_5 and dir10 < 0.10 and trend5 > -0.10 and exh5 < cfg.exh_th_5:
        return "EMERGING_DOWN", 5
    return "NOISE", 0


def compute_forward_trendness(close_series: pd.Series, horizon: int) -> np.ndarray:
    r1 = np.log(close_series).diff()
    vals = np.full(len(close_series), np.nan, dtype=float)
    arr = r1.to_numpy(dtype=float)
    for i in range(len(arr)):
        if i + horizon >= len(arr):
            continue
        future = arr[i + 1 : i + 1 + horizon]
        if len(future) < horizon or np.isnan(future).any():
            continue
        denom = np.abs(future).sum()
        if denom <= 0:
            continue
        vals[i] = abs(future.sum()) / denom
    return vals


def compute_forward_to_close(open_series: pd.Series, close_series: pd.Series) -> np.ndarray:
    entry_open = open_series.shift(-1)
    eod_close = float(close_series.iloc[-1]) if len(close_series) else np.nan
    vals = np.full(len(close_series), np.nan, dtype=float)
    if not np.isfinite(eod_close) or eod_close <= 0:
        return vals
    entry = entry_open.to_numpy(dtype=float)
    mask = np.isfinite(entry) & (entry > 0)
    vals[mask] = np.log(eod_close / entry[mask])
    return vals


def compute_horizon_features(
    close: np.ndarray,
    volume: np.ndarray,
    money: np.ndarray,
    active_now: np.ndarray,
    log_close: np.ndarray,
    ret1: np.ndarray,
    ret1_df: pd.DataFrame,
    active_df: pd.DataFrame,
    base_weights: np.ndarray,
    sector_matrix: np.ndarray,
    bin_masks: List[np.ndarray],
    leader_mask: np.ndarray,
    etf_close: np.ndarray,
    etf_log_close: np.ndarray,
    etf_r1: np.ndarray,
    minute_index: pd.DatetimeIndex,
    cfg: StrategyConfig,
    horizon: int,
) -> Dict[str, np.ndarray]:
    eps_floor = cfg.eps_floor_5 if horizon == 5 else cfg.eps_floor_10
    trade_recent = (
        pd.DataFrame((volume > 0).astype(float), index=minute_index)
        .rolling(horizon, min_periods=1)
        .sum()
        .to_numpy(dtype=float)
        > 0
    )
    ret_h = log_close - shift_matrix(log_close, horizon)
    eps = np.maximum((1.5 * cfg.tick_size_stock) / np.maximum(close, 1e-6), eps_floor)
    signed = np.zeros_like(ret_h, dtype=float)
    signed[ret_h > eps] = 1.0
    signed[ret_h < -eps] = -1.0

    active_h = active_now & trade_recent & np.isfinite(ret_h)
    active_count = active_h.sum(axis=1).astype(float)
    valid_rows = active_count >= cfg.min_active_constituents
    row_weights = normalize_rows(active_h.astype(float) * base_weights[None, :])
    direction_sign = np.sign((row_weights * signed).sum(axis=1))

    up = np.divide(((signed > 0) & active_h).sum(axis=1), active_count, out=np.full(len(active_count), np.nan), where=active_count > 0)
    down = np.divide(((signed < 0) & active_h).sum(axis=1), active_count, out=np.full(len(active_count), np.nan), where=active_count > 0)
    breadth = np.abs(up - down)

    accel_window = 5 if horizon == 5 else 10
    db = breadth - pd.Series(breadth, index=minute_index).rolling(accel_window, min_periods=1).mean().to_numpy(dtype=float)
    wsb = (row_weights * signed).sum(axis=1)
    direction_sign = np.where(direction_sign == 0.0, np.sign(wsb), direction_sign)

    weighted_ret = row_weights * np.where(np.isfinite(ret_h), ret_h, 0.0)
    weighted_abs = np.abs(weighted_ret)
    total_abs = weighted_abs.sum(axis=1)
    total_sq = np.square(weighted_abs).sum(axis=1)

    ep_raw = np.divide(
        total_abs ** 2,
        active_count * total_sq,
        out=np.full(len(active_count), np.nan),
        where=(active_count > 1) & (total_sq > 0),
    )
    ep = np.divide(
        ep_raw - 1.0 / active_count,
        1.0 - 1.0 / active_count,
        out=np.full(len(active_count), np.nan),
        where=active_count > 1,
    )

    k = min(cfg.topk_k, weighted_abs.shape[1])
    if k > 0:
        topk_sum = np.partition(weighted_abs, weighted_abs.shape[1] - k, axis=1)[:, -k:].sum(axis=1)
        topk = np.divide(topk_sum, total_abs, out=np.full(len(active_count), np.nan), where=total_abs > 0)
    else:
        topk = np.full(len(active_count), np.nan)

    sector_contrib = weighted_abs @ sector_matrix
    sector_signed = weighted_ret @ sector_matrix
    sector_total = sector_contrib.sum(axis=1, keepdims=True)
    sector_prob = np.divide(sector_contrib, sector_total, out=np.zeros_like(sector_contrib), where=sector_total > 0)
    active_sector_n = (sector_contrib > 0).sum(axis=1).astype(float)
    entropy_num = -np.nansum(np.where(sector_prob > 0, sector_prob * np.log(sector_prob), 0.0), axis=1)
    entropy_den = np.where(active_sector_n > 1, np.log(active_sector_n), np.nan)
    se = entropy_num / entropy_den

    money_base_window = max(20, horizon * 6)
    money_h = rolling_sum_matrix(money, horizon)
    money_base_mean = rolling_mean_matrix(money, money_base_window)
    money_expected_h = np.maximum(np.where(np.isfinite(money_base_mean), money_base_mean, 0.0) * horizon, 1.0)
    money_surprise = np.log1p(np.maximum(money_h, 0.0)) - np.log1p(money_expected_h)
    pos_money_surprise = np.maximum(money_surprise, 0.0)

    flow_pressure = (row_weights * signed * pos_money_surprise).sum(axis=1)
    aligned_flow = (signed == direction_sign[:, None]) & active_h & (money_surprise > 0.0)
    flow_breadth_mag = (row_weights * aligned_flow.astype(float)).sum(axis=1)
    flow_breadth = direction_sign * flow_breadth_mag
    flow_accel = direction_sign * (flow_pressure - shift_vector(flow_pressure, 1))

    eps_1m = np.maximum((1.5 * cfg.tick_size_stock) / np.maximum(close, 1e-6), eps_floor * 0.5)
    signed_1m = np.zeros_like(ret1, dtype=float)
    signed_1m[ret1 > eps_1m] = 1.0
    signed_1m[ret1 < -eps_1m] = -1.0
    prev_signed_h = shift_matrix(signed, 1)
    aligned_prev_h = prev_signed_h == direction_sign[:, None]
    aligned_now_h = signed == direction_sign[:, None]
    stock_stick_mag = np.divide(
        (row_weights * (aligned_prev_h & aligned_now_h & active_h).astype(float)).sum(axis=1),
        (row_weights * (aligned_prev_h & active_h).astype(float)).sum(axis=1),
        out=np.full(len(active_count), np.nan),
        where=(row_weights * (aligned_prev_h & active_h).astype(float)).sum(axis=1) > 0,
    )
    sector_sign = np.sign(sector_signed)
    prev_sector_sign = shift_matrix(sector_sign, 1)
    sector_active = np.isfinite(sector_signed) & (np.abs(sector_signed) > 0)
    sector_stick_mag = np.divide(
        ((sector_sign == direction_sign[:, None]) & (prev_sector_sign == direction_sign[:, None]) & sector_active).sum(axis=1),
        ((prev_sector_sign == direction_sign[:, None]) & sector_active).sum(axis=1),
        out=np.full(len(active_count), np.nan),
        where=((prev_sector_sign == direction_sign[:, None]) & sector_active).sum(axis=1) > 0,
    )
    res_stick_mag = 0.6 * np.nan_to_num(stock_stick_mag, nan=0.0) + 0.4 * np.nan_to_num(sector_stick_mag, nan=0.0)
    res_stick = direction_sign * res_stick_mag

    money_1m_expected = np.maximum(np.where(np.isfinite(money_base_mean), money_base_mean, 0.0), 1.0)
    money_1m_surprise = np.log1p(np.maximum(money, 0.0)) - np.log1p(money_1m_expected)
    prev_signed_1m = shift_matrix(signed_1m, 1)
    prev_money_1m_surprise = shift_matrix(money_1m_surprise, 1)
    aligned_prev = prev_signed_1m == direction_sign[:, None]
    sticky_mask = (
        (signed_1m == direction_sign[:, None])
        & aligned_prev
        & (money_1m_surprise > 0.0)
        & (prev_money_1m_surprise > 0.0)
        & active_now
    )
    sticky_denom = (row_weights * aligned_prev.astype(float)).sum(axis=1)
    flow_stick_mag = np.divide(
        (row_weights * sticky_mask.astype(float)).sum(axis=1),
        sticky_denom,
        out=np.full(len(sticky_denom), np.nan),
        where=sticky_denom > 0,
    )
    flow_stick = direction_sign * flow_stick_mag

    corr = compute_corr_fast(ret1_df, active_df, active_count, horizon)

    etf_ret_h = etf_log_close - shift_vector(etf_log_close, horizon)
    etf_abs_path = pd.Series(np.abs(etf_r1), index=minute_index).rolling(horizon, min_periods=horizon).sum().to_numpy(dtype=float)
    dir_strength = np.divide(np.abs(etf_ret_h), etf_abs_path, out=np.full(len(etf_ret_h), np.nan), where=etf_abs_path > 0)

    lead_span = max(2, horizon // 2)
    leader_weights = normalize_rows(active_now.astype(float) * leader_mask[None, :] * base_weights[None, :])
    leader_ret = log_close - shift_matrix(log_close, lead_span)
    leader_prev = shift_vector((leader_weights * np.where(np.isfinite(leader_ret), leader_ret, 0.0)).sum(axis=1), 1)

    follower_mask = (~leader_mask).astype(float)
    follower_weights = normalize_rows(active_now.astype(float) * follower_mask[None, :] * base_weights[None, :])
    follower_ret_now = (follower_weights * np.where(np.isfinite(ret1), ret1, 0.0)).sum(axis=1)
    follower_vol = pd.Series(follower_ret_now, index=minute_index).rolling(20, min_periods=5).std(ddof=0).to_numpy(dtype=float)
    lfr = np.sign(leader_prev) * np.divide(
        follower_ret_now,
        follower_vol,
        out=np.full(len(follower_ret_now), np.nan),
        where=follower_vol > 0,
    )
    follower_flow_now = (follower_weights * signed_1m * np.maximum(money_1m_surprise, 0.0)).sum(axis=1)
    follower_flow_vol = pd.Series(follower_flow_now, index=minute_index).rolling(20, min_periods=5).std(ddof=0).to_numpy(dtype=float)
    crowd_follow = np.sign(leader_prev) * np.divide(
        follower_flow_now,
        follower_flow_vol,
        out=np.full(len(follower_flow_now), np.nan),
        where=follower_flow_vol > 0,
    )

    bin_wsb = np.full((len(minute_index), len(bin_masks)), np.nan, dtype=float)
    for j, mask in enumerate(bin_masks):
        mask_float = mask.astype(float)
        numer = (signed * active_h.astype(float) * mask_float[None, :] * base_weights[None, :]).sum(axis=1)
        denom = (active_h.astype(float) * mask_float[None, :] * base_weights[None, :]).sum(axis=1)
        bin_wsb[:, j] = np.divide(numer, denom, out=np.full(len(denom), np.nan), where=denom > 0)

    x = np.arange(len(bin_masks), dtype=float) - (len(bin_masks) - 1) / 2.0
    wps_slope = np.full(len(minute_index), np.nan, dtype=float)
    for i in range(len(minute_index)):
        y = bin_wsb[i]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        xx = x[mask]
        yy = y[mask]
        xx = xx - xx.mean()
        yy = yy - yy.mean()
        denom = float(np.dot(xx, xx))
        if denom > 0:
            wps_slope[i] = float(np.dot(xx, yy) / denom)
    wps = np.sign(wsb) * wps_slope

    out = {
        f"active_count_{horizon}": active_count,
        f"B_{horizon}_raw": breadth,
        f"dB_{horizon}_raw": db,
        f"Corr_{horizon}_raw": corr,
        f"EP_{horizon}_raw": ep,
        f"TopK_{horizon}_raw": topk,
        f"SE_{horizon}_raw": se,
        f"Dir_{horizon}_raw": dir_strength,
        f"WSB_{horizon}_raw": wsb,
        f"LFR_{horizon}_raw": lfr,
        f"WPS_{horizon}_raw": wps,
        f"FlowPressure_{horizon}_raw": flow_pressure,
        f"FlowAccel_{horizon}_raw": flow_accel,
        f"FlowBreadth_{horizon}_raw": flow_breadth,
        f"FlowStick_{horizon}_raw": flow_stick,
        f"CrowdFollow_{horizon}_raw": crowd_follow,
        f"ResStick_{horizon}_raw": res_stick,
        f"ETFRet_{horizon}_raw": etf_ret_h,
    }
    for key, arr in out.items():
        out[key] = np.where(valid_rows, arr, np.nan)
    return out


def compute_market_context_features(
    close: np.ndarray,
    volume: np.ndarray,
    money: np.ndarray,
    paused: np.ndarray,
    pre_close: np.ndarray,
    high_limit: np.ndarray,
    low_limit: np.ndarray,
    log_close: np.ndarray,
    minute_index: pd.DatetimeIndex,
    mcap_weights: np.ndarray,
    large_mask: np.ndarray,
    sector_matrix: np.ndarray,
    cfg: StrategyConfig,
    horizon: int,
) -> Dict[str, np.ndarray]:
    eps_floor = cfg.eps_floor_5 if horizon == 5 else cfg.eps_floor_10
    active_now = np.isfinite(close) & (paused <= 0.0) & (volume > 0.0) & (money >= cfg.amount_min)
    trade_recent = (
        pd.DataFrame((volume > 0).astype(float), index=minute_index)
        .rolling(horizon, min_periods=1)
        .sum()
        .to_numpy(dtype=float)
        > 0
    )
    ret_h = log_close - shift_matrix(log_close, horizon)
    ret_clean = np.where(np.isfinite(ret_h), ret_h, np.nan)
    eps = np.maximum((1.5 * cfg.tick_size_stock) / np.maximum(close, 1e-6), eps_floor)
    signed = np.zeros_like(ret_h, dtype=float)
    signed[ret_h > eps] = 1.0
    signed[ret_h < -eps] = -1.0

    active_h = active_now & trade_recent & np.isfinite(ret_h)
    active_count = active_h.sum(axis=1).astype(float)
    valid_rows = active_count >= cfg.min_market_active
    row_eq = normalize_rows(active_h.astype(float))

    market_net = (row_eq * signed).sum(axis=1)
    market_b = np.abs(market_net)
    accel_window = 5 if horizon == 5 else 10
    market_db = market_b - pd.Series(market_b, index=minute_index).rolling(accel_window, min_periods=1).mean().to_numpy(dtype=float)
    market_ret = (row_eq * np.where(np.isfinite(ret_h), ret_h, 0.0)).sum(axis=1)
    market_abs = (row_eq * np.abs(np.where(np.isfinite(ret_h), ret_h, 0.0))).sum(axis=1)
    market_coh = np.divide(np.abs(market_ret), market_abs, out=np.full(len(market_ret), np.nan), where=market_abs > 0)
    market_disp = np.nanstd(np.where(active_h, ret_clean, np.nan), axis=1)
    market_sign = np.sign(market_ret)
    market_sign = np.where(market_sign == 0.0, np.sign(market_net), market_sign)

    large_weights = normalize_rows(active_h.astype(float) * large_mask[None, :] * np.maximum(mcap_weights[None, :], 0.0))
    rest_weights = normalize_rows(active_h.astype(float) * (~large_mask)[None, :])
    large_ret = (large_weights * np.where(np.isfinite(ret_h), ret_h, 0.0)).sum(axis=1)
    rest_ret = (rest_weights * np.where(np.isfinite(ret_h), ret_h, 0.0)).sum(axis=1)
    style_lead = market_sign * (large_ret - rest_ret)

    money_base_window = max(20, horizon * 6)
    money_h = rolling_sum_matrix(money, horizon)
    money_base_mean = rolling_mean_matrix(money, money_base_window)
    money_expected_h = np.maximum(np.where(np.isfinite(money_base_mean), money_base_mean, 0.0) * horizon, 1.0)
    money_surprise = np.log1p(np.maximum(money_h, 0.0)) - np.log1p(money_expected_h)
    pos_money_surprise = np.maximum(money_surprise, 0.0)
    market_flow_pressure = (row_eq * signed * pos_money_surprise).sum(axis=1)
    market_flow_breadth_mag = (row_eq * ((signed == market_sign[:, None]) & active_h & (money_surprise > 0.0)).astype(float)).sum(axis=1)
    market_flow_breadth = market_sign * market_flow_breadth_mag
    market_flow_accel = market_sign * (market_flow_pressure - shift_vector(market_flow_pressure, 1))

    sector_ret = (row_eq * np.where(np.isfinite(ret_h), ret_h, 0.0)) @ sector_matrix
    sector_abs = np.abs(sector_ret)
    sector_total = sector_abs.sum(axis=1, keepdims=True)
    sector_prob = np.divide(sector_abs, sector_total, out=np.zeros_like(sector_abs), where=sector_total > 0)
    active_sector_n = (sector_abs > 0).sum(axis=1).astype(float)
    sector_entropy_num = -np.nansum(np.where(sector_prob > 0, sector_prob * np.log(sector_prob), 0.0), axis=1)
    sector_entropy_den = np.where(active_sector_n > 1, np.log(active_sector_n), np.nan)
    market_se = sector_entropy_num / sector_entropy_den

    sector_sign = np.sign(sector_ret)
    sector_confirm = np.divide(
        (sector_sign * market_sign[:, None]).sum(axis=1),
        active_sector_n,
        out=np.full(len(active_sector_n), np.nan),
        where=active_sector_n > 0,
    )

    near_high = active_now & np.isfinite(high_limit) & (close >= high_limit * (1.0 - cfg.market_limit_band))
    near_low = active_now & np.isfinite(low_limit) & (close <= low_limit * (1.0 + cfg.market_limit_band))
    active_now_count = active_now.sum(axis=1).astype(float)
    limit_imb = np.divide(
        near_high.sum(axis=1) - near_low.sum(axis=1),
        active_now_count,
        out=np.full(len(active_now_count), np.nan),
        where=active_now_count > 0,
    )

    up_from_pre = np.divide(close, np.where(pre_close > 0, pre_close, np.nan)) - 1.0
    surge_up = active_now & np.isfinite(up_from_pre) & (up_from_pre >= 0.04)
    surge_down = active_now & np.isfinite(up_from_pre) & (up_from_pre <= -0.04)
    surge_imb = np.divide(
        surge_up.sum(axis=1) - surge_down.sum(axis=1),
        active_now_count,
        out=np.full(len(active_now_count), np.nan),
        where=active_now_count > 0,
    )

    out = {
        f"MKTNet_{horizon}_raw": market_net,
        f"MKTB_{horizon}_raw": market_b,
        f"MKTdB_{horizon}_raw": market_db,
        f"MKTRet_{horizon}_raw": market_ret,
        f"MKTDisp_{horizon}_raw": market_disp,
        f"MKTCoh_{horizon}_raw": market_coh,
        f"MKTSE_{horizon}_raw": market_se,
        f"MKTLimitImb_{horizon}_raw": limit_imb,
        f"MKTSurgeImb_{horizon}_raw": surge_imb,
        f"MKTLargeRet_{horizon}_raw": large_ret,
        f"MKTStyleLead_{horizon}_raw": style_lead,
        f"MKTSectorConfirm_{horizon}_raw": sector_confirm,
        f"MKTFlowPressure_{horizon}_raw": market_flow_pressure,
        f"MKTFlowAccel_{horizon}_raw": market_flow_accel,
        f"MKTFlowBreadth_{horizon}_raw": market_flow_breadth,
        f"market_active_count_{horizon}": active_count,
    }
    for key, arr in out.items():
        out[key] = np.where(valid_rows, arr, np.nan)
    return out


def compute_day_features(
    date_str: str,
    static_df: pd.DataFrame,
    cfg: StrategyConfig,
    mcap_panel: pd.DataFrame,
    industry_df: pd.DataFrame,
) -> pd.DataFrame | None:
    stock_path = Path(cfg.stock_data_dir) / f"{date_str}.parquet"
    etf_path = Path(cfg.etf_data_dir) / f"{date_str}.parquet"

    stock_df = read_stock_day(
        stock_path,
        columns=["code", "datetime", "close", "volume", "money", "paused", "pre_close", "high_limit", "low_limit"],
    )
    etf_df = read_etf_day(etf_path, cfg.target_etf_code)
    if stock_df.empty or etf_df.empty:
        return None

    close_df_all = stock_df.pivot_table(index="datetime", columns="code", values="close")
    volume_df_all = stock_df.pivot_table(index="datetime", columns="code", values="volume")
    money_df_all = stock_df.pivot_table(index="datetime", columns="code", values="money")
    paused_df_all = stock_df.pivot_table(index="datetime", columns="code", values="paused")
    pre_close_df_all = stock_df.pivot_table(index="datetime", columns="code", values="pre_close")
    high_limit_df_all = stock_df.pivot_table(index="datetime", columns="code", values="high_limit")
    low_limit_df_all = stock_df.pivot_table(index="datetime", columns="code", values="low_limit")

    common_cols = [c for c in static_df["code"] if c in close_df_all.columns]
    if len(common_cols) < cfg.min_active_constituents:
        return None

    etf_df = etf_df.set_index("datetime").sort_index()
    common_times = close_df_all.index.intersection(etf_df.index)
    if len(common_times) < 120:
        return None

    close_df_all = close_df_all.reindex(index=common_times)
    volume_df_all = volume_df_all.reindex(index=common_times).fillna(0.0)
    money_df_all = money_df_all.reindex(index=common_times).fillna(0.0)
    paused_df_all = paused_df_all.reindex(index=common_times).fillna(0.0)
    pre_close_df_all = pre_close_df_all.reindex(index=common_times)
    high_limit_df_all = high_limit_df_all.reindex(index=common_times)
    low_limit_df_all = low_limit_df_all.reindex(index=common_times)
    etf_df = etf_df.reindex(common_times)

    close_df = close_df_all.reindex(columns=common_cols)
    volume_df = volume_df_all.reindex(columns=common_cols)
    money_df = money_df_all.reindex(columns=common_cols)
    paused_df = paused_df_all.reindex(columns=common_cols).fillna(0.0)

    static_day = static_df.set_index("code").reindex(common_cols).reset_index()
    base_weights = static_day["weight"].to_numpy(dtype=float)
    leader_mask = static_day["leader_flag"].to_numpy(dtype=int).astype(bool)
    industries = static_day["industry"].fillna("Unknown").astype(str)
    industry_codes = industries.astype("category").cat.codes.to_numpy(dtype=int)
    n_sectors = int(industry_codes.max()) + 1 if len(industry_codes) else 0
    sector_matrix = np.zeros((len(common_cols), max(n_sectors, 1)), dtype=float)
    sector_matrix[np.arange(len(common_cols)), industry_codes] = 1.0

    bin_masks: List[np.ndarray] = []
    for k in range(static_day["weight_bin"].max() + 1):
        bin_masks.append((static_day["weight_bin"].to_numpy(dtype=int) == k))

    close = close_df.to_numpy(dtype=float)
    volume = volume_df.fillna(0.0).to_numpy(dtype=float)
    money = money_df.fillna(0.0).to_numpy(dtype=float)
    paused = paused_df.fillna(0.0).to_numpy(dtype=float)
    active_now = np.isfinite(close) & (paused <= 0.0) & (volume > 0.0) & (money >= cfg.amount_min)
    log_close = np.log(close)
    ret1 = log_close - shift_matrix(log_close, 1)
    ret1_df = pd.DataFrame(ret1, index=common_times, columns=common_cols)
    active_df = pd.DataFrame(active_now, index=common_times, columns=common_cols)

    etf_open = etf_df["open"].to_numpy(dtype=float)
    etf_close = etf_df["close"].to_numpy(dtype=float)
    etf_log_close = np.log(etf_close)
    etf_r1 = etf_log_close - shift_vector(etf_log_close, 1)

    market_cols = close_df_all.columns.tolist()
    market_meta = build_market_meta_day(date_str, market_cols, mcap_panel, industry_df, cfg).set_index("code").reindex(market_cols).reset_index()
    market_ind_codes = market_meta["industry"].fillna("Unknown").astype("category").cat.codes.to_numpy(dtype=int)
    n_market_sectors = int(market_ind_codes.max()) + 1 if len(market_ind_codes) else 0
    market_sector_matrix = np.zeros((len(market_cols), max(n_market_sectors, 1)), dtype=float)
    market_sector_matrix[np.arange(len(market_cols)), market_ind_codes] = 1.0
    market_mcap_weights = market_meta["mcap_weight"].fillna(0.0).to_numpy(dtype=float)
    market_large_mask = market_meta["large_flag"].fillna(0).to_numpy(dtype=int).astype(bool)

    close_all = close_df_all.to_numpy(dtype=float)
    volume_all = volume_df_all.to_numpy(dtype=float)
    money_all = money_df_all.to_numpy(dtype=float)
    paused_all = paused_df_all.to_numpy(dtype=float)
    pre_close_all = pre_close_df_all.to_numpy(dtype=float)
    high_limit_all = high_limit_df_all.to_numpy(dtype=float)
    low_limit_all = low_limit_df_all.to_numpy(dtype=float)
    log_close_all = np.log(close_all)
    market_active_now = np.isfinite(close_all) & (paused_all <= 0.0) & (volume_all > 0.0) & (money_all >= cfg.amount_min)

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(date_str),
            "datetime": common_times,
            "minute_idx": np.arange(len(common_times), dtype=int),
            "etf_code": cfg.target_etf_code,
            "etf_open": etf_open,
            "etf_close": etf_close,
            "hs300_available": len(common_cols),
            "hs300_active_now": active_now.sum(axis=1).astype(int),
            "market_available": len(market_cols),
            "market_active_now": market_active_now.sum(axis=1).astype(int),
        }
    )

    for horizon in HORIZONS:
        feats = compute_horizon_features(
            close=close,
            volume=volume,
            money=money,
            active_now=active_now,
            log_close=log_close,
            ret1=ret1,
            ret1_df=ret1_df,
            active_df=active_df,
            base_weights=base_weights,
            sector_matrix=sector_matrix,
            bin_masks=bin_masks,
            leader_mask=leader_mask,
            etf_close=etf_close,
            etf_log_close=etf_log_close,
            etf_r1=etf_r1,
            minute_index=common_times,
            cfg=cfg,
            horizon=horizon,
        )
        for k, v in feats.items():
            panel[k] = v

        market_feats = compute_market_context_features(
            close=close_all,
            volume=volume_all,
            money=money_all,
            paused=paused_all,
            pre_close=pre_close_all,
            high_limit=high_limit_all,
            low_limit=low_limit_all,
            log_close=log_close_all,
            minute_index=common_times,
            mcap_weights=market_mcap_weights,
            large_mask=market_large_mask,
            sector_matrix=market_sector_matrix,
            cfg=cfg,
            horizon=horizon,
        )
        for k, v in market_feats.items():
            panel[k] = v

    entry_open = panel["etf_open"].shift(-1)
    for horizon in DIAG_HORIZONS:
        exit_open = panel["etf_open"].shift(-(horizon + 1))
        panel[f"fwd_open_ret_{horizon}"] = np.log(exit_open / entry_open)
        panel[f"trendness_{horizon}"] = compute_forward_trendness(panel["etf_close"], horizon)
    panel["fwd_eod_ret"] = compute_forward_to_close(panel["etf_open"], panel["etf_close"])

    return panel


def bucket_robust_zscore(panel: pd.DataFrame, raw_cols: List[str], cfg: StrategyConfig) -> pd.DataFrame:
    out = panel.sort_values(["date", "minute_idx"]).reset_index(drop=True).copy()
    for col in raw_cols:
        z = np.full(len(out), np.nan, dtype=float)
        hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.z_window_days))
        vals = out[col].to_numpy(dtype=float)
        buckets = out["minute_idx"].to_numpy(dtype=int)
        for i, val in enumerate(vals):
            bucket = int(buckets[i])
            history = hist[bucket]
            if np.isfinite(val):
                arr = np.asarray(history, dtype=float)
                if len(arr) >= cfg.z_min_history:
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med)))
                    scale = 1.4826 * mad
                    if scale < 1e-8:
                        std = float(arr.std(ddof=0))
                        scale = std if std > 1e-8 else 1.0
                    z[i] = (val - med) / scale
                elif len(arr) >= 2:
                    mu = float(arr.mean())
                    sd = float(arr.std(ddof=0))
                    z[i] = (val - mu) / (sd if sd > 1e-8 else 1.0)
                else:
                    z[i] = 0.0
                history.append(float(val))
        out[f"z_{col}"] = z
    return out


def score_panel(panel: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    raw_cols: List[str] = []
    for horizon in HORIZONS:
        raw_cols.extend(
            [
                f"B_{horizon}_raw",
                f"dB_{horizon}_raw",
                f"Corr_{horizon}_raw",
                f"EP_{horizon}_raw",
                f"TopK_{horizon}_raw",
                f"SE_{horizon}_raw",
                f"Dir_{horizon}_raw",
                f"WSB_{horizon}_raw",
                f"LFR_{horizon}_raw",
                f"WPS_{horizon}_raw",
                f"FlowPressure_{horizon}_raw",
                f"FlowAccel_{horizon}_raw",
                f"FlowBreadth_{horizon}_raw",
                f"FlowStick_{horizon}_raw",
                f"CrowdFollow_{horizon}_raw",
                f"ResStick_{horizon}_raw",
                f"ETFRet_{horizon}_raw",
                f"MKTNet_{horizon}_raw",
                f"MKTB_{horizon}_raw",
                f"MKTdB_{horizon}_raw",
                f"MKTRet_{horizon}_raw",
                f"MKTDisp_{horizon}_raw",
                f"MKTCoh_{horizon}_raw",
                f"MKTSE_{horizon}_raw",
                f"MKTLimitImb_{horizon}_raw",
                f"MKTSurgeImb_{horizon}_raw",
                f"MKTLargeRet_{horizon}_raw",
                f"MKTStyleLead_{horizon}_raw",
                f"MKTSectorConfirm_{horizon}_raw",
                f"MKTFlowPressure_{horizon}_raw",
                f"MKTFlowAccel_{horizon}_raw",
                f"MKTFlowBreadth_{horizon}_raw",
            ]
        )
    scored = bucket_robust_zscore(panel, raw_cols, cfg)

    for horizon in HORIZONS:
        z_wsb = scored[f"z_WSB_{horizon}_raw"].fillna(0.0)
        z_db = scored[f"z_dB_{horizon}_raw"].fillna(0.0)
        z_corr = scored[f"z_Corr_{horizon}_raw"].fillna(0.0)
        z_ep = scored[f"z_EP_{horizon}_raw"].fillna(0.0)
        z_se = scored[f"z_SE_{horizon}_raw"].fillna(0.0)
        z_etf_ret = scored[f"z_ETFRet_{horizon}_raw"].fillna(0.0)
        z_b = scored[f"z_B_{horizon}_raw"].fillna(0.0)
        z_dir = scored[f"z_Dir_{horizon}_raw"].fillna(0.0)
        z_topk = scored[f"z_TopK_{horizon}_raw"].fillna(0.0)
        z_lfr = scored[f"z_LFR_{horizon}_raw"].fillna(0.0)
        z_wps = scored[f"z_WPS_{horizon}_raw"].fillna(0.0)
        z_flow_pressure = scored[f"z_FlowPressure_{horizon}_raw"].fillna(0.0)
        z_flow_accel = scored[f"z_FlowAccel_{horizon}_raw"].fillna(0.0)
        z_flow_breadth = scored[f"z_FlowBreadth_{horizon}_raw"].fillna(0.0)
        z_flow_stick = scored[f"z_FlowStick_{horizon}_raw"].fillna(0.0)
        z_crowd_follow = scored[f"z_CrowdFollow_{horizon}_raw"].fillna(0.0)
        z_res_stick = scored[f"z_ResStick_{horizon}_raw"].fillna(0.0)
        z_mkt_net = scored[f"z_MKTNet_{horizon}_raw"].fillna(0.0)
        z_mkt_b = scored[f"z_MKTB_{horizon}_raw"].fillna(0.0)
        z_mkt_db = scored[f"z_MKTdB_{horizon}_raw"].fillna(0.0)
        z_mkt_ret = scored[f"z_MKTRet_{horizon}_raw"].fillna(0.0)
        z_mkt_disp = scored[f"z_MKTDisp_{horizon}_raw"].fillna(0.0)
        z_mkt_coh = scored[f"z_MKTCoh_{horizon}_raw"].fillna(0.0)
        z_mkt_se = scored[f"z_MKTSE_{horizon}_raw"].fillna(0.0)
        z_mkt_limit = scored[f"z_MKTLimitImb_{horizon}_raw"].fillna(0.0)
        z_mkt_surge = scored[f"z_MKTSurgeImb_{horizon}_raw"].fillna(0.0)
        z_mkt_large = scored[f"z_MKTLargeRet_{horizon}_raw"].fillna(0.0)
        z_mkt_style = scored[f"z_MKTStyleLead_{horizon}_raw"].fillna(0.0)
        z_mkt_sector = scored[f"z_MKTSectorConfirm_{horizon}_raw"].fillna(0.0)
        z_mkt_flow_pressure = scored[f"z_MKTFlowPressure_{horizon}_raw"].fillna(0.0)
        z_mkt_flow_accel = scored[f"z_MKTFlowAccel_{horizon}_raw"].fillna(0.0)
        z_mkt_flow_breadth = scored[f"z_MKTFlowBreadth_{horizon}_raw"].fillna(0.0)

        pressure_sign = np.sign(z_wsb)
        pressure_sign = pressure_sign.mask(pressure_sign == 0.0, np.sign(z_lfr)).fillna(0.0)
        aligned_strength = 0.18 * z_db + 0.18 * z_corr + 0.14 * z_ep + 0.14 * z_se + 0.10 * z_flow_breadth
        internal = 0.26 * z_wsb + pressure_sign * aligned_strength + 0.12 * z_flow_pressure + 0.08 * z_crowd_follow
        ipg = internal - z_etf_ret
        pad = pressure_sign * z_etf_ret - np.abs(internal)
        market_pressure_sign = np.sign(z_mkt_ret)
        market_pressure_sign = market_pressure_sign.mask(market_pressure_sign == 0.0, np.sign(z_mkt_net)).fillna(0.0)
        market_aligned_strength = 0.12 * z_mkt_db + 0.10 * (-z_mkt_disp) + 0.10 * z_mkt_coh + 0.08 * z_mkt_se + 0.08 * z_mkt_style + 0.08 * z_mkt_sector + 0.08 * z_mkt_flow_breadth
        market_pressure = 0.18 * z_mkt_net + 0.14 * z_mkt_ret + 0.08 * z_mkt_large + 0.06 * z_mkt_flow_pressure + 0.04 * z_mkt_limit + 0.04 * z_mkt_surge
        market_pressure = market_pressure + market_pressure_sign * market_aligned_strength
        market_ipg = market_pressure - z_etf_ret
        context_gate = (
            0.14 * z_mkt_b
            + 0.10 * z_mkt_db
            + 0.08 * (-z_mkt_disp)
            + 0.08 * z_mkt_coh
            + 0.08 * z_mkt_se
            + 0.04 * np.abs(z_mkt_limit)
            + 0.04 * np.abs(z_mkt_surge)
            + 0.04 * np.maximum(z_mkt_style, 0.0)
            + 0.04 * np.abs(z_mkt_flow_pressure)
        )
        behavior_pressure = (
            0.32 * z_flow_pressure
            + 0.16 * z_flow_accel
            + 0.14 * z_flow_breadth
            + 0.12 * z_flow_stick
            + 0.10 * z_crowd_follow
            + 0.10 * z_mkt_flow_pressure
            + 0.06 * z_mkt_flow_accel
        )

        internal_delta = internal - internal.groupby(scored["date"]).shift(1)
        prev_internal_delta = internal_delta.groupby(scored["date"]).shift(1)
        res_vel = pressure_sign * internal_delta.fillna(0.0)
        res_acc = pressure_sign * (internal_delta - prev_internal_delta).fillna(0.0)
        diff_acc = pressure_sign * (
            0.30 * (z_b - z_b.groupby(scored["date"]).shift(1).fillna(0.0))
            + 0.20 * (z_ep - z_ep.groupby(scored["date"]).shift(1).fillna(0.0))
            + 0.20 * (z_se - z_se.groupby(scored["date"]).shift(1).fillna(0.0))
            + 0.15 * (z_flow_breadth - z_flow_breadth.groupby(scored["date"]).shift(1).fillna(0.0))
            + 0.15 * (z_flow_pressure - z_flow_pressure.groupby(scored["date"]).shift(1).fillna(0.0))
        )

        gate = (
            0.16 * z_b
            + 0.14 * z_db
            + 0.14 * z_corr
            + 0.10 * z_ep
            + 0.08 * z_se
            + 0.06 * z_dir
            + 0.05 * (-z_topk)
            + 0.05 * np.abs(z_flow_pressure)
            + 0.04 * np.abs(z_flow_breadth)
            + context_gate
        )
        exhaustion = (
            0.38 * np.maximum(pad, 0.0)
            + 0.18 * np.maximum(z_topk, 0.0)
            + 0.12 * np.maximum(-z_db, 0.0)
            + 0.12 * np.maximum(-market_ipg, 0.0)
            + 0.10 * np.maximum(z_mkt_disp, 0.0)
            + 0.10 * np.maximum(np.abs(z_flow_pressure) - np.abs(z_flow_breadth), 0.0)
        )
        crowd_inertia = (
            0.24 * res_vel
            + 0.16 * res_acc
            + 0.16 * z_res_stick
            + 0.14 * diff_acc
            + 0.12 * z_crowd_follow
            + 0.10 * z_flow_accel
            + 0.08 * z_flow_breadth
            - 0.18 * exhaustion
        )
        healthy_inertia = (
            0.22 * res_vel
            + 0.16 * res_acc
            + 0.12 * z_res_stick
            + 0.10 * z_flow_stick
            + 0.14 * diff_acc
            + 0.18 * behavior_pressure
            + 0.10 * z_crowd_follow
            - 0.18 * exhaustion
        )
        direction_core = 0.28 * ipg + 0.18 * z_lfr + 0.12 * z_wps + 0.08 * z_wsb + 0.12 * behavior_pressure + 0.10 * res_vel + 0.08 * res_acc + 0.08 * crowd_inertia
        direction = direction_core + 0.10 * market_ipg + 0.04 * z_mkt_ret + 0.02 * z_mkt_limit + 0.08 * healthy_inertia
        dir_sign = np.sign(direction).mask(np.sign(direction) == 0.0, pressure_sign).fillna(0.0)
        trend_support = (
            0.28 * gate
            + 0.28 * np.abs(direction)
            + 0.18 * np.maximum(dir_sign * market_pressure, 0.0)
            + 0.22 * np.maximum(dir_sign * healthy_inertia, 0.0)
            - 0.18 * exhaustion
        )

        scored[f"InternalPressure_{horizon}"] = internal
        scored[f"WSB_{horizon}"] = z_wsb
        scored[f"FlowPressure_{horizon}"] = z_flow_pressure
        scored[f"FlowAccel_{horizon}"] = z_flow_accel
        scored[f"FlowBreadth_{horizon}"] = z_flow_breadth
        scored[f"FlowStick_{horizon}"] = z_flow_stick
        scored[f"CrowdFollow_{horizon}"] = z_crowd_follow
        scored[f"ResStick_{horizon}"] = z_res_stick
        scored[f"MKTFlowPressure_{horizon}"] = z_mkt_flow_pressure
        scored[f"MKTFlowAccel_{horizon}"] = z_mkt_flow_accel
        scored[f"MKTFlowBreadth_{horizon}"] = z_mkt_flow_breadth
        scored[f"ResVel_{horizon}"] = res_vel
        scored[f"ResAcc_{horizon}"] = res_acc
        scored[f"DiffAcc_{horizon}"] = diff_acc
        scored[f"BehavioralPressure_{horizon}"] = behavior_pressure
        scored[f"CrowdInertia_{horizon}"] = crowd_inertia
        scored[f"HealthyInertia_{horizon}"] = healthy_inertia
        scored[f"FollowThrough_{horizon}"] = (
            0.45 * res_vel
            + 0.25 * z_crowd_follow
            + 0.15 * z_flow_accel
            + 0.15 * res_acc
            - 0.25 * z_flow_pressure
        )
        scored[f"TrendContinuation_{horizon}"] = (
            0.40 * res_vel
            + 0.20 * z_crowd_follow
            + 0.15 * res_acc
            + 0.15 * diff_acc
            - 0.15 * z_flow_pressure
            - 0.15 * exhaustion
        )
        scored[f"CrowdFollowLowExh_{horizon}"] = (
            0.50 * res_vel
            + 0.30 * z_crowd_follow
            + 0.20 * z_flow_accel
            - 0.20 * exhaustion
            - 0.15 * z_flow_pressure
        )
        scored[f"CascadeBalance_{horizon}"] = (
            0.30 * res_vel
            + 0.20 * res_acc
            + 0.20 * z_crowd_follow
            + 0.10 * z_flow_accel
            - 0.10 * z_flow_pressure
            - 0.10 * exhaustion
        )
        scored[f"IPG_{horizon}"] = ipg
        scored[f"PAD_{horizon}"] = pad
        scored[f"MarketPressure_{horizon}"] = market_pressure
        scored[f"MarketIPG_{horizon}"] = market_ipg
        scored[f"TrendSupport_{horizon}"] = trend_support
        scored[f"GateScore_{horizon}"] = gate
        scored[f"DirScore_{horizon}"] = direction
        scored[f"Exhaustion_{horizon}"] = exhaustion

    # Session-level trend-day features answer a different question than continuation:
    # is the market evolving into a half-day / full-day one-sided regime?
    date_key = scored["date"]
    etf_log_close = np.log(scored["etf_close"].replace(0.0, np.nan))
    etf_r1 = etf_log_close.groupby(date_key).diff()
    first_open = scored.groupby("date", sort=False)["etf_open"].transform("first").replace(0.0, np.nan)
    open_ret = np.log(scored["etf_close"] / first_open)
    open_path_abs = etf_r1.abs().groupby(date_key).cumsum()
    day_path_eff = np.divide(
        open_ret,
        open_path_abs,
        out=np.full(len(scored), np.nan, dtype=float),
        where=open_path_abs.to_numpy(dtype=float) > 0,
    )

    day_dir = pd.Series(np.sign(scored["TrendContinuation_10"]), index=scored.index, dtype=float)
    day_dir = day_dir.where(day_dir != 0.0, np.sign(scored["DirScore_10"]))
    day_dir = day_dir.where(day_dir != 0.0, np.sign(scored["InternalPressure_10"]))
    day_dir = day_dir.fillna(0.0)

    aligned_internal = day_dir * scored["InternalPressure_10"]
    aligned_market = day_dir * scored["MarketPressure_10"]
    aligned_wsb = day_dir * scored["WSB_10"]
    aligned_follow = day_dir * scored["FollowThrough_10"]
    aligned_res_vel = day_dir * scored["ResVel_10"]
    aligned_path_eff = day_dir * pd.Series(day_path_eff, index=scored.index).fillna(0.0)

    day_internal_mean = aligned_internal.groupby(date_key).expanding().mean().reset_index(level=0, drop=True).fillna(0.0)
    day_market_mean = aligned_market.groupby(date_key).expanding().mean().reset_index(level=0, drop=True).fillna(0.0)
    day_wsb_mean = aligned_wsb.groupby(date_key).expanding().mean().reset_index(level=0, drop=True).fillna(0.0)
    day_follow_mean = aligned_follow.groupby(date_key).expanding().mean().reset_index(level=0, drop=True).fillna(0.0)
    day_res_vel_mean = aligned_res_vel.groupby(date_key).expanding().mean().reset_index(level=0, drop=True).fillna(0.0)

    dir_shift = day_dir.groupby(date_key).shift(1)
    stable_same_dir = ((day_dir == dir_shift) & (day_dir != 0.0)).astype(float)
    direction_stability = (
        stable_same_dir.groupby(date_key)
        .rolling(10, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )
    direction_stability = 2.0 * direction_stability - 1.0

    day_gap = day_internal_mean - aligned_path_eff
    trend_day = (
        0.20 * scored["GateScore_10"].fillna(0.0)
        + 0.16 * scored["DirScore_10"].abs().fillna(0.0)
        + 0.16 * day_internal_mean
        + 0.12 * day_market_mean
        + 0.10 * aligned_path_eff
        + 0.08 * day_wsb_mean
        + 0.08 * day_follow_mean
        + 0.10 * direction_stability
        - 0.12 * scored["Exhaustion_10"].fillna(0.0)
    )
    day_cont = (
        0.28 * scored["TrendContinuation_10"].fillna(0.0)
        + 0.18 * scored["FollowThrough_10"].fillna(0.0)
        + 0.14 * day_internal_mean
        + 0.12 * day_market_mean
        + 0.10 * aligned_path_eff
        + 0.08 * day_gap
        + 0.06 * direction_stability
        + 0.06 * day_res_vel_mean
        - 0.12 * scored["Exhaustion_10"].fillna(0.0)
    )

    scored["DayOpenRet"] = open_ret
    scored["DayPathEff"] = aligned_path_eff
    scored["DayInternalMean_10"] = day_internal_mean
    scored["DayMarketMean_10"] = day_market_mean
    scored["DayBreadthMean_10"] = day_wsb_mean
    scored["DayFollowMean_10"] = day_follow_mean
    scored["DayDirStability_10"] = direction_stability
    scored["DayGap_10"] = day_gap
    scored["TrendDayScore_10"] = trend_day
    scored["DayContinuation_10"] = day_cont

    scored["Gate_5"] = scored["GateScore_5"] > cfg.gate_enter_5
    scored["Gate_10"] = scored["GateScore_10"] > cfg.gate_enter_10
    scored["Keep_5"] = scored["GateScore_5"] > cfg.gate_keep_5
    scored["Keep_10"] = scored["GateScore_10"] > cfg.gate_keep_10

    scored["signal_side"] = np.where(
        scored["DirScore_10"].abs() >= scored["DirScore_5"].abs(),
        np.sign(scored["DirScore_10"]),
        np.sign(scored["DirScore_5"]),
    ).astype(int)

    states: List[str] = []
    preferred_h: List[int] = []
    for _, row in scored.iterrows():
        state, horizon = compute_state(row, cfg)
        states.append(state)
        preferred_h.append(horizon)
    scored["state"] = states
    scored["preferred_horizon"] = preferred_h
    return scored


def choose_confirmed_trade_holds(row: pd.Series, side: int, cfg: StrategyConfig) -> Tuple[int, int, str]:
    trend_support = float(row.get("TrendSupport_10", 0.0))
    follow_through = float(row.get("FollowThrough_10", 0.0))
    trend_cont = float(row.get("TrendContinuation_10", 0.0))
    day_cont = float(row.get("DayContinuation_10", 0.0))
    trend_day = float(row.get("TrendDayScore_10", 0.0))
    if side > 0 and trend_day >= cfg.trade_trend_day_long_10 and day_cont >= cfg.trade_day_cont_long_10:
        return cfg.trade_min_hold_long_trendday, cfg.trade_hold_long_trendday, "TRENDDAY"
    if side < 0 and trend_day >= cfg.trade_trend_day_short_10 and day_cont <= -cfg.trade_day_cont_short_10:
        return cfg.trade_min_hold_short_trendday, cfg.trade_hold_short_trendday, "TRENDDAY"
    if side > 0 and follow_through >= cfg.trade_followthrough_long_10 and trend_cont >= cfg.trade_trend_cont_long_10:
        return cfg.trade_min_hold_long_combo, cfg.trade_hold_long_combo, "COMBO"
    if trend_support >= cfg.trade_strong_trend_support_10:
        if side > 0:
            return cfg.trade_min_hold_long_strong, cfg.trade_hold_long_strong, "STRONG"
        return cfg.trade_min_hold_short_strong, cfg.trade_hold_short_strong, "STRONG"
    if side > 0:
        return cfg.trade_min_hold_long_confirmed, cfg.trade_hold_long_confirmed, "BASE"
    return cfg.trade_min_hold_short_confirmed, cfg.trade_hold_short_confirmed, "BASE"


def backtest_etf(scored: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trades: List[dict] = []
    daily_rows: List[dict] = []
    cost = cfg.round_trip_cost_bps / 10000.0

    for date, day in scored.groupby("date", sort=True):
        day = day.sort_values("datetime").reset_index(drop=True).copy()

        if cfg.trade_confirmed_only:
            day_pnl = 0.0
            position = 0
            entry_bar = -1
            entry_px = np.nan
            entry_state = ""
            planned_hold = 0
            min_hold = 0
            hold_profile = ""
            pending: dict | None = None
            trades_today = 0
            next_ok_minute = -1
            for bar in range(len(day)):
                open_px = float(day.loc[bar, "etf_open"])
                minute_idx = int(day.loc[bar, "minute_idx"])

                if pending is not None:
                    if pending["type"] == "EXIT" and position != 0:
                        gross = position * (open_px / entry_px - 1.0)
                        net = gross - cost
                        day_pnl += net
                        trades.append(
                            {
                                "date": date,
                                "entry_time": day.loc[entry_bar, "datetime"],
                                "exit_time": day.loc[bar, "datetime"],
                                "entry_state": entry_state,
                                "exit_state": pending.get("reason", day.loc[bar, "state"]),
                                "direction": "LONG" if position > 0 else "SHORT",
                                "planned_hold": planned_hold,
                                "bars_held": bar - entry_bar,
                                "entry_px": entry_px,
                                "exit_px": open_px,
                                "hold_profile": hold_profile,
                                "gross_bps": gross * 10000.0,
                                "net_bps": net * 10000.0,
                            }
                        )
                        position = 0
                        entry_bar = -1
                        entry_px = np.nan
                        entry_state = ""
                        planned_hold = 0
                        min_hold = 0
                        hold_profile = ""
                        trades_today += 1
                        next_ok_minute = minute_idx + cfg.trade_cooldown_bars
                    elif pending["type"] == "ENTER" and position == 0:
                        position = pending["side"]
                        entry_bar = bar
                        entry_px = open_px
                        entry_state = pending["state"]
                        planned_hold = pending["hold"]
                        min_hold = pending["min_hold"]
                        hold_profile = pending["hold_profile"]
                    pending = None

                if bar >= len(day) - 1:
                    break

                minute_idx = int(day.loc[bar, "minute_idx"])
                state = day.loc[bar, "state"]
                dir10 = float(day.loc[bar, "DirScore_10"])
                exh10 = float(day.loc[bar, "Exhaustion_10"])
                trend10 = float(day.loc[bar, "TrendSupport_10"])
                keep10 = bool(day.loc[bar, "Keep_10"])
                market10 = float(day.loc[bar, "MarketPressure_10"])
                follow10 = float(day.loc[bar, "FollowThrough_10"])
                trend_cont10 = float(day.loc[bar, "TrendContinuation_10"])
                day_cont10 = float(day.loc[bar, "DayContinuation_10"])
                trend_day10 = float(day.loc[bar, "TrendDayScore_10"])
                side = 0
                if (
                    cfg.trade_enable_long
                    and
                    state == "CONFIRMED_UP"
                    and trend10 >= cfg.trade_entry_trend_support_10
                    and market10 > -0.25
                    and exh10 <= cfg.trade_exhaustion_max_10
                    and (
                        trend_cont10 >= cfg.trade_trend_cont_long_10
                        or (trend_day10 >= cfg.trade_trend_day_long_10 and day_cont10 >= cfg.trade_day_cont_long_10)
                    )
                ):
                    side = 1
                elif (
                    cfg.trade_enable_short
                    and
                    state == "CONFIRMED_DOWN"
                    and dir10 <= cfg.trade_dir_short_10
                    and trend10 >= cfg.trade_entry_trend_support_10
                    and market10 < 0.25
                    and exh10 <= cfg.trade_exhaustion_max_10
                    and (day_cont10 <= -cfg.trade_day_cont_short_10 or trend_day10 >= cfg.trade_trend_day_short_10)
                ):
                    side = -1

                if position == 0:
                    if minute_idx < next_ok_minute or trades_today >= cfg.trade_max_per_day or side == 0:
                        continue
                    min_hold_sel, max_hold_sel, hold_profile_sel = choose_confirmed_trade_holds(day.loc[bar], side, cfg)
                    if bar + max_hold_sel + 1 >= len(day):
                        continue
                    pending = {
                        "type": "ENTER",
                        "side": side,
                        "state": state,
                        "hold": max_hold_sel,
                        "min_hold": min_hold_sel,
                        "hold_profile": hold_profile_sel,
                    }
                    continue

                held = bar - entry_bar
                exit_reason = ""
                if hold_profile in {"COMBO", "TRENDDAY"}:
                    if held >= planned_hold:
                        exit_reason = f"MAX_HOLD_{planned_hold}"
                    elif held >= min_hold and side == -position:
                        exit_reason = "OPPOSITE_CONFIRMED"
                    elif held >= min_hold and state == "EXHAUSTED" and exh10 > max(cfg.exh_th_10, 1.20):
                        exit_reason = "SEVERE_EXHAUSTED"
                    elif hold_profile == "TRENDDAY" and held >= min_hold and trend_day10 < 0.0:
                        exit_reason = "DAY_TREND_FADE"
                else:
                    if held >= min_hold and state == "EXHAUSTED":
                        exit_reason = "EXHAUSTED"
                    elif held >= min_hold and side == -position:
                        exit_reason = "OPPOSITE_CONFIRMED"
                    elif held >= min_hold and (not keep10):
                        exit_reason = "KEEP_LOST"
                    elif held >= min_hold and trend10 < cfg.trade_keep_trend_support_10:
                        exit_reason = "TREND_FADE"
                    elif held >= planned_hold:
                        exit_reason = f"MAX_HOLD_{planned_hold}"

                if exit_reason:
                    pending = {"type": "EXIT", "reason": exit_reason}

            if position != 0:
                close_px = float(day.loc[len(day) - 1, "etf_close"])
                gross = position * (close_px / entry_px - 1.0)
                net = gross - cost
                day_pnl += net
                trades.append(
                    {
                        "date": date,
                        "entry_time": day.loc[entry_bar, "datetime"],
                        "exit_time": day.loc[len(day) - 1, "datetime"],
                        "entry_state": entry_state,
                        "exit_state": "FORCE_EOD",
                        "direction": "LONG" if position > 0 else "SHORT",
                        "planned_hold": planned_hold,
                        "bars_held": len(day) - 1 - entry_bar,
                        "entry_px": entry_px,
                        "exit_px": close_px,
                        "hold_profile": hold_profile,
                        "gross_bps": gross * 10000.0,
                        "net_bps": net * 10000.0,
                    }
                )
                trades_today += 1
            daily_rows.append(
                {
                    "date": date,
                    "day_pnl_bps": day_pnl * 10000.0,
                    "n_trades": trades_today,
                }
            )
            continue

        position = 0
        entry_bar = -1
        entry_px = np.nan
        entry_state = ""
        planned_hold = 0
        min_hold = 0
        pending: dict | None = None
        day_pnl = 0.0
        trades_today = 0

        for bar in range(len(day)):
            open_px = float(day.loc[bar, "etf_open"])

            if pending is not None:
                if pending["type"] == "EXIT" and position != 0:
                    gross = position * (open_px / entry_px - 1.0)
                    net = gross - cost
                    day_pnl += net
                    trades.append(
                        {
                            "date": date,
                            "entry_time": day.loc[entry_bar, "datetime"],
                            "exit_time": day.loc[bar, "datetime"],
                            "entry_state": entry_state,
                            "exit_state": day.loc[bar, "state"],
                            "direction": "LONG" if position > 0 else "SHORT",
                            "planned_hold": planned_hold,
                            "bars_held": bar - entry_bar,
                            "entry_px": entry_px,
                            "exit_px": open_px,
                            "gross_bps": gross * 10000.0,
                            "net_bps": net * 10000.0,
                        }
                    )
                    position = 0
                    trades_today += 1
                    entry_bar = -1
                    entry_px = np.nan
                    entry_state = ""
                    planned_hold = 0
                    min_hold = 0

                elif pending["type"] == "ENTER" and position == 0:
                    position = pending["side"]
                    entry_bar = bar
                    entry_px = open_px
                    entry_state = pending["state"]
                    planned_hold = pending["hold"]
                    min_hold = pending["min_hold"]
                pending = None

            if bar >= len(day) - 1:
                break

            state = day.loc[bar, "state"]
            dir10 = float(day.loc[bar, "DirScore_10"])
            exh10 = float(day.loc[bar, "Exhaustion_10"])
            keep10 = bool(day.loc[bar, "Keep_10"])
            side_signal = 0
            if cfg.trade_enable_long and state == "CONFIRMED_UP" and dir10 >= cfg.trade_dir_long_10 and exh10 <= cfg.trade_exhaustion_max_10:
                side_signal = 1
            elif cfg.trade_enable_short and state == "CONFIRMED_DOWN" and dir10 <= cfg.trade_dir_short_10 and exh10 <= cfg.trade_exhaustion_max_10:
                side_signal = -1
            elif not cfg.trade_confirmed_only:
                pref_h = int(day.loc[bar, "preferred_horizon"])
                if cfg.trade_enable_long and state.endswith("_UP"):
                    side_signal = 1
                elif cfg.trade_enable_short and state.endswith("_DOWN"):
                    side_signal = -1
            else:
                pref_h = 10

            if position == 0:
                if side_signal != 0 and trades_today < cfg.trade_max_per_day:
                    pending = {
                        "type": "ENTER",
                        "side": side_signal,
                        "state": state,
                        "hold": cfg.trade_max_hold_10 if cfg.trade_confirmed_only else (10 if pref_h == 10 else 5),
                        "min_hold": cfg.trade_min_hold_10 if cfg.trade_confirmed_only else (5 if pref_h == 10 else 3),
                    }
                continue

            held = bar - entry_bar
            min_hold_signal = max(min_hold - 1, 0)
            max_hold_signal = max(planned_hold - 1, 0)
            if not cfg.trade_confirmed_only and side_signal == position and pref_h == 10:
                planned_hold = max(planned_hold, 10)
                min_hold = max(min_hold, 5)

            should_exit = False
            if state == "EXHAUSTED" and held >= min_hold_signal:
                should_exit = True
            elif side_signal == -position and held >= min_hold_signal:
                should_exit = True
            elif not cfg.trade_confirmed_only and position > 0 and not bool(day.loc[bar, "Keep_5"]) and held >= min_hold_signal:
                should_exit = True
            elif not cfg.trade_confirmed_only and position < 0 and not bool(day.loc[bar, "Keep_5"]) and held >= min_hold_signal:
                should_exit = True
            elif held >= max_hold_signal:
                should_exit = True
            elif not cfg.trade_confirmed_only and held >= max_hold_signal and state == "NOISE":
                should_exit = True

            if should_exit:
                pending = {"type": "EXIT"}

        if position != 0:
            open_px = float(day.loc[len(day) - 1, "etf_close"])
            gross = position * (open_px / entry_px - 1.0)
            net = gross - cost
            day_pnl += net
            trades.append(
                {
                    "date": date,
                    "entry_time": day.loc[entry_bar, "datetime"],
                    "exit_time": day.loc[len(day) - 1, "datetime"],
                    "entry_state": entry_state,
                    "exit_state": "FORCE_EOD",
                    "direction": "LONG" if position > 0 else "SHORT",
                    "planned_hold": planned_hold,
                    "bars_held": len(day) - 1 - entry_bar,
                    "entry_px": entry_px,
                    "exit_px": open_px,
                    "gross_bps": gross * 10000.0,
                    "net_bps": net * 10000.0,
                }
            )
            trades_today += 1

        daily_rows.append(
            {
                "date": date,
                "day_pnl_bps": day_pnl * 10000.0,
                "n_trades": trades_today,
            }
        )

    return pd.DataFrame(trades), pd.DataFrame(daily_rows)


def summarize_performance(trades: pd.DataFrame, daily: pd.DataFrame, scored: pd.DataFrame, cfg: StrategyConfig) -> dict:
    out: dict = {}
    out["sample"] = {
        "date_start": str(scored["date"].min().date()) if len(scored) else None,
        "date_end": str(scored["date"].max().date()) if len(scored) else None,
        "n_days": int(scored["date"].nunique()),
        "n_rows": int(len(scored)),
        "target_etf_code": cfg.target_etf_code,
    }

    state_counts = scored["state"].value_counts(dropna=False).to_dict()
    out["state_counts"] = {str(k): int(v) for k, v in state_counts.items()}

    if daily.empty:
        out["backtest"] = {}
        return out

    pnl = daily["day_pnl_bps"].to_numpy(dtype=float) / 10000.0
    cum = (1.0 + pd.Series(pnl)).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0) if len(cum) else 0.0
    ann_ret = float((1.0 + total_ret) ** (242 / max(len(daily), 1)) - 1.0) if len(daily) else 0.0
    sharpe = float(pnl.mean() / pnl.std(ddof=0) * np.sqrt(242)) if pnl.std(ddof=0) > 0 else 0.0
    dd = float((cum / cum.cummax() - 1.0).min()) if len(cum) else 0.0

    out["backtest"] = {
        "n_trades": int(len(trades)),
        "n_trade_days": int((daily["n_trades"] > 0).sum()),
        "avg_trades_per_trade_day": float(daily.loc[daily["n_trades"] > 0, "n_trades"].mean()) if (daily["n_trades"] > 0).any() else 0.0,
        "win_rate": float((trades["net_bps"] > 0).mean()) if len(trades) else 0.0,
        "avg_trade_bps": float(trades["net_bps"].mean()) if len(trades) else 0.0,
        "median_trade_bps": float(trades["net_bps"].median()) if len(trades) else 0.0,
        "avg_day_bps": float(daily["day_pnl_bps"].mean()) if len(daily) else 0.0,
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": dd,
    }

    state_stats = []
    for state, grp in scored.groupby("state", sort=False):
        side = 1
        if "DOWN" in state:
            side = -1
        elif state in {"NOISE", "EXHAUSTED"}:
            side = 0

        row = {
            "state": state,
            "count": int(len(grp)),
        }
        for horizon in DIAG_HORIZONS:
            col = f"fwd_open_ret_{horizon}"
            if col not in grp.columns:
                continue
            signed = grp[col] * side if side != 0 else grp[col]
            row[f"mean_fwd_{horizon}_bps"] = float(grp[col].mean() * 10000.0)
            row[f"mean_signed_fwd_{horizon}_bps"] = float(signed.mean() * 10000.0)
            row[f"hit_{horizon}"] = float((signed > 0).mean()) if side != 0 else float((grp[col] > 0).mean())
            row[f"tradable_{horizon}_over_cost"] = float((signed.abs() * 10000.0 > cfg.round_trip_cost_bps).mean())
        if "fwd_eod_ret" in grp.columns:
            signed = grp["fwd_eod_ret"] * side if side != 0 else grp["fwd_eod_ret"]
            row["mean_fwd_eod_bps"] = float(grp["fwd_eod_ret"].mean() * 10000.0)
            row["mean_signed_fwd_eod_bps"] = float(signed.mean() * 10000.0)
            row["hit_eod"] = float((signed > 0).mean()) if side != 0 else float((grp["fwd_eod_ret"] > 0).mean())
            row["tradable_eod_over_cost"] = float((signed.abs() * 10000.0 > cfg.round_trip_cost_bps).mean())
        state_stats.append(row)
    out["state_forward_stats"] = state_stats
    return out


def main() -> None:
    t0 = time.time()
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 120)
    print("Explore 14: HS300 Resonance Propagation Strategy")
    print("=" * 120)
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    static_df = load_static_universe(cfg)
    industry_df = load_industry_map(cfg.industry_map_path)
    mcap_panel = pd.DataFrame()
    if cfg.universe_mode == "dynamic_mcap_proxy":
        mcap_panel = load_mcap_panel(cfg.daily_mcap_panel_path, cfg.start_date, cfg.end_date)
        print(f"Loaded mcap proxy panel: rows={len(mcap_panel)}, dates={mcap_panel['date'].nunique()}")
    dates = select_dates(cfg)
    print(f"Selected {len(dates)} overlapping stock/ETF dates.")

    day_panels: List[pd.DataFrame] = []
    skipped = 0
    for i, d in enumerate(dates, start=1):
        try:
            day_static = static_df
            if cfg.universe_mode == "dynamic_mcap_proxy":
                day_static = build_dynamic_proxy_universe(d, mcap_panel, industry_df, cfg)
                if day_static.empty or len(day_static) < cfg.min_active_constituents:
                    skipped += 1
                    continue
            panel = compute_day_features(d, day_static, cfg, mcap_panel=mcap_panel, industry_df=industry_df)
            if panel is None or panel.empty:
                skipped += 1
                continue
            panel["universe_mode"] = cfg.universe_mode
            panel["universe_size"] = int(len(day_static))
            day_panels.append(panel)
            if i <= cfg.preview_days:
                print(f"[{i}/{len(dates)}] loaded {d} rows={len(panel)} active_mean={panel['hs300_active_now'].mean():.1f}")
        except Exception as e:
            skipped += 1
            print(f"[warn] skip {d}: {e}")

    if not day_panels:
        raise RuntimeError("No valid day panels built.")

    raw_panel = pd.concat(day_panels, axis=0, ignore_index=True)
    print(f"Built raw panel: rows={len(raw_panel)}, days={raw_panel['date'].nunique()}, skipped={skipped}")

    scored = score_panel(raw_panel, cfg)
    trades, daily = backtest_etf(scored, cfg)
    summary = summarize_performance(trades, daily, scored, cfg)

    raw_path = Path(cfg.output_dir) / "raw_panel.parquet"
    scored_path = Path(cfg.output_dir) / "scored_panel.parquet"
    trades_path = Path(cfg.output_dir) / "trades.csv"
    daily_path = Path(cfg.output_dir) / "daily_pnl.csv"
    state_path = Path(cfg.output_dir) / "state_forward_stats.csv"
    summary_path = Path(cfg.output_dir) / "summary.json"

    raw_panel.to_parquet(raw_path, index=False)
    scored.to_parquet(scored_path, index=False)
    trades.to_csv(trades_path, index=False)
    daily.to_csv(daily_path, index=False)
    pd.DataFrame(summary.get("state_forward_stats", [])).to_csv(state_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "artifacts": {
                    "raw_panel": str(raw_path.resolve()),
                    "scored_panel": str(scored_path.resolve()),
                    "trades": str(trades_path.resolve()),
                    "daily_pnl": str(daily_path.resolve()),
                    "state_forward_stats": str(state_path.resolve()),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nBacktest summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved results under: {Path(cfg.output_dir).resolve()}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
