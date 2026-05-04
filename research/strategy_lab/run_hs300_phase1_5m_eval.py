#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Phase 1 evaluator for HS300 intraday propagation signals.

This is the lightweight local counterpart of jq_hs300_propagation_phase1.py.
It reads local parquet minute data, computes the core propagation features, and
evaluates high-confidence long/short signals for the next 5-minute return.

It intentionally avoids the heavier Explore14 full-market context so a 90-day
diagnostic run stays fast and readable.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path("/Users/daweizhong/Documents/projects")
REPO_ROOT = Path(__file__).resolve().parents[2]

HORIZONS = (5, 10)
RAW_NAMES = ("B", "dB", "Corr", "EP", "TopK", "SE", "Dir", "WSB", "LFR", "WPS", "ETFRet")


@dataclass
class Config:
    stock_data_dir: str = str(PROJECT_ROOT / "stock_data")
    etf_data_dir: str = str(PROJECT_ROOT / "ETF data core7")
    weights_csv: str = str(REPO_ROOT / "research" / "strategy_lab" / "data" / "hs300_weights.csv")
    industry_map_path: str = str(
        PROJECT_ROOT / "artifacts" / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun" / "industry_map_baostock.parquet"
    )
    output_dir: str = str(REPO_ROOT / "results" / "510300_breadth_regime" / "hs300_phase1_5m_90d_v1")
    target_etf_code: str = "510300.XSHG"
    eval_days: int = 90
    warmup_days: int = 25
    end_date: str = ""
    amount_min: float = 100000.0
    min_active_constituents: int = 120
    tick_size_stock: float = 0.01
    z_window_days: int = 20
    z_min_history: int = 5
    topk_k: int = 10
    leader_n: int = 30
    weight_bins: int = 5
    gate_5: float = 0.60
    gate_10: float = 0.55
    dir_5: float = 0.50
    dir_10: float = 0.45
    high_conf_dir_5: float = 0.75
    exh_5: float = 0.80
    exh_10: float = 0.90


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Evaluate HS300 Phase 1 propagation signals for 5m forward returns")
    p.add_argument("--stock-data-dir", default=Config.stock_data_dir)
    p.add_argument("--etf-data-dir", default=Config.etf_data_dir)
    p.add_argument("--weights-csv", default=Config.weights_csv)
    p.add_argument("--industry-map-path", default=Config.industry_map_path)
    p.add_argument("--output-dir", default=Config.output_dir)
    p.add_argument("--target-etf-code", default=Config.target_etf_code)
    p.add_argument("--eval-days", type=int, default=Config.eval_days)
    p.add_argument("--warmup-days", type=int, default=Config.warmup_days)
    p.add_argument("--end-date", default="")
    p.add_argument("--amount-min", type=float, default=Config.amount_min)
    p.add_argument("--high-conf-dir-5", type=float, default=Config.high_conf_dir_5)
    return Config(**vars(p.parse_args()))


def select_dates(cfg: Config) -> List[str]:
    stock_dates = {p.stem for p in Path(cfg.stock_data_dir).glob("*.parquet")}
    etf_dates = {p.stem for p in Path(cfg.etf_data_dir).glob("*.parquet")}
    dates = sorted(stock_dates & etf_dates)
    if cfg.end_date:
        dates = [d for d in dates if d <= cfg.end_date]
    need = cfg.eval_days + cfg.warmup_days
    if len(dates) < need:
        raise ValueError(f"Not enough overlapping dates: have={len(dates)} need={need}")
    return dates[-need:]


def load_universe(cfg: Config) -> pd.DataFrame:
    w = pd.read_csv(cfg.weights_csv)
    w = w[["code", "weight_pct"]].copy()
    w["code"] = w["code"].astype(str)
    w["weight"] = w["weight_pct"].astype(float) / 100.0
    w = w[w["weight"] > 0].copy()
    w["weight"] = w["weight"] / w["weight"].sum()

    industry = load_industry_map(cfg.industry_map_path)
    if not industry.empty:
        w = w.merge(industry, on="code", how="left")
    w["industry"] = w.get("industry", pd.Series("Unknown", index=w.index)).fillna("Unknown").astype(str)

    w = w.sort_values("weight", ascending=False).reset_index(drop=True)
    w["leader_flag"] = 0
    w.loc[: max(cfg.leader_n - 1, 0), "leader_flag"] = 1
    w["weight_bin"] = build_weight_bins(w["weight"], cfg.weight_bins)
    return w[["code", "weight", "industry", "leader_flag", "weight_bin"]]


def load_industry_map(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["code", "industry"])
    df = pd.read_parquet(p)
    if "code" not in df.columns or "industry" not in df.columns:
        return pd.DataFrame(columns=["code", "industry"])
    out = df[["code", "industry"]].copy()
    out["code"] = out["code"].astype(str)
    out["industry"] = out["industry"].fillna("Unknown").astype(str)
    return out.drop_duplicates("code", keep="last")


def build_weight_bins(weights: pd.Series, bins: int) -> np.ndarray:
    q = min(int(bins), len(weights))
    ranks = weights.rank(method="first", ascending=True)
    try:
        return pd.qcut(ranks, q=q, labels=False, duplicates="drop").astype(int).values
    except Exception:
        return np.zeros(len(weights), dtype=int)


def read_stock_day(path: Path, codes: Iterable[str]) -> pd.DataFrame:
    cols = ["code", "datetime", "close", "volume", "money", "paused"]
    df = pd.read_parquet(path, columns=cols, filters=[("code", "in", list(codes))])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["code"] = df["code"].astype(str)
    return df


def read_etf_day(path: Path, target_code: str) -> pd.DataFrame:
    cols = ["code", "datetime", "open", "close", "volume", "money", "paused"]
    df = pd.read_parquet(path, columns=cols, filters=[("code", "==", target_code)])
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").drop_duplicates("datetime", keep="last")


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    total = np.nansum(mat, axis=1, keepdims=True)
    return np.divide(mat, total, out=np.zeros_like(mat, dtype=float), where=total > 0)


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    total = float(np.nansum(vec))
    if total <= 0:
        return np.zeros_like(vec, dtype=float)
    return np.nan_to_num(vec / total)


def rolling_recent_trade(volume: np.ndarray, horizon: int) -> np.ndarray:
    has_trade = (volume > 0).astype(float)
    out = np.zeros_like(has_trade, dtype=float)
    for i in range(len(has_trade)):
        start = max(0, i - horizon + 1)
        out[i] = has_trade[start : i + 1].sum(axis=0)
    return out > 0


def compute_corr_series(ret1: np.ndarray, active_h: np.ndarray, window: int, min_active: int) -> np.ndarray:
    out = np.full(len(ret1), np.nan, dtype=float)
    for i in range(len(ret1)):
        if i < window or active_h[i].sum() < min_active:
            continue
        cols = active_h[i]
        r = ret1[i - window + 1 : i + 1, :][:, cols]
        if r.shape[1] < 3:
            continue
        ew = np.nanmean(r, axis=1)
        var_ew = float(np.nanvar(ew))
        var_i = float(np.nanmean(np.nanvar(r, axis=0)))
        n = float(r.shape[1])
        if var_i <= 0 or n <= 1:
            continue
        out[i] = np.clip((n * var_ew / var_i - 1.0) / (n - 1.0), -1.0, 1.0)
    return out


def compute_sector_entropy(weighted_ret: np.ndarray, sector_matrix: np.ndarray) -> np.ndarray:
    contrib = np.abs(weighted_ret) @ sector_matrix
    total = contrib.sum(axis=1, keepdims=True)
    prob = np.divide(contrib, total, out=np.zeros_like(contrib), where=total > 0)
    active_n = (contrib > 0).sum(axis=1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        num = -np.nansum(np.where(prob > 0, prob * np.log(prob), 0.0), axis=1)
        den = np.where(active_n > 1, np.log(active_n), np.nan)
    return num / den


def compute_lfr(log_close: np.ndarray, ret1: np.ndarray, active_now: np.ndarray, base_weights: np.ndarray, leader_mask: np.ndarray) -> np.ndarray:
    out = np.full(len(log_close), np.nan, dtype=float)
    for i in range(21, len(log_close)):
        leaders = leader_mask & active_now[i]
        followers = (~leader_mask) & active_now[i]
        if leaders.sum() < 3 or followers.sum() < 20:
            continue
        w_lead = normalize_vec(base_weights * leaders.astype(float))
        w_follow = normalize_vec(base_weights * followers.astype(float))
        lead_prev = float(np.sum(w_lead * np.nan_to_num(log_close[i - 1] - log_close[i - 3])))
        follow_now = float(np.sum(w_follow * np.nan_to_num(ret1[i])))
        hist = []
        for j in range(i - 19, i + 1):
            hist.append(float(np.sum(w_follow * np.nan_to_num(ret1[j]))))
        sigma = float(np.std(hist))
        if sigma > 1e-10:
            out[i] = np.sign(lead_prev) * follow_now / sigma
    return out


def compute_wps(sign_h: np.ndarray, active_h: np.ndarray, base_weights: np.ndarray, weight_bins: np.ndarray, wsb: np.ndarray) -> np.ndarray:
    out = np.full(len(sign_h), np.nan, dtype=float)
    bins = sorted(set(weight_bins.tolist()))
    x_axis = np.linspace(-2.0, 2.0, len(bins))
    for i in range(len(sign_h)):
        xs = []
        ys = []
        for x, b in zip(x_axis, bins):
            mask = active_h[i] & (weight_bins == b)
            if mask.sum() == 0:
                continue
            w = normalize_vec(base_weights * mask.astype(float))
            xs.append(float(x))
            ys.append(float(np.sum(w * sign_h[i])))
        if len(xs) < 2:
            continue
        x = np.asarray(xs)
        y = np.asarray(ys)
        x = x - x.mean()
        y = y - y.mean()
        denom = float(np.dot(x, x))
        if denom > 0:
            out[i] = np.sign(wsb[i]) * float(np.dot(x, y) / denom)
    return out


def compute_day_features(date_str: str, universe: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
    stock_path = Path(cfg.stock_data_dir) / f"{date_str}.parquet"
    etf_path = Path(cfg.etf_data_dir) / f"{date_str}.parquet"
    stocks = universe["code"].tolist()
    stock = read_stock_day(stock_path, stocks)
    etf = read_etf_day(etf_path, cfg.target_etf_code)
    if stock.empty or etf.empty:
        return None

    close_df = stock.pivot_table(index="datetime", columns="code", values="close").reindex(columns=stocks)
    volume_df = stock.pivot_table(index="datetime", columns="code", values="volume").reindex(columns=stocks).fillna(0.0)
    money_df = stock.pivot_table(index="datetime", columns="code", values="money").reindex(columns=stocks).fillna(0.0)
    paused_df = stock.pivot_table(index="datetime", columns="code", values="paused").reindex(columns=stocks).fillna(0.0)

    etf = etf.set_index("datetime").sort_index()
    times = close_df.index.intersection(etf.index)
    if len(times) < 120:
        return None

    close_df = close_df.reindex(times)
    volume_df = volume_df.reindex(times).fillna(0.0)
    money_df = money_df.reindex(times).fillna(0.0)
    paused_df = paused_df.reindex(times).fillna(0.0)
    etf = etf.reindex(times)

    close = close_df.values.astype(float)
    volume = volume_df.values.astype(float)
    money = money_df.values.astype(float)
    paused = paused_df.values.astype(float)
    log_close = np.log(close)
    ret1 = np.full_like(log_close, np.nan, dtype=float)
    ret1[1:] = log_close[1:] - log_close[:-1]

    etf_close = etf["close"].values.astype(float)
    etf_open = etf["open"].values.astype(float)
    etf_log = np.log(etf_close)
    etf_r1 = np.full(len(etf_log), np.nan, dtype=float)
    etf_r1[1:] = etf_log[1:] - etf_log[:-1]

    base_weights = universe["weight"].values.astype(float)
    leader_mask = universe["leader_flag"].values.astype(int).astype(bool)
    weight_bins = universe["weight_bin"].values.astype(int)
    industries = universe["industry"].fillna("Unknown").astype(str)
    sector_codes = industries.astype("category").cat.codes.values.astype(int)
    sector_matrix = np.zeros((len(stocks), max(int(sector_codes.max()) + 1, 1)), dtype=float)
    sector_matrix[np.arange(len(stocks)), sector_codes] = 1.0

    active_now = np.isfinite(close) & (paused <= 0.0) & (volume > 0.0) & (money >= cfg.amount_min)
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(date_str),
            "datetime": times,
            "minute_idx": np.arange(len(times), dtype=int),
            "etf_code": cfg.target_etf_code,
            "etf_open": etf_open,
            "etf_close": etf_close,
            "active_now": active_now.sum(axis=1).astype(int),
        }
    )

    for horizon in HORIZONS:
        eps_floor = 0.00020 if horizon == 5 else 0.00030
        ret_h = np.full_like(log_close, np.nan, dtype=float)
        ret_h[horizon:] = log_close[horizon:] - log_close[:-horizon]
        eps = np.maximum((1.5 * cfg.tick_size_stock) / np.maximum(close, 1e-6), eps_floor)
        sign_h = np.zeros_like(ret_h, dtype=float)
        sign_h[ret_h > eps] = 1.0
        sign_h[ret_h < -eps] = -1.0

        trade_recent = rolling_recent_trade(volume, horizon)
        active_h = active_now & trade_recent & np.isfinite(ret_h)
        active_count = active_h.sum(axis=1).astype(float)
        valid = active_count >= cfg.min_active_constituents
        row_weights = normalize_rows(active_h.astype(float) * base_weights[None, :])

        up = np.divide(((sign_h > 0) & active_h).sum(axis=1), active_count, out=np.full(len(active_count), np.nan), where=active_count > 0)
        down = np.divide(((sign_h < 0) & active_h).sum(axis=1), active_count, out=np.full(len(active_count), np.nan), where=active_count > 0)
        b = np.abs(up - down)
        m = 5 if horizon == 5 else 10
        db = b - pd.Series(b).shift(1).rolling(m, min_periods=1).mean().values
        db = np.where(np.isfinite(db), db, 0.0)
        corr = compute_corr_series(ret1, active_h, horizon, cfg.min_active_constituents)

        weighted_ret = row_weights * np.where(np.isfinite(ret_h), ret_h, 0.0)
        contrib_abs = np.abs(weighted_ret)
        total_abs = contrib_abs.sum(axis=1)
        total_sq = np.square(contrib_abs).sum(axis=1)
        ep_raw = np.divide(total_abs**2, active_count * total_sq, out=np.full(len(total_abs), np.nan), where=(active_count > 1) & (total_sq > 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ep = np.divide(ep_raw - 1.0 / active_count, 1.0 - 1.0 / active_count, out=np.full(len(total_abs), np.nan), where=active_count > 1)

        k = min(cfg.topk_k, contrib_abs.shape[1])
        topk = np.full(len(total_abs), np.nan, dtype=float)
        if k > 0:
            topk_sum = np.partition(contrib_abs, contrib_abs.shape[1] - k, axis=1)[:, -k:].sum(axis=1)
            topk = np.divide(topk_sum, total_abs, out=np.full(len(total_abs), np.nan), where=total_abs > 0)

        se = compute_sector_entropy(weighted_ret, sector_matrix)
        etf_ret_h = np.full(len(etf_log), np.nan, dtype=float)
        etf_ret_h[horizon:] = etf_log[horizon:] - etf_log[:-horizon]
        etf_path = pd.Series(np.abs(etf_r1)).rolling(horizon, min_periods=horizon).sum().values
        dir_strength = np.divide(np.abs(etf_ret_h), etf_path, out=np.full(len(etf_ret_h), np.nan), where=etf_path > 0)
        wsb = (row_weights * sign_h).sum(axis=1)
        lfr = compute_lfr(log_close, ret1, active_now, base_weights, leader_mask)
        wps = compute_wps(sign_h, active_h, base_weights, weight_bins, wsb)

        values = {
            f"active_count_{horizon}": active_count,
            f"B_{horizon}_raw": b,
            f"dB_{horizon}_raw": db,
            f"Corr_{horizon}_raw": corr,
            f"EP_{horizon}_raw": ep,
            f"TopK_{horizon}_raw": topk,
            f"SE_{horizon}_raw": se,
            f"Dir_{horizon}_raw": dir_strength,
            f"WSB_{horizon}_raw": wsb,
            f"LFR_{horizon}_raw": lfr,
            f"WPS_{horizon}_raw": wps,
            f"ETFRet_{horizon}_raw": etf_ret_h,
        }
        for col, arr in values.items():
            panel[col] = np.where(valid, arr, np.nan)

    for horizon in HORIZONS:
        panel[f"fwd_ret_{horizon}"] = np.log(pd.Series(etf_close).shift(-horizon) / pd.Series(etf_close)).values
        future_trend = []
        for i in range(len(panel)):
            r = etf_r1[i + 1 : i + 1 + horizon]
            if len(r) < horizon or np.isnan(r).any():
                future_trend.append(np.nan)
                continue
            denom = float(np.abs(r).sum())
            future_trend.append(float(abs(r.sum()) / denom) if denom > 0 else np.nan)
        panel[f"trendness_fwd_{horizon}"] = future_trend

    return panel


def bucket_robust_zscore(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    scored = panel.sort_values(["date", "minute_idx"]).reset_index(drop=True).copy()
    raw_cols = [f"{name}_{h}_raw" for h in HORIZONS for name in RAW_NAMES]
    for col in raw_cols:
        hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.z_window_days))
        vals = scored[col].values.astype(float)
        buckets = scored["minute_idx"].values.astype(int)
        z = np.full(len(scored), np.nan, dtype=float)
        for i, val in enumerate(vals):
            bucket = int(buckets[i])
            history = hist[bucket]
            if np.isfinite(val):
                arr = np.asarray(history, dtype=float)
                if len(arr) >= cfg.z_min_history:
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med)))
                    scale = 1.4826 * mad
                    if scale <= 1e-10:
                        sd = float(np.std(arr))
                        scale = sd if sd > 1e-10 else 1.0
                    z[i] = (val - med) / scale
                elif len(arr) >= 2:
                    mu = float(np.mean(arr))
                    sd = float(np.std(arr))
                    z[i] = (val - mu) / (sd if sd > 1e-10 else 1.0)
                else:
                    z[i] = 0.0
                history.append(float(val))
        scored[f"z_{col}"] = np.clip(z, -8.0, 8.0)
    return scored


def score_panel(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    scored = bucket_robust_zscore(panel, cfg)
    for horizon in HORIZONS:
        z = {name: scored[f"z_{name}_{horizon}_raw"].fillna(0.0) for name in RAW_NAMES}
        internal = 0.30 * z["WSB"] + 0.20 * z["dB"] + 0.20 * z["Corr"] + 0.15 * z["EP"] + 0.15 * z["SE"]
        ipg = internal - z["ETFRet"]
        pad = z["ETFRet"] - internal
        gate = 0.22 * z["B"] + 0.18 * z["dB"] + 0.18 * z["Corr"] + 0.14 * z["EP"] + 0.12 * z["SE"] + 0.08 * z["Dir"] - 0.08 * z["TopK"]
        direction = 0.40 * ipg + 0.30 * z["LFR"] + 0.20 * z["WPS"] + 0.10 * z["WSB"]
        exhaustion = 0.50 * np.maximum(pad, 0.0) + 0.30 * np.maximum(z["TopK"], 0.0) + 0.20 * np.maximum(-z["dB"], 0.0)
        scored[f"InternalPressure_{horizon}"] = internal
        scored[f"IPG_{horizon}"] = ipg
        scored[f"PAD_{horizon}"] = pad
        scored[f"GateScore_{horizon}"] = gate
        scored[f"DirScore_{horizon}"] = direction
        scored[f"Exhaustion_{horizon}"] = exhaustion
        scored[f"Gate_{horizon}"] = gate > (cfg.gate_5 if horizon == 5 else cfg.gate_10)

    state = []
    for _, r in scored.iterrows():
        gate5 = bool(r["Gate_5"])
        gate10 = bool(r["Gate_10"])
        dir5 = float(r["DirScore_5"])
        dir10 = float(r["DirScore_10"])
        exh5 = float(r["Exhaustion_5"])
        exh10 = float(r["Exhaustion_10"])
        if (gate5 or gate10) and (exh5 > cfg.exh_5 or exh10 > cfg.exh_10 or (r["z_dB_5_raw"] < -0.25 and r["z_LFR_5_raw"] < -0.25)):
            state.append("EXHAUSTED")
        elif gate5 and gate10 and dir5 > cfg.dir_5 and dir10 > cfg.dir_10 and exh10 < cfg.exh_10:
            state.append("CONFIRMED_UP")
        elif gate5 and gate10 and dir5 < -cfg.dir_5 and dir10 < -cfg.dir_10 and exh10 < cfg.exh_10:
            state.append("CONFIRMED_DOWN")
        elif gate5 and dir5 > cfg.dir_5 and dir10 > -cfg.dir_10 and exh5 < cfg.exh_5:
            state.append("EMERGING_UP")
        elif gate5 and dir5 < -cfg.dir_5 and dir10 < cfg.dir_10 and exh5 < cfg.exh_5:
            state.append("EMERGING_DOWN")
        else:
            state.append("NOISE")
    scored["state"] = state

    signal_side = np.zeros(len(scored), dtype=int)
    high_conf = (
        (scored["Gate_5"])
        & (scored["Exhaustion_5"] < cfg.exh_5)
        & (scored["DirScore_5"].abs() >= cfg.high_conf_dir_5)
        & ((scored["DirScore_5"] * scored["DirScore_10"]) >= -0.10)
    )
    signal_side[high_conf & (scored["DirScore_5"] > 0)] = 1
    signal_side[high_conf & (scored["DirScore_5"] < 0)] = -1
    scored["signal_side_5"] = signal_side
    scored["signal_name_5"] = np.where(signal_side > 0, "LONG", np.where(signal_side < 0, "SHORT", "FLAT"))
    scored["signal_strength_5"] = scored["DirScore_5"].abs()
    scored["signed_fwd_ret_5"] = scored["signal_side_5"] * scored["fwd_ret_5"]
    return scored


def regular_signal_mask(df: pd.DataFrame) -> pd.Series:
    dt = pd.to_datetime(df["datetime"])
    hhmm = dt.dt.hour * 100 + dt.dt.minute
    return (hhmm >= 935) & ~((hhmm >= 1125) & (hhmm <= 1305)) & (hhmm < 1450)


def summarize_eval(eval_panel: pd.DataFrame, cfg: Config) -> dict:
    sig = eval_panel[(eval_panel["signal_side_5"] != 0) & eval_panel["fwd_ret_5"].notna() & regular_signal_mask(eval_panel)].copy()
    throttled = throttle_signals(sig, cooldown_bars=5)
    all_valid = eval_panel[eval_panel["fwd_ret_5"].notna() & regular_signal_mask(eval_panel)].copy()
    by_side = {}
    for side_name, side_value in [("LONG", 1), ("SHORT", -1)]:
        sub = sig[sig["signal_side_5"] == side_value]
        by_side[side_name] = summarize_signal_slice(sub)
    by_state = {}
    for state, sub in sig.groupby("state", sort=True):
        by_state[state] = summarize_signal_slice(sub)
    summary = {
        "eval_start": str(eval_panel["date"].min().date()),
        "eval_end": str(eval_panel["date"].max().date()),
        "eval_days": int(eval_panel["date"].nunique()),
        "valid_minutes": int(len(all_valid)),
        "signal_minutes": int(len(sig)),
        "signal_coverage": float(len(sig) / len(all_valid)) if len(all_valid) else 0.0,
        "all_signal": summarize_signal_slice(sig),
        "throttled_5bar_signal": summarize_signal_slice(throttled),
        "by_side": by_side,
        "by_state": by_state,
        "baseline_all_minutes": {
            "mean_fwd_ret_5_bps": float(all_valid["fwd_ret_5"].mean() * 10000.0),
            "abs_mean_fwd_ret_5_bps": float(all_valid["fwd_ret_5"].abs().mean() * 10000.0),
            "up_rate": float((all_valid["fwd_ret_5"] > 0).mean()),
            "n": int(len(all_valid)),
        },
        "config": asdict(cfg),
    }
    return summary


def summarize_signal_slice(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n": 0,
            "nonzero_n": 0,
            "zero_rate": None,
            "hit_rate": None,
            "nonzero_hit_rate": None,
            "mean_signed_bps": None,
            "nonzero_mean_signed_bps": None,
            "median_signed_bps": None,
            "p25_signed_bps": None,
            "p75_signed_bps": None,
            "avg_abs_fwd_bps": None,
            "mean_raw_fwd_bps": None,
        }
    s = df["signed_fwd_ret_5"].astype(float)
    nz = df[s.abs() > 1e-12].copy()
    nz_s = nz["signed_fwd_ret_5"].astype(float) if len(nz) else pd.Series(dtype=float)
    return {
        "n": int(len(df)),
        "days": int(df["date"].nunique()),
        "nonzero_n": int(len(nz)),
        "zero_rate": float(1.0 - len(nz) / len(df)) if len(df) else None,
        "hit_rate": float((s > 0).mean()),
        "nonzero_hit_rate": float((nz_s > 0).mean()) if len(nz_s) else None,
        "mean_signed_bps": float(s.mean() * 10000.0),
        "nonzero_mean_signed_bps": float(nz_s.mean() * 10000.0) if len(nz_s) else None,
        "median_signed_bps": float(s.median() * 10000.0),
        "p25_signed_bps": float(s.quantile(0.25) * 10000.0),
        "p75_signed_bps": float(s.quantile(0.75) * 10000.0),
        "avg_abs_fwd_bps": float(df["fwd_ret_5"].abs().mean() * 10000.0),
        "mean_raw_fwd_bps": float(df["fwd_ret_5"].mean() * 10000.0),
    }


def throttle_signals(signals: pd.DataFrame, cooldown_bars: int = 5) -> pd.DataFrame:
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


def threshold_scan(eval_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = eval_panel[eval_panel["fwd_ret_5"].notna() & regular_signal_mask(eval_panel)].copy()
    for th in [0.50, 0.60, 0.75, 0.90, 1.10, 1.30, 1.60, 2.00]:
        mask = (
            (base["Gate_5"])
            & (base["Exhaustion_5"] < 0.80)
            & (base["DirScore_5"].abs() >= th)
            & ((base["DirScore_5"] * base["DirScore_10"]) >= -0.10)
        )
        sub = base[mask].copy()
        sub["scan_side"] = np.sign(sub["DirScore_5"])
        sub["scan_signed"] = sub["scan_side"] * sub["fwd_ret_5"]
        nz = sub[sub["scan_signed"].abs() > 1e-12]
        rows.append(
            {
                "dir_abs_threshold": th,
                "n": int(len(sub)),
                "nonzero_n": int(len(nz)),
                "coverage": float(len(sub) / len(base)) if len(base) else 0.0,
                "hit_rate": float((sub["scan_signed"] > 0).mean()) if len(sub) else np.nan,
                "nonzero_hit_rate": float((nz["scan_signed"] > 0).mean()) if len(nz) else np.nan,
                "mean_signed_bps": float(sub["scan_signed"].mean() * 10000.0) if len(sub) else np.nan,
                "nonzero_mean_signed_bps": float(nz["scan_signed"].mean() * 10000.0) if len(nz) else np.nan,
                "long_n": int((sub["scan_side"] > 0).sum()) if len(sub) else 0,
                "short_n": int((sub["scan_side"] < 0).sum()) if len(sub) else 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    universe = load_universe(cfg)
    dates = select_dates(cfg)
    eval_dates = set(dates[-cfg.eval_days :])

    print(f"Selected {len(dates)} dates: {dates[0]} -> {dates[-1]}; eval last {cfg.eval_days} days.")
    print(f"Universe size={len(universe)}, target={cfg.target_etf_code}")

    panels: List[pd.DataFrame] = []
    skipped: List[str] = []
    for i, d in enumerate(dates, start=1):
        try:
            panel = compute_day_features(d, universe, cfg)
        except Exception as exc:
            print(f"[warn] skip {d}: {exc}")
            skipped.append(d)
            continue
        if panel is None or panel.empty:
            skipped.append(d)
            continue
        panels.append(panel)
        if i <= 3 or i == len(dates):
            print(f"[{i}/{len(dates)}] {d} rows={len(panel)} active_mean={panel['active_now'].mean():.1f}")

    if not panels:
        raise RuntimeError("No panels built")

    raw = pd.concat(panels, ignore_index=True)
    scored = score_panel(raw, cfg)
    scored["is_eval"] = scored["date"].dt.strftime("%Y-%m-%d").isin(eval_dates)
    eval_panel = scored[scored["is_eval"]].copy()
    summary = summarize_eval(eval_panel, cfg)
    scan = threshold_scan(eval_panel)

    signals = eval_panel[(eval_panel["signal_side_5"] != 0) & eval_panel["fwd_ret_5"].notna() & regular_signal_mask(eval_panel)].copy()
    throttled = throttle_signals(signals, cooldown_bars=5)
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
        "LFR_5_raw",
        "WPS_5_raw",
        "WSB_5_raw",
        "EP_5_raw",
        "SE_5_raw",
        "TopK_5_raw",
        "ETFRet_5_raw",
        "fwd_ret_5",
        "signed_fwd_ret_5",
    ]
    signal_cols = [c for c in signal_cols if c in signals.columns]

    scored_path = out_dir / "scored_panel.parquet"
    eval_path = out_dir / "eval_panel.parquet"
    signals_path = out_dir / "signals_5m.csv"
    throttled_path = out_dir / "signals_5m_throttled.csv"
    scan_path = out_dir / "threshold_scan.csv"
    summary_path = out_dir / "summary.json"

    scored.to_parquet(scored_path, index=False)
    eval_panel.to_parquet(eval_path, index=False)
    signals[signal_cols].to_csv(signals_path, index=False)
    throttled[signal_cols].to_csv(throttled_path, index=False)
    scan.to_csv(scan_path, index=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary,
                "skipped_dates": skipped,
                "artifacts": {
                    "scored_panel": str(scored_path.resolve()),
                    "eval_panel": str(eval_path.resolve()),
                    "signals_5m": str(signals_path.resolve()),
                    "signals_5m_throttled": str(throttled_path.resolve()),
                    "threshold_scan": str(scan_path.resolve()),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
