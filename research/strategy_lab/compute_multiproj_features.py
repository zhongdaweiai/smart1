#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-projection cross-sectional features for HS300 intraday wave research.

Phase 14.5 companion to run_hs300_phase1_5m_eval.py. The phase-1 engine
projects everything through HS300 cap weights and produces a single
weighted-breadth view. The Wave Framework (sec. 12) requires a useful wave
to be visible across several projections; this module computes the missing
projections so a consensus filter can be layered on top of V1 signals.

Per (date, datetime, minute_idx) the engine computes:

  Sector projection (industry from baostock map):
    Sector_Concord_h     fraction of active sectors whose signed breadth
                          agrees in sign with the HS300 weighted breadth
    Sector_Down_Frac_h   fraction of active sectors with signed breadth < -0.10
    Sector_Up_Frac_h     fraction of active sectors with signed breadth > +0.10
    Sector_N_Active_h    count of sectors with >=3 active stocks

  Weight-bucket projection (5 buckets by HS300 weight rank):
    Bucket_Top_Sign_h     signed breadth in top weight bucket
    Bucket_Bottom_Sign_h  signed breadth in bottom weight bucket
    Bucket_Penetration_h  top - bottom (positive = core stronger than tail)
    Bucket_Slope_h        OLS slope across bucket index
                          (positive = move concentrated in larger weights)

  Equal-weight whole-market projection over active HS300 constituents:
    EW_SignedBreadth_h        equal-weight (up - down) / active
    EW_to_W_Penetration_h     index-weighted - equal-weight signed breadth
                              (positive = wave reaching the index core)

  Wavefront (5m horizon only):
    NewDown_5             count of stocks turning sign>=0 -> sign<0
    NewUp_5               count of stocks turning sign<=0 -> sign>0
    WavefrontDown_Frac_5  NewDown / active count
    WavefrontUp_Frac_5    NewUp / active count

Universe construction follows run_hs300_downside_walkforward.py: daily HS300
weights are lagged one trading day, top-30 by weight are leaders, weights are
binned by qcut into 5 buckets. Only days with both stock and ETF parquet
present are processed. Days with weight-date strictly before the trade date
are required (no same-day weight leak).

Output:
  multiproj_panel.parquet     joined panel for the requested date range
  weight_usage.csv            mirror of run_hs300_downside_walkforward usage
  skipped_dates.csv           anything skipped, for diagnosis

Hard rules (mirroring AGENTS.md):
- Use only the precise ETF directory (ETF data core7 precise).
- Never apply same-day or future weight snapshots.
- Compute features only from data available at or before the bar.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from run_hs300_phase1_5m_eval import (
    Config as FeatureConfig,
    build_weight_bins,
    load_industry_map,
    normalize_rows,
    normalize_vec,
    read_etf_day,
    read_stock_day,
    rolling_recent_trade,
)
from run_hs300_downside_walkforward import (
    PROJECT_ROOT,
    REPO_ROOT,
    list_trade_dates,
    load_weight_table,
    make_universe,
    previous_weight_date,
)


HORIZONS = (5, 10)
TICK_SIZE_STOCK = 0.01
SECTOR_DOWN_THRESHOLD = -0.10
SECTOR_UP_THRESHOLD = 0.10


@dataclass
class Config:
    stock_data_dir: str = str(PROJECT_ROOT / "stock_data")
    etf_data_dir: str = str(PROJECT_ROOT / "ETF data core7 precise")
    daily_weights_csv: str = str(
        REPO_ROOT
        / "research"
        / "strategy_lab"
        / "data"
        / "hs300_daily_weights"
        / "hs300_weights_2025-01-01_2026-04-30.csv"
    )
    industry_map_path: str = str(
        PROJECT_ROOT
        / "artifacts"
        / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun"
        / "industry_map_baostock.parquet"
    )
    output_dir: str = str(
        REPO_ROOT
        / "results"
        / "510300_breadth_regime"
        / "hs300_multiproj_panel_v1"
    )
    target_etf_code: str = "510300.XSHG"
    start_date: str = "2025-06-09"
    end_date: str = "2026-04-30"
    amount_min: float = 100000.0
    min_active_constituents: int = 120
    weight_bins: int = 5
    leader_n: int = 30


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Phase 14.5 multi-projection feature engine")
    for field, default in Config().__dict__.items():
        arg = "--" + field.replace("_", "-")
        if isinstance(default, bool):
            p.add_argument(arg, type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=default)
        elif isinstance(default, int):
            p.add_argument(arg, type=int, default=default)
        elif isinstance(default, float):
            p.add_argument(arg, type=float, default=default)
        else:
            p.add_argument(arg, default=default)
    return Config(**vars(p.parse_args()))


def compute_multiproj_day(date_str: str, universe: pd.DataFrame, cfg: Config) -> pd.DataFrame | None:
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

    close = close_df.reindex(times).values.astype(float)
    volume = volume_df.reindex(times).fillna(0.0).values.astype(float)
    money = money_df.reindex(times).fillna(0.0).values.astype(float)
    paused = paused_df.reindex(times).fillna(0.0).values.astype(float)

    log_close = np.log(close)

    base_weights = universe["weight"].values.astype(float)
    weight_bin_arr = universe["weight_bin"].values.astype(int)
    industries = universe["industry"].fillna("Unknown").astype(str)
    sector_codes = industries.astype("category").cat.codes.values.astype(int)
    n_sectors = max(int(sector_codes.max()) + 1, 1)
    n_buckets = int(weight_bin_arr.max()) + 1

    active_now = np.isfinite(close) & (paused <= 0.0) & (volume > 0.0) & (money >= cfg.amount_min)

    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(date_str),
            "datetime": times,
            "minute_idx": np.arange(len(times), dtype=int),
            "active_now_total": active_now.sum(axis=1).astype(int),
        }
    )

    for horizon in HORIZONS:
        eps_floor = 0.00020 if horizon == 5 else 0.00030
        ret_h = np.full_like(log_close, np.nan, dtype=float)
        ret_h[horizon:] = log_close[horizon:] - log_close[:-horizon]
        eps = np.maximum((1.5 * TICK_SIZE_STOCK) / np.maximum(close, 1e-6), eps_floor)
        sign_h = np.zeros_like(ret_h, dtype=float)
        sign_h[ret_h > eps] = 1.0
        sign_h[ret_h < -eps] = -1.0

        trade_recent = rolling_recent_trade(volume, horizon)
        active_h = active_now & trade_recent & np.isfinite(ret_h)
        active_count_h = active_h.sum(axis=1).astype(float)
        valid = active_count_h >= cfg.min_active_constituents

        # ============================================================
        # A. Sector projection
        # ============================================================
        sector_signed = np.zeros((len(times), n_sectors), dtype=float)
        sector_active_mask = np.zeros((len(times), n_sectors), dtype=bool)
        for s in range(n_sectors):
            cols_mask = sector_codes == s
            if cols_mask.sum() < 3:
                continue
            sec_active_h = active_h[:, cols_mask]
            sec_sign_h = sign_h[:, cols_mask]
            sec_count = sec_active_h.sum(axis=1).astype(float)
            up = ((sec_sign_h > 0) & sec_active_h).sum(axis=1)
            dn = ((sec_sign_h < 0) & sec_active_h).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                signed = np.divide(
                    up.astype(float) - dn.astype(float),
                    sec_count,
                    out=np.zeros(len(times)),
                    where=sec_count >= 3,
                )
            sector_signed[:, s] = signed
            sector_active_mask[:, s] = sec_count >= 3

        n_active_sectors = sector_active_mask.sum(axis=1).astype(float)
        sector_down_count = ((sector_signed < SECTOR_DOWN_THRESHOLD) & sector_active_mask).sum(axis=1)
        sector_up_count = ((sector_signed > SECTOR_UP_THRESHOLD) & sector_active_mask).sum(axis=1)
        sector_down_frac = np.divide(
            sector_down_count.astype(float),
            n_active_sectors,
            out=np.full(len(times), np.nan),
            where=n_active_sectors > 0,
        )
        sector_up_frac = np.divide(
            sector_up_count.astype(float),
            n_active_sectors,
            out=np.full(len(times), np.nan),
            where=n_active_sectors > 0,
        )

        row_w = normalize_rows(active_h.astype(float) * base_weights[None, :])
        weighted_sign = (row_w * sign_h).sum(axis=1)  # equivalent to WSB in phase-1
        sgn_w = np.sign(weighted_sign)
        same_sign = ((sector_signed * sgn_w[:, None]) > 0) & sector_active_mask
        sector_concord = np.divide(
            same_sign.sum(axis=1).astype(float),
            n_active_sectors,
            out=np.full(len(times), np.nan),
            where=n_active_sectors > 0,
        )

        panel[f"Sector_Concord_{horizon}"] = np.where(valid, sector_concord, np.nan)
        panel[f"Sector_Down_Frac_{horizon}"] = np.where(valid, sector_down_frac, np.nan)
        panel[f"Sector_Up_Frac_{horizon}"] = np.where(valid, sector_up_frac, np.nan)
        panel[f"Sector_N_Active_{horizon}"] = np.where(valid, n_active_sectors, np.nan)

        # ============================================================
        # B. Weight-bucket projection
        # ============================================================
        bucket_signed = np.full((len(times), n_buckets), np.nan, dtype=float)
        for b in range(n_buckets):
            cols_mask = weight_bin_arr == b
            if cols_mask.sum() < 5:
                continue
            buc_active_h = active_h[:, cols_mask]
            buc_sign_h = sign_h[:, cols_mask]
            buc_count = buc_active_h.sum(axis=1).astype(float)
            up = ((buc_sign_h > 0) & buc_active_h).sum(axis=1)
            dn = ((buc_sign_h < 0) & buc_active_h).sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                bucket_signed[:, b] = np.divide(
                    up.astype(float) - dn.astype(float),
                    buc_count,
                    out=np.full(len(times), np.nan),
                    where=buc_count >= 5,
                )

        # qcut places larger weight ranks into larger bin indices (build_weight_bins
        # uses ranks ascending), so b=0 is bottom (smallest weights), b=n-1 is top.
        top_b = n_buckets - 1
        bot_b = 0
        bucket_top = bucket_signed[:, top_b]
        bucket_bot = bucket_signed[:, bot_b]
        panel[f"Bucket_Top_Sign_{horizon}"] = np.where(valid, bucket_top, np.nan)
        panel[f"Bucket_Bottom_Sign_{horizon}"] = np.where(valid, bucket_bot, np.nan)
        panel[f"Bucket_Penetration_{horizon}"] = np.where(valid, bucket_top - bucket_bot, np.nan)

        # OLS slope of bucket signed breadth vs centered bucket index
        x = np.arange(n_buckets, dtype=float)
        x_c = x - x.mean()
        ymean = np.nanmean(bucket_signed, axis=1, keepdims=True)
        y_c = bucket_signed - ymean
        x_c_b = x_c[None, :]
        with np.errstate(invalid="ignore"):
            xy = np.nansum(x_c_b * y_c, axis=1)
            denom = float(np.nansum(x_c * x_c))
            slope = xy / denom if denom > 0 else np.full(len(times), np.nan)
        panel[f"Bucket_Slope_{horizon}"] = np.where(valid, slope, np.nan)

        # ============================================================
        # C. Equal-weight whole-market projection
        # ============================================================
        ew_up = ((sign_h > 0) & active_h).sum(axis=1)
        ew_dn = ((sign_h < 0) & active_h).sum(axis=1)
        ew_signed = np.divide(
            ew_up.astype(float) - ew_dn.astype(float),
            active_count_h,
            out=np.full(len(times), np.nan),
            where=active_count_h >= cfg.min_active_constituents,
        )
        panel[f"EW_SignedBreadth_{horizon}"] = np.where(valid, ew_signed, np.nan)
        panel[f"EW_to_W_Penetration_{horizon}"] = np.where(valid, weighted_sign - ew_signed, np.nan)
        panel[f"WeightedSign_{horizon}"] = np.where(valid, weighted_sign, np.nan)

        # ============================================================
        # D. Wavefront (h=5 only -- shorter horizon catches new joiners)
        # ============================================================
        if horizon == 5:
            sign_prev = np.full_like(sign_h, 0.0)
            sign_prev[1:, :] = sign_h[:-1, :]
            new_down = ((sign_h < 0) & (sign_prev >= 0) & active_h).sum(axis=1).astype(float)
            new_up = ((sign_h > 0) & (sign_prev <= 0) & active_h).sum(axis=1).astype(float)
            wf_down = np.divide(
                new_down,
                active_count_h,
                out=np.full(len(times), np.nan),
                where=active_count_h > 0,
            )
            wf_up = np.divide(
                new_up,
                active_count_h,
                out=np.full(len(times), np.nan),
                where=active_count_h > 0,
            )
            panel["NewDown_5"] = np.where(valid, new_down, np.nan)
            panel["NewUp_5"] = np.where(valid, new_up, np.nan)
            panel["WavefrontDown_Frac_5"] = np.where(valid, wf_down, np.nan)
            panel["WavefrontUp_Frac_5"] = np.where(valid, wf_up, np.nan)

    return panel


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[multiproj] window {cfg.start_date}..{cfg.end_date}")
    print(f"[multiproj] stock dir: {cfg.stock_data_dir}")
    print(f"[multiproj] etf dir:   {cfg.etf_data_dir}")
    print(f"[multiproj] weights:   {cfg.daily_weights_csv}")

    weights_by_date = load_weight_table(cfg.daily_weights_csv)
    weight_dates = sorted(weights_by_date.keys())
    industry = load_industry_map(cfg.industry_map_path)
    trade_dates = list_trade_dates(
        Path(cfg.stock_data_dir),
        Path(cfg.etf_data_dir),
        cfg.start_date,
        cfg.end_date,
    )
    print(f"[multiproj] trade dates in window: {len(trade_dates)}  weights known: {len(weight_dates)}")

    panels: list[pd.DataFrame] = []
    usage_rows: list[dict] = []
    skipped: list[dict] = []

    for i, d in enumerate(trade_dates, start=1):
        trade_dt = pd.Timestamp(d).normalize()
        wdt = previous_weight_date(weight_dates, trade_dt)
        if wdt is None:
            skipped.append({"date": d, "reason": "no_previous_weight_date"})
            continue
        try:
            universe = make_universe(weights_by_date[wdt], industry, cfg.leader_n, cfg.weight_bins)
            day = compute_multiproj_day(d, universe, cfg)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"date": d, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if day is None or day.empty:
            skipped.append({"date": d, "reason": "empty_day_panel"})
            continue
        day["weight_date"] = wdt
        panels.append(day)
        usage_rows.append(
            {
                "date": d,
                "weight_date": str(wdt.date()),
                "n_constituents": int(len(universe)),
                "rows": int(len(day)),
                "active_mean": float(day["active_now_total"].mean()),
            }
        )
        if i <= 3 or i == len(trade_dates) or i % 25 == 0:
            print(f"[multiproj] {i}/{len(trade_dates)} {d} weight={wdt.date()} rows={len(day)}")

    if not panels:
        raise RuntimeError(f"no multiproj panels built; skipped={skipped[:5]}")

    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "minute_idx"]).reset_index(drop=True)
    panel_path = out_dir / "multiproj_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    usage = pd.DataFrame(usage_rows)
    usage.to_csv(out_dir / "weight_usage.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_dates.csv", index=False)

    summary = {
        "config": asdict(cfg),
        "rows": int(len(panel)),
        "days": int(panel["date"].nunique()),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
        "skipped_count": int(len(skipped)),
        "feature_cols": [c for c in panel.columns if c not in ("date", "datetime", "minute_idx", "weight_date", "active_now_total")],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Saved: {panel_path.resolve()}")


if __name__ == "__main__":
    main()
