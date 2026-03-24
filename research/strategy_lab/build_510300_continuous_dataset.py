#!/usr/bin/env python3
"""
Build a continuous 510300 minute dataset.

Behavior:
1. For dates that exist in the original ETF directory, extract real 510300 rows.
2. For later dates where ETF data is missing, synthesize a 510300 proxy from stock
   minute data using a daily float-mcap proxy universe.

The output directory contains one parquet per trading date with the same schema the
research scripts already expect, but only for `510300.XSHG`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from explore14_hs300_propagation_strategy import (
    ROOT_DIR,
    ETF_DATA_DIR,
    STOCK_DATA_DIR,
    StrategyConfig,
    build_dynamic_proxy_universe,
    load_mcap_panel,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "ETF data 510300 continuous"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a continuous 510300 minute dataset")
    p.add_argument("--stock-data-dir", type=str, default=str(STOCK_DATA_DIR))
    p.add_argument("--etf-data-dir", type=str, default=str(ETF_DATA_DIR))
    p.add_argument(
        "--daily-mcap-panel-path",
        type=str,
        default=str(
            ROOT_DIR / "artifacts" / "ashare_t1_xgb_stfree_mcap10_500_v2_fullrun" / "feature_table_stfree_mcap_advanced.parquet"
        ),
    )
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--target-code", type=str, default="510300.XSHG")
    p.add_argument("--start-date", type=str, default="2010-01-04")
    p.add_argument("--end-date", type=str, default="2026-03-05")
    p.add_argument("--proxy-top-n", type=int, default=320)
    p.add_argument("--leader-n", type=int, default=30)
    p.add_argument("--weight-bins", type=int, default=5)
    p.add_argument("--seed-close", type=float, default=np.nan)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def select_dates(stock_dir: Path, start_date: str, end_date: str) -> List[str]:
    dates = sorted(p.stem for p in stock_dir.glob("*.parquet"))
    return [d for d in dates if start_date <= d <= end_date]


def extract_real_510300(etf_path: Path, target_code: str) -> pd.DataFrame:
    df = pd.read_parquet(etf_path)
    df = df[df["code"] == target_code].copy()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def _pivot_field(stock_df: pd.DataFrame, field: str, codes: Iterable[str]) -> pd.DataFrame:
    return stock_df.pivot_table(index="datetime", columns="code", values=field).reindex(columns=list(codes))


def synthesize_proxy_day(
    stock_path: Path,
    date_str: str,
    prev_close: float,
    cfg: StrategyConfig,
    mcap_panel: pd.DataFrame,
    target_code: str,
) -> pd.DataFrame:
    universe = build_dynamic_proxy_universe(date_str, mcap_panel, pd.DataFrame(columns=["code", "industry"]), cfg)
    if universe.empty:
        raise ValueError(f"No proxy universe for {date_str}")
    codes = universe["code"].astype(str).tolist()
    table = pq.read_table(
        stock_path,
        columns=["code", "datetime", "open", "high", "low", "close", "money", "volume", "paused", "pre_close"],
        filters=[("code", "in", codes)],
    )
    stock_df = table.to_pandas()
    stock_df["datetime"] = pd.to_datetime(stock_df["datetime"])
    codes = [c for c in codes if c in set(stock_df["code"])]
    if len(codes) < 150:
        raise ValueError(f"Too few proxy names on {date_str}: {len(codes)}")

    weights = universe.set_index("code").reindex(codes)["weight"].fillna(0.0)
    weights = weights / weights.sum()
    weight_vec = weights.to_numpy(dtype=float)

    open_df = _pivot_field(stock_df, "open", codes)
    high_df = _pivot_field(stock_df, "high", codes)
    low_df = _pivot_field(stock_df, "low", codes)
    close_df = _pivot_field(stock_df, "close", codes)
    money_df = _pivot_field(stock_df, "money", codes).fillna(0.0)
    volume_df = _pivot_field(stock_df, "volume", codes).fillna(0.0)
    paused_df = _pivot_field(stock_df, "paused", codes).fillna(0.0)
    pre_close_df = _pivot_field(stock_df, "pre_close", codes)

    common_times = close_df.index
    open_arr = open_df.to_numpy(dtype=float)
    high_arr = high_df.to_numpy(dtype=float)
    low_arr = low_df.to_numpy(dtype=float)
    close_arr = close_df.to_numpy(dtype=float)
    money_arr = money_df.to_numpy(dtype=float)
    volume_arr = volume_df.to_numpy(dtype=float)
    paused_arr = paused_df.to_numpy(dtype=float)
    pre_close_arr = pre_close_df.to_numpy(dtype=float)

    valid = np.isfinite(pre_close_arr) & (pre_close_arr > 0)
    norm_open = np.divide(open_arr, pre_close_arr, out=np.full_like(open_arr, np.nan), where=valid)
    norm_high = np.divide(high_arr, pre_close_arr, out=np.full_like(high_arr, np.nan), where=valid)
    norm_low = np.divide(low_arr, pre_close_arr, out=np.full_like(low_arr, np.nan), where=valid)
    norm_close = np.divide(close_arr, pre_close_arr, out=np.full_like(close_arr, np.nan), where=valid)

    active = np.isfinite(norm_close)
    row_weights = active.astype(float) * weight_vec[None, :]
    row_sum = row_weights.sum(axis=1, keepdims=True)
    row_weights = np.divide(row_weights, row_sum, out=np.zeros_like(row_weights), where=row_sum > 0)

    basket_open = prev_close * np.nansum(row_weights * norm_open, axis=1)
    basket_high = prev_close * np.nansum(row_weights * norm_high, axis=1)
    basket_low = prev_close * np.nansum(row_weights * norm_low, axis=1)
    basket_close = prev_close * np.nansum(row_weights * norm_close, axis=1)
    basket_money = np.nansum(row_weights * money_arr, axis=1)
    basket_volume = np.nansum(row_weights * volume_arr, axis=1)
    basket_avg = np.copy(basket_close)
    basket_paused = np.where((row_weights * (paused_arr > 0).astype(float)).sum(axis=1) > 0.95, 1.0, 0.0)

    out = pd.DataFrame(
        {
            "code": target_code,
            "datetime": common_times,
            "open": basket_open,
            "high": basket_high,
            "low": basket_low,
            "close": basket_close,
            "money": basket_money,
            "volume": basket_volume,
            "factor": 1.0,
            "high_limit": prev_close * 1.10,
            "low_limit": prev_close * 0.90,
            "avg": basket_avg,
            "pre_close": prev_close,
            "paused": basket_paused,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    stock_dir = Path(args.stock_data_dir)
    etf_dir = Path(args.etf_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = StrategyConfig(
        stock_data_dir=str(stock_dir),
        etf_data_dir=str(etf_dir),
        daily_mcap_panel_path=args.daily_mcap_panel_path,
        proxy_top_n=args.proxy_top_n,
        leader_n=args.leader_n,
        weight_bins=args.weight_bins,
    )
    mcap_panel = load_mcap_panel(args.daily_mcap_panel_path, start_date=args.start_date, end_date=args.end_date)

    dates = select_dates(stock_dir, args.start_date, args.end_date)
    summary: Dict[str, int | str | float] = {
        "target_code": args.target_code,
        "date_start": dates[0] if dates else "",
        "date_end": dates[-1] if dates else "",
        "n_dates": len(dates),
        "n_real_days": 0,
        "n_proxy_days": 0,
        "n_skipped": 0,
    }

    prev_close = np.nan
    for idx, date_str in enumerate(dates, start=1):
        out_path = output_dir / f"{date_str}.parquet"
        if out_path.exists() and not args.overwrite:
            existing = pd.read_parquet(out_path, columns=["close"])
            if not existing.empty:
                prev_close = float(existing["close"].iloc[-1])
            continue

        real_df = pd.DataFrame()
        etf_path = etf_dir / f"{date_str}.parquet"
        if etf_path.exists():
            real_df = extract_real_510300(etf_path, args.target_code)

        if not real_df.empty:
            real_df.to_parquet(out_path, index=False)
            prev_close = float(real_df["close"].iloc[-1])
            summary["n_real_days"] = int(summary["n_real_days"]) + 1
        else:
            if not np.isfinite(prev_close):
                if np.isfinite(args.seed_close):
                    prev_close = float(args.seed_close)
                else:
                    summary["n_skipped"] = int(summary["n_skipped"]) + 1
                    continue
            try:
                proxy_df = synthesize_proxy_day(
                    stock_path=stock_dir / f"{date_str}.parquet",
                    date_str=date_str,
                    prev_close=prev_close,
                    cfg=cfg,
                    mcap_panel=mcap_panel,
                    target_code=args.target_code,
                )
            except Exception:
                summary["n_skipped"] = int(summary["n_skipped"]) + 1
                continue
            if proxy_df.empty:
                summary["n_skipped"] = int(summary["n_skipped"]) + 1
                continue
            proxy_df.to_parquet(out_path, index=False)
            prev_close = float(proxy_df["close"].iloc[-1])
            summary["n_proxy_days"] = int(summary["n_proxy_days"]) + 1

        if idx <= 5 or idx % 250 == 0:
            print(f"[{idx}/{len(dates)}] {date_str} prev_close={prev_close:.4f}")

    summary_path = output_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": str(output_dir.resolve()), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
