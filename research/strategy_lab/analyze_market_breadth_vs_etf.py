#!/usr/bin/env python3
"""
Analyze market-wide minute breadth versus core ETF minute returns.

Breadth definition per minute:
- For each stock on a given minute, sign(close - open):
  +1 if positive, -1 if negative, 0 otherwise
- Aggregate over all stocks to get:
  breadth_sum = up_count - down_count
  breadth_ratio = breadth_sum / total_count

Experiments:
1. Same-minute correlation between breadth and ETF returns.
2. Lead-lag correlation between breadth features and future ETF returns.
3. Simple single-factor quantile spread tests for future 1-5 minute ETF returns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path("/Users/daweizhong/Documents/projects")
STOCK_DIR = ROOT / "stock_data"
ETF_DIR = ROOT / "ETF data core7"
OUTPUT_DIR = ROOT / "artifacts" / "market_breadth_vs_etf"

ETF_CODES = [
    "510300.XSHG",
    "510500.XSHG",
    "510050.XSHG",
]


@dataclass
class Config:
    stock_dir: str = str(STOCK_DIR)
    etf_dir: str = str(ETF_DIR)
    output_dir: str = str(OUTPUT_DIR)
    start_date: str = "2020-01-01"
    end_date: str = "2026-03-05"
    max_days: int = 0
    quantiles: int = 5
    min_obs_per_day: int = 80


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Analyze market breadth vs ETF minute returns")
    parser.add_argument("--stock-dir", default=str(STOCK_DIR))
    parser.add_argument("--etf-dir", default=str(ETF_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2026-03-05")
    parser.add_argument("--max-days", type=int, default=0)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-obs-per-day", type=int, default=80)
    return Config(**vars(parser.parse_args()))


def select_dates(cfg: Config) -> List[str]:
    stock_dates = {p.stem for p in Path(cfg.stock_dir).glob("*.parquet")}
    etf_dates = {p.stem for p in Path(cfg.etf_dir).glob("*.parquet")}
    dates = sorted(stock_dates & etf_dates)
    dates = [d for d in dates if cfg.start_date <= d <= cfg.end_date]
    if cfg.max_days > 0:
        dates = dates[-cfg.max_days :]
    return dates


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    if xv.nunique() <= 1 or yv.nunique() <= 1:
        return np.nan
    return float(xv.corr(yv))


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    xv = x[mask]
    yv = y[mask]
    if xv.nunique() <= 1 or yv.nunique() <= 1:
        return np.nan
    stat = spearmanr(xv, yv).statistic
    return float(stat) if np.isfinite(stat) else np.nan


def build_breadth_day(stock_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(stock_path, columns=["datetime", "open", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    diff = df["close"] - df["open"]
    df["sign"] = 0
    df.loc[diff > 0, "sign"] = 1
    df.loc[diff < 0, "sign"] = -1
    grouped = df.groupby("datetime", sort=True)["sign"].agg(["sum", "count"])
    up = df.loc[df["sign"] == 1].groupby("datetime").size()
    down = df.loc[df["sign"] == -1].groupby("datetime").size()
    flat = df.loc[df["sign"] == 0].groupby("datetime").size()
    out = grouped.rename(columns={"sum": "breadth_sum", "count": "stock_count"})
    out["up_count"] = up
    out["down_count"] = down
    out["flat_count"] = flat
    out = out.fillna(0).astype(int).reset_index()
    out["breadth_ratio"] = out["breadth_sum"] / out["stock_count"].replace(0, np.nan)
    out["breadth_up_ratio"] = out["up_count"] / out["stock_count"].replace(0, np.nan)
    out["breadth_down_ratio"] = out["down_count"] / out["stock_count"].replace(0, np.nan)
    out["breadth_diff_1"] = out["breadth_sum"].diff()
    out["breadth_ratio_diff_1"] = out["breadth_ratio"].diff()
    out["breadth_sum_ma3"] = out["breadth_sum"].rolling(3, min_periods=1).mean()
    out["breadth_sum_ma5"] = out["breadth_sum"].rolling(5, min_periods=1).mean()
    out["breadth_ratio_ma3"] = out["breadth_ratio"].rolling(3, min_periods=1).mean()
    out["breadth_ratio_ma5"] = out["breadth_ratio"].rolling(5, min_periods=1).mean()
    return out


def build_etf_day(etf_path: Path, code: str) -> pd.DataFrame:
    df = pd.read_parquet(etf_path, columns=["code", "datetime", "open", "close"])
    df = df[df["code"] == code].copy()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    df["ret_oc"] = np.log(df["close"] / df["open"])
    df["ret_cc_1"] = np.log(df["close"] / df["close"].shift(1))
    for h in range(1, 6):
        df[f"fwd_ret_{h}m"] = np.log(df["close"].shift(-h) / df["close"])
        df[f"fwd_dir_{h}m"] = np.sign(df[f"fwd_ret_{h}m"])
    return df


def build_panel(dates: Iterable[str], cfg: Config) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for i, date_str in enumerate(dates, start=1):
        breadth = build_breadth_day(Path(cfg.stock_dir) / f"{date_str}.parquet")
        if breadth.empty:
            continue
        breadth["date"] = pd.to_datetime(date_str)
        for code in ETF_CODES:
            etf = build_etf_day(Path(cfg.etf_dir) / f"{date_str}.parquet", code)
            if etf.empty:
                continue
            merged = breadth.merge(etf, on="datetime", how="inner")
            if merged.empty:
                continue
            merged["date"] = pd.to_datetime(date_str)
            merged["etf_code"] = code
            merged["minute_idx"] = np.arange(len(merged), dtype=int)
            frames.append(merged)
        if i % 200 == 0:
            print(f"processed {i} days")
    if not frames:
        raise RuntimeError("No panel built")
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["etf_code", "date", "datetime"]).reset_index(drop=True)


def summarize_same_minute(panel: pd.DataFrame) -> pd.DataFrame:
    features = [
        "breadth_sum",
        "breadth_ratio",
        "breadth_diff_1",
        "breadth_ratio_diff_1",
        "breadth_sum_ma3",
        "breadth_sum_ma5",
        "breadth_ratio_ma3",
        "breadth_ratio_ma5",
    ]
    rows: List[dict] = []
    for code, grp in panel.groupby("etf_code"):
        for feat in features:
            rows.append(
                {
                    "etf_code": code,
                    "feature": feat,
                    "corr_with_ret_oc": safe_corr(grp[feat], grp["ret_oc"]),
                    "corr_with_ret_cc_1": safe_corr(grp[feat], grp["ret_cc_1"]),
                    "spearman_with_ret_oc": safe_spearman(grp[feat], grp["ret_oc"]),
                    "spearman_with_ret_cc_1": safe_spearman(grp[feat], grp["ret_cc_1"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_forward(panel: pd.DataFrame) -> pd.DataFrame:
    features = [
        "breadth_sum",
        "breadth_ratio",
        "breadth_diff_1",
        "breadth_ratio_diff_1",
        "breadth_sum_ma3",
        "breadth_sum_ma5",
        "breadth_ratio_ma3",
        "breadth_ratio_ma5",
    ]
    rows: List[dict] = []
    for code, grp in panel.groupby("etf_code"):
        for feat in features:
            for h in range(1, 6):
                target = f"fwd_ret_{h}m"
                dir_target = f"fwd_dir_{h}m"
                signed = np.sign(grp[feat]) * grp[target]
                hit = ((np.sign(grp[feat]) == np.sign(grp[target])) & grp[target].notna() & grp[feat].notna()).mean()
                rows.append(
                    {
                        "etf_code": code,
                        "feature": feat,
                        "horizon_min": h,
                        "corr_with_fwd_ret": safe_corr(grp[feat], grp[target]),
                        "spearman_with_fwd_ret": safe_spearman(grp[feat], grp[target]),
                        "mean_signed_fwd_bps": float(signed.mean() * 10000.0),
                        "hit_rate": float(hit),
                        "pos_rate": float((grp[dir_target] > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def summarize_quantiles(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    features = [
        "breadth_sum",
        "breadth_ratio",
        "breadth_diff_1",
        "breadth_ratio_diff_1",
        "breadth_sum_ma3",
        "breadth_sum_ma5",
        "breadth_ratio_ma3",
        "breadth_ratio_ma5",
    ]
    rows: List[dict] = []
    for code, code_df in panel.groupby("etf_code"):
        for feat in features:
            for h in range(1, 6):
                ret_col = f"fwd_ret_{h}m"
                daily_spreads = []
                daily_ic = []
                for _, day in code_df.groupby("date"):
                    x = day[feat]
                    y = day[ret_col]
                    mask = x.notna() & y.notna()
                    if mask.sum() < cfg.min_obs_per_day:
                        continue
                    xv = x[mask]
                    yv = y[mask]
                    if xv.nunique() < cfg.quantiles:
                        continue
                    ic = safe_spearman(xv, yv)
                    if np.isfinite(ic):
                        daily_ic.append(ic)
                    labels = pd.qcut(xv.rank(method="first"), cfg.quantiles, labels=[f"Q{i+1}" for i in range(cfg.quantiles)])
                    qret = yv.groupby(labels).mean()
                    daily_spreads.append(float(qret.get(f"Q{cfg.quantiles}", np.nan) - qret.get("Q1", np.nan)))
                if not daily_spreads:
                    continue
                spread = pd.Series(daily_spreads, dtype=float)
                ic_s = pd.Series(daily_ic, dtype=float)
                rows.append(
                    {
                        "etf_code": code,
                        "feature": feat,
                        "horizon_min": h,
                        "daily_ic_mean": float(ic_s.mean()) if len(ic_s) else np.nan,
                        "daily_icir": float(ic_s.mean() / ic_s.std(ddof=0)) if len(ic_s) and ic_s.std(ddof=0) > 0 else np.nan,
                        "spread_mean_bps": float(spread.mean() * 10000.0),
                        "spread_sharpe": float(spread.mean() / spread.std(ddof=0) * np.sqrt(242)) if spread.std(ddof=0) > 0 else np.nan,
                        "n_days": int(len(spread)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = select_dates(cfg)
    print(f"selected_dates={len(dates)}")
    panel = build_panel(dates, cfg)
    panel.to_parquet(out_dir / "panel.parquet", index=False)

    same_minute = summarize_same_minute(panel)
    forward = summarize_forward(panel)
    quantiles = summarize_quantiles(panel, cfg)

    same_minute.to_csv(out_dir / "same_minute_corr.csv", index=False)
    forward.to_csv(out_dir / "forward_corr.csv", index=False)
    quantiles.to_csv(out_dir / "quantile_summary.csv", index=False)

    best_forward = (
        quantiles.sort_values(["spread_mean_bps", "daily_icir"], ascending=False)
        .groupby("etf_code", as_index=False)
        .head(10)
    )
    best_forward.to_csv(out_dir / "best_quantile_rules.csv", index=False)

    report = {
        "config": asdict(cfg),
        "sample": {
            "n_dates": len(dates),
            "date_start": dates[0] if dates else None,
            "date_end": dates[-1] if dates else None,
            "n_rows": int(len(panel)),
        },
        "artifacts": {
            "panel": str((out_dir / "panel.parquet").resolve()),
            "same_minute_corr": str((out_dir / "same_minute_corr.csv").resolve()),
            "forward_corr": str((out_dir / "forward_corr.csv").resolve()),
            "quantile_summary": str((out_dir / "quantile_summary.csv").resolve()),
            "best_quantile_rules": str((out_dir / "best_quantile_rules.csv").resolve()),
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
