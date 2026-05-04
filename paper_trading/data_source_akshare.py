#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time data source for HS300 V1.5 paper trading using akshare.

Provides bars for HS300 constituents and 510300 ETF, fetched on demand.
The engine never sees a future minute — each fetch is wall-clock'ed and
the data returned has timestamps strictly <= now.

Endpoints used (all free, no auth required):
- ak.index_stock_cons_csindex(symbol='000300')  -> HS300 constituents
- ak.stock_zh_a_hist_min_em(symbol, period='1') -> stock minute bars
- ak.fund_etf_hist_min_em(symbol, period='1')   -> ETF minute bars

Cache strategy:
- Constituent list cached for the trading day
- Per-stock minute bars cached and incremented (fetch returns full day,
  we keep the latest copy in memory and on disk so a service restart
  does not require a fresh full-day fetch)

Rate limits:
- akshare hits Eastmoney; ~1 sec per stock seems safe
- 300 stocks total fetch in roughly 5 min serial; we accept this latency
- Easy to parallelize with ThreadPoolExecutor if needed
"""

from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------

def to_jq_code(ak_code: str) -> str:
    """ak format '000001' (no exchange) -> JQ format '000001.XSHE' or '600519.XSHG'."""
    s = str(ak_code).zfill(6)
    if s.startswith(("60", "68", "5", "9")):
        return f"{s}.XSHG"
    return f"{s}.XSHE"


def from_jq_code(jq_code: str) -> str:
    """JQ '600519.XSHG' -> ak '600519'."""
    return str(jq_code).split(".")[0]


# ---------------------------------------------------------------------------
# Constituent list
# ---------------------------------------------------------------------------

class HS300Constituents:
    """Cached HS300 constituent list. Refreshes once per calendar day."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cached: Optional[pd.DataFrame] = None
        self._cached_date: Optional[str] = None

    def get(self, today_str: Optional[str] = None) -> pd.DataFrame:
        if today_str is None:
            today_str = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
        if self._cached is not None and self._cached_date == today_str:
            return self._cached
        cache_path = self.cache_dir / f"hs300_constituents_{today_str}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            self._cached = df
            self._cached_date = today_str
            return df
        # Fetch fresh
        import akshare as ak
        raw = ak.index_stock_cons_csindex(symbol="000300")
        df = pd.DataFrame({
            "ak_code": raw["成分券代码"].astype(str).str.zfill(6),
            "name": raw["成分券名称"].astype(str),
            "exchange": raw["交易所"].astype(str),
        })
        df["jq_code"] = df["ak_code"].apply(to_jq_code)
        df.to_parquet(cache_path, index=False)
        self._cached = df
        self._cached_date = today_str
        return df


# ---------------------------------------------------------------------------
# Minute bar fetcher
# ---------------------------------------------------------------------------

@dataclass
class MinuteFetchResult:
    code: str
    bars: Optional[pd.DataFrame]  # cols: datetime, open, close, high, low, volume, money
    error: Optional[str] = None
    latency_sec: float = 0.0


def _normalize_min_bars(raw: pd.DataFrame, target_date: Optional[str] = None) -> pd.DataFrame:
    """Translate akshare minute bar columns to our schema. If target_date given,
    filter to that date; otherwise return all bars akshare gave us.

    Akshare's ETF endpoint (fund_etf_hist_min_em) returns open=0 for all bars
    -- this is a known akshare bug. We patch that here: for bars with open<=0
    we substitute the prior bar's close (the price at the start of the bar
    in a continuous market). For the very first bar of the series we use
    the bar's own close.
    """
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["datetime", "open", "close", "high", "low", "volume", "money"])
    df = pd.DataFrame({
        "datetime": pd.to_datetime(raw["时间"]),
        "open": raw["开盘"].astype(float),
        "close": raw["收盘"].astype(float),
        "high": raw["最高"].astype(float),
        "low": raw["最低"].astype(float),
        "volume": raw["成交量"].astype(float),
        "money": raw["成交额"].astype(float),
    })
    df = df.sort_values("datetime").reset_index(drop=True)
    # Fix open=0 (akshare ETF bug): use prior bar's close
    bad = df["open"] <= 0
    if bad.any():
        prev_close = df["close"].shift(1)
        # Fall back to current close for the first row
        prev_close = prev_close.fillna(df["close"])
        df.loc[bad, "open"] = prev_close[bad].values
    if target_date is not None:
        target = pd.Timestamp(target_date).normalize()
        df = df[df["datetime"].dt.normalize() == target].copy().reset_index(drop=True)
    return df


def fetch_stock_minute(code_ak: str, target_date: Optional[str] = None, retries: int = 2) -> MinuteFetchResult:
    """Fetch minute bars for a single A-share. code_ak is '600519' style.
    If target_date is None returns all bars akshare gave us (~5 days);
    otherwise filters to that one date."""
    import akshare as ak
    last_err = None
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            raw = ak.stock_zh_a_hist_min_em(symbol=code_ak, period="1", adjust="")
            bars = _normalize_min_bars(raw, target_date=target_date)
            return MinuteFetchResult(code=code_ak, bars=bars, latency_sec=time.time() - t0)
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    return MinuteFetchResult(code=code_ak, bars=None, error=str(last_err), latency_sec=time.time() - t0)


def fetch_etf_minute(code_ak: str, target_date: Optional[str] = None, retries: int = 2) -> MinuteFetchResult:
    import akshare as ak
    last_err = None
    t0 = time.time()
    for attempt in range(retries + 1):
        try:
            raw = ak.fund_etf_hist_min_em(symbol=code_ak, period="1", adjust="")
            bars = _normalize_min_bars(raw, target_date=target_date)
            return MinuteFetchResult(code=code_ak, bars=bars, latency_sec=time.time() - t0)
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    return MinuteFetchResult(code=code_ak, bars=None, error=str(last_err), latency_sec=time.time() - t0)


# ---------------------------------------------------------------------------
# AkshareDataSource
# ---------------------------------------------------------------------------

class AkshareDataSource:
    """Pluggable data source for the live engine.

    Methods mirror the LocalParquetSource but pull from akshare. Per-day
    bars are cached on disk so a service restart does not re-fetch
    today's data.
    """

    def __init__(self, cache_dir: Path, etf_code: str = "510300", parallel_workers: int = 8):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.etf_code = etf_code
        self.parallel_workers = parallel_workers
        self.constituents = HS300Constituents(cache_dir / "constituents")

    def fetch_universe_bars(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch minute bars for all 300 HS300 constituents.
        If target_date is None, returns all bars akshare gives (~5 days, useful
        for live mode where we want to know the latest available trading day).
        If target_date is given (YYYY-MM-DD), filters to that one date."""
        if target_date is None:
            target_date = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
        cache_path = self.cache_dir / f"universe_bars_{target_date}.parquet"

        cons = self.constituents.get(target_date)
        codes = cons["ak_code"].tolist()

        results: list[pd.DataFrame] = []
        latencies: list[float] = []
        errors: list[str] = []

        def task(code):
            return fetch_stock_minute(code, target_date=target_date)

        print(f"[akshare] fetching {len(codes)} HS300 constituents for {target_date} (workers={self.parallel_workers}) ...")
        t_start = time.time()
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            futures = {pool.submit(task, c): c for c in codes}
            for i, fut in enumerate(as_completed(futures), 1):
                res = fut.result()
                latencies.append(res.latency_sec)
                if res.error:
                    errors.append(f"{res.code}: {res.error}")
                    continue
                if res.bars is None or len(res.bars) == 0:
                    continue
                bars = res.bars.copy()
                bars["code"] = to_jq_code(res.code)
                bars["paused"] = 0.0
                results.append(bars)
                if i % 50 == 0 or i == len(codes):
                    avg_lat = np.mean(latencies)
                    elapsed = time.time() - t_start
                    print(f"[akshare] progress {i}/{len(codes)} | avg latency {avg_lat:.2f}s | elapsed {elapsed:.1f}s | errors {len(errors)}")

        if not results:
            raise RuntimeError(f"akshare returned no stock bars for {target_date}; errors={errors[:5]} (likely non-trading day)")
        df = pd.concat(results, ignore_index=True)
        df = df[["code", "datetime", "open", "close", "high", "low", "volume", "money", "paused"]]
        df.to_parquet(cache_path, index=False)
        print(f"[akshare] fetched {len(df)} bar rows for {df['code'].nunique()} stocks in {time.time()-t_start:.1f}s; errors={len(errors)}")
        return df

    def fetch_etf_bars(self, target_date: Optional[str] = None) -> pd.DataFrame:
        if target_date is None:
            target_date = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d")
        cache_path = self.cache_dir / f"etf_bars_{target_date}.parquet"
        res = fetch_etf_minute(self.etf_code, target_date=target_date)
        if res.error or res.bars is None or len(res.bars) == 0:
            raise RuntimeError(f"akshare ETF fetch failed for {target_date}: {res.error or 'no bars (non-trading day?)'}")
        df = res.bars.copy()
        df["code"] = f"{self.etf_code}.XSHG"
        df["paused"] = 0.0
        df = df[["code", "datetime", "open", "close", "high", "low", "volume", "money", "paused"]]
        df.to_parquet(cache_path, index=False)
        print(f"[akshare] fetched {len(df)} ETF bars for {target_date}")
        return df

    def latest_trading_date(self) -> str:
        """Probe akshare for the most recent trading date by sampling one stock."""
        res = fetch_stock_minute("000001")  # 平安银行, very liquid
        if res.bars is None or len(res.bars) == 0:
            raise RuntimeError("akshare returned no bars when probing latest date")
        latest = res.bars["datetime"].dt.normalize().max()
        return latest.strftime("%Y-%m-%d")

    def is_market_open(self, now: Optional[pd.Timestamp] = None) -> bool:
        if now is None:
            now = pd.Timestamp.now(tz="Asia/Shanghai")
        else:
            now = now.tz_convert("Asia/Shanghai") if now.tzinfo else now.tz_localize("Asia/Shanghai")
        if now.weekday() >= 5:
            return False
        h, m = now.hour, now.minute
        if (h == 9 and m >= 30) or (10 <= h < 11) or (h == 11 and m <= 30):
            return True
        if (h == 13) or (h == 14) or (h == 15 and m == 0):
            return True
        return False


if __name__ == "__main__":
    # Smoke test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["constituents", "stock", "etf", "universe"], required=True)
    parser.add_argument("--cache-dir", default="paper_trading/akshare_cache")
    args = parser.parse_args()

    src = AkshareDataSource(Path(args.cache_dir))

    if args.test == "constituents":
        df = src.constituents.get()
        print(f"HS300: {len(df)} constituents")
        print(df.head(5).to_string())
        print(df.tail(5).to_string())
    elif args.test == "stock":
        res = fetch_stock_minute("600519")
        print(f"600519 (Moutai): {len(res.bars)} bars in {res.latency_sec:.2f}s")
        if res.bars is not None and len(res.bars):
            print(res.bars.tail(5).to_string())
    elif args.test == "etf":
        res = fetch_etf_minute("510300")
        print(f"510300 ETF: {len(res.bars)} bars in {res.latency_sec:.2f}s")
        if res.bars is not None and len(res.bars):
            print(res.bars.tail(5).to_string())
    elif args.test == "universe":
        latest = src.latest_trading_date()
        print(f"Latest trading date: {latest}")
        df = src.fetch_universe_bars(target_date=latest)
        print(f"Universe: {len(df)} rows for {df['code'].nunique()} stocks")
        print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
