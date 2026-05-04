#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict downside walk-forward research for the HS300 propagation thesis.

This script is deliberately conservative:
- ETF minute prices must come from the high-precision rebuilt directory.
- Constituents and weights are lagged one trading day.
- Same-minute z-scores are computed only from previous days.
- Trade entries occur on the next minute open after a completed signal bar.
- Candidate filters are fit on the training window and then frozen for OOS.

The search space is intentionally restricted to the interpretable edge that
survived the ETF precision audit: downside propagation over roughly 20-60
minutes, not generic five-minute direction chasing.
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
    compute_day_features,
    load_industry_map,
    regular_signal_mask,
    score_panel,
)


PROJECT_ROOT = Path("/Users/daweizhong/Documents/projects")
REPO_ROOT = Path(__file__).resolve().parents[2]


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
        / "hs300_downside_wf_daily_lag_precise_v1"
    )
    target_etf_code: str = "510300.XSHG"
    start_date: str = "2025-01-03"
    end_date: str = "2026-04-30"
    oos_start: str = "2026-03-06"
    reuse_panel: bool = False
    input_panel_files: str = ""
    use_is_eval_only: bool = True
    panel_file: str = "scored_panel_daily_lag.parquet"
    amount_min: float = 100000.0
    min_active_constituents: int = 120
    round_trip_cost_bps: float = 4.5
    initial_capital: float = 500000.0
    if_margin_rate: float = 0.08
    if_utilization: float = 0.80
    if_multiplier: float = 300.0
    etf_to_index_scale: float = 1000.0
    min_train_trades: int = 24
    wf_train_days: int = 120
    wf_test_days: int = 20
    max_trades_per_day: int = 2


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="HS300 downside propagation strict walk-forward")
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


def list_trade_dates(stock_dir: Path, etf_dir: Path, start: str, end: str) -> List[str]:
    stock_dates = {p.stem for p in stock_dir.glob("*.parquet")}
    etf_dates = {p.stem for p in etf_dir.glob("*.parquet")}
    dates = sorted(d for d in stock_dates & etf_dates if start <= d <= end)
    if not dates:
        raise ValueError(f"no overlapping stock/ETF dates in {start}..{end}")
    return dates


def load_weight_table(path: str) -> dict[pd.Timestamp, pd.DataFrame]:
    df = pd.read_csv(path)
    needed = {"date", "code", "weight_pct"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"daily weights missing columns: {missing}")
    df = df[["date", "code", "weight_pct"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = df["code"].astype(str)
    df["weight_pct"] = df["weight_pct"].astype(float)
    return {d: g.drop(columns=["date"]).reset_index(drop=True) for d, g in df.groupby("date", sort=True)}


def previous_weight_date(weight_dates: list[pd.Timestamp], trade_date: pd.Timestamp) -> pd.Timestamp | None:
    # 320 dates only, so a simple reverse scan is clearer than keeping another dependency.
    for d in reversed(weight_dates):
        if d < trade_date:
            return d
    return None


def make_universe(weight_df: pd.DataFrame, industry: pd.DataFrame, leader_n: int, weight_bins: int) -> pd.DataFrame:
    w = weight_df[["code", "weight_pct"]].copy()
    w["code"] = w["code"].astype(str)
    w["weight"] = w["weight_pct"].astype(float) / 100.0
    w = w[w["weight"] > 0].copy()
    if w.empty:
        raise ValueError("empty positive-weight universe")
    w["weight"] = w["weight"] / w["weight"].sum()
    if not industry.empty:
        w = w.merge(industry, on="code", how="left")
    w["industry"] = w.get("industry", pd.Series("Unknown", index=w.index)).fillna("Unknown").astype(str)
    w = w.sort_values("weight", ascending=False).reset_index(drop=True)
    w["leader_flag"] = 0
    w.loc[: max(int(leader_n) - 1, 0), "leader_flag"] = 1
    w["weight_bin"] = build_weight_bins(w["weight"], int(weight_bins))
    return w[["code", "weight", "industry", "leader_flag", "weight_bin"]]


def build_or_load_panel(cfg: Config, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cfg.input_panel_files.strip():
        parts: list[pd.DataFrame] = []
        for raw in cfg.input_panel_files.split(","):
            path = Path(raw.strip())
            if not path.is_absolute():
                path = REPO_ROOT / path
            part = pd.read_parquet(path)
            if cfg.use_is_eval_only and "is_eval" in part.columns:
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
        usage = pd.DataFrame(
            {
                "source": [str(p) for p in cfg.input_panel_files.split(",")],
                "mode": ["input_panel_files"] * len(cfg.input_panel_files.split(",")),
            }
        )
        out_path = out_dir / cfg.panel_file
        panel.to_parquet(out_path, index=False)
        return panel, usage

    panel_path = out_dir / cfg.panel_file
    usage_path = out_dir / "weight_usage.csv"
    if cfg.reuse_panel and panel_path.exists() and usage_path.exists():
        return pd.read_parquet(panel_path), pd.read_csv(usage_path)

    feature_cfg = FeatureConfig(
        stock_data_dir=cfg.stock_data_dir,
        etf_data_dir=cfg.etf_data_dir,
        industry_map_path=cfg.industry_map_path,
        output_dir=str(out_dir),
        target_etf_code=cfg.target_etf_code,
        amount_min=cfg.amount_min,
        min_active_constituents=cfg.min_active_constituents,
    )
    weights_by_date = load_weight_table(cfg.daily_weights_csv)
    weight_dates = sorted(weights_by_date.keys())
    industry = load_industry_map(cfg.industry_map_path)
    trade_dates = list_trade_dates(Path(cfg.stock_data_dir), Path(cfg.etf_data_dir), cfg.start_date, cfg.end_date)

    raw_panels: list[pd.DataFrame] = []
    usage_rows: list[dict] = []
    skipped: list[dict] = []
    for i, d in enumerate(trade_dates, start=1):
        trade_dt = pd.Timestamp(d).normalize()
        wdt = previous_weight_date(weight_dates, trade_dt)
        if wdt is None:
            skipped.append({"date": d, "reason": "no_previous_weight_date"})
            continue
        try:
            universe = make_universe(weights_by_date[wdt], industry, feature_cfg.leader_n, feature_cfg.weight_bins)
            day = compute_day_features(d, universe, feature_cfg)
        except Exception as exc:
            skipped.append({"date": d, "reason": str(exc)})
            continue
        if day is None or day.empty:
            skipped.append({"date": d, "reason": "empty_day_panel"})
            continue
        day["weight_date"] = wdt
        raw_panels.append(day)
        usage_rows.append(
            {
                "date": d,
                "weight_date": str(wdt.date()),
                "n_constituents": int(len(universe)),
                "top_weight_code": str(universe.iloc[0]["code"]),
                "top_weight": float(universe.iloc[0]["weight"]),
                "rows": int(len(day)),
                "active_mean": float(day["active_now"].mean()),
            }
        )
        if i <= 3 or i == len(trade_dates) or i % 50 == 0:
            print(f"[panel] {i}/{len(trade_dates)} {d} weight={wdt.date()} rows={len(day)}")

    if not raw_panels:
        raise RuntimeError(f"no panels built; skipped={skipped[:5]}")
    raw = pd.concat(raw_panels, ignore_index=True).sort_values(["date", "minute_idx"]).reset_index(drop=True)
    scored = score_panel(raw, feature_cfg)
    scored["regular_time"] = regular_signal_mask(scored).astype(int)
    usage = pd.DataFrame(usage_rows)

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(panel_path, index=False)
    usage.to_csv(usage_path, index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "skipped_panel_dates.csv", index=False)
    return scored, usage


def candidate_grid() -> pd.DataFrame:
    rows: list[dict] = []
    state_sets = {
        "ED": ("EMERGING_DOWN",),
        "ED_CD": ("EMERGING_DOWN", "CONFIRMED_DOWN"),
    }
    for name, states in state_sets.items():
        for require_highconf in (0, 1):
            for minute_min in (30, 45):
                for hold_bars in (20, 30, 45):
                    for dir5_abs_min in (0.75, 1.00):
                        for dir10_max in (0.45, 0.00):
                            for exh10_q in (0.80, 0.90, 0.95, 1.00):
                                for ipg10_q in (-1.0, 0.25, 0.40):
                                    rows.append(
                                        {
                                            "rule": name,
                                            "states": "|".join(states),
                                            "require_highconf": int(require_highconf),
                                            "minute_min": int(minute_min),
                                            "hold_bars": int(hold_bars),
                                            "dir5_abs_min": float(dir5_abs_min),
                                            "dir10_max": float(dir10_max),
                                            "exh10_q": float(exh10_q),
                                            "ipg10_q": float(ipg10_q),
                                        }
                                    )
    return pd.DataFrame(rows)


def split_state_set(raw: str) -> set[str]:
    return {x for x in str(raw).split("|") if x}


def candidate_thresholds(train: pd.DataFrame, cand: pd.Series) -> dict:
    states = split_state_set(cand["states"])
    base = train[train["state"].isin(states)].copy()
    if bool(cand["require_highconf"]):
        base = base[base["signal_side_5"] == -1]
    base = base[base["DirScore_5"] <= -float(cand["dir5_abs_min"])]
    base = base[base["DirScore_10"] <= float(cand["dir10_max"])]
    base = base[base["regular_time"].astype(bool)]
    if base.empty:
        return {"exh10_max": np.nan, "ipg10_min": np.nan}

    exh_q = float(cand["exh10_q"])
    exh_max = float(base["Exhaustion_10"].quantile(exh_q)) if exh_q < 0.999 else float("inf")
    ipg_q = float(cand["ipg10_q"])
    ipg_min = float(base["IPG_10"].quantile(ipg_q)) if ipg_q >= 0.0 else -float("inf")
    return {"exh10_max": exh_max, "ipg10_min": ipg_min}


def candidate_mask(panel: pd.DataFrame, cand: pd.Series | dict, thresholds: dict) -> pd.Series:
    states = split_state_set(cand["states"])
    mask = panel["state"].isin(states)
    mask &= panel["regular_time"].astype(bool)
    mask &= panel["minute_idx"].astype(int) >= int(cand["minute_min"])
    mask &= panel["DirScore_5"].astype(float) <= -float(cand["dir5_abs_min"])
    mask &= panel["DirScore_10"].astype(float) <= float(cand["dir10_max"])
    if bool(cand["require_highconf"]):
        mask &= panel["signal_side_5"].astype(int) == -1
    if np.isfinite(thresholds.get("exh10_max", np.nan)):
        mask &= panel["Exhaustion_10"].astype(float) <= float(thresholds["exh10_max"])
    if np.isfinite(thresholds.get("ipg10_min", np.nan)):
        mask &= panel["IPG_10"].astype(float) >= float(thresholds["ipg10_min"])
    return mask


def simulate_short_trades(panel: pd.DataFrame, cand: pd.Series | dict, thresholds: dict, cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []
    hold = int(cand["hold_bars"])
    mask = candidate_mask(panel, cand, thresholds)
    base = panel[mask].sort_values(["date", "minute_idx"]).copy()
    if base.empty:
        return pd.DataFrame()

    for date, day in panel.sort_values(["date", "minute_idx"]).groupby("date", sort=True):
        day = day.reset_index(drop=True)
        idxs = day.index[candidate_mask(day, cand, thresholds)].tolist()
        if not idxs:
            continue
        next_ok_pos = -1
        trades_today = 0
        for signal_pos in idxs:
            if signal_pos < next_ok_pos or trades_today >= cfg.max_trades_per_day:
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
                    "net_bps": gross_bps - float(cfg.round_trip_cost_bps),
                    "state": day.loc[signal_pos, "state"],
                    "DirScore_5": float(day.loc[signal_pos, "DirScore_5"]),
                    "DirScore_10": float(day.loc[signal_pos, "DirScore_10"]),
                    "GateScore_5": float(day.loc[signal_pos, "GateScore_5"]),
                    "GateScore_10": float(day.loc[signal_pos, "GateScore_10"]),
                    "Exhaustion_10": float(day.loc[signal_pos, "Exhaustion_10"]),
                    "IPG_10": float(day.loc[signal_pos, "IPG_10"]),
                    "signal_side_5": int(day.loc[signal_pos, "signal_side_5"]),
                }
            )
            next_ok_pos = exit_pos
            trades_today += 1
    return pd.DataFrame(rows)


def daily_bps_series(panel: pd.DataFrame, trades: pd.DataFrame) -> pd.Series:
    dates = pd.Index(sorted(pd.to_datetime(panel["date"]).dt.normalize().unique()), name="date")
    if trades.empty:
        return pd.Series(0.0, index=dates, name="day_net_bps")
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"]).dt.normalize()
    return t.groupby("date")["net_bps"].sum().reindex(dates, fill_value=0.0)


def summarize_bps(panel: pd.DataFrame, trades: pd.DataFrame) -> dict:
    daily = daily_bps_series(panel, trades)
    ret = daily.to_numpy(dtype=float) / 10000.0
    equity = pd.Series((1.0 + pd.Series(ret, index=daily.index)).cumprod().values, index=daily.index)
    total = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    vol = float(np.std(ret, ddof=0))
    sharpe = float(np.mean(ret) / vol * np.sqrt(242)) if vol > 1e-12 else 0.0
    max_dd = float((equity / equity.cummax() - 1.0).min()) if len(equity) else 0.0
    calmar = float(((1.0 + total) ** (242 / max(len(daily), 1)) - 1.0) / abs(max_dd)) if max_dd < 0 else np.nan
    first_half = trades.iloc[: len(trades) // 2] if len(trades) >= 2 else trades
    second_half = trades.iloc[len(trades) // 2 :] if len(trades) >= 2 else trades
    return {
        "n_days": int(len(daily)),
        "n_trade_days": int((daily != 0).sum()),
        "n_trades": int(len(trades)),
        "win_rate": float((trades["net_bps"] > 0).mean()) if len(trades) else 0.0,
        "avg_net_bps": float(trades["net_bps"].mean()) if len(trades) else 0.0,
        "median_net_bps": float(trades["net_bps"].median()) if len(trades) else 0.0,
        "p25_net_bps": float(trades["net_bps"].quantile(0.25)) if len(trades) else 0.0,
        "p75_net_bps": float(trades["net_bps"].quantile(0.75)) if len(trades) else 0.0,
        "first_half_avg_net_bps": float(first_half["net_bps"].mean()) if len(first_half) else 0.0,
        "second_half_avg_net_bps": float(second_half["net_bps"].mean()) if len(second_half) else 0.0,
        "total_return_unlevered": total,
        "annual_return_unlevered": float((1.0 + total) ** (242 / max(len(daily), 1)) - 1.0) if len(daily) else 0.0,
        "sharpe_unlevered": sharpe,
        "max_drawdown_unlevered": max_dd,
        "calmar_unlevered": calmar,
    }


def compound_if(panel: pd.DataFrame, trades: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    all_dates = pd.Index(sorted(pd.to_datetime(panel["date"]).dt.normalize().unique()), name="date")
    equity = float(cfg.initial_capital)
    rows: list[dict] = []
    if trades.empty:
        daily = pd.DataFrame({"date": all_dates, "equity": equity, "day_pnl": 0.0, "contracts": 0})
    else:
        t = trades.sort_values(["entry_datetime"]).copy()
        t["date"] = pd.to_datetime(t["date"]).dt.normalize()
        grouped = {d: g.copy() for d, g in t.groupby("date", sort=True)}
        for d in all_dates:
            day_pnl = 0.0
            contracts_total = 0
            if d in grouped:
                for _, tr in grouped[d].iterrows():
                    index_level = float(tr["entry_px"]) * float(cfg.etf_to_index_scale)
                    notional = index_level * float(cfg.if_multiplier)
                    margin = notional * float(cfg.if_margin_rate)
                    contracts = int(math.floor(equity * float(cfg.if_utilization) / margin)) if margin > 0 else 0
                    if contracts <= 0:
                        continue
                    pnl = contracts * notional * (float(tr["net_bps"]) / 10000.0)
                    equity += pnl
                    day_pnl += pnl
                    contracts_total += contracts
            rows.append({"date": d, "equity": equity, "day_pnl": day_pnl, "contracts": contracts_total})
        daily = pd.DataFrame(rows)

    eq = daily["equity"].astype(float)
    daily_ret = eq.pct_change().fillna((eq.iloc[0] / cfg.initial_capital - 1.0) if len(eq) else 0.0)
    total = float(eq.iloc[-1] / cfg.initial_capital - 1.0) if len(eq) else 0.0
    max_dd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    vol = float(daily_ret.std(ddof=0))
    sharpe = float(daily_ret.mean() / vol * np.sqrt(242)) if vol > 1e-12 else 0.0
    ann = float((1.0 + total) ** (242 / max(len(eq), 1)) - 1.0) if len(eq) else 0.0
    calmar = float(ann / abs(max_dd)) if max_dd < 0 else np.nan
    summary = {
        "initial_capital": float(cfg.initial_capital),
        "ending_equity": float(eq.iloc[-1]) if len(eq) else float(cfg.initial_capital),
        "total_return": total,
        "annual_return": ann,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "trade_days": int((daily["day_pnl"] != 0).sum()) if len(daily) else 0,
        "avg_contracts_on_trade_day": float(daily.loc[daily["contracts"] > 0, "contracts"].mean())
        if (len(daily) and (daily["contracts"] > 0).any())
        else 0.0,
    }
    return daily, summary


def score_candidate(summary: dict, cfg: Config) -> float:
    n = int(summary["n_trades"])
    if n < cfg.min_train_trades:
        return -1e12
    if summary["avg_net_bps"] <= 0:
        return -1e12
    # Enforce some temporal robustness inside the training period.
    half_penalty = 0.0
    if summary["first_half_avg_net_bps"] <= 0:
        half_penalty += 50.0
    if summary["second_half_avg_net_bps"] <= 0:
        half_penalty += 50.0
    dd_penalty = 80.0 * abs(float(summary["max_drawdown_unlevered"]))
    tail_penalty = max(0.0, -float(summary["p25_net_bps"])) * 0.20
    return (
        float(summary["avg_net_bps"]) * math.sqrt(n)
        + 12.0 * float(summary["win_rate"])
        + 150.0 * float(summary["total_return_unlevered"])
        - dd_penalty
        - tail_penalty
        - half_penalty
    )


def evaluate_grid(train: pd.DataFrame, test: pd.DataFrame, grid: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []
    for i, (_, cand) in enumerate(grid.iterrows(), start=1):
        if i == 1 or i % 250 == 0 or i == len(grid):
            print(f"[grid] {i}/{len(grid)}", flush=True)
        thresholds = candidate_thresholds(train, cand)
        train_trades = simulate_short_trades(train, cand, thresholds, cfg)
        train_summary = summarize_bps(train, train_trades)
        score = score_candidate(train_summary, cfg)
        test_trades = simulate_short_trades(test, cand, thresholds, cfg)
        test_summary = summarize_bps(test, test_trades)
        row = cand.to_dict()
        row.update({f"th_{k}": v for k, v in thresholds.items()})
        row.update({f"train_{k}": v for k, v in train_summary.items()})
        row.update({f"test_{k}": v for k, v in test_summary.items()})
        row["score"] = score
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def run_walk_forward(panel: pd.DataFrame, grid: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dates = sorted(pd.to_datetime(panel["date"]).dt.normalize().unique())
    fold_rows: list[dict] = []
    trade_parts: list[pd.DataFrame] = []
    for train_end in range(cfg.wf_train_days, len(dates) - cfg.wf_test_days + 1, cfg.wf_test_days):
        train_dates = dates[train_end - cfg.wf_train_days : train_end]
        test_dates = dates[train_end : train_end + cfg.wf_test_days]
        train = panel[pd.to_datetime(panel["date"]).dt.normalize().isin(train_dates)]
        test = panel[pd.to_datetime(panel["date"]).dt.normalize().isin(test_dates)]
        evaluated = evaluate_grid(train, test, grid, cfg)
        best = evaluated.iloc[0]
        cand = best[grid.columns].to_dict()
        thresholds = {"exh10_max": best["th_exh10_max"], "ipg10_min": best["th_ipg10_min"]}
        trades = simulate_short_trades(test, cand, thresholds, cfg)
        if not trades.empty:
            trades["fold_train_start"] = train_dates[0]
            trades["fold_train_end"] = train_dates[-1]
            trades["fold_test_start"] = test_dates[0]
            trades["fold_test_end"] = test_dates[-1]
            trade_parts.append(trades)
        fold_rows.append(
            {
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                **{f"selected_{k}": best[k] for k in grid.columns},
                "th_exh10_max": best["th_exh10_max"],
                "th_ipg10_min": best["th_ipg10_min"],
                "train_n_trades": best["train_n_trades"],
                "train_avg_net_bps": best["train_avg_net_bps"],
                "train_total_return": best["train_total_return_unlevered"],
                "test_n_trades": best["test_n_trades"],
                "test_avg_net_bps": best["test_avg_net_bps"],
                "test_total_return": best["test_total_return_unlevered"],
                "score": best["score"],
            }
        )
        print(
            "[wf] "
            f"{pd.Timestamp(test_dates[0]).date()}..{pd.Timestamp(test_dates[-1]).date()} "
            f"rule={best['rule']} hold={int(best['hold_bars'])} "
            f"test_n={int(best['test_n_trades'])} test_avg={best['test_avg_net_bps']:.2f}bps"
        )

    folds = pd.DataFrame(fold_rows)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    wf_panel = panel[panel["date"].between(folds["test_start"].min(), folds["test_end"].max())] if not folds.empty else panel.iloc[0:0]
    summary = summarize_bps(wf_panel, trades)
    _, if_summary = compound_if(wf_panel, trades, cfg)
    summary = {**summary, **{f"if_{k}": v for k, v in if_summary.items()}}
    return folds, trades, summary


def save_trade_set(name: str, panel: pd.DataFrame, trades: pd.DataFrame, cfg: Config, out_dir: Path) -> dict:
    trades_path = out_dir / f"{name}_trades.csv"
    daily_path = out_dir / f"{name}_if_daily.csv"
    trades.to_csv(trades_path, index=False)
    daily, if_summary = compound_if(panel, trades, cfg)
    daily.to_csv(daily_path, index=False)
    summary = summarize_bps(panel, trades)
    return {
        "trade_csv": str(trades_path.resolve()),
        "if_daily_csv": str(daily_path.resolve()),
        "summary": summary,
        "if_summary": if_summary,
    }


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel, usage = build_or_load_panel(cfg, out_dir)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["date", "minute_idx"]).reset_index(drop=True)

    oos_start = pd.Timestamp(cfg.oos_start).normalize()
    train = panel[panel["date"] < oos_start].copy()
    oos = panel[panel["date"] >= oos_start].copy()
    grid = candidate_grid()

    print(
        f"[split] train {train['date'].min().date()}..{train['date'].max().date()} "
        f"days={train['date'].nunique()} | oos {oos['date'].min().date()}..{oos['date'].max().date()} "
        f"days={oos['date'].nunique()} | candidates={len(grid)}"
    )
    evaluated = evaluate_grid(train, oos, grid, cfg)
    evaluated_path = out_dir / "candidate_grid_train_oos.csv"
    evaluated.to_csv(evaluated_path, index=False)
    top_path = out_dir / "top_candidates.csv"
    evaluated.head(50).to_csv(top_path, index=False)

    selected = evaluated.iloc[0]
    selected_cand = selected[grid.columns].to_dict()
    thresholds = {"exh10_max": selected["th_exh10_max"], "ipg10_min": selected["th_ipg10_min"]}
    train_trades = simulate_short_trades(train, selected_cand, thresholds, cfg)
    oos_trades = simulate_short_trades(oos, selected_cand, thresholds, cfg)
    selected_train = save_trade_set("selected_train", train, train_trades, cfg, out_dir)
    selected_oos = save_trade_set("selected_oos", oos, oos_trades, cfg, out_dir)

    folds, wf_trades, wf_summary = run_walk_forward(panel, grid, cfg)
    folds_path = out_dir / "walkforward_folds.csv"
    wf_trades_path = out_dir / "walkforward_trades.csv"
    wf_daily, wf_if_summary = compound_if(
        panel[panel["date"].between(folds["test_start"].min(), folds["test_end"].max())] if not folds.empty else panel.iloc[0:0],
        wf_trades,
        cfg,
    )
    wf_daily_path = out_dir / "walkforward_if_daily.csv"
    folds.to_csv(folds_path, index=False)
    wf_trades.to_csv(wf_trades_path, index=False)
    wf_daily.to_csv(wf_daily_path, index=False)

    summary = {
        "config": asdict(cfg),
        "data": {
            "panel_rows": int(len(panel)),
            "panel_days": int(panel["date"].nunique()),
            "panel_start": str(panel["date"].min().date()),
            "panel_end": str(panel["date"].max().date()),
            "oos_start": str(oos_start.date()),
            "weight_usage_rows": int(len(usage)),
            "etf_data_dir": cfg.etf_data_dir,
            "daily_weights_csv": cfg.daily_weights_csv,
        },
        "selected_candidate": selected_cand,
        "selected_thresholds": thresholds,
        "selected_train": selected_train,
        "selected_oos": selected_oos,
        "walkforward_summary": wf_summary,
        "walkforward_if_summary": wf_if_summary,
        "artifacts": {
            "panel": str((out_dir / cfg.panel_file).resolve()),
            "weight_usage": str((out_dir / "weight_usage.csv").resolve()),
            "candidate_grid": str(evaluated_path.resolve()),
            "top_candidates": str(top_path.resolve()),
            "walkforward_folds": str(folds_path.resolve()),
            "walkforward_trades": str(wf_trades_path.resolve()),
            "walkforward_if_daily": str(wf_daily_path.resolve()),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"Saved: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
