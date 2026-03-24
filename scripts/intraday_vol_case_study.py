#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "stock_data"
OUTPUT_DIR = ROOT_DIR / "artifacts" / "intraday_vol_case_study"


@dataclass
class Config:
    start: str = "2024-01-01"
    end: str = "2025-12-31"
    primary_cutoff: int = 1500
    max_cutoff: int = 3000
    liquidity_cutoffs: tuple[int, ...] = (500, 1000, 1500, 3000)
    rolling_days: int = 20
    min_history: int = 10
    quantiles: int = 5
    regime_warmup: int = 60
    progress_every: int = 50
    output_dir: str = str(OUTPUT_DIR)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Intraday volatility factor case study")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--primary-cutoff", type=int, default=1500)
    parser.add_argument("--max-cutoff", type=int, default=3000)
    parser.add_argument("--liquidity-cutoffs", default="500,1000,1500,3000")
    parser.add_argument("--rolling-days", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--regime-warmup", type=int, default=60)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    cutoffs = tuple(sorted({int(x) for x in args.liquidity_cutoffs.split(",") if x.strip()}))
    return Config(
        start=args.start,
        end=args.end,
        primary_cutoff=args.primary_cutoff,
        max_cutoff=args.max_cutoff,
        liquidity_cutoffs=cutoffs,
        rolling_days=args.rolling_days,
        min_history=args.min_history,
        quantiles=args.quantiles,
        regime_warmup=args.regime_warmup,
        progress_every=args.progress_every,
        output_dir=args.output_dir,
    )


def iter_files(data_dir: Path, start: str, end: str) -> list[Path]:
    return [
        fp
        for fp in sorted(data_dir.glob("*.parquet"))
        if start <= fp.stem <= end
    ]


def load_daily_features(path: Path, max_cutoff: int) -> pd.DataFrame | None:
    df = pd.read_parquet(path, columns=["code", "open", "close", "money", "paused"])
    df = df[df["paused"].fillna(0) == 0].copy()
    if df.empty:
        return None

    df["minute_no"] = df.groupby("code").cumcount() + 1
    log_close = np.log(df["close"].replace(0, np.nan))
    df["log_ret"] = log_close.groupby(df["code"]).diff()

    day_money = df.groupby("code")["money"].sum(min_count=1).sort_values(ascending=False)
    top_codes = day_money.index[:max_cutoff]
    liq_rank = pd.Series(np.arange(1, len(day_money) + 1), index=day_money.index)

    df = df[df["code"].isin(top_codes)].copy()
    if df.empty:
        return None

    morning = df[df["minute_no"] <= 120].copy()
    morning["log_ret_sq"] = np.square(morning["log_ret"])
    rv_raw = np.sqrt(morning.groupby("code")["log_ret_sq"].sum(min_count=20))

    open_1 = df.loc[df["minute_no"] == 1, ["code", "open"]].set_index("code")["open"]
    close_120 = df.loc[df["minute_no"] == 120, ["code", "close"]].set_index("code")["close"]
    close_121 = df.loc[df["minute_no"] == 121, ["code", "close"]].set_index("code")["close"]
    close_240 = df.loc[df["minute_no"] == 240, ["code", "close"]].set_index("code")["close"]

    out = pd.concat(
        [
            rv_raw.rename("rv_raw"),
            (close_120 / open_1 - 1.0).rename("ret_am"),
            (close_240 / close_121 - 1.0).rename("ret_pm"),
            np.log(day_money).rename("log_money"),
            liq_rank.rename("liq_rank"),
        ],
        axis=1,
    ).dropna()
    if out.empty:
        return None
    out["date"] = pd.to_datetime(path.stem)
    out["abs_ret_am"] = out["ret_am"].abs()
    return out.reset_index()


def build_panel(cfg: Config) -> pd.DataFrame:
    files = iter_files(DATA_DIR, cfg.start, cfg.end)
    rows: list[pd.DataFrame] = []
    print(f"Loading {len(files)} trading days from {DATA_DIR}")
    for i, fp in enumerate(files, 1):
        if i % cfg.progress_every == 0:
            print(f"  processed {i}/{len(files)}: {fp.name}")
        daily = load_daily_features(fp, cfg.max_cutoff)
        if daily is not None:
            rows.append(daily)
    if not rows:
        raise RuntimeError("No valid daily rows were produced.")
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["code", "date"]).reset_index(drop=True)
    panel["rv_roll_mean"] = panel.groupby("code")["rv_raw"].transform(
        lambda s: s.rolling(cfg.rolling_days, min_periods=cfg.min_history).mean().shift(1)
    )
    panel["rv_shock"] = panel["rv_raw"] / panel["rv_roll_mean"] - 1.0
    return panel


def residualize_daily(panel: pd.DataFrame, factor_col: str, x_cols: Sequence[str], out_col: str) -> pd.DataFrame:
    panel = panel.copy()
    pieces: list[pd.Series] = []
    for _, day in panel.groupby("date", sort=True):
        cols = [factor_col, *x_cols]
        sub = day[cols].dropna()
        if len(sub) < 100:
            continue
        x = np.column_stack([np.ones(len(sub)), *[sub[col].to_numpy() for col in x_cols]])
        y = sub[factor_col].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        pieces.append(pd.Series(resid, index=sub.index))
    panel[out_col] = pd.concat(pieces).sort_index() if pieces else np.nan
    return panel


def rank_ic(day: pd.DataFrame, factor_col: str, ret_col: str = "ret_pm") -> float:
    sample = day[[factor_col, ret_col]].dropna()
    if len(sample) < 100:
        return np.nan
    return sample[factor_col].rank().corr(sample[ret_col].rank(), method="pearson")


def summarize_ic(panel: pd.DataFrame, factor_col: str) -> dict[str, float]:
    daily = panel.groupby("date").apply(lambda x: rank_ic(x, factor_col)).dropna()
    if daily.empty:
        return {
            "factor": factor_col,
            "n_days": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "ic_pos_rate": np.nan,
            "t_stat": np.nan,
        }
    std = float(daily.std(ddof=0))
    mean = float(daily.mean())
    return {
        "factor": factor_col,
        "n_days": int(daily.shape[0]),
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else np.nan,
        "ic_pos_rate": float((daily > 0).mean()),
        "t_stat": mean / (std / np.sqrt(len(daily))) if std > 0 else np.nan,
    }


def quantile_summary(panel: pd.DataFrame, factor_col: str, quantiles: int) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for date, day in panel[["date", factor_col, "ret_pm"]].dropna().groupby("date", sort=True):
        if len(day) < quantiles * 20:
            continue
        ranked = day[factor_col].rank(method="first")
        groups = pd.qcut(ranked, quantiles, labels=False) + 1
        avg_ret = day.groupby(groups)["ret_pm"].mean()
        row = {"date": date}
        for q in range(1, quantiles + 1):
            row[f"Q{q}"] = float(avg_ret.get(q, np.nan))
        rows.append(row)
    if not rows:
        return {
            "spread_mean_bps": np.nan,
            "spread_sharpe": np.nan,
            "q1_mean_bps": np.nan,
            "q5_mean_bps": np.nan,
        }
    qdf = pd.DataFrame(rows)
    qdf["spread"] = qdf[f"Q{quantiles}"] - qdf["Q1"]
    spread_std = qdf["spread"].std(ddof=0)
    return {
        "spread_mean_bps": float(qdf["spread"].mean() * 10000.0),
        "spread_sharpe": float(qdf["spread"].mean() / spread_std * np.sqrt(242)) if spread_std > 0 else np.nan,
        "q1_mean_bps": float(qdf["Q1"].mean() * 10000.0),
        "q5_mean_bps": float(qdf[f"Q{quantiles}"].mean() * 10000.0),
    }


def yearly_summary(panel: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    rows = []
    daily = panel.groupby("date").apply(lambda x: rank_ic(x, factor_col)).dropna().rename("daily_ic")
    if daily.empty:
        return pd.DataFrame(columns=["year", "factor", "n_days", "ic_mean", "icir", "ic_pos_rate"])
    grouped = daily.groupby(daily.index.year)
    for year, series in grouped:
        std = float(series.std(ddof=0))
        mean = float(series.mean())
        rows.append(
            {
                "year": int(year),
                "factor": factor_col,
                "n_days": int(series.shape[0]),
                "ic_mean": mean,
                "icir": mean / std if std > 0 else np.nan,
                "ic_pos_rate": float((series > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def regime_summary(panel: pd.DataFrame, factor_col: str, warmup: int) -> pd.DataFrame:
    market = panel.groupby("date").agg(mkt_rv_shock=("rv_shock", "mean"))
    pct = market["mkt_rv_shock"].expanding(min_periods=warmup).rank(pct=True)
    market["regime"] = "mid"
    market.loc[pct <= 1 / 3, "regime"] = "low"
    market.loc[pct > 2 / 3, "regime"] = "high"

    merged = panel.merge(market["regime"], left_on="date", right_index=True, how="left")
    rows = []
    for regime, sub in merged.groupby("regime", sort=False):
        daily = sub.groupby("date").apply(lambda x: rank_ic(x, factor_col)).dropna()
        if daily.empty:
            continue
        std = float(daily.std(ddof=0))
        mean = float(daily.mean())
        rows.append(
            {
                "regime": regime,
                "n_days": int(daily.shape[0]),
                "ic_mean": mean,
                "icir": mean / std if std > 0 else np.nan,
                "ic_pos_rate": float((daily > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def liquidity_sensitivity(panel: pd.DataFrame, factor_col: str, cutoffs: Iterable[int]) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs:
        sub = panel[panel["liq_rank"] <= cutoff]
        stats = summarize_ic(sub, factor_col)
        stats["liquidity_cutoff"] = int(cutoff)
        rows.append(stats)
    return pd.DataFrame(rows)[["liquidity_cutoff", "n_days", "ic_mean", "ic_std", "icir", "ic_pos_rate", "t_stat"]]


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel(cfg)
    panel = residualize_daily(panel, "rv_shock", ["abs_ret_am", "log_money"], "rv_shock_resid")

    primary = panel[panel["liq_rank"] <= cfg.primary_cutoff].copy()
    factor_list = ["ret_am", "abs_ret_am", "rv_raw", "rv_shock", "rv_shock_resid"]

    summary_rows = []
    yearly_frames = []
    for factor in factor_list:
        stats = summarize_ic(primary, factor)
        stats.update(quantile_summary(primary, factor, cfg.quantiles))
        summary_rows.append(stats)
        yearly_frames.append(yearly_summary(primary, factor))
    summary = pd.DataFrame(summary_rows)
    yearly = pd.concat(yearly_frames, ignore_index=True)
    regime = regime_summary(primary, "rv_shock", cfg.regime_warmup)
    liquidity = liquidity_sensitivity(panel, "rv_shock", cfg.liquidity_cutoffs)

    summary.to_csv(out_dir / "summary.csv", index=False)
    yearly.to_csv(out_dir / "yearly_summary.csv", index=False)
    regime.to_csv(out_dir / "regime_summary.csv", index=False)
    liquidity.to_csv(out_dir / "liquidity_sensitivity.csv", index=False)
    panel.to_parquet(out_dir / "panel.parquet", index=False)

    payload = {
        "config": asdict(cfg),
        "corr_rv_shock_abs_ret_am": float(primary[["rv_shock", "abs_ret_am"]].corr().iloc[0, 1]),
        "corr_rv_shock_log_money": float(primary[["rv_shock", "log_money"]].corr().iloc[0, 1]),
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\nPrimary summary")
    print(summary.to_string(index=False))
    print("\nLiquidity sensitivity")
    print(liquidity.to_string(index=False))
    print("\nRegime summary")
    print(regime.to_string(index=False))
    print(f"\nSaved outputs to {out_dir}")


if __name__ == "__main__":
    main()
