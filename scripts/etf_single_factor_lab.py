#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF / 指数日内单因子研究

目标:
1. 读取 explore14 产出的 scored_panel.parquet
2. 对给定因子做日内时序单因子测试
3. 输出:
   - 每日 IC 序列与汇总
   - 日内分层效果
   - 分样本效果对比
   - 图表与 CSV 报告

默认因子:
  - IPG_10
  - DirScore_10

默认样本:
  - 2015 单边上涨窗口
  - 2019 震荡窗口
  - 2020-2021 强趋势窗口
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "PingFang SC", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_PATHS = [
    str(ROOT_DIR / "artifacts" / "strategy_lab_explore14_fullmkt_2015win" / "scored_panel.parquet"),
    str(ROOT_DIR / "artifacts" / "strategy_lab_explore14_fullmkt_2019win" / "scored_panel.parquet"),
    str(ROOT_DIR / "artifacts" / "strategy_lab_explore14_fullmkt_240d_v1" / "scored_panel.parquet"),
]
DEFAULT_OUTPUT_DIR = str(ROOT_DIR / "artifacts" / "etf_single_factor_lab_v1")


@dataclass
class LabConfig:
    panel_paths: List[str]
    output_dir: str
    factors: List[str]
    horizons: List[int]
    quantiles: int = 5
    min_obs_per_day: int = 80
    rolling_ic_days: int = 20


def parse_args() -> LabConfig:
    parser = argparse.ArgumentParser(description="ETF intraday single factor lab")
    parser.add_argument("--panel-path", action="append", dest="panel_paths")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--factor", action="append", dest="factors")
    parser.add_argument("--horizon", action="append", dest="horizons", type=int)
    parser.add_argument("--quantiles", default=5, type=int)
    parser.add_argument("--min-obs-per-day", default=80, type=int)
    parser.add_argument("--rolling-ic-days", default=20, type=int)
    args = parser.parse_args()
    return LabConfig(
        panel_paths=args.panel_paths or DEFAULT_PANEL_PATHS,
        output_dir=args.output_dir,
        factors=args.factors or ["IPG_10", "DirScore_10"],
        horizons=args.horizons or [10, 20, 30, 45],
        quantiles=args.quantiles,
        min_obs_per_day=args.min_obs_per_day,
        rolling_ic_days=args.rolling_ic_days,
    )


def simplify_sample_name(path: str) -> str:
    name = Path(path).parent.name
    mapping = {
        "strategy_lab_explore14_fullmkt_2015win": "2015",
        "strategy_lab_explore14_fullmkt_2019win": "2019",
        "strategy_lab_explore14_fullmkt_240d_v1": "2020_2021",
    }
    return mapping.get(name, name)


def load_panel(cfg: LabConfig) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    required_cols = ["date", "datetime", "minute_idx"] + cfg.factors + [f"fwd_open_ret_{h}" for h in cfg.horizons]
    for path in cfg.panel_paths:
        sample = simplify_sample_name(path)
        df = pd.read_parquet(path, columns=required_cols)
        df["sample"] = sample
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["datetime"] = pd.to_datetime(df["datetime"])
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["sample", "date", "datetime"]).reset_index(drop=True)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan
    xs = x[mask]
    ys = y[mask]
    if xs.nunique() <= 1 or ys.nunique() <= 1:
        return np.nan
    try:
        val = spearmanr(xs, ys).statistic
    except Exception:
        return np.nan
    return float(val) if np.isfinite(val) else np.nan


def compute_daily_ic(panel: pd.DataFrame, factor: str, horizon: int, min_obs: int) -> pd.DataFrame:
    rows: List[dict] = []
    ret_col = f"fwd_open_ret_{horizon}"
    for (sample, date), day in panel.groupby(["sample", "date"], sort=True):
        x = day[factor]
        y = day[ret_col]
        mask = x.notna() & y.notna()
        if mask.sum() < min_obs:
            continue
        ic = safe_spearman(x[mask], y[mask])
        if np.isnan(ic):
            continue
        rows.append({"sample": sample, "date": date, "factor": factor, "horizon": horizon, "daily_ic": ic})
    return pd.DataFrame(rows)


def compute_daily_quantiles(
    panel: pd.DataFrame,
    factor: str,
    horizon: int,
    quantiles: int,
    min_obs: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    ret_col = f"fwd_open_ret_{horizon}"
    for (sample, date), day in panel.groupby(["sample", "date"], sort=True):
        x = day[factor]
        y = day[ret_col]
        mask = x.notna() & y.notna()
        if mask.sum() < min_obs:
            continue
        x_valid = x[mask]
        y_valid = y[mask]
        if x_valid.nunique() < quantiles:
            continue
        try:
            labels = pd.qcut(
                x_valid.rank(method="first"),
                quantiles,
                labels=[f"Q{i+1}" for i in range(quantiles)],
            )
        except Exception:
            continue
        grp = y_valid.groupby(labels).mean()
        row = {"sample": sample, "date": date, "factor": factor, "horizon": horizon}
        for q in range(quantiles):
            row[f"Q{q+1}"] = float(grp.get(f"Q{q+1}", np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_ic(ic_df: pd.DataFrame) -> Dict[str, float]:
    s = ic_df["daily_ic"].dropna()
    if s.empty:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "t_stat": np.nan,
            "ic_pos_rate": np.nan,
            "n_days": 0,
        }
    std = float(s.std(ddof=0))
    mean = float(s.mean())
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else np.nan,
        "t_stat": mean / (std / np.sqrt(len(s))) if std > 0 else np.nan,
        "ic_pos_rate": float((s > 0).mean()),
        "n_days": int(len(s)),
    }


def summarize_quantiles(q_df: pd.DataFrame, quantiles: int) -> Dict[str, float]:
    quantile_cols = [f"Q{i+1}" for i in range(quantiles)]
    mean_bps = q_df[quantile_cols].mean() * 10000.0
    ls_daily = q_df[f"Q{quantiles}"] - q_df["Q1"]
    ls_mean = float(ls_daily.mean() * 10000.0) if len(ls_daily) else np.nan
    ls_std = float(ls_daily.std(ddof=0) * 10000.0) if len(ls_daily) else np.nan
    ls_ann = float(ls_daily.mean() * 242) if len(ls_daily) else np.nan
    ls_sharpe = float(ls_daily.mean() / ls_daily.std(ddof=0) * np.sqrt(242)) if len(ls_daily) and ls_daily.std(ddof=0) > 0 else np.nan
    ls_eq = (1.0 + ls_daily.fillna(0.0)).cumprod() if len(ls_daily) else pd.Series(dtype=float)
    ls_mdd = float((ls_eq / ls_eq.cummax() - 1.0).min()) if len(ls_eq) else np.nan
    monotonic = safe_spearman(
        pd.Series(np.arange(1, quantiles + 1), index=quantile_cols, dtype=float),
        mean_bps,
    )
    out: Dict[str, float] = {
        "spread_mean_bps": ls_mean,
        "spread_std_bps": ls_std,
        "spread_ann_return": ls_ann,
        "spread_sharpe": ls_sharpe,
        "spread_max_drawdown": ls_mdd,
        "quantile_monotonicity": monotonic,
    }
    for col in quantile_cols:
        out[f"{col}_mean_bps"] = float(mean_bps[col])
    return out


def build_summaries(panel: pd.DataFrame, cfg: LabConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[dict] = []
    sample_rows: List[dict] = []
    all_ic_rows: List[pd.DataFrame] = []
    all_quantile_rows: List[pd.DataFrame] = []

    for factor in cfg.factors:
        for horizon in cfg.horizons:
            ic_df = compute_daily_ic(panel, factor, horizon, cfg.min_obs_per_day)
            q_df = compute_daily_quantiles(panel, factor, horizon, cfg.quantiles, cfg.min_obs_per_day)
            if ic_df.empty or q_df.empty:
                continue
            all_ic_rows.append(ic_df)
            all_quantile_rows.append(q_df)

            row = {"factor": factor, "horizon": horizon}
            row.update(summarize_ic(ic_df))
            row.update(summarize_quantiles(q_df, cfg.quantiles))
            summary_rows.append(row)

            for sample in sorted(panel["sample"].unique()):
                ic_sub = ic_df[ic_df["sample"] == sample]
                q_sub = q_df[q_df["sample"] == sample]
                if ic_sub.empty or q_sub.empty:
                    continue
                sub_row = {"sample": sample, "factor": factor, "horizon": horizon}
                sub_row.update(summarize_ic(ic_sub))
                sub_row.update(summarize_quantiles(q_sub, cfg.quantiles))
                sample_rows.append(sub_row)

    return (
        pd.DataFrame(summary_rows).sort_values(["factor", "icir"], ascending=[True, False]),
        pd.DataFrame(sample_rows).sort_values(["factor", "sample", "icir"], ascending=[True, True, False]),
        pd.concat(all_ic_rows, ignore_index=True) if all_ic_rows else pd.DataFrame(),
        pd.concat(all_quantile_rows, ignore_index=True) if all_quantile_rows else pd.DataFrame(),
    )


def pick_best_horizon(summary_df: pd.DataFrame, factor: str) -> int:
    sub = summary_df[summary_df["factor"] == factor].copy()
    if sub.empty:
        raise ValueError(f"No summary rows for factor={factor}")
    sub["score"] = sub["icir"].fillna(-999.0) + 0.02 * sub["spread_mean_bps"].fillna(0.0)
    return int(sub.sort_values(["score", "spread_sharpe"], ascending=False).iloc[0]["horizon"])


def plot_factor_report(
    factor: str,
    summary_df: pd.DataFrame,
    sample_summary_df: pd.DataFrame,
    ic_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    cfg: LabConfig,
    output_dir: Path,
) -> None:
    best_horizon = pick_best_horizon(summary_df, factor)
    factor_summary = summary_df[summary_df["factor"] == factor].sort_values("horizon")
    factor_ic = ic_df[(ic_df["factor"] == factor) & (ic_df["horizon"] == best_horizon)].sort_values("date")
    factor_q = quantile_df[(quantile_df["factor"] == factor) & (quantile_df["horizon"] == best_horizon)].sort_values("date")
    factor_sample = sample_summary_df[(sample_summary_df["factor"] == factor) & (sample_summary_df["horizon"] == best_horizon)]

    quantile_cols = [f"Q{i+1}" for i in range(cfg.quantiles)]
    heatmap_df = (
        summary_df[summary_df["factor"] == factor]
        .set_index("horizon")[[f"Q{i+1}_mean_bps" for i in range(cfg.quantiles)]]
        .rename(columns={f"Q{i+1}_mean_bps": f"Q{i+1}" for i in range(cfg.quantiles)})
        .sort_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"{factor} — ETF日内单因子测试", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(factor_ic["date"], factor_ic["daily_ic"], color="#90caf9", linewidth=1.0, label="日IC")
    rolling = factor_ic["daily_ic"].rolling(cfg.rolling_ic_days, min_periods=5).mean()
    ax.plot(factor_ic["date"], rolling, color="#1565c0", linewidth=2.2, label=f"{cfg.rolling_ic_days}日滚动均值")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ic_mean = factor_summary.loc[factor_summary["horizon"] == best_horizon, "ic_mean"].iloc[0]
    icir = factor_summary.loc[factor_summary["horizon"] == best_horizon, "icir"].iloc[0]
    ax.set_title(f"最佳周期 {best_horizon} 分钟 日IC\nIC均值={ic_mean:.4f}  ICIR={icir:.3f}")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    x = np.arange(len(factor_summary))
    ax.bar(x - 0.18, factor_summary["icir"], width=0.36, color="#42a5f5", label="ICIR")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, factor_summary["spread_mean_bps"], width=0.36, color="#ef6c00", alpha=0.75, label="Q5-Q1 bps")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(h)) for h in factor_summary["horizon"]])
    ax.set_xlabel("预测周期(分钟)")
    ax.set_title("各周期 ICIR 与分层价差")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 0]
    mat = heatmap_df.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(list(heatmap_df.columns))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels([f"{int(h)}m" for h in heatmap_df.index])
    ax.set_title("分层均值收益热力图 (bps)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="black")
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    q_daily = factor_q.set_index("date")[quantile_cols].sort_index()
    spread = q_daily[f"Q{cfg.quantiles}"] - q_daily["Q1"]
    spread_eq = (1.0 + spread.fillna(0.0)).cumprod()
    top_eq = (1.0 + q_daily[f"Q{cfg.quantiles}"].fillna(0.0)).cumprod()
    bot_eq = (1.0 + q_daily["Q1"].fillna(0.0)).cumprod()
    ax.plot(spread_eq.index, spread_eq.values, color="#d32f2f", linewidth=2.2, label=f"Q{cfg.quantiles}-Q1")
    ax.plot(top_eq.index, top_eq.values, color="#2e7d32", linewidth=1.6, label=f"Q{cfg.quantiles}")
    ax.plot(bot_eq.index, bot_eq.values, color="#1565c0", linewidth=1.3, label="Q1")
    ax.set_title(f"最佳周期 {best_horizon} 分层累计曲线")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / f"{factor}_report.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not factor_sample.empty:
        fig2, ax = plt.subplots(figsize=(10, 5))
        sample_plot = factor_sample.sort_values("sample")
        bars = ax.bar(sample_plot["sample"], sample_plot["spread_mean_bps"], color="#fb8c00", edgecolor="white")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_ylabel("Q5-Q1 平均收益 (bps)")
        ax.set_title(f"{factor} — {best_horizon}分钟 分样本分层价差")
        for bar, val in zip(bars, sample_plot["spread_mean_bps"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top")
        plt.tight_layout()
        fig2.savefig(output_dir / f"{factor}_sample_compare.png", dpi=160, bbox_inches="tight")
        plt.close(fig2)


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)
    output_dir = Path(cfg.output_dir)

    panel = load_panel(cfg)
    summary_df, sample_summary_df, ic_df, quantile_df = build_summaries(panel, cfg)
    if summary_df.empty:
        raise RuntimeError("No valid factor summary generated.")

    summary_path = output_dir / "factor_summary.csv"
    sample_path = output_dir / "factor_sample_summary.csv"
    ic_path = output_dir / "daily_ic_series.csv"
    quantile_path = output_dir / "daily_quantile_returns.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    sample_summary_df.to_csv(sample_path, index=False, encoding="utf-8-sig")
    ic_df.to_csv(ic_path, index=False, encoding="utf-8-sig")
    quantile_df.to_csv(quantile_path, index=False, encoding="utf-8-sig")

    for factor in cfg.factors:
        plot_factor_report(factor, summary_df, sample_summary_df, ic_df, quantile_df, cfg, output_dir)

    best_rows = []
    for factor in cfg.factors:
        best_horizon = pick_best_horizon(summary_df, factor)
        best_rows.append(summary_df[(summary_df["factor"] == factor) & (summary_df["horizon"] == best_horizon)].iloc[0].to_dict())
    best_df = pd.DataFrame(best_rows)
    best_path = output_dir / "best_factor_horizons.csv"
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")

    report = {
        "config": asdict(cfg),
        "best_factors": best_df.to_dict(orient="records"),
        "artifacts": {
            "factor_summary": str(summary_path.resolve()),
            "factor_sample_summary": str(sample_path.resolve()),
            "daily_ic_series": str(ic_path.resolve()),
            "daily_quantile_returns": str(quantile_path.resolve()),
            "best_factor_horizons": str(best_path.resolve()),
        },
    }
    report_path = output_dir / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
