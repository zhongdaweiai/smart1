#!/usr/bin/env python3
"""
Refine walk-forward breadth predictions with post-filters.

This script is intentionally simple:
- load out-of-sample predictions from breadth_confidence_model.py
- add regime / stability filters found during diagnostics
- compare high-confidence gates including top 1%
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/Users/daweizhong/Documents/projects")
PRED_PATH = ROOT / "artifacts" / "breadth_confidence_model_v1" / "predictions.parquet"
PANEL_PATH = ROOT / "artifacts" / "market_breadth_vs_etf_2023_2026" / "panel.parquet"
OUTPUT_DIR = ROOT / "artifacts" / "breadth_confidence_model_refined"


@dataclass
class Config:
    pred_path: str = str(PRED_PATH)
    panel_path: str = str(PANEL_PATH)
    output_dir: str = str(OUTPUT_DIR)
    etf_codes: tuple[str, ...] = ("510300.XSHG", "510500.XSHG")
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)
    confidence_quantiles: tuple[float, ...] = (0.90, 0.95, 0.98, 0.99)
    shock_caps: tuple[float, ...] = (1.00, 0.95, 0.90)
    per_day_caps: tuple[int, ...] = (0, 5)
    sides: tuple[str, ...] = ("BOTH", "LONG", "SHORT")
    regime_flip_caps: tuple[float, ...] = (1.00, 0.80, 0.70)
    regime_bdiff_caps: tuple[float, ...] = (1.00, 0.90, 0.80)
    regime_trend_mins: tuple[float, ...] = (0.00, 0.10, 0.20)
    gap_quantiles: tuple[float, ...] = (0.00, 0.30, 0.40, 0.50)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Refine high-confidence breadth signals")
    parser.add_argument("--pred-path", default=str(PRED_PATH))
    parser.add_argument("--panel-path", default=str(PANEL_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return Config(**vars(parser.parse_args()))


def load_merged(cfg: Config) -> pd.DataFrame:
    pred = pd.read_parquet(cfg.pred_path)
    panel = pd.read_parquet(
        cfg.panel_path,
        columns=[
            "etf_code",
            "date",
            "datetime",
            "minute_idx",
            "breadth_ratio",
            "breadth_diff_1",
            "ret_oc",
            "ret_cc_1",
        ],
    )
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()

    panel = panel.sort_values(["etf_code", "date", "datetime"]).copy()
    panel["breadth_sign"] = np.sign(panel["breadth_diff_1"]).fillna(0.0)
    panel["prev_breadth_sign"] = panel.groupby(["etf_code", "date"])["breadth_sign"].shift(1).fillna(0.0)
    panel["breadth_flip"] = (
        (panel["breadth_sign"] != 0)
        & (panel["prev_breadth_sign"] != 0)
        & (panel["breadth_sign"] != panel["prev_breadth_sign"])
    ).astype(int)
    panel["gap_abs"] = (panel["breadth_ratio"] - panel["ret_oc"]).abs()

    def summarize_open30(day: pd.DataFrame) -> pd.Series:
        first30 = day[day["minute_idx"] < 30]
        ret = first30["ret_cc_1"].dropna()
        trendness = np.nan
        if len(ret) and ret.abs().sum() > 0:
            trendness = float(abs(ret.sum()) / ret.abs().sum())
        return pd.Series(
            {
                "open30_flip_rate": float(first30["breadth_flip"].mean()) if len(first30) else np.nan,
                "open30_abs_bdiff_mean": float(first30["breadth_diff_1"].abs().mean()) if len(first30) else np.nan,
                "open30_trendness": trendness,
            }
        )

    open30 = (
        panel.groupby(["etf_code", "date"], group_keys=False)
        .apply(summarize_open30, include_groups=False)
        .reset_index()
    )

    merged = pred.merge(panel, on=["etf_code", "date", "datetime"], how="left")
    merged = merged.merge(open30, on=["etf_code", "date"], how="left")
    if "minute_idx" not in merged.columns:
        if "minute_idx_x" in merged.columns:
            merged["minute_idx"] = merged["minute_idx_x"]
        elif "minute_idx_y" in merged.columns:
            merged["minute_idx"] = merged["minute_idx_y"]
    merged["pred_sign"] = np.sign(merged["pred_ret"])
    merged["breadth_sign"] = np.sign(merged["breadth_diff_1"])
    merged["etf_sign"] = np.sign(merged["ret_oc"])
    merged["breadth_align"] = (
        (merged["pred_sign"] != 0)
        & (merged["breadth_sign"] != 0)
        & (merged["pred_sign"] == merged["breadth_sign"])
    )
    merged["diverge_now"] = (
        (merged["pred_sign"] != 0)
        & (merged["etf_sign"] != 0)
        & (merged["pred_sign"] != merged["etf_sign"])
    )
    merged["no_open30"] = merged["minute_idx"] >= 30
    merged["abs_bdiff"] = merged["breadth_diff_1"].abs()
    return merged


def apply_filter_rule(
    df: pd.DataFrame,
    conf_q: float,
    require_align: bool,
    require_diverge: bool,
    require_no_open30: bool,
    shock_cap_q: float,
    per_day_cap: int,
    side: str,
    regime_flip_cap: float,
    regime_bdiff_cap: float,
    regime_trend_min: float,
    gap_q: float,
) -> pd.DataFrame:
    conf_thr = float(df["confidence"].quantile(conf_q))
    out = df[df["confidence"] >= conf_thr].copy()
    if require_align:
        out = out[out["breadth_align"]]
    if require_diverge:
        out = out[out["diverge_now"]]
    if require_no_open30:
        out = out[out["no_open30"]]
    if regime_flip_cap < 1.0:
        flip_cap = float(out["open30_flip_rate"].quantile(regime_flip_cap))
        out = out[out["open30_flip_rate"] <= flip_cap]
    if regime_bdiff_cap < 1.0:
        bdiff_cap = float(out["open30_abs_bdiff_mean"].quantile(regime_bdiff_cap))
        out = out[out["open30_abs_bdiff_mean"] <= bdiff_cap]
    if regime_trend_min > 0.0:
        out = out[out["open30_trendness"] >= regime_trend_min]
    if shock_cap_q < 1.0:
        cap = float(out["abs_bdiff"].quantile(shock_cap_q))
        out = out[out["abs_bdiff"] <= cap]
    if gap_q > 0.0:
        gap_thr = float(out["gap_abs"].quantile(gap_q))
        out = out[out["gap_abs"] >= gap_thr]
    if side == "LONG":
        out = out[out["pred_sign"] > 0]
    elif side == "SHORT":
        out = out[out["pred_sign"] < 0]
    if per_day_cap > 0 and not out.empty:
        out = out.sort_values(["date", "confidence"], ascending=[True, False]).groupby("date", group_keys=False).head(per_day_cap)
    return out


def summarize_rule(sub: pd.DataFrame) -> Dict[str, float]:
    return {
        "n_preds": int(len(sub)),
        "n_days": int(sub["date"].nunique()) if len(sub) else 0,
        "hit_rate": float((sub["signed_ret_bps"] > 0).mean()) if len(sub) else np.nan,
        "zero_rate": float((sub["signed_ret_bps"] == 0).mean()) if len(sub) else np.nan,
        "avg_signed_bps_gross": float(sub["signed_ret_bps"].mean()) if len(sub) else np.nan,
        "median_signed_bps_gross": float(sub["signed_ret_bps"].median()) if len(sub) else np.nan,
        "avg_signed_bps_net": float(sub["net_signed_ret_bps"].mean()) if len(sub) else np.nan,
        "median_signed_bps_net": float(sub["net_signed_ret_bps"].median()) if len(sub) else np.nan,
    }


def run_grid(merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for code in cfg.etf_codes:
        for horizon in cfg.horizons:
            base = merged[(merged["etf_code"] == code) & (merged["horizon_min"] == horizon)].copy()
            if base.empty:
                continue
            for conf_q in cfg.confidence_quantiles:
                for require_align in (False, True):
                    for require_diverge in (False, True):
                        for require_no_open30 in (False, True):
                            for shock_cap_q in cfg.shock_caps:
                                for per_day_cap in cfg.per_day_caps:
                                    for side in cfg.sides:
                                        for regime_flip_cap in cfg.regime_flip_caps:
                                            for regime_bdiff_cap in cfg.regime_bdiff_caps:
                                                for regime_trend_min in cfg.regime_trend_mins:
                                                    for gap_q in cfg.gap_quantiles:
                                                        uses_regime = (
                                                            regime_flip_cap < 1.0
                                                            or regime_bdiff_cap < 1.0
                                                            or regime_trend_min > 0.0
                                                        )
                                                        if uses_regime and not require_no_open30:
                                                            continue
                                                        sub = apply_filter_rule(
                                                            base,
                                                            conf_q=conf_q,
                                                            require_align=require_align,
                                                            require_diverge=require_diverge,
                                                            require_no_open30=require_no_open30,
                                                            shock_cap_q=shock_cap_q,
                                                            per_day_cap=per_day_cap,
                                                            side=side,
                                                            regime_flip_cap=regime_flip_cap,
                                                            regime_bdiff_cap=regime_bdiff_cap,
                                                            regime_trend_min=regime_trend_min,
                                                            gap_q=gap_q,
                                                        )
                                                        if len(sub) < 60:
                                                            continue
                                                        row = {
                                                            "etf_code": code,
                                                            "horizon_min": int(horizon),
                                                            "confidence_quantile": conf_q,
                                                            "require_align": require_align,
                                                            "require_diverge": require_diverge,
                                                            "require_no_open30": require_no_open30,
                                                            "shock_cap_q": shock_cap_q,
                                                            "per_day_cap": per_day_cap,
                                                            "side": side,
                                                            "regime_flip_cap": regime_flip_cap,
                                                            "regime_bdiff_cap": regime_bdiff_cap,
                                                            "regime_trend_min": regime_trend_min,
                                                            "gap_q": gap_q,
                                                        }
                                                        row.update(summarize_rule(sub))
                                                        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["etf_code", "avg_signed_bps_gross", "avg_signed_bps_net", "hit_rate"],
        ascending=[True, False, False, False],
    )


def summarize_best(grid: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for code in sorted(grid["etf_code"].unique()):
        sub = grid[grid["etf_code"] == code].copy()
        if sub.empty:
            continue
        rows.append(
            sub.sort_values(
                ["avg_signed_bps_gross", "avg_signed_bps_net", "hit_rate"],
                ascending=False,
            ).iloc[0]
        )
    return pd.DataFrame(rows)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = load_merged(cfg)
    grid = run_grid(merged, cfg)
    best = summarize_best(grid)

    merged.to_parquet(out_dir / "merged_predictions.parquet", index=False)
    grid.to_csv(out_dir / "refined_filter_grid.csv", index=False)
    best.to_csv(out_dir / "best_refined_rules.csv", index=False)

    report = {
        "config": asdict(cfg),
        "best_rules": best.to_dict(orient="records"),
        "artifacts": {
            "merged_predictions": str((out_dir / "merged_predictions.parquet").resolve()),
            "refined_filter_grid": str((out_dir / "refined_filter_grid.csv").resolve()),
            "best_refined_rules": str((out_dir / "best_refined_rules.csv").resolve()),
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
