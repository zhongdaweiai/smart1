#!/usr/bin/env python3
"""
Walk-forward confidence-gated ETF prediction model using market breadth features.

Goal:
- Predict future 1-5 minute ETF returns for 510300 / 510500.
- Only act on the most confident predictions.
- Report average signed return in bps under different confidence gates.
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
PANEL_PATH = ROOT / "artifacts" / "market_breadth_vs_etf_2023_2026" / "panel.parquet"
OUTPUT_DIR = ROOT / "artifacts" / "breadth_confidence_model"


@dataclass
class Config:
    panel_path: str = str(PANEL_PATH)
    output_dir: str = str(OUTPUT_DIR)
    etf_codes: tuple[str, ...] = ("510300.XSHG", "510500.XSHG")
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5)
    train_days: int = 60
    ridge_alpha: float = 10.0
    round_trip_cost_bps: float = 6.0


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Walk-forward breadth confidence model")
    parser.add_argument("--panel-path", default=str(PANEL_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--round-trip-cost-bps", type=float, default=6.0)
    return Config(**vars(parser.parse_args()))


FEATURES = [
    "breadth_ratio",
    "breadth_diff_1",
    "breadth_ratio_diff_1",
    "breadth_ratio_ma3",
    "breadth_ratio_ma5",
    "ret_oc",
    "ret_cc_1",
    "breadth_ret_gap",
    "breadth_impulse_x_gap",
    "breadth_level_x_ret",
    "minute_frac",
]


def load_panel(cfg: Config) -> pd.DataFrame:
    cols = [
        "etf_code",
        "date",
        "datetime",
        "minute_idx",
        "breadth_sum",
        "breadth_ratio",
        "breadth_diff_1",
        "breadth_ratio_diff_1",
        "breadth_sum_ma3",
        "breadth_sum_ma5",
        "breadth_ratio_ma3",
        "breadth_ratio_ma5",
        "ret_oc",
        "ret_cc_1",
    ] + [f"fwd_ret_{h}m" for h in cfg.horizons]
    panel = pd.read_parquet(cfg.panel_path, columns=cols)
    panel = panel[panel["etf_code"].isin(cfg.etf_codes)].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel["minute_frac"] = panel["minute_idx"] / panel["minute_idx"].max()
    panel["breadth_ret_gap"] = panel["breadth_ratio"] - panel["ret_oc"]
    panel["breadth_impulse_x_gap"] = panel["breadth_ratio_diff_1"] * panel["breadth_ret_gap"]
    panel["breadth_level_x_ret"] = panel["breadth_ratio"] * panel["ret_oc"]
    return panel.sort_values(["etf_code", "date", "datetime"]).reset_index(drop=True)


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_mean = np.nanmean(x_train, axis=0)
    x_std = np.nanstd(x_train, axis=0)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)

    y_mean = float(np.nanmean(y_train))
    y_std = float(np.nanstd(y_train))
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    x_train_z = (x_train - x_mean) / x_std
    x_test_z = (x_test - x_mean) / x_std
    y_train_z = (y_train - y_mean) / y_std

    x_train_aug = np.column_stack([np.ones(len(x_train_z)), x_train_z])
    x_test_aug = np.column_stack([np.ones(len(x_test_z)), x_test_z])

    eye = np.eye(x_train_aug.shape[1], dtype=float)
    eye[0, 0] = 0.0
    beta = np.linalg.solve(x_train_aug.T @ x_train_aug + alpha * eye, x_train_aug.T @ y_train_z)

    train_pred_z = x_train_aug @ beta
    test_pred_z = x_test_aug @ beta
    resid_std = float(np.std(y_train_z - train_pred_z, ddof=0))
    resid_std = resid_std if resid_std > 1e-8 else 1.0
    return test_pred_z * y_std + y_mean, train_pred_z * y_std + y_mean, resid_std


def walk_forward_predictions(panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for code, code_df in panel.groupby("etf_code"):
        dates = sorted(code_df["date"].unique())
        for horizon in cfg.horizons:
            pred_rows: List[pd.DataFrame] = []
            target_col = f"fwd_ret_{horizon}m"
            for idx in range(cfg.train_days, len(dates)):
                train_dates = dates[idx - cfg.train_days : idx]
                test_date = dates[idx]
                train_df = code_df[code_df["date"].isin(train_dates)].copy()
                test_df = code_df[code_df["date"] == test_date].copy()
                train_df = train_df.dropna(subset=FEATURES + [target_col])
                test_df = test_df.dropna(subset=FEATURES + [target_col])
                if len(train_df) < 5000 or len(test_df) < 100:
                    continue

                x_train = train_df[FEATURES].to_numpy(dtype=float)
                y_train = train_df[target_col].to_numpy(dtype=float)
                x_test = test_df[FEATURES].to_numpy(dtype=float)

                pred, train_pred, resid_std = fit_ridge_predict(x_train, y_train, x_test, cfg.ridge_alpha)
                out = test_df[["etf_code", "date", "datetime", "minute_idx", target_col]].copy()
                out["horizon_min"] = horizon
                out["pred_ret"] = pred
                out["pred_dir"] = np.sign(pred)
                out["confidence"] = np.abs(pred) / max(resid_std, 1e-8)
                out["signed_ret_bps"] = np.sign(pred) * out[target_col] * 10000.0
                out["net_signed_ret_bps"] = out["signed_ret_bps"] - cfg.round_trip_cost_bps
                pred_rows.append(out)
            if pred_rows:
                rows.append(pd.concat(pred_rows, ignore_index=True))
    if not rows:
        raise RuntimeError("No walk-forward predictions generated.")
    return pd.concat(rows, ignore_index=True)


def summarize_confidence(pred_df: pd.DataFrame) -> pd.DataFrame:
    quantiles = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    rows: List[Dict[str, float]] = []
    for (code, horizon), grp in pred_df.groupby(["etf_code", "horizon_min"]):
        conf = grp["confidence"].to_numpy(dtype=float)
        for q in quantiles:
            threshold = float(np.nanquantile(conf, q))
            sub = grp[grp["confidence"] >= threshold].copy()
            if sub.empty:
                continue
            rows.append(
                {
                    "etf_code": code,
                    "horizon_min": int(horizon),
                    "confidence_quantile": q,
                    "confidence_threshold": threshold,
                    "n_preds": int(len(sub)),
                    "hit_rate": float((sub["signed_ret_bps"] > 0).mean()),
                    "avg_signed_bps_gross": float(sub["signed_ret_bps"].mean()),
                    "median_signed_bps_gross": float(sub["signed_ret_bps"].median()),
                    "avg_signed_bps_net": float(sub["net_signed_ret_bps"].mean()),
                    "median_signed_bps_net": float(sub["net_signed_ret_bps"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["etf_code", "avg_signed_bps_gross", "hit_rate"],
        ascending=[True, False, False],
    )


def summarize_best(pred_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    best_rows: List[pd.Series] = []
    for code in sorted(summary_df["etf_code"].unique()):
        sub = summary_df[summary_df["etf_code"] == code].copy()
        if sub.empty:
            continue
        best_rows.append(sub.sort_values(["avg_signed_bps_gross", "hit_rate"], ascending=False).iloc[0])
    return pd.DataFrame(best_rows)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel(cfg)
    pred_df = walk_forward_predictions(panel, cfg)
    summary_df = summarize_confidence(pred_df)
    best_df = summarize_best(pred_df, summary_df)

    panel.to_parquet(out_dir / "model_panel.parquet", index=False)
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    summary_df.to_csv(out_dir / "confidence_summary.csv", index=False)
    best_df.to_csv(out_dir / "best_rules.csv", index=False)

    report = {
        "config": asdict(cfg),
        "sample": {
            "n_rows": int(len(panel)),
            "date_start": str(panel["date"].min().date()) if len(panel) else None,
            "date_end": str(panel["date"].max().date()) if len(panel) else None,
        },
        "best_rules": best_df.to_dict(orient="records"),
        "artifacts": {
            "model_panel": str((out_dir / "model_panel.parquet").resolve()),
            "predictions": str((out_dir / "predictions.parquet").resolve()),
            "confidence_summary": str((out_dir / "confidence_summary.csv").resolve()),
            "best_rules": str((out_dir / "best_rules.csv").resolve()),
        },
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
