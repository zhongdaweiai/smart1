#!/usr/bin/env python3
"""
Reproduce the 510300 Breadth Regime Strategy from spec.

Uses pre-built predictions and panel, matching the original implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path("/Users/daweizhong/Documents/projects")
PANEL_PATH = ROOT / "artifacts" / "market_breadth_vs_etf_2023_2026" / "panel.parquet"
PRED_PATH = ROOT / "artifacts" / "breadth_confidence_model_v1" / "predictions.parquet"
REF_REPORT = ROOT / "artifacts" / "breadth_confidence_510300_regime_fixedthreshold_v1" / "report.json"
OUTPUT_DIR = ROOT / "artifacts" / "breadth_regime_reproduction"

# ── strategy constants ─────────────────────────────────────────────────
CODE = "510300.XSHG"
HORIZON = 5
LOOKBACK_DAYS = 120
COST_BPS = 6.0

PARAMS = {
    "conf_q": 0.99,
    "shock_q": 0.90,
    "flip_q": 0.70,
    "bdiff_q": 0.90,
    "trend_min": 0.10,
    "gap_q": 0.50,
}


def build_dataset() -> pd.DataFrame:
    """Merge panel and predictions, compute regime features."""
    panel = pd.read_parquet(PANEL_PATH)
    pred = pd.read_parquet(PRED_PATH)

    panel = panel[panel["etf_code"] == CODE].copy()
    pred = pred[(pred["etf_code"] == CODE) & (pred["horizon_min"] == HORIZON)].copy()

    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    panel = panel.sort_values(["date", "datetime"]).copy()

    # breadth flip (day-level shift for previous sign)
    panel["breadth_sign"] = np.sign(panel["breadth_diff_1"]).fillna(0.0)
    panel["prev_breadth_sign"] = panel.groupby("date")["breadth_sign"].shift(1).fillna(0.0)
    panel["breadth_flip"] = (
        (panel["breadth_sign"] != 0)
        & (panel["prev_breadth_sign"] != 0)
        & (panel["breadth_sign"] != panel["prev_breadth_sign"])
    ).astype(int)
    panel["gap_abs"] = (panel["breadth_ratio"] - panel["ret_oc"]).abs()

    # day-level regime features
    regime_rows = []
    for date, day in panel.groupby("date"):
        first30 = day[day["minute_idx"] < 30]
        if first30.empty:
            continue
        ret = first30["ret_cc_1"].dropna()
        trendness = np.nan
        if len(ret) and ret.abs().sum() > 0:
            trendness = float(abs(ret.sum()) / ret.abs().sum())
        regime_rows.append({
            "date": date,
            "open30_flip_rate": float(first30["breadth_flip"].mean()),
            "open30_abs_bdiff_mean": float(first30["breadth_diff_1"].abs().mean()),
            "open30_trendness": trendness,
        })
    regime = pd.DataFrame(regime_rows)

    # merge predictions with panel columns
    df = pred.merge(
        panel[["date", "datetime", "breadth_diff_1", "ret_oc", "gap_abs"]],
        on=["date", "datetime"],
        how="left",
    )
    df = df.merge(regime, on="date", how="left")
    df["pred_sign"] = np.sign(df["pred_ret"])
    df["breadth_sign"] = np.sign(df["breadth_diff_1"])
    df["breadth_align"] = (
        (df["pred_sign"] != 0)
        & (df["breadth_sign"] != 0)
        & (df["pred_sign"] == df["breadth_sign"])
    )
    return df.sort_values(["date", "datetime"]).reset_index(drop=True)


def apply_rule(df: pd.DataFrame, params: dict, threshold_source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the filter chain exactly as in the original."""
    work = df.copy()
    hist = threshold_source.copy()

    conf_thr = float(hist["confidence"].quantile(params["conf_q"]))
    work = work[work["confidence"] >= conf_thr]
    hist = hist[hist["confidence"] >= conf_thr]

    work = work[work["breadth_align"]]
    hist = hist[hist["breadth_align"]]

    work = work[work["minute_idx"] >= 30]
    hist = hist[hist["minute_idx"] >= 30]

    work = work[work["pred_sign"] > 0]
    hist = hist[hist["pred_sign"] > 0]

    if hist.empty:
        return work.iloc[0:0].copy(), {
            "conf_thr": conf_thr,
            "shock_thr": np.nan,
            "flip_thr": np.nan,
            "bdiff_thr": np.nan,
            "gap_thr": np.nan,
        }

    shock_thr = float(hist["breadth_diff_1"].abs().quantile(params["shock_q"]))
    work = work[work["breadth_diff_1"].abs() <= shock_thr]
    hist = hist[hist["breadth_diff_1"].abs() <= shock_thr]
    if hist.empty:
        return work.iloc[0:0].copy(), {
            "conf_thr": conf_thr,
            "shock_thr": shock_thr,
            "flip_thr": np.nan,
            "bdiff_thr": np.nan,
            "gap_thr": np.nan,
        }

    # Regime thresholds computed on minute-level rows (same as original)
    flip_thr = float(hist["open30_flip_rate"].quantile(params["flip_q"]))
    bdiff_thr = float(hist["open30_abs_bdiff_mean"].quantile(params["bdiff_q"]))
    work = work[work["open30_flip_rate"] <= flip_thr]
    work = work[work["open30_abs_bdiff_mean"] <= bdiff_thr]
    work = work[work["open30_trendness"] >= params["trend_min"]]
    hist = hist[hist["open30_flip_rate"] <= flip_thr]
    hist = hist[hist["open30_abs_bdiff_mean"] <= bdiff_thr]
    hist = hist[hist["open30_trendness"] >= params["trend_min"]]
    if hist.empty:
        return work.iloc[0:0].copy(), {
            "conf_thr": conf_thr,
            "shock_thr": shock_thr,
            "flip_thr": flip_thr,
            "bdiff_thr": bdiff_thr,
            "gap_thr": np.nan,
        }

    gap_thr = np.nan
    if params["gap_q"] > 0.0:
        gap_thr = float(hist["gap_abs"].quantile(params["gap_q"]))
        work = work[work["gap_abs"] >= gap_thr]

    return work, {
        "conf_thr": conf_thr,
        "shock_thr": shock_thr,
        "flip_thr": flip_thr,
        "bdiff_thr": bdiff_thr,
        "gap_thr": gap_thr,
    }


def run_walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward with fixed params and rolling thresholds."""
    dates = sorted(df["date"].unique())
    chosen_rows = []
    trade_rows = []

    for idx in range(LOOKBACK_DAYS, len(dates)):
        hist_dates = dates[idx - LOOKBACK_DAYS : idx]
        test_date = dates[idx]

        hist = df[df["date"].isin(hist_dates)].copy()
        test = df[df["date"] == test_date].copy()

        if hist.empty or test.empty:
            continue

        selected, thresholds = apply_rule(test, PARAMS, hist)
        chosen_rows.append({
            "date": test_date,
            **PARAMS,
            **thresholds,
            "test_n_preds": int(len(selected)),
        })

        if not selected.empty:
            day_trades = selected.copy()
            for k, v in thresholds.items():
                day_trades[k] = v
            trade_rows.append(day_trades)

    chosen_df = pd.DataFrame(chosen_rows).sort_values("date").reset_index(drop=True)
    trade_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return chosen_df, trade_df


def evaluate(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"error": "no trades"}

    day_pnl = trades.groupby("date")["net_signed_ret_bps"].sum()
    nonzero = trades[trades["signed_ret_bps"] != 0]
    return {
        "n_preds": int(len(trades)),
        "n_days": int(trades["date"].nunique()),
        "hit_rate": float((trades["signed_ret_bps"] > 0).mean()),
        "zero_rate": float((trades["signed_ret_bps"] == 0).mean()),
        "nonzero_hit_rate": float((nonzero["signed_ret_bps"] > 0).mean()) if len(nonzero) else np.nan,
        "avg_signed_bps_gross": float(trades["signed_ret_bps"].mean()),
        "avg_signed_bps_net": float(trades["net_signed_ret_bps"].mean()),
        "median_signed_bps_net": float(trades["net_signed_ret_bps"].median()),
        "worst_day_net_bps": float(day_pnl.min()),
        "best_day_net_bps": float(day_pnl.max()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Step 1: Building dataset (panel + predictions + regime)...")
    df = build_dataset()
    print(f"  dataset: {len(df)} rows, {df['date'].nunique()} dates")

    print("Step 2: Running walk-forward filter chain...")
    chosen_df, trade_df = run_walkforward(df)
    print(f"  trades: {len(trade_df)}, trade days: {trade_df['date'].nunique() if len(trade_df) > 0 else 0}")

    print("Step 3: Evaluating...")
    result = evaluate(trade_df)

    # Save artifacts
    trade_df.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    chosen_df.to_csv(OUTPUT_DIR / "thresholds.csv", index=False)

    # Load reference
    with open(REF_REPORT) as f:
        ref = json.load(f)

    print("\n" + "=" * 60)
    print("REPRODUCTION RESULTS vs REFERENCE")
    print("=" * 60)
    ref_summary = ref["summary"]
    for key in ["n_preds", "n_days", "hit_rate", "zero_rate", "nonzero_hit_rate",
                "avg_signed_bps_gross", "avg_signed_bps_net", "median_signed_bps_net",
                "worst_day_net_bps", "best_day_net_bps"]:
        repro = result.get(key, "N/A")
        reference = ref_summary.get(key, "N/A")
        match = ""
        if isinstance(repro, (int, float)) and isinstance(reference, (int, float)):
            if isinstance(repro, int) and isinstance(reference, int):
                match = " MATCH" if repro == reference else " DIFF"
            else:
                match = " MATCH" if abs(repro - reference) < 1e-10 else " DIFF"
        print(f"  {key:30s}  repro={repro!s:>20s}  ref={reference!s:>20s}{match}")

    report = {"params": PARAMS, "result": result, "reference": ref_summary}
    with open(OUTPUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nArtifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
