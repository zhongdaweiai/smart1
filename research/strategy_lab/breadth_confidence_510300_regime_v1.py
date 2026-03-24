#!/usr/bin/env python3
"""
Targeted refinement for 510300 5-minute breadth-confidence trades.

Focus:
- top 1% confidence only
- long-only, breadth-aligned continuation
- explicit open30 regime gating
- current breadth/ETF gap gate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/daweizhong/Documents/projects")
PANEL_PATH = ROOT / "artifacts" / "market_breadth_vs_etf_2023_2026" / "panel.parquet"
PRED_PATH = ROOT / "artifacts" / "breadth_confidence_model_v1" / "predictions.parquet"
OUT_DIR = ROOT / "artifacts" / "breadth_confidence_510300_regime_v1"
CODE = "510300.XSHG"
HORIZON = 5
ROUND_TRIP_COST_BPS = 6.0


def build_dataset() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_PATH)
    pred = pd.read_parquet(PRED_PATH)

    panel = panel[panel["etf_code"] == CODE].copy()
    pred = pred[(pred["etf_code"] == CODE) & (pred["horizon_min"] == HORIZON)].copy()

    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    panel = panel.sort_values(["date", "datetime"]).copy()

    panel["breadth_sign"] = np.sign(panel["breadth_diff_1"]).fillna(0.0)
    panel["prev_breadth_sign"] = panel.groupby("date")["breadth_sign"].shift(1).fillna(0.0)
    panel["breadth_flip"] = (
        (panel["breadth_sign"] != 0)
        & (panel["prev_breadth_sign"] != 0)
        & (panel["breadth_sign"] != panel["prev_breadth_sign"])
    ).astype(int)
    panel["gap_abs"] = (panel["breadth_ratio"] - panel["ret_oc"]).abs()

    reg_rows = []
    for date, day in panel.groupby("date"):
        first30 = day[day["minute_idx"] < 30]
        if first30.empty:
            continue
        ret = first30["ret_cc_1"].dropna()
        trendness = np.nan
        if len(ret) and ret.abs().sum() > 0:
            trendness = float(abs(ret.sum()) / ret.abs().sum())
        reg_rows.append(
            {
                "date": date,
                "open30_flip_rate": float(first30["breadth_flip"].mean()),
                "open30_abs_bdiff_mean": float(first30["breadth_diff_1"].abs().mean()),
                "open30_trendness": trendness,
            }
        )
    regime = pd.DataFrame(reg_rows)

    pred = pred.merge(
        panel[["date", "datetime", "breadth_diff_1", "ret_oc", "gap_abs"]],
        on=["date", "datetime"],
        how="left",
    )
    pred = pred.merge(regime, on="date", how="left")
    pred["pred_sign"] = np.sign(pred["pred_ret"])
    pred["breadth_sign"] = np.sign(pred["breadth_diff_1"])
    pred["breadth_align"] = (
        (pred["pred_sign"] != 0)
        & (pred["breadth_sign"] != 0)
        & (pred["pred_sign"] == pred["breadth_sign"])
    )
    pred["diverge_now"] = (
        (pred["pred_sign"] != 0)
        & (np.sign(pred["ret_oc"]) != 0)
        & (pred["pred_sign"] != np.sign(pred["ret_oc"]))
    )
    return pred.sort_values(["date", "datetime"]).reset_index(drop=True)


def build_base(df: pd.DataFrame) -> pd.DataFrame:
    conf_thr = float(df["confidence"].quantile(0.99))
    out = df[df["confidence"] >= conf_thr].copy()
    out = out[out["breadth_align"]]
    out = out[out["minute_idx"] >= 30]
    shock_cap = float(out["breadth_diff_1"].abs().quantile(0.90))
    out = out[out["breadth_diff_1"].abs() <= shock_cap]
    out = out[out["pred_sign"] > 0]
    return out


def summarize(sub: pd.DataFrame) -> dict[str, float]:
    day_net = sub.groupby("date")["net_signed_ret_bps"].sum()
    return {
        "n_preds": int(len(sub)),
        "n_days": int(sub["date"].nunique()),
        "hit_rate": float((sub["signed_ret_bps"] > 0).mean()),
        "zero_rate": float((sub["signed_ret_bps"] == 0).mean()),
        "avg_signed_bps_gross": float(sub["signed_ret_bps"].mean()),
        "avg_signed_bps_net": float(sub["net_signed_ret_bps"].mean()),
        "median_signed_bps_net": float(sub["net_signed_ret_bps"].median()),
        "nonzero_hit_rate": float((sub.loc[sub["signed_ret_bps"] != 0, "signed_ret_bps"] > 0).mean())
        if (sub["signed_ret_bps"] != 0).any()
        else np.nan,
        "worst_day_net_bps": float(day_net.min()),
        "best_day_net_bps": float(day_net.max()),
    }


def run_search(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for flip_cap in (1.00, 0.80, 0.70):
        flip_thr = float(base["open30_flip_rate"].quantile(flip_cap))
        for bdiff_cap in (1.00, 0.90, 0.80):
            bdiff_thr = float(base["open30_abs_bdiff_mean"].quantile(bdiff_cap))
            for trend_min in (0.00, 0.10, 0.20):
                for gap_q in (0.00, 0.20, 0.30, 0.40, 0.50):
                    sub = base.copy()
                    sub = sub[sub["open30_flip_rate"] <= flip_thr]
                    sub = sub[sub["open30_abs_bdiff_mean"] <= bdiff_thr]
                    sub = sub[sub["open30_trendness"] >= trend_min]
                    if gap_q > 0:
                        gap_thr = float(sub["gap_abs"].quantile(gap_q))
                        sub = sub[sub["gap_abs"] >= gap_thr]
                    if len(sub) < 60:
                        continue
                    row = {
                        "flip_cap_q": flip_cap,
                        "bdiff_cap_q": bdiff_cap,
                        "trend_min": trend_min,
                        "gap_q": gap_q,
                    }
                    row.update(summarize(sub))
                    rows.append(row)
    grid = pd.DataFrame(rows).sort_values(
        ["avg_signed_bps_net", "worst_day_net_bps", "n_preds"],
        ascending=[False, False, False],
    )
    return grid, grid.iloc[0]


def apply_best_rule(base: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    flip_thr = float(base["open30_flip_rate"].quantile(float(best["flip_cap_q"])))
    bdiff_thr = float(base["open30_abs_bdiff_mean"].quantile(float(best["bdiff_cap_q"])))
    out = base.copy()
    out = out[out["open30_flip_rate"] <= flip_thr]
    out = out[out["open30_abs_bdiff_mean"] <= bdiff_thr]
    out = out[out["open30_trendness"] >= float(best["trend_min"])]
    if float(best["gap_q"]) > 0:
        gap_thr = float(out["gap_abs"].quantile(float(best["gap_q"])))
        out = out[out["gap_abs"] >= gap_thr]
    return out.sort_values(["date", "datetime"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    base = build_base(df)
    grid, best = run_search(base)
    best_trades = apply_best_rule(base, best)
    day_pnl = best_trades.groupby("date", as_index=False)["net_signed_ret_bps"].sum().rename(
        columns={"net_signed_ret_bps": "day_net_signed_ret_bps"}
    )

    grid.to_csv(OUT_DIR / "rule_grid.csv", index=False)
    pd.DataFrame([best]).to_csv(OUT_DIR / "best_rule.csv", index=False)
    best_trades.to_csv(OUT_DIR / "best_rule_trades.csv", index=False)
    day_pnl.to_csv(OUT_DIR / "best_rule_day_pnl.csv", index=False)

    report = {
        "code": CODE,
        "horizon_min": HORIZON,
        "base_rule": {
            "description": "top1% confidence + breadth_align + minute_idx>=30 + shock_cap_q90 + long_only",
            **summarize(base),
        },
        "best_rule": best.to_dict(),
        "artifacts": {
            "rule_grid": str((OUT_DIR / "rule_grid.csv").resolve()),
            "best_rule": str((OUT_DIR / "best_rule.csv").resolve()),
            "best_rule_trades": str((OUT_DIR / "best_rule_trades.csv").resolve()),
            "best_rule_day_pnl": str((OUT_DIR / "best_rule_day_pnl.csv").resolve()),
        },
    }
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
