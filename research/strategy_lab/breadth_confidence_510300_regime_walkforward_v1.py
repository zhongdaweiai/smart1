#!/usr/bin/env python3
"""
Strict walk-forward rule selection for 510300 5-minute breadth strategy.

Rules:
- base model predictions are already day-by-day out-of-sample
- all confidence thresholds and filter thresholds are fit on trailing history only
- parameter selection is also done on trailing history only
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/daweizhong/Documents/projects")
PANEL_PATH = ROOT / "artifacts" / "market_breadth_vs_etf_2023_2026" / "panel.parquet"
PRED_PATH = ROOT / "artifacts" / "breadth_confidence_model_v1" / "predictions.parquet"
OUT_DIR = ROOT / "artifacts" / "breadth_confidence_510300_regime_walkforward_v1"
CODE = "510300.XSHG"
HORIZON = 5
LOOKBACK_DAYS = 120
MIN_HIST_TRADES = 20


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

    regime_rows = []
    for date, day in panel.groupby("date"):
        first30 = day[day["minute_idx"] < 30]
        if first30.empty:
            continue
        ret = first30["ret_cc_1"].dropna()
        trendness = np.nan
        if len(ret) and ret.abs().sum() > 0:
            trendness = float(abs(ret.sum()) / ret.abs().sum())
        regime_rows.append(
            {
                "date": date,
                "open30_flip_rate": float(first30["breadth_flip"].mean()),
                "open30_abs_bdiff_mean": float(first30["breadth_diff_1"].abs().mean()),
                "open30_trendness": trendness,
            }
        )
    regime = pd.DataFrame(regime_rows)

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


def summarize(sub: pd.DataFrame) -> dict[str, float]:
    if sub.empty:
        return {
            "n_preds": 0,
            "n_days": 0,
            "hit_rate": np.nan,
            "zero_rate": np.nan,
            "avg_signed_bps_gross": np.nan,
            "avg_signed_bps_net": np.nan,
            "median_signed_bps_net": np.nan,
            "nonzero_hit_rate": np.nan,
            "worst_day_net_bps": np.nan,
            "best_day_net_bps": np.nan,
        }
    day_net = sub.groupby("date")["net_signed_ret_bps"].sum()
    nonzero = sub[sub["signed_ret_bps"] != 0]
    return {
        "n_preds": int(len(sub)),
        "n_days": int(sub["date"].nunique()),
        "hit_rate": float((sub["signed_ret_bps"] > 0).mean()),
        "zero_rate": float((sub["signed_ret_bps"] == 0).mean()),
        "avg_signed_bps_gross": float(sub["signed_ret_bps"].mean()),
        "avg_signed_bps_net": float(sub["net_signed_ret_bps"].mean()),
        "median_signed_bps_net": float(sub["net_signed_ret_bps"].median()),
        "nonzero_hit_rate": float((nonzero["signed_ret_bps"] > 0).mean()) if len(nonzero) else np.nan,
        "worst_day_net_bps": float(day_net.min()),
        "best_day_net_bps": float(day_net.max()),
    }


def apply_rule(df: pd.DataFrame, params: dict[str, float], threshold_source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
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
        hist = hist[hist["gap_abs"] >= gap_thr]

    return work, {
        "conf_thr": conf_thr,
        "shock_thr": shock_thr,
        "flip_thr": flip_thr,
        "bdiff_thr": bdiff_thr,
        "gap_thr": gap_thr,
    }


def candidate_grid() -> list[dict[str, float]]:
    grid = []
    for conf_q, shock_q, flip_q, bdiff_q, trend_min, gap_q in product(
        (0.99,),
        (0.90,),
        (1.00, 0.70),
        (1.00, 0.90),
        (0.00, 0.10, 0.20),
        (0.00, 0.30, 0.50),
    ):
        grid.append(
            {
                "conf_q": conf_q,
                "shock_q": shock_q,
                "flip_q": flip_q,
                "bdiff_q": bdiff_q,
                "trend_min": trend_min,
                "gap_q": gap_q,
            }
        )
    return grid


def pick_best_params(hist: pd.DataFrame, grid: list[dict[str, float]]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    best_params = None
    best_summary = None
    best_key = None

    for params in grid:
        selected, _ = apply_rule(hist, params, hist)
        stats = summarize(selected)
        if stats["n_preds"] < MIN_HIST_TRADES:
            continue
        key = (
            stats["avg_signed_bps_net"],
            stats["worst_day_net_bps"],
            stats["n_preds"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_params = params
            best_summary = stats
    return best_params, best_summary


def run_walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    grid = candidate_grid()
    chosen_rows = []
    trade_rows = []

    for idx in range(LOOKBACK_DAYS, len(dates)):
        hist_dates = dates[idx - LOOKBACK_DAYS : idx]
        test_date = dates[idx]

        hist = df[df["date"].isin(hist_dates)].copy()
        test = df[df["date"] == test_date].copy()

        params, hist_stats = pick_best_params(hist, grid)
        if params is None:
            continue

        selected, thresholds = apply_rule(test, params, hist)
        chosen_row = {
            "date": test_date,
            **params,
            **thresholds,
        }
        if hist_stats:
            chosen_row.update({f"hist_{k}": v for k, v in hist_stats.items()})
        chosen_row["test_n_preds"] = int(len(selected))
        chosen_rows.append(chosen_row)

        if not selected.empty:
            day_trades = selected.copy()
            for k, v in params.items():
                day_trades[k] = v
            for k, v in thresholds.items():
                day_trades[k] = v
            trade_rows.append(day_trades)

    chosen_df = pd.DataFrame(chosen_rows).sort_values("date").reset_index(drop=True)
    trade_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    return chosen_df, trade_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    chosen_df, trade_df = run_walkforward(df)
    summary = summarize(trade_df)
    day_pnl = (
        trade_df.groupby("date", as_index=False)["net_signed_ret_bps"].sum().rename(
            columns={"net_signed_ret_bps": "day_net_signed_ret_bps"}
        )
        if not trade_df.empty
        else pd.DataFrame(columns=["date", "day_net_signed_ret_bps"])
    )

    chosen_df.to_csv(OUT_DIR / "chosen_params_by_day.csv", index=False)
    trade_df.to_csv(OUT_DIR / "walkforward_trades.csv", index=False)
    day_pnl.to_csv(OUT_DIR / "walkforward_day_pnl.csv", index=False)

    report = {
        "code": CODE,
        "horizon_min": HORIZON,
        "lookback_days": LOOKBACK_DAYS,
        "min_hist_trades": MIN_HIST_TRADES,
        "summary": summary,
        "artifacts": {
            "chosen_params_by_day": str((OUT_DIR / "chosen_params_by_day.csv").resolve()),
            "walkforward_trades": str((OUT_DIR / "walkforward_trades.csv").resolve()),
            "walkforward_day_pnl": str((OUT_DIR / "walkforward_day_pnl.csv").resolve()),
        },
    }
    with open(OUT_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
