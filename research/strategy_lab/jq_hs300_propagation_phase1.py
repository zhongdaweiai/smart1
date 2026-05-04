#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JoinQuant Phase 1 research script for HS300 intraday propagation signals.

Paste this file into a JoinQuant minute-level backtest/research strategy.
It does not trade. It builds a minute panel for 510300.XSHG using HS300
constituent propagation features, then writes the panel to:

    hs300_propagation_phase1.csv

Design focus:
- H = 5 and H = 10 minute horizons.
- Gate / Direction / Exhaustion are separated.
- The primary alpha candidates are IPG, LFR, WPS, EP, SE, and PAD.
- Same-minute-bucket rolling robust z-score uses only previous days/bars.

Important JoinQuant assumptions:
- Run at minute frequency.
- `history` returns completed minute bars. Signals are therefore computed from
  the latest completed bar and labels are added after the day is finished.
- `get_index_weights` availability differs by account/data plan. If it fails,
  the script falls back to equal-weight HS300 constituents and records
  `weight_source = "equal_fallback"`.
"""

from collections import deque
import math

try:
    from jqdata import *  # type: ignore  # noqa: F401,F403
except Exception:
    # Allows local syntax checks outside JoinQuant.
    pass

import numpy as np
import pandas as pd


INDEX_CODE = "000300.XSHG"
ETF_CODE = "510300.XSHG"
OUTPUT_FILE = "hs300_propagation_phase1.csv"
HORIZONS = (5, 10)


def initialize(context):
    set_benchmark(INDEX_CODE)
    set_option("use_real_price", True)

    g.index_code = INDEX_CODE
    g.etf_code = ETF_CODE
    g.output_file = OUTPUT_FILE

    # Data / feature settings.
    g.history_count = 36
    g.amount_min = 100000.0
    g.min_active_constituents = 120
    g.tick_size_stock = 0.01
    g.eps_floor = {5: 0.00020, 10: 0.00030}
    g.accel_window = {5: 5, 10: 10}
    g.corr_window = {5: 5, 10: 10}
    g.topk_k = 10
    g.leader_n = 30
    g.weight_bins = 5

    # Same time bucket robust z-score settings.
    g.z_window_days = 20
    g.z_min_history = 5
    g.z_hist = {}

    # Label settings. These are deliberately conservative defaults; in later
    # research you can replace them with same-bucket rolling quantile thresholds.
    g.gamma = {5: 0.00040, 10: 0.00060}
    g.trendness_th = {5: 0.58, 10: 0.62}

    # Score thresholds from the design memo.
    g.gate_enter = {5: 0.60, 10: 0.55}
    g.gate_keep = {5: 0.40, 10: 0.35}
    g.dir_th = {5: 0.50, 10: 0.45}
    g.exh_th = {5: 0.80, 10: 0.90}

    # Optional IF controls are intentionally off in Phase 1 scoring. Turn on
    # only after the stock/ETF signal panel is stable.
    g.use_if_features = False

    g.current_date = None
    g.universe = None
    g.daily_rows = []
    g.output_initialized = False
    g.row_counter = 0


def before_trading_start(context):
    trade_date = context.current_dt.date()
    g.current_date = trade_date
    g.daily_rows = []
    g.row_counter = 0
    g.universe = load_daily_universe(trade_date)
    log.info(
        "HS300 propagation Phase1 universe: date=%s n=%s source=%s"
        % (trade_date, len(g.universe), getattr(g, "weight_source", "unknown"))
    )


def handle_data(context, data):
    if g.universe is None or len(g.universe) == 0:
        return
    row = build_signal_row(context)
    if row is None:
        return
    g.daily_rows.append(row)
    g.row_counter += 1

    # Lightweight chart diagnostics in the JoinQuant backtest UI.
    try:
        record(
            Gate5=row.get("GateScore_5", np.nan),
            Dir5=row.get("DirScore_5", np.nan),
            Exh5=row.get("Exhaustion_5", np.nan),
        )
    except Exception:
        pass


def after_trading_end(context):
    if not g.daily_rows:
        return

    day = pd.DataFrame(g.daily_rows).sort_values("datetime").reset_index(drop=True)
    day = add_forward_labels(day)
    append_output(day)
    log.info(
        "HS300 propagation Phase1 wrote %s rows for %s to %s"
        % (len(day), g.current_date, g.output_file)
    )


def load_daily_universe(trade_date):
    stocks = list(get_index_stocks(g.index_code, date=trade_date))
    if not stocks:
        return pd.DataFrame(columns=["code", "weight", "industry", "leader_flag", "weight_bin"])

    weights, weight_source = load_index_weights(stocks, trade_date)
    industry = load_industry_map(stocks, trade_date)

    uni = pd.DataFrame({"code": stocks})
    uni["weight"] = uni["code"].map(weights).astype(float)
    if not np.isfinite(uni["weight"]).all() or uni["weight"].sum() <= 0:
        uni["weight"] = 1.0 / len(uni)
        g.weight_source = "equal_fallback"
    else:
        uni["weight"] = uni["weight"] / uni["weight"].sum()
        g.weight_source = weight_source

    uni["industry"] = uni["code"].map(industry).fillna("Unknown").astype(str)
    uni = uni.sort_values("weight", ascending=False).reset_index(drop=True)
    uni["leader_flag"] = 0
    uni.loc[: max(g.leader_n - 1, 0), "leader_flag"] = 1
    uni["weight_bin"] = build_weight_bins(uni["weight"], g.weight_bins)
    return uni


def load_index_weights(stocks, trade_date):
    weights = pd.Series(1.0 / len(stocks), index=stocks, dtype=float)
    try:
        wdf = get_index_weights(g.index_code, date=trade_date)
    except Exception as exc:
        log.warn("get_index_weights failed, using equal weights: %s" % exc)
        return weights, "equal_fallback"

    if wdf is None or len(wdf) == 0:
        return weights, "equal_fallback"

    try:
        if "code" in wdf.columns:
            code_col = "code"
        elif "con_code" in wdf.columns:
            code_col = "con_code"
        else:
            code_col = None

        if "weight" in wdf.columns:
            weight_col = "weight"
        elif "weight_pct" in wdf.columns:
            weight_col = "weight_pct"
        else:
            weight_col = None

        if weight_col is None:
            return weights, "equal_fallback"

        if code_col is not None:
            raw = pd.Series(wdf[weight_col].astype(float).values, index=wdf[code_col].astype(str))
        else:
            raw = pd.Series(wdf[weight_col].astype(float).values, index=wdf.index.astype(str))

        raw = raw.reindex(stocks).fillna(0.0)
        if raw.max() > 1.0:
            raw = raw / 100.0
        if raw.sum() <= 0:
            return weights, "equal_fallback"
        raw = raw / raw.sum()
        return raw, "index_weights"
    except Exception as exc:
        log.warn("parse get_index_weights failed, using equal weights: %s" % exc)
        return weights, "equal_fallback"


def load_industry_map(stocks, trade_date):
    out = {s: "Unknown" for s in stocks}
    try:
        ind = get_industry(stocks, date=trade_date)
    except Exception:
        return pd.Series(out)

    for code, payload in ind.items():
        name = "Unknown"
        try:
            if payload.get("sw_l1"):
                name = payload["sw_l1"].get("industry_name", "Unknown")
            elif payload.get("zjw"):
                name = payload["zjw"].get("industry_name", "Unknown")
        except Exception:
            name = "Unknown"
        out[code] = name or "Unknown"
    return pd.Series(out)


def build_weight_bins(weights, bins):
    n = len(weights)
    if n == 0:
        return np.array([], dtype=int)
    q = min(int(bins), n)
    ranks = pd.Series(weights).rank(method="first", ascending=True)
    try:
        return pd.qcut(ranks, q=q, labels=False, duplicates="drop").astype(int).values
    except Exception:
        return np.zeros(n, dtype=int)


def build_signal_row(context):
    stocks = list(g.universe["code"])
    securities = stocks + [g.etf_code]
    close = safe_history(g.history_count, "close", securities)
    volume = safe_history(g.history_count, "volume", stocks)
    money = safe_history(g.history_count, "money", stocks)

    if close is None or volume is None or money is None:
        return None
    if g.etf_code not in close.columns:
        return None

    stock_close = close.reindex(columns=stocks)
    volume = volume.reindex(columns=stocks).fillna(0.0)
    money = money.reindex(columns=stocks).fillna(0.0)
    etf_close = close[g.etf_code].astype(float)

    if len(stock_close) < max(HORIZONS) + 2 or len(etf_close.dropna()) < max(HORIZONS) + 2:
        return None

    minute_idx = g.row_counter
    now = pd.Timestamp(stock_close.index[-1])
    row = {
        "date": pd.Timestamp(g.current_date),
        "datetime": now,
        "minute_idx": minute_idx,
        "index_code": g.index_code,
        "etf_code": g.etf_code,
        "weight_source": getattr(g, "weight_source", "unknown"),
        "etf_close": float(etf_close.iloc[-1]),
        "valid_signal_time": int(is_regular_signal_time(now)),
    }

    base_weights = g.universe["weight"].values.astype(float)
    industries = g.universe["industry"].fillna("Unknown").astype(str).values
    leader_mask = g.universe["leader_flag"].values.astype(int).astype(bool)
    weight_bins = g.universe["weight_bin"].values.astype(int)

    close_arr = stock_close.values.astype(float)
    volume_arr = volume.values.astype(float)
    money_arr = money.values.astype(float)
    log_close = np.log(close_arr)
    ret1 = log_close[-1] - log_close[-2]

    active_now = (
        np.isfinite(close_arr[-1])
        & (volume_arr[-1] > 0.0)
        & (money_arr[-1] >= g.amount_min)
    )

    for horizon in HORIZONS:
        hvals = compute_horizon_row(
            horizon=horizon,
            close_arr=close_arr,
            log_close=log_close,
            ret1=ret1,
            volume_arr=volume_arr,
            money_arr=money_arr,
            active_now=active_now,
            base_weights=base_weights,
            industries=industries,
            leader_mask=leader_mask,
            weight_bins=weight_bins,
            etf_close=etf_close,
            minute_idx=minute_idx,
        )
        row.update(hvals)

    score_row(row, minute_idx)
    row["state"] = compute_state(row)

    if g.use_if_features:
        add_if_features(row, context, minute_idx)

    return row


def safe_history(count, field, securities):
    try:
        df = history(count, unit="1m", field=field, security_list=securities, df=True)
    except Exception as exc:
        log.warn("history failed field=%s: %s" % (field, exc))
        return None
    if isinstance(df, pd.Series):
        df = df.to_frame(securities[0])
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def compute_horizon_row(
    horizon,
    close_arr,
    log_close,
    ret1,
    volume_arr,
    money_arr,
    active_now,
    base_weights,
    industries,
    leader_mask,
    weight_bins,
    etf_close,
    minute_idx,
):
    n_stock = close_arr.shape[1]
    trade_recent = np.nansum(volume_arr[-horizon:] > 0.0, axis=0) > 0
    ret_h = log_close[-1] - log_close[-1 - horizon]
    eps = np.maximum((1.5 * g.tick_size_stock) / np.maximum(close_arr[-1], 1e-6), g.eps_floor[horizon])

    sign_h = np.zeros(n_stock, dtype=float)
    sign_h[ret_h > eps] = 1.0
    sign_h[ret_h < -eps] = -1.0

    active = active_now & trade_recent & np.isfinite(ret_h)
    active_count = int(active.sum())
    out = {"active_count_%s" % horizon: active_count}
    suffix = "_%s_raw" % horizon

    if active_count < g.min_active_constituents:
        for name in ["B", "dB", "Corr", "EP", "TopK", "SE", "Dir", "WSB", "LFR", "WPS", "ETFRet"]:
            out[name + suffix] = np.nan
        return out

    w = normalize_weights(base_weights * active.astype(float))
    active_sign = sign_h[active]
    up = float(np.mean(active_sign == 1.0))
    down = float(np.mean(active_sign == -1.0))
    breadth = abs(up - down)

    hist_b = [r.get("B_%s_raw" % horizon) for r in g.daily_rows[-g.accel_window[horizon] :]]
    hist_b = [x for x in hist_b if is_finite(x)]
    db = breadth - float(np.mean(hist_b)) if hist_b else 0.0

    corr = compute_fast_corr(log_close, active, horizon)

    weighted_ret = w * np.where(np.isfinite(ret_h), ret_h, 0.0)
    contrib_abs = np.abs(weighted_ret)
    total_abs = contrib_abs.sum()
    total_sq = np.square(contrib_abs).sum()
    ep_raw = (total_abs * total_abs) / (active_count * total_sq) if total_sq > 0 else np.nan
    ep = (ep_raw - 1.0 / active_count) / (1.0 - 1.0 / active_count) if active_count > 1 and is_finite(ep_raw) else np.nan

    k = min(g.topk_k, len(contrib_abs))
    topk = np.nan
    if total_abs > 0 and k > 0:
        topk = float(np.sort(contrib_abs)[-k:].sum() / total_abs)

    se = compute_sector_entropy(weighted_ret, industries, active)

    etf_log = np.log(etf_close.astype(float))
    etf_ret_h = float(etf_log.iloc[-1] - etf_log.iloc[-1 - horizon])
    etf_r1 = etf_log.diff().iloc[-horizon:]
    etf_path_abs = float(np.abs(etf_r1).sum())
    dir_strength = abs(etf_ret_h) / etf_path_abs if etf_path_abs > 0 else np.nan

    wsb = float(np.sum(w * sign_h))
    lfr = compute_lfr(log_close, ret1, active_now, base_weights, leader_mask)
    wps = compute_wps(sign_h, active, base_weights, weight_bins, wsb)

    out.update(
        {
            "B" + suffix: breadth,
            "dB" + suffix: db,
            "Corr" + suffix: corr,
            "EP" + suffix: ep,
            "TopK" + suffix: topk,
            "SE" + suffix: se,
            "Dir" + suffix: dir_strength,
            "WSB" + suffix: wsb,
            "LFR" + suffix: lfr,
            "WPS" + suffix: wps,
            "ETFRet" + suffix: etf_ret_h,
        }
    )
    return out


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = np.nansum(weights)
    if total <= 0:
        return np.zeros_like(weights, dtype=float)
    return np.nan_to_num(weights / total)


def compute_fast_corr(log_close, active, window):
    if len(log_close) < window + 1 or active.sum() < 3:
        return np.nan
    r1 = np.diff(log_close[-(window + 1) :], axis=0)
    r1 = r1[:, active]
    if r1.shape[1] < 3:
        return np.nan
    ew = np.nanmean(r1, axis=1)
    var_ew = float(np.nanvar(ew))
    var_i = float(np.nanmean(np.nanvar(r1, axis=0)))
    n = float(r1.shape[1])
    if var_i <= 0 or n <= 1:
        return np.nan
    corr = (n * var_ew / var_i - 1.0) / (n - 1.0)
    return float(np.clip(corr, -1.0, 1.0))


def compute_sector_entropy(weighted_ret, industries, active):
    sector_sum = {}
    for ret, ind, ok in zip(weighted_ret, industries, active):
        if not ok or not is_finite(ret):
            continue
        sector_sum[ind] = sector_sum.get(ind, 0.0) + float(ret)
    vals = np.array([abs(v) for v in sector_sum.values() if abs(v) > 0], dtype=float)
    if len(vals) <= 1:
        return np.nan
    p = vals / vals.sum()
    return float(-np.sum(p * np.log(p)) / math.log(len(vals)))


def compute_lfr(log_close, ret1, active_now, base_weights, leader_mask):
    if len(log_close) < 22:
        return np.nan
    leaders = leader_mask & active_now
    followers = (~leader_mask) & active_now
    if leaders.sum() < 3 or followers.sum() < 20:
        return np.nan

    w_lead = normalize_weights(base_weights * leaders.astype(float))
    w_follow = normalize_weights(base_weights * followers.astype(float))

    # Previous 2-minute leader move: close(t-1) vs close(t-3).
    r_lead_prev_2m = log_close[-2] - log_close[-4]
    lead_prev = float(np.sum(w_lead * np.nan_to_num(r_lead_prev_2m)))

    follow_now = float(np.sum(w_follow * np.nan_to_num(ret1)))

    follow_series = []
    for j in range(1, 21):
        rr = log_close[-j] - log_close[-j - 1]
        follow_series.append(float(np.sum(w_follow * np.nan_to_num(rr))))
    sigma = float(np.std(follow_series))
    if sigma <= 1e-10:
        return np.nan
    return float(np.sign(lead_prev) * follow_now / sigma)


def compute_wps(sign_h, active, base_weights, weight_bins, wsb):
    xs = []
    ys = []
    unique_bins = sorted(set(weight_bins.tolist()))
    if len(unique_bins) < 2:
        return np.nan
    centered_x = np.linspace(-2.0, 2.0, len(unique_bins))
    for x, b in zip(centered_x, unique_bins):
        mask = active & (weight_bins == b)
        if mask.sum() == 0:
            continue
        w_bin = normalize_weights(base_weights * mask.astype(float))
        ys.append(float(np.sum(w_bin * sign_h)))
        xs.append(float(x))
    if len(xs) < 2:
        return np.nan
    x = np.asarray(xs)
    y = np.asarray(ys)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    slope = float(np.dot(x, y) / denom)
    return float(np.sign(wsb) * slope)


def score_row(row, minute_idx):
    for horizon in HORIZONS:
        raw_names = ["B", "dB", "Corr", "EP", "TopK", "SE", "Dir", "WSB", "LFR", "WPS", "ETFRet"]
        z = {}
        for name in raw_names:
            raw_col = "%s_%s_raw" % (name, horizon)
            z_col = "z_%s_%s" % (name, horizon)
            z[name] = zscore_update(z_col, minute_idx, row.get(raw_col, np.nan))
            row[z_col] = z[name]

        internal = (
            0.30 * z["WSB"]
            + 0.20 * z["dB"]
            + 0.20 * z["Corr"]
            + 0.15 * z["EP"]
            + 0.15 * z["SE"]
        )
        ipg = internal - z["ETFRet"]
        pad = z["ETFRet"] - internal

        row["InternalPressure_%s" % horizon] = internal
        row["IPG_%s_raw" % horizon] = ipg
        row["PAD_%s_raw" % horizon] = pad
        row["z_IPG_%s" % horizon] = zscore_update("z_IPG_%s" % horizon, minute_idx, ipg)
        row["z_PAD_%s" % horizon] = zscore_update("z_PAD_%s" % horizon, minute_idx, pad)

        row["GateScore_%s" % horizon] = (
            0.22 * z["B"]
            + 0.18 * z["dB"]
            + 0.18 * z["Corr"]
            + 0.14 * z["EP"]
            + 0.12 * z["SE"]
            + 0.08 * z["Dir"]
            - 0.08 * z["TopK"]
        )
        row["DirScore_%s" % horizon] = (
            0.40 * row["z_IPG_%s" % horizon]
            + 0.30 * z["LFR"]
            + 0.20 * z["WPS"]
            + 0.10 * z["WSB"]
        )
        row["Exhaustion_%s" % horizon] = (
            0.50 * max(row["z_PAD_%s" % horizon], 0.0)
            + 0.30 * max(z["TopK"], 0.0)
            + 0.20 * max(-z["dB"], 0.0)
        )
        row["Gate_%s" % horizon] = int(row["GateScore_%s" % horizon] > g.gate_enter[horizon])
        row["Keep_%s" % horizon] = int(row["GateScore_%s" % horizon] > g.gate_keep[horizon])


def zscore_update(name, bucket, value):
    if not is_finite(value):
        return 0.0

    by_bucket = g.z_hist.setdefault(name, {})
    hist = by_bucket.setdefault(int(bucket), deque(maxlen=g.z_window_days))
    arr = np.asarray(list(hist), dtype=float)

    if len(arr) >= g.z_min_history:
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        scale = 1.4826 * mad
        if scale <= 1e-10:
            sd = float(np.std(arr))
            scale = sd if sd > 1e-10 else 1.0
        z = (float(value) - med) / scale
    elif len(arr) >= 2:
        mu = float(np.mean(arr))
        sd = float(np.std(arr))
        z = (float(value) - mu) / (sd if sd > 1e-10 else 1.0)
    else:
        z = 0.0

    hist.append(float(value))
    return float(np.clip(z, -8.0, 8.0))


def compute_state(row):
    gate5 = bool(row.get("Gate_5", 0))
    gate10 = bool(row.get("Gate_10", 0))
    dir5 = float(row.get("DirScore_5", 0.0))
    dir10 = float(row.get("DirScore_10", 0.0))
    exh5 = float(row.get("Exhaustion_5", 0.0))
    exh10 = float(row.get("Exhaustion_10", 0.0))
    db5 = float(row.get("z_dB_5", 0.0))
    lfr5 = float(row.get("z_LFR_5", 0.0))

    if (gate5 or gate10) and (exh5 > g.exh_th[5] or exh10 > g.exh_th[10] or (db5 < -0.25 and lfr5 < -0.25)):
        return "EXHAUSTED"
    if gate5 and gate10 and dir5 > g.dir_th[5] and dir10 > g.dir_th[10] and exh10 < g.exh_th[10]:
        return "CONFIRMED_UP"
    if gate5 and gate10 and dir5 < -g.dir_th[5] and dir10 < -g.dir_th[10] and exh10 < g.exh_th[10]:
        return "CONFIRMED_DOWN"
    if gate5 and dir5 > g.dir_th[5] and dir10 > -g.dir_th[10] and exh5 < g.exh_th[5]:
        return "EMERGING_UP"
    if gate5 and dir5 < -g.dir_th[5] and dir10 < g.dir_th[10] and exh5 < g.exh_th[5]:
        return "EMERGING_DOWN"
    return "NOISE"


def add_if_features(row, context, minute_idx):
    try:
        if_code = get_dominant_future("IF")
        if_close = safe_history(g.history_count, "close", [if_code])
        if if_close is None or if_code not in if_close.columns:
            return
        ser = if_close[if_code].astype(float)
        if len(ser) < max(HORIZONS) + 1:
            return
        log_ser = np.log(ser)
        for horizon in HORIZONS:
            raw = float(log_ser.iloc[-1] - log_ser.iloc[-1 - horizon])
            row["IFRet_%s_raw" % horizon] = raw
            row["z_IFRet_%s" % horizon] = zscore_update("z_IFRet_%s" % horizon, minute_idx, raw)
            row["IF_minus_ETF_%s_raw" % horizon] = raw - float(row.get("ETFRet_%s_raw" % horizon, np.nan))
    except Exception:
        return


def add_forward_labels(day):
    close = day["etf_close"].astype(float)
    log_close = np.log(close)
    r1 = log_close.diff()

    for horizon in HORIZONS:
        fwd = log_close.shift(-horizon) - log_close
        trendness = []
        for i in range(len(day)):
            future = r1.iloc[i + 1 : i + 1 + horizon]
            if len(future) < horizon or future.isna().any():
                trendness.append(np.nan)
                continue
            denom = float(np.abs(future).sum())
            trendness.append(float(abs(future.sum()) / denom) if denom > 0 else np.nan)

        day["R_%s_fwd" % horizon] = fwd
        day["Trendness_%s_fwd" % horizon] = trendness
        day["Y_gate_%s" % horizon] = (
            (day["R_%s_fwd" % horizon].abs() > g.gamma[horizon])
            & (day["Trendness_%s_fwd" % horizon] > g.trendness_th[horizon])
        ).astype(int)
        day["Y_dir_%s" % horizon] = np.where(
            day["Y_gate_%s" % horizon] == 1,
            np.sign(day["R_%s_fwd" % horizon]),
            0,
        ).astype(int)
    return day


def append_output(day):
    content = day.to_csv(index=False, header=not g.output_initialized)
    try:
        write_file(g.output_file, content, append=g.output_initialized)
        g.output_initialized = True
    except Exception as exc:
        log.warn("write_file failed: %s" % exc)
        log.info(day.tail(3).to_string())


def is_regular_signal_time(dt):
    hhmm = dt.hour * 100 + dt.minute
    if hhmm < 935:
        return False
    if 1125 <= hhmm <= 1305:
        return False
    if hhmm >= 1450:
        return False
    return True


def is_finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False
