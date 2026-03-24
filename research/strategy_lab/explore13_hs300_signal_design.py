"""
Explore 13: HS300 Signal Redesign
==================================
Explore 12 showed breadth_resid doesn't predict real HS300 index.
Now systematically test MANY breadth signal variants against HS300 returns.

Approach: IC study first (cheap), then backtest top signals.

Signal variants:
  1. breadth_resid_ew:     original (equal-weight binary, regress on ew_return)
  2. breadth_resid_cw:     cap-weighted binary breadth, regress on cw_return
  3. breadth_top30:        breadth from top 30 stocks only
  4. breadth_top50:        breadth from top 50 stocks only
  5. breadth_bottom200:    breadth from bottom 200 stocks only
  6. breadth_deficit:      breadth_bottom200 - breadth_top30 (cross-tier)
  7. breadth_deficit_resid: breadth_deficit regressed on cw_return
  8. intensity_ew:         avg(return) signed, not binary
  9. intensity_cw:         cap-weighted avg(return), regressed on cw_return
  10. breadth_cw_resid_ew: cap-weighted breadth resid, regressed on EQUAL-weight return
  11. breadth_ew_resid_cw: equal-weight breadth resid, regressed on CAP-weight return
  12. top30_breadth_resid: breadth_top30 regressed on top30 return (pure large-cap)
  13. neg_breadth_resid:   FLIPPED original signal (reversal hypothesis)

Target: HS300 cap-weighted return, forward 1/2/3/5/8 bars
"""

import os, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore13')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROUND_TRIP_COST_BPS = 4.5
MAX_SCALE = 3.0
DAILY_LOSS_LIMIT_BPS = -50


def load_hs300_weights():
    fp = os.path.join(DATA_DIR, 'hs300_weights.csv')
    df = pd.read_csv(fp)
    weights = dict(zip(df['code'], df['weight_pct'] / 100.0))
    codes = set(df['code'].values)
    # Sort by weight for tier assignment
    df_sorted = df.sort_values('weight_pct', ascending=False)
    top30 = set(df_sorted.head(30)['code'].values)
    top50 = set(df_sorted.head(50)['code'].values)
    bottom200 = set(df_sorted.tail(200)['code'].values)
    return codes, weights, top30, top50, bottom200


def load_day(fp):
    df = pd.read_parquet(fp)
    df = df[df['paused'] == 0].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def compute_signals(fp, hs300_codes, hs300_weights, top30, top50, bottom200):
    """Compute ALL signal variants for one day."""
    df = load_day(fp)
    df = df[df['code'].isin(hs300_codes)].copy()

    pc = df.pivot_table(index='datetime', columns='code', values='close')
    ret1 = pc.pct_change()

    n_bars = len(ret1)
    if n_bars < 30:
        return None

    # --- Index returns ---
    # Cap-weighted (HS300 official)
    avail = [c for c in pc.columns if c in hs300_weights]
    w = np.array([hs300_weights[c] for c in avail])
    w = w / w.sum()
    cw_ret = pd.Series(ret1[avail].values @ w, index=ret1.index)
    cw_lvl = (1 + cw_ret).cumprod()
    cw_lvl.iloc[0] = 1.0

    # Equal-weighted
    ew_ret = ret1.mean(axis=1)

    # --- Tier-specific returns and breadth ---
    top30_avail = [c for c in avail if c in top30]
    top50_avail = [c for c in avail if c in top50]
    bot200_avail = [c for c in avail if c in bottom200]

    # Top 30 cap-weighted return
    w30 = np.array([hs300_weights[c] for c in top30_avail])
    w30 = w30 / w30.sum()
    top30_ret = pd.Series(ret1[top30_avail].values @ w30, index=ret1.index)

    # Forward returns for IC calculation
    fwd = {}
    for h in [1, 2, 3, 5, 8]:
        fwd[h] = cw_ret.shift(-h - 1)  # -1 because we enter NEXT bar (no look-ahead)
        # Actually: signal at bar i → enter at i+1 → exit at i+1+h
        # Forward return = sum of cw_ret from i+1 to i+h (inclusive? need to be careful)
        # Simpler: forward cumulative return from bar i+1 over h bars
        fwd_cum = cw_lvl.shift(-h - 1) / cw_lvl.shift(-1) - 1
        fwd[h] = fwd_cum

    # --- Compute all signal variants ---
    ns_all = ret1.count(axis=1)
    bu_all = (ret1 > 0).sum(axis=1) / ns_all  # equal-weight binary breadth

    # Cap-weighted binary breadth
    up_mask = (ret1[avail] > 0).astype(float)
    bu_cw = (up_mask.values * w).sum(axis=1)
    bu_cw = pd.Series(bu_cw, index=ret1.index)

    # Tier breadth
    ns_t30 = ret1[top30_avail].count(axis=1)
    bu_t30 = (ret1[top30_avail] > 0).sum(axis=1) / ns_t30.clip(lower=1)
    ns_t50 = ret1[top50_avail].count(axis=1)
    bu_t50 = (ret1[top50_avail] > 0).sum(axis=1) / ns_t50.clip(lower=1)
    ns_b200 = ret1[bot200_avail].count(axis=1)
    bu_b200 = (ret1[bot200_avail] > 0).sum(axis=1) / ns_b200.clip(lower=1)

    # Intensity (average return, not binary)
    int_ew = ret1.mean(axis=1)  # same as ew_ret
    int_cw = cw_ret  # same as cw_ret

    signals = {}

    # Helper: expanding regression residual
    def resid(y_series, x_series, min_obs=10):
        r = y_series.copy()
        r[:] = np.nan
        for i in range(min_obs, len(y_series)):
            yy = y_series.iloc[:i]
            xx = x_series.iloc[:i]
            mask = yy.notna() & xx.notna()
            if mask.sum() > 5:
                sl = np.polyfit(xx[mask], yy[mask], 1)
                r.iloc[i] = y_series.iloc[i] - np.polyval(sl, x_series.iloc[i])
        return r

    # 1. Original: EW breadth resid on EW return
    signals['breadth_resid_ew'] = resid(bu_all, ew_ret)

    # 2. CW breadth resid on CW return
    signals['breadth_resid_cw'] = resid(bu_cw, cw_ret)

    # 3-5. Tier breadth (raw, no regression for speed)
    signals['breadth_top30'] = bu_t30
    signals['breadth_top50'] = bu_t50
    signals['breadth_bottom200'] = bu_b200

    # 6. Cross-tier deficit
    signals['breadth_deficit'] = bu_b200 - bu_t30

    # 7. Deficit regressed on CW return
    deficit = bu_b200 - bu_t30
    signals['deficit_resid_cw'] = resid(deficit, cw_ret)

    # 8-9. Intensity residuals
    signals['intensity_ew_resid'] = resid(int_ew, cw_ret)  # EW intensity resid on CW
    # intensity_cw_resid would be residual of cw_ret on itself = 0, skip

    # 10. CW breadth resid on EW return
    signals['breadth_cw_resid_ew'] = resid(bu_cw, ew_ret)

    # 11. EW breadth resid on CW return
    signals['breadth_ew_resid_cw'] = resid(bu_all, cw_ret)

    # 12. Top30 breadth resid on top30 return
    signals['top30_breadth_resid'] = resid(bu_t30, top30_ret)

    # 13. Flipped original
    r_ew = resid(bu_all, ew_ret)
    signals['neg_breadth_resid'] = -r_ew

    # 14. Bottom200 breadth resid on CW return (cross-tier prediction)
    signals['bot200_resid_cw'] = resid(bu_b200, cw_ret)

    # 15. Top30 breadth resid on CW return
    signals['top30_resid_cw'] = resid(bu_t30, cw_ret)

    # 16. Breadth spread: all-stock breadth - top30 breadth (consensus gap)
    signals['breadth_spread'] = bu_all - bu_t30

    # 17. Breadth spread resid on CW return
    spread = bu_all - bu_t30
    signals['spread_resid_cw'] = resid(spread, cw_ret)

    return pd.DataFrame(signals, index=ret1.index), fwd, cw_lvl, cw_ret


def backtest_day_fixed(signals_series, price_series, th_long, th_short,
                        hold_bars, scale, max_sc, cost_bps, loss_limit_bps):
    """No look-ahead backtest."""
    n = len(signals_series)
    pos = 0.0; ebar = -1; eprice = 0
    cum = 0; n_trades = 0; stopped = False
    pending_entry = None

    for bar in range(10, n):
        px = price_series.iloc[bar]
        if stopped: break
        if pending_entry is not None and pos == 0:
            pos = pending_entry; ebar = bar; eprice = px; pending_entry = None
        if pos != 0:
            held = bar - ebar
            eod = bar >= n - 2
            if held >= hold_bars or eod:
                d = 1 if pos > 0 else -1; sz = abs(pos)
                gr = d * (px / eprice - 1)
                nr = gr * sz - cost_bps / 10000
                cum += nr; n_trades += 1; pos = 0
                if cum * 10000 < loss_limit_bps:
                    stopped = True; continue
        if pos == 0 and pending_entry is None and bar < n - hold_bars - 2 and not stopped:
            sig = signals_series.iloc[bar]
            if np.isnan(sig): continue
            if sig > th_long:
                sc = min(abs(sig) / abs(th_long), max_sc) if scale else 1.0
                pending_entry = sc
            elif sig < th_short:
                sc = min(abs(sig) / abs(th_short), max_sc) if scale else 1.0
                pending_entry = -sc
    if pos != 0:
        px = price_series.iloc[-1]
        d = 1 if pos > 0 else -1; sz = abs(pos)
        gr = d * (px / eprice - 1)
        nr = gr * sz - cost_bps / 10000
        cum += nr; n_trades += 1
    return cum, n_trades


def compute_metrics(pnls):
    arr = np.array(pnls)
    if len(arr) == 0 or arr.std() == 0:
        return {'ret': 0, 'sharpe': 0, 'dd': 0, 'wr': 0, 'avg_bps': 0}
    cum = (1 + pd.Series(pnls)).cumprod()
    ret = cum.iloc[-1] - 1
    sh = arr.mean() / arr.std() * np.sqrt(242)
    dd = (cum / cum.cummax() - 1).min()
    wr = (arr > 0).mean()
    return {'ret': ret, 'sharpe': sh, 'dd': dd, 'wr': wr, 'avg_bps': arr.mean() * 10000}


def main():
    t0 = time.time()
    print("=" * 120)
    print("Explore 13: HS300 Signal Redesign — IC Study + Backtest")
    print("=" * 120)

    hs300_codes, hs300_weights, top30, top50, bottom200 = load_hs300_weights()
    print(f"HS300: {len(hs300_codes)} stocks, top30: {len(top30)}, bottom200: {len(bottom200)}")

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    N = 201
    recent = all_files[-N:]

    # Process all days
    print("Processing daily signals... ", end="", flush=True)
    all_signals = []
    all_fwd = []
    all_cw_lvl = []
    all_cw_ret = []
    all_dates = []

    for i in range(1, len(recent)):
        fp = os.path.join(STOCK_DATA_DIR, recent[i])
        date_str = recent[i].replace('.parquet', '')
        try:
            result = compute_signals(fp, hs300_codes, hs300_weights, top30, top50, bottom200)
            if result is None:
                continue
            sigs, fwd, cw_lvl, cw_ret = result
            all_signals.append(sigs)
            all_fwd.append(fwd)
            all_cw_lvl.append(cw_lvl)
            all_cw_ret.append(cw_ret)
            all_dates.append(date_str)
        except Exception as e:
            continue

    print(f"done ({len(all_signals)} days, {time.time()-t0:.0f}s)")

    # Split
    split = len(all_signals) // 2
    test_signals = all_signals[split:]
    test_fwd = all_fwd[split:]
    test_cw_lvl = all_cw_lvl[split:]
    test_dates = all_dates[split:]

    # ============================
    # PART 1: IC Study
    # ============================
    print(f"\n{'='*120}")
    print("PART 1: IC STUDY — Which signals predict HS300 forward returns?")
    print("=" * 120)
    print("  IC = rank correlation (Spearman) between signal and forward return")
    print("  Computed on OOS test data, bar-by-bar within each day, then averaged")

    signal_names = list(all_signals[0].columns)
    horizons = [1, 2, 3, 5, 8]

    ic_results = {sig: {h: [] for h in horizons} for sig in signal_names}

    for day_idx in range(len(test_signals)):
        sigs = test_signals[day_idx]
        fwd = test_fwd[day_idx]

        for sig_name in signal_names:
            s = sigs[sig_name].iloc[10:]  # skip warmup
            for h in horizons:
                f = fwd[h].iloc[10:]
                # Align
                mask = s.notna() & f.notna()
                if mask.sum() > 20:
                    # Time-series IC within the day (not cross-sectional)
                    # Use Spearman rank correlation
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(s[mask], f[mask])
                    if not np.isnan(corr):
                        ic_results[sig_name][h].append(corr)

    # Print IC table
    print(f"\n{'Signal':<28} ", end="")
    for h in horizons:
        print(f"{'IC_'+str(h):>8} {'ICIR_'+str(h):>8} {'%+_'+str(h):>7}  ", end="")
    print()
    print("-" * 120)

    ic_summary = []
    for sig_name in signal_names:
        row = {'signal': sig_name}
        print(f"{sig_name:<28} ", end="")
        for h in horizons:
            ics = ic_results[sig_name][h]
            if len(ics) > 5:
                ic_mean = np.mean(ics)
                ic_std = np.std(ics)
                icir = ic_mean / ic_std * np.sqrt(242) if ic_std > 0 else 0
                pct_pos = np.mean([x > 0 for x in ics])
                row[f'ic_{h}'] = ic_mean
                row[f'icir_{h}'] = icir
                row[f'pctpos_{h}'] = pct_pos
                # Color coding
                star = "**" if abs(icir) > 1.0 else "  "
                print(f"{ic_mean:>+7.3f}{star}{icir:>+7.1f}  {pct_pos:>5.0%}   ", end="")
            else:
                print(f"{'N/A':>8} {'N/A':>8} {'N/A':>7}  ", end="")
        print()
        ic_summary.append(row)

    ic_df = pd.DataFrame(ic_summary)
    ic_df.to_csv(os.path.join(OUTPUT_DIR, 'ic_study.csv'), index=False)

    # ============================
    # PART 2: Top signals → Backtest
    # ============================
    # Identify signals with |ICIR| > 0.5 at any horizon
    promising = []
    for row in ic_summary:
        for h in horizons:
            key = f'icir_{h}'
            if key in row and abs(row[key]) > 0.5:
                promising.append(row['signal'])
                break

    # Also always test a few key ones
    must_test = ['breadth_resid_ew', 'breadth_resid_cw', 'breadth_ew_resid_cw',
                 'top30_breadth_resid', 'breadth_deficit', 'deficit_resid_cw',
                 'bot200_resid_cw', 'spread_resid_cw', 'neg_breadth_resid',
                 'breadth_spread', 'top30_resid_cw']
    to_test = list(set(promising + must_test))

    print(f"\n{'='*120}")
    print(f"PART 2: BACKTEST — {len(to_test)} signals on HS300 index")
    print("=" * 120)

    # Calibrate on training data
    train_signals = all_signals[:split]
    calibration = {}
    for sig_name in to_test:
        vals = pd.concat([d[sig_name] for d in train_signals]).dropna()
        calibration[sig_name] = {'mean': vals.mean(), 'std': vals.std()}

    # Test configs (focused set)
    test_configs = [
        (3, 0.5, 999, True, "h3_L0.5_LO_sc"),
        (3, 0.7, 999, False, "h3_L0.7_LO_1x"),
        (5, 0.5, 999, True, "h5_L0.5_LO_sc"),
        (5, 0.5, 999, False, "h5_L0.5_LO_1x"),
        (5, 0.7, 999, True, "h5_L0.7_LO_sc"),
        (5, 0.7, 999, False, "h5_L0.7_LO_1x"),
        (5, 1.0, 999, False, "h5_L1.0_LO_1x"),
        (8, 0.5, 999, False, "h8_L0.5_LO_1x"),
        (8, 0.7, 999, False, "h8_L0.7_LO_1x"),
        (8, 1.0, 999, False, "h8_L1.0_LO_1x"),
    ]

    # Also test with NEGATIVE threshold (reversal)
    # If IC is negative, we want to SHORT when signal is high
    # Implementation: use negative thresholds → when signal < -th, go long
    # Or simpler: just flip the signal (we already have neg_breadth_resid)

    bt_results = []

    for sig_name in sorted(to_test):
        cal = calibration[sig_name]
        for hold, tl_m, ts_m, sc, config_label in test_configs:
            th_l = cal['mean'] + tl_m * cal['std']
            th_s = cal['mean'] - ts_m * cal['std'] if ts_m < 100 else -9999

            pnls = []
            tot_trades = 0
            for day_idx in range(len(test_signals)):
                sig_series = test_signals[day_idx][sig_name]
                px_series = test_cw_lvl[day_idx]
                p, nt = backtest_day_fixed(sig_series, px_series,
                                             th_l, th_s, hold, sc, MAX_SCALE,
                                             ROUND_TRIP_COST_BPS, DAILY_LOSS_LIMIT_BPS)
                pnls.append(p)
                tot_trades += nt

            if sum(1 for p in pnls if p != 0) < 10:
                continue

            m = compute_metrics(pnls)
            bt_results.append({
                'signal': sig_name, 'config': config_label,
                'hold': hold, 'th_long': tl_m, 'scaled': sc,
                **m, 'trades': tot_trades,
                'trades_per_day': tot_trades / len(test_signals),
            })

    bt_df = pd.DataFrame(bt_results)
    bt_df = bt_df.sort_values('sharpe', ascending=False)
    bt_df.to_csv(os.path.join(OUTPUT_DIR, 'backtest_results.csv'), index=False)

    # Print top results grouped by signal
    print(f"\n{'='*120}")
    print("TOP RESULTS BY SIGNAL (best config per signal)")
    print("=" * 120)
    print(f"{'Signal':<28} {'Config':<20} {'Return':>9} {'Sharpe':>7} {'MaxDD':>7} "
          f"{'WR':>5} {'T/Day':>6} {'AvgBps':>8}")
    print("-" * 100)

    seen_signals = set()
    for _, r in bt_df.iterrows():
        if r['signal'] not in seen_signals:
            seen_signals.add(r['signal'])
            star = " 🔥" if r['sharpe'] > 1.0 else (" ⚠" if r['sharpe'] > 0 else "")
            print(f"{r['signal']:<28} {r['config']:<20} {r['ret']*100:>+8.1f}% "
                  f"{r['sharpe']:>7.2f} {r['dd']*100:>6.1f}% {r['wr']:>4.0%} "
                  f"{r['trades_per_day']:>6.1f} {r['avg_bps']:>+7.1f}{star}")

    # Print overall top 20
    print(f"\n{'='*120}")
    print("OVERALL TOP 20 (all signal×config combos)")
    print("=" * 120)
    print(f"{'Rank':>4} {'Signal':<28} {'Config':<20} {'Return':>9} {'Sharpe':>7} "
          f"{'MaxDD':>7} {'WR':>5} {'T/Day':>6}")
    print("-" * 105)
    for i, (_, r) in enumerate(bt_df.head(20).iterrows()):
        print(f"{i+1:>4} {r['signal']:<28} {r['config']:<20} "
              f"{r['ret']*100:>+8.1f}% {r['sharpe']:>7.2f} "
              f"{r['dd']*100:>6.1f}% {r['wr']:>4.0%} {r['trades_per_day']:>6.1f}")

    # Any positive Sharpe?
    positive = bt_df[bt_df['sharpe'] > 0]
    print(f"\n{'='*120}")
    print(f"SUMMARY: {len(positive)}/{len(bt_df)} combos have positive Sharpe")
    if len(positive) > 0:
        print(f"Best: {positive.iloc[0]['signal']} × {positive.iloc[0]['config']} "
              f"→ Sharpe {positive.iloc[0]['sharpe']:.2f}")
    else:
        print("❌ NO signal variant produces positive Sharpe on real HS300 index")
    print("=" * 120)

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
