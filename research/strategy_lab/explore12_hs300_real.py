"""
Explore 12: Real CSI 300 (HS300) Backtest
==========================================
Previous explores used top-300-by-turnover stocks and a synthetic index.
This explore uses:
  1. REAL CSI 300 constituent stocks (from akshare)
  2. CSI 300 official weights for index construction (proxy for IF futures)
  3. Same no-look-ahead execution as Explore 11

Key question: Does the alpha exist when we use the actual HS300 stocks
and trade the HS300-weighted index (≈ IF futures)?

Note: We use the 2026-02-27 constituent snapshot as proxy for the full
backtest period. HS300 rebalances every 6 months with ~90% overlap,
so this introduces minor survivorship bias for older dates.
"""

import os, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore12')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROUND_TRIP_COST_BPS = 4.5
MAX_SCALE = 3.0
DAILY_LOSS_LIMIT_BPS = -50


def load_hs300_weights():
    """Load real CSI 300 weights (snapshot from 2026-02-27)."""
    fp = os.path.join(DATA_DIR, 'hs300_weights.csv')
    df = pd.read_csv(fp)
    codes = set(df['code'].values)
    weights = dict(zip(df['code'], df['weight_pct'] / 100.0))  # normalize to sum=1
    return codes, weights


def load_day(fp):
    df = pd.read_parquet(fp)
    df = df[df['paused'] == 0].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def process_day_hs300(fp, hs300_codes, hs300_weights):
    """
    Process one day using HS300 constituents.
    - breadth: computed from HS300 stocks (equal-weight binary)
    - index: constructed using HS300 official weights (proxy for IF futures)
    """
    df = load_day(fp)
    df = df[df['code'].isin(hs300_codes)].copy()

    pc = df.pivot_table(index='datetime', columns='code', values='close')
    ret1 = pc.pct_change()

    # --- HS300-weighted index (using official weights) ---
    # Normalize weights for available stocks
    avail_codes = [c for c in pc.columns if c in hs300_weights]
    w = np.array([hs300_weights[c] for c in avail_codes])
    w = w / w.sum()  # renormalize to sum=1

    # Weight matrix (static weights, as in real index)
    wt_ret = ret1[avail_codes].values @ w  # shape: (n_bars,)
    wt_ret = pd.Series(wt_ret, index=ret1.index)
    wt_lvl = (1 + wt_ret).cumprod()
    wt_lvl.iloc[0] = 1.0

    # --- Breadth signal (equal-weight binary from HS300 stocks) ---
    ew_ret = ret1.mean(axis=1)  # equal-weight for regression
    ns = ret1.count(axis=1)
    bu = (ret1 > 0).sum(axis=1) / ns

    br = bu.copy()
    for i in range(10, len(bu)):
        b, m = bu.iloc[:i], ew_ret.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 5:
            sl = np.polyfit(m[mask], b[mask], 1)
            br.iloc[i] = bu.iloc[i] - np.polyval(sl, ew_ret.iloc[i])

    return pd.DataFrame({
        'hs300_level': wt_lvl,
        'hs300_return': wt_ret,
        'breadth_resid': br,
    }, index=pc.index)


def process_day_top300(fp, prev_fp):
    """
    Process one day using top-300-by-turnover (original method from Explore 11).
    For comparison.
    """
    # Get previous day's universe
    prev_df = load_day(prev_fp)
    dm = prev_df.groupby('code')['money'].sum()
    universe = set(dm.nlargest(300).index)

    df = load_day(fp)
    df = df[df['code'].isin(universe)].copy()

    pc = df.pivot_table(index='datetime', columns='code', values='close')
    pm = df.pivot_table(index='datetime', columns='code', values='money')
    ret1 = pc.pct_change()

    bw = pm.div(pm.sum(axis=1), axis=0)
    tw_ret = (ret1 * bw).sum(axis=1)
    tw_lvl = (1 + tw_ret).cumprod()
    tw_lvl.iloc[0] = 1.0

    ew_ret = ret1.mean(axis=1)
    ns = ret1.count(axis=1)
    bu = (ret1 > 0).sum(axis=1) / ns
    br = bu.copy()
    for i in range(10, len(bu)):
        b, m = bu.iloc[:i], ew_ret.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 5:
            sl = np.polyfit(m[mask], b[mask], 1)
            br.iloc[i] = bu.iloc[i] - np.polyval(sl, ew_ret.iloc[i])

    return pd.DataFrame({
        'tw_level': tw_lvl, 'tw_return': tw_ret,
        'breadth_resid': br,
    }, index=pc.index)


def backtest_day_fixed(day_df, price_col, th_long, th_short, hold_bars, scale,
                        max_sc, cost_bps, loss_limit_bps):
    """No look-ahead backtest (same as Explore 11)."""
    n = len(day_df)
    pos = 0.0; ebar = -1; eprice = 0
    cum = 0; n_trades = 0; stopped = False
    pending_entry = None

    for bar in range(10, n):
        px = day_df[price_col].iloc[bar]

        if stopped:
            break

        if pending_entry is not None and pos == 0:
            pos = pending_entry
            ebar = bar
            eprice = px
            pending_entry = None

        if pos != 0:
            held = bar - ebar
            eod = bar >= n - 2
            if held >= hold_bars or eod:
                d = 1 if pos > 0 else -1
                sz = abs(pos)
                gr = d * (px / eprice - 1)
                nr = gr * sz - cost_bps / 10000
                cum += nr; n_trades += 1
                pos = 0
                if cum * 10000 < loss_limit_bps:
                    stopped = True; continue

        if pos == 0 and pending_entry is None and bar < n - hold_bars - 2 and not stopped:
            sig = day_df['breadth_resid'].iloc[bar]
            if sig > th_long:
                sc = min(abs(sig) / abs(th_long), max_sc) if scale else 1.0
                pending_entry = sc
            elif sig < th_short:
                sc = min(abs(sig) / abs(th_short), max_sc) if scale else 1.0
                pending_entry = -sc

    if pos != 0:
        px = day_df[price_col].iloc[-1]
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
    print("Explore 12: Real CSI 300 (HS300) Backtest")
    print("=" * 120)
    print("""
  对比两种方法:
  A) 真实沪深300成分股 + 官方权重构建指数 (≈ IF期货)
  B) 前一天成交额top300 + 成交额加权指数 (原方法)

  均使用无未来函数的回测 (Explore 11 修复)
""")

    hs300_codes, hs300_weights = load_hs300_weights()
    print(f"HS300 stocks: {len(hs300_codes)}")

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    N = 201
    recent = all_files[-N:]

    # Process both methods
    print("Processing daily data (both methods)... ", end="", flush=True)
    hs300_daily = []
    top300_daily = []
    all_dates = []

    for i in range(1, len(recent)):
        fp = os.path.join(STOCK_DATA_DIR, recent[i])
        prev_fp = os.path.join(STOCK_DATA_DIR, recent[i-1])
        date_str = recent[i].replace('.parquet', '')
        try:
            d_hs = process_day_hs300(fp, hs300_codes, hs300_weights)
            d_top = process_day_top300(fp, prev_fp)
            hs300_daily.append(d_hs)
            top300_daily.append(d_top)
            all_dates.append(date_str)
        except Exception as e:
            continue

    print(f"done ({len(hs300_daily)} days, {time.time()-t0:.0f}s)")

    # Split
    split = len(hs300_daily) // 2
    hs_train = hs300_daily[:split]
    hs_test = hs300_daily[split:]
    top_train = top300_daily[:split]
    top_test = top300_daily[split:]
    test_dates = all_dates[split:]

    # Calibration from training data
    hs_all_r = pd.concat([d['breadth_resid'] for d in hs_train]).dropna()
    hs_mean, hs_std = hs_all_r.mean(), hs_all_r.std()
    top_all_r = pd.concat([d['breadth_resid'] for d in top_train]).dropna()
    top_mean, top_std = top_all_r.mean(), top_all_r.std()

    print(f"HS300 calibration: mean={hs_mean:.4f}, std={hs_std:.4f}")
    print(f"Top300 calibration: mean={top_mean:.4f}, std={top_std:.4f}")

    # === First: correlation between the two indices ===
    print(f"\n{'='*120}")
    print("INDEX CORRELATION: HS300-weighted vs Turnover-weighted")
    print("=" * 120)

    hs_rets = []
    top_rets = []
    for i in range(len(hs_test)):
        hr = hs300_daily[split + i]['hs300_return'].dropna()
        tr = top300_daily[split + i]['tw_return'].dropna()
        # Align
        common = hr.index.intersection(tr.index)
        hs_rets.extend(hr.loc[common].values)
        top_rets.extend(tr.loc[common].values)

    corr = np.corrcoef(hs_rets, top_rets)[0, 1]
    print(f"  Minute-bar return correlation: {corr:.4f}")
    print(f"  (1.0 = identical, <0.95 = significantly different)")

    # === Sweep on key configurations ===
    configs = [
        # (hold, th_long_mult, th_short_mult, scale, label)
        (2, 0.5, 1.0, True, "h2_L0.5_S1.0_sc"),
        (2, 0.5, 999, True, "h2_L0.5_LO_sc"),
        (2, 0.5, 999, False, "h2_L0.5_LO_1x"),
        (2, 0.7, 999, True, "h2_L0.7_LO_sc"),
        (2, 0.7, 999, False, "h2_L0.7_LO_1x"),
        (3, 0.5, 999, True, "h3_L0.5_LO_sc"),
        (3, 0.5, 999, False, "h3_L0.5_LO_1x"),
        (3, 0.7, 999, True, "h3_L0.7_LO_sc"),
        (3, 0.7, 999, False, "h3_L0.7_LO_1x"),
        (5, 0.5, 1.0, True, "h5_L0.5_S1.0_sc"),
        (5, 0.5, 999, True, "h5_L0.5_LO_sc"),
        (5, 0.5, 999, False, "h5_L0.5_LO_1x"),
        (5, 0.7, 1.0, True, "h5_L0.7_S1.0_sc"),
        (5, 0.7, 999, True, "h5_L0.7_LO_sc"),
        (5, 0.7, 999, False, "h5_L0.7_LO_1x"),
        (5, 1.0, 999, True, "h5_L1.0_LO_sc"),
        (5, 1.0, 999, False, "h5_L1.0_LO_1x"),
        (8, 0.5, 999, True, "h8_L0.5_LO_sc"),
        (8, 0.5, 999, False, "h8_L0.5_LO_1x"),
        (8, 0.7, 999, True, "h8_L0.7_LO_sc"),
        (8, 0.7, 999, False, "h8_L0.7_LO_1x"),
        (8, 1.0, 999, True, "h8_L1.0_LO_sc"),
        (8, 1.0, 999, False, "h8_L1.0_LO_1x"),
    ]

    print(f"\nTesting {len(configs)} configurations on both methods...")
    print(f"\n{'='*120}")
    print(f"{'Config':<24} │ {'HS300 (real index)':^40} │ {'Top300 (synthetic)':^40} │ {'Δ Sharpe':>8}")
    print(f"{'':24} │ {'Return':>9} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} {'T/D':>5} │ "
          f"{'Return':>9} {'Sharpe':>7} {'MaxDD':>7} {'WR':>5} {'T/D':>5} │")
    print("-" * 120)

    results = []

    for hold, tl_m, ts_m, sc, label in configs:
        # HS300 method
        th_l_hs = hs_mean + tl_m * hs_std
        th_s_hs = hs_mean - ts_m * hs_std if ts_m < 100 else -9999
        pnls_hs = []
        trades_hs = 0
        for ddf in hs_test:
            p, nt = backtest_day_fixed(ddf, 'hs300_level', th_l_hs, th_s_hs,
                                         hold, sc, MAX_SCALE,
                                         ROUND_TRIP_COST_BPS, DAILY_LOSS_LIMIT_BPS)
            pnls_hs.append(p)
            trades_hs += nt
        m_hs = compute_metrics(pnls_hs)

        # Top300 method (for comparison)
        th_l_top = top_mean + tl_m * top_std
        th_s_top = top_mean - ts_m * top_std if ts_m < 100 else -9999
        pnls_top = []
        trades_top = 0
        for ddf in top_test:
            p, nt = backtest_day_fixed(ddf, 'tw_level', th_l_top, th_s_top,
                                         hold, sc, MAX_SCALE,
                                         ROUND_TRIP_COST_BPS, DAILY_LOSS_LIMIT_BPS)
            pnls_top.append(p)
            trades_top += nt
        m_top = compute_metrics(pnls_top)

        delta_sh = m_hs['sharpe'] - m_top['sharpe']

        print(f"{label:<24} │ {m_hs['ret']*100:>+8.1f}% {m_hs['sharpe']:>7.2f} "
              f"{m_hs['dd']*100:>6.1f}% {m_hs['wr']:>4.0%} {trades_hs/len(hs_test):>5.1f} │ "
              f"{m_top['ret']*100:>+8.1f}% {m_top['sharpe']:>7.2f} "
              f"{m_top['dd']*100:>6.1f}% {m_top['wr']:>4.0%} {trades_top/len(top_test):>5.1f} │ "
              f"{delta_sh:>+7.1f}")

        results.append({
            'label': label, 'hold': hold, 'th_long_mult': tl_m,
            'th_short_mult': ts_m, 'scaled': sc,
            'hs300_ret': m_hs['ret'], 'hs300_sharpe': m_hs['sharpe'],
            'hs300_dd': m_hs['dd'], 'hs300_wr': m_hs['wr'],
            'hs300_avg_bps': m_hs['avg_bps'], 'hs300_trades': trades_hs,
            'top300_ret': m_top['ret'], 'top300_sharpe': m_top['sharpe'],
            'top300_dd': m_top['dd'], 'top300_wr': m_top['wr'],
            'top300_avg_bps': m_top['avg_bps'], 'top300_trades': trades_top,
            'delta_sharpe': delta_sh,
        })

    # === Summary ===
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUT_DIR, 'hs300_vs_top300.csv'), index=False)

    print(f"\n{'='*120}")
    print("SUMMARY")
    print("=" * 120)
    avg_delta = df['delta_sharpe'].mean()
    hs_wins = (df['hs300_sharpe'] > df['top300_sharpe']).sum()
    print(f"  Average Δ Sharpe (HS300 - Top300): {avg_delta:+.2f}")
    print(f"  HS300 wins: {hs_wins}/{len(df)} configs")

    # Best HS300 configs
    print(f"\n  Top 5 HS300 configs by Sharpe:")
    df_sorted = df.sort_values('hs300_sharpe', ascending=False)
    for i, (_, r) in enumerate(df_sorted.head(5).iterrows()):
        print(f"    {i+1}. {r['label']:<24} Sharpe={r['hs300_sharpe']:>6.2f}, "
              f"Ret={r['hs300_ret']*100:>+8.1f}%, DD={r['hs300_dd']*100:>5.1f}%")

    # === Daily P&L comparison for best config ===
    best = df_sorted.iloc[0]
    print(f"\n{'='*120}")
    print(f"DAILY P&L: {best['label']} (best HS300 config)")
    print("=" * 120)

    hold = int(best['hold'])
    tl_m = best['th_long_mult']
    ts_m = best['th_short_mult']
    sc = best['scaled']

    th_l_hs = hs_mean + tl_m * hs_std
    th_s_hs = hs_mean - ts_m * hs_std if ts_m < 100 else -9999

    daily_pnls = []
    for i, ddf in enumerate(hs_test):
        p, nt = backtest_day_fixed(ddf, 'hs300_level', th_l_hs, th_s_hs,
                                     hold, sc, MAX_SCALE,
                                     ROUND_TRIP_COST_BPS, DAILY_LOSS_LIMIT_BPS)
        daily_pnls.append({'date': test_dates[i], 'pnl': p, 'pnl_bps': p * 10000, 'trades': nt})

    dpnl = pd.DataFrame(daily_pnls)
    dpnl.to_csv(os.path.join(OUTPUT_DIR, 'best_daily_pnl.csv'), index=False)

    cum = (1 + dpnl['pnl']).cumprod()
    print(f"  Days: {len(dpnl)}")
    print(f"  Positive days: {(dpnl['pnl'] > 0).sum()} ({(dpnl['pnl'] > 0).mean():.0%})")
    print(f"  Best day: {dpnl.loc[dpnl['pnl'].idxmax(), 'date']} ({dpnl['pnl_bps'].max():+.1f} bps)")
    print(f"  Worst day: {dpnl.loc[dpnl['pnl'].idxmin(), 'date']} ({dpnl['pnl_bps'].min():+.1f} bps)")
    print(f"  Avg daily P&L: {dpnl['pnl_bps'].mean():+.1f} bps")
    print(f"  Std daily P&L: {dpnl['pnl_bps'].std():.1f} bps")

    # Monthly breakdown
    dpnl['month'] = dpnl['date'].str[:7]
    monthly = dpnl.groupby('month').agg({'pnl_bps': ['sum', 'count', 'mean']})
    monthly.columns = ['total_bps', 'days', 'avg_bps']
    print(f"\n  Monthly breakdown:")
    for month, row in monthly.iterrows():
        print(f"    {month}: {row['total_bps']:>+8.1f} bps ({int(row['days'])} days, "
              f"avg {row['avg_bps']:>+5.1f} bps/day)")

    print(f"\n  Saved to {OUTPUT_DIR}/")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
