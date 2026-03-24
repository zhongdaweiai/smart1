"""
Explore 11: No-Lookahead Backtest
==================================
Fix two look-ahead biases from previous explores:

1. EXECUTION LAG: Signal at bar i → enter at bar i+1 (not bar i)
   - In reality, you observe bar i's close, compute signal, then trade at next bar
   - P&L: from bar (i+1) to bar (i+1+hold)

2. STOCK UNIVERSE: Use previous day's top 300 (not same day's full turnover)

Test same parameter sweep as Explore 10 on the fixed version.
"""

import os, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore11')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 300
ROUND_TRIP_COST_BPS = 4.5
MAX_SCALE = 3.0
DAILY_LOSS_LIMIT_BPS = -50


def load_day(fp):
    df = pd.read_parquet(fp)
    df = df[df['paused'] == 0].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def get_top_codes_from_day(fp, n=300):
    """Get top N stock codes by turnover for a given day (used for NEXT day's universe)."""
    df = load_day(fp)
    dm = df.groupby('code')['money'].sum()
    return set(dm.nlargest(n).index)


def process_day_with_universe(fp, universe_codes):
    """Process one day using a PRE-DETERMINED stock universe (from previous day)."""
    df = load_day(fp)
    df = df[df['code'].isin(universe_codes)].copy()

    pc = df.pivot_table(index='datetime', columns='code', values='close')
    pm = df.pivot_table(index='datetime', columns='code', values='money')
    ret1 = pc.pct_change()

    ew_ret = ret1.mean(axis=1)
    bw = pm.div(pm.sum(axis=1), axis=0)
    tw_ret = (ret1 * bw).sum(axis=1)
    tw_lvl = (1 + tw_ret).cumprod(); tw_lvl.iloc[0] = 1.0

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


def backtest_day_fixed(day_df, th_long, th_short, hold_bars, scale, max_sc,
                        cost_bps, loss_limit_bps):
    """
    FIXED backtest: signal at bar i → enter at bar i+1 → exit at bar i+1+hold.
    This eliminates the same-bar execution look-ahead.
    """
    n = len(day_df)
    pos = 0.0; ebar = -1; eprice = 0
    cum = 0; n_trades = 0; stopped = False
    pending_entry = None  # (direction_scale,) — signal triggered, will enter next bar

    for bar in range(10, n):
        px = day_df['tw_level'].iloc[bar]

        if stopped: break

        # === EXECUTE PENDING ENTRY (from previous bar's signal) ===
        if pending_entry is not None and pos == 0:
            pos = pending_entry
            ebar = bar
            eprice = px
            pending_entry = None

        # === EXIT ===
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

        # === SIGNAL CHECK (will execute NEXT bar) ===
        if pos == 0 and pending_entry is None and bar < n - hold_bars - 2 and not stopped:
            sig = day_df['breadth_resid'].iloc[bar]
            if sig > th_long:
                sc = min(abs(sig) / abs(th_long), max_sc) if scale else 1.0
                pending_entry = sc  # will enter long next bar
            elif sig < th_short:
                sc = min(abs(sig) / abs(th_short), max_sc) if scale else 1.0
                pending_entry = -sc  # will enter short next bar

    # Force close
    if pos != 0:
        px = day_df['tw_level'].iloc[-1]
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
    print("Explore 11: No-Lookahead Backtest (FIXED)")
    print("=" * 120)
    print("""
  修复的未来信息:
  1. 开仓延迟1bar: 信号在bar i产生 → bar i+1开仓 → bar i+1+hold平仓
  2. 股票池用前一天的top 300 (不用当天全天成交额)
""")

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    N = 201  # need 1 extra for previous day's universe
    recent = all_files[-N:]

    # Step 1: Get previous day's universe for each day
    print("Building stock universes (previous day's top 300)... ", end="", flush=True)
    universes = {}
    for i in range(1, len(recent)):
        prev_fp = os.path.join(STOCK_DATA_DIR, recent[i-1])
        date_str = recent[i].replace('.parquet', '')
        universes[date_str] = get_top_codes_from_day(prev_fp, TOP_N)
    print(f"done ({len(universes)} days)")

    # Step 2: Process each day with correct universe
    print("Processing daily data... ", end="", flush=True)
    all_daily = []
    all_dates = []
    for i in range(1, len(recent)):
        date_str = recent[i].replace('.parquet', '')
        fp = os.path.join(STOCK_DATA_DIR, recent[i])
        try:
            d = process_day_with_universe(fp, universes[date_str])
            all_daily.append(d)
            all_dates.append(date_str)
        except Exception as e:
            continue
    print(f"done ({len(all_daily)} days, {time.time()-t0:.0f}s)")

    # Split
    split = len(all_daily) // 2
    train = all_daily[:split]
    test = all_daily[split:]
    test_dates = all_dates[split:]

    all_r = pd.concat([d['breadth_resid'] for d in train]).dropna()
    r_mean, r_std = all_r.mean(), all_r.std()
    print(f"Calibration: mean={r_mean:.4f}, std={r_std:.4f}")
    print(f"Train: {split} days, Test: {len(test)} days")

    # === SWEEP (same as Explore 10) ===
    hold_vals = [1, 2, 3, 5, 8]
    th_long_mults = [0.5, 0.7, 1.0, 1.5]
    th_short_mults = [1.0, 1.5, 999]  # 999 = long-only
    scale_opts = [True, False]

    results = []
    total = len(hold_vals) * len(th_long_mults) * len(th_short_mults) * len(scale_opts)
    print(f"\nTesting {total} configurations (no look-ahead)...")

    for hold in hold_vals:
        for tl_m in th_long_mults:
            for ts_m in th_short_mults:
                for sc in scale_opts:
                    th_l = r_mean + tl_m * r_std
                    th_s = r_mean - ts_m * r_std if ts_m < 100 else -9999

                    pnls = []
                    tot_trades = 0
                    for ddf in test:
                        p, nt = backtest_day_fixed(ddf, th_l, th_s, hold, sc, MAX_SCALE,
                                                    ROUND_TRIP_COST_BPS, DAILY_LOSS_LIMIT_BPS)
                        pnls.append(p)
                        tot_trades += nt

                    if sum(1 for p in pnls if p != 0) < 10:
                        continue

                    m = compute_metrics(pnls)
                    short_label = 'LO' if ts_m > 100 else f'S{ts_m}'
                    sc_label = 'sc' if sc else '1x'
                    label = f"h{hold}_L{tl_m}_{short_label}_{sc_label}"

                    results.append({
                        'label': label, 'hold': hold,
                        'th_long_mult': tl_m, 'th_short_mult': ts_m,
                        'scaled': sc, **m,
                        'trades': tot_trades,
                        'trades_per_day': tot_trades / len(test),
                    })

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    print(f"\nCompleted {len(df)} configs in {time.time()-t0:.0f}s")

    # === TOP 30 by Sharpe ===
    print("\n" + "=" * 120)
    print("TOP 30 BY SHARPE — NO LOOK-AHEAD")
    print("=" * 120)
    print(f"{'Rank':>4} {'Config':<24} {'Hold':>4} {'ThL':>5} {'ThS':>5} {'Scale':>5} "
          f"{'Return':>9} {'Sharpe':>7} {'MaxDD':>7} {'WinR':>6} "
          f"{'AvgBps':>8} {'Trades':>7} {'T/Day':>6}")
    print("-" * 110)
    for i, (_, r) in enumerate(df.head(30).iterrows()):
        ts = 'LO' if r['th_short_mult'] > 100 else f"{r['th_short_mult']:.1f}"
        print(f"{i+1:>4} {r['label']:<24} {r['hold']:>4} {r['th_long_mult']:>5.1f} {ts:>5} "
              f"{'Yes' if r['scaled'] else 'No':>5} "
              f"{r['ret']*100:>8.1f}% {r['sharpe']:>7.2f} "
              f"{r['dd']*100:>6.1f}% {r['wr']:>5.0%} "
              f"{r['avg_bps']:>8.1f} {r['trades']:>7} {r['trades_per_day']:>6.1f}")

    # === HOLD PERIOD EFFECT ===
    print(f"\n{'='*120}")
    print("HOLD PERIOD EFFECT (L0.5 S1.0 scaled, NO look-ahead)")
    print("=" * 120)
    for hold in hold_vals:
        subset = df[(df['hold'] == hold) & (df['scaled'] == True) &
                     (df['th_long_mult'] == 0.5) & (df['th_short_mult'] == 1.0)]
        if len(subset) > 0:
            r = subset.iloc[0]
            print(f"  Hold={hold}: Ret={r['ret']*100:>+8.1f}%, Sharpe={r['sharpe']:>6.2f}, "
                  f"DD={r['dd']*100:>6.1f}%, T/day={r['trades_per_day']:>5.1f}, "
                  f"AvgBps={r['avg_bps']:>+7.1f}")

    # === THRESHOLD EFFECT at hold=2 ===
    print(f"\n{'='*120}")
    print("THRESHOLD EFFECT (hold=2, scaled, NO look-ahead)")
    print("=" * 120)
    for tl in th_long_mults:
        for ts in th_short_mults:
            subset = df[(df['hold'] == 2) & (df['scaled'] == True) &
                         (df['th_long_mult'] == tl) & (df['th_short_mult'] == ts)]
            if len(subset) > 0:
                r = subset.iloc[0]
                ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                print(f"  L{tl:.1f} S{ts_str}: Ret={r['ret']*100:>+8.1f}%, "
                      f"Sharpe={r['sharpe']:>6.2f}, DD={r['dd']*100:>6.1f}%, "
                      f"T/day={r['trades_per_day']:>5.1f}, AvgBps={r['avg_bps']:>+7.1f}")

    # === THRESHOLD EFFECT at hold=5 ===
    print(f"\n{'='*120}")
    print("THRESHOLD EFFECT (hold=5, scaled, NO look-ahead)")
    print("=" * 120)
    for tl in th_long_mults:
        for ts in th_short_mults:
            subset = df[(df['hold'] == 5) & (df['scaled'] == True) &
                         (df['th_long_mult'] == tl) & (df['th_short_mult'] == ts)]
            if len(subset) > 0:
                r = subset.iloc[0]
                ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                print(f"  L{tl:.1f} S{ts_str}: Ret={r['ret']*100:>+8.1f}%, "
                      f"Sharpe={r['sharpe']:>6.2f}, DD={r['dd']*100:>6.1f}%, "
                      f"T/day={r['trades_per_day']:>5.1f}, AvgBps={r['avg_bps']:>+7.1f}")

    # === COMPARISON: Fixed vs Original (selected configs) ===
    print(f"\n{'='*120}")
    print("COMPARISON: NO LOOK-AHEAD vs ORIGINAL (key configs)")
    print("=" * 120)
    # Load explore10 results for comparison
    e10_path = os.path.join(os.path.dirname(__file__), 'results', 'explore10', 'sweep_results.csv')
    if os.path.exists(e10_path):
        e10 = pd.read_csv(e10_path)
        compare_configs = [
            (1, 0.5, 1.0, True), (2, 0.5, 1.0, True), (5, 0.5, 1.0, True),
            (2, 0.5, 999, True), (5, 0.5, 999, True),
            (2, 0.7, 999, True), (5, 0.7, 999, True),
            (2, 1.0, 999, True), (5, 1.0, 999, True),
            (2, 0.7, 999, False), (5, 0.7, 999, False),
        ]

        print(f"  {'Config':<24} {'ORIGINAL':>30}  {'NO LOOK-AHEAD':>30}  {'Δ Sharpe':>10}")
        print("  " + "-" * 100)

        for hold, tl, ts, sc in compare_configs:
            # Original (explore10)
            orig = e10[(e10['hold'] == hold) & (e10['th_long_mult'] == tl) &
                        (e10['th_short_mult'] == ts) & (e10['scaled'] == sc)]
            # Fixed
            fixed = df[(df['hold'] == hold) & (df['th_long_mult'] == tl) &
                        (df['th_short_mult'] == ts) & (df['scaled'] == sc)]

            if len(orig) > 0 and len(fixed) > 0:
                o, f_ = orig.iloc[0], fixed.iloc[0]
                ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                sc_str = 'sc' if sc else '1x'
                label = f"h{hold}_L{tl}_{ts_str}_{sc_str}"
                o_str = f"Ret={o['ret']*100:>+7.1f}% Sh={o['sharpe']:>5.1f}"
                f_str = f"Ret={f_['ret']*100:>+7.1f}% Sh={f_['sharpe']:>5.1f}"
                delta = f_['sharpe'] - o['sharpe']
                print(f"  {label:<24} {o_str:>30}  {f_str:>30}  {delta:>+9.1f}")

    # === Best practical configs ===
    print(f"\n{'='*120}")
    print("RECOMMENDED CONFIGS (NO LOOK-AHEAD)")
    print("=" * 120)

    # Top by sharpe with wr > 50%
    good = df[df['wr'] > 0.5].head(10)
    print("\n  Top 10 by Sharpe (win rate > 50%):")
    for i, (_, r) in enumerate(good.iterrows()):
        ts = 'LO' if r['th_short_mult'] > 100 else f"{r['th_short_mult']:.1f}"
        print(f"  {i+1:>2}. {r['label']:<24} Ret={r['ret']*100:>+8.1f}%, "
              f"Sh={r['sharpe']:>6.2f}, DD={r['dd']*100:>5.1f}%, "
              f"WR={r['wr']:>4.0%}, T/day={r['trades_per_day']:>5.1f}")

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'fixed_sweep.csv'), index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/fixed_sweep.csv")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
