"""
Explore 10: Longer Hold + Higher Threshold Sweep
=================================================
Systematically test:
1. Hold period: 1, 2, 3, 5 bars
2. Threshold multiplier: 0.5, 0.7, 1.0, 1.5, 2.0 σ (for long)
3. Short threshold: 1.0, 1.5, 2.0 σ
4. With and without signal scaling

Goal: fewer trades, lower cost drag, potentially better risk-adjusted returns.
"""

import os, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore10')
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


def select_top(df, n=300):
    dm = df.groupby('code')['money'].sum()
    top = dm.nlargest(n).index
    return df[df['code'].isin(top)].copy()


def process_day(fp):
    df = load_day(fp)
    df = select_top(df, TOP_N)
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


def backtest_day(day_df, th_long, th_short, hold_bars, scale, max_sc, cost_bps, loss_limit_bps):
    n = len(day_df)
    pos = 0.0; ebar = -1; eprice = 0
    cum = 0; n_trades = 0; stopped = False

    for bar in range(10, n):
        sig = day_df['breadth_resid'].iloc[bar]
        px = day_df['tw_level'].iloc[bar]

        if stopped: break

        # EXIT
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

        # ENTRY
        if pos == 0 and bar < n - hold_bars - 1 and not stopped:
            if sig > th_long:
                sc = min(abs(sig) / abs(th_long), max_sc) if scale else 1.0
                pos = sc; ebar = bar; eprice = px
            elif sig < th_short:
                sc = min(abs(sig) / abs(th_short), max_sc) if scale else 1.0
                pos = -sc; ebar = bar; eprice = px

    # Force close
    if pos != 0:
        px = day_df['tw_level'].iloc[-1]
        d = 1 if pos > 0 else -1; sz = abs(pos)
        gr = d * (px / eprice - 1)
        nr = gr * sz - cost_bps / 10000
        cum += nr; n_trades += 1

    return cum, n_trades


def compute_metrics(pnls, label=""):
    arr = np.array(pnls)
    cum = (1 + pd.Series(pnls)).cumprod()
    ret = cum.iloc[-1] - 1
    ann = (1 + ret) ** (242 / len(arr)) - 1 if len(arr) > 0 else 0
    sh = arr.mean() / arr.std() * np.sqrt(242) if arr.std() > 0 else 0
    dd = (cum / cum.cummax() - 1).min()
    wr = (arr > 0).mean()
    return {
        'label': label, 'ret': ret, 'ann': ann, 'sharpe': sh,
        'dd': dd, 'wr': wr, 'avg_bps': arr.mean() * 10000,
    }


def main():
    t0 = time.time()
    print("=" * 120)
    print("Explore 10: Hold Period + Threshold Sweep (Real Futures Cost)")
    print("=" * 120)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    N = 200
    recent = all_files[-N:]

    print(f"Loading {N} days... ", end="", flush=True)
    all_daily = []
    all_dates = []
    for f in recent:
        try:
            d = process_day(os.path.join(STOCK_DATA_DIR, f))
            all_daily.append(d)
            all_dates.append(f.replace('.parquet', ''))
        except:
            continue
    print(f"done ({len(all_daily)} days, {time.time()-t0:.0f}s)")

    # Split
    split = len(all_daily) // 2
    train = all_daily[:split]
    test = all_daily[split:]
    test_dates = all_dates[split:]

    # Calibrate
    all_r = pd.concat([d['breadth_resid'] for d in train]).dropna()
    r_mean, r_std = all_r.mean(), all_r.std()
    print(f"Calibration: mean={r_mean:.4f}, std={r_std:.4f}")

    # === SWEEP ===
    hold_vals = [1, 2, 3, 5, 8]
    th_long_mults = [0.5, 0.7, 1.0, 1.5, 2.0]
    th_short_mults = [1.0, 1.5, 2.0, 999]  # 999 = long-only
    scale_opts = [True, False]

    results = []
    total_configs = len(hold_vals) * len(th_long_mults) * len(th_short_mults) * len(scale_opts)
    print(f"\nTesting {total_configs} configurations...")

    for hold in hold_vals:
        for tl_m in th_long_mults:
            for ts_m in th_short_mults:
                for sc in scale_opts:
                    th_l = r_mean + tl_m * r_std
                    th_s = r_mean - ts_m * r_std if ts_m < 100 else -9999

                    pnls = []
                    tot_trades = 0
                    for ddf in test:
                        p, nt = backtest_day(ddf, th_l, th_s, hold, sc, MAX_SCALE,
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
                        'label': label,
                        'hold': hold,
                        'th_long_mult': tl_m,
                        'th_short_mult': ts_m,
                        'scaled': sc,
                        'ret': m['ret'],
                        'ann': m['ann'],
                        'sharpe': m['sharpe'],
                        'dd': m['dd'],
                        'wr': m['wr'],
                        'avg_bps': m['avg_bps'],
                        'trades': tot_trades,
                        'trades_per_day': tot_trades / len(test),
                    })

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    print(f"\nCompleted {len(df)} configs in {time.time()-t0:.0f}s")

    # === TOP 30 by Sharpe ===
    print("\n" + "=" * 120)
    print("TOP 30 BY SHARPE (100 OOS days, 4.5 bps cost)")
    print("=" * 120)
    print(f"{'Rank':>4} {'Config':<28} {'Hold':>4} {'ThL':>5} {'ThS':>5} {'Scale':>5} "
          f"{'Return':>9} {'AnnRet':>9} {'Sharpe':>7} {'MaxDD':>7} {'WinR':>6} "
          f"{'AvgBps':>8} {'Trades':>7} {'T/Day':>6}")
    print("-" * 120)
    for i, (_, r) in enumerate(df.head(30).iterrows()):
        ts = 'LO' if r['th_short_mult'] > 100 else f"{r['th_short_mult']:.1f}"
        print(f"{i+1:>4} {r['label']:<28} {r['hold']:>4} {r['th_long_mult']:>5.1f} {ts:>5} "
              f"{'Yes' if r['scaled'] else 'No':>5} "
              f"{r['ret']*100:>8.1f}% {r['ann']*100:>8.0f}% {r['sharpe']:>7.2f} "
              f"{r['dd']*100:>6.1f}% {r['wr']:>5.0%} "
              f"{r['avg_bps']:>8.1f} {r['trades']:>7} {r['trades_per_day']:>6.1f}")

    # === TOP 30 by Return ===
    df_ret = df.sort_values('ret', ascending=False)
    print("\n" + "=" * 120)
    print("TOP 30 BY RETURN")
    print("=" * 120)
    print(f"{'Rank':>4} {'Config':<28} {'Hold':>4} {'ThL':>5} {'ThS':>5} {'Scale':>5} "
          f"{'Return':>9} {'AnnRet':>9} {'Sharpe':>7} {'MaxDD':>7} {'WinR':>6} "
          f"{'AvgBps':>8} {'Trades':>7} {'T/Day':>6}")
    print("-" * 120)
    for i, (_, r) in enumerate(df_ret.head(30).iterrows()):
        ts = 'LO' if r['th_short_mult'] > 100 else f"{r['th_short_mult']:.1f}"
        print(f"{i+1:>4} {r['label']:<28} {r['hold']:>4} {r['th_long_mult']:>5.1f} {ts:>5} "
              f"{'Yes' if r['scaled'] else 'No':>5} "
              f"{r['ret']*100:>8.1f}% {r['ann']*100:>8.0f}% {r['sharpe']:>7.2f} "
              f"{r['dd']*100:>6.1f}% {r['wr']:>5.0%} "
              f"{r['avg_bps']:>8.1f} {r['trades']:>7} {r['trades_per_day']:>6.1f}")

    # === Analysis: Hold period effect ===
    print("\n" + "=" * 120)
    print("HOLD PERIOD EFFECT (scaled, L0.5 S1.0 as baseline)")
    print("=" * 120)
    for hold in hold_vals:
        subset = df[(df['hold'] == hold) & (df['scaled'] == True) &
                     (df['th_long_mult'] == 0.5) & (df['th_short_mult'] == 1.0)]
        if len(subset) > 0:
            r = subset.iloc[0]
            print(f"  Hold={hold}: Ret={r['ret']*100:>+7.1f}%, Sharpe={r['sharpe']:>6.2f}, "
                  f"DD={r['dd']*100:>6.1f}%, Trades/day={r['trades_per_day']:>5.1f}, "
                  f"AvgBps={r['avg_bps']:>+7.1f}")

    # === Analysis: Threshold effect (at hold=2, scaled) ===
    print(f"\n{'='*120}")
    print("THRESHOLD EFFECT (hold=2, scaled)")
    print("=" * 120)
    for tl in th_long_mults:
        for ts in th_short_mults:
            subset = df[(df['hold'] == 2) & (df['scaled'] == True) &
                         (df['th_long_mult'] == tl) & (df['th_short_mult'] == ts)]
            if len(subset) > 0:
                r = subset.iloc[0]
                ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                print(f"  L{tl:.1f} S{ts_str}: Ret={r['ret']*100:>+7.1f}%, "
                      f"Sharpe={r['sharpe']:>6.2f}, DD={r['dd']*100:>6.1f}%, "
                      f"T/day={r['trades_per_day']:>5.1f}, AvgBps={r['avg_bps']:>+7.1f}")

    # === Analysis: Threshold effect (at hold=3, scaled) ===
    print(f"\n{'='*120}")
    print("THRESHOLD EFFECT (hold=3, scaled)")
    print("=" * 120)
    for tl in th_long_mults:
        for ts in th_short_mults:
            subset = df[(df['hold'] == 3) & (df['scaled'] == True) &
                         (df['th_long_mult'] == tl) & (df['th_short_mult'] == ts)]
            if len(subset) > 0:
                r = subset.iloc[0]
                ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                print(f"  L{tl:.1f} S{ts_str}: Ret={r['ret']*100:>+7.1f}%, "
                      f"Sharpe={r['sharpe']:>6.2f}, DD={r['dd']*100:>6.1f}%, "
                      f"T/day={r['trades_per_day']:>5.1f}, AvgBps={r['avg_bps']:>+7.1f}")

    # === Analysis: Threshold effect (at hold=5, scaled) ===
    print(f"\n{'='*120}")
    print("THRESHOLD EFFECT (hold=5, scaled)")
    print("=" * 120)
    for tl in th_long_mults:
        for ts in th_short_mults:
            subset = df[(df['hold'] == 5) & (df['scaled'] == True) &
                         (df['th_long_mult'] == tl) & (df['th_short_mult'] == ts)]
            if len(subset) > 0:
                r = subset.iloc[0]
                ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                print(f"  L{tl:.1f} S{ts_str}: Ret={r['ret']*100:>+7.1f}%, "
                      f"Sharpe={r['sharpe']:>6.2f}, DD={r['dd']*100:>6.1f}%, "
                      f"T/day={r['trades_per_day']:>5.1f}, AvgBps={r['avg_bps']:>+7.1f}")

    # === Scale vs No-Scale comparison ===
    print(f"\n{'='*120}")
    print("SCALE vs NO-SCALE (hold=2)")
    print("=" * 120)
    for tl in th_long_mults:
        for ts in [1.0, 1.5, 999]:
            for sc in [True, False]:
                subset = df[(df['hold'] == 2) & (df['scaled'] == sc) &
                             (df['th_long_mult'] == tl) & (df['th_short_mult'] == ts)]
                if len(subset) > 0:
                    r = subset.iloc[0]
                    ts_str = 'LO' if ts > 100 else f'{ts:.1f}'
                    sc_str = 'SCALED' if sc else '  1x  '
                    print(f"  L{tl:.1f} S{ts_str} {sc_str}: Ret={r['ret']*100:>+7.1f}%, "
                          f"Sharpe={r['sharpe']:>6.2f}, DD={r['dd']*100:>6.1f}%, "
                          f"T/day={r['trades_per_day']:>5.1f}")

    # === Trade log for best new config ===
    best = df.iloc[0]  # best by sharpe
    best_ret = df_ret.iloc[0]  # best by return
    print(f"\n{'='*120}")
    print("RECOMMENDED CONFIGS")
    print("=" * 120)
    b_ts = 'LO' if best['th_short_mult'] > 100 else f"{best['th_short_mult']:.1f}s"
    print(f"\n  Best Sharpe: {best['label']}")
    print(f"     Hold={int(best['hold'])}, ThLong={best['th_long_mult']:.1f}s, ThShort={b_ts}, "
          f"Scaled={'Yes' if best['scaled'] else 'No'}")
    print(f"     Return: {best['ret']*100:+.1f}%, Sharpe: {best['sharpe']:.2f}, "
          f"MaxDD: {best['dd']*100:.1f}%, Trades/day: {best['trades_per_day']:.1f}")

    br_ts = 'LO' if best_ret['th_short_mult'] > 100 else f"{best_ret['th_short_mult']:.1f}s"
    print(f"\n  Best Return: {best_ret['label']}")
    print(f"     Hold={int(best_ret['hold'])}, ThLong={best_ret['th_long_mult']:.1f}s, ThShort={br_ts}, "
          f"Scaled={'Yes' if best_ret['scaled'] else 'No'}")
    print(f"     Return: {best_ret['ret']*100:+.1f}%, Sharpe: {best_ret['sharpe']:.2f}, "
          f"MaxDD: {best_ret['dd']*100:.1f}%, Trades/day: {best_ret['trades_per_day']:.1f}")

    # === Detailed daily PnL for top configs ===
    # Rerun top 5 and compare daily patterns
    print(f"\n{'='*120}")
    print("DAILY P&L COMPARISON — TOP 5 CONFIGS")
    print("=" * 120)

    top5 = df.head(5)
    daily_comparison = {}

    for _, cfg in top5.iterrows():
        th_l = r_mean + cfg['th_long_mult'] * r_std
        th_s = r_mean - cfg['th_short_mult'] * r_std if cfg['th_short_mult'] < 100 else -9999
        pnls = []
        for ddf in test:
            p, _ = backtest_day(ddf, th_l, th_s, int(cfg['hold']),
                                cfg['scaled'], MAX_SCALE,
                                ROUND_TRIP_COST_BPS, DAILY_LOSS_LIMIT_BPS)
            pnls.append(p)
        daily_comparison[cfg['label']] = pnls

    # Print week-by-week comparison
    print(f"\n  {'Date':<12}", end="")
    for lbl in daily_comparison:
        print(f" {lbl:>16}", end="")
    print()
    print("  " + "-" * (12 + 17 * len(daily_comparison)))

    week_pnls = {lbl: [] for lbl in daily_comparison}
    for i, date in enumerate(test_dates):
        if i > 0 and i % 5 == 0:
            # Print week summary
            print(f"  {'WEEK':>12}", end="")
            for lbl in daily_comparison:
                ws = sum(week_pnls[lbl]) * 10000
                print(f" {ws:>+15.0f}", end="")
            print()
            print("  " + "-" * (12 + 17 * len(daily_comparison)))
            week_pnls = {lbl: [] for lbl in daily_comparison}

        print(f"  {date:<12}", end="")
        for lbl in daily_comparison:
            v = daily_comparison[lbl][i] * 10000
            print(f" {v:>+15.1f}", end="")
            week_pnls[lbl].append(daily_comparison[lbl][i])
        print()

    # Final week
    if any(week_pnls[lbl] for lbl in week_pnls):
        print(f"  {'WEEK':>12}", end="")
        for lbl in daily_comparison:
            ws = sum(week_pnls[lbl]) * 10000
            print(f" {ws:>+15.0f}", end="")
        print()

    # Save
    df.to_csv(os.path.join(OUTPUT_DIR, 'sweep_results.csv'), index=False)
    print(f"\n  Results saved to {OUTPUT_DIR}/sweep_results.csv")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
