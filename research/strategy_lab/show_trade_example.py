"""
Show detailed trade-by-trade example for one specific day.
"""
import os
import numpy as np
import pandas as pd

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
TOP_N_LIQUID = 300
COST_BPS = 1.0

# Use a recent day
TARGET_DAY = '2026-01-20.parquet'

# Use expanding calibration from prior days (simulate 200 days of history for thresholds)
HISTORY_DAYS = 200


def load_day(filepath):
    df = pd.read_parquet(filepath)
    df = df[df['paused'] == 0].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def select_liquid_stocks(df, top_n=300):
    daily_money = df.groupby('code')['money'].sum()
    top_codes = daily_money.nlargest(top_n).index
    return df[df['code'].isin(top_codes)].copy()


def process_day(filepath):
    df = load_day(filepath)
    df = select_liquid_stocks(df, TOP_N_LIQUID)
    pivot_close = df.pivot_table(index='datetime', columns='code', values='close')
    stock_returns_1 = pivot_close.pct_change()
    index_return_1 = stock_returns_1.mean(axis=1)
    index_level = (1 + index_return_1).cumprod()
    index_level.iloc[0] = 1.0
    n_stocks = stock_returns_1.count(axis=1)
    breadth_up = (stock_returns_1 > 0).sum(axis=1) / n_stocks

    breadth_resid = breadth_up.copy()
    for i in range(10, len(breadth_up)):
        b = breadth_up.iloc[:i]
        m = index_return_1.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 5:
            slope = np.polyfit(m[mask], b[mask], 1)
            breadth_resid.iloc[i] = breadth_up.iloc[i] - np.polyval(slope, index_return_1.iloc[i])

    return pd.DataFrame({
        'index_level': index_level,
        'index_return': index_return_1,
        'breadth_up': breadth_up,
        'breadth_resid': breadth_resid,
        'n_up': (stock_returns_1 > 0).sum(axis=1),
        'n_down': (stock_returns_1 < 0).sum(axis=1),
        'n_flat': (stock_returns_1 == 0).sum(axis=1),
    }, index=pivot_close.index)


def main():
    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])

    # Find target day index
    target_idx = all_files.index(TARGET_DAY)

    # Calibrate thresholds from prior days
    print("Calibrating thresholds from history...")
    history_files = all_files[max(0, target_idx - HISTORY_DAYS):target_idx]
    all_resids = []
    for fname in history_files[-50:]:  # use last 50 days for speed
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_day(filepath)
            all_resids.append(day_df['breadth_resid'].dropna())
        except:
            continue

    all_resid = pd.concat(all_resids)
    resid_mean = all_resid.mean()
    resid_std = all_resid.std()
    th_long = resid_mean + 1.0 * resid_std
    th_short = resid_mean - 1.0 * resid_std

    print(f"Threshold calibration (from {len(history_files)} days):")
    print(f"  breadth_resid mean = {resid_mean:.4f}")
    print(f"  breadth_resid std  = {resid_std:.4f}")
    print(f"  threshold_long  = {th_long:.4f}")
    print(f"  threshold_short = {th_short:.4f}")

    # Process target day
    print(f"\n{'='*80}")
    print(f"TRADE EXAMPLE: {TARGET_DAY.replace('.parquet', '')}")
    print(f"{'='*80}")

    filepath = os.path.join(STOCK_DATA_DIR, TARGET_DAY)
    day_df = process_day(filepath)

    # Run strategy and log every trade
    n_bars = len(day_df)
    position = 0
    entry_bar = -1
    entry_price = 0
    entry_time = None
    entry_signal = 0
    trades = []
    total_pnl = 0

    print(f"\nSignal range today: [{day_df['breadth_resid'].min():.4f}, {day_df['breadth_resid'].max():.4f}]")
    print(f"Thresholds: long > {th_long:.4f}, short < {th_short:.4f}")
    print()

    for bar in range(10, n_bars):
        signal = day_df['breadth_resid'].iloc[bar]
        price = day_df['index_level'].iloc[bar]
        time = day_df.index[bar]
        idx_ret = day_df['index_return'].iloc[bar]
        b_up = day_df['breadth_up'].iloc[bar]
        n_up = day_df['n_up'].iloc[bar]
        n_down = day_df['n_down'].iloc[bar]

        if position != 0:
            bars_held = bar - entry_bar
            is_eod = (bar >= n_bars - 2)
            if bars_held >= 1 or is_eod:
                exit_price = price
                trade_return = position * (exit_price / entry_price - 1)
                gross_return = trade_return
                trade_return -= COST_BPS / 10000
                total_pnl += trade_return

                direction = "LONG" if position == 1 else "SHORT"
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': time,
                    'direction': direction,
                    'entry_signal': entry_signal,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_bps': gross_return * 10000,
                    'net_bps': trade_return * 10000,
                    'cum_pnl_bps': total_pnl * 10000,
                })
                position = 0

        if position == 0 and bar < n_bars - 2:
            if signal > th_long:
                position = 1
                entry_bar = bar
                entry_price = price
                entry_time = time
                entry_signal = signal
            elif signal < th_short:
                position = -1
                entry_bar = bar
                entry_price = price
                entry_time = time
                entry_signal = signal

    if position != 0:
        price = day_df['index_level'].iloc[-1]
        time = day_df.index[-1]
        trade_return = position * (price / entry_price - 1)
        gross_return = trade_return
        trade_return -= COST_BPS / 10000
        total_pnl += trade_return
        direction = "LONG" if position == 1 else "SHORT"
        trades.append({
            'entry_time': entry_time,
            'exit_time': time,
            'direction': direction,
            'entry_signal': entry_signal,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_bps': gross_return * 10000,
            'net_bps': trade_return * 10000,
            'cum_pnl_bps': total_pnl * 10000,
        })

    # Print all trades
    print(f"Total trades today: {len(trades)}")
    print(f"Total PnL: {total_pnl*10000:.2f} bps ({total_pnl*100:.4f}%)")
    print()

    # Show first 15 trades in detail
    n_show = min(15, len(trades))
    print(f"First {n_show} trades:")
    print(f"{'#':>3} {'Entry Time':<22} {'Exit Time':<22} {'Dir':<6} {'Signal':>8} {'Entry':>10} {'Exit':>10} {'Gross(bps)':>11} {'Net(bps)':>10} {'CumPnL':>10}")
    print("-" * 120)

    for i, t in enumerate(trades[:n_show]):
        print(f"{i+1:>3} {str(t['entry_time']):<22} {str(t['exit_time']):<22} "
              f"{t['direction']:<6} {t['entry_signal']:>8.4f} "
              f"{t['entry_price']:>10.6f} {t['exit_price']:>10.6f} "
              f"{t['gross_bps']:>11.2f} {t['net_bps']:>10.2f} {t['cum_pnl_bps']:>10.2f}")

    if len(trades) > n_show:
        print(f"  ... ({len(trades) - n_show} more trades) ...")
        print(f"\nLast 5 trades:")
        for i, t in enumerate(trades[-5:]):
            idx = len(trades) - 5 + i
            print(f"{idx+1:>3} {str(t['entry_time']):<22} {str(t['exit_time']):<22} "
                  f"{t['direction']:<6} {t['entry_signal']:>8.4f} "
                  f"{t['entry_price']:>10.6f} {t['exit_price']:>10.6f} "
                  f"{t['gross_bps']:>11.2f} {t['net_bps']:>10.2f} {t['cum_pnl_bps']:>10.2f}")

    # Summary stats
    if trades:
        trades_df = pd.DataFrame(trades)
        print(f"\n{'='*80}")
        print("DAY SUMMARY")
        print(f"{'='*80}")
        print(f"  Total trades:   {len(trades)}")
        print(f"  Long trades:    {(trades_df['direction']=='LONG').sum()}")
        print(f"  Short trades:   {(trades_df['direction']=='SHORT').sum()}")
        print(f"  Winners:        {(trades_df['net_bps']>0).sum()} ({(trades_df['net_bps']>0).mean():.1%})")
        print(f"  Losers:         {(trades_df['net_bps']<0).sum()} ({(trades_df['net_bps']<0).mean():.1%})")
        print(f"  Avg gross(bps): {trades_df['gross_bps'].mean():.2f}")
        print(f"  Avg net(bps):   {trades_df['net_bps'].mean():.2f}")
        print(f"  Total PnL(bps): {total_pnl*10000:.2f}")
        print(f"  Total PnL(%):   {total_pnl*100:.4f}%")

    # Show one detailed trade narrative
    if len(trades) >= 3:
        print(f"\n{'='*80}")
        print("DETAILED TRADE NARRATIVE (Trade #1)")
        print(f"{'='*80}")
        t = trades[0]
        entry_bar_idx = day_df.index.get_loc(t['entry_time'])
        exit_bar_idx = day_df.index.get_loc(t['exit_time'])

        # Context: 3 bars before entry
        print(f"\n  Context (3 bars before entry):")
        for b in range(max(10, entry_bar_idx-3), entry_bar_idx):
            row = day_df.iloc[b]
            print(f"    {day_df.index[b].strftime('%H:%M')} | index_ret={row['index_return']*10000:>+6.1f}bps | "
                  f"up={row['n_up']:.0f} down={row['n_down']:.0f} | "
                  f"breadth_up={row['breadth_up']:.3f} | breadth_resid={row['breadth_resid']:>+.4f}")

        # Entry bar
        row = day_df.iloc[entry_bar_idx]
        print(f"  → {day_df.index[entry_bar_idx].strftime('%H:%M')} | index_ret={row['index_return']*10000:>+6.1f}bps | "
              f"up={row['n_up']:.0f} down={row['n_down']:.0f} | "
              f"breadth_up={row['breadth_up']:.3f} | breadth_resid={row['breadth_resid']:>+.4f} "
              f"{'> LONG' if t['direction']=='LONG' else '< SHORT'} THRESHOLD")

        # Exit bar
        row = day_df.iloc[exit_bar_idx]
        print(f"  ← {day_df.index[exit_bar_idx].strftime('%H:%M')} | index_ret={row['index_return']*10000:>+6.1f}bps | "
              f"EXIT → gross={t['gross_bps']:+.2f}bps, net={t['net_bps']:+.2f}bps")

        # Interpretation
        print(f"\n  Interpretation:")
        if t['direction'] == 'LONG':
            print(f"    Signal = {t['entry_signal']:.4f} > threshold {th_long:.4f}")
            print(f"    Meaning: {row['n_up']:.0f}/{TOP_N_LIQUID} stocks were rising, MORE than expected")
            print(f"    given the index move. This 'excess breadth' predicts upward continuation.")
        else:
            print(f"    Signal = {t['entry_signal']:.4f} < threshold {th_short:.4f}")
            print(f"    Meaning: Only {row['n_up']:.0f}/{TOP_N_LIQUID} stocks were rising, FEWER than expected")
            print(f"    given the index move. This 'deficient breadth' predicts downward continuation.")


if __name__ == '__main__':
    main()
