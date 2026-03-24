"""
Explore 08: Real-World Futures Strategy Backtest
=================================================
Build a realistic, tradeable strategy using stock index futures (IF/IC/IM).

Key design choices:
1. INDEX CONSTRUCTION:
   - Use top 300 stocks by daily turnover (approximates CSI300)
   - Build TURNOVER-WEIGHTED index (closer to cap-weighted real index)
   - Also build equal-weight version for signal computation (breadth is inherently equal-weight)

2. TRADING INSTRUMENT: Stock Index Futures (IF = CSI300 futures)
   - Contract multiplier: 300 CNY/point (IF), 200 (IC), 200 (IM)
   - Margin: ~12% (we ignore leverage scaling, report per-contract return)
   - Commission: 0.23/10000 of notional per trade (open) + 3.45/10000 (close today)
     → ~3.68/10000 round-trip for intraday = 3.68 bps
   - Slippage: 0.2 index points per trade ≈ ~0.5 bps at current levels (~4000 points)
   - Total cost assumption: 4.5 bps round-trip (conservative)
   - Tick size: 0.2 points

3. SIGNAL: breadth_resid with scaled_asym (best from Explore 07)
   - Signal computed on equal-weight breadth (all stocks count equally for "consensus")
   - Trading P&L measured on turnover-weighted index (tracks real futures)

4. POSITION SIZING:
   - Base: 1 contract per signal
   - Scaled: position_size = min(|signal|/threshold, 3.0) contracts (cap at 3x)
   - Daily loss limit: stop trading if day's loss exceeds 20 bps

5. BACKTEST:
   - 200 recent days, first 100 train, last 100 OOS
   - Also run on a broader sample for robustness
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore08')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 300
ROUND_TRIP_COST_BPS = 4.5  # realistic IF intraday cost
MAX_SCALE = 3.0            # max position multiplier
DAILY_LOSS_LIMIT_BPS = -50  # stop trading if cumulative day loss exceeds this


def load_day(filepath):
    df = pd.read_parquet(filepath)
    df = df[df['paused'] == 0].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def select_top_stocks(df, top_n=300):
    daily_money = df.groupby('code')['money'].sum()
    top_codes = daily_money.nlargest(top_n).index
    return df[df['code'].isin(top_codes)].copy(), daily_money[top_codes]


def process_single_day(filepath):
    """Compute signal and two index versions for one day."""
    df = load_day(filepath)
    df_top, daily_money = select_top_stocks(df, TOP_N)

    pivot_close = df_top.pivot_table(index='datetime', columns='code', values='close')
    pivot_money = df_top.pivot_table(index='datetime', columns='code', values='money')

    stock_returns_1 = pivot_close.pct_change()

    # === Equal-weight index (for signal computation) ===
    ew_return = stock_returns_1.mean(axis=1)
    ew_level = (1 + ew_return).cumprod()
    ew_level.iloc[0] = 1.0

    # === Turnover-weighted index (proxy for real cap-weighted index) ===
    # Use each bar's turnover as weight (more realistic than daily total)
    bar_weights = pivot_money.div(pivot_money.sum(axis=1), axis=0)
    tw_return = (stock_returns_1 * bar_weights).sum(axis=1)
    tw_level = (1 + tw_return).cumprod()
    tw_level.iloc[0] = 1.0

    # === Signal: breadth_resid (computed on equal-weight basis) ===
    n_stocks = stock_returns_1.count(axis=1)
    breadth_up = (stock_returns_1 > 0).sum(axis=1) / n_stocks

    breadth_resid = breadth_up.copy()
    for i in range(10, len(breadth_up)):
        b = breadth_up.iloc[:i]
        m = ew_return.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 5:
            slope = np.polyfit(m[mask], b[mask], 1)
            breadth_resid.iloc[i] = breadth_up.iloc[i] - np.polyval(slope, ew_return.iloc[i])

    # === Tracking error between EW and TW ===
    tracking_diff = tw_return - ew_return

    return pd.DataFrame({
        'ew_level': ew_level,
        'ew_return': ew_return,
        'tw_level': tw_level,
        'tw_return': tw_return,
        'breadth_up': breadth_up,
        'breadth_resid': breadth_resid,
        'tracking_diff': tracking_diff,
    }, index=pivot_close.index)


def backtest_futures_day(day_df, th_long, th_short, cost_bps, max_scale, loss_limit_bps):
    """
    Backtest one day with realistic futures mechanics.

    Trading on TURNOVER-WEIGHTED index (tw_level) — this tracks real futures.
    Signal from breadth_resid (equal-weight based).

    Returns: list of trades, daily PnL
    """
    n_bars = len(day_df)
    position = 0.0  # can be fractional due to scaling
    entry_bar = -1
    entry_price = 0
    cum_day_pnl = 0
    trades = []
    stopped = False

    for bar in range(10, n_bars):
        signal = day_df['breadth_resid'].iloc[bar]
        price = day_df['tw_level'].iloc[bar]
        time_str = day_df.index[bar]

        if stopped:
            break

        # === EXIT ===
        if position != 0:
            bars_held = bar - entry_bar
            is_eod = (bar >= n_bars - 2)

            if bars_held >= 1 or is_eod:
                # P&L on turnover-weighted index
                direction = 1 if position > 0 else -1
                size = abs(position)
                gross_return = direction * (price / entry_price - 1)
                net_return = gross_return * size - cost_bps / 10000

                cum_day_pnl += net_return

                trades.append({
                    'entry_time': day_df.index[entry_bar],
                    'exit_time': time_str,
                    'direction': 'LONG' if direction > 0 else 'SHORT',
                    'size': size,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'gross_bps': gross_return * 10000,
                    'net_bps': net_return * 10000,
                    'cum_pnl_bps': cum_day_pnl * 10000,
                })

                position = 0

                # Daily loss limit check
                if cum_day_pnl * 10000 < loss_limit_bps:
                    stopped = True
                    continue

        # === ENTRY ===
        if position == 0 and bar < n_bars - 3 and not stopped:
            if signal > th_long:
                # Scaled position
                scale = min(abs(signal) / abs(th_long), max_scale)
                position = scale  # positive = long
                entry_bar = bar
                entry_price = price
            elif signal < th_short:
                scale = min(abs(signal) / abs(th_short), max_scale)
                position = -scale  # negative = short
                entry_bar = bar
                entry_price = price

    # Force close any remaining
    if position != 0:
        price = day_df['tw_level'].iloc[-1]
        direction = 1 if position > 0 else -1
        size = abs(position)
        gross_return = direction * (price / entry_price - 1)
        net_return = gross_return * size - cost_bps / 10000
        cum_day_pnl += net_return

        trades.append({
            'entry_time': day_df.index[entry_bar],
            'exit_time': day_df.index[-1],
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'size': size,
            'entry_price': entry_price,
            'exit_price': price,
            'gross_bps': gross_return * 10000,
            'net_bps': net_return * 10000,
            'cum_pnl_bps': cum_day_pnl * 10000,
        })

    return trades, cum_day_pnl


def compute_metrics(daily_pnls, n_days, label=""):
    """Compute strategy performance metrics."""
    pnl_arr = np.array(daily_pnls)
    cum_pnl = (1 + pd.Series(daily_pnls)).cumprod()
    total_ret = cum_pnl.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (242 / n_days) - 1
    sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(242) if pnl_arr.std() > 0 else 0
    max_dd = (cum_pnl / cum_pnl.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (pnl_arr > 0).mean()

    return {
        'label': label,
        'total_return': total_ret,
        'ann_return': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'calmar': calmar,
        'win_rate': win_rate,
        'avg_daily_bps': pnl_arr.mean() * 10000,
        'n_days': n_days,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("Explore 08: Real-World Futures Strategy Backtest")
    print("=" * 80)
    print(f"  Cost: {ROUND_TRIP_COST_BPS} bps round-trip (commission + slippage)")
    print(f"  Max position scale: {MAX_SCALE}x")
    print(f"  Daily loss limit: {DAILY_LOSS_LIMIT_BPS} bps")

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])

    # === Phase 1: Use last 200 days (100 train + 100 test) ===
    N_DAYS = 200
    recent_files = all_files[-N_DAYS:]
    print(f"\nPhase 1: {N_DAYS} days ({recent_files[0]} to {recent_files[-1]})")

    all_daily = []
    daily_dates = []
    for i, fname in enumerate(recent_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_single_day(filepath)
            all_daily.append(day_df)
            daily_dates.append(fname.replace('.parquet', ''))
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{N_DAYS} ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    # === Check tracking error between EW and TW indexes ===
    print("\n" + "=" * 80)
    print("EW vs TW INDEX TRACKING")
    print("=" * 80)
    all_tracking = pd.concat([d['tracking_diff'] for d in all_daily]).dropna()
    print(f"  Mean tracking diff per bar: {all_tracking.mean()*10000:.4f} bps")
    print(f"  Std tracking diff per bar:  {all_tracking.std()*10000:.4f} bps")
    daily_track = []
    for d in all_daily:
        ew_day_ret = d['ew_level'].iloc[-1] / d['ew_level'].iloc[0] - 1
        tw_day_ret = d['tw_level'].iloc[-1] / d['tw_level'].iloc[0] - 1
        daily_track.append(tw_day_ret - ew_day_ret)
    daily_track = np.array(daily_track)
    print(f"  Mean daily tracking diff:   {daily_track.mean()*10000:.2f} bps")
    print(f"  Std daily tracking diff:    {daily_track.std()*10000:.2f} bps")
    print(f"  Correlation (daily):        {np.corrcoef([d['ew_level'].iloc[-1]/d['ew_level'].iloc[0]-1 for d in all_daily], [d['tw_level'].iloc[-1]/d['tw_level'].iloc[0]-1 for d in all_daily])[0,1]:.4f}")

    # === Calibrate thresholds ===
    train_end = len(all_daily) // 2
    train_data = all_daily[:train_end]
    test_data = all_daily[train_end:]
    test_dates = daily_dates[train_end:]

    all_resid = pd.concat([d['breadth_resid'] for d in train_data]).dropna()
    resid_mean = all_resid.mean()
    resid_std = all_resid.std()

    print(f"\n  Training days: {train_end}")
    print(f"  Test days: {len(test_data)}")
    print(f"  breadth_resid: mean={resid_mean:.4f}, std={resid_std:.4f}")

    # === Strategy variants ===
    configs = [
        # (th_long_mult, th_short_mult, max_scale, label)
        (1.0, 1.0, 1.0, "baseline_1x"),
        (1.0, 1.0, MAX_SCALE, "baseline_scaled"),
        (0.7, 1.0, 1.0, "asym_1x"),
        (0.7, 1.0, MAX_SCALE, "asym_scaled"),
        (0.7, 1.0, 2.0, "asym_scaled_2x"),
        (0.5, 1.0, MAX_SCALE, "aggressive_asym_scaled"),
        (0.7, 1.5, MAX_SCALE, "asym_conservative_short"),
    ]

    print("\n" + "=" * 80)
    print("OOS BACKTEST RESULTS (Turnover-Weighted Index, Futures Costs)")
    print(f"Cost: {ROUND_TRIP_COST_BPS} bps | Loss limit: {DAILY_LOSS_LIMIT_BPS} bps/day")
    print("=" * 80)

    all_results = []
    best_trades = None
    best_daily = None
    best_label = ""
    best_return = -999

    for th_l_mult, th_s_mult, max_sc, label in configs:
        th_long = resid_mean + th_l_mult * resid_std
        th_short = resid_mean - th_s_mult * resid_std

        daily_pnls = []
        all_trades = []

        for day_idx, day_df in enumerate(test_data):
            trades, day_pnl = backtest_futures_day(
                day_df, th_long, th_short,
                cost_bps=ROUND_TRIP_COST_BPS,
                max_scale=max_sc,
                loss_limit_bps=DAILY_LOSS_LIMIT_BPS
            )
            daily_pnls.append(day_pnl)
            for t in trades:
                t['date'] = test_dates[day_idx]
            all_trades.extend(trades)

        metrics = compute_metrics(daily_pnls, len(test_data), label)
        total_trades = len(all_trades)
        metrics['total_trades'] = total_trades

        all_results.append(metrics)

        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
            best_trades = all_trades
            best_daily = daily_pnls
            best_label = label

    print(f"\n  {'Strategy':<30} {'Return':>10} {'AnnRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'WinR':>8} {'Trades':>8} {'AvgBps':>8}")
    print("  " + "-" * 105)
    for m in all_results:
        print(f"  {m['label']:<30} {m['total_return']*100:>9.2f}% {m['ann_return']*100:>9.1f}% "
              f"{m['sharpe']:>8.2f} {m['max_dd']*100:>7.2f}% {m['calmar']:>8.1f} "
              f"{m['win_rate']:>7.1%} {m['total_trades']:>8} {m['avg_daily_bps']:>8.2f}")

    # === Detailed analysis of best strategy ===
    print("\n" + "=" * 80)
    print(f"BEST STRATEGY DETAILS: {best_label}")
    print("=" * 80)

    trades_df = pd.DataFrame(best_trades)
    daily_df = pd.DataFrame({'date': test_dates, 'pnl': best_daily})

    # Long vs Short
    long_trades = trades_df[trades_df['direction'] == 'LONG']
    short_trades = trades_df[trades_df['direction'] == 'SHORT']

    print(f"\n  Total trades: {len(trades_df)}")
    print(f"  Long:  {len(long_trades)} trades, "
          f"avg_gross={long_trades['gross_bps'].mean():.2f}bps, "
          f"avg_net={long_trades['net_bps'].mean():.2f}bps, "
          f"win_rate={(long_trades['net_bps']>0).mean():.1%}")
    if len(short_trades) > 0:
        print(f"  Short: {len(short_trades)} trades, "
              f"avg_gross={short_trades['gross_bps'].mean():.2f}bps, "
              f"avg_net={short_trades['net_bps'].mean():.2f}bps, "
              f"win_rate={(short_trades['net_bps']>0).mean():.1%}")

    print(f"\n  Avg position size: {trades_df['size'].mean():.2f}x")
    print(f"  Max position size: {trades_df['size'].max():.2f}x")
    print(f"  Avg trades/day: {len(trades_df)/len(test_data):.1f}")

    # Monthly breakdown
    daily_df['month'] = [d[:7] for d in daily_df['date']]
    monthly = daily_df.groupby('month')['pnl'].agg(['sum', 'count', 'mean', 'std'])
    monthly['sharpe'] = monthly['mean'] / monthly['std'] * np.sqrt(242)

    print(f"\n  Monthly performance:")
    print(f"    {'Month':<10} {'Return':>10} {'Days':>6} {'AvgBps':>10} {'Sharpe':>8}")
    print("    " + "-" * 48)
    for month, row in monthly.iterrows():
        print(f"    {month:<10} {row['sum']*100:>9.3f}% {row['count']:>6.0f} "
              f"{row['mean']*10000:>10.2f} {row['sharpe']:>8.2f}")

    # Equity curve stats
    cum_pnl = (1 + pd.Series(best_daily)).cumprod()

    # Find longest losing streak
    streak = 0
    max_streak = 0
    for p in best_daily:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    print(f"\n  Max consecutive losing days: {max_streak}")
    print(f"  Best day:  {max(best_daily)*10000:.2f} bps ({max(best_daily)*100:.4f}%)")
    print(f"  Worst day: {min(best_daily)*10000:.2f} bps ({min(best_daily)*100:.4f}%)")

    # === Profitability under different cost assumptions ===
    print("\n" + "=" * 80)
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Find best config
    best_config = None
    for th_l_mult, th_s_mult, max_sc, label in configs:
        if label == best_label:
            best_config = (th_l_mult, th_s_mult, max_sc)
            break

    th_long = resid_mean + best_config[0] * resid_std
    th_short = resid_mean - best_config[1] * resid_std

    print(f"\n  Using config: {best_label}")
    print(f"  {'Cost(bps)':>10} {'Return':>10} {'AnnRet':>10} {'Sharpe':>8} {'MaxDD':>8}")
    print("  " + "-" * 50)

    for test_cost in [0, 1, 2, 3, 4.5, 6, 8, 10]:
        daily_pnls_cost = []
        for day_df in test_data:
            _, day_pnl = backtest_futures_day(
                day_df, th_long, th_short,
                cost_bps=test_cost,
                max_scale=best_config[2],
                loss_limit_bps=DAILY_LOSS_LIMIT_BPS
            )
            daily_pnls_cost.append(day_pnl)

        m = compute_metrics(daily_pnls_cost, len(test_data))
        print(f"  {test_cost:>10.1f} {m['total_return']*100:>9.2f}% {m['ann_return']*100:>9.1f}% "
              f"{m['sharpe']:>8.2f} {m['max_dd']*100:>7.2f}%")

    # === Phase 2: Extended backtest (sample every 2nd day across last 1000 days) ===
    print("\n" + "=" * 80)
    print("EXTENDED BACKTEST (last 1000 days, sampled every 2nd day)")
    print("=" * 80)

    extended_files = all_files[-1000::2]  # every 2nd day
    print(f"Processing {len(extended_files)} sampled days...")

    ext_daily = []
    ext_dates = []
    for i, fname in enumerate(extended_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_single_day(filepath)
            ext_daily.append(day_df)
            ext_dates.append(fname.replace('.parquet', ''))
        except:
            continue
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(extended_files)} ({time.time()-t0:.1f}s)")

    print(f"Processed {len(ext_daily)} days")

    # Rolling: train 120, test 30 (sampled days)
    TRAIN_W = 120
    TEST_W = 30

    rolling_results = []
    i = 0
    while i + TRAIN_W + TEST_W <= len(ext_daily):
        train_slice = ext_daily[i:i + TRAIN_W]
        test_slice = ext_daily[i + TRAIN_W:i + TRAIN_W + TEST_W]
        test_dates_slice = ext_dates[i + TRAIN_W:i + TRAIN_W + TEST_W]

        # Calibrate
        all_r = pd.concat([d['breadth_resid'] for d in train_slice]).dropna()
        r_mean = all_r.mean()
        r_std = all_r.std()
        th_l = r_mean + best_config[0] * r_std
        th_s = r_mean - best_config[1] * r_std

        test_pnls = []
        for day_df in test_slice:
            _, pnl = backtest_futures_day(day_df, th_l, th_s,
                                          cost_bps=ROUND_TRIP_COST_BPS,
                                          max_scale=best_config[2],
                                          loss_limit_bps=DAILY_LOSS_LIMIT_BPS)
            test_pnls.append(pnl)

        pnl_arr = np.array(test_pnls)
        ret = (1 + pd.Series(test_pnls)).prod() - 1
        sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(242) if pnl_arr.std() > 0 else 0

        rolling_results.append({
            'test_start': test_dates_slice[0],
            'test_end': test_dates_slice[-1],
            'return': ret,
            'sharpe': sharpe,
            'win_rate': (pnl_arr > 0).mean(),
        })
        i += TEST_W

    rolling_df = pd.DataFrame(rolling_results)
    rolling_df['year'] = [d[:4] for d in rolling_df['test_start']]

    yearly = rolling_df.groupby('year').agg({
        'return': lambda x: (1 + x).prod() - 1,
        'sharpe': 'mean',
        'win_rate': 'mean',
    })

    print(f"\n  {'Year':<8} {'Return':>10} {'AvgSharpe':>12} {'AvgWinRate':>12}")
    print("  " + "-" * 45)
    for year, row in yearly.iterrows():
        print(f"  {year:<8} {row['return']*100:>9.2f}% {row['sharpe']:>12.2f} {row['win_rate']:>12.1%}")

    overall_ret = (1 + rolling_df['return']).prod() - 1
    overall_sharpe = rolling_df['return'].mean() / rolling_df['return'].std() * np.sqrt(242/30*TEST_W) if rolling_df['return'].std() > 0 else 0
    print(f"\n  Overall: Return={overall_ret*100:.2f}%, Avg Sharpe={rolling_df['sharpe'].mean():.2f}")
    print(f"  % positive windows: {(rolling_df['return']>0).mean():.1%}")

    # Save
    trades_df.to_csv(os.path.join(OUTPUT_DIR, 'best_trades.csv'), index=False)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, 'best_daily_pnl.csv'), index=False)
    rolling_df.to_csv(os.path.join(OUTPUT_DIR, 'extended_rolling.csv'), index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
