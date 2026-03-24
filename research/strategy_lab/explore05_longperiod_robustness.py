"""
Explore 05: Long-Period Robustness Test
========================================
Critical question: Does the strategy work across 16 years and different market regimes?

Approach:
- Rolling window: train on 250 days (1 year), test on next 60 days (~3 months)
- Roll forward by 60 days each time
- Recalibrate thresholds at each roll
- Test across all available data: 2010-2026
- Use ±1.0σ (most robust from Explore 04) as primary config
- Also test P90/P10 for comparison

Report:
- Annual Sharpe ratios
- Year-by-year performance
- Regime analysis (high vol vs low vol periods)
- Drawdown analysis

NOTE: Full 3910 days × 5000 stocks would be very slow.
Optimization: sample 500 days evenly spaced across full history for initial scan,
then do full test on interesting subperiods.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore05')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N_LIQUID = 300
COST_BPS = 1.0


def load_day(filepath):
    df = pd.read_parquet(filepath)
    df = df[df['paused'] == 0].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def select_liquid_stocks(df, top_n=300):
    daily_money = df.groupby('code')['money'].sum()
    top_codes = daily_money.nlargest(top_n).index
    return df[df['code'].isin(top_codes)].copy()


def process_single_day(filepath):
    df = load_day(filepath)
    df = select_liquid_stocks(df, TOP_N_LIQUID)

    pivot_close = df.pivot_table(index='datetime', columns='code', values='close')
    stock_returns_1 = pivot_close.pct_change()

    index_return_1 = stock_returns_1.mean(axis=1)
    index_level = (1 + index_return_1).cumprod()
    index_level.iloc[0] = 1.0

    n_stocks = stock_returns_1.count(axis=1)
    breadth_up = (stock_returns_1 > 0).sum(axis=1) / n_stocks

    # Breadth residual
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
    }, index=pivot_close.index)


def backtest_day(day_df, threshold_long, threshold_short, holding_bars=1, cost_bps=1.0):
    """Backtest one day. Returns daily PnL and trade count."""
    n_bars = len(day_df)
    position = 0
    entry_bar = -1
    entry_price = 0
    total_pnl = 0
    n_trades = 0

    for bar in range(10, n_bars):
        signal = day_df['breadth_resid'].iloc[bar]
        price = day_df['index_level'].iloc[bar]

        if position != 0:
            bars_held = bar - entry_bar
            is_eod = (bar >= n_bars - 2)
            if bars_held >= holding_bars or is_eod:
                trade_return = position * (price / entry_price - 1)
                trade_return -= cost_bps / 10000
                total_pnl += trade_return
                n_trades += 1
                position = 0

        if position == 0 and bar < n_bars - holding_bars - 1:
            if signal > threshold_long:
                position = 1
                entry_bar = bar
                entry_price = price
            elif signal < threshold_short:
                position = -1
                entry_bar = bar
                entry_price = price

    if position != 0:
        price = day_df['index_level'].iloc[-1]
        trade_return = position * (price / entry_price - 1)
        trade_return -= cost_bps / 10000
        total_pnl += trade_return
        n_trades += 1

    return total_pnl, n_trades


def main():
    t0 = time.time()
    print("=" * 70)
    print("Explore 05: Long-Period Robustness Test")
    print("=" * 70)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    print(f"Total available days: {len(all_files)}")

    # Strategy 1: Sample evenly — every 4th day across full history for speed
    # This gives ~977 days to test
    step = 4
    sampled_files = all_files[::step]
    print(f"Sampling every {step}th day: {len(sampled_files)} days")

    # Process all sampled days
    print(f"\nProcessing {len(sampled_files)} days...")
    all_daily = []
    daily_dates = []
    for i, fname in enumerate(sampled_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_single_day(filepath)
            all_daily.append(day_df)
            daily_dates.append(fname.replace('.parquet', ''))
        except Exception as e:
            if i % 100 == 0:
                print(f"  Error on {fname}: {e}")
            continue
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(sampled_files)} days ({time.time()-t0:.1f}s)")

    print(f"Successfully processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    # ===== Rolling window test =====
    print("\n" + "=" * 70)
    print("ROLLING WINDOW BACKTEST")
    print("Train=60 sampled days (~240 actual days), Test=15 sampled days (~60 actual days)")
    print("=" * 70)

    TRAIN_SIZE = 60  # sampled days
    TEST_SIZE = 15   # sampled days

    rolling_results = []
    i = 0
    while i + TRAIN_SIZE + TEST_SIZE <= len(all_daily):
        train_data = all_daily[i:i + TRAIN_SIZE]
        train_dates_slice = daily_dates[i:i + TRAIN_SIZE]
        test_data = all_daily[i + TRAIN_SIZE:i + TRAIN_SIZE + TEST_SIZE]
        test_dates_slice = daily_dates[i + TRAIN_SIZE:i + TRAIN_SIZE + TEST_SIZE]

        # Calibrate thresholds on training data
        all_resid = pd.concat([d['breadth_resid'] for d in train_data]).dropna()
        resid_mean = all_resid.mean()
        resid_std = all_resid.std()

        # Config: ±1.0σ
        th_long = resid_mean + 1.0 * resid_std
        th_short = resid_mean - 1.0 * resid_std

        # Test
        test_pnls = []
        test_ntrades = []
        for day_df in test_data:
            pnl, nt = backtest_day(day_df, th_long, th_short, holding_bars=1, cost_bps=COST_BPS)
            test_pnls.append(pnl)
            test_ntrades.append(nt)

        test_pnl_arr = np.array(test_pnls)
        total_ret = (1 + pd.Series(test_pnls)).prod() - 1
        sharpe = test_pnl_arr.mean() / test_pnl_arr.std() * np.sqrt(242) if test_pnl_arr.std() > 0 else 0

        rolling_results.append({
            'train_start': train_dates_slice[0],
            'train_end': train_dates_slice[-1],
            'test_start': test_dates_slice[0],
            'test_end': test_dates_slice[-1],
            'total_return': total_ret,
            'sharpe': sharpe,
            'mean_daily_pnl': test_pnl_arr.mean(),
            'n_trades': sum(test_ntrades),
            'win_rate': (test_pnl_arr > 0).mean(),
        })

        i += TEST_SIZE  # roll forward

    rolling_df = pd.DataFrame(rolling_results)
    rolling_df.to_csv(os.path.join(OUTPUT_DIR, 'rolling_results.csv'), index=False)

    print(f"\nRolling windows: {len(rolling_df)}")
    print(f"\n  {'Test Period':<30} {'Return':>10} {'Sharpe':>8} {'WinR':>8} {'Trades':>8}")
    print("  " + "-" * 70)
    for _, row in rolling_df.iterrows():
        print(f"  {row['test_start']} ~ {row['test_end']}  "
              f"{row['total_return']:>10.4f} {row['sharpe']:>8.2f} {row['win_rate']:>8.2%} {row['n_trades']:>8}")

    # ===== Year-by-year analysis =====
    print("\n" + "=" * 70)
    print("YEAR-BY-YEAR ANALYSIS")
    print("=" * 70)

    rolling_df['test_year'] = [d[:4] for d in rolling_df['test_start']]
    yearly = rolling_df.groupby('test_year').agg({
        'total_return': lambda x: (1 + x).prod() - 1,
        'sharpe': 'mean',
        'n_trades': 'sum',
        'win_rate': 'mean',
    })

    print(f"\n  {'Year':<8} {'Cum Return':>12} {'Avg Sharpe':>12} {'Trades':>10} {'Avg WinR':>10}")
    print("  " + "-" * 55)
    for year, row in yearly.iterrows():
        print(f"  {year:<8} {row['total_return']:>12.4f} {row['sharpe']:>12.2f} "
              f"{row['n_trades']:>10.0f} {row['win_rate']:>10.2%}")

    # ===== Overall statistics =====
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    pct_positive_windows = (rolling_df['total_return'] > 0).mean()
    pct_positive_sharpe = (rolling_df['sharpe'] > 0).mean()
    avg_sharpe = rolling_df['sharpe'].mean()
    median_sharpe = rolling_df['sharpe'].median()

    print(f"  Total rolling windows: {len(rolling_df)}")
    print(f"  % windows with positive return: {pct_positive_windows:.2%}")
    print(f"  % windows with Sharpe > 0: {pct_positive_sharpe:.2%}")
    print(f"  Average Sharpe across windows: {avg_sharpe:.2f}")
    print(f"  Median Sharpe across windows: {median_sharpe:.2f}")
    print(f"  Worst window Sharpe: {rolling_df['sharpe'].min():.2f}")
    print(f"  Best window Sharpe: {rolling_df['sharpe'].max():.2f}")

    # Overall cumulative return
    all_window_returns = rolling_df['total_return'].values
    cum_return = (1 + pd.Series(all_window_returns)).prod() - 1
    print(f"  Cumulative return across all windows: {cum_return:.4f} ({cum_return*100:.2f}%)")

    # ===== Regime analysis =====
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS (high vol vs low vol)")
    print("=" * 70)

    # Use average dispersion as volatility proxy
    rolling_df['is_high_vol'] = rolling_df['sharpe'] != rolling_df['sharpe']  # placeholder

    # Compute volatility of daily PnLs for each window
    for idx in rolling_df.index:
        window_data = all_daily[idx * TEST_SIZE + TRAIN_SIZE:idx * TEST_SIZE + TRAIN_SIZE + TEST_SIZE]
        if len(window_data) > 0:
            avg_disp = np.mean([d['index_return'].std() for d in window_data if len(d) > 0])
            rolling_df.loc[idx, 'avg_vol'] = avg_disp

    if 'avg_vol' in rolling_df.columns:
        median_vol = rolling_df['avg_vol'].median()
        high_vol = rolling_df[rolling_df['avg_vol'] >= median_vol]
        low_vol = rolling_df[rolling_df['avg_vol'] < median_vol]

        print(f"\n  Low volatility regime ({len(low_vol)} windows):")
        print(f"    Avg Sharpe: {low_vol['sharpe'].mean():.2f}")
        print(f"    Avg Return: {low_vol['total_return'].mean():.4f}")
        print(f"    % positive: {(low_vol['total_return'] > 0).mean():.2%}")

        print(f"\n  High volatility regime ({len(high_vol)} windows):")
        print(f"    Avg Sharpe: {high_vol['sharpe'].mean():.2f}")
        print(f"    Avg Return: {high_vol['total_return'].mean():.4f}")
        print(f"    % positive: {(high_vol['total_return'] > 0).mean():.2%}")

    print(f"\nTotal time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
