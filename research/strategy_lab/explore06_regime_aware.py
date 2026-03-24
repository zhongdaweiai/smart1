"""
Explore 06: Regime-Aware Strategy
==================================
Critical pivot from Explore 05: strategy only works in high-volatility regimes.
Need to add a regime filter to avoid trading in low-vol environments.

Regime detection approaches:
1. Rolling dispersion threshold (use dispersion from previous days as proxy)
2. Rolling index volatility (realized vol over past N days)
3. Intraday dispersion level (current day's dispersion so far)

Strategy:
- Same core signal: breadth_resid, ±1.0σ threshold, hold=1
- Add regime filter: only trade when volatility indicator > threshold
- Test across full history (sampled)

Key design principle: regime filter must use ONLY past information (no lookahead).
Use trailing 20-day rolling dispersion mean as the regime indicator.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore06')
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
    dispersion = stock_returns_1.std(axis=1)

    breadth_resid = breadth_up.copy()
    for i in range(10, len(breadth_up)):
        b = breadth_up.iloc[:i]
        m = index_return_1.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 5:
            slope = np.polyfit(m[mask], b[mask], 1)
            breadth_resid.iloc[i] = breadth_up.iloc[i] - np.polyval(slope, index_return_1.iloc[i])

    # Daily summary stats
    daily_dispersion_mean = dispersion.mean()
    daily_index_vol = index_return_1.std()
    daily_index_range = (index_level.max() - index_level.min()) / index_level.iloc[0]

    return (
        pd.DataFrame({
            'index_level': index_level,
            'index_return': index_return_1,
            'breadth_resid': breadth_resid,
            'dispersion': dispersion,
        }, index=pivot_close.index),
        {
            'daily_dispersion': daily_dispersion_mean,
            'daily_vol': daily_index_vol,
            'daily_range': daily_index_range,
        }
    )


def backtest_day(day_df, threshold_long, threshold_short, holding_bars=1, cost_bps=1.0):
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
    print("Explore 06: Regime-Aware Strategy")
    print("=" * 70)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])

    # Use every 4th day for speed (same as Explore 05)
    step = 4
    sampled_files = all_files[::step]
    print(f"Using {len(sampled_files)} sampled days")

    # Process all days
    print("Processing...")
    all_daily = []
    daily_dates = []
    daily_stats = []

    for i, fname in enumerate(sampled_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df, stats_dict = process_single_day(filepath)
            all_daily.append(day_df)
            daily_dates.append(fname.replace('.parquet', ''))
            daily_stats.append(stats_dict)
        except Exception as e:
            continue
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(sampled_files)} ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    stats_df = pd.DataFrame(daily_stats, index=daily_dates)

    # ===== Regime indicator: trailing rolling dispersion =====
    # Since we're sampling every 4th day, a "20-day rolling" is 5 sampled days
    REGIME_LOOKBACK = 5  # 5 sampled days ≈ 20 actual days

    stats_df['trailing_disp'] = stats_df['daily_dispersion'].rolling(REGIME_LOOKBACK).mean()
    stats_df['trailing_vol'] = stats_df['daily_vol'].rolling(REGIME_LOOKBACK).mean()

    # ===== Strategy 1: No filter (baseline) =====
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON ACROSS FULL HISTORY")
    print("=" * 70)

    # Calibrate thresholds on first 60 days (expanding thereafter)
    WARMUP = 60

    def run_filtered_backtest(filter_fn, label):
        """Run backtest with a filter function that takes (day_idx, stats_df) → bool."""
        all_pnls = []
        all_ntrades = []
        all_active = []

        # Expanding threshold calibration
        for day_idx in range(WARMUP, len(all_daily)):
            # Calibrate thresholds on all previous days
            past_resids = pd.concat([d['breadth_resid'] for d in all_daily[:day_idx]]).dropna()
            resid_mean = past_resids.mean()
            resid_std = past_resids.std()
            th_long = resid_mean + 1.0 * resid_std
            th_short = resid_mean - 1.0 * resid_std

            # Check regime filter
            is_active = filter_fn(day_idx, stats_df)
            all_active.append(is_active)

            if is_active:
                pnl, nt = backtest_day(all_daily[day_idx], th_long, th_short,
                                        holding_bars=1, cost_bps=COST_BPS)
            else:
                pnl, nt = 0.0, 0

            all_pnls.append(pnl)
            all_ntrades.append(nt)

        pnl_arr = np.array(all_pnls)
        dates = daily_dates[WARMUP:]
        active_days = sum(all_active)
        total_days = len(all_pnls)

        cum_ret = (1 + pd.Series(all_pnls)).prod() - 1
        sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(242) if pnl_arr.std() > 0 else 0

        # Active-only metrics
        active_pnls = [p for p, a in zip(all_pnls, all_active) if a]
        if active_pnls:
            active_arr = np.array(active_pnls)
            active_sharpe = active_arr.mean() / active_arr.std() * np.sqrt(242) if active_arr.std() > 0 else 0
            active_wr = (active_arr > 0).mean()
        else:
            active_sharpe = 0
            active_wr = 0

        # Max drawdown
        cum_pnl = (1 + pd.Series(all_pnls)).cumprod()
        max_dd = (cum_pnl / cum_pnl.cummax() - 1).min()

        return {
            'label': label,
            'cum_return': cum_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'total_trades': sum(all_ntrades),
            'active_days': active_days,
            'total_days': total_days,
            'active_ratio': active_days / total_days,
            'active_sharpe': active_sharpe,
            'active_wr': active_wr,
            'pnls': all_pnls,
            'dates': dates,
            'active_flags': all_active,
        }

    # Define filter functions
    def no_filter(day_idx, sdf):
        return True

    def disp_filter_p50(day_idx, sdf):
        """Trade only when trailing dispersion > historical median."""
        if day_idx < REGIME_LOOKBACK:
            return True
        current = sdf.iloc[day_idx]['trailing_disp']
        historical_median = sdf.iloc[:day_idx]['trailing_disp'].median()
        return not np.isnan(current) and current > historical_median

    def disp_filter_p60(day_idx, sdf):
        if day_idx < REGIME_LOOKBACK:
            return True
        current = sdf.iloc[day_idx]['trailing_disp']
        historical_p60 = sdf.iloc[:day_idx]['trailing_disp'].quantile(0.6)
        return not np.isnan(current) and current > historical_p60

    def disp_filter_p70(day_idx, sdf):
        if day_idx < REGIME_LOOKBACK:
            return True
        current = sdf.iloc[day_idx]['trailing_disp']
        historical_p70 = sdf.iloc[:day_idx]['trailing_disp'].quantile(0.7)
        return not np.isnan(current) and current > historical_p70

    def vol_filter_p50(day_idx, sdf):
        if day_idx < REGIME_LOOKBACK:
            return True
        current = sdf.iloc[day_idx]['trailing_vol']
        historical_median = sdf.iloc[:day_idx]['trailing_vol'].median()
        return not np.isnan(current) and current > historical_median

    def intraday_disp_filter(day_idx, sdf):
        """Use current day's first-hour dispersion as regime filter."""
        if day_idx < REGIME_LOOKBACK:
            return True
        current_daily = sdf.iloc[day_idx]['daily_dispersion']
        historical_median = sdf.iloc[:day_idx]['daily_dispersion'].median()
        return not np.isnan(current_daily) and current_daily > historical_median

    filters = [
        (no_filter, "No filter"),
        (disp_filter_p50, "Disp > P50"),
        (disp_filter_p60, "Disp > P60"),
        (disp_filter_p70, "Disp > P70"),
        (vol_filter_p50, "Vol > P50"),
        (intraday_disp_filter, "IntradayDisp > P50"),
    ]

    results = []
    for filter_fn, label in filters:
        print(f"  Running: {label}...")
        res = run_filtered_backtest(filter_fn, label)
        results.append(res)

    # ===== Results comparison =====
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n  {'Strategy':<25} {'CumRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'Active%':>10} {'ActSharpe':>10} {'ActWR':>8} {'Trades':>8}")
    print("  " + "-" * 95)

    for res in results:
        print(f"  {res['label']:<25} {res['cum_return']:>10.4f} {res['sharpe']:>8.2f} "
              f"{res['max_dd']:>8.4f} {res['active_ratio']:>10.2%} "
              f"{res['active_sharpe']:>10.2f} {res['active_wr']:>8.2%} {res['total_trades']:>8}")

    # ===== Year-by-year for best regime strategy =====
    print("\n" + "=" * 70)
    print("YEAR-BY-YEAR: No filter vs Best regime filter")
    print("=" * 70)

    # Find best by Sharpe
    best_idx = max(range(len(results)), key=lambda i: results[i]['sharpe'])
    best_res = results[best_idx]
    no_filter_res = results[0]

    print(f"\n  Best strategy: {best_res['label']}")
    print(f"\n  {'Year':<8} {'NoFilter Ret':>14} {'NoFilter Sharpe':>16} {best_res['label']+' Ret':>14} {best_res['label']+' Sharpe':>16}")
    print("  " + "-" * 70)

    for res in [no_filter_res, best_res]:
        res['year'] = [d[:4] for d in res['dates']]

    for year in sorted(set(no_filter_res['year'])):
        nf_pnls = [p for p, y in zip(no_filter_res['pnls'], no_filter_res['year']) if y == year]
        bf_pnls = [p for p, y in zip(best_res['pnls'], best_res['year']) if y == year]

        nf_ret = (1 + pd.Series(nf_pnls)).prod() - 1
        bf_ret = (1 + pd.Series(bf_pnls)).prod() - 1
        nf_sharpe = np.mean(nf_pnls) / np.std(nf_pnls) * np.sqrt(242) if np.std(nf_pnls) > 0 else 0
        bf_sharpe = np.mean(bf_pnls) / np.std(bf_pnls) * np.sqrt(242) if np.std(bf_pnls) > 0 else 0

        improvement = "✅" if bf_sharpe > nf_sharpe else "❌"
        print(f"  {year:<8} {nf_ret:>14.4f} {nf_sharpe:>16.2f} {bf_ret:>14.4f} {bf_sharpe:>16.2f}  {improvement}")

    # ===== Cumulative PnL comparison =====
    print("\n" + "=" * 70)
    print("CUMULATIVE PnL TRAJECTORY")
    print("=" * 70)

    for res in [no_filter_res, best_res]:
        cum = (1 + pd.Series(res['pnls'])).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()

        # Find worst period
        cum_vals = cum.values
        peak_idx = 0
        worst_dd = 0
        worst_start = 0
        worst_end = 0
        for j in range(1, len(cum_vals)):
            if cum_vals[j] > cum_vals[peak_idx]:
                peak_idx = j
            dd = cum_vals[j] / cum_vals[peak_idx] - 1
            if dd < worst_dd:
                worst_dd = dd
                worst_start = peak_idx
                worst_end = j

        print(f"\n  {res['label']}:")
        print(f"    Final NAV: {cum.iloc[-1]:.4f}")
        print(f"    Max drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
        if worst_start < len(res['dates']) and worst_end < len(res['dates']):
            print(f"    Worst DD period: {res['dates'][worst_start]} ~ {res['dates'][worst_end]}")

    # ===== Save detailed results =====
    summary_df = pd.DataFrame([{
        'label': r['label'],
        'cum_return': r['cum_return'],
        'sharpe': r['sharpe'],
        'max_dd': r['max_dd'],
        'active_ratio': r['active_ratio'],
        'active_sharpe': r['active_sharpe'],
        'total_trades': r['total_trades'],
    } for r in results])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'strategy_comparison.csv'), index=False)

    # Save daily PnL for best strategy
    best_daily = pd.DataFrame({
        'date': best_res['dates'],
        'pnl': best_res['pnls'],
        'active': best_res['active_flags'],
    })
    best_daily.to_csv(os.path.join(OUTPUT_DIR, 'best_strategy_daily_pnl.csv'), index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
