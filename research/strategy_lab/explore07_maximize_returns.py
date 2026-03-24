"""
Explore 07: Maximize Returns — Creative Parameter Search
=========================================================
Goals: push the strategy to its return ceiling by exploring:

1. Holding periods: 1, 2, 3, 5, 10, 20, 30, 60 bars
2. Asymmetric thresholds: long signal is stronger, use lower entry threshold for longs
3. Session-specific trading: only trade in close session (IC is 4x stronger there)
4. Adaptive position sizing: scale position by |signal| strength
5. Re-entry: if still in signal zone after exit, immediately re-enter
6. Multi-timeframe: combine 1-bar and 5-bar breadth signals
7. Momentum regime overlay: only long when index trending up, only short when down
8. Cumulative intraday breadth: running sum of breadth_resid through the day

Use 200-day dataset: first 100 train, last 100 OOS test.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore07')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_DAYS = 200
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


def process_single_day_rich(filepath):
    """Compute rich feature set for creative strategy variants."""
    df = load_day(filepath)
    df = select_liquid_stocks(df, TOP_N_LIQUID)

    pivot_close = df.pivot_table(index='datetime', columns='code', values='close')
    pivot_money = df.pivot_table(index='datetime', columns='code', values='money')
    stock_returns_1 = pivot_close.pct_change()

    index_return_1 = stock_returns_1.mean(axis=1)
    index_level = (1 + index_return_1).cumprod()
    index_level.iloc[0] = 1.0

    n_stocks = stock_returns_1.count(axis=1)
    breadth_up = (stock_returns_1 > 0).sum(axis=1) / n_stocks
    dispersion = stock_returns_1.std(axis=1)

    # breadth_resid (expanding regression within day)
    breadth_resid = breadth_up.copy()
    for i in range(10, len(breadth_up)):
        b = breadth_up.iloc[:i]
        m = index_return_1.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 5:
            slope = np.polyfit(m[mask], b[mask], 1)
            breadth_resid.iloc[i] = breadth_up.iloc[i] - np.polyval(slope, index_return_1.iloc[i])

    # Multi-timeframe: 5-bar rolling breadth
    stock_returns_5 = pivot_close.pct_change(5)
    breadth_up_5 = (stock_returns_5 > 0).sum(axis=1) / stock_returns_5.count(axis=1)

    # Money-weighted breadth
    weights = pivot_money.div(pivot_money.sum(axis=1), axis=0)
    weighted_breadth = ((stock_returns_1 > 0).astype(float) * weights).sum(axis=1)

    # Cumulative intraday breadth_resid (running sum through day)
    cum_breadth_resid = breadth_resid.cumsum()

    # Index momentum (rolling 10-bar return)
    index_mom_10 = index_level.pct_change(10)
    index_mom_30 = index_level.pct_change(30)

    # Intraday trend: index return from open
    index_from_open = index_level / index_level.iloc[0] - 1

    # Breadth acceleration: change in breadth_resid
    breadth_accel = breadth_resid.diff()

    # Bar position
    bar_pos = pd.Series(range(len(pivot_close.index)), index=pivot_close.index)

    # Extreme breadth: how far from 0.5
    breadth_extreme = (breadth_up - 0.5).abs()

    return pd.DataFrame({
        'index_level': index_level,
        'index_return': index_return_1,
        'breadth_up': breadth_up,
        'breadth_resid': breadth_resid,
        'breadth_up_5': breadth_up_5,
        'weighted_breadth': weighted_breadth,
        'cum_breadth_resid': cum_breadth_resid,
        'breadth_accel': breadth_accel,
        'breadth_extreme': breadth_extreme,
        'dispersion': dispersion,
        'index_mom_10': index_mom_10,
        'index_mom_30': index_mom_30,
        'index_from_open': index_from_open,
        'bar_pos': bar_pos,
    }, index=pivot_close.index)


def backtest_advanced(day_df, config):
    """
    Advanced backtester with multiple strategy modes.

    config dict:
    - signal_col: which column to use as signal
    - th_long: long threshold
    - th_short: short threshold
    - holding_bars: how long to hold
    - cost_bps: round-trip cost
    - session_filter: None or (start_bar, end_bar) to restrict trading
    - allow_reentry: bool, if True re-enter immediately after exit if signal persists
    - long_only: bool
    - short_only: bool
    - scale_by_signal: bool, if True scale PnL by |signal| (proxy for position sizing)
    - trend_filter: None or 'index_mom_10' to only trade in trend direction
    - combo_signal: None or secondary signal column to combine
    - combo_weight: weight for secondary signal (primary=1.0)
    """
    signal_col = config.get('signal_col', 'breadth_resid')
    th_long = config['th_long']
    th_short = config['th_short']
    holding_bars = config.get('holding_bars', 1)
    cost_bps = config.get('cost_bps', 1.0)
    session_filter = config.get('session_filter', None)
    allow_reentry = config.get('allow_reentry', False)
    long_only = config.get('long_only', False)
    short_only = config.get('short_only', False)
    scale_by_signal = config.get('scale_by_signal', False)
    trend_filter = config.get('trend_filter', None)
    combo_signal = config.get('combo_signal', None)
    combo_weight = config.get('combo_weight', 0.5)

    n_bars = len(day_df)
    position = 0
    entry_bar = -1
    entry_price = 0
    entry_signal_strength = 1.0
    total_pnl = 0
    n_trades = 0

    for bar in range(10, n_bars):
        # Session filter
        if session_filter:
            bar_pos = day_df['bar_pos'].iloc[bar]
            if bar_pos < session_filter[0] or bar_pos > session_filter[1]:
                # Force close if out of session
                if position != 0:
                    price = day_df['index_level'].iloc[bar]
                    trade_return = position * (price / entry_price - 1) * entry_signal_strength
                    trade_return -= cost_bps / 10000
                    total_pnl += trade_return
                    n_trades += 1
                    position = 0
                continue

        # Build signal
        signal = day_df[signal_col].iloc[bar]
        if combo_signal and combo_signal in day_df.columns:
            combo_val = day_df[combo_signal].iloc[bar]
            if not np.isnan(combo_val):
                signal = signal + combo_weight * combo_val

        price = day_df['index_level'].iloc[bar]

        # Check exit
        if position != 0:
            bars_held = bar - entry_bar
            is_eod = (bar >= n_bars - 2)
            if bars_held >= holding_bars or is_eod:
                trade_return = position * (price / entry_price - 1) * entry_signal_strength
                trade_return -= cost_bps / 10000
                total_pnl += trade_return
                n_trades += 1
                position = 0

                # Re-entry check
                if not allow_reentry:
                    continue

        # Check entry
        if position == 0 and bar < n_bars - max(holding_bars, 2) - 1:
            # Trend filter
            if trend_filter and trend_filter in day_df.columns:
                trend = day_df[trend_filter].iloc[bar]
                if not np.isnan(trend):
                    if trend > 0 and signal < th_short:  # uptrend but short signal → skip
                        continue
                    if trend < 0 and signal > th_long:  # downtrend but long signal → skip
                        continue

            if signal > th_long and not short_only:
                position = 1
                entry_bar = bar
                entry_price = price
                entry_signal_strength = abs(signal) / abs(th_long) if scale_by_signal else 1.0
            elif signal < th_short and not long_only:
                position = -1
                entry_bar = bar
                entry_price = price
                entry_signal_strength = abs(signal) / abs(th_short) if scale_by_signal else 1.0

    # Force close
    if position != 0:
        price = day_df['index_level'].iloc[-1]
        trade_return = position * (price / entry_price - 1) * entry_signal_strength
        trade_return -= cost_bps / 10000
        total_pnl += trade_return
        n_trades += 1

    return total_pnl, n_trades


def evaluate_strategy(all_daily, daily_dates, config, label, train_end_idx):
    """Evaluate a strategy config on train and test splits."""
    train_data = all_daily[:train_end_idx]
    test_data = all_daily[train_end_idx:]
    test_dates = daily_dates[train_end_idx:]

    # Calibrate thresholds
    sig_col = config.get('signal_col', 'breadth_resid')
    all_sig = pd.concat([d[sig_col] for d in train_data]).dropna()
    sig_mean = all_sig.mean()
    sig_std = all_sig.std()

    th_mult_long = config.get('th_mult_long', 1.0)
    th_mult_short = config.get('th_mult_short', 1.0)
    config['th_long'] = sig_mean + th_mult_long * sig_std
    config['th_short'] = sig_mean - th_mult_short * sig_std

    # Run on test
    test_pnls = []
    test_ntrades = []
    for day_df in test_data:
        pnl, nt = backtest_advanced(day_df, config)
        test_pnls.append(pnl)
        test_ntrades.append(nt)

    pnl_arr = np.array(test_pnls)
    total_ret = (1 + pd.Series(test_pnls)).prod() - 1

    # Avoid division by zero
    if pnl_arr.std() > 0:
        sharpe = pnl_arr.mean() / pnl_arr.std() * np.sqrt(242)
    else:
        sharpe = 0

    cum_pnl = (1 + pd.Series(test_pnls)).cumprod()
    max_dd = (cum_pnl / cum_pnl.cummax() - 1).min()

    total_trades = sum(test_ntrades)
    win_rate = (pnl_arr > 0).mean()

    return {
        'label': label,
        'total_return': total_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_daily_ret': pnl_arr.mean(),
        'n_days': len(test_data),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("Explore 07: Maximize Returns — Creative Parameter Search")
    print("=" * 80)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    recent_files = all_files[-N_DAYS:]
    print(f"Using {len(recent_files)} days: {recent_files[0]} to {recent_files[-1]}")

    # Process all days
    all_daily = []
    daily_dates = []
    for i, fname in enumerate(recent_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_single_day_rich(filepath)
            all_daily.append(day_df)
            daily_dates.append(fname.replace('.parquet', ''))
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(recent_files)} days ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    train_end = len(all_daily) // 2

    # ===== STRATEGY VARIANTS =====
    strategies = []

    # --- Group 1: Holding period sweep ---
    for hold in [1, 2, 3, 5, 10, 20, 30, 60]:
        strategies.append((
            {'signal_col': 'breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
             'holding_bars': hold, 'cost_bps': COST_BPS},
            f"hold={hold}"
        ))

    # --- Group 2: Threshold sweep ---
    for th in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        strategies.append((
            {'signal_col': 'breadth_resid', 'th_mult_long': th, 'th_mult_short': th,
             'holding_bars': 1, 'cost_bps': COST_BPS},
            f"th=±{th}σ_h1"
        ))

    # --- Group 3: Asymmetric (long-favored) ---
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS},
        "asym_L0.7/S1.0_h1"
    ))
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.5, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS},
        "asym_L0.5/S1.0_h1"
    ))
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.5,
         'holding_bars': 1, 'cost_bps': COST_BPS},
        "asym_L0.7/S1.5_h1"
    ))

    # --- Group 4: Long-only ---
    for hold in [1, 3, 5, 10]:
        strategies.append((
            {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
             'holding_bars': hold, 'cost_bps': COST_BPS, 'long_only': True},
            f"long_only_h{hold}"
        ))

    # --- Group 5: Close session only (bar 180-238, i.e. 14:01-14:59) ---
    for hold in [1, 3, 5]:
        strategies.append((
            {'signal_col': 'breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
             'holding_bars': hold, 'cost_bps': COST_BPS,
             'session_filter': (180, 238)},
            f"close_only_h{hold}"
        ))

    # --- Group 6: Afternoon session (bar 120-238, i.e. 13:01-14:59) ---
    for hold in [1, 3, 5]:
        strategies.append((
            {'signal_col': 'breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
             'holding_bars': hold, 'cost_bps': COST_BPS,
             'session_filter': (120, 238)},
            f"afternoon_h{hold}"
        ))

    # --- Group 7: Re-entry allowed ---
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS, 'allow_reentry': True},
        "reentry_h1"
    ))
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS, 'allow_reentry': True},
        "reentry_asym_h1"
    ))

    # --- Group 8: Signal-scaled position ---
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS, 'scale_by_signal': True},
        "scaled_h1"
    ))
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS, 'scale_by_signal': True},
        "scaled_asym_h1"
    ))

    # --- Group 9: Trend filter (only trade with trend) ---
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS, 'trend_filter': 'index_mom_10'},
        "trend10_h1"
    ))
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS, 'trend_filter': 'index_mom_30'},
        "trend30_asym_h1"
    ))

    # --- Group 10: Alternative signals ---
    strategies.append((
        {'signal_col': 'cum_breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 5, 'cost_bps': COST_BPS},
        "cum_resid_h5"
    ))
    strategies.append((
        {'signal_col': 'cum_breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 10, 'cost_bps': COST_BPS},
        "cum_resid_h10"
    ))
    strategies.append((
        {'signal_col': 'cum_breadth_resid', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 20, 'cost_bps': COST_BPS},
        "cum_resid_h20"
    ))
    strategies.append((
        {'signal_col': 'breadth_up_5', 'th_mult_long': 1.0, 'th_mult_short': 1.0,
         'holding_bars': 5, 'cost_bps': COST_BPS},
        "breadth5_h5"
    ))

    # --- Group 11: Combo signal (breadth_resid + breadth_accel) ---
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS,
         'combo_signal': 'breadth_accel', 'combo_weight': 0.5},
        "combo_accel_h1"
    ))

    # --- Group 12: Session + asymmetric + scale (kitchen sink) ---
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.7, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS,
         'session_filter': (150, 238), 'scale_by_signal': True, 'allow_reentry': True},
        "kitchen_sink_pm"
    ))
    strategies.append((
        {'signal_col': 'breadth_resid', 'th_mult_long': 0.5, 'th_mult_short': 1.0,
         'holding_bars': 1, 'cost_bps': COST_BPS,
         'session_filter': (150, 238), 'scale_by_signal': True, 'allow_reentry': True},
        "kitchen_sink_pm_v2"
    ))

    # --- Group 13: Wider threshold + longer hold (fewer trades, lower cost sensitivity) ---
    for hold in [5, 10, 20, 30]:
        strategies.append((
            {'signal_col': 'breadth_resid', 'th_mult_long': 0.5, 'th_mult_short': 0.5,
             'holding_bars': hold, 'cost_bps': COST_BPS},
            f"wide_th0.5_h{hold}"
        ))

    # ===== RUN ALL =====
    print(f"\nRunning {len(strategies)} strategy variants...")
    results = []
    for config, label in strategies:
        res = evaluate_strategy(all_daily, daily_dates, config, label, train_end)
        results.append(res)

    print(f"Done in {time.time()-t0:.1f}s")

    # ===== RESULTS =====
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('total_return', ascending=False)

    print("\n" + "=" * 80)
    print("ALL RESULTS (sorted by total return, OOS 100 days)")
    print("=" * 80)

    print(f"\n  {'Strategy':<25} {'TotRet':>10} {'AnnRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinR':>8} {'AvgDailyBps':>13}")
    print("  " + "-" * 100)

    for _, row in results_df.iterrows():
        ann_ret = (1 + row['total_return']) ** (242 / row['n_days']) - 1
        avg_bps = row['avg_daily_ret'] * 10000
        print(f"  {row['label']:<25} {row['total_return']:>10.4f} {ann_ret:>10.4f} "
              f"{row['sharpe']:>8.2f} {row['max_dd']:>8.4f} {row['total_trades']:>8.0f} "
              f"{row['win_rate']:>8.2%} {avg_bps:>13.2f}")

    # ===== TOP 10 ANALYSIS =====
    print("\n" + "=" * 80)
    print("TOP 10 by TOTAL RETURN")
    print("=" * 80)
    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        ann_ret = (1 + row['total_return']) ** (242 / row['n_days']) - 1
        avg_bps = row['avg_daily_ret'] * 10000
        calmar = ann_ret / abs(row['max_dd']) if row['max_dd'] != 0 else 0
        print(f"\n  {row['label']}:")
        print(f"    Return: {row['total_return']*100:.2f}%  Ann: {ann_ret*100:.1f}%  "
              f"Sharpe: {row['sharpe']:.2f}  MaxDD: {row['max_dd']*100:.2f}%  Calmar: {calmar:.1f}")
        print(f"    Trades: {row['total_trades']:.0f}  WinRate: {row['win_rate']:.1%}  AvgDaily: {avg_bps:.2f}bps")

    print("\n" + "=" * 80)
    print("TOP 10 by SHARPE")
    print("=" * 80)
    top10_sharpe = results_df.sort_values('sharpe', ascending=False).head(10)
    for _, row in top10_sharpe.iterrows():
        ann_ret = (1 + row['total_return']) ** (242 / row['n_days']) - 1
        print(f"  {row['label']:<25} Sharpe={row['sharpe']:.2f}  Return={row['total_return']*100:.2f}%  "
              f"MaxDD={row['max_dd']*100:.2f}%  Trades={row['total_trades']:.0f}")

    # Save
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'all_results.csv'), index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
