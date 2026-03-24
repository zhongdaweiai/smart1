"""
Explore 04: Intraday Trading Strategy Backtest
===============================================
Convert breadth_resid signal into a tradeable intraday strategy.

Strategy design:
- Universe: trade a synthetic "index" (equal-weight top 300 liquid stocks)
  In practice, would trade index ETF (510300/510500) options or stock index futures (IF/IC/IM)
- Signal: breadth_resid (breadth_up orthogonalized to index momentum)
- Timing: generate signal every bar, but only trade when signal exceeds threshold

Trading rules:
1. Signal generation: breadth_resid = breadth_up - beta * index_mom (expanding regression)
2. Entry:
   - Long when breadth_resid > upper threshold (many stocks up despite weak index → expect catch-up)
   - Short when breadth_resid < lower threshold (few stocks up despite strong index → expect reversal)
   - No position when signal is in the neutral zone
3. Exit: after fixed holding period (K bars), or at end of day
4. No overnight positions (pure intraday)

Backtest:
- Train period: first 100 days (calibrate thresholds)
- Test period: last 100 days (out-of-sample)
- Transaction cost: 0.5 bps per trade (one-way, for futures)
- Slippage: 0.5 bps

Metrics:
- Total return, annualized return, Sharpe ratio
- Max drawdown
- Win rate, profit factor
- Average trade return, number of trades
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore04')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_DAYS = 200
TOP_N_LIQUID = 300
COST_BPS = 1.0        # round-trip cost in bps (0.5 each way for futures)
HOLDING_BARS = 5       # hold for 5 bars then exit


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
    """Compute signals and index for one day."""
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

    # Breadth residual (simple approach: breadth - expanding beta * mom)
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
        'dispersion': dispersion,
    }, index=pivot_close.index)


def backtest_strategy(daily_data_list, daily_dates, threshold_long, threshold_short,
                      holding_bars=5, cost_bps=1.0, label=""):
    """
    Run intraday backtest across multiple days.

    For each bar:
    - If no position and signal > threshold_long → go long
    - If no position and signal < threshold_short → go short
    - Exit after holding_bars, or at end of day

    Returns performance metrics dict.
    """
    all_trades = []
    daily_pnl = []

    for day_idx, (day_df, date_str) in enumerate(zip(daily_data_list, daily_dates)):
        idx = day_df.index
        n_bars = len(idx)
        position = 0  # 0=flat, 1=long, -1=short
        entry_bar = -1
        entry_price = 0
        day_trades = []

        for bar in range(10, n_bars):  # skip first 10 bars for signal warmup
            signal = day_df['breadth_resid'].iloc[bar]
            price = day_df['index_level'].iloc[bar]

            # Check exit
            if position != 0:
                bars_held = bar - entry_bar
                is_eod = (bar >= n_bars - 2)  # exit 2 bars before close

                if bars_held >= holding_bars or is_eod:
                    # Exit
                    exit_price = price
                    trade_return = position * (exit_price / entry_price - 1)
                    trade_return -= cost_bps / 10000  # round-trip cost
                    day_trades.append({
                        'date': date_str,
                        'entry_bar': entry_bar,
                        'exit_bar': bar,
                        'direction': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': trade_return,
                        'bars_held': bars_held,
                        'signal_at_entry': day_df['breadth_resid'].iloc[entry_bar],
                    })
                    position = 0

            # Check entry (only if flat)
            if position == 0 and bar < n_bars - holding_bars - 1:
                if signal > threshold_long:
                    position = 1
                    entry_bar = bar
                    entry_price = price
                elif signal < threshold_short:
                    position = -1
                    entry_bar = bar
                    entry_price = price

        # Force close any remaining position at EOD
        if position != 0:
            exit_price = day_df['index_level'].iloc[-1]
            trade_return = position * (exit_price / entry_price - 1)
            trade_return -= cost_bps / 10000
            day_trades.append({
                'date': date_str,
                'entry_bar': entry_bar,
                'exit_bar': n_bars - 1,
                'direction': position,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return': trade_return,
                'bars_held': n_bars - 1 - entry_bar,
                'signal_at_entry': day_df['breadth_resid'].iloc[entry_bar],
            })

        all_trades.extend(day_trades)
        day_ret = sum(t['return'] for t in day_trades)
        daily_pnl.append({'date': date_str, 'pnl': day_ret, 'n_trades': len(day_trades)})

    if not all_trades:
        return None

    trades_df = pd.DataFrame(all_trades)
    daily_df = pd.DataFrame(daily_pnl)

    # Compute metrics
    total_return = (1 + daily_df['pnl']).prod() - 1
    n_days = len(daily_df)
    ann_factor = 242  # A-share trading days per year
    ann_return = (1 + total_return) ** (ann_factor / n_days) - 1

    daily_returns = daily_df['pnl']
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(ann_factor) if daily_returns.std() > 0 else 0

    cum_pnl = (1 + daily_returns).cumprod()
    max_dd = (cum_pnl / cum_pnl.cummax() - 1).min()

    n_trades = len(trades_df)
    win_rate = (trades_df['return'] > 0).mean() if n_trades > 0 else 0
    avg_trade_ret = trades_df['return'].mean() if n_trades > 0 else 0

    winning = trades_df[trades_df['return'] > 0]['return'].sum()
    losing = abs(trades_df[trades_df['return'] < 0]['return'].sum())
    profit_factor = winning / losing if losing > 0 else float('inf')

    avg_trades_per_day = n_trades / n_days

    metrics = {
        'label': label,
        'total_return': total_return,
        'ann_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'avg_trades_per_day': avg_trades_per_day,
        'win_rate': win_rate,
        'avg_trade_return': avg_trade_ret,
        'profit_factor': profit_factor,
        'n_days': n_days,
    }

    return metrics, trades_df, daily_df


def main():
    t0 = time.time()
    print("=" * 70)
    print("Explore 04: Intraday Trading Strategy Backtest")
    print("=" * 70)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    recent_files = all_files[-N_DAYS:]
    print(f"Using {len(recent_files)} days: {recent_files[0]} to {recent_files[-1]}")

    # Process all days
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
            print(f"  Processed {i+1}/{len(recent_files)} days ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    # Split into train/test
    n_train = len(all_daily) // 2
    train_data = all_daily[:n_train]
    train_dates = daily_dates[:n_train]
    test_data = all_daily[n_train:]
    test_dates = daily_dates[n_train:]
    print(f"Train: {len(train_data)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"Test:  {len(test_data)} days ({test_dates[0]} to {test_dates[-1]})")

    # ===== Calibrate thresholds on training data =====
    print("\n" + "=" * 70)
    print("CALIBRATING THRESHOLDS ON TRAINING DATA")
    print("=" * 70)

    # Get distribution of breadth_resid
    all_resid = pd.concat([d['breadth_resid'] for d in train_data]).dropna()
    print(f"\n  breadth_resid distribution (train):")
    print(f"    mean = {all_resid.mean():.4f}")
    print(f"    std  = {all_resid.std():.4f}")
    for pct in [5, 10, 20, 25, 50, 75, 80, 90, 95]:
        print(f"    P{pct} = {all_resid.quantile(pct/100):.4f}")

    # Test different threshold strategies on training data
    print("\n" + "=" * 70)
    print("TRAINING DATA: Strategy sweep")
    print("=" * 70)

    resid_std = all_resid.std()
    resid_mean = all_resid.mean()

    threshold_configs = [
        # (long_threshold, short_threshold, label)
        (resid_mean + 0.5*resid_std, resid_mean - 0.5*resid_std, '±0.5σ'),
        (resid_mean + 1.0*resid_std, resid_mean - 1.0*resid_std, '±1.0σ'),
        (resid_mean + 1.5*resid_std, resid_mean - 1.5*resid_std, '±1.5σ'),
        (resid_mean + 2.0*resid_std, resid_mean - 2.0*resid_std, '±2.0σ'),
        (all_resid.quantile(0.8), all_resid.quantile(0.2), 'P80/P20'),
        (all_resid.quantile(0.9), all_resid.quantile(0.1), 'P90/P10'),
        (all_resid.quantile(0.7), all_resid.quantile(0.3), 'P70/P30'),
    ]

    holding_periods = [1, 3, 5, 10]

    print(f"\n  {'Config':<15} {'Hold':>6} {'TotRet':>10} {'AnnRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinR':>8} {'AvgRet':>10} {'PF':>8}")
    print("  " + "-" * 105)

    best_sharpe = -999
    best_config = None

    for th_long, th_short, label in threshold_configs:
        for hold in holding_periods:
            result = backtest_strategy(
                train_data, train_dates,
                threshold_long=th_long,
                threshold_short=th_short,
                holding_bars=hold,
                cost_bps=COST_BPS,
                label=f"{label}_h{hold}"
            )
            if result is None:
                continue
            m, _, _ = result
            print(f"  {m['label']:<15} {hold:>6} {m['total_return']:>10.4f} {m['ann_return']:>10.4f} "
                  f"{m['sharpe']:>8.2f} {m['max_drawdown']:>8.4f} {m['n_trades']:>8} "
                  f"{m['win_rate']:>8.2%} {m['avg_trade_return']:>10.6f} {m['profit_factor']:>8.2f}")

            if m['sharpe'] > best_sharpe and m['n_trades'] >= 50:
                best_sharpe = m['sharpe']
                best_config = (th_long, th_short, hold, label)

    if best_config is None:
        print("\n  No valid configuration found!")
        return

    print(f"\n  Best config (train): {best_config[3]} hold={best_config[2]}, Sharpe={best_sharpe:.2f}")

    # ===== Test on out-of-sample data =====
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE TEST")
    print("=" * 70)

    th_long, th_short, hold, label = best_config

    # Also test all configs on OOS for comparison
    print(f"\n  {'Config':<15} {'Hold':>6} {'TotRet':>10} {'AnnRet':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'WinR':>8} {'AvgRet':>10} {'PF':>8}")
    print("  " + "-" * 105)

    for th_long_t, th_short_t, label_t in threshold_configs:
        for hold_t in holding_periods:
            result = backtest_strategy(
                test_data, test_dates,
                threshold_long=th_long_t,
                threshold_short=th_short_t,
                holding_bars=hold_t,
                cost_bps=COST_BPS,
                label=f"{label_t}_h{hold_t}"
            )
            if result is None:
                continue
            m, trades_df, daily_df = result
            marker = " ← BEST_TRAIN" if (th_long_t == th_long and th_short_t == th_short and hold_t == hold) else ""
            print(f"  {m['label']:<15} {hold_t:>6} {m['total_return']:>10.4f} {m['ann_return']:>10.4f} "
                  f"{m['sharpe']:>8.2f} {m['max_drawdown']:>8.4f} {m['n_trades']:>8} "
                  f"{m['win_rate']:>8.2%} {m['avg_trade_return']:>10.6f} {m['profit_factor']:>8.2f}{marker}")

    # ===== Detailed analysis of best config on test data =====
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS: Best config on test data")
    print(f"  Config: {label}, hold={hold}")
    print("=" * 70)

    result = backtest_strategy(test_data, test_dates,
                                threshold_long=th_long, threshold_short=th_short,
                                holding_bars=hold, cost_bps=COST_BPS,
                                label=f"{label}_h{hold}")
    if result:
        m, trades_df, daily_df = result

        print(f"\n  Total return: {m['total_return']:.4f} ({m['total_return']*100:.2f}%)")
        print(f"  Annualized return: {m['ann_return']:.4f} ({m['ann_return']*100:.2f}%)")
        print(f"  Sharpe ratio: {m['sharpe']:.2f}")
        print(f"  Max drawdown: {m['max_drawdown']:.4f} ({m['max_drawdown']*100:.2f}%)")
        print(f"  Total trades: {m['n_trades']}")
        print(f"  Avg trades/day: {m['avg_trades_per_day']:.1f}")
        print(f"  Win rate: {m['win_rate']:.2%}")
        print(f"  Avg trade return: {m['avg_trade_return']*10000:.2f} bps")
        print(f"  Profit factor: {m['profit_factor']:.2f}")

        # Long vs Short breakdown
        long_trades = trades_df[trades_df['direction'] == 1]
        short_trades = trades_df[trades_df['direction'] == -1]
        print(f"\n  Long trades:  n={len(long_trades)}, "
              f"win_rate={((long_trades['return']>0).mean()):.2%}, "
              f"avg_ret={long_trades['return'].mean()*10000:.2f}bps")
        if len(short_trades) > 0:
            print(f"  Short trades: n={len(short_trades)}, "
                  f"win_rate={((short_trades['return']>0).mean()):.2%}, "
                  f"avg_ret={short_trades['return'].mean()*10000:.2f}bps")

        # Monthly breakdown
        daily_df['month'] = [d[:7] for d in daily_df['date']]
        monthly = daily_df.groupby('month').agg({
            'pnl': ['sum', 'count'],
            'n_trades': 'sum'
        })
        monthly.columns = ['return', 'days', 'trades']
        print(f"\n  Monthly returns:")
        for idx, row in monthly.iterrows():
            print(f"    {idx}: ret={row['return']*100:.3f}%, "
                  f"days={row['days']}, trades={row['trades']}")

        # Save results
        trades_df.to_csv(os.path.join(OUTPUT_DIR, 'test_trades.csv'), index=False)
        daily_df.to_csv(os.path.join(OUTPUT_DIR, 'test_daily_pnl.csv'), index=False)

    # ===== Long-only variant (simpler, for ETF trading) =====
    print("\n" + "=" * 70)
    print("LONG-ONLY VARIANT (for ETF trading)")
    print("=" * 70)

    print(f"\n  {'Config':<15} {'Hold':>6} {'TotRet':>10} {'Sharpe':>8} {'Trades':>8} {'WinR':>8} {'AvgRet':>10}")
    print("  " + "-" * 80)

    for th_pct in [0.6, 0.7, 0.8, 0.9]:
        th_val = all_resid.quantile(th_pct)
        for hold_t in [1, 3, 5]:
            result = backtest_strategy(
                test_data, test_dates,
                threshold_long=th_val,
                threshold_short=-999,  # never short
                holding_bars=hold_t,
                cost_bps=COST_BPS * 2,  # higher cost for ETF
                label=f"Long_P{int(th_pct*100)}_h{hold_t}"
            )
            if result is None:
                continue
            m, _, _ = result
            print(f"  {m['label']:<15} {hold_t:>6} {m['total_return']:>10.4f} "
                  f"{m['sharpe']:>8.2f} {m['n_trades']:>8} "
                  f"{m['win_rate']:>8.2%} {m['avg_trade_return']:>10.6f}")

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
