"""
Explore 01: Co-movement Metrics Foundation
==========================================
Goal: Define basic co-movement (同涨同跌) metrics across all stocks,
      construct a synthetic "market index" from data, and test if
      co-movement metrics have predictive power for future index returns.

Metrics to compute (per minute bar):
1. breadth_up: fraction of stocks with positive return over trailing N bars
2. breadth_down: fraction of stocks with negative return over trailing N bars
3. breadth_net: breadth_up - breadth_down (net breadth, range [-1, 1])
4. dispersion: cross-sectional std of stock returns (low = high co-movement)
5. breadth_change: delta of breadth_net vs previous bar (momentum of consensus)

Target: forward index return over next K bars

Analysis:
- Rank IC (Spearman correlation) of each metric vs forward return
- Binned analysis: sort bars by metric quintile, compute avg forward return
- Multiple lookback windows (N) and forward horizons (K)

Data: Use most recent 100 trading days from stock_data/ for speed.
      Construct equal-weight "index" from top ~300 liquid stocks per day.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore01')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameters
N_DAYS = 200          # number of recent trading days to use
LOOKBACKS = [1, 3, 5, 10, 20]   # lookback windows for computing stock returns (in bars)
FORWARDS = [1, 3, 5, 10, 20]    # forward horizons for index return prediction (in bars)
TOP_N_LIQUID = 300    # top N stocks by daily turnover to approximate index


def load_day(filepath):
    """Load one day's minute data."""
    df = pd.read_parquet(filepath)
    df = df[df['paused'] == 0].copy()  # exclude suspended stocks
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def select_liquid_stocks(df, top_n=300):
    """Select top N stocks by daily money (turnover)."""
    daily_money = df.groupby('code')['money'].sum()
    top_codes = daily_money.nlargest(top_n).index
    return df[df['code'].isin(top_codes)].copy()


def compute_index(df):
    """Compute equal-weight index from selected stocks.
    Returns a Series indexed by datetime with the index level.
    """
    # Compute per-stock return from first bar of the day
    pivot = df.pivot_table(index='datetime', columns='code', values='close')
    # Equal-weight return: average of individual stock returns
    stock_returns = pivot.pct_change()
    index_return = stock_returns.mean(axis=1)
    index_level = (1 + index_return).cumprod()
    index_level.iloc[0] = 1.0
    return index_level, stock_returns


def compute_comovement_metrics(stock_returns, lookback):
    """Compute co-movement metrics for a given lookback window.

    stock_returns: DataFrame (datetime x code) of per-bar returns
    lookback: number of bars to look back for computing rolling returns

    Returns DataFrame with columns: breadth_up, breadth_down, breadth_net, dispersion, breadth_change
    """
    if lookback == 1:
        rolling_ret = stock_returns
    else:
        # Rolling cumulative return over lookback bars
        rolling_ret = stock_returns.rolling(lookback).sum()  # approximate with sum for small returns

    n_stocks = rolling_ret.count(axis=1)

    breadth_up = (rolling_ret > 0).sum(axis=1) / n_stocks
    breadth_down = (rolling_ret < 0).sum(axis=1) / n_stocks
    breadth_net = breadth_up - breadth_down
    dispersion = rolling_ret.std(axis=1)
    breadth_change = breadth_net.diff()

    # Additional: signed dispersion (dispersion * sign of mean return)
    mean_ret = rolling_ret.mean(axis=1)
    signed_cohesion = mean_ret / (dispersion + 1e-10)  # like a cross-sectional t-stat

    result = pd.DataFrame({
        'breadth_up': breadth_up,
        'breadth_down': breadth_down,
        'breadth_net': breadth_net,
        'dispersion': dispersion,
        'breadth_change': breadth_change,
        'signed_cohesion': signed_cohesion,
    })
    return result


def compute_forward_returns(index_level, forward):
    """Compute forward return of the index over next `forward` bars."""
    return index_level.pct_change(forward).shift(-forward)


def rank_ic(factor, forward_ret):
    """Compute rank IC (Spearman correlation)."""
    mask = factor.notna() & forward_ret.notna()
    if mask.sum() < 30:
        return np.nan
    return stats.spearmanr(factor[mask], forward_ret[mask])[0]


def binned_analysis(factor, forward_ret, n_bins=5):
    """Bin factor into quintiles, compute mean forward return per bin."""
    mask = factor.notna() & forward_ret.notna()
    f = factor[mask]
    r = forward_ret[mask]
    try:
        bins = pd.qcut(f, n_bins, labels=False, duplicates='drop')
    except ValueError:
        return None
    result = r.groupby(bins).agg(['mean', 'std', 'count'])
    result.index.name = 'quintile'
    return result


def process_single_day(filepath, lookbacks):
    """Process one day: compute index + co-movement metrics."""
    df = load_day(filepath)
    df = select_liquid_stocks(df, TOP_N_LIQUID)

    index_level, stock_returns = compute_index(df)

    all_metrics = {}
    for lb in lookbacks:
        metrics = compute_comovement_metrics(stock_returns, lb)
        for col in metrics.columns:
            all_metrics[f'{col}_lb{lb}'] = metrics[col]

    metrics_df = pd.DataFrame(all_metrics, index=index_level.index)
    metrics_df['index_level'] = index_level

    return metrics_df


def main():
    t0 = time.time()
    print("=" * 70)
    print("Explore 01: Co-movement Metrics Foundation")
    print("=" * 70)

    # Get recent N_DAYS files
    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    recent_files = all_files[-N_DAYS:]
    print(f"Using {len(recent_files)} trading days: {recent_files[0]} to {recent_files[-1]}")

    # Process each day
    all_daily = []
    for i, fname in enumerate(recent_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_metrics = process_single_day(filepath, LOOKBACKS)
            all_daily.append(day_metrics)
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(recent_files)} days ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    # Concatenate all days
    full_df = pd.concat(all_daily)
    full_df = full_df.sort_index()
    print(f"Total bars: {len(full_df)}")

    # Compute forward returns for the index
    # Note: forward returns should NOT cross day boundaries for intraday strategy
    # So we compute within each day
    forward_returns = {}
    for fwd in FORWARDS:
        fwd_ret = []
        for day_df in all_daily:
            idx = day_df['index_level']
            fr = compute_forward_returns(idx, fwd)
            fwd_ret.append(fr)
        forward_returns[f'fwd_{fwd}'] = pd.concat(fwd_ret)

    fwd_df = pd.DataFrame(forward_returns)

    # Merge
    analysis_df = full_df.join(fwd_df)

    # ===== Rank IC Analysis =====
    print("\n" + "=" * 70)
    print("RANK IC ANALYSIS (Spearman correlation with forward returns)")
    print("=" * 70)

    metric_cols = [c for c in full_df.columns if c != 'index_level']
    fwd_cols = [f'fwd_{f}' for f in FORWARDS]

    ic_results = []
    for metric in metric_cols:
        row = {'metric': metric}
        for fwd_col in fwd_cols:
            ic = rank_ic(analysis_df[metric], analysis_df[fwd_col])
            row[fwd_col] = ic
        ic_results.append(row)

    ic_df = pd.DataFrame(ic_results).set_index('metric')

    # Sort by absolute IC for fwd_5
    ic_df['abs_ic_fwd5'] = ic_df['fwd_5'].abs()
    ic_df = ic_df.sort_values('abs_ic_fwd5', ascending=False)

    print("\nTop 20 metrics by |IC| with fwd_5:")
    print(ic_df.head(20).to_string(float_format='{:.4f}'.format))

    # Save IC results
    ic_df.to_csv(os.path.join(OUTPUT_DIR, 'rank_ic_results.csv'))

    # ===== Binned Analysis for Top Metrics =====
    print("\n" + "=" * 70)
    print("BINNED ANALYSIS (quintile returns) for top metrics vs fwd_5")
    print("=" * 70)

    top_metrics = ic_df.head(10).index.tolist()
    for metric in top_metrics:
        print(f"\n--- {metric} ---")
        result = binned_analysis(analysis_df[metric], analysis_df['fwd_5'])
        if result is not None:
            print(result.to_string(float_format='{:.6f}'.format))

    # ===== Daily IC Time Series =====
    print("\n" + "=" * 70)
    print("DAILY IC STABILITY (IC per day for top metrics vs fwd_5)")
    print("=" * 70)

    analysis_df['date'] = analysis_df.index.date
    daily_ic = {}
    for metric in top_metrics[:5]:
        daily_ics = []
        for date, group in analysis_df.groupby('date'):
            ic = rank_ic(group[metric], group['fwd_5'])
            daily_ics.append({'date': date, 'ic': ic})
        dic = pd.DataFrame(daily_ics).set_index('date')['ic']
        daily_ic[metric] = dic
        mean_ic = dic.mean()
        std_ic = dic.std()
        icir = mean_ic / std_ic if std_ic > 0 else 0
        pct_positive = (dic > 0).mean()
        print(f"  {metric}: mean_IC={mean_ic:.4f}, std={std_ic:.4f}, ICIR={icir:.4f}, pct_positive={pct_positive:.2%}")

    daily_ic_df = pd.DataFrame(daily_ic)
    daily_ic_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_ic_timeseries.csv'))

    # ===== Summary Statistics =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Date range: {recent_files[0]} to {recent_files[-1]}")
    print(f"Total days: {len(all_daily)}")
    print(f"Total bars: {len(full_df)}")
    print(f"Stocks per day: ~{TOP_N_LIQUID}")
    print(f"Lookback windows: {LOOKBACKS}")
    print(f"Forward horizons: {FORWARDS}")

    print(f"\nBest metric by |IC(fwd_5)|: {ic_df.index[0]} = {ic_df.iloc[0]['fwd_5']:.4f}")
    print(f"Best metric by |IC(fwd_10)|: ", end="")
    ic_df_10 = ic_df.sort_values(ic_df.columns[ic_df.columns.str.contains('fwd_10')][0],
                                  key=abs, ascending=False)
    print(f"{ic_df_10.index[0]} = {ic_df_10.iloc[0]['fwd_10']:.4f}")

    # Save the full analysis dataframe (sampled for size)
    sample_df = analysis_df.drop(columns=['date'], errors='ignore').iloc[::5]  # every 5th bar
    sample_df.to_parquet(os.path.join(OUTPUT_DIR, 'analysis_sample.parquet'))

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
