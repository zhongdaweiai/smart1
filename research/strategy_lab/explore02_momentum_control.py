"""
Explore 02: Momentum Control — Isolate Pure Co-movement Signal
==============================================================
Key question from Explore 01: Is breadth just a proxy for price momentum?
If breadth_up is high, the index is already going up, so future up is just
momentum continuation. We need to separate:

  (A) "Index is going up" (momentum)
  (B) "Stocks are moving together" (co-movement / cohesion)

Approach:
1. Compute index return over the same lookback window as the breadth signal
2. Orthogonalize breadth against index return (residual analysis)
3. Check: does breadth have alpha AFTER controlling for momentum?
4. Design a "pure cohesion" signal: high breadth GIVEN the index return level

Additional insight to test:
- When the index goes up by the same amount, does it matter HOW it went up?
  (a) All stocks +0.1% → breadth near 1.0, low dispersion
  (b) Half stocks +0.5%, half flat → breadth ~0.5, higher dispersion
  Both produce the same index return, but (a) may predict continuation better.
  This is the PURE co-movement alpha hypothesis.

Also:
- Analyze breadth vs dispersion interaction
- Check if low dispersion + positive return is a stronger signal than either alone
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore02')
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_DAYS = 200
TOP_N_LIQUID = 300


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
    """Process one day: compute index, breadth, dispersion, and momentum."""
    df = load_day(filepath)
    df = select_liquid_stocks(df, TOP_N_LIQUID)

    pivot_close = df.pivot_table(index='datetime', columns='code', values='close')
    stock_returns_1 = pivot_close.pct_change()  # 1-bar returns

    # Equal-weight index
    index_return_1 = stock_returns_1.mean(axis=1)
    index_level = (1 + index_return_1).cumprod()
    index_level.iloc[0] = 1.0

    n_stocks = stock_returns_1.count(axis=1)

    # === Core metrics (lb=1) ===
    breadth_up = (stock_returns_1 > 0).sum(axis=1) / n_stocks
    breadth_net = ((stock_returns_1 > 0).sum(axis=1) - (stock_returns_1 < 0).sum(axis=1)) / n_stocks
    dispersion = stock_returns_1.std(axis=1)

    # === Momentum signal (index return over same period) ===
    index_mom_1 = index_return_1  # 1-bar index return
    index_mom_3 = index_level.pct_change(3)
    index_mom_5 = index_level.pct_change(5)

    # === "Pure cohesion" metric: breadth unexplained by momentum ===
    # Simple approach: residual of breadth ~ momentum regression per bar
    # Better approach: for each bar, compute breadth CONDITIONAL on the index return level

    # Approach 1: Residual
    # (We'll compute this on the full dataset later)

    # Approach 2: Dispersion-adjusted momentum
    # When dispersion is low AND momentum is positive → strong cohesive up move
    # When dispersion is high AND momentum is positive → divergent up move (less reliable)
    inv_dispersion = 1.0 / (dispersion + 1e-8)
    cohesion_momentum = index_mom_1 * inv_dispersion  # momentum scaled by cohesion

    # Approach 3: Breadth - expected breadth given momentum
    # If index goes up 0.1%, we'd "expect" some breadth_up level
    # Excess breadth = actual - expected = pure consensus signal

    # === Money-weighted metrics ===
    pivot_money = df.pivot_table(index='datetime', columns='code', values='money')
    # Weight by money (turnover)
    weights = pivot_money.div(pivot_money.sum(axis=1), axis=0)
    weighted_return = (stock_returns_1 * weights).sum(axis=1)
    # Breadth weighted by money
    up_mask = (stock_returns_1 > 0).astype(float)
    weighted_breadth = (up_mask * weights).sum(axis=1)  # money-weighted fraction of up stocks
    breadth_money_diff = weighted_breadth - breadth_up  # big stocks vs small stocks bias

    # === Forward returns ===
    fwd_5 = index_level.pct_change(5).shift(-5)
    fwd_10 = index_level.pct_change(10).shift(-10)
    fwd_1 = index_return_1.shift(-1)
    fwd_3 = index_level.pct_change(3).shift(-3)

    result = pd.DataFrame({
        'breadth_up': breadth_up,
        'breadth_net': breadth_net,
        'dispersion': dispersion,
        'index_mom_1': index_mom_1,
        'index_mom_3': index_mom_3,
        'index_mom_5': index_mom_5,
        'cohesion_momentum': cohesion_momentum,
        'weighted_breadth': weighted_breadth,
        'breadth_money_diff': breadth_money_diff,
        'inv_dispersion': inv_dispersion,
        'index_level': index_level,
        'fwd_1': fwd_1,
        'fwd_3': fwd_3,
        'fwd_5': fwd_5,
        'fwd_10': fwd_10,
    }, index=pivot_close.index)

    return result


def rank_ic(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return np.nan
    return stats.spearmanr(x[mask], y[mask])[0]


def partial_rank_ic(factor, fwd_ret, control):
    """Partial rank correlation: IC of factor with fwd_ret, controlling for control variable."""
    mask = factor.notna() & fwd_ret.notna() & control.notna()
    if mask.sum() < 30:
        return np.nan
    f = factor[mask].rank()
    r = fwd_ret[mask].rank()
    c = control[mask].rank()

    # Residualize both f and r on c
    from numpy.polynomial.polynomial import polyfit
    # f_resid = f - proj(f, c)
    slope_f = np.polyfit(c, f, 1)
    f_resid = f - np.polyval(slope_f, c)
    slope_r = np.polyfit(c, r, 1)
    r_resid = r - np.polyval(slope_r, c)

    return stats.spearmanr(f_resid, r_resid)[0]


def conditional_analysis(df, factor_col, fwd_col, condition_col, n_cond_bins=3, n_factor_bins=5):
    """Analyze factor's predictive power within each condition bin."""
    mask = df[factor_col].notna() & df[fwd_col].notna() & df[condition_col].notna()
    subset = df[mask].copy()

    try:
        subset['cond_bin'] = pd.qcut(subset[condition_col], n_cond_bins, labels=['low', 'mid', 'high'],
                                      duplicates='drop')
    except ValueError:
        return None

    results = {}
    for cond_val, group in subset.groupby('cond_bin'):
        try:
            group['factor_bin'] = pd.qcut(group[factor_col], n_factor_bins, labels=False, duplicates='drop')
        except ValueError:
            continue
        binned = group.groupby('factor_bin')[fwd_col].mean()
        ic = rank_ic(group[factor_col], group[fwd_col])
        results[cond_val] = {'binned_returns': binned, 'ic': ic, 'count': len(group)}

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("Explore 02: Momentum Control — Isolate Pure Co-movement Signal")
    print("=" * 70)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    recent_files = all_files[-N_DAYS:]
    print(f"Using {len(recent_files)} days: {recent_files[0]} to {recent_files[-1]}")

    all_daily = []
    for i, fname in enumerate(recent_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_single_day(filepath)
            all_daily.append(day_df)
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(recent_files)} days ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    full_df = pd.concat(all_daily).sort_index()
    print(f"Total bars: {len(full_df)}")

    # ===== Test 1: Raw IC vs Partial IC (controlling for momentum) =====
    print("\n" + "=" * 70)
    print("TEST 1: Raw IC vs Partial IC (controlling for index momentum)")
    print("=" * 70)

    factors = ['breadth_up', 'breadth_net', 'dispersion', 'cohesion_momentum',
               'weighted_breadth', 'breadth_money_diff', 'inv_dispersion']
    fwd_cols = ['fwd_1', 'fwd_3', 'fwd_5', 'fwd_10']

    print(f"\n{'Metric':<25} {'raw_IC(fwd5)':>14} {'partial_IC(fwd5)':>16} {'raw_IC(fwd1)':>14} {'partial_IC(fwd1)':>16}")
    print("-" * 90)

    for factor in factors:
        raw_5 = rank_ic(full_df[factor], full_df['fwd_5'])
        partial_5 = partial_rank_ic(full_df[factor], full_df['fwd_5'], full_df['index_mom_1'])
        raw_1 = rank_ic(full_df[factor], full_df['fwd_1'])
        partial_1 = partial_rank_ic(full_df[factor], full_df['fwd_1'], full_df['index_mom_1'])
        print(f"  {factor:<23} {raw_5:>14.4f} {partial_5:>16.4f} {raw_1:>14.4f} {partial_1:>16.4f}")

    # ===== Test 2: Conditional analysis — breadth within momentum buckets =====
    print("\n" + "=" * 70)
    print("TEST 2: Breadth predictive power WITHIN momentum buckets")
    print("=" * 70)

    cond_result = conditional_analysis(full_df, 'breadth_up', 'fwd_5', 'index_mom_1',
                                        n_cond_bins=3, n_factor_bins=5)
    if cond_result:
        for cond, info in cond_result.items():
            print(f"\n  Momentum = {cond} (n={info['count']}, IC={info['ic']:.4f}):")
            print(f"    Breadth quintile returns: {info['binned_returns'].to_dict()}")

    # ===== Test 3: Dispersion conditional analysis =====
    print("\n" + "=" * 70)
    print("TEST 3: Momentum predictive power WITHIN dispersion buckets")
    print("(Does low dispersion = high co-movement make momentum more reliable?)")
    print("=" * 70)

    cond_result2 = conditional_analysis(full_df, 'index_mom_1', 'fwd_5', 'dispersion',
                                         n_cond_bins=3, n_factor_bins=5)
    if cond_result2:
        for cond, info in cond_result2.items():
            print(f"\n  Dispersion = {cond} (n={info['count']}, IC={info['ic']:.4f}):")
            print(f"    Momentum quintile returns: {info['binned_returns'].to_dict()}")

    # ===== Test 4: Interaction signal — cohesion_momentum =====
    print("\n" + "=" * 70)
    print("TEST 4: Combined signal analysis")
    print("=" * 70)

    combined_factors = ['index_mom_1', 'breadth_up', 'cohesion_momentum',
                        'weighted_breadth', 'breadth_money_diff', 'inv_dispersion']

    for fwd_col in ['fwd_1', 'fwd_5', 'fwd_10']:
        print(f"\n  Forward: {fwd_col}")
        for factor in combined_factors:
            ic = rank_ic(full_df[factor], full_df[fwd_col])
            print(f"    {factor:<25} IC = {ic:.4f}")

    # ===== Test 5: Double sort — momentum x dispersion =====
    print("\n" + "=" * 70)
    print("TEST 5: Double sort — Momentum × Dispersion → fwd_5 return")
    print("=" * 70)

    mask = full_df['index_mom_1'].notna() & full_df['fwd_5'].notna() & full_df['dispersion'].notna()
    subset = full_df[mask].copy()

    try:
        subset['mom_q'] = pd.qcut(subset['index_mom_1'], 5, labels=['very_neg', 'neg', 'flat', 'pos', 'very_pos'],
                                   duplicates='drop')
        subset['disp_q'] = pd.qcut(subset['dispersion'], 3, labels=['low_disp', 'mid_disp', 'high_disp'],
                                    duplicates='drop')

        double_sort = subset.groupby(['mom_q', 'disp_q'])['fwd_5'].agg(['mean', 'count'])
        print("\nMean fwd_5 return by (Momentum quintile × Dispersion tercile):")
        pivot_table = double_sort['mean'].unstack('disp_q')
        print(pivot_table.to_string(float_format='{:.6f}'.format))
        print("\nCounts:")
        count_table = double_sort['count'].unstack('disp_q')
        print(count_table.to_string())

        # Key question: within the same momentum level, does lower dispersion predict better?
        print("\n  Within each momentum quintile, low_disp - high_disp return spread:")
        if 'low_disp' in pivot_table.columns and 'high_disp' in pivot_table.columns:
            spread = pivot_table['low_disp'] - pivot_table['high_disp']
            for idx in spread.index:
                print(f"    {idx}: {spread[idx]:.6f}")
    except Exception as e:
        print(f"  Double sort failed: {e}")

    # ===== Test 6: Daily partial IC stability =====
    print("\n" + "=" * 70)
    print("TEST 6: Daily Partial IC stability (breadth controlling for momentum)")
    print("=" * 70)

    full_df['date'] = full_df.index.date
    daily_partial_ics = []
    daily_raw_ics = []
    for date, group in full_df.groupby('date'):
        raw = rank_ic(group['breadth_up'], group['fwd_5'])
        partial = partial_rank_ic(group['breadth_up'], group['fwd_5'], group['index_mom_1'])
        daily_raw_ics.append(raw)
        daily_partial_ics.append(partial)

    raw_arr = pd.Series(daily_raw_ics).dropna()
    partial_arr = pd.Series(daily_partial_ics).dropna()

    print(f"  Raw breadth_up IC:     mean={raw_arr.mean():.4f}, ICIR={raw_arr.mean()/raw_arr.std():.4f}, pct+={((raw_arr>0).mean()):.2%}")
    print(f"  Partial breadth_up IC: mean={partial_arr.mean():.4f}, ICIR={partial_arr.mean()/partial_arr.std():.4f}, pct+={((partial_arr>0).mean()):.2%}")

    # Save results
    results_summary = {
        'raw_ic_mean': raw_arr.mean(),
        'raw_ic_icir': raw_arr.mean() / raw_arr.std(),
        'partial_ic_mean': partial_arr.mean(),
        'partial_ic_icir': partial_arr.mean() / partial_arr.std(),
    }
    pd.Series(results_summary).to_csv(os.path.join(OUTPUT_DIR, 'partial_ic_summary.csv'))

    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
