"""
Explore 03: Intraday Session Analysis + Composite Signal
========================================================
Key questions from Explore 02:
1. Do breadth/dispersion signals work differently at open/midday/close?
2. Can we combine breadth_residual + dispersion into a stronger composite signal?
3. Is the dispersion effect a volatility trend (persistence) or mean reversion?

Sessions defined:
- Open:  09:31 - 10:00 (first 30 bars)
- Morning: 10:01 - 11:30 (next 90 bars)
- Afternoon open: 13:01 - 13:30 (first 30 bars after lunch)
- Afternoon: 13:31 - 14:30 (60 bars)
- Close: 14:31 - 15:00 (last 30 bars)

Additional analysis:
- Dispersion autocorrelation (is high dispersion followed by high dispersion?)
- Composite signal: z-score(breadth_up) + z-score(dispersion)
- Signal decay analysis: how quickly does the signal lose its power?
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore03')
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


def get_session(dt):
    """Classify a datetime into intraday session."""
    t = dt.time()
    from datetime import time as dtime
    if t <= dtime(10, 0):
        return 'open'
    elif t <= dtime(11, 30):
        return 'morning'
    elif t <= dtime(13, 30):
        return 'pm_open'
    elif t <= dtime(14, 30):
        return 'afternoon'
    else:
        return 'close'


def rank_ic(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 20:
        return np.nan
    return stats.spearmanr(x[mask], y[mask])[0]


def process_single_day(filepath):
    """Process one day: compute all signals."""
    df = load_day(filepath)
    df = select_liquid_stocks(df, TOP_N_LIQUID)

    pivot_close = df.pivot_table(index='datetime', columns='code', values='close')
    stock_returns_1 = pivot_close.pct_change()

    # Index
    index_return_1 = stock_returns_1.mean(axis=1)
    index_level = (1 + index_return_1).cumprod()
    index_level.iloc[0] = 1.0

    n_stocks = stock_returns_1.count(axis=1)

    # Core metrics
    breadth_up = (stock_returns_1 > 0).sum(axis=1) / n_stocks
    dispersion = stock_returns_1.std(axis=1)
    index_mom = index_return_1

    # Rolling metrics (5-bar rolling)
    breadth_up_5 = breadth_up.rolling(5).mean()
    dispersion_5 = dispersion.rolling(5).mean()
    index_mom_5 = index_level.pct_change(5)

    # Z-score within day (expanding)
    breadth_z = (breadth_up - breadth_up.expanding().mean()) / (breadth_up.expanding().std() + 1e-10)
    disp_z = (dispersion - dispersion.expanding().mean()) / (dispersion.expanding().std() + 1e-10)

    # Composite signals
    composite_add = breadth_z + disp_z  # both breadth and dispersion are positive signals
    composite_breadth_only = breadth_z
    composite_disp_only = disp_z

    # Breadth residual (orthogonalized to momentum)
    # Simple per-bar approach: breadth_resid = breadth_up - beta * index_mom
    # Use expanding regression
    breadth_resid = breadth_up.copy()
    for i in range(20, len(breadth_up)):
        b = breadth_up.iloc[:i]
        m = index_mom.iloc[:i]
        mask = b.notna() & m.notna()
        if mask.sum() > 10:
            slope = np.polyfit(m[mask], b[mask], 1)
            breadth_resid.iloc[i] = breadth_up.iloc[i] - np.polyval(slope, index_mom.iloc[i])

    # Session label
    sessions = pd.Series([get_session(dt) for dt in pivot_close.index], index=pivot_close.index)

    # Forward returns
    fwd_1 = index_return_1.shift(-1)
    fwd_3 = index_level.pct_change(3).shift(-3)
    fwd_5 = index_level.pct_change(5).shift(-5)
    fwd_10 = index_level.pct_change(10).shift(-10)

    # Bar position within day (0-239)
    bar_pos = pd.Series(range(len(pivot_close.index)), index=pivot_close.index)

    result = pd.DataFrame({
        'breadth_up': breadth_up,
        'dispersion': dispersion,
        'index_mom': index_mom,
        'breadth_up_5': breadth_up_5,
        'dispersion_5': dispersion_5,
        'index_mom_5': index_mom_5,
        'breadth_z': breadth_z,
        'disp_z': disp_z,
        'composite_add': composite_add,
        'breadth_resid': breadth_resid,
        'session': sessions,
        'bar_pos': bar_pos,
        'fwd_1': fwd_1,
        'fwd_3': fwd_3,
        'fwd_5': fwd_5,
        'fwd_10': fwd_10,
    }, index=pivot_close.index)

    return result


def main():
    t0 = time.time()
    print("=" * 70)
    print("Explore 03: Session Analysis + Composite Signal")
    print("=" * 70)

    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    recent_files = all_files[-N_DAYS:]
    print(f"Using {len(recent_files)} days: {recent_files[0]} to {recent_files[-1]}")

    all_daily = []
    for i, fname in enumerate(recent_files):
        filepath = os.path.join(STOCK_DATA_DIR, fname)
        try:
            day_df = process_single_day(filepath)
            day_df['date'] = fname.replace('.parquet', '')
            all_daily.append(day_df)
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(recent_files)} days ({time.time()-t0:.1f}s)")

    print(f"Processed {len(all_daily)} days in {time.time()-t0:.1f}s")

    full_df = pd.concat(all_daily).sort_index()
    print(f"Total bars: {len(full_df)}")

    # ===== Analysis 1: IC by session =====
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Signal IC by Intraday Session")
    print("=" * 70)

    metrics = ['breadth_up', 'dispersion', 'index_mom', 'breadth_resid',
               'composite_add', 'breadth_up_5', 'dispersion_5']
    fwd_cols = ['fwd_1', 'fwd_3', 'fwd_5']
    sessions = ['open', 'morning', 'pm_open', 'afternoon', 'close']

    for fwd_col in fwd_cols:
        print(f"\n  === {fwd_col} ===")
        header = f"  {'Metric':<20}" + "".join(f"  {s:>12}" for s in sessions)
        print(header)
        print("  " + "-" * (20 + 14 * len(sessions)))

        for metric in metrics:
            row = f"  {metric:<20}"
            for session in sessions:
                mask = full_df['session'] == session
                ic = rank_ic(full_df.loc[mask, metric], full_df.loc[mask, fwd_col])
                row += f"  {ic:>12.4f}"
            print(row)

    # ===== Analysis 2: IC by bar position (decay analysis) =====
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Signal IC by bar position (decay within day)")
    print("=" * 70)

    # Group bars into 30-min blocks
    full_df['block'] = full_df['bar_pos'] // 30

    print(f"\n  IC of breadth_up vs fwd_5 by 30-min block:")
    for block in sorted(full_df['block'].unique()):
        mask = full_df['block'] == block
        ic = rank_ic(full_df.loc[mask, 'breadth_up'], full_df.loc[mask, 'fwd_5'])
        n = mask.sum()
        bar_range = f"bar {block*30}-{min((block+1)*30-1, 239)}"
        print(f"    Block {block} ({bar_range}): IC = {ic:.4f}, n = {n}")

    # ===== Analysis 3: Composite signal vs individual signals =====
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Composite vs Individual Signal IC")
    print("=" * 70)

    signals = ['breadth_up', 'dispersion', 'index_mom', 'breadth_resid',
               'composite_add', 'breadth_z', 'disp_z']

    for fwd_col in ['fwd_1', 'fwd_5', 'fwd_10']:
        print(f"\n  === {fwd_col} ===")
        for sig in signals:
            ic = rank_ic(full_df[sig], full_df[fwd_col])
            # Daily ICIR
            daily_ics = []
            for date, group in full_df.groupby('date'):
                dic = rank_ic(group[sig], group[fwd_col])
                daily_ics.append(dic)
            dic_arr = pd.Series(daily_ics).dropna()
            icir = dic_arr.mean() / dic_arr.std() if dic_arr.std() > 0 else 0
            pct_pos = (dic_arr > 0).mean()
            print(f"    {sig:<20} IC={ic:.4f}, daily_ICIR={icir:.4f}, pct+={pct_pos:.2%}")

    # ===== Analysis 4: Dispersion autocorrelation =====
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Dispersion autocorrelation (persistence)")
    print("=" * 70)

    # Within-day autocorrelation
    ac_lags = [1, 3, 5, 10, 20]
    print("  Lag  | dispersion AC | breadth_up AC")
    print("  -----+---------------+--------------")
    for lag in ac_lags:
        ac_disp = full_df['dispersion'].autocorr(lag)
        ac_breadth = full_df['breadth_up'].autocorr(lag)
        print(f"    {lag:>2}  |    {ac_disp:.4f}     |    {ac_breadth:.4f}")

    # ===== Analysis 5: Optimal composite weights via rolling OOS =====
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Optimal composite signal (breadth + dispersion)")
    print("=" * 70)

    # Try different weight combinations
    weights_to_try = [
        (1.0, 0.0, 'breadth_z only'),
        (0.0, 1.0, 'disp_z only'),
        (0.5, 0.5, '50/50'),
        (0.7, 0.3, '70/30 breadth-heavy'),
        (0.3, 0.7, '30/70 disp-heavy'),
        (1.0, 1.0, 'equal sum'),
        (1.0, -1.0, 'breadth - disp'),
    ]

    print(f"\n  {'Weights':<25} {'IC(fwd_5)':>12} {'ICIR':>8} {'pct+':>8}")
    print("  " + "-" * 55)

    best_icir = -999
    best_label = ""
    for w_b, w_d, label in weights_to_try:
        combo = w_b * full_df['breadth_z'] + w_d * full_df['disp_z']
        ic = rank_ic(combo, full_df['fwd_5'])
        daily_ics = []
        for date, group in full_df.groupby('date'):
            c = w_b * group['breadth_z'] + w_d * group['disp_z']
            dic = rank_ic(c, group['fwd_5'])
            daily_ics.append(dic)
        dic_arr = pd.Series(daily_ics).dropna()
        icir = dic_arr.mean() / dic_arr.std() if dic_arr.std() > 0 else 0
        pct_pos = (dic_arr > 0).mean()
        print(f"  {label:<25} {ic:>12.4f} {icir:>8.4f} {pct_pos:>8.2%}")
        if icir > best_icir:
            best_icir = icir
            best_label = label

    print(f"\n  Best by ICIR: {best_label} (ICIR={best_icir:.4f})")

    # ===== Analysis 6: Signal decay — how many bars is the signal useful? =====
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Signal decay — IC at different forward horizons")
    print("=" * 70)

    horizons = [1, 2, 3, 5, 7, 10, 15, 20, 30]
    signals_to_check = ['breadth_up', 'dispersion', 'composite_add']

    # We need to compute forward returns for all horizons
    print(f"\n  {'Horizon':<10}", end="")
    for sig in signals_to_check:
        print(f"  {sig:>18}", end="")
    print()
    print("  " + "-" * (10 + 20 * len(signals_to_check)))

    for day_df in all_daily:
        idx = day_df['index_level'] if 'index_level' in day_df.columns else None

    # Recompute with more forward horizons
    for h in horizons:
        row = f"  fwd_{h:<5}"
        for sig in signals_to_check:
            # Compute per-day IC then average
            day_ics = []
            for day_df in all_daily:
                if 'index_level' not in day_df.columns:
                    # Reconstruct index level from index_mom
                    il = (1 + day_df['index_mom']).cumprod()
                    il.iloc[0] = 1.0
                else:
                    continue
                fwd_h = il.pct_change(h).shift(-h)
                ic = rank_ic(day_df[sig], fwd_h)
                day_ics.append(ic)
            # Use the pre-computed forward returns for available horizons
            if f'fwd_{h}' in full_df.columns:
                ic = rank_ic(full_df[sig], full_df[f'fwd_{h}'])
            else:
                ic = np.nan
            row += f"  {ic:>18.4f}" if not np.isnan(ic) else f"  {'N/A':>18}"
        print(row)

    # Save summary
    print(f"\nResults saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time()-t0:.1f}s")

    # Save full analysis for next explore
    full_df.to_parquet(os.path.join(OUTPUT_DIR, 'full_analysis.parquet'))


if __name__ == '__main__':
    main()
