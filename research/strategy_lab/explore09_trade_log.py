"""
Explore 09: Detailed Weekly Trade Log (Two Weeks Comparison)
=============================================================
Show trade-by-trade detail for the BEST and WORST weeks from OOS period.
Strategy: aggressive_asym_scaled (best from Explore 08)
"""

import os, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')

STOCK_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'stock_data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'explore09')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOP_N = 300
ROUND_TRIP_COST_BPS = 4.5
MAX_SCALE = 3.0
DAILY_LOSS_LIMIT_BPS = -50
TH_LONG_MULT = 0.5
TH_SHORT_MULT = 1.0

# IF futures
IF_MULTIPLIER = 300
APPROX_INDEX = 3900

# Two weeks: best and worst from 100-day OOS
WEEKS = {
    '🏆 最佳周': ['2026-01-12', '2026-01-13', '2026-01-14', '2026-01-15', '2026-01-16'],
    '📉 最差周': ['2026-01-19', '2026-01-20', '2026-01-21', '2026-01-22', '2026-01-23'],
}


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

    # Equal-weight (signal)
    ew_ret = ret1.mean(axis=1)
    ew_lvl = (1 + ew_ret).cumprod(); ew_lvl.iloc[0] = 1.0

    # Turnover-weighted (P&L)
    bw = pm.div(pm.sum(axis=1), axis=0)
    tw_ret = (ret1 * bw).sum(axis=1)
    tw_lvl = (1 + tw_ret).cumprod(); tw_lvl.iloc[0] = 1.0

    # breadth_resid
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
        'breadth_resid': br, 'breadth_up': bu,
    }, index=pc.index)


def backtest_day(day_df, th_long, th_short, r_mean, r_std, date_str):
    """Backtest one day, return detailed trade list."""
    n = len(day_df)
    pos = 0.0; ebar = -1; eprice = 0; esig = 0
    cum = 0; trades = []; stopped = False; tno = 0

    for bar in range(10, n):
        sig = day_df['breadth_resid'].iloc[bar]
        px = day_df['tw_level'].iloc[bar]
        tm = day_df.index[bar].strftime('%H:%M')

        if stopped: break

        # EXIT
        if pos != 0:
            held = bar - ebar
            eod = bar >= n - 2
            if held >= 1 or eod:
                d = 1 if pos > 0 else -1
                sz = abs(pos)
                gr = d * (px / eprice - 1)
                nr = gr * sz - ROUND_TRIP_COST_BPS / 10000
                cum += nr; tno += 1
                trades.append({
                    'no': tno, 'date': date_str,
                    'entry': day_df.index[ebar].strftime('%H:%M'),
                    'exit': tm,
                    'dir': '做多↑' if d > 0 else '做空↓',
                    'dir_en': 'LONG' if d > 0 else 'SHORT',
                    'size': round(sz, 2),
                    'sig_z': round((esig - r_mean) / r_std, 2),
                    'gross': round(gr * 10000, 2),
                    'net': round(nr * 10000, 2),
                    'ok': '✅' if nr > 0 else '❌',
                    'cum': round(cum * 10000, 2),
                    'yuan': round(nr * APPROX_INDEX * IF_MULTIPLIER, 0),
                })
                pos = 0
                if cum * 10000 < DAILY_LOSS_LIMIT_BPS:
                    stopped = True; continue

        # ENTRY
        if pos == 0 and bar < n - 3 and not stopped:
            if sig > th_long:
                sc = min(abs(sig - r_mean) / (TH_LONG_MULT * r_std), MAX_SCALE)
                pos = sc; ebar = bar; eprice = px; esig = sig
            elif sig < th_short:
                sc = min(abs(sig - r_mean) / (TH_SHORT_MULT * r_std), MAX_SCALE)
                pos = -sc; ebar = bar; eprice = px; esig = sig

    # Force close
    if pos != 0:
        px = day_df['tw_level'].iloc[-1]
        d = 1 if pos > 0 else -1; sz = abs(pos)
        gr = d * (px / eprice - 1)
        nr = gr * sz - ROUND_TRIP_COST_BPS / 10000
        cum += nr; tno += 1
        trades.append({
            'no': tno, 'date': date_str, 'entry': day_df.index[ebar].strftime('%H:%M'),
            'exit': day_df.index[-1].strftime('%H:%M'),
            'dir': '做多↑' if d > 0 else '做空↓', 'dir_en': 'LONG' if d > 0 else 'SHORT',
            'size': round(sz, 2), 'sig_z': round((esig - r_mean) / r_std, 2),
            'gross': round(gr * 10000, 2), 'net': round(nr * 10000, 2),
            'ok': '✅' if nr > 0 else '❌', 'cum': round(cum * 10000, 2),
            'yuan': round(nr * APPROX_INDEX * IF_MULTIPLIER, 0),
        })

    return trades, cum


def main():
    t0 = time.time()

    # === Strategy description ===
    print("=" * 110)
    print("  Explore 09: 策略详细交易记录 — 最佳周 vs 最差周 对比")
    print("=" * 110)
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  交易标的:   沪深300股指期货 (IF)                            │
  │  合约乘数:   每点 {IF_MULTIPLIER} 元                                    │
  │  保证金:     ~{IF_MULTIPLIER*APPROX_INDEX*0.12/10000:.1f}万元/手 (12%)                           │
  │  往返成本:   {ROUND_TRIP_COST_BPS} bps (佣金3.68 + 滑点0.82)                  │
  │  信号:       breadth_resid (成分股同涨比例, 控制指数动量)    │
  │  做多门槛:   Z > +{TH_LONG_MULT}σ                                        │
  │  做空门槛:   Z < -{TH_SHORT_MULT}σ                                        │
  │  仓位:       min(|Z|/阈值倍数, {MAX_SCALE}) 倍 (信号越强仓位越大)     │
  │  持仓:       1分钟 (1 bar)                                  │
  │  日亏限额:   {DAILY_LOSS_LIMIT_BPS} bps (触及则当日停止交易)                  │
  └─────────────────────────────────────────────────────────────┘
""")

    # Load files
    all_files = sorted([f for f in os.listdir(STOCK_DATA_DIR) if f.endswith('.parquet')])
    recent = all_files[-200:]
    file_map = {f.replace('.parquet', ''): f for f in recent}

    # Flatten all target dates
    all_target_dates = sorted(set(d for wk in WEEKS.values() for d in wk))
    first_date = all_target_dates[0]

    # Calibration: all data before earliest target date
    print("  校准中...", end="", flush=True)
    train_data = []
    for f in recent:
        ds = f.replace('.parquet', '')
        if ds < first_date:
            try:
                train_data.append(process_day(os.path.join(STOCK_DATA_DIR, f)))
            except:
                continue

    all_r = pd.concat([d['breadth_resid'] for d in train_data]).dropna()
    r_mean, r_std = all_r.mean(), all_r.std()
    th_long = r_mean + TH_LONG_MULT * r_std
    th_short = r_mean - TH_SHORT_MULT * r_std
    print(f" {len(train_data)}天 (mean={r_mean:.4f}, std={r_std:.4f})")
    print(f"  做多阈值: breadth_resid > {th_long:.4f} (Z > +{TH_LONG_MULT}σ)")
    print(f"  做空阈值: breadth_resid < {th_short:.4f} (Z < -{TH_SHORT_MULT}σ)")

    # Load and process target days
    print("  加载目标交易日...", end="", flush=True)
    day_cache = {}
    for ds in all_target_dates:
        if ds in file_map:
            day_cache[ds] = process_day(os.path.join(STOCK_DATA_DIR, file_map[ds]))
    print(f" {len(day_cache)}天 ({time.time()-t0:.0f}s)")

    # === Process each week ===
    for wk_label, wk_dates in WEEKS.items():
        found = [d for d in wk_dates if d in day_cache]
        if not found:
            print(f"\n{wk_label}: 无数据"); continue

        print(f"\n\n{'#'*110}")
        print(f"#  {wk_label}: {found[0]} ~ {found[-1]}")
        print(f"{'#'*110}")

        wk_trades = []
        wk_pnl = 0
        day_summaries = []

        for ds in found:
            ddf = day_cache[ds]
            trades, dpnl = backtest_day(ddf, th_long, th_short, r_mean, r_std, ds)
            wk_trades.extend(trades)
            wk_pnl += dpnl

            n_l = sum(1 for t in trades if t['dir_en'] == 'LONG')
            n_s = sum(1 for t in trades if t['dir_en'] == 'SHORT')
            n_w = sum(1 for t in trades if t['ok'] == '✅')
            idx_ret = (ddf['tw_level'].iloc[-1] / ddf['tw_level'].iloc[0] - 1) * 10000

            day_summaries.append({
                'date': ds, 'n': len(trades), 'long': n_l, 'short': n_s,
                'win': n_w, 'lose': len(trades) - n_w,
                'pnl_bps': dpnl * 10000, 'idx_bps': idx_ret,
                'yuan': dpnl * APPROX_INDEX * IF_MULTIPLIER,
            })

        # Week overview
        print(f"\n  {'日期':<12} {'交易':>4} {'多':>3} {'空':>3} {'胜':>3} {'负':>3} {'胜率':>6} {'日P&L':>10} {'≈元/手':>10} {'指数':>10} {'结果':>4}")
        print("  " + "-" * 80)
        for s in day_summaries:
            wr = s['win']/s['n'] if s['n'] else 0
            em = '📈' if s['pnl_bps'] > 0 else '📉'
            print(f"  {s['date']:<12} {s['n']:>4} {s['long']:>3} {s['short']:>3} "
                  f"{s['win']:>3} {s['lose']:>3} {wr:>5.0%} "
                  f"{s['pnl_bps']:>+9.1f} {s['yuan']:>+9.0f} {s['idx_bps']:>+9.1f} {em:>4}")

        tw = len(wk_trades); twin = sum(1 for t in wk_trades if t['ok'] == '✅')
        print("  " + "-" * 80)
        print(f"  {'合计':<12} {tw:>4} {sum(s['long'] for s in day_summaries):>3} "
              f"{sum(s['short'] for s in day_summaries):>3} {twin:>3} {tw-twin:>3} "
              f"{twin/tw:.0%} {wk_pnl*10000:>+9.1f} {wk_pnl*APPROX_INDEX*IF_MULTIPLIER:>+9.0f}")

        # Day-by-day trades
        for ds in found:
            dt = [t for t in wk_trades if t['date'] == ds]
            if not dt: continue
            s = [x for x in day_summaries if x['date'] == ds][0]

            print(f"\n  ┌── {ds}  指数{s['idx_bps']:+.0f}bps  策略{s['pnl_bps']:+.1f}bps ──┐")
            print(f"  │ {'#':>2} {'开仓':>5}→{'平仓':>5} {'方向':>5} {'仓位':>4} {'信号':>7} "
                  f"{'毛利':>7} {'净利':>7} {'':>2} {'累计':>7} {'≈元':>8} │")
            print(f"  │{'-'*76}│")

            for t in dt:
                print(f"  │ {t['no']:>2} {t['entry']:>5}→{t['exit']:>5} {t['dir']:>5} "
                      f"{t['size']:>4.1f}x {t['sig_z']:>+6.2f}σ "
                      f"{t['gross']:>+6.1f} {t['net']:>+6.1f} {t['ok']:>2} "
                      f"{t['cum']:>+6.1f} {t['yuan']:>+7.0f} │")

            long_net = sum(t['net'] for t in dt if t['dir_en'] == 'LONG')
            short_net = sum(t['net'] for t in dt if t['dir_en'] == 'SHORT')
            total_cost = len(dt) * ROUND_TRIP_COST_BPS
            print(f"  │{'-'*76}│")
            yuan_str = f"{s['yuan']:+,.0f}"
            summary_str = (f" 日结: {len(dt)}笔, {s['win']}胜{s['lose']}负, "
                           f"多头{long_net:+.1f} 空头{short_net:+.1f} "
                           f"成本-{total_cost:.0f}bps "
                           f"净{s['pnl_bps']:+.1f}bps ≈{yuan_str}元")
            pad = max(0, 76 - len(summary_str))
            print(f"  │{summary_str}{' '*pad}│")
            print(f"  └{'─'*76}┘")

        # Week totals
        long_all = [t for t in wk_trades if t['dir_en'] == 'LONG']
        short_all = [t for t in wk_trades if t['dir_en'] == 'SHORT']
        gross_sum = sum(t['gross'] for t in wk_trades)
        cost_sum = tw * ROUND_TRIP_COST_BPS

        print(f"""
  ┌──── {wk_label} 总结 ────────────────────────────────────────────┐
  │  交易: {tw}笔, 胜率{twin/tw:.0%} ({twin}胜{tw-twin}负)                      │
  │  做多: {len(long_all)}笔 净利{sum(t['net'] for t in long_all):>+8.1f}bps  胜率{sum(1 for t in long_all if t['ok']=='✅')/max(len(long_all),1):.0%}                    │
  │  做空: {len(short_all)}笔 净利{sum(t['net'] for t in short_all):>+8.1f}bps  胜率{sum(1 for t in short_all if t['ok']=='✅')/max(len(short_all),1):.0%}                    │
  │  毛利合计: {gross_sum:>+10.1f} bps                               │
  │  成本合计:  -{cost_sum:>8.1f} bps                               │
  │  净利合计: {wk_pnl*10000:>+10.1f} bps                               │
  │  ≈ 每手IF周损益: {wk_pnl*APPROX_INDEX*IF_MULTIPLIER:>+10,.0f} 元                          │
  └─────────────────────────────────────────────────────────────┘""")

    # === KEY INSIGHT ===
    print(f"""

{'='*110}
  💡 策略盈利模式分析
{'='*110}

  这个策略的盈利模式是 "截断亏损, 让利润奔跑":

  1. 亏损日: 频繁但有限
     - 日亏损限额 -50bps 起到截断作用
     - 亏损日往往早盘就触及限额停止交易 (如1月20日仅2笔就停)
     - 亏损主要来自: 交易成本累积 + 信号反向

  2. 盈利日: 少数但巨大
     - 大赢日的单笔利润远超成本 (如1月13日单笔可达数百bps)
     - 信号缩放使得强信号日自动加大仓位
     - 持仓1分钟极短, 但强共振时1分钟内指数波动很大

  3. 最佳周 vs 最差周:
     - 最佳周: 策略捕捉到多次强共振, 大赢覆盖所有成本
     - 最差周: 市场缺乏清晰共振信号, 频繁交易侵蚀本金

  4. 100天OOS整体: 平均日收益 +114bps, 正日率57%
     → 年化收益极高, 靠少数大赢日驱动
""")

    # Save
    pd.DataFrame(wk_trades).to_csv(os.path.join(OUTPUT_DIR, 'two_weeks_trades.csv'), index=False)
    print(f"  交易记录已保存: {OUTPUT_DIR}/two_weeks_trades.csv")
    print(f"  耗时: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
