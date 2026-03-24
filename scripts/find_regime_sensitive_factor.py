"""
构建多个经典因子，找出条件性差异最大的那个。

候选因子:
  1. 动量_20日 (Momentum_20d)  — 过去20日收益率
  2. 短期反转_5日 (Reversal_5d) — 过去5日收益率取负
  3. 波动率_20日 (Vol_20d)      — 过去20日日收益率标准差
  4. 换手率_20日 (Turnover_20d) — 过去20日平均换手率
  5. 成交额变化 (MoneyChg)     — 近5日 vs 前15日 成交额比值
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = '/Users/daweizhong/Documents/projects/stock_data'
ARTIFACTS = '/Users/daweizhong/Documents/projects/artifacts'

# 获取已有因子面板的月频日期 (和 RV_Skew 对齐)
existing_fp = pd.read_parquet(os.path.join(ARTIFACTS, 'factor_panel.parquet'))
monthly_dates = sorted(existing_fp.index)
print(f"月频日期: {len(monthly_dates)} 期, {monthly_dates[0].date()} ~ {monthly_dates[-1].date()}")

# 获取所有交易日
all_files = sorted(f.replace('.parquet', '') for f in os.listdir(DATA_DIR) if f.endswith('.parquet'))
all_trade_dates = all_files

def get_n_days_before(date_str, n=20):
    """获取某日期之前 n 个交易日的日期列表"""
    idx = all_trade_dates.index(date_str)
    start = max(0, idx - n)
    return all_trade_dates[start:idx+1]

def read_daily_close_and_money(date_str):
    """从分钟线读取日级别 close 和 money"""
    fp = os.path.join(DATA_DIR, f"{date_str}.parquet")
    df = pd.read_parquet(fp, columns=['code', 'close', 'money', 'volume', 'paused'])
    df = df[df['paused'] == 0]
    agg = df.groupby('code').agg(
        close=('close', 'last'),
        money=('money', 'sum'),
        volume=('volume', 'sum'),
    )
    return agg[agg['close'] > 0]


# ======================================================
# 构建因子
# ======================================================
print("\n逐月构建因子...")

factor_panels = {
    'Momentum_20d': {},
    'Reversal_5d': {},
    'Vol_20d': {},
    'Turnover_20d': {},
    'MoneyChg_5v15': {},
}

for i_m, dt in enumerate(monthly_dates):
    date_str = dt.strftime('%Y-%m-%d')
    print(f"  {i_m+1}/{len(monthly_dates)}: {date_str}")

    # 取过去25个交易日的数据
    past_dates = get_n_days_before(date_str, n=25)
    if len(past_dates) < 21:
        print(f"    跳过: 历史数据不足 ({len(past_dates)} 天)")
        continue

    # 读取这些天的日级别数据
    daily_data = {}
    for d in past_dates:
        try:
            daily_data[d] = read_daily_close_and_money(d)
        except Exception as e:
            pass

    if len(daily_data) < 21:
        print(f"    跳过: 有效数据不足 ({len(daily_data)} 天)")
        continue

    sorted_days = sorted(daily_data.keys())

    # 构建 close 面板 和 money 面板
    close_panel = pd.DataFrame({d: daily_data[d]['close'] for d in sorted_days})
    money_panel = pd.DataFrame({d: daily_data[d]['money'] for d in sorted_days})
    volume_panel = pd.DataFrame({d: daily_data[d]['volume'] for d in sorted_days})

    # 日收益率面板
    ret_panel = close_panel.pct_change(axis=1)

    # --- 因子1: 动量_20d ---
    # 过去20日累计收益
    if len(sorted_days) >= 21:
        d_start = sorted_days[-21]
        d_end = sorted_days[-1]
        mom = close_panel[d_end] / close_panel[d_start] - 1
        mom = mom.replace([np.inf, -np.inf], np.nan)
        factor_panels['Momentum_20d'][dt] = mom

    # --- 因子2: 短期反转_5d ---
    if len(sorted_days) >= 6:
        d_start5 = sorted_days[-6]
        d_end5 = sorted_days[-1]
        rev = -(close_panel[d_end5] / close_panel[d_start5] - 1)
        rev = rev.replace([np.inf, -np.inf], np.nan)
        factor_panels['Reversal_5d'][dt] = rev

    # --- 因子3: 波动率_20d ---
    if ret_panel.shape[1] >= 20:
        last20_ret = ret_panel[sorted_days[-20:]]
        vol = last20_ret.std(axis=1)
        factor_panels['Vol_20d'][dt] = vol

    # --- 因子4: 换手率_20d (用成交额/收盘价 作为代理) ---
    if money_panel.shape[1] >= 20:
        last20_money = money_panel[sorted_days[-20:]]
        avg_turnover = last20_money.mean(axis=1)
        factor_panels['Turnover_20d'][dt] = np.log1p(avg_turnover)

    # --- 因子5: 成交额变化 近5日/前15日 ---
    if money_panel.shape[1] >= 20:
        recent5 = money_panel[sorted_days[-5:]].mean(axis=1)
        prev15 = money_panel[sorted_days[-20:-5]].mean(axis=1)
        ratio = recent5 / prev15.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        factor_panels['MoneyChg_5v15'][dt] = ratio


# 转为 DataFrame
print("\n转为面板...")
for name in factor_panels:
    factor_panels[name] = pd.DataFrame(factor_panels[name]).T
    factor_panels[name].index.name = 'date'
    print(f"  {name}: {factor_panels[name].shape}")

# 保存
for name, panel in factor_panels.items():
    panel.to_parquet(os.path.join(ARTIFACTS, f'factor_{name}.parquet'))
    print(f"  已保存: factor_{name}.parquet")


# ======================================================
# 因子预处理 + 条件评价
# ======================================================
from conditional_factor_test import ConditionalFactorEvaluator

def preprocess(panel):
    """MAD去极值 + z-score"""
    processed = panel.copy()
    for dt in processed.index:
        row = processed.loc[dt].dropna()
        if len(row) < 100:
            continue
        median = row.median()
        mad = (row - median).abs().median()
        if mad == 0:
            continue
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        row = row.clip(lower, upper)
        std = row.std()
        if std > 0:
            row = (row - row.mean()) / std
        processed.loc[dt, row.index] = row
    return processed

# 计算前向收益率
from scipy import stats

print("\n计算前向收益率...")
factor_dates = sorted(monthly_dates)
returns_dict = {}
for i in range(len(factor_dates) - 1):
    cur, nxt = factor_dates[i], factor_dates[i+1]
    cur_f = os.path.join(DATA_DIR, f"{cur.strftime('%Y-%m-%d')}.parquet")
    nxt_f = os.path.join(DATA_DIR, f"{nxt.strftime('%Y-%m-%d')}.parquet")
    if not os.path.exists(cur_f) or not os.path.exists(nxt_f):
        continue
    df_cur = pd.read_parquet(cur_f, columns=['code', 'close'])
    df_nxt = pd.read_parquet(nxt_f, columns=['code', 'close'])
    close_cur = df_cur.groupby('code')['close'].last()
    close_nxt = df_nxt.groupby('code')['close'].last()
    common = close_cur.index.intersection(close_nxt.index)
    ret = (close_nxt.loc[common] - close_cur.loc[common]) / close_cur.loc[common]
    returns_dict[cur] = ret.replace([np.inf, -np.inf], np.nan)

return_panel = pd.DataFrame(returns_dict).T
print(f"  收益率面板: {return_panel.shape}")


# 构建评价器
print("\n\n" + "="*60)
print("开始条件性评价")
print("="*60)

evaluator = ConditionalFactorEvaluator(
    output_dir=os.path.join(ARTIFACTS, 'conditional_eval_multi')
)

for name, panel in factor_panels.items():
    evaluator.add_factor(name, preprocess(panel))

# 也加上 RV_Skew 做对比
rv_panel = pd.read_parquet(os.path.join(ARTIFACTS, 'factor_panel.parquet'))
evaluator.add_factor('RV_Skew', preprocess(rv_panel))

evaluator.set_returns(return_panel)

# 用缓存的市场状态
cache_path = os.path.join(ARTIFACTS, 'daily_market_stats.parquet')
evaluator.compute_regimes_from_data(cache_path=cache_path)

# 运行
evaluator.run()

# 额外分析: 找出 regime 差异最大的因子
print("\n\n" + "="*60)
print("Regime 差异分析: 哪个因子的条件性差异最大?")
print("="*60)

for name in evaluator.conditional_ic:
    ic_s = evaluator.ic_series[name]
    overall_icir = ic_s.mean() / ic_s.std() if ic_s.std() > 0 else 0

    all_icirs = []
    for dim in evaluator.conditional_ic[name]:
        for val, st in evaluator.conditional_ic[name][dim].items():
            all_icirs.append((dim, val, st['icir'], st['ic_mean']))

    if not all_icirs:
        continue

    best = max(all_icirs, key=lambda x: x[2])
    worst = min(all_icirs, key=lambda x: x[2])
    spread = best[2] - worst[2]

    print(f"\n  {name}:")
    print(f"    全局 ICIR = {overall_icir:+.3f}")
    print(f"    最佳: {best[0]}:{best[1]}  ICIR={best[2]:+.3f}")
    print(f"    最差: {worst[0]}:{worst[1]}  ICIR={worst[2]:+.3f}")
    print(f"    差距 = {spread:.3f}")
    if worst[3] * ic_s.mean() < 0:
        print(f"    *** 最差状态下方向反转! IC均值从 {ic_s.mean():+.4f} 变为 {worst[3]:+.4f} ***")
