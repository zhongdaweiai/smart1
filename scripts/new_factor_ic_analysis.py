# -*- coding: utf-8 -*-
"""
扩展版因子 IC 分析 (v2)

在原有因子基础上，新增 new_factor_fetcher.py 生成的 10 类新因子，
输出完整的新旧因子对比 IC 报告。

数据源:
  原有 cache: alpha_factors / style_exposures / stock_connect /
             specific_risk / insider_selling
  新增 cache: new_holder / new_capflow / new_margin / new_stk_connect /
             new_unlock / new_staff / new_insider / new_consensus /
             new_news / new_esg

输出目录: output/ic_analysis_v2/
  factor_ic_summary_v2.csv  — 全因子统计表 (按|ICIR|降序)
  factor_ic_series_v2.csv   — 逐期IC序列
  ic_report_v2.txt          — 文字报告
  ic_bar_chart_v2.png       — IC均值/ICIR柱状图
  ic_rolling_top_v2.png     — Top因子滚动IC折线图
  ic_heatmap_annual_v2.png  — 年度IC热力图
"""

import os
import sys
import io
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors as mcolors

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import V1_STYLE_FACTORS

# ── 路径配置 ──────────────────────────────────────────────────
CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
LABEL_PATH = r'D:\HotStockStrategy\external_data\all_codelist_label_filter300500_jump_all_20260205_ddb_label20.pkl'
OUTPUT_DIR = r'D:\HotStockStrategy\multifactors\output\ic_analysis_v2'
NS         = '866015_ri'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 因子名称映射（原有 + 新增）────────────────────────────────
FACTOR_NAMES = {
    # ── 原有：财务 alpha 因子 ──
    'ep_ratio_ttm':                               'EToPGrowth(EP比TTM)',
    'net_profit_parent_company_growth_ratio_ttm': 'EarnVari(净利润增速)',
    'inc_return_on_equity_ttm':                   'ROEG(ROE增长率)',
    'peg_ratio_ttm':                              'PEG2',
    'operating_revenue_growth_ratio_ttm':         'SalesChng(营收增速)',
    'gross_profit_margin_ttm':                    'GrossProfitMargin(毛利率)',
    'return_on_asset_net_profit_ttm':             'ROA(总资产净利率)',
    'net_profit_margin_ttm':                      'NetProfitMargin(净利率)',
    'total_asset_turnover_ttm':                   'AssetTurnover(总资产周转)',
    'operating_profit_growth_ratio_ttm':          'OpProfitGrowth(营业利润增速)',
    'dividend_yield_ttm':                         'DividendYield(股息率)',
    'book_to_market_ratio_lf':                    'BookToMarket(账面市值比)',
    'quick_ratio_lf':                             'QuickRatio(速动比率)',
    # ── 原有：衍生 alpha 因子 ──
    'ShsTurnQtr':                'ShsTurnQtr(季度换手率)',
    'cvtor':                     'cvtor(换手标准化)',
    'mktcapAdjustedTurnover':    'MktcapAdjTurnover(市值调整换手)',
    'specificityIndex_500':      'SpecificityIdx500(1-R²)',
    'capflow5d':                 'Capflow5d(5日净流入旧)',
    'FIHoldings':                'FIHoldings(北向持股水平)',
    'SaleInsiderAmount2Yr':      'InsiderSell旧(高管减持)',
    'idiosyncraticRisk':         'IdioRisk(特异波动率)',
    # ── 原有：v1 风格因子 ──
    'beta':               'Style:Beta',
    'momentum':           'Style:Momentum',
    'size':               'Style:Size',
    'book_to_price':      'Style:Value(BtoP)',
    'earnings_yield':     'Style:EarningsYield',
    'residual_volatility':'Style:ResidVol',
    'growth':             'Style:Growth',
    'leverage':           'Style:Leverage',
    'liquidity':          'Style:Liquidity',
    'non_linear_size':    'Style:NonLinearSize',
    # ── 新增：股东户数 ──
    'holder_count_chg_qoq':  'New:HolderConc(股东集中化)',
    'avg_holder_shares_log':  'New:AvgHolderLog(户均持股log)',
    # ── 新增：资金流 ──
    'capflow_net_20d':        'New:CapFlow20d(20日净流入)',
    'capflow_net_5d':         'New:CapFlow5d(5日净流入)',
    # ── 新增：融资融券 ──
    'margin_balance_chg_20d': 'New:MarginChg20d(融资变动)',
    'short_balance_chg_20d':  'New:ShortChg20d(融券变动)',
    # ── 新增：北向持股变动 ──
    'northbound_chg_20d':     'New:NBChg20d(北向20日变动)',
    'northbound_chg_60d':     'New:NBChg60d(北向60日变动)',
    # ── 新增：解禁压力 ──
    'unlock_ratio_90d':       'New:UnlockRatio90d(解禁压力)',
    # ── 新增：员工数 ──
    'staff_yoy_growth':       'New:StaffGrowth(员工增速)',
    # ── 新增：高管持股变动 ──
    'insider_net_buy_12m':    'New:InsiderBuy12m(高管净买12月)',
    'insider_net_buy_3m':     'New:InsiderBuy3m(高管净买3月)',
    # ── 新增：一致预期 ──
    'eps_revision_3m':        'New:EPSRev3m(EPS预期修正)',
    'con_grd_coef_inv':       'New:AnalystRating(机构评级)',
    'target_price_upside':    'New:TargetUpside(目标价涨幅)',
    # ── 新增：新闻舆情 ──
    'news_sentiment_20d':     'New:NewsSentiment20d(新闻情绪)',
    # ── 新增：ESG ──
    'esg_overall_score':      'New:ESG_Overall(ESG综合)',
    'governance_score':       'New:ESG_Gov(治理得分)',
}


# ── 1. 加载 label ─────────────────────────────────────────────
print('加载 label_num20 ...')
raw = pd.read_pickle(LABEL_PATH)
label_wide = (raw[['code', 'day', 'label_num20']]
              .pivot(index='day', columns='code', values='label_num20'))
label_wide.index = pd.to_datetime(label_wide.index)
print(f'  label shape: {label_wide.shape}  '
      f'日期: {label_wide.index.min().date()} ~ {label_wide.index.max().date()}')


# ── 2. 枚举可用日期 ───────────────────────────────────────────
def load_pkl(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def list_dates(category: str) -> dict:
    """返回 {date_str: pkl_path} 字典"""
    pattern = os.path.join(CACHE_ROOT, category, NS, '*.pkl')
    files = sorted(glob.glob(pattern))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}


# 原有缓存
alpha_dates      = list_dates('alpha_factors')
style_dates      = list_dates('style_exposures')
stockconn_dates  = list_dates('stock_connect')
spec_risk_dates  = list_dates('specific_risk')
insider_dates    = list_dates('insider_selling')

# 新增缓存
new_holder_dates  = list_dates('new_holder')
new_capflow_dates = list_dates('new_capflow')
new_margin_dates  = list_dates('new_margin')
new_stkconn_dates = list_dates('new_stk_connect')
new_unlock_dates  = list_dates('new_unlock')
new_staff_dates   = list_dates('new_staff')
new_insider_dates = list_dates('new_insider')
new_cons_dates    = list_dates('new_consensus')
new_news_dates    = list_dates('new_news')
new_esg_dates     = list_dates('new_esg')

# 基准日期集合：alpha + style（与原有分析保持一致）
all_dates = sorted(set(alpha_dates) & set(style_dates))
print(f'\n基准月末日期: {len(all_dates)} 期  ({all_dates[0]} ~ {all_dates[-1]})')

# 统计新因子覆盖情况
for cat_name, cat_dates in [
    ('new_holder', new_holder_dates), ('new_capflow', new_capflow_dates),
    ('new_margin', new_margin_dates), ('new_stk_connect', new_stkconn_dates),
    ('new_unlock', new_unlock_dates), ('new_staff', new_staff_dates),
    ('new_insider', new_insider_dates), ('new_consensus', new_cons_dates),
    ('new_news', new_news_dates), ('new_esg', new_esg_dates),
]:
    overlap = len(set(cat_dates) & set(all_dates))
    print(f'  {cat_name:20s}: {overlap:3d} 期已缓存')


# ── 3. 截面 IC 计算 ───────────────────────────────────────────
def rank_ic(factor_series: pd.Series, return_series: pd.Series) -> float:
    """Spearman IC：截面排名相关"""
    common = factor_series.dropna().index.intersection(return_series.dropna().index)
    if len(common) < 30:
        return np.nan
    r, _ = spearmanr(factor_series.loc[common], return_series.loc[common])
    return float(r)


def load_df_cols(path: str, exclude_cols: list = None) -> dict:
    """加载 pkl DataFrame，按列名返回 {col: Series} 字典"""
    obj = load_pkl(path)
    if obj is None:
        return {}
    if isinstance(obj, pd.Series):
        return {'__series__': obj}
    if isinstance(obj, pd.DataFrame):
        result = {}
        for col in obj.columns:
            if exclude_cols and col in exclude_cols:
                continue
            result[col] = obj[col]
        return result
    return {}


print('\n计算各期截面 IC ...')
ic_records = []

for date_str in all_dates:
    dt = pd.Timestamp(date_str)

    # 获取对应 label（未来21日收益）
    if dt not in label_wide.index:
        avail = label_wide.index[label_wide.index <= dt]
        if len(avail) == 0:
            continue
        dt_label = avail[-1]
    else:
        dt_label = dt
    future_ret = label_wide.loc[dt_label].dropna()
    if len(future_ret) < 50:
        continue

    row = {'date': dt}

    # ── 原有：alpha_factors ──────────────────────────────────
    if date_str in alpha_dates:
        af = load_pkl(alpha_dates[date_str])
        if isinstance(af, pd.DataFrame):
            for col in af.columns:
                if col == 'market_cap_3':
                    continue
                row[col] = rank_ic(af[col], future_ret)

    # ── 原有：style_exposures ────────────────────────────────
    if date_str in style_dates:
        sf = load_pkl(style_dates[date_str])
        if isinstance(sf, pd.DataFrame):
            for col in V1_STYLE_FACTORS:
                if col in sf.columns:
                    row[col] = rank_ic(sf[col], future_ret)

    # ── 原有：stock_connect (FIHoldings 水平值) ─────────────
    if date_str in stockconn_dates:
        fi = load_pkl(stockconn_dates[date_str])
        if isinstance(fi, pd.Series):
            row['FIHoldings'] = rank_ic(fi, future_ret)

    # ── 原有：specific_risk (idiosyncraticRisk) ─────────────
    if date_str in spec_risk_dates:
        sr = load_pkl(spec_risk_dates[date_str])
        if isinstance(sr, pd.Series):
            row['idiosyncraticRisk'] = rank_ic(sr, future_ret)

    # ── 原有：insider_selling ────────────────────────────────
    if date_str in insider_dates:
        ins = load_pkl(insider_dates[date_str])
        if isinstance(ins, pd.Series):
            row['SaleInsiderAmount2Yr'] = rank_ic(ins, future_ret)

    # ── 新增：通用 DataFrame 加载 (每列→一个因子) ────────────
    new_caches = [
        (new_holder_dates,  date_str),
        (new_capflow_dates, date_str),
        (new_margin_dates,  date_str),
        (new_stkconn_dates, date_str),
        (new_unlock_dates,  date_str),
        (new_staff_dates,   date_str),
        (new_insider_dates, date_str),
        (new_cons_dates,    date_str),
        (new_news_dates,    date_str),
        (new_esg_dates,     date_str),
    ]
    for cat_dict, d in new_caches:
        if d in cat_dict:
            obj = load_pkl(cat_dict[d])
            if isinstance(obj, pd.DataFrame):
                for col in obj.columns:
                    row[col] = rank_ic(obj[col], future_ret)
            elif isinstance(obj, pd.Series):
                row[obj.name if obj.name else d] = rank_ic(obj, future_ret)

    ic_records.append(row)
    if len(ic_records) % 20 == 0:
        print(f'  已处理 {len(ic_records)}/{len(all_dates)} 期')

ic_df = pd.DataFrame(ic_records).set_index('date').sort_index()
print(f'\nIC 矩阵: {ic_df.shape}  (期数 × 因子数)')


# ── 4. 统计汇总 ───────────────────────────────────────────────
ic_mean  = ic_df.mean()
ic_std   = ic_df.std()
ic_ir    = ic_mean / ic_std
ic_t     = ic_mean / (ic_std / np.sqrt(ic_df.count()))
pos_rate = (ic_df > 0).mean()

summary = pd.DataFrame({
    'IC均值':   ic_mean.round(4),
    'IC标准差': ic_std.round(4),
    'ICIR':     ic_ir.round(3),
    't统计量':  ic_t.round(2),
    'IC>0占比': pos_rate.round(3),
    '有效期数': ic_df.count(),
}).sort_values('ICIR', ascending=False)

# 添加可读名称（保留原始名作为索引备份）
summary_named = summary.copy()
summary_named.index = [FACTOR_NAMES.get(c, c) for c in summary.index]

print('\n=== 因子 IC 统计（按 ICIR 排序）===')
print(summary_named.to_string())

# 保存 CSV
summary_path = os.path.join(OUTPUT_DIR, 'factor_ic_summary_v2.csv')
summary_named.to_csv(summary_path, encoding='utf-8-sig')

ic_named = ic_df.rename(columns=FACTOR_NAMES)
ic_path  = os.path.join(OUTPUT_DIR, 'factor_ic_series_v2.csv')
ic_named.to_csv(ic_path, encoding='utf-8-sig')
print(f'\n已保存: {summary_path}')
print(f'已保存: {ic_path}')


# ── 5. 可视化 ─────────────────────────────────────────────────
print('\n绘图 ...')

# ── 5a. IC均值 / ICIR 柱状图 ─────────────────────────────────
n_factors = len(summary_named)
fig, axes = plt.subplots(1, 2, figsize=(22, max(8, n_factors * 0.32)))

colors_mean = ['#e74c3c' if v < 0 else '#2ecc71' for v in summary_named['IC均值']]
summary_named['IC均值'].plot(kind='barh', ax=axes[0],
                              color=colors_mean, edgecolor='gray', linewidth=0.4)
axes[0].axvline(0, color='black', linewidth=1)
axes[0].set_title('各因子 IC 均值', fontsize=12, fontproperties='SimHei')
axes[0].set_xlabel('IC均值', fontproperties='SimHei')
axes[0].tick_params(axis='y', labelsize=7)
for i, (v, t) in enumerate(zip(summary_named['IC均值'], summary_named['t统计量'])):
    axes[0].text(v + 0.001 if v >= 0 else v - 0.001, i,
                 f'{v:.4f}(t={t:.1f})', va='center',
                 ha='left' if v >= 0 else 'right', fontsize=6)

colors_icir = ['#e74c3c' if v < 0 else '#2ecc71' for v in summary_named['ICIR']]
summary_named['ICIR'].plot(kind='barh', ax=axes[1],
                            color=colors_icir, edgecolor='gray', linewidth=0.4)
axes[1].axvline(0.3, color='orange', linewidth=1.5, linestyle='--', label='|ICIR|=0.3 门槛')
axes[1].axvline(-0.3, color='orange', linewidth=1.5, linestyle='--')
axes[1].axvline(0, color='black', linewidth=1)
axes[1].set_title('各因子 ICIR', fontsize=12, fontproperties='SimHei')
axes[1].set_xlabel('ICIR', fontproperties='SimHei')
axes[1].legend(prop={'family': 'SimHei'}, fontsize=8)
axes[1].tick_params(axis='y', labelsize=7)

plt.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, 'ic_bar_chart_v2.png')
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  已保存: {bar_path}')

# ── 5b. Top 因子滚动 IC 折线图 ────────────────────────────────
top_raw = ic_ir.abs().nlargest(12).index.tolist()
top_names = [FACTOR_NAMES.get(f, f) for f in top_raw]

ncols = 3
nrows = (len(top_raw) + ncols - 1) // ncols
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
axes2 = axes2.flatten()

for i, (raw_name, disp_name) in enumerate(zip(top_raw, top_names)):
    ax = axes2[i]
    if raw_name not in ic_df.columns:
        continue
    ic_s = ic_df[raw_name].dropna()
    cum_ic = ic_s.cumsum()
    roll12 = ic_s.rolling(12).mean()

    ax.bar(ic_s.index, ic_s.values,
           color=['#e74c3c' if v < 0 else '#2ecc71' for v in ic_s.values],
           alpha=0.5, width=20, label='月度IC')
    ax.plot(roll12.index, roll12.values, 'b-', linewidth=1.8, label='12期滚动均值')
    ax2 = ax.twinx()
    ax2.plot(cum_ic.index, cum_ic.values, 'k--', linewidth=1.2, alpha=0.7, label='累计IC')
    ax2.set_ylabel('累计IC', fontsize=7, fontproperties='SimHei')

    icir_v = ic_ir.get(raw_name, np.nan)
    ax.set_title(f'{disp_name}\nIC均值={ic_s.mean():.4f}  ICIR={icir_v:.3f}',
                 fontsize=8, fontproperties='SimHei')
    ax.axhline(0, color='gray', linewidth=0.8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6)

for j in range(i + 1, len(axes2)):
    axes2[j].set_visible(False)

plt.suptitle('Top 因子月度 IC 与累计 IC (v2)', fontsize=13,
             fontproperties='SimHei', y=1.01)
plt.tight_layout()
rolling_path = os.path.join(OUTPUT_DIR, 'ic_rolling_top_v2.png')
plt.savefig(rolling_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  已保存: {rolling_path}')

# ── 5c. 年度IC热力图 ──────────────────────────────────────────
ic_ann = ic_named.copy()
ic_ann['year'] = ic_ann.index.year
annual_ic = ic_ann.groupby('year').mean().drop(columns='year', errors='ignore')

# 只显示有效期数 >= 3 的因子列
valid_cols = [c for c in annual_ic.columns
              if ic_named[c].count() >= 3 if c in ic_named.columns]
annual_ic = annual_ic[valid_cols]

fig3, ax3 = plt.subplots(figsize=(max(14, len(annual_ic.columns) * 0.55),
                                   max(5, len(annual_ic) * 0.5)))
norm = mcolors.TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)
im = ax3.imshow(annual_ic.values, aspect='auto', cmap='RdYlGn', norm=norm)
plt.colorbar(im, ax=ax3, label='年均IC')
ax3.set_xticks(range(len(annual_ic.columns)))
ax3.set_xticklabels(annual_ic.columns, rotation=45, ha='right', fontsize=7)
ax3.set_yticks(range(len(annual_ic.index)))
ax3.set_yticklabels(annual_ic.index, fontsize=9)
ax3.set_title('因子年均IC热力图 (v2)', fontsize=13, fontproperties='SimHei')
for yi in range(len(annual_ic.index)):
    for xi in range(len(annual_ic.columns)):
        v = annual_ic.values[yi, xi]
        if not np.isnan(v):
            ax3.text(xi, yi, f'{v:.3f}', ha='center', va='center',
                     fontsize=5, color='black' if abs(v) < 0.03 else 'white')
plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, 'ic_heatmap_annual_v2.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  已保存: {heatmap_path}')


# ── 6. 筛选建议报告 ───────────────────────────────────────────
valid   = summary_named[summary_named['ICIR'].abs() > 0.3]
weak    = summary_named[(summary_named['ICIR'].abs() >= 0.15) & (summary_named['ICIR'].abs() <= 0.3)]
invalid = summary_named[summary_named['ICIR'].abs() < 0.15]

# 分离新旧因子
new_prefix = 'New:'
valid_new   = valid[valid.index.str.startswith(new_prefix)]
valid_old   = valid[~valid.index.str.startswith(new_prefix)]
weak_new    = weak[weak.index.str.startswith(new_prefix)]

print('\n' + '=' * 65)
print('因子筛选建议（ICIR > 0.3 为有效）')
print('=' * 65)
print(f'\n[有效-原有] {len(valid_old)} 个')
print(valid_old[['IC均值', 'ICIR', 'IC>0占比']].to_string())
print(f'\n[有效-新增] {len(valid_new)} 个')
print(valid_new[['IC均值', 'ICIR', 'IC>0占比']].to_string())
print(f'\n[弱效-新增] {len(weak_new)} 个')
print(weak_new[['IC均值', 'ICIR', 'IC>0占比']].to_string())

# 文字报告
report_path = os.path.join(OUTPUT_DIR, 'ic_report_v2.txt')
with open(report_path, 'w', encoding='utf-8') as rpt:
    rpt.write('因子 IC 分析报告 (v2 扩展版)\n')
    rpt.write('=' * 65 + '\n\n')
    rpt.write('全部因子统计（按ICIR排序）\n')
    rpt.write(summary_named.to_string() + '\n\n')
    rpt.write('-' * 65 + '\n')
    rpt.write(f'[有效] 因子（|ICIR|>0.3）: {len(valid)} 个\n')
    rpt.write(valid[['IC均值', 'ICIR', 'IC>0占比']].to_string() + '\n\n')
    rpt.write(f'[弱效] 因子（0.15<|ICIR|<0.3）: {len(weak)} 个\n')
    rpt.write(weak[['IC均值', 'ICIR', 'IC>0占比']].to_string() + '\n\n')
    rpt.write(f'[无效] 因子（|ICIR|<0.15）: {len(invalid)} 个\n')
    rpt.write(invalid[['IC均值', 'ICIR', 'IC>0占比']].to_string() + '\n')

print(f'\n文字报告: {report_path}')
print(f'输出目录: {OUTPUT_DIR}')
print('\n完成！')
