"""
条件性因子评价框架 (Conditional Factor Evaluation)
==================================================
核心思想：一个因子不应该用"一个数字"来评价。
因子的价值取决于它在不同市场环境下的表现。

市场状态维度：
  1. 波动率状态 (volatility)   — 近期市场波动高/中/低
  2. 趋势状态 (trend)          — 近期市场涨/平/跌
  3. 宽度状态 (breadth)        — 上涨股票多/中/少
  4. 换手状态 (turnover)       — 市场活跃/一般/清淡
  5. 离散度状态 (dispersion)   — 个股分化大/中/小

评价输出：
  - 条件IC热力图：因子 × 状态 的 IC 矩阵
  - Stability Score: 跨状态一致性评分
  - Worst-Case IC: 最差状态下的表现
  - 因子分类: 全天候 / 条件有效 / 弱效 / 无效

使用方式：
  evaluator = ConditionalFactorEvaluator()
  evaluator.add_factor('RV_Skew', factor_panel)  # (dates × stocks)
  evaluator.set_returns(return_panel)             # (dates × stocks)
  evaluator.run()                                 # 一键分析
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = '/Users/daweizhong/Documents/projects/stock_data'
OUTPUT_DIR = '/Users/daweizhong/Documents/projects/artifacts/conditional_eval'


# =====================================================
# 第一部分：市场状态计算
# =====================================================

def compute_daily_market_stats(dates, data_dir=DATA_DIR):
    """
    从分钟线数据逐日计算市场级别统计量。
    全部用向量化操作，不做 per-stock 循环。

    Returns: DataFrame(index=date, columns=[mkt_ret, breadth, dispersion, ...])
    """
    records = []
    for i, date_str in enumerate(dates):
        if i % 50 == 0:
            print(f"  市场状态: {i}/{len(dates)} ({date_str})")

        filepath = os.path.join(data_dir, f"{date_str}.parquet")
        if not os.path.exists(filepath):
            continue

        df = pd.read_parquet(filepath, columns=['code', 'open', 'close', 'high', 'low',
                                                  'volume', 'money', 'paused'])
        df = df[df['paused'] == 0]
        if len(df) == 0:
            continue

        # 向量化聚合
        agg = df.groupby('code').agg(
            o=('open', 'first'), c=('close', 'last'),
            h=('high', 'max'), l=('low', 'min'),
            money=('money', 'sum')
        )
        agg = agg[(agg['o'] > 0) & (agg['c'] > 0)]
        if len(agg) < 100:
            continue

        ret = agg['c'] / agg['o'] - 1
        amp = (agg['h'] - agg['l']) / agg['o']

        records.append({
            'date': date_str,
            'mkt_ret': ret.mean(),
            'breadth': (ret > 0).mean(),
            'dispersion': ret.std(),
            'amplitude': amp.mean(),
            'median_money': agg['money'].median(),
            'n_stocks': len(agg),
        })

    df_out = pd.DataFrame(records)
    df_out['date'] = pd.to_datetime(df_out['date'])
    return df_out.set_index('date').sort_index()


def compute_regime_labels(daily_stats, lookback=20):
    """
    从日级别统计量 → 滚动指标 → expanding分位数分类。
    使用 expanding percentile 避免未来信息泄漏。

    Returns: (regime_indicators, regime_labels)
    """
    ri = pd.DataFrame(index=daily_stats.index)
    ri['volatility'] = daily_stats['mkt_ret'].rolling(lookback).std()
    ri['trend'] = daily_stats['mkt_ret'].rolling(lookback).sum()
    ri['breadth'] = daily_stats['breadth'].rolling(lookback).mean()
    ri['turnover'] = np.log1p(daily_stats['median_money']).rolling(lookback).mean()
    ri['dispersion'] = daily_stats['dispersion'].rolling(lookback).mean()
    ri = ri.dropna()

    # 三分位标签 (expanding percentile, 避免前视偏差)
    labels = pd.DataFrame(index=ri.index, columns=ri.columns)
    for col in ri.columns:
        pct = ri[col].expanding(min_periods=60).rank(pct=True)
        labels[col] = 'mid'
        labels.loc[pct <= 1/3, col] = 'low'
        labels.loc[pct > 2/3, col] = 'high'

    return ri, labels


# =====================================================
# 第二部分：条件性因子评价器
# =====================================================

class ConditionalFactorEvaluator:
    """
    条件性因子评价主类。

    支持多因子: 每个因子是一个 (dates × stocks) 的 DataFrame。
    """

    REGIME_DISPLAY = {
        'volatility': '波动率', 'trend': '趋势', 'breadth': '宽度',
        'turnover': '换手', 'dispersion': '离散度',
    }
    VAL_DISPLAY = {'low': '低', 'mid': '中', 'high': '高'}

    def __init__(self, output_dir=OUTPUT_DIR):
        self.factors = {}          # {name: DataFrame(dates × stocks)}
        self.return_panel = None   # DataFrame(dates × stocks)
        self.regime_labels = None  # DataFrame(dates × regime_dims)
        self.regime_indicators = None
        self.output_dir = output_dir

        # 结果
        self.ic_series = {}        # {factor_name: Series(dates → IC)}
        self.conditional_ic = {}   # {factor: {dim: {val: dict of stats}}}
        self.summary = None        # DataFrame

    def add_factor(self, name, panel):
        """添加一个因子面板 (dates × stocks)"""
        self.factors[name] = panel
        print(f"  已添加因子 '{name}': {panel.shape[0]}期 × {panel.shape[1]}只股票")

    def set_returns(self, panel):
        """设置收益率面板 (dates × stocks)"""
        self.return_panel = panel
        print(f"  收益率面板: {panel.shape[0]}期 × {panel.shape[1]}只股票")

    def set_regimes(self, labels, indicators=None):
        """直接设置 regime 标签"""
        self.regime_labels = labels
        self.regime_indicators = indicators
        print(f"  市场状态: {labels.shape[0]}天 × {labels.shape[1]}个维度")

    def compute_regimes_from_data(self, data_dir=DATA_DIR, lookback=20, cache_path=None):
        """从分钟线数据自动计算市场状态"""
        # 确定需要的日期范围
        all_files = sorted(f.replace('.parquet', '') for f in os.listdir(data_dir)
                           if f.endswith('.parquet'))

        # 因子面板的所有日期
        all_factor_dates = set()
        for panel in self.factors.values():
            all_factor_dates.update(panel.index.strftime('%Y-%m-%d'))

        min_date = min(all_factor_dates)
        max_date = max(all_factor_dates)

        # 往前多取 lookback+30 天
        start_idx = max(0, next(i for i, d in enumerate(all_files) if d >= min_date) - lookback - 30)
        needed = [d for d in all_files[start_idx:] if d <= max_date]

        if cache_path and os.path.exists(cache_path):
            print(f"读取市场状态缓存: {cache_path}")
            daily_stats = pd.read_parquet(cache_path)
            # 检查是否覆盖所需范围
            cached_dates = set(daily_stats.index.strftime('%Y-%m-%d'))
            missing = [d for d in needed if d not in cached_dates]
            if missing:
                print(f"  缓存缺少 {len(missing)} 天，补充计算...")
                new_stats = compute_daily_market_stats(missing, data_dir)
                daily_stats = pd.concat([daily_stats, new_stats]).sort_index()
                daily_stats = daily_stats[~daily_stats.index.duplicated(keep='last')]
                daily_stats.to_parquet(cache_path)
        else:
            daily_stats = compute_daily_market_stats(needed, data_dir)
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                daily_stats.to_parquet(cache_path)
                print(f"  市场状态已缓存: {cache_path}")

        self.regime_indicators, self.regime_labels = compute_regime_labels(daily_stats, lookback)
        print(f"  市场状态: {self.regime_labels.shape[0]}天 × {self.regime_labels.shape[1]}个维度")

    # --------------------------------------------------
    # IC 计算
    # --------------------------------------------------

    def _rank_ic(self, f, r):
        """截面 Spearman IC"""
        common = f.dropna().index.intersection(r.dropna().index)
        if len(common) < 50:
            return np.nan
        return stats.spearmanr(f.loc[common], r.loc[common])[0]

    def compute_ic(self):
        """计算每个因子每一期的截面 IC"""
        for name, panel in self.factors.items():
            dates = panel.index.intersection(self.return_panel.index)
            ic_list = []
            for dt in dates:
                ic = self._rank_ic(panel.loc[dt], self.return_panel.loc[dt])
                ic_list.append({'date': dt, 'IC': ic})
            self.ic_series[name] = pd.DataFrame(ic_list).set_index('date')['IC'].dropna()
            print(f"  {name}: {len(self.ic_series[name])} 期IC, "
                  f"均值={self.ic_series[name].mean():.4f}, "
                  f"ICIR={self.ic_series[name].mean()/self.ic_series[name].std():.3f}")

    def compute_conditional_ic(self):
        """计算每个因子在每种市场状态下的 IC 统计"""
        if not self.ic_series:
            self.compute_ic()

        for name, ic_s in self.ic_series.items():
            self.conditional_ic[name] = {}
            # 对齐: 因子IC是月频，regime是日频 → 找最近的regime日期
            for dim in self.regime_labels.columns:
                self.conditional_ic[name][dim] = {}
                for val in ['low', 'mid', 'high']:
                    # 对每个IC日期，找对应的regime标签
                    ic_in_regime = []
                    for dt in ic_s.index:
                        # 找 <= dt 的最近regime日期
                        avail = self.regime_labels.index[self.regime_labels.index <= dt]
                        if len(avail) == 0:
                            continue
                        regime_dt = avail[-1]
                        if self.regime_labels.loc[regime_dt, dim] == val:
                            ic_in_regime.append(ic_s.loc[dt])

                    ic_arr = np.array(ic_in_regime)
                    ic_arr = ic_arr[~np.isnan(ic_arr)]
                    if len(ic_arr) < 3:
                        continue

                    self.conditional_ic[name][dim][val] = {
                        'ic_mean': ic_arr.mean(),
                        'ic_std': ic_arr.std(),
                        'icir': ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else 0,
                        'ic_pos_rate': (ic_arr > 0).mean(),
                        'n': len(ic_arr),
                    }

    def compute_summary(self):
        """生成综合评估表"""
        if not self.conditional_ic:
            self.compute_conditional_ic()

        rows = []
        for name in self.factors:
            ic_s = self.ic_series[name]
            overall_icir = ic_s.mean() / ic_s.std() if ic_s.std() > 0 else 0

            # 收集所有条件下的ICIR
            all_icirs = []
            all_ic_means = []
            for dim in self.conditional_ic.get(name, {}):
                for val, st in self.conditional_ic[name][dim].items():
                    all_icirs.append(st['icir'])
                    all_ic_means.append(st['ic_mean'])

            if not all_icirs:
                continue

            worst_icir = min(all_icirs)
            all_positive = all(m > 0 for m in all_ic_means)

            # 稳定性调整
            if overall_icir != 0 and worst_icir * overall_icir > 0:
                stability_adj = min(1.0, abs(worst_icir / overall_icir))
            elif worst_icir * overall_icir <= 0:
                stability_adj = 0.3  # 某些regime下方向反转
            else:
                stability_adj = 0.5

            composite = overall_icir * stability_adj

            # 分类
            if abs(composite) > 0.5 and all_positive:
                cat = 'A_全天候'
            elif abs(composite) > 0.3:
                cat = 'B_多数有效'
            elif abs(overall_icir) > 0.3 and not all_positive:
                cat = 'C_条件有效'
            elif abs(overall_icir) > 0.15:
                cat = 'D_弱效'
            else:
                cat = 'E_无效'

            rows.append({
                'factor': name,
                'ic_mean': round(ic_s.mean(), 4),
                'ic_std': round(ic_s.std(), 4),
                'overall_icir': round(overall_icir, 3),
                'worst_regime_icir': round(worst_icir, 3),
                'all_positive': all_positive,
                'stability_adj': round(stability_adj, 3),
                'composite_score': round(composite, 3),
                'category': cat,
                'n_periods': len(ic_s),
            })

        self.summary = pd.DataFrame(rows).set_index('factor').sort_values('composite_score', ascending=False)
        return self.summary

    # --------------------------------------------------
    # 可视化
    # --------------------------------------------------

    def plot_conditional_heatmap(self, factor_name=None):
        """因子 × 市场状态 的 IC 热力图"""
        os.makedirs(self.output_dir, exist_ok=True)

        if factor_name is None:
            factor_name = list(self.factors.keys())[0]

        cond = self.conditional_ic.get(factor_name, {})
        if not cond:
            print(f"  无条件IC数据: {factor_name}")
            return

        # 构建热力图矩阵: 行=状态维度:值, 列=统计指标
        # 但更直观的是: 一个因子的 dim × val 热力图

        rows_data = []
        row_labels = []
        for dim in ['volatility', 'trend', 'breadth', 'turnover', 'dispersion']:
            if dim not in cond:
                continue
            for val in ['low', 'mid', 'high']:
                if val not in cond[dim]:
                    continue
                st = cond[dim][val]
                dim_name = self.REGIME_DISPLAY.get(dim, dim)
                val_name = self.VAL_DISPLAY.get(val, val)
                row_labels.append(f"{dim_name}:{val_name}")
                rows_data.append(st)

        if not rows_data:
            return

        # 也加上全局
        ic_s = self.ic_series[factor_name]
        global_stats = {
            'ic_mean': ic_s.mean(),
            'icir': ic_s.mean()/ic_s.std() if ic_s.std() > 0 else 0,
            'ic_pos_rate': (ic_s > 0).mean(),
            'n': len(ic_s),
        }
        row_labels.insert(0, '【全局】')
        rows_data.insert(0, global_stats)

        fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(row_labels) * 0.4)))
        fig.suptitle(f'{factor_name} — 条件性评价', fontsize=16, fontweight='bold')

        # 子图1: IC均值柱状图
        ax = axes[0]
        ic_means = [r['ic_mean'] for r in rows_data]
        colors = ['#1565c0' if i == 0 else ('#4caf50' if v > 0 else '#f44336')
                  for i, v in enumerate(ic_means)]
        y_pos = range(len(row_labels))
        bars = ax.barh(y_pos, ic_means, color=colors, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.set_xlabel('IC均值')
        ax.set_title('各状态下的 IC 均值', fontsize=12)
        ax.invert_yaxis()
        for bar, val in zip(bars, ic_means):
            ax.text(val + 0.001 if val >= 0 else val - 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)

        # 子图2: ICIR柱状图
        ax = axes[1]
        icirs = [r['icir'] for r in rows_data]
        colors2 = ['#1565c0' if i == 0 else ('#4caf50' if v > 0 else '#f44336')
                   for i, v in enumerate(icirs)]
        bars2 = ax.barh(y_pos, icirs, color=colors2, alpha=0.8, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.axvline(0, color='grey', linewidth=0.5)
        ax.axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='ICIR=0.3')
        ax.axvline(-0.3, color='orange', linestyle='--', alpha=0.5)
        ax.set_xlabel('ICIR')
        ax.set_title('各状态下的 ICIR', fontsize=12)
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        for bar, val in zip(bars2, icirs):
            ax.text(val + 0.02 if val >= 0 else val - 0.02, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)

        # 子图3: 统计表格
        ax = axes[2]
        ax.axis('off')
        table_data = []
        for lbl, st in zip(row_labels, rows_data):
            table_data.append([
                lbl,
                f"{st['ic_mean']:.4f}",
                f"{st['icir']:.3f}",
                f"{st['ic_pos_rate']:.0%}",
                f"{st['n']}",
            ])
        col_labels = ['状态', 'IC均值', 'ICIR', 'IC>0占比', '期数']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#1565c0')
                cell.set_text_props(color='white', fontweight='bold')
            elif row == 1:
                cell.set_facecolor('#bbdefb')
            else:
                cell.set_facecolor('#f5f5f5' if row % 2 == 0 else 'white')
                # 高亮 IC 为负的行
                if rows_data[row-1]['ic_mean'] < 0:
                    cell.set_facecolor('#ffebee')
        ax.set_title('统计汇总', fontsize=12)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f'conditional_{factor_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  条件IC图已保存: {path}")

    def plot_ic_by_regime_timeseries(self, factor_name=None):
        """IC 时间序列 + regime 着色"""
        os.makedirs(self.output_dir, exist_ok=True)

        if factor_name is None:
            factor_name = list(self.factors.keys())[0]

        ic_s = self.ic_series[factor_name]

        fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
        fig.suptitle(f'{factor_name} — IC时序 与 市场状态', fontsize=16, fontweight='bold')

        # 1. IC 时序 + 波动率着色
        ax = axes[0]
        vol_colors = []
        for dt in ic_s.index:
            avail = self.regime_labels.index[self.regime_labels.index <= dt]
            if len(avail) > 0 and 'volatility' in self.regime_labels.columns:
                regime = self.regime_labels.loc[avail[-1], 'volatility']
                vol_colors.append({'low': '#4caf50', 'mid': '#ff9800', 'high': '#f44336'}.get(regime, '#999'))
            else:
                vol_colors.append('#42a5f5')
        ax.bar(range(len(ic_s)), ic_s.values, color=vol_colors, alpha=0.8, width=1.0)
        ax.axhline(ic_s.mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f'IC均值={ic_s.mean():.4f}')
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.set_ylabel('IC')
        ax.set_title('IC 时序 (颜色=波动率: 绿=低波, 橙=中, 红=高波)', fontsize=11)
        ax.legend(fontsize=9)

        # 2. IC 时序 + 趋势着色
        ax = axes[1]
        trend_colors = []
        for dt in ic_s.index:
            avail = self.regime_labels.index[self.regime_labels.index <= dt]
            if len(avail) > 0 and 'trend' in self.regime_labels.columns:
                regime = self.regime_labels.loc[avail[-1], 'trend']
                trend_colors.append({'low': '#f44336', 'mid': '#ff9800', 'high': '#4caf50'}.get(regime, '#999'))
            else:
                trend_colors.append('#42a5f5')
        ax.bar(range(len(ic_s)), ic_s.values, color=trend_colors, alpha=0.8, width=1.0)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.set_ylabel('IC')
        ax.set_title('IC 时序 (颜色=趋势: 绿=上涨, 橙=震荡, 红=下跌)', fontsize=11)

        # 3. 滚动IC + cumIC
        ax = axes[2]
        roll_ic = ic_s.rolling(6, min_periods=3).mean()
        cum_ic = ic_s.cumsum()
        ax.bar(range(len(ic_s)), ic_s.values, color='#90caf9', alpha=0.4, width=1.0, label='单期IC')
        ax.plot(range(len(roll_ic)), roll_ic.values, 'b-', linewidth=2, label='6期滚动IC')
        ax2 = ax.twinx()
        ax2.plot(range(len(cum_ic)), cum_ic.values, 'k--', linewidth=1.5, alpha=0.6, label='累计IC')
        ax2.set_ylabel('累计IC')
        ax.set_ylabel('IC')
        ax.set_title('滚动IC & 累计IC', fontsize=11)
        ax.axhline(0, color='grey', linewidth=0.5)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, fontsize=9)

        # x轴标签
        tick_positions = list(range(0, len(ic_s), max(1, len(ic_s)//12)))
        tick_labels = [ic_s.index[i].strftime('%Y-%m') for i in tick_positions]
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels(tick_labels, rotation=45, fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f'ic_regime_ts_{factor_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  IC时序图已保存: {path}")

    def plot_regime_boxplot(self, factor_name=None):
        """各regime下的IC箱线图"""
        os.makedirs(self.output_dir, exist_ok=True)

        if factor_name is None:
            factor_name = list(self.factors.keys())[0]

        ic_s = self.ic_series[factor_name]
        cond = self.conditional_ic.get(factor_name, {})

        # 收集数据: 需要原始IC序列在各regime下的分组
        box_groups = {}
        for dim in ['volatility', 'trend', 'breadth', 'turnover', 'dispersion']:
            if dim not in self.regime_labels.columns:
                continue
            for val in ['low', 'mid', 'high']:
                label = f"{self.REGIME_DISPLAY.get(dim,dim)}:{self.VAL_DISPLAY.get(val,val)}"
                ic_in_regime = []
                for dt in ic_s.index:
                    avail = self.regime_labels.index[self.regime_labels.index <= dt]
                    if len(avail) == 0:
                        continue
                    if self.regime_labels.loc[avail[-1], dim] == val:
                        ic_in_regime.append(ic_s.loc[dt])
                if len(ic_in_regime) >= 3:
                    box_groups[label] = ic_in_regime

        if not box_groups:
            return

        fig, ax = plt.subplots(figsize=(max(12, len(box_groups) * 0.8), 7))

        data = list(box_groups.values())
        labels = list(box_groups.keys())
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)

        # 着色: 根据中位数正负
        for i, (patch, d) in enumerate(zip(bp['boxes'], data)):
            median = np.median(d)
            if median > 0.01:
                patch.set_facecolor('#c8e6c9')
            elif median < -0.01:
                patch.set_facecolor('#ffcdd2')
            else:
                patch.set_facecolor('#fff9c4')

        ax.axhline(0, color='grey', linewidth=1)
        ax.set_ylabel('IC', fontsize=12)
        ax.set_title(f'{factor_name} — 各市场状态下的 IC 分布 (箱线图)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(True, alpha=0.2, axis='y')

        # 在每个箱子上方标注均值和期数
        for i, (lbl, d) in enumerate(zip(labels, data)):
            mean_val = np.mean(d)
            ax.text(i+1, ax.get_ylim()[1]*0.95, f'μ={mean_val:.3f}\nn={len(d)}',
                    ha='center', va='top', fontsize=7, color='#333')

        plt.tight_layout()
        path = os.path.join(self.output_dir, f'regime_boxplot_{factor_name}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  箱线图已保存: {path}")

    # --------------------------------------------------
    # 主流程
    # --------------------------------------------------

    def run(self):
        """一键运行全部分析"""
        os.makedirs(self.output_dir, exist_ok=True)

        print("=" * 60)
        print("条件性因子评价框架")
        print("=" * 60)

        # 1. IC
        print("\n>>> 第一步: 计算截面IC")
        self.compute_ic()

        # 2. 条件IC
        print("\n>>> 第二步: 计算条件IC")
        self.compute_conditional_ic()

        # 打印条件IC结果
        for name in self.factors:
            print(f"\n  === {name} ===")
            cond = self.conditional_ic.get(name, {})
            for dim in ['volatility', 'trend', 'breadth', 'turnover', 'dispersion']:
                if dim not in cond:
                    continue
                dim_name = self.REGIME_DISPLAY.get(dim, dim)
                for val in ['low', 'mid', 'high']:
                    if val not in cond[dim]:
                        continue
                    st = cond[dim][val]
                    val_name = self.VAL_DISPLAY.get(val, val)
                    print(f"    {dim_name}:{val_name:2s}  "
                          f"IC={st['ic_mean']:+.4f}  "
                          f"ICIR={st['icir']:+.3f}  "
                          f"IC>0={st['ic_pos_rate']:.0%}  "
                          f"n={st['n']}")

        # 3. 综合评估
        print("\n>>> 第三步: 综合评估")
        self.compute_summary()
        print(self.summary.to_string())

        # 4. 可视化
        print("\n>>> 第四步: 生成可视化")
        for name in self.factors:
            self.plot_conditional_heatmap(name)
            self.plot_ic_by_regime_timeseries(name)
            self.plot_regime_boxplot(name)

        # 5. 保存
        self.summary.to_csv(os.path.join(self.output_dir, 'summary.csv'), encoding='utf-8-sig')

        # 保存条件IC明细
        rows = []
        for name in self.conditional_ic:
            for dim in self.conditional_ic[name]:
                for val, st in self.conditional_ic[name][dim].items():
                    rows.append({'factor': name, 'regime_dim': dim, 'regime_val': val, **st})
        pd.DataFrame(rows).to_csv(os.path.join(self.output_dir, 'conditional_ic_detail.csv'),
                                   index=False, encoding='utf-8-sig')

        print(f"\n所有结果已保存至: {self.output_dir}")
        print("=" * 60)
        return self.summary


# =====================================================
# 第三部分：独立运行 — 用日内波动率因子演示
# =====================================================

if __name__ == '__main__':
    print("加载已有因子面板...")
    ARTIFACTS = '/Users/daweizhong/Documents/projects/artifacts'
    factor_panel = pd.read_parquet(os.path.join(ARTIFACTS, 'factor_panel.parquet'))
    print(f"  因子面板: {factor_panel.shape}")

    # 预处理因子 (MAD去极值 + z-score)
    print("因子预处理...")
    processed = factor_panel.copy()
    for dt in processed.index:
        row = processed.loc[dt].dropna()
        if len(row) < 100:
            continue
        median = row.median()
        mad = (row - median).abs().median()
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        row = row.clip(lower, upper)
        row = (row - row.mean()) / row.std()
        processed.loc[dt, row.index] = row

    # 计算月频前向收益率
    print("计算月频前向收益率...")
    factor_dates = sorted(factor_panel.index)
    returns_dict = {}
    for i in range(len(factor_dates) - 1):
        cur = factor_dates[i]
        nxt = factor_dates[i + 1]
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
    evaluator = ConditionalFactorEvaluator()
    evaluator.add_factor('RV_Skew(日内波动率不对称)', processed)
    evaluator.set_returns(return_panel)

    # 计算市场状态
    cache_path = os.path.join(ARTIFACTS, 'daily_market_stats.parquet')
    evaluator.compute_regimes_from_data(cache_path=cache_path)

    # 运行
    evaluator.run()
