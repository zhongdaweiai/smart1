"""
单因子测试全流程示范
====================
因子：日内波动率不对称因子 (Realized Volatility Skew)
数据：A股全量分钟线
测试区间：2020-01-01 ~ 2025-12-31
调仓频率：月频（每月最后一个交易日调仓）

因子逻辑：
    将每天的分钟收益率分为正收益和负收益两部分，
    分别计算上行已实现波动率(RV+)和下行已实现波动率(RV-)，
    因子 = 过去20个交易日的 mean(RV-) / mean(RV+)

    直觉：下行波动率占比高 → 下跌时波动大 → 市场对该股票偏悲观 → 可能被过度抛售 → 未来有反转机会
    这是一个典型的"反转类"因子。
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False

DATA_DIR = '/Users/daweizhong/Documents/projects/stock_data'
OUTPUT_DIR = '/Users/daweizhong/Documents/projects/artifacts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 第一步：从分钟线数据构造因子
# =====================================================
def calc_daily_rv_skew(df_day):
    """
    计算单日每只股票的上行/下行已实现波动率

    Parameters
    ----------
    df_day : pd.DataFrame — 一天的全市场分钟数据

    Returns
    -------
    pd.DataFrame — columns=['code', 'rv_up', 'rv_down']
    """
    # 过滤停牌和无交易的股票
    df_day = df_day[df_day['paused'] == 0].copy()

    results = []
    for code, group in df_day.groupby('code'):
        if len(group) < 60:  # 交易分钟数太少则跳过
            continue

        # 计算分钟收益率
        close_prices = group.sort_values('datetime')['close'].values
        minute_returns = np.diff(close_prices) / close_prices[:-1]
        minute_returns = minute_returns[np.isfinite(minute_returns)]

        if len(minute_returns) < 30:
            continue

        # 上行已实现波动率: 只用正收益
        up_returns = minute_returns[minute_returns > 0]
        rv_up = np.sqrt(np.sum(up_returns ** 2)) if len(up_returns) > 0 else 0.0

        # 下行已实现波动率: 只用负收益
        down_returns = minute_returns[minute_returns < 0]
        rv_down = np.sqrt(np.sum(down_returns ** 2)) if len(down_returns) > 0 else 0.0

        results.append({'code': code, 'rv_up': rv_up, 'rv_down': rv_down})

    return pd.DataFrame(results)


def build_factor(start_date='2020-01-01', end_date='2025-12-31', lookback=20):
    """
    构造因子：过去 lookback 日的 mean(RV-) / mean(RV+)

    每月最后一个交易日计算因子值（月频调仓）
    """
    # 获取所有数据文件
    all_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    all_dates = [f.replace('.parquet', '') for f in all_files]

    # 筛选日期范围（多取一些用于lookback计算）
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=60)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    valid_dates = [d for d in all_dates if start_dt.strftime('%Y-%m-%d') <= d <= end_date]
    print(f"数据日期范围: {valid_dates[0]} ~ {valid_dates[-1]}, 共 {len(valid_dates)} 天")

    # 第一步：逐日计算 RV_up 和 RV_down
    daily_rv_up = {}
    daily_rv_down = {}

    print("正在逐日计算已实现波动率...")
    for i, date_str in enumerate(valid_dates):
        if i % 100 == 0:
            print(f"  进度: {i}/{len(valid_dates)} ({date_str})")

        filepath = os.path.join(DATA_DIR, f"{date_str}.parquet")
        df_day = pd.read_parquet(filepath)
        rv_df = calc_daily_rv_skew(df_day)

        if len(rv_df) > 0:
            daily_rv_up[date_str] = rv_df.set_index('code')['rv_up']
            daily_rv_down[date_str] = rv_df.set_index('code')['rv_down']

    rv_up_panel = pd.DataFrame(daily_rv_up).T
    rv_down_panel = pd.DataFrame(daily_rv_down).T
    rv_up_panel.index = pd.to_datetime(rv_up_panel.index)
    rv_down_panel.index = pd.to_datetime(rv_down_panel.index)

    print(f"RV面板: {rv_up_panel.shape[0]} 天 x {rv_up_panel.shape[1]} 只股票")

    # 第二步：滚动计算因子值 (每月末计算)
    # 找到每月最后一个交易日
    monthly_dates = rv_up_panel.resample('ME').last().index  # 月末
    # 但要取实际的交易日
    monthly_trade_dates = []
    for m_end in monthly_dates:
        actual_dates = rv_up_panel.index[rv_up_panel.index.month == m_end.month]
        actual_dates = actual_dates[actual_dates.year == m_end.year]
        if len(actual_dates) > 0:
            monthly_trade_dates.append(actual_dates[-1])

    # 筛选在目标范围内的月末日期
    start_pd = pd.Timestamp(start_date)
    monthly_trade_dates = [d for d in monthly_trade_dates if d >= start_pd]

    print(f"月频调仓日: {len(monthly_trade_dates)} 个月")

    factor_dict = {}
    for dt in monthly_trade_dates:
        # 取过去 lookback 个交易日
        loc = rv_up_panel.index.get_loc(dt)
        if loc < lookback:
            continue

        rv_up_window = rv_up_panel.iloc[loc - lookback + 1: loc + 1]
        rv_down_window = rv_down_panel.iloc[loc - lookback + 1: loc + 1]

        mean_rv_up = rv_up_window.mean()
        mean_rv_down = rv_down_window.mean()

        # 因子 = RV_down / RV_up (下行波动占比)
        factor_val = mean_rv_down / mean_rv_up.replace(0, np.nan)
        factor_val = factor_val.replace([np.inf, -np.inf], np.nan)

        factor_dict[dt] = factor_val

    factor_panel = pd.DataFrame(factor_dict).T
    factor_panel.index.name = 'date'
    print(f"因子面板: {factor_panel.shape[0]} 期 x {factor_panel.shape[1]} 只股票")

    return factor_panel, rv_up_panel, rv_down_panel


# =====================================================
# 第二步：计算下一期收益率
# =====================================================
def calc_forward_returns(factor_dates):
    """
    计算每个调仓日到下一个调仓日的区间收益率
    """
    print("正在计算各期收益率...")
    returns_dict = {}

    for i in range(len(factor_dates) - 1):
        current_date = factor_dates[i]
        next_date = factor_dates[i + 1]

        # 读取两个日期的收盘价
        cur_file = os.path.join(DATA_DIR, f"{current_date.strftime('%Y-%m-%d')}.parquet")
        nxt_file = os.path.join(DATA_DIR, f"{next_date.strftime('%Y-%m-%d')}.parquet")

        if not os.path.exists(cur_file) or not os.path.exists(nxt_file):
            continue

        df_cur = pd.read_parquet(cur_file)
        df_nxt = pd.read_parquet(nxt_file)

        # 取每天最后一根分钟线的收盘价作为当天收盘价
        close_cur = df_cur.groupby('code')['close'].last()
        close_nxt = df_nxt.groupby('code')['close'].last()

        common = close_cur.index.intersection(close_nxt.index)
        ret = (close_nxt.loc[common] - close_cur.loc[common]) / close_cur.loc[common]
        ret = ret.replace([np.inf, -np.inf], np.nan)

        returns_dict[current_date] = ret

    return_panel = pd.DataFrame(returns_dict).T
    print(f"收益率面板: {return_panel.shape[0]} 期 x {return_panel.shape[1]} 只股票")
    return return_panel


# =====================================================
# 第三步：因子预处理
# =====================================================
def preprocess_factor(factor_panel):
    """去极值 + 标准化"""
    print("正在进行因子预处理...")
    processed = factor_panel.copy()

    for dt in processed.index:
        row = processed.loc[dt].dropna()
        if len(row) < 100:
            continue

        # MAD 去极值
        median = row.median()
        mad = (row - median).abs().median()
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        row = row.clip(lower, upper)

        # Z-Score 标准化
        row = (row - row.mean()) / row.std()

        processed.loc[dt, row.index] = row

    print("因子预处理完成")
    return processed


# =====================================================
# 第四步：IC 分析
# =====================================================
def calc_ic_series(factor_panel, return_panel):
    """计算每期 Rank IC (Spearman)"""
    dates = factor_panel.index.intersection(return_panel.index)
    ic_list = []

    for dt in dates:
        f = factor_panel.loc[dt].dropna()
        r = return_panel.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < 100:
            continue
        ic, _ = stats.spearmanr(f.loc[common], r.loc[common])
        ic_list.append({'date': dt, 'IC': ic})

    return pd.DataFrame(ic_list).set_index('date')['IC']


def plot_ic_analysis(ic_series, factor_name):
    """绘制 IC 分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'{factor_name} — IC 分析', fontsize=16, fontweight='bold')

    # 1. IC 时间序列
    ax = axes[0, 0]
    colors = ['#d32f2f' if v > 0 else '#1976d2' for v in ic_series.values]
    ax.bar(range(len(ic_series)), ic_series.values, color=colors, alpha=0.7, width=1.0)
    ax.axhline(ic_series.mean(), color='black', linestyle='--', linewidth=1.5,
               label=f'IC均值={ic_series.mean():.4f}')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_title('IC 时间序列', fontsize=13)
    ax.set_ylabel('IC')
    ax.legend()
    # 设置x轴为年份
    n = len(ic_series)
    tick_positions = []
    tick_labels = []
    for i, dt in enumerate(ic_series.index):
        if dt.month == 1:
            tick_positions.append(i)
            tick_labels.append(str(dt.year))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # 2. IC 累计曲线
    ax = axes[0, 1]
    cum_ic = ic_series.cumsum()
    ax.plot(range(len(cum_ic)), cum_ic.values, color='#1565c0', linewidth=2)
    ax.fill_between(range(len(cum_ic)), cum_ic.values, alpha=0.15, color='#1565c0')
    ax.set_title('IC 累计曲线', fontsize=13)
    ax.set_ylabel('累计IC')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # 3. IC 分布直方图
    ax = axes[1, 0]
    ax.hist(ic_series.values, bins=30, color='#42a5f5', edgecolor='white', alpha=0.8)
    ax.axvline(ic_series.mean(), color='red', linestyle='--', linewidth=2,
               label=f'均值={ic_series.mean():.4f}')
    ax.axvline(0, color='grey', linewidth=0.5)
    ax.set_title('IC 分布', fontsize=13)
    ax.set_xlabel('IC')
    ax.set_ylabel('频次')
    ax.legend()

    # 4. IC 统计表
    ax = axes[1, 1]
    ax.axis('off')
    ic_stats = {
        'IC均值': f'{ic_series.mean():.4f}',
        'IC标准差': f'{ic_series.std():.4f}',
        'ICIR': f'{ic_series.mean() / ic_series.std():.4f}',
        'IC>0 占比': f'{(ic_series > 0).mean():.1%}',
        '|IC|>0.02 占比': f'{(ic_series.abs() > 0.02).mean():.1%}',
        't统计量': f'{ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))):.2f}',
        '最大IC': f'{ic_series.max():.4f}',
        '最小IC': f'{ic_series.min():.4f}',
        '测试期数': f'{len(ic_series)}',
    }
    table_data = [[k, v] for k, v in ic_stats.items()]
    table = ax.table(cellText=table_data, colLabels=['指标', '值'],
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565c0')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#f5f5f5' if row % 2 == 0 else 'white')
    ax.set_title('IC 统计汇总', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ic_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"IC分析图已保存: {OUTPUT_DIR}/ic_analysis.png")

    return ic_stats


# =====================================================
# 第五步：分层回测
# =====================================================
def layered_backtest(factor_panel, return_panel, n_groups=5):
    """分层回测：按因子值分组，计算各组收益"""
    dates = sorted(factor_panel.index.intersection(return_panel.index))
    all_group_rets = []

    for dt in dates:
        f = factor_panel.loc[dt].dropna()
        r = return_panel.loc[dt].dropna()
        common = f.index.intersection(r.index)
        if len(common) < n_groups * 30:
            continue

        f_common = f.loc[common]
        r_common = r.loc[common]

        # 等频分组
        try:
            labels = pd.qcut(f_common.rank(method='first'), n_groups,
                             labels=[f'G{i+1}' for i in range(n_groups)])
        except Exception:
            continue

        group_ret = r_common.groupby(labels).mean()
        group_ret.name = dt
        all_group_rets.append(group_ret)

    group_returns = pd.DataFrame(all_group_rets)
    return group_returns


def plot_layered_backtest(group_returns, factor_name):
    """绘制分层回测结果图"""
    n_groups = group_returns.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'{factor_name} — 分层回测 (月频调仓, {n_groups}分组)',
                 fontsize=16, fontweight='bold')

    # 颜色方案: 从绿到红
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n_groups))

    # 1. 各组累计净值曲线
    ax = axes[0, 0]
    cumulative = (1 + group_returns).cumprod()
    for i, col in enumerate(cumulative.columns):
        ax.plot(cumulative.index, cumulative[col], color=colors[i],
                linewidth=2, label=col, marker='o', markersize=2)
    ax.set_title('各组累计净值', fontsize=13)
    ax.set_ylabel('累计净值')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # 2. 多空组合净值 (G5 - G1)
    ax = axes[0, 1]
    long_short = group_returns.iloc[:, -1] - group_returns.iloc[:, 0]
    ls_cumulative = (1 + long_short).cumprod()
    ax.plot(ls_cumulative.index, ls_cumulative.values, color='#d32f2f', linewidth=2.5)
    ax.fill_between(ls_cumulative.index, 1, ls_cumulative.values, alpha=0.1, color='#d32f2f')
    ann_ret = (ls_cumulative.iloc[-1]) ** (12 / len(ls_cumulative)) - 1
    ann_vol = long_short.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    ax.set_title(f'多空组合 (G{n_groups}-G1)\n年化收益:{ann_ret:.1%} | 年化波动:{ann_vol:.1%} | 夏普:{sharpe:.2f}',
                 fontsize=12)
    ax.set_ylabel('累计净值')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    # 3. 各组年化收益柱状图
    ax = axes[1, 0]
    ann_returns = group_returns.mean() * 12
    bars = ax.bar(range(n_groups), ann_returns.values, color=colors, edgecolor='white', width=0.7)
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(group_returns.columns)
    ax.set_title('各组年化收益率', fontsize=13)
    ax.set_ylabel('年化收益率')
    # 在柱子上标数值
    for bar, val in zip(bars, ann_returns.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 统计表格
    ax = axes[1, 1]
    ax.axis('off')
    stats_data = []
    for col in group_returns.columns:
        g = group_returns[col]
        ann_r = g.mean() * 12
        ann_v = g.std() * np.sqrt(12)
        sr = ann_r / ann_v if ann_v > 0 else 0
        cum = (1 + g).cumprod().iloc[-1]
        max_dd = ((1 + g).cumprod() / (1 + g).cumprod().cummax() - 1).min()
        stats_data.append([col, f'{ann_r:.2%}', f'{ann_v:.2%}', f'{sr:.2f}',
                           f'{cum:.3f}', f'{max_dd:.2%}'])

    # 加上多空组合
    ls = long_short
    ann_r_ls = ls.mean() * 12
    ann_v_ls = ls.std() * np.sqrt(12)
    sr_ls = ann_r_ls / ann_v_ls if ann_v_ls > 0 else 0
    cum_ls = (1 + ls).cumprod().iloc[-1]
    max_dd_ls = ((1 + ls).cumprod() / (1 + ls).cumprod().cummax() - 1).min()
    stats_data.append([f'G{n_groups}-G1', f'{ann_r_ls:.2%}', f'{ann_v_ls:.2%}',
                       f'{sr_ls:.2f}', f'{cum_ls:.3f}', f'{max_dd_ls:.2%}'])

    col_labels = ['分组', '年化收益', '年化波动', '夏普比率', '累计净值', '最大回撤']
    table = ax.table(cellText=stats_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1565c0')
            cell.set_text_props(color='white', fontweight='bold')
        elif row == len(stats_data):  # 多空行高亮
            cell.set_facecolor('#fff3e0')
        else:
            cell.set_facecolor('#f5f5f5' if row % 2 == 0 else 'white')
    ax.set_title('分组绩效统计', fontsize=13)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'layered_backtest.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分层回测图已保存: {OUTPUT_DIR}/layered_backtest.png")


# =====================================================
# 第六步：换手率分析
# =====================================================
def calc_turnover(factor_panel, n_groups=5):
    """计算头部组和尾部组的换手率"""
    dates = sorted(factor_panel.dropna(how='all').index)

    prev_top = None
    prev_bot = None
    turnover_list = []

    for dt in dates:
        f = factor_panel.loc[dt].dropna()
        if len(f) < 100:
            continue

        threshold_top = f.quantile(1 - 1.0/n_groups)
        threshold_bot = f.quantile(1.0/n_groups)

        top_stocks = set(f[f >= threshold_top].index)
        bot_stocks = set(f[f <= threshold_bot].index)

        if prev_top is not None and len(top_stocks) > 0:
            top_turnover = len(top_stocks - prev_top) / len(top_stocks)
            bot_turnover = len(bot_stocks - prev_bot) / len(bot_stocks)
            turnover_list.append({
                'date': dt,
                'top_turnover': top_turnover,
                'bot_turnover': bot_turnover,
            })

        prev_top = top_stocks
        prev_bot = bot_stocks

    return pd.DataFrame(turnover_list).set_index('date')


def plot_turnover(turnover_df, factor_name):
    """绘制换手率分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{factor_name} — 换手率分析', fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.plot(turnover_df.index, turnover_df['top_turnover'],
            label=f'头部组 (均值:{turnover_df["top_turnover"].mean():.1%})',
            color='#d32f2f', linewidth=1.5)
    ax.plot(turnover_df.index, turnover_df['bot_turnover'],
            label=f'尾部组 (均值:{turnover_df["bot_turnover"].mean():.1%})',
            color='#1976d2', linewidth=1.5)
    ax.set_title('月度换手率时序', fontsize=12)
    ax.set_ylabel('换手率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

    ax = axes[1]
    ax.hist(turnover_df['top_turnover'], bins=20, alpha=0.6, color='#d32f2f', label='头部组')
    ax.hist(turnover_df['bot_turnover'], bins=20, alpha=0.6, color='#1976d2', label='尾部组')
    ax.set_title('换手率分布', fontsize=12)
    ax.set_xlabel('换手率')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'turnover_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"换手率分析图已保存: {OUTPUT_DIR}/turnover_analysis.png")


# =====================================================
# 第七步：IC衰减分析
# =====================================================
def ic_decay_analysis(factor_panel, n_months=6):
    """
    IC衰减：检验因子对未来1~N个月收益率的预测能力
    """
    factor_dates = sorted(factor_panel.dropna(how='all').index)
    decay_ics = {}

    for lag in range(1, n_months + 1):
        ic_list = []
        for i in range(len(factor_dates) - lag):
            dt = factor_dates[i]
            dt_future = factor_dates[i + lag]

            # 读取收盘价
            cur_file = os.path.join(DATA_DIR, f"{dt.strftime('%Y-%m-%d')}.parquet")
            fut_file = os.path.join(DATA_DIR, f"{dt_future.strftime('%Y-%m-%d')}.parquet")

            if not os.path.exists(cur_file) or not os.path.exists(fut_file):
                continue

            df_cur = pd.read_parquet(cur_file)
            df_fut = pd.read_parquet(fut_file)

            close_cur = df_cur.groupby('code')['close'].last()
            close_fut = df_fut.groupby('code')['close'].last()

            common = close_cur.index.intersection(close_fut.index)
            ret = (close_fut.loc[common] - close_cur.loc[common]) / close_cur.loc[common]

            f = factor_panel.loc[dt].dropna()
            common2 = f.index.intersection(ret.dropna().index)
            if len(common2) < 100:
                continue

            ic, _ = stats.spearmanr(f.loc[common2], ret.loc[common2])
            ic_list.append(ic)

        if ic_list:
            decay_ics[lag] = np.mean(ic_list)

    return pd.Series(decay_ics)


def plot_ic_decay(decay_series, factor_name):
    """绘制IC衰减图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#1565c0' if v > 0 else '#d32f2f' for v in decay_series.values]
    bars = ax.bar(decay_series.index, decay_series.values, color=colors,
                  edgecolor='white', width=0.6)
    for bar, val in zip(bars, decay_series.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001 * np.sign(val),
                f'{val:.4f}', ha='center', va='bottom' if val > 0 else 'top', fontsize=11)
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.set_xlabel('滞后月数')
    ax.set_ylabel('IC均值')
    ax.set_title(f'{factor_name} — IC衰减分析', fontsize=14, fontweight='bold')
    ax.set_xticks(decay_series.index)
    ax.set_xticklabels([f'{i}个月' for i in decay_series.index])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ic_decay.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"IC衰减图已保存: {OUTPUT_DIR}/ic_decay.png")


# =====================================================
# 主流程
# =====================================================
def main():
    factor_name = '日内波动率不对称因子 (RV Skew)'

    print("=" * 60)
    print(f"单因子测试: {factor_name}")
    print("=" * 60)

    # 1. 构造因子
    print("\n>>> 第一步: 构造因子")
    factor_panel, rv_up_panel, rv_down_panel = build_factor(
        start_date='2020-01-01', end_date='2025-12-31', lookback=20
    )

    # 保存中间结果
    factor_panel.to_parquet(os.path.join(OUTPUT_DIR, 'factor_panel.parquet'))

    # 2. 因子预处理
    print("\n>>> 第二步: 因子预处理")
    factor_processed = preprocess_factor(factor_panel)

    # 3. 计算收益率
    print("\n>>> 第三步: 计算下一期收益率")
    return_panel = calc_forward_returns(factor_panel.index.tolist())

    # 4. IC 分析
    print("\n>>> 第四步: IC 分析")
    ic_series = calc_ic_series(factor_processed, return_panel)
    ic_stats = plot_ic_analysis(ic_series, factor_name)
    print("\nIC统计:")
    for k, v in ic_stats.items():
        print(f"  {k}: {v}")

    # 5. 分层回测
    print("\n>>> 第五步: 分层回测")
    group_returns = layered_backtest(factor_processed, return_panel, n_groups=5)
    plot_layered_backtest(group_returns, factor_name)

    # 6. 换手率
    print("\n>>> 第六步: 换手率分析")
    turnover_df = calc_turnover(factor_processed, n_groups=5)
    plot_turnover(turnover_df, factor_name)

    # 7. IC衰减
    print("\n>>> 第七步: IC衰减分析")
    decay_series = ic_decay_analysis(factor_processed, n_months=6)
    plot_ic_decay(decay_series, factor_name)

    print("\n" + "=" * 60)
    print("全部测试完成! 图表已保存至:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
