# -*- coding: utf-8 -*-
"""
新因子数据获取器

从 rqdata API 批量获取并缓存 10 类新因子。
每类因子独立缓存到:
  cache/new_{类别}/866015_ri/{YYYY-MM-DD}.pkl  →  pd.DataFrame(index=order_book_id, columns=因子名)

运行方式:
  python new_factor_fetcher.py [--start 2013-01-31] [--end 2023-12-29]

增量运行: 已缓存的 (类别, 日期) 自动跳过。

数据源与因子:
  1. get_holder_number      → holder_count_chg_qoq, avg_holder_shares_log
  2. get_capital_flow       → capflow_net_20d, capflow_net_5d
  3. get_securities_margin  → margin_balance_chg_20d, short_balance_chg_20d
  4. get_stock_connect      → northbound_chg_20d, northbound_chg_60d
  5. get_restricted_shares  → unlock_ratio_90d
  6. get_staff_count        → staff_yoy_growth
  7. get_leader_shares_change → insider_net_buy_12m, insider_net_buy_3m
  8. consensus.get_comp_indicators → eps_revision_3m, con_grd_coef_inv, target_price_upside
  9. news.get_stock_news    → news_sentiment_20d   [需 pip install rqdatac_news]
  10. esg.get_rating        → esg_overall_score, governance_score [需 pip install rqdatac_esg]
"""

import os
import sys
import io
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import rqdatac

# Windows GBK 终端中文支持
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cache as _cache
from data_fetcher import get_universe_stocks

# ── 可选依赖 ──────────────────────────────────────────────────
try:
    import rqdatac_news  # noqa: F401
    HAS_NEWS = True
except ImportError:
    HAS_NEWS = False
    print('[提示] rqdatac_news 未安装，跳过新闻舆情因子')

try:
    import rqdatac_esg  # noqa: F401
    HAS_ESG = True
except ImportError:
    HAS_ESG = False
    print('[提示] rqdatac_esg 未安装，跳过 ESG 因子')

# ── 常量 ──────────────────────────────────────────────────────
CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
NS = '866015_ri'


# ── 工具函数 ──────────────────────────────────────────────────
def _list_dates(category: str) -> list:
    pattern = os.path.join(CACHE_ROOT, category, NS, '*.pkl')
    files = sorted(glob.glob(pattern))
    return [os.path.splitext(os.path.basename(f))[0] for f in files]


def _get_all_dates() -> list:
    alpha_dates = set(_list_dates('alpha_factors'))
    style_dates = set(_list_dates('style_exposures'))
    return sorted(alpha_dates & style_dates)


def _empty(stocks: list, cols: list) -> pd.DataFrame:
    """返回全 NaN 的空 DataFrame，index=stocks, columns=cols"""
    return pd.DataFrame(np.nan, index=stocks, columns=cols)


# ══════════════════════════════════════════════════════════════
# 因子 1：股东户数
# ══════════════════════════════════════════════════════════════
def fetch_holder_count(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_holder_number，全量 stocks 一次传入
    因子:
      holder_count_chg_qoq  股东户数季环比变动率 (负=集中化=看多)
      avg_holder_shares_log 户均持股数对数 (高=筹码集中)
    """
    COLS = ['holder_count_chg_qoq', 'avg_holder_shares_log']
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=200)).strftime('%Y-%m-%d')
        df = rqdatac.get_holder_number(stocks, start_date=start, end_date=date_str)
        if df is None or df.empty:
            return _empty(stocks, COLS)

        df = df.sort_index()
        # 按 order_book_id 分组，取最近两期
        latest = df.groupby(level=0).last()
        prev   = df.groupby(level=0).nth(-2)

        # 股东户数季环比变动率
        sh_now  = latest.get('a_share_holders', pd.Series(dtype=float))
        sh_prev = prev.get('a_share_holders',   pd.Series(dtype=float))
        chg_qoq = (sh_now - sh_prev) / sh_prev.abs().replace(0, np.nan)

        # 户均持股对数
        avg_sh  = latest.get('avg_a_share_holders', pd.Series(dtype=float))
        avg_log = np.log(avg_sh.replace(0, np.nan))

        result = pd.DataFrame({
            'holder_count_chg_qoq':  chg_qoq,
            'avg_holder_shares_log': avg_log,
        }).reindex(stocks)
        result.index.name = 'order_book_id'
        return result
    except Exception as e:
        print(f'  [错误] fetch_holder_count({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 2：资金流
# ══════════════════════════════════════════════════════════════
def fetch_capital_flow(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_capital_flow (日线)，全量 stocks 一次传入
    因子:
      capflow_net_20d  过去20交易日 (buy_value-sell_value)/(buy+sell)
      capflow_net_5d   过去5交易日同上
    """
    COLS = ['capflow_net_20d', 'capflow_net_5d']
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=35)).strftime('%Y-%m-%d')
        df = rqdatac.get_capital_flow(stocks, start_date=start, end_date=date_str, frequency='1d')
        if df is None or df.empty:
            return _empty(stocks, COLS)

        # 转为宽表: date × stock
        buy  = df['buy_value'].unstack(level='order_book_id')
        sell = df['sell_value'].unstack(level='order_book_id')
        net  = buy - sell
        tot  = (buy + sell).replace(0, np.nan)

        cf_20 = net.tail(20).sum() / tot.tail(20).sum()
        cf_5  = net.tail(5).sum()  / tot.tail(5).sum()

        result = pd.DataFrame({'capflow_net_20d': cf_20, 'capflow_net_5d': cf_5})
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_capital_flow({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 3：融资融券
# ══════════════════════════════════════════════════════════════
def fetch_margin(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_securities_margin，全量 stocks 一次传入
    因子:
      margin_balance_chg_20d  融资余额20日变动率 (加杠杆情绪)
      short_balance_chg_20d   融券余量20日变动率 (做空情绪)
    """
    COLS = ['margin_balance_chg_20d', 'short_balance_chg_20d']
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=35)).strftime('%Y-%m-%d')
        df = rqdatac.get_securities_margin(
            stocks, start_date=start, end_date=date_str,
            fields=['margin_balance', 'short_balance_quantity'],
            expect_df=True,
        )
        if df is None or df.empty:
            return _empty(stocks, COLS)

        mb = df['margin_balance'].unstack(level='order_book_id')
        sb = df['short_balance_quantity'].unstack(level='order_book_id')
        if len(mb) < 2:
            return _empty(stocks, COLS)

        n = min(21, len(mb))
        mb_chg = (mb.iloc[-1] - mb.iloc[-n]) / mb.iloc[-n].abs().replace(0, np.nan)
        sb_chg = (sb.iloc[-1] - sb.iloc[-n]) / sb.iloc[-n].abs().replace(0, np.nan)

        result = pd.DataFrame({
            'margin_balance_chg_20d': mb_chg,
            'short_balance_chg_20d':  sb_chg,
        })
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_margin({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 4：北向持股变动
# ══════════════════════════════════════════════════════════════
def fetch_stock_connect_change(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_stock_connect，全量 stocks 一次传入
    因子:
      northbound_chg_20d  北向持股比例20日变动 (pct point)
      northbound_chg_60d  北向持股比例60日变动 (pct point)
    """
    COLS = ['northbound_chg_20d', 'northbound_chg_60d']
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=95)).strftime('%Y-%m-%d')
        df = rqdatac.get_stock_connect(stocks, start_date=start, end_date=date_str,
                                       fields=['holding_ratio'])
        if df is None or df.empty:
            return _empty(stocks, COLS)

        hr = df['holding_ratio'].unstack(level='order_book_id')
        if len(hr) < 2:
            return _empty(stocks, COLS)

        hr_now = hr.iloc[-1]
        n20 = min(21, len(hr))
        n60 = min(61, len(hr))

        result = pd.DataFrame({
            'northbound_chg_20d': hr_now - hr.iloc[-n20],
            'northbound_chg_60d': hr_now - hr.iloc[-n60],
        })
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_stock_connect_change({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 5：解禁压力
# ══════════════════════════════════════════════════════════════
def fetch_unlock_pressure(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_restricted_shares，全量 stocks 一次传入
    因子:
      unlock_ratio_90d  未来90日解禁股/总股本 (高=解禁压力大=负向信号)
    """
    COLS = ['unlock_ratio_90d']
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=730)).strftime('%Y-%m-%d')
        df = rqdatac.get_restricted_shares(stocks, start_date=start, end_date=date_str)
        if df is None or df.empty:
            return _empty(stocks, COLS)

        # 筛选 relieve_date 在未来 90 天内
        end_window = dt + pd.Timedelta(days=90)
        df['relieve_date'] = pd.to_datetime(df['relieve_date'])
        future = df[(df['relieve_date'] > dt) & (df['relieve_date'] <= end_window)]
        if future.empty:
            return _empty(stocks, COLS)

        unlock_shares = future.groupby(level='order_book_id')['relieve_shares'].sum()

        # 归一化：总股本
        try:
            shares_df = rqdatac.get_shares(stocks, start_date=date_str, end_date=date_str)
            if shares_df is not None and not shares_df.empty:
                # get_shares 可能返回 MultiIndex (order_book_id, date)
                if isinstance(shares_df.index, pd.MultiIndex):
                    shares_df = shares_df.groupby(level='order_book_id').last()
                total_col = 'total' if 'total' in shares_df.columns else shares_df.columns[0]
                total_shares = shares_df[total_col]
                unlock_ratio = (unlock_shares / total_shares.replace(0, np.nan)).reindex(stocks)
            else:
                unlock_ratio = unlock_shares.reindex(stocks)
        except Exception:
            # 归一化失败则使用原始值（IC 仍有效，因为是截面排名）
            unlock_ratio = unlock_shares.reindex(stocks)

        result = pd.DataFrame({'unlock_ratio_90d': unlock_ratio})
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_unlock_pressure({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 6：员工数
# ══════════════════════════════════════════════════════════════
def fetch_staff_count(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_staff_count，全量 stocks 一次传入
    因子:
      staff_yoy_growth  员工数YoY增速 (正=扩张)
    """
    COLS = ['staff_yoy_growth']
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=548)).strftime('%Y-%m-%d')  # ~18个月
        df = rqdatac.get_staff_count(stocks, start_date=start, end_date=date_str)
        if df is None or df.empty:
            return _empty(stocks, COLS)

        df = df.sort_index()
        staff_col = ('total_staff' if 'total_staff' in df.columns
                     else 'staff_count' if 'staff_count' in df.columns
                     else df.columns[0])

        # 最新值
        latest = df.groupby(level='order_book_id')[staff_col].last()

        # 约 1 年前的值
        dt_1y = dt - pd.Timedelta(days=365)
        info_dates = df.index.get_level_values('info_date')
        older = df[info_dates <= dt_1y]
        if not older.empty:
            prev = older.groupby(level='order_book_id')[staff_col].last()
        else:
            prev = df.groupby(level='order_book_id')[staff_col].first()

        yoy = ((latest - prev) / prev.abs().replace(0, np.nan)).reindex(stocks)
        result = pd.DataFrame({'staff_yoy_growth': yoy})
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_staff_count({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 7：高管持股变动
# ══════════════════════════════════════════════════════════════
def fetch_insider_trades(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 get_leader_shares_change，全量 stocks 一次传入
    因子:
      insider_net_buy_12m  过去12个月高管净买入/总股本 (正=增持)
      insider_net_buy_3m   过去3个月高管净买入/总股本
    """
    COLS = ['insider_net_buy_12m', 'insider_net_buy_3m']
    try:
        dt = pd.Timestamp(date_str)
        start_12m = (dt - pd.Timedelta(days=365)).strftime('%Y-%m-%d')

        df = rqdatac.get_leader_shares_change(stocks, start_date=start_12m, end_date=date_str)
        if df is None or df.empty:
            return _empty(stocks, COLS)

        df = df.sort_index()

        # 12 个月净买入
        net_12m = df.groupby(level='order_book_id')['shares_change'].sum()

        # 3 个月净买入
        start_3m = (dt - pd.Timedelta(days=92)).strftime('%Y-%m-%d')
        change_dates = df.index.get_level_values('change_date')
        df_3m = df[change_dates >= pd.Timestamp(start_3m)]
        net_3m = (df_3m.groupby(level='order_book_id')['shares_change'].sum()
                  if not df_3m.empty else pd.Series(dtype=float))

        # 用 current_shares 归一化（取最新一条记录的持股数作为分母）
        if 'current_shares' in df.columns:
            denom = df.groupby(level='order_book_id')['current_shares'].last().replace(0, np.nan)
            net_12m = (net_12m / denom).reindex(stocks)
            net_3m  = (net_3m  / denom).reindex(stocks)
        else:
            net_12m = net_12m.reindex(stocks)
            net_3m  = net_3m.reindex(stocks)

        result = pd.DataFrame({'insider_net_buy_12m': net_12m, 'insider_net_buy_3m': net_3m})
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_insider_trades({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 8：一致预期
# ══════════════════════════════════════════════════════════════
def fetch_consensus(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 consensus.get_comp_indicators (两次：当期 + 3个月前)
    因子:
      eps_revision_3m     T+1年EPS预期3月变动率 (上调动能)
      con_grd_coef_inv    评级系数倒置 (6-原始值, 越高=越看多)
      target_price_upside 目标价涨幅空间 = (目标价-现价)/现价
    """
    COLS = ['eps_revision_3m', 'con_grd_coef_inv', 'target_price_upside']
    try:
        dt = pd.Timestamp(date_str)
        date_3m_ago = (dt - pd.Timedelta(days=91)).strftime('%Y-%m-%d')

        fields_all = ['comp_con_eps_t1', 'con_grd_coef', 'con_targ_price']
        fields_eps  = ['comp_con_eps_t1']

        # 当期值
        curr_df = rqdatac.consensus.get_comp_indicators(
            stocks, start_date=date_str, end_date=date_str, fields=fields_all
        )
        if curr_df is None or curr_df.empty:
            return _empty(stocks, COLS)

        if isinstance(curr_df.index, pd.MultiIndex):
            curr = curr_df.groupby(level='order_book_id').last()
        else:
            curr = curr_df

        # 3 个月前值（用于 EPS 修正率）
        past_df = rqdatac.consensus.get_comp_indicators(
            stocks, start_date=date_3m_ago, end_date=date_3m_ago, fields=fields_eps
        )
        if past_df is not None and not past_df.empty:
            if isinstance(past_df.index, pd.MultiIndex):
                past = past_df.groupby(level='order_book_id').last()
            else:
                past = past_df
            eps_now  = curr.get('comp_con_eps_t1', pd.Series(dtype=float))
            eps_prev = past.get('comp_con_eps_t1', pd.Series(dtype=float))
            eps_rev  = ((eps_now - eps_prev) / eps_prev.abs().replace(0, np.nan)).reindex(stocks)
        else:
            eps_rev = pd.Series(np.nan, index=stocks)

        # 评级系数倒置 (原始: 1=强力买入 … 5=卖出 → 倒置: 5=强力买入)
        grd = curr.get('con_grd_coef', pd.Series(dtype=float))
        grd_inv = (6 - grd).reindex(stocks)

        # 目标价涨幅：需当期收盘价
        targ_price = curr.get('con_targ_price', pd.Series(dtype=float))
        try:
            price_df = rqdatac.get_price(
                stocks, start_date=date_str, end_date=date_str,
                fields=['close'], expect_df=True,
            )
            if price_df is not None and not price_df.empty:
                if isinstance(price_df.index, pd.MultiIndex):
                    close_px = price_df['close'].groupby(level='order_book_id').last()
                else:
                    close_px = price_df['close']
                upside = ((targ_price - close_px) / close_px.replace(0, np.nan)).reindex(stocks)
            else:
                upside = pd.Series(np.nan, index=stocks)
        except Exception:
            upside = pd.Series(np.nan, index=stocks)

        result = pd.DataFrame({
            'eps_revision_3m':     eps_rev,
            'con_grd_coef_inv':    grd_inv,
            'target_price_upside': upside,
        })
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_consensus({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 9：新闻舆情（可选，需 rqdatac_news）
# ══════════════════════════════════════════════════════════════
def fetch_news_sentiment(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 news.get_stock_news (需 rqdatac_news)，全量 stocks 一次传入
    因子:
      news_sentiment_20d  过去20日公司相关度加权情绪净值
    """
    COLS = ['news_sentiment_20d']
    if not HAS_NEWS:
        return _empty(stocks, COLS)
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        df = rqdatac.news.get_stock_news(
            stocks, start_date=start, end_date=date_str,
            fields=['company_positive_weight', 'company_negative_weight', 'company_relevance'],
        )
        if df is None or df.empty:
            return _empty(stocks, COLS)

        df = df.reset_index()
        df['sentiment'] = (df['company_positive_weight'] - df['company_negative_weight']
                           ) * df['company_relevance']
        agg = df.groupby('order_book_id').agg(
            w_sum=('company_relevance', 'sum'),
            s_sum=('sentiment', 'sum'),
        )
        score = (agg['s_sum'] / agg['w_sum'].replace(0, np.nan)).reindex(stocks)
        result = pd.DataFrame({'news_sentiment_20d': score})
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_news_sentiment({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子 10：ESG（可选，需 rqdatac_esg）
# ══════════════════════════════════════════════════════════════
def fetch_esg(stocks: list, date_str: str) -> pd.DataFrame:
    """
    调用 esg.get_rating (需 rqdatac_esg)，全量 stocks 一次传入
    因子:
      esg_overall_score  ESG 综合得分 (level=0)
      governance_score   治理维度得分 (level=1, type='G')
    """
    COLS = ['esg_overall_score', 'governance_score']
    if not HAS_ESG:
        return _empty(stocks, COLS)
    try:
        dt = pd.Timestamp(date_str)
        start = (dt - pd.Timedelta(days=400)).strftime('%Y-%m-%d')

        def _latest_score(df_esg, name_filter):
            if df_esg is None or df_esg.empty:
                return pd.Series(np.nan, index=stocks)
            sub = df_esg[df_esg['name'] == name_filter] if 'name' in df_esg.columns else df_esg
            if sub.empty:
                return pd.Series(np.nan, index=stocks)
            if isinstance(sub.index, pd.MultiIndex):
                s = sub.groupby(level='order_book_id')['score'].last()
            else:
                s = sub.groupby('order_book_id')['score'].last()
            return s.reindex(stocks)

        df0 = rqdatac.esg.get_rating(stocks, start_date=start, end_date=date_str, level=[0])
        df1 = rqdatac.esg.get_rating(stocks, start_date=start, end_date=date_str,
                                     level=[1], type=['G'])

        result = pd.DataFrame({
            'esg_overall_score': _latest_score(df0, 'esg_overall'),
            'governance_score':  _latest_score(df1, 'governance'),
        })
        result.index.name = 'order_book_id'
        return result.reindex(stocks)
    except Exception as e:
        print(f'  [错误] fetch_esg({date_str}): {e}')
        return _empty(stocks, COLS)


# ══════════════════════════════════════════════════════════════
# 因子类别注册表：(缓存目录名, 获取函数)
# ══════════════════════════════════════════════════════════════
FETCH_REGISTRY = [
    ('new_holder',      fetch_holder_count),
    ('new_capflow',     fetch_capital_flow),
    ('new_margin',      fetch_margin),
    ('new_stk_connect', fetch_stock_connect_change),
    ('new_unlock',      fetch_unlock_pressure),
    ('new_staff',       fetch_staff_count),
    ('new_insider',     fetch_insider_trades),
    ('new_consensus',   fetch_consensus),
    ('new_news',        fetch_news_sentiment),
    ('new_esg',         fetch_esg),
]


# ══════════════════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='新因子数据获取器')
    parser.add_argument('--start', default=None, help='起始日期 (YYYY-MM-DD)')
    parser.add_argument('--end',   default=None, help='截止日期 (YYYY-MM-DD)')
    parser.add_argument('--cat',   default=None,
                        help='仅运行指定类别 (如 new_capflow)，默认全部')
    args = parser.parse_args()

    rqdatac.init()

    all_dates = _get_all_dates()
    if args.start:
        all_dates = [d for d in all_dates if d >= args.start]
    if args.end:
        all_dates = [d for d in all_dates if d <= args.end]

    registry = ([(n, f) for n, f in FETCH_REGISTRY if n == args.cat]
                if args.cat else FETCH_REGISTRY)

    print(f'\n待处理月末日期: {len(all_dates)} 期  ({all_dates[0]} ~ {all_dates[-1]})')
    print(f'因子类别数: {len(registry)}\n{"="*60}')

    for i, date_str in enumerate(all_dates):
        print(f'\n[{i+1}/{len(all_dates)}] {date_str}')

        # 取当天 866015.RI 成分股（约4000只，已排除北交所和ST）
        stocks = get_universe_stocks(date_str)
        if not stocks:
            print(f'  [跳过] 无宇宙数据')
            continue
        print(f'  宇宙股票数: {len(stocks)}')

        for cat_name, fetch_fn in registry:
            cache_key = f'{cat_name}/{NS}/{date_str}'
            if _cache.exists(cache_key):
                print(f'  [已缓存] {cat_name}')
                continue

            print(f'  [获取中] {cat_name} ...', end=' ', flush=True)
            try:
                df = fetch_fn(stocks, date_str)
                non_null = int(df.notna().any(axis=1).sum()) if df is not None else 0
                if df is not None and non_null > 0:
                    _cache.save(cache_key, df)
                    print(f'OK  (有效股票: {non_null})')
                else:
                    print('无数据')
            except Exception as e:
                print(f'异常: {e}')

    print(f'\n{"="*60}')
    print('完成！')


if __name__ == '__main__':
    main()
