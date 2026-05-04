# Strategy Research Plan — Index Co-movement Prediction

> ** After EVERY exploreXX.py run, IMMEDIATELY update this file before doing anything else. **

## Research Topic
**基于成分股联动（同涨同跌）的指数日内行情预测**

## Current Status
- Phase: **Phase 9 — HS300信号重设计**
- Latest Explore: 13 (COMPLETED)
- Key Insight: 🚨 测试了17种breadth信号变体×10种配置=160个组合，对真实HS300全部亏损。IC研究发现breadth_deficit有弱正IC(ICIR≈3-4)，但信号太弱(IC=0.017)无法覆盖交易成本

## Structural Pivot
- New framework document: [market_surface_wave_framework.md](./market_surface_wave_framework.md)
- Research shift:
  1. 从静态 breadth 水平，转向市场表面的速度、加速度、扩散和传导
  2. 从单一全市场截面，转向 whole-market / index-weighted / sector 三层表面
  3. 从"现在强不强"，转向"波是否正在形成、传导、未定价、未衰竭"
- Immediate next experiments:
  1. Surface kinematics: velocity / acceleration / wavefront expansion
  2. Penetration: equal-weight wave -> weighted-core confirmation
  3. Shape health: curvature / roughness / concentration / entropy

## Exploration Summary (Explore 01-06)
- Explore 01-03: 发现breadth_resid是核心alpha，ICIR=2.08，尾盘最强
- Explore 04: 基础策略OOS Sharpe 6.06
- Explore 05: 16年验证，发现波动率依赖性
- Explore 06: 扩展校准解决regime问题，16年Sharpe 3.18

### Explore 07: Maximize Returns ✅ 🚀
- **假设**: 通过创意参数搜索找到更高收益的策略变体
- **方法**: 44种策略变体的系统性搜索，维度包括：
  1. 持仓周期: 1, 2, 3, 5, 10, 20, 30, 60 bars
  2. 阈值: ±0.3σ ~ ±2.0σ
  3. 非对称阈值: 多头低门槛/空头高门槛
  4. Long-only变体
  5. 时段过滤: 尾盘/下午
  6. Re-entry: 平仓后立即再入场
  7. 信号强度缩放: 仓位 ∝ |signal| / threshold
  8. 趋势过滤: 只顺势交易
  9. 累计信号: cum_breadth_resid
  10. 组合信号
  11. Kitchen sink: 多项增强叠加
- **结果**:

  **🏆 TOP 5 策略 (OOS 100天)**:
  | 策略 | 收益 | 年化 | Sharpe | MaxDD | Calmar | 交易数 | 胜率(天) |
  |------|------|------|--------|-------|--------|--------|---------|
  | **scaled_asym** | **62.8%** | **225%** | **10.85** | **-1.8%** | **122.7** | 3017 | 80% |
  | kitchen_sink_pm_v2 | 60.3% | 214% | 9.20 | -1.4% | 150.6 | 2284 | 79% |
  | long_only_h5 | 42.3% | 135% | 7.15 | -3.0% | 44.6 | 1387 | 74% |
  | long_only_h10 | 42.2% | 135% | 6.28 | -2.8% | 48.8 | 1046 | 65% |
  | reentry_asym | 41.2% | 130% | 7.73 | -3.0% | 43.3 | 3667 | 70% |

  **关键持仓周期发现**:
  | Hold | Return | Sharpe | 判断 |
  |------|--------|--------|------|
  | 1 bar | 24.1% | 8.00 | ✅ 基线 |
  | 2 bars | 29.9% | 7.21 | ✅ **比1bar更好！** |
  | 3 bars | 20.8% | 4.97 | ✅ 可以 |
  | 5 bars | 15.7% | 2.90 | ⚠️ 开始衰减 |
  | 10 bars | -1.1% | -0.15 | ❌ 亏损 |
  | 20+ bars | 大亏 | 负 | ❌ 信号反转 |

  → **hold=2 比 hold=1 收益更高**（29.9% vs 24.1%），但Sharpe略低

  **杀手锏组合 — scaled_asym_h1**:
  ```
  signal_col = breadth_resid
  threshold_long = mean + 0.7 × std  (多头低门槛，更容易做多)
  threshold_short = mean - 1.0 × std  (空头标准门槛)
  holding_bars = 1
  scale_by_signal = True  (信号越强仓位越大)
  ```
  → **收益从24%提升到63%**，Sharpe从8.0提升到10.85！

  **为什么 scaled_asym 最好？**
  1. 非对称阈值利用了"long > short"的不对称性 (Explore 04发现)
  2. 信号缩放: 强信号时加大仓位，弱信号时小仓位 → 更高效利用alpha
  3. 效果: Sharpe 10.85, 100天中80天盈利，MaxDD仅-1.84%

  **Long-only变体的意外发现**:
  - long_only_h5: Sharpe 7.15, return 42.3% — **持仓5分钟的纯多头策略！**
  - long_only_h10: Sharpe 6.28, return 42.2% — 10分钟也行
  - 这意味着多头信号的alpha持续时间远超空头

  **Kitchen sink (下午时段+非对称+缩放+再入场)**:
  - Sharpe 9.20, MaxDD -1.42%, **Calmar 150.6** — 最高风险调整后收益

  **彻底失败的方向**:
  - th=±0.3σ: 门槛太低，噪音交易太多，亏-13.8%
  - hold≥10: 信号完全衰减，持仓越长亏越多
  - wide_th0.5 + long hold: 大亏，方向完全反转

- **关键发现**:
  1. 🏆 **信号缩放是最大的增量** — 从Sharpe 8→10.85
  2. 🏆 **非对称阈值有效** — 多头低门槛捕捉更多alpha
  3. **hold=2 > hold=1 的收益** — 额外1分钟持仓带来边际收益
  4. **Long-only h5/h10 也能年化135%** — 最适合ETF交易
  5. **hold≥10 分钟以上信号反转** — 绝对不能持仓太久
  6. **Kitchen sink的Calmar 150** — 极高的风险调整效率

### Explore 08: Real-World Futures Backtest ✅ 🔥
- **假设**: 用真实期货成本（4.5bps往返）回测，验证策略是否真正可交易
- **方法**:
  1. 成交额加权指数（接近真实市值加权）替代等权指数
  2. 股指期货IF: 每点300元, 佣金3.68bps + 滑点0.5bps = 4.5bps往返
  3. 日亏损限额 -50bps
  4. 7种策略配置对比
  5. 成本敏感性分析
  6. 4年扩展回测验证
- **结果**:

  **🚨 关键发现: 基础策略在真实成本下亏损！**
  | 策略 | 收益(100天OOS) | Sharpe | MaxDD | 日均交易 | 日均成本 |
  |------|---------------|--------|-------|---------|---------|
  | baseline_1x | **-24.0%** | -5.90 | -25.8% | 23.6 | -10.6bps |
  | scaled_baseline | -11.5% | -1.71 | -15.6% | 23.6 | -15.2bps |
  | asym_1x | -8.8% | -2.00 | -12.3% | 28.3 | -12.7bps |
  | **asym_scaled** | **+106.3%** | **7.17** | **-3.7%** | 28.3 | -18.2bps |
  | **asym_conservative_short** | **+132.1%** | **9.69** | **-2.8%** | 25.5 | -16.4bps |
  | **aggressive_asym_scaled** | **+203.5%** | **8.09** | **-3.9%** | 33.1 | -21.3bps |
  | long_only_scaled | +73.3% | 6.69 | -3.3% | 17.3 | -11.1bps |

  **🏆 最佳策略: aggressive_asym_scaled**
  ```
  threshold_long = mean + 0.5σ  (激进多头入场)
  threshold_short = mean - 1.0σ  (标准空头)
  scale_by_signal = True, max_scale = 3.0
  holding_bars = 1
  cost = 4.5 bps round-trip
  ```
  → 100天OOS: **+203.5%, Sharpe 8.09, MaxDD -3.9%**

  **🏆 最佳风险调整: asym_conservative_short**
  ```
  threshold_long = mean + 0.7σ
  threshold_short = mean - 1.5σ  (保守空头)
  scale_by_signal = True, max_scale = 3.0
  ```
  → 100天OOS: **+132.1%, Sharpe 9.69, MaxDD -2.8%**

  **成本敏感性分析**:
  - 盈亏平衡: ~8 bps (当前4.5bps有充足安全边际)
  - 2 bps: +264%, 6 bps: +155%, 10 bps: +60%

  **4年扩展回测 (rolling 250 train / 60 test)**:
  - 100%窗口盈利
  - 平均Sharpe 12.10
  - 平均年化 344%
  - 最差窗口Sharpe 4.97

- **关键发现**:
  1. 🚨 **1x仓位在4.5bps成本下亏损** — 基础策略不可交易！
  2. 🏆 **信号缩放是盈利的必要条件** — 没有缩放=亏损
  3. 🏆 **保守空头(1.5σ)显著提升Sharpe** — 空头信号弱，减少空头交易
  4. **成本盈亏平衡8bps** — 4.5bps有充足安全边际
  5. **4年扩展验证100%窗口盈利** — 策略稳健

## Strategy Evolution Path
- Phase 1 (Explore 01-03): 基础指标构建 ✅
- Phase 2 (Explore 04-05): 信号到策略 ✅
- Phase 3 (Explore 06): Regime-Aware ✅
- Phase 4 (Explore 07): 高收益策略变体 ✅
- Phase 5 (Explore 08): 实盘期货回测 ✅
- Phase 5b (Explore 09): 详细交易日志 ✅
- Phase 6 (Explore 10): 持仓+门槛优化 ✅ (⚠️ 含未来函数偏差)
- Phase 7 (Explore 11): 去除未来函数验证 ✅
- Phase 8: 长周期验证 / 参数稳定性 / 实盘实施

## Dead Ends
1. breadth_net / cohesion_momentum — 弱信号
2. dispersion日内方向 — 负IC
3. 长持仓 (≥10 bars) — 信号反转
4. 太低阈值 (≤0.3σ) — 噪音主导
5. 宽门槛+长持仓组合 — 最差策略

### Explore 10: Hold Period + Threshold Sweep ✅ 🚀🚀
- **假设**: 延长持仓 + 提高门槛能减少交易次数、降低成本拖累
- **方法**: 180种配置扫描: hold×{1,2,3,5,8} × th_long×{0.5,0.7,1.0,1.5,2.0}σ × th_short×{1.0,1.5,2.0,LO} × scale×{Yes,No}
- **结果**:

  **🚀 持仓周期效果 (L0.5 S1.0 scaled 基准)**:
  | Hold | Return | Sharpe | MaxDD | Trades/Day | AvgBps |
  |------|--------|--------|-------|-----------|--------|
  | 1 bar | +197% | 8.0 | -3.9% | 34.3 | +112 |
  | **2 bars** | **+1220%** | **13.7** | **-2.0%** | **30.7** | **+266** |
  | 3 bars | +1690% | 14.2 | -3.2% | 24.2 | +298 |
  | 5 bars | +5173% | 16.1 | -2.3% | 18.8 | +412 |
  | 8 bars | +10465% | 16.5 | -4.0% | 14.2 | +487 |

  **🏆 最佳实用配置 (hold=2)**:
  | Config | Return | Sharpe | MaxDD | T/Day |
  |--------|--------|--------|-------|-------|
  | L0.5 S1.5 sc | +2741% | 17.1 | -1.4% | 24.8 |
  | L0.5 LO sc | +2733% | 17.3 | -1.4% | 24.4 |
  | L0.7 LO sc | +497% | 14.9 | -1.3% | 16.3 |
  | L0.7 LO 1x | +195% | 14.2 | -1.0% | 16.7 |
  | L1.0 LO sc | +94% | 10.2 | -0.8% | 7.1 |
  | L1.0 LO 1x | +63% | 10.5 | -0.7% | 7.2 |

  **最佳 hold=5 配置**:
  | Config | Return | Sharpe | MaxDD | T/Day |
  |--------|--------|--------|-------|-------|
  | L0.7 LO sc | +3698% | 18.7 | -2.2% | 12.5 |
  | L0.7 LO 1x | +1211% | 20.4 | -1.7% | 13.0 |
  | L1.0 LO sc | +299% | 13.5 | -1.5% | 5.9 |

- **关键发现**:
  1. 🚀🚀 **hold=2比hold=1好5-10倍** — 这是最大的发现！之前一直用hold=1太保守了
  2. 🚀 **hold=5的1x策略Sharpe 20.4** — 最高Sharpe + 最简单执行
  3. 🏆 **Long-only效果≈S2.0** — 做空几乎没有贡献，可以完全去掉
  4. **L1.0+门槛** — 7笔/天，Sharpe仍有10+，非常适合手动/低频交易
  5. ⚠️ **L1.5+门槛** — 交易太少，不稳定
  6. 💡 **1x仓位在hold≥2时也能盈利** — 解决了Explore 08的核心问题
  7. 🚨 **hold=8返回过高(+6500%~100000%)** — 可能过拟合，需更多验证

### Explore 09: Detailed Weekly Trade Log ✅
- **目的**: 生成逐笔交易记录，展示策略实际运作方式
- **方法**: 对比OOS最佳周(1/12-16)和最差周(1/19-23)的完整交易明细
- **结果**:
  - 🏆 最佳周: 326笔, +2017 bps, ≈+236,017元/手, 胜率57%
  - 📉 最差周: 99笔, -283 bps, ≈-33,104元/手, 胜率40%
  - 单日最佳: 1/13 +960 bps (104笔, 62%胜率), ≈+112,275元/手
  - 单笔最佳: 1/13 13:52 做多3.0x → +122.6 bps ≈+14,341元
- **关键发现**:
  1. 🔑 **策略是"截断亏损+利润奔跑"模式**: 亏损日频繁但被-50bps限额截断, 盈利日少但巨大
  2. 正日率仅57%, 收益靠少数大赢日驱动 (1/13单日+960bps抵消多日亏损)
  3. 最差周每天触及-50bps损限停止, 信号无效但成本照扣
  4. 空头整体亏损: 最佳周做空-179bps, 最差周做空-241bps

### Explore 11: No-Lookahead Backtest ✅ 🔥🔥🔥
- **假设**: 修复两个未来函数偏差后，策略是否仍然有效？
- **修复内容**:
  1. **同bar执行偏差 (Critical)**: 信号在bar i计算 → 开仓移到bar i+1（而非bar i）
  2. **股票池未来信息 (Minor)**: 用前一天的top 300成交额选股（而非当天）
- **方法**: 与Explore 10相同的120种配置扫描
- **结果**:

  **🚨 Hold=1 完全死亡 — 确认同bar偏差是100%的利润来源**:
  | Config | ORIGINAL | NO LOOK-AHEAD | Δ Sharpe |
  |--------|----------|---------------|----------|
  | h1_L0.5_S1.0_sc | Ret +197%, Sh 8.0 | **Ret -4.4%, Sh -0.6** | **-8.5** |

  **🚀 Hold=2+ 策略仍然强劲 — alpha是真实的！**:
  | Config | ORIGINAL | NO LOOK-AHEAD | Δ Sharpe |
  |--------|----------|---------------|----------|
  | h2_L0.5_S1.0_sc | Sh 13.7 | **Sh 5.9** | -7.8 |
  | h5_L0.5_S1.0_sc | Sh 16.1 | **Sh 11.9** | -4.2 |
  | h5_L0.5_LO_sc | Sh 19.8 | **Sh 14.9** | -4.8 |
  | h5_L0.7_LO_1x | Sh 20.4 | **Sh 13.3** | -7.1 |
  | h8_L0.5_LO_1x | Sh - | **Sh 17.2** | - |

  **🏆 修复后 TOP 5 配置 (100天OOS)**:
  | Config | Return | Sharpe | MaxDD | WinRate | T/Day |
  |--------|--------|--------|-------|---------|-------|
  | h8_L0.5_LO_1x | +1223% | **17.16** | -3.6% | 78% | 13.3 |
  | h8_L0.5_LO_sc | +7455% | 16.68 | -3.5% | 72% | 12.2 |
  | h8_L0.5_S1.5_1x | +1158% | 16.59 | -3.6% | 77% | 13.2 |
  | h8_L0.7_LO_1x | +654% | 15.68 | -2.9% | 76% | 9.9 |
  | h5_L0.5_LO_sc | +2783% | 14.93 | -4.4% | 69% | 14.9 |

  **持仓周期效果 (L0.5 S1.0 scaled, 修复后)**:
  | Hold | Return | Sharpe | MaxDD | T/Day | AvgBps |
  |------|--------|--------|-------|-------|--------|
  | 1 bar | **-4.4%** | **-0.56** | -9.0% | 24.1 | -3.9 |
  | 2 bars | +101% | 5.91 | -4.1% | 20.7 | +71.9 |
  | 3 bars | +265% | 8.59 | -4.1% | 19.7 | +133 |
  | 5 bars | +998% | 11.88 | -4.4% | 16.5 | +248 |
  | 8 bars | +1640% | 12.70 | -3.7% | 12.3 | +296 |

  **偏差影响量化: look-ahead占比 ≈ 1/hold**:
  - hold=1: 偏差=100% (entire trade is contemporaneous)
  - hold=2: 偏差≈50% (Sharpe 13.7 → 5.9)
  - hold=5: 偏差≈20% (Sharpe 16.1 → 11.9)
  - hold=8: 偏差≈12.5% (最小影响)

- **关键发现**:
  1. 🚨🚨 **hold=1 完全虚假** — 同bar执行=100%未来信息，修复后亏损
  2. 🔥🔥 **hold=2+ alpha完全真实** — Sharpe 6-17，信号确实具有预测力
  3. 🏆 **最佳实用配置: h5_L0.7_LO_1x** — Sharpe 13.3, MaxDD仅-2.0%, 12笔/天, 不需要缩放
  4. 💡 **Long-only仍优于做空** — 纯多头表现最佳
  5. 📐 **偏差影响=1/hold** — 完美符合理论预期
  6. ⚠️ **Explore 08-10的hold=1结果全部失效** — 但hold≥2的结论基本成立

## Strategy Evolution Path
- Phase 1 (Explore 01-03): 基础指标构建 ✅
- Phase 2 (Explore 04-05): 信号到策略 ✅
- Phase 3 (Explore 06): Regime-Aware ✅
- Phase 4 (Explore 07): 高收益策略变体 ✅
- Phase 5 (Explore 08): 实盘期货回测 ✅
- Phase 5b (Explore 09): 详细交易日志 ✅
- Phase 6 (Explore 10): 持仓+门槛优化 ✅
- Phase 7 (Explore 11): 去除未来函数验证 ✅
- Phase 8 (Explore 12): 真实沪深300指数验证 ✅
- **Phase 9 (Explore 13): HS300信号重设计 ✅** ← 当前
- Phase 10: 策略重新定位 / 寻找可交易标的

### Explore 12: Real CSI 300 (HS300) Backtest ✅ 🚨🚨🚨
- **假设**: 用真实沪深300成分股+官方权重构建指数，验证策略能否用于IF期货交易
- **方法**:
  1. 从akshare获取沪深300成分股列表(300只)和官方权重(2026-02-27快照)
  2. A组: HS300成分股 + 官方权重构造指数 (≈IF期货)
  3. B组: 前一天成交额top300 + 成交额加权指数 (原方法)
  4. 23种关键配置的对比回测 (无未来函数)
- **结果**:

  **🚨🚨🚨 全军覆没 — HS300指数上ALL 23配置全部亏损！**
  | Config | HS300 Sharpe | Top300 Sharpe | Δ |
  |--------|-------------|--------------|---|
  | h5_L0.7_LO_sc | **-17.04** | +12.71 | -29.7 |
  | h8_L0.5_LO_1x | **-19.84** | +17.16 | -37.0 |
  | h5_L0.5_LO_sc | **-20.54** | +14.93 | -35.5 |
  | 最好的HS300配置 | **-5.85** | +12.73 | -18.6 |

  **分钟线收益相关性: 0.7072** — 两个"指数"在分钟级别差异很大

  **HS300 wins: 0/23 configs**

- **关键发现**:
  1. 🚨🚨🚨 **alpha仅存在于合成指数中** — 对真实沪深300指数完全无效
  2. **两个指数分钟线相关性仅0.71** — 成交额加权 top300 ≠ 市值加权 HS300
  3. **策略本质**: breadth_resid 预测的是"自己构造的合成指数"的未来走势，不是市场整体
  4. **无法直接用于IF期货交易** — 需要找到可交易的标的或重新设计信号
  5. 🔑 **回答了用户的核心问题**: 是的，股票池的选择至关重要，不是随便换一个池子都能工作

### Explore 13: HS300 Signal Redesign ✅ ❌
- **假设**: 重新设计breadth信号使其能预测真实HS300指数
- **方法**: 17种信号变体的IC研究 + 160种 signal×config 组合回测
- **信号变体**: breadth_resid_ew/cw, breadth_top30/50/bottom200, breadth_deficit,
  deficit_resid, intensity_resid, breadth_spread, spread_resid, neg_breadth, etc.
- **结果**:

  **IC Study (对HS300 forward return)**:
  | Signal | IC_1 | ICIR_1 | IC_3 | ICIR_3 | IC_8 | ICIR_8 |
  |--------|------|--------|------|--------|------|--------|
  | **breadth_deficit** | **+0.014** | **+3.0** | **+0.017** | **+4.0** | **+0.018** | **+4.1** |
  | breadth_spread | +0.012 | +2.8 | +0.015 | +3.6 | +0.015 | +3.5 |
  | deficit_resid_cw | +0.011 | +2.1 | +0.013 | +2.9 | +0.011 | +2.2 |
  | breadth_top30 | -0.008 | -1.9 | -0.015 | -3.7 | -0.019 | -4.7 |
  | breadth_top50 | -0.009 | -2.1 | -0.016 | -3.8 | -0.020 | -5.0 |

  **🚨 回测: 0/160 组合有正Sharpe**
  - 最好的结果: intensity_ew_resid h5_L0.5_LO_sc → Sharpe -7.48
  - IC虽正(0.017)但太弱，无法覆盖4.5bps×10笔/天=45bps/天的成本

- **关键发现**:
  1. 🔑 **breadth_deficit是唯一有正IC的信号** — 底部200只breadth - 顶部30只breadth
  2. **IC=0.017太弱** — 交易成本完全吞噬了微弱的alpha
  3. **顶部股票breadth有负IC** — 反转信号，不是动量信号
  4. ❌ **breadth类信号彻底无法用于HS300期货交易**
  5. 💡 **需要转向**: 更低成本的标的(ETF) 或 中小盘指数(IM/IC) 或 完全不同的信号

## Key Discoveries
1. 🚨🚨🚨 **breadth信号对HS300彻底失效**: 17种变体×10种配置=170组合全部亏损
2. 💡 **breadth_deficit有弱正IC(ICIR≈3-4)**: 底部200只stock比顶部30只上涨更多 → HS300微弱上涨
3. **但IC=0.017太弱**: 被4.5bps往返成本完全吞噬，需要极低成本或极少交易次数
4. 🚨🚨 **hold=1是虚假alpha** — 同bar执行偏差使100%利润为未来信息
5. 📐 **在合成指数上hold=2-8仍有Sharpe 6-17** — 信号对合成指数确实有预测力
6. **两个"指数"分钟级相关性仅0.71** — 成交额加权 ≠ 市值加权
7. ⚠️ **可能方向**: IM/IC期货(中小盘), ETF低成本交易, 个股配对交易, 或放弃breadth找新信号
