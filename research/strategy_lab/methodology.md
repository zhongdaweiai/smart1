# Iterative Strategy Research Methodology

> Based on Knuth "Claude's Cycles" (2026) — Filip Stappers & Claude Opus 4.6

## Core Principles

### 1. Forced Documentation (plan.md)
Every exploration must update `plan.md` BEFORE starting the next one.
This prevents "forgetting" previous attempts and lessons in long sessions.

### 2. Numbered Explorations
- Each experiment is a standalone script: `explore01_xxx.py`, `explore02_xxx.py`, ...
- Scripts are self-contained and reproducible
- Old scripts are NEVER deleted — they form the research audit trail

### 3. Progressive Strategy Evolution
Typical evolution path for quant strategy research:

| Phase | Approach | Example |
|-------|----------|---------|
| 1 | Brute force / naive | 单因子全样本回测 |
| 2 | Pattern discovery | 发现某些市值区间/行业有效 |
| 3 | Framework abstraction | 构建因子组合框架 |
| 4 | Critical pivot | "回测能找到解但无法泛化，需要更深的逻辑" |
| 5 | Structural insight | 从数据模式中提炼出可解释的交易逻辑 |
| 6 | General construction | 参数化、泛化的策略构造 |

### 4. Know When to Pivot
Key signal: "SA can find solutions but cannot give a general construction. Need pure math."
Quant equivalent: "过拟合能找到好参数，但样本外失效。需要更深的逻辑框架。"

### 5. Role Division
- **Human (Coach)**: 提出研究方向、约束条件、在陷入局部最优时重新引导、最终验证
- **AI (Explorer)**: 执行探索、记录进展、自我反思、策略转换、代码实现

### 6. Self-Reflection Triggers
When to step back and rethink:
- 连续3次探索无显著改善
- 样本内表现好但样本外崩溃
- 策略逻辑无法给出直觉解释
- 参数敏感性过高

## Workflow Per Exploration

```
1. State hypothesis clearly
2. Write explore script (explore{NN}_{topic}.py)
3. Run experiment, collect results
4. Update plan.md IMMEDIATELY:
   - What was tried
   - What the results were
   - Why it succeeded/failed
   - What to try next
5. Reflect: pivot strategy if needed
6. Proceed to next exploration
```

## File Structure

```
Research/strategy_lab/
├── plan.md                    # Central progress tracker (ALWAYS update first)
├── methodology.md             # This file
├── explore01_xxx.py           # Exploration scripts
├── explore02_xxx.py
├── ...
├── results/                   # Output data, charts, metrics
│   ├── explore01/
│   ├── explore02/
│   └── ...
└── utils/                     # Shared utility functions
```
