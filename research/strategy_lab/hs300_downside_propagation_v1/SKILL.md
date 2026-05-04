---
name: "hs300-downside-propagation"
description: "Continue the HS300 intraday downside propagation research line, reproduce the current best 30-minute short strategy, and extend it without reintroducing ETF precision or lookahead bugs."
---

# HS300 Downside Propagation Skill

Use this skill whenever continuing the A-share HS300 intraday propagation
strategy research.

## First Principles

The current strategy is not a generic five-minute predictor. Treat five-minute
signals as confirmation. The money horizon is around 30 minutes, specifically
on downside propagation.

The core edge is:

```text
EMERGING_DOWN + not open noise + not over-exhausted + IPG still positive
```

where `IPG_10` means internal constituent pressure is still ahead of ETF price.

## Non-Negotiable Data Rules

1. Use high-precision ETF data only:

   ```text
   /Users/daweizhong/Documents/projects/ETF data core7 precise
   ```

2. Do not use the old rounded ETF directory:

   ```text
   /Users/daweizhong/Documents/projects/ETF data core7
   ```

3. Any threshold used in OOS must be fit on training data only.
4. Any trade must enter no earlier than the next minute open after the signal
   bar.
5. Do not report option results unless option bid/ask and IV costs are modeled.

## Current Best Strategy

Load the canonical parameters from:

```text
research/strategy_lab/hs300_downside_propagation_v1/BEST_STRATEGY.json
```

Rule summary:

```text
SHORT only
state == EMERGING_DOWN
minute_idx >= 30
DirScore_5 <= -0.75
DirScore_10 <= 0.45
Exhaustion_10 <= train q90
IPG_10 >= train q40
enter next open
hold 30 bars
max 2 non-overlapping trades per day
cost 4.5 bps
```

## Reproduce

From repo root:

```bash
python3 research/strategy_lab/run_hs300_downside_walkforward.py \
  --input-panel-files \
results/510300_breadth_regime/hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet,results/510300_breadth_regime/hs300_phase1_5m_oos_20260306_precise_static_v1/scored_panel.parquet \
  --output-dir results/510300_breadth_regime/hs300_downside_wf_static_precise_v1 \
  --oos-start 2026-03-06 \
  --min-train-trades 24 \
  --wf-train-days 120 \
  --wf-test-days 20
```

For stricter daily-lag validation, run the same script without
`--input-panel-files`. Expect it to be slow because it rebuilds minute features
from all-market stock parquet.

## How To Extend Safely

- Prefer small, interpretable changes around the current rule:
  `IPG_10`, `Exhaustion_10`, hold length, first 30-60 minute filters, and IF
  execution assumptions.
- Keep a fixed-rule walk-forward table and a final partial fold table.
- Always show train, OOS, rolling, and latest partial fold separately.
- If adding stops, prove they improve both train and OOS. The current test found
  fixed 30-minute hold beats hard stops and trailing exits.
- Avoid expanding into a huge unconstrained grid. That will likely overfit.

## Required Output For Future Agents

Every serious update should write:

```text
candidate_shortlist_metrics.csv
fixed_rule_walkforward_summary.csv
partial_<date_range>_summary.csv
trade logs for selected candidate
README or summary notes explaining what changed
```

Never summarize only the best-looking table.
