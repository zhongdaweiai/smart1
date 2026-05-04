---
name: "hs300-downside-propagation-v1-5"
description: "Continue the HS300 intraday downside propagation V1.5 line (V1 + Bucket_Penetration filter). Extend without reintroducing ETF precision, lookahead, or candidate-cherry-pick bugs."
---

# HS300 Downside Propagation V1.5 Skill

Use this skill whenever continuing the A-share HS300 intraday
propagation strategy research that uses the Bucket_Penetration filter
on top of the V1 base rule.

## First Principles

The strategy is not a generic intraday short. The money horizon is
30 minutes after a propagation signal that has visibly reached the
HS300 weight core (top weight bucket is at least as down as bottom
weight bucket).

The core edge stack:

```
EMERGING_DOWN
  + not open noise (minute_idx >= 30)
  + not over-exhausted (Exhaustion_10 < train q90)
  + IPG_10 still positive (internal pressure ahead of price)
  + Bucket_Penetration_10 negative beyond train q50  <-- V1.5 addition
```

## Non-Negotiable Data Rules

1. Use precise ETF data only:
   `/Users/daweizhong/Documents/projects/ETF data core7 precise`
2. Never use `/Users/daweizhong/Documents/projects/ETF data core7`.
3. All thresholds (Exhaustion_10, IPG_10, Bucket_Penetration_10) are
   fit on training data only. Walk-forward refits per fold.
4. Trade entry must be at minute t+1 open; exit at t+1+hold open.
5. Do not report option results unless option bid/ask and IV costs
   are modeled.
6. The candidate name `ipg0_oos_strong` was selected from a candidate
   shortlist showing inverse train-OOS rank correlation; treat its
   nominal OOS metrics as cherry-picked. Use BP-only walk-forward
   numbers as the cleanest evidence.

## Canonical Parameters

Load from:

```
research/strategy_lab/hs300_downside_propagation_v1_5/BEST_STRATEGY.json
```

Rule summary:

```
SHORT only
state == EMERGING_DOWN
minute_idx >= 30
DirScore_5  <= -0.75
DirScore_10 <= +0.45
Exhaustion_10 <= train q90  (~0.66 on initial fit)
IPG_10 >= train q40         (~0.0)
Bucket_Penetration_10 <= train q50  (~ -0.033)
enter next minute open
hold 30 bars
max 2 non-overlapping trades per day
cost 4.5 bps
```

## Reproduce

From repo root:

```bash
python3 research/strategy_lab/audit_phase14_5_robustness.py
```

This builds (or reuses) the multiproj panel and runs:
- BP filter applied to all 5 V1 candidates (cherry-pick check)
- BP-only walk-forward (cleanest evidence: 16 trades, 75% win, +17.77 bps)
- In-train held-out chunk stability check

Outputs to `results/.../hs300_phase14_5_audit_v1/`.

## How To Extend Safely

- Prefer small, interpretable changes around the current rule:
  q-threshold sweeps for BP, longer rolling walk-forward, mirror
  long side.
- Always show train, OOS, walk-forward, AND in-train held-out chunks
  separately. The strategy already had a cherry-pick scare; do not
  repeat it.
- If adding stops, prove they improve both train AND walk-forward.
  The current evidence is fixed 30-minute hold beats hard stops and
  trailing exits (V1 finding).
- Avoid expanding into a huge unconstrained grid. The 95% profit
  concentration in 6 trades makes the strategy especially overfit-prone
  if too many params are tuned.

## Required Output For Future Agents

Every serious update should write:

```
audit_grid.csv               (per-rule metrics with t-stat + Wilson CI)
walkforward_folds.csv        (per-fold summary, threshold refit per fold)
walkforward_trades.csv       (trade-level log)
walkforward_if_daily.csv     (IF equity curve)
README.md or summary notes explaining what changed
```

Never summarize only the best-looking table. Always include the
profit-concentration ratio (top-N trades as % of total bps).

## Honest Sample Size

OOS sample is 7 trades. Walk-forward sample is 16 trades. None of
these are large enough to claim "the strategy works" with confidence.
The 5-7 month forward window is the next required evidence; until at
least 25 forward trades accumulate post-2026-05, treat all numbers
as research-grade, not production-grade.
