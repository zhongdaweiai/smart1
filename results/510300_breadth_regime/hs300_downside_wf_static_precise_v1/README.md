# HS300 Downside Walk-Forward Static Precise V1 Results

This result folder contains the curated small outputs for the current best
HS300 downside propagation strategy.

Large parquet panels are intentionally not versioned. The key unversioned local
file produced by reproduction is:

```text
scored_panel_daily_lag.parquet
```

## Included Result Files

- `candidate_shortlist_metrics.csv`
  Human-readable comparison of the main candidate rules.
- `fixed_rule_walkforward_summary.csv`
  Fixed-rule rolling walk-forward summary for selected candidates.
- `partial_20260403_0430_summary.csv`
  Latest incomplete fold from 2026-04-03 to 2026-04-30.
- `ipg0_stop_trail_grid.csv`
  Stop/trailing-stop experiment showing fixed 30-minute hold remains best.
- `ipg0_oos_strong_oos_trades.csv`
  OOS trades for the current preferred candidate.
- `ipg0_oos_strong_oos_if_daily.csv`
  OOS IF proxy daily equity for the current preferred candidate.
- `summary.json`
  Full script summary from the strict walk-forward run.

## Preferred Candidate

`ipg0_oos_strong` is the practical preferred candidate:

```text
state == EMERGING_DOWN
minute_idx >= 30
DirScore_5 <= -0.75
DirScore_10 <= 0.45
Exhaustion_10 <= train q90
IPG_10 >= train q40
hold 30 bars
```

It is more selective than the score-selected baseline and avoided several
negative-IPG OOS trades.

## Caution

Do not annualize the OOS result as if it were a stable production estimate.
There are only 9 OOS trades in the preferred candidate. The result is promising
because it matches the propagation thesis and survived the ETF precision audit,
not because the sample is large.
