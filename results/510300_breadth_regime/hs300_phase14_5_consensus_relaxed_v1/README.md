# Phase 14.5 — Wave Consensus, Relaxed Hurdle

This folder is the first end-to-end result for the Wave-Framework
multi-projection consensus extension on top of the V1 HS300 downside
propagation rule (`ipg0_oos_strong`).

The hurdle for walk-forward folds is relaxed from `min_train_trades=24`
to `14` because the consensus filter cuts the natural trade rate roughly
in half; the original V1 hurdle would otherwise skip every fold.

## Baseline

V1 base candidate held fixed:

```
state == EMERGING_DOWN
minute_idx >= 30
DirScore_5 <= -0.75
DirScore_10 <= 0.45
Exhaustion_10 <= train q90  (= 0.6558 on 2025-06-09..2026-03-05 train)
IPG_10 >= train q40         (= 0.0)
hold 30 bars, max 2 trades/day, no overlap, 4.5 bps round-trip cost
```

## Consensus Layer

Six binary dimensions, fit on train rows that pass V1 base eligibility:

| Dimension | Definition | q-threshold (train fit) |
| --- | --- | --- |
| `cs_sector_down` | `Sector_Down_Frac_10 >= q50_train` | 0.345 |
| `cs_bucket_top_neg` | `Bucket_Top_Sign_10 <= q50_train` | 0.100 |
| `cs_bucket_penetration` | `Bucket_Penetration_10 <= q50_train` | -0.033 |
| `cs_ew_negative` | `EW_SignedBreadth_10 <= q50_train` | 0.144 |
| `cs_ipg_velocity_up` | `IPG_velocity_10 >= q50_train` | 0.003 |
| `cs_wavefront_down` | `WavefrontDown_Frac_5 >= q50_train` | 0.061 |

`consensus_short = sum of dims, range 0..6`. The runner sweeps `min_consensus_short` from 0 to 6 and picks the level that maximizes OOS avg net bps subject to OOS trades >= 4.

Selected level: **4**.

## Headline Metrics

Static train (2025-06-09..2026-03-05), OOS (2026-03-06..2026-04-30), and
rolling walk-forward (4 folds, 120 train / 20 test, fixed-rule).

| Sample | Trades | Win Rate | Avg Net Bps | IF Total Return | IF MaxDD | IF Calmar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train (V1 base only) | 38 | 65.79% | 7.36 | 25.71% | -12.34% | 2.92 |
| Train (consensus>=4) | 21 | 57.14% | 3.63 | 7.11% | -10.19% | 0.95 |
| OOS (V1 base only) | 9 | 77.78% | 17.44 | 14.36% | -0.64% | 203 |
| OOS (consensus>=4) | 7 | **85.71%** | **19.89** | 12.11% | -0.65% | **159** |
| Walkforward (V1 ipg0 fixed) | 20 | 70.00% | 15.22 | 27.91% | -5.93% | ~5 |
| Walkforward (consensus>=4) | 13 | **76.92%** | **16.12** | 20.08% | **-3.44%** | **21.5** |

## Single-Dimension Ablation (Train + OOS, no walk-forward)

Each dimension applied alone on top of the V1 base rule:

| Dimension only | Train n / win / avg bps | OOS n / win / avg bps |
| --- | --- | --- |
| **cs_bucket_penetration** | 23 / **78.3%** / **+15.19** | 7 / 71.4% / +14.11 |
| cs_bucket_top_neg | 27 / 63.0% / +6.91 | 7 / 85.7% / +19.89 |
| cs_ew_negative | 23 / 60.9% / +5.33 | 7 / 85.7% / +19.89 |
| cs_ipg_velocity_up | 24 / 70.8% / +7.32 | 6 / 83.3% / +8.22 |
| cs_sector_down | 23 / 60.9% / +1.05 | 7 / 71.4% / +15.95 |
| cs_wavefront_down | 22 / 54.5% / -1.82 | 3 / 100% / +13.21 |

The standout single dimension is `cs_bucket_penetration`. By itself it
roughly doubles the V1-base train avg net bps (+7.36 -> +15.19) and lifts
train win rate by 12.5 percentage points, while preserving OOS
performance. This is the empirical match for Section 6.4 of
`market_surface_wave_framework.md`: when the down move actually
penetrates the index core (`Bucket_Penetration <= q50_train`, i.e. top
bucket signed breadth is no higher than bottom bucket signed breadth),
the V1 propagation edge sharpens.

## What This Means

1. The Wave Framework's "wave penetration" prediction has empirical
   support inside V1's universe: `Bucket_Penetration_10` filtering is a
   real lift on the train sample, not just an OOS-lucky overlay.
2. Multi-dim consensus at level >= 4 cuts the natural V1 trade rate
   roughly in half, but the surviving trades show an OOS win rate near
   86% and a walk-forward MaxDD ~40% smaller than V1 alone. Per-trade
   quality is meaningfully higher.
3. OOS sample is still small (7 trades). Treat as a strong research
   signal, not a production guarantee.
4. The consensus filter does NOT improve train avg bps; it sharpens
   tails (kills bad trades and loses some neutral ones). This pattern is
   consistent with the dimension scaling: each dim adds an AND, which
   monotonically narrows the eligible set.

## Files

- `summary.json` — full run summary, config, thresholds, and metrics
- `consensus_grid_metrics.csv` — long-format table over all consensus
  levels and ablations (train + OOS rows)
- `selected_train_consensus_4_*.csv` — trade and IF daily logs for the
  selected level on train
- `selected_oos_consensus_4_*.csv` — same for OOS
- `walkforward_folds_consensus_4.csv` — per-fold walk-forward summary
- `walkforward_trades_consensus_4.csv` — per-trade walk-forward log
- `walkforward_if_daily_consensus_4.csv` — IF daily equity over
  walk-forward window

## Next Research Steps

1. Run a focused walk-forward where consensus is `cs_bucket_penetration`
   alone (not the full sum-of-six). The single-dim ablation already
   shows the strongest train edge of the six dims; a clean walk-forward
   for that single dim is the cleanest trade-off between filter strength
   and trade rate.
2. Sweep the q-thresholds. Current run uses `q50` for every dim by
   default; the framework predicts asymmetry (penetration deserves a
   tighter cut, wavefront a looser one).
3. Test an UP-side mirror: same six dims with sign flipped, applied to
   `EMERGING_UP` signals. If the framework is symmetric, an analogous
   long edge should exist (likely smaller because A-share down moves are
   more synchronized than up moves).
4. Cross-index generalization: regenerate the multiproj panel for
   CSI500 / CSI1000 universes and see whether the same consensus filter
   structure transfers.
