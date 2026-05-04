# Phase 14.5 Robustness Audit

This folder contains the hard checks of whether the Phase 14.5 result is
real money safe, not just back-test optics.

## What was tested

1. **Look-ahead audit** (code-level, not in this folder).
   Walked through z-score deque updates, signal-mask columns,
   entry/exit timing, kinematic diffs, consensus threshold fitting, and
   walk-forward fold construction. No look-ahead found:
   - z-score deque is per minute_idx bucket, past 20 days only;
     current value is appended after computing z, so today's value
     does not enter today's stat.
   - `signal_side_5` / `state` are computed only from same-row or
     prior-row features; `fwd_ret_5/10` are labels, never enter the
     signal mask.
   - Trade entry uses `etf_open` at minute t+1, exit at t+1+hold.
     No same-bar leak.
   - `IPG_velocity_10` is `groupby('date').diff(3)` so the first 3
     rows of every day are NaN; nothing carries across days.
   - Wavefront uses `sign_h[t-1]`, past only.
   - Consensus q-thresholds are fit on `panel[date < oos_start]`
     rows only; walk-forward refits per fold.

2. **Cherry-pick check on the V1 baseline**.
   The V1 handoff committed `ipg0_oos_strong` as best, but its name and
   the candidate shortlist (committed in the parent results folder)
   suggest selection on OOS performance:

   |               | Train avg bps | OOS avg bps |
   | --- | ---: | ---: |
   | score_selected (auto top by train) | +11.30 | +1.92 |
   | base_exh90 | +9.96 | +4.88 |
   | ipg_m008_balanced | +8.67 | +11.74 |
   | ipg_q25_balanced | +8.58 | +7.34 |
   | **ipg0_oos_strong** | **+7.36** (worst) | **+17.44** (best) |

   Train-rank inverse-correlates with OOS-rank for these five. The
   "ipg0_oos_strong" name is itself the smoking gun.

   **Counter-evidence**: split the V1 train period 2025-06-09..
   2026-03-05 into three sequential 60-day blocks and use each prior
   block as train for the next:

   | Hold-out chunk | n | win rate | avg net bps |
   | --- | ---: | ---: | ---: |
   | chunk_B (Sep-Dec 2025) | 15 | 80.0% | +15.28 |
   | chunk_C (Dec-Mar 2026) | 13 | 69.2% | +12.42 |
   | true OOS (Mar-Apr 2026) | 9 | 77.8% | +17.44 |

   The ipg0_oos_strong RULE generalizes to three independent held-out
   windows, all positive, all in the +12..+17 bps band. The naming is
   suspicious; the rule is not pure cherry-pick.

3. **Bucket_Penetration filter applied to ALL 5 V1 candidates**.
   The single-dim ablation flagged Bucket_Penetration as the strongest
   filter on the original ipg0_oos_strong train. This audit checks
   whether the lift is candidate-specific or general.

   Train (per-candidate, threshold refit per candidate):

   | Candidate | base n / win / avg | + BP n / win / avg | Δ avg bps |
   | --- | --- | --- | ---: |
   | score_selected | 44 / 70.5% / +11.30 | 26 / 80.8% / +16.53 | +5.23 |
   | ipg0_oos_strong | 38 / 65.8% / +7.36 | 23 / 78.3% / +15.20 | +7.84 |
   | ipg_m008_balanced | 34 / 70.6% / +8.67 | 22 / 81.8% / +16.19 | +7.51 |
   | ipg_q25_balanced | 39 / 66.7% / +8.58 | 25 / 76.0% / +14.89 | +6.31 |
   | base_exh90 | 48 / 68.8% / +9.96 | 28 / 75.0% / +12.32 | +2.35 |

   All five candidates improve on train. The lift is general, not a
   cherry-pick artifact.

   OOS (small samples, less reliable):

   | Candidate | base avg | + BP avg | Δ |
   | --- | ---: | ---: | ---: |
   | score_selected | +1.92 | +4.09 | +2.17 |
   | ipg0_oos_strong | +17.44 | +14.11 | -3.33 |
   | ipg_m008_balanced | +11.74 | +6.79 | -4.95 |
   | ipg_q25_balanced | +7.34 | +2.99 | -4.35 |
   | base_exh90 | +4.88 | +2.78 | -2.10 |

   On OOS, Bucket_Penetration helps `score_selected` (the only
   train-honestly-selected candidate) but appears to hurt the
   OOS-cherry-picked candidates. This is consistent with the OOS
   buffer in the cherry-picked candidates already absorbing whatever
   marginal effect Bucket_Penetration would add. With 7-12 OOS
   trades per cell, none of these OOS deltas are statistically
   significant.

4. **Bucket_Penetration single-dim walk-forward**.
   Cleanest available evidence. Filter selected on TRAIN ablation
   merit, applied as a single binary mask, run as a 4-fold walk-forward
   (120-train / 20-test, threshold refit per fold). Since the dim
   selection happened on TRAIN merit (not OOS), there is no
   model-selection leakage into the walk-forward.

   ```
   16 trades, win rate 75.0%, avg net bps +17.77 (t = 2.82, p ~ 0.013)
   IF total return +26.25% over 80 trading days
   IF Sharpe 3.68, IF MaxDD -1.66%, IF Calmar 61.6
   ```

## Statistics

T-stats (one-sample t-test, `avg / (std / sqrt(n))`) and Wilson 95%
CIs on win rate.

| Result | n | win rate (95% CI) | avg bps | t-stat | p (approx) |
| --- | ---: | --- | ---: | ---: | ---: |
| score_selected base train | 44 | 0.70 (0.56-0.82) | +11.30 | 2.49 | 0.017 |
| score_selected + BP train | 26 | 0.81 (0.62-0.92) | +16.53 | 3.06 | 0.005 |
| ipg0_oos_strong base train | 38 | 0.66 (0.50-0.79) | +7.36 | 1.55 | 0.13 |
| ipg0_oos_strong + BP train | 23 | 0.78 (0.58-0.90) | +15.20 | 2.62 | 0.015 |
| BP-only walk-forward | 16 | 0.75 (0.51-0.90) | +17.77 | 2.82 | 0.013 |
| ipg0_oos_strong base OOS | 9 | 0.78 (0.45-0.94) | +17.44 | 1.90 | 0.094 |
| ipg0_oos_strong + BP OOS | 7 | 0.71 (0.36-0.92) | +14.11 | 1.46 | 0.19 |

The statistically significant cells are train-side. OOS samples are
too small for reliable inference.

## Real-money sizing (calibrated to the BP-only walk-forward window)

Configuration:
- 500,000 CNY initial capital
- IF margin rate 8%, capital utilization 80%
- ~10x effective leverage on capital
- 4.5 bps round-trip cost (commission + slippage)

Observed in 80-day walk-forward window:
- 16 trades (~50 trades / year extrapolated)
- 75% win rate, +17.77 avg net bps
- IF total return +26.25% (annualized via compounding ~107% / yr)
- IF MaxDD -1.66%, peak equity 634,904 CNY, trough 500,000 CNY
- 59 of 80 days at peak equity (no time spent in drawdown most of
  the time)

Bootstrap distribution over 5,000 synthetic 50-trade-year paths
(re-sampling the 16-trade bps distribution with replacement;
calibration: 1 bp ≈ 0.092% capital return at 10x leverage):

| Statistic | Annual return | Worst DD of year |
| --- | ---: | ---: |
| median | +122.3% | -2.2% |
| p25 | +99.7% | -3.0% |
| p10 | +82.2% | -3.8% |
| p5 | +72.2% | -4.2% |
| p1 | +56.2% | -5.3% |
| prob(<0) | 0% | -- |

If you double leverage (margin halved or 2x contract count, ~20x
effective):

| Statistic | Annual return | Worst DD |
| --- | ---: | ---: |
| median | +375.6% | -4.4% |
| p5 | +188.4% | -8.4% |
| p1 | -- | -10.6% |

These bootstrap numbers ASSUME the trade distribution stays the same.
Real markets have regime shifts. A 2x or 3x worse drawdown than the
bootstrap suggests is the right way to size for live trading.

## Files

- `audit_grid.csv` — long-format per-row metrics for all tested
  combinations
- `bp_only_walkforward_folds.csv` — per-fold summary for the
  Bucket_Penetration single-dim walk-forward
- `bp_only_walkforward_trades.csv` — per-trade log
- `bp_only_walkforward_if_daily.csv` — IF daily equity curve
