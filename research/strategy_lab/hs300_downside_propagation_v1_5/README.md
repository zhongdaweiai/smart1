# HS300 Downside Propagation V1.5 — BP-Filter Edition

This folder is the canonical handoff for the Phase 14.5 strategy that
extends V1 with the Wave-Framework Bucket-Penetration filter.

It is intentionally written so you can hand it to a new agent or a new
analyst tomorrow and they can rebuild the strategy from raw data without
this conversation.

## TL;DR

A short-only intraday HS300 strategy. After 10:00am, every minute, check
8 conditions on cross-sectional HS300 features. When all 8 fire, short
510300 ETF (or equivalent IF futures) at the next minute's open, hold
exactly 30 minutes, then exit at open.

```
state == EMERGING_DOWN
minute_idx >= 30
regular trading hours (no first 5min, no lunch, no last 10min)
DirScore_5  <= -0.75
DirScore_10 <= +0.45
Exhaustion_10 <= train q90
IPG_10 >= train q40
Bucket_Penetration_10 <= train q50    <-- this is what V1.5 adds
```

Walk-forward 80 trading days: 16 trades, 75% win rate, +17.77 avg net
bps, t = 2.82 (p ≈ 0.013), IF MaxDD -1.66%, Calmar 61.6.

## What V1.5 changes from V1

V1 (`hs300_downside_ipg0_v1`) projects the wave only through HS300 cap
weights. The Wave Framework (`market_surface_wave_framework.md`) Sec. 12
says a useful wave should be visible across several projections.

V1.5 adds **one** projection-based gate:

```
Bucket_Penetration_10 = signed_breadth(top weight bucket)
                      - signed_breadth(bottom weight bucket)
fire only if Bucket_Penetration_10 <= train_q50  (≈ -0.033 on initial fit)
```

Plain meaning: only fire when the down move has reached the largest
weight stocks at least as much as the smaller ones. If only the tail is
crashing, do not trade — that is not a real index-relevant wave.

**Empirical proof this is general, not cherry-picked:** apply the BP
filter on top of all 5 V1 candidate rules, threshold refit per
candidate. All 5 improve on train (avg net bps lift +2.4 to +7.8).
This rules out "BP filter only helps the cherry-picked candidate"
artifact.

## How to recompute every threshold from scratch

Two parquets are pre-built and live in the parent results directory:

```
results/510300_breadth_regime/hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet
results/510300_breadth_regime/hs300_phase1_5m_oos_20260306_precise_static_v1/scored_panel.parquet
```

These contain V1's score panel (state, DirScore, Exhaustion, IPG, etc.).

The multi-projection panel (Bucket_Penetration_10, etc.) is rebuilt by:

```
cd /Users/daweizhong/Documents/projects/smart1
python3 research/strategy_lab/compute_multiproj_features.py \
  --start-date 2025-06-09 --end-date 2026-04-30 \
  --output-dir results/510300_breadth_regime/hs300_multiproj_panel_v1
```

This reads stock_data/ and ETF data core7 precise/ day-by-day, builds
the universe from lagged daily HS300 weights, and emits
`multiproj_panel.parquet`. About 3 minutes wall time.

The walk-forward run that produces the headline 16-trade result:

```
python3 research/strategy_lab/audit_phase14_5_robustness.py
```

This audit script (a) tests the BP filter on all 5 V1 candidates,
(b) runs the BP-only walk-forward, (c) does in-train held-out chunk
checks. Output goes to `results/.../hs300_phase14_5_audit_v1/`.

For the Phase 14.5 consensus-of-six runner (different rule, sweeps over
how many of six dims must agree):

```
python3 research/strategy_lab/run_phase14_5_consensus_walkforward.py \
  --input-panel-files results/.../hs300_phase1_5m_180d_precise_static_v1/scored_panel.parquet,results/.../hs300_phase1_5m_oos_20260306_precise_static_v1/scored_panel.parquet \
  --multiproj-panel results/.../hs300_multiproj_panel_v1/multiproj_panel.parquet \
  --output-dir results/.../hs300_phase14_5_consensus_relaxed_v1 \
  --oos-start 2026-03-06 --min-train-trades 14 --wf-train-days 120 --wf-test-days 20
```

## Order mechanics, in plain language

Every trading minute, we already know everything from this minute and
earlier (no lookahead). We compute features at minute t. If all 8
conditions in the signal rule are true:

1. Place a SHORT market-on-open order for minute (t+1).
2. Position size: in IF, `n_contracts = floor(equity * 0.80 / (index_pt * 300 * 0.08))`. At 500K equity and index near 4500, this is 3 contracts.
3. Hold for 30 minutes.
4. Place a BUY market-on-open order for minute (t+31) to close.
5. Track P&L per trade as `-log(exit_open / entry_open) * 10000` bps gross, then subtract 4.5 bps for round-trip cost.
6. Constraint: max 2 trades per day, no overlap (next signal must be at minute > exit_minute).

The strategy never holds across:
- the lunch break (signal grid only fires inside trading hours and
  exit_pos must stay inside the day)
- overnight (max 2 same-day trades, exit must complete before close)
- futures roll (irrelevant for 30-minute intraday holds)

## Honest profit profile (16-trade walk-forward)

| # | Date | Time | DirScore_5/10 | IPG_10 | Bucket_Pen | Net bps |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 2025-12-22 | 10:25 | -0.82 / -0.02 | +0.75 | -0.70 | +3.97 |
| 2 | 2025-12-22 | 14:26 | -1.77 / -1.49 | +0.88 | -0.27 | +1.84 |
| 3 | 2025-12-23 | 13:57 | -1.12 / -0.13 | +2.23 | -0.17 | +6.05 |
| 4 | **2026-01-13** | **10:03** | -0.95 / -0.20 | +0.04 | -0.35 | **+56.38** |
| 5 | 2026-01-20 | 13:52 | -1.17 / -0.83 | +0.09 | -0.10 | +18.79 |
| 6 | **2026-01-22** | **10:07** | -1.22 / -1.64 | +0.58 | -0.55 | **+63.15** |
| 7 | 2026-01-27 | 11:00 | -0.80 / -0.26 | +0.51 | -0.06 | +12.42 |
| 8 | 2026-02-10 | 10:08 | -0.76 / -0.49 | +0.05 | -0.25 | -0.27 |
| 9 | 2026-03-03 | 10:58 | -0.77 / +0.41 | +0.38 | -0.38 | -6.61 |
| 10 | **2026-03-03** | **13:08** | -0.77 / -0.10 | +0.66 | -0.15 | **+39.99** |
| 11 | 2026-03-06 | 10:33 | -0.79 / -0.54 | +0.41 | -0.20 | -17.38 |
| 12 | 2026-03-11 | 11:04 | -0.79 / +0.29 | +2.64 | -0.09 | +14.61 |
| 13 | 2026-03-20 | 10:31 | -0.87 / -0.39 | +1.80 | -0.54 | +6.35 |
| 14 | **2026-03-23** | **10:20** | -0.98 / -0.15 | +1.25 | -0.28 | **+59.97** |
| 15 | 2026-03-23 | 14:20 | -1.03 / -0.00 | +1.31 | -0.37 | +31.71 |
| 16 | 2026-04-01 | 10:35 | -0.88 / -0.02 | +0.47 | -0.62 | -6.71 |

**Profit concentration: top 6 trades = +269.98 bps = 95.0% of total.
Other 10 trades net only +14.27 bps.**

Best day: 2026-03-23, +49,319 CNY (+9.86% of starting capital, 2 trades
in one day). Worst day: 2026-03-06, -9,709 CNY (-1.66% of peak equity,
trade #11). The 80-day MaxDD of -1.66% is exactly the bracket from
#11 alone — there is no other meaningful drawdown episode.

## What this profile means for live trading

The strategy is fat-tailed, not steady. It will almost always look
flat-ish for stretches of weeks, then have one big day. If you put it
live and your first month has 4 trades all under +5 bps each, that
is not a failure — that is the median month.

Live performance evaluation should require:
- at least 25 trades (~6 months of forward data) before any "the
  strategy works" or "the strategy fails" call
- inclusion of at least one >40 bps single trade for the strategy to
  count as confirmed; the absence of any large winner over 6 months
  is evidence the regime has shifted

## What was checked for look-ahead

`results/.../hs300_phase14_5_audit_v1/README.md` lists the full audit.
Summary:

- Z-score deque is per minute_idx, past 20 days only; current value
  appended after computing z. No same-bar leak.
- `signal_side_5` and `state` use only same-row or prior-row features.
- `fwd_ret_5/10` are labels, never enter the signal mask.
- Trade entry uses `etf_open` at minute t+1, exit at t+1+hold.
- Kinematics use `groupby(date).diff(lag)`, NaN for first lag rows of
  every day; nothing crosses day boundaries.
- Wavefront uses `sign[t-1]`, past only.
- Consensus q-thresholds fit on `panel[date < oos_start]` only.
- Walk-forward refits thresholds per fold.

No look-ahead found.

## Open questions for future research

These are NOT fixed; they are the next legitimate experiments:

1. **BP-only walk-forward with longer windows.** Current 4 folds
   covers 80 days. Add 5-month forward as it accumulates from
   2026-05 onward.
2. **q-threshold sensitivity sweep.** Current uses q50 for BP cut.
   Test q30, q40, q60. The framework predicts more aggressive cuts
   for penetration than for other dims.
3. **Long mirror.** EMERGING_UP has ~6x more candidate minutes than
   EMERGING_DOWN. A long-side BP filter (sign-flipped) is the
   natural symmetry test.
4. **Cross-index port to CSI500/CSI1000.** Build the same
   compute_multiproj_features pipeline on a different universe.
   If the BP filter survives, this is universal Wave-Framework
   evidence; if it dies, the edge is HS300-specific.
5. **Real IF minute data substitution.** Current uses ETF returns
   mapped to IF via index proxy. With real IF minute data, basis
   and futures-specific noise can be measured.

## Lineage

- Phase 1: HS300 propagation feature engine (`run_hs300_phase1_5m_eval.py`)
- Phase 14: V1 ipg0_oos_strong rule (`hs300_downside_propagation_v1/`)
- Phase 14.5: this folder, BP filter + consensus walk-forward
- Wave Framework theoretical doc: `../market_surface_wave_framework.md`

The named candidate `hs300_downside_propagation_v1_5_bp_filter` is what
this folder freezes; the canonical parameters are in `BEST_STRATEGY.json`
in this folder.
