# HS300 Downside Propagation V1

This folder is the handoff package for the current best HS300 intraday strategy
line as of 2026-05-04.

The strategy is not a generic five-minute direction predictor. The earlier
five-minute long/short signal lost most of its edge after fixing ETF price
precision. The surviving edge is narrower and more interpretable:

> Detect an emerging downside propagation state from HS300 constituents, require
> internal pressure to still be ahead of the ETF price, then short the ETF/IF
> expression for roughly 30 minutes.

## Best Candidate

Canonical name: `hs300_downside_ipg0_v1`

Rule:

```text
state == EMERGING_DOWN
minute_idx >= 30
DirScore_5 <= -0.75
DirScore_10 <= 0.45
Exhaustion_10 <= train q90 of eligible signals
IPG_10 >= train q40 of eligible signals
enter short on next minute open
exit on open after 30 bars
max 2 non-overlapping trades per day
round-trip cost = 4.5 bps
```

For the fixed train/OOS split used in this handoff:

```text
train window: 2025-06-09 .. 2026-03-05
OOS window:   2026-03-06 .. 2026-04-30
Exhaustion_10 threshold: 0.6558249921746806
IPG_10 threshold:        0.0
```

The preferred execution model is IF futures, estimated from `510300.XSHG` ETF
returns using:

```text
initial capital = 500000 CNY
IF multiplier = 300
ETF-to-index scale = 1000
margin rate = 8%
capital utilization = 80%
```

## Why This Is The Current Best Line

The prior apparently strong five-minute strategy was contaminated by ETF close
prices rounded to two decimals. `510300.XSHG` trades at 0.001 tick increments,
so the old data created artificial zero returns and artificial jumps. After
rebuilding high-precision ETF data, the broad five-minute strategy was too thin
to cover cost.

The downside propagation rule survived the precision audit because it is more
specific:

- `EMERGING_DOWN` catches constituent-level downside synchronization.
- `Exhaustion_10` filters terminal/overdone moves.
- `IPG_10` keeps only cases where internal pressure has not yet been fully
  reflected in ETF price.
- The 30-minute hold captures propagation, while five minutes is mostly a
  confirmation horizon.

## Key Results

All numbers below use high-precision ETF data and 4.5 bps round-trip cost.

| Sample | Trades | Win Rate | Avg Net Bps | IF Ending Equity | IF Return | IF Max DD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train 2025-06-09..2026-03-05 | 38 | 65.79% | 7.36 | 628,527.91 | 25.71% | -12.34% |
| OOS 2026-03-06..2026-04-30 | 9 | 77.78% | 17.44 | 571,814.63 | 14.36% | -0.64% |
| Fixed-rule rolling 80 trading days | 20 | 70.00% | 15.22 | 639,529.03 | 27.91% | -5.93% |
| Partial live-like fold 2026-04-03..2026-04-30 | 1 | 100.00% | 10.20 | 504,374.00 | 0.87% | 0.00% |

The very small OOS trade count matters. Treat this as a promising research
candidate, not a production guarantee.

## Main Code

- `../run_hs300_downside_walkforward.py`
  Strict downside walk-forward runner and IF proxy compounding.
- `../run_hs300_phase1_5m_eval.py`
  Feature engine used to build the propagation panel.
- `../download_hs300_weights_from_joinquant.py`
  JoinQuant weight downloader used for daily HS300 weights.
- `../jq_hs300_propagation_phase1.py`
  Optional JoinQuant-side research script for producing the original Phase 1
  propagation panel inside JoinQuant.

## Reproduction

The current lightweight reproduction uses the already built high-precision
scored panels:

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

The stricter but slower path rebuilds the feature panel from local stock/ETF
minute parquet and lagged daily HS300 weights:

```bash
python3 research/strategy_lab/run_hs300_downside_walkforward.py \
  --start-date 2025-01-03 \
  --end-date 2026-04-30 \
  --oos-start 2026-03-06 \
  --output-dir results/510300_breadth_regime/hs300_downside_wf_daily_lag_precise_v1
```

The second command can be slow because it reads large all-market minute parquet
files day by day.

## Required Local Data

These datasets are intentionally not committed:

```text
/Users/daweizhong/Documents/projects/stock_data
/Users/daweizhong/Documents/projects/ETF data core7 precise
/Users/daweizhong/Documents/projects/smart1/research/strategy_lab/data/hs300_daily_weights
```

`ETF data core7 precise` is critical. Do not use the old rounded
`ETF data core7` directory for this strategy.

## Known Limitations

- Current best metrics are still based on a short OOS window.
- The handoff result uses high-precision scored panels generated from a static
  HS300 universe path. The script supports lagged daily weights, but full
  lagged-weight rebuild should be rerun before treating the result as final.
- IF transaction model is a proxy. It uses ETF returns, approximate index level,
  fixed round-trip bps cost, and integer contract sizing.
- Options are not backtested here. If mapped to options, start with put spreads,
  not naked puts, until live slippage/IV behavior is measured.

## Next Research Steps

1. Run the full lagged daily-weight rebuild for 2025-06-09..2026-04-30.
2. Continue walk-forward monitoring from 2026-05 onward without changing the
   rule after seeing each new outcome.
3. Add IF real minute data when available and compare ETF proxy vs IF execution.
4. Only after the futures proxy remains alive, test option expression with
   liquidity, bid-ask, IV, and gamma/theta costs.
