# HS300 V1.5 Paper Trading

Local + Render deployment for the wave-framework V1.5 strategy. Runs the
strategy in pure forward-stream mode (proven bit-exact against batch
backtest by `streaming_lookahead_audit.py`) and serves a live dashboard
showing signals, trades, and the equity curve.

## What's in here

- `live_signal_engine.py` — the streaming engine. Three CLI modes:
  - `replay <YYYY-MM-DD>` — replay a single historical day from local
    parquet, log signals + trades to SQLite.
  - `backfill <START> <END>` — replay a date range.
  - `status` — print DB stats.
- `dashboard.py` — Flask web app that reads SQLite and serves a dashboard
  with auto-refresh, equity curve, recent trades, fired signals, and
  open positions.
- `requirements.txt` — Python deps for Render.
- `render.yaml` — Render Blueprint that provisions one web service
  (the dashboard) plus a daily cron (the replay engine), backed by a
  shared 1 GB persistent disk for SQLite.

## Local quickstart

```bash
cd /Users/daweizhong/Documents/projects/smart1

# 1. Backfill the entire walk-forward window so you can see the
#    historical 16-trade equity curve right away.
python3 paper_trading/live_signal_engine.py backfill 2025-12-01 2026-04-30

# 2. Check status
python3 paper_trading/live_signal_engine.py status

# 3. Start the dashboard
python3 paper_trading/dashboard.py
# Open http://localhost:5000
```

## How the engine generates signals (no look-ahead, ever)

For each trading day:

1. Load 25 prior trading days of HS300 minute data.
2. Build per-`minute_idx` z-score deques (length 20) from those prior
   days only. Today's data is NOT in the deque when today's signals
   are evaluated — today's per-minute values are appended to the deque
   AFTER its z-score is computed for that minute.
3. Walk through today's 240 minutes in chronological order. At each
   minute t:
   - compute per-stock 5/10-min returns through t
   - aggregate to HS300-weighted breadth + sector + weight-bucket +
     equal-weight projections
   - compute composite IPG_10, DirScore, Exhaustion, Bucket_Penetration
   - check the 8 signal conditions (state == EMERGING_DOWN,
     minute_idx>=30, dir5<=-0.75, dir10<=+0.45, exh10<=q90,
     ipg10>=q40, bp10<=q50)
4. If the signal fires at minute t, schedule entry at minute (t+1)
   open. Entry price is recorded only when minute (t+1) arrives.
5. Schedule exit at minute (t+31) open. Exit price is recorded only
   when minute (t+31) arrives.
6. All thresholds (`exh10_max`, `ipg10_min`, `bp10_max`) are refit on
   the prior 120 trading days only. No future data ever enters the
   threshold fit.

Verified bit-exact against the batch walk-forward by
`research/strategy_lab/streaming_lookahead_audit.py`. See
`results/.../hs300_streaming_audit_v1/` for the reconciliation.

## Live mode via akshare

The engine's `live-once` subcommand fetches today's HS300 universe and
510300 ETF minute bars from akshare (free, no auth) and streams them
through the same signal pipeline as replay mode. Verified bit-aligned
against the local backfill on 2026-04-28: same trade fires, +10.19 vs
+10.20 bps net. Akshare's ETF endpoint has a known bug (open=0 for all
bars); the data adapter patches this by substituting the prior bar's
close, which is the price at the start of the next bar in a continuous
market.

```bash
# fetch + stream the latest available trading day
python3 paper_trading/live_signal_engine.py live-once

# or fetch a specific date
python3 paper_trading/live_signal_engine.py live-once --target-date 2026-04-30 --reset-today
```

## Background scheduler

For continuous paper-trading mode, `scheduler.py` spawns a daemon thread
that runs the live cycle every 5 minutes during 09:30-15:00 Beijing time
and every 30 minutes outside market hours. The dashboard auto-spawns
this scheduler when started with `RUN_SCHEDULER=1`.

```bash
# run scheduler standalone
python3 paper_trading/scheduler.py

# or one cycle and exit
python3 paper_trading/scheduler.py --once

# integrated with dashboard (the Render-ready way):
RUN_SCHEDULER=1 python3 paper_trading/dashboard.py
```

## Deploying to Render

Render is a cloud platform with a free tier sufficient for this
dashboard + daily cron. Steps:

### 1. Push the smart1 repo to a Render-connected GitHub remote

The repo at `https://github.com/zhongdaweiai/smart1` works.

### 2. Connect the repo to Render

- Render dashboard → New → Blueprint → connect your GitHub
- Choose the `wave-framework-extensions` branch (or main once merged)
- Render reads `paper_trading/render.yaml` and creates two services:
  - `hs300-paper-trading` (web) — the dashboard
  - `hs300-daily-replay` (cron) — runs at 07:30 UTC = 15:30 CN
    every weekday (after market close)
- Both share `/var/data/paper_trading.db` on a 1 GB persistent disk.

### 3. Provision data

The cron job calls `live_signal_engine.py replay <today>` which
expects local minute parquet under `/Users/daweizhong/Documents/projects/`.
This path does NOT exist on Render. There are two ways forward.

**Option A (recommended for first month): backfill from your local
machine, then push the SQLite DB to Render's persistent disk.**

- Run `live_signal_engine.py backfill 2025-12-01 2026-04-30` locally.
- Use Render's shell access (or `rsync` via the disk mount) to copy
  `paper_trading.db` to `/var/data/paper_trading.db`.
- Disable the daily cron until a real-time data source is wired.
- The dashboard will then show the historical 16 trades.

**Option B (full live mode): wire a real-time data source adapter.**

The engine has a `DataSource` abstract class. Implement one of:

- `JoinQuantSource` (paid) — wrap `jqdata` API for real-time minute
  data. Highest quality, has the full constituent set.
- `TushareSource` (paid) — wrap Tushare Pro real-time API.
- `AkshareSource` (free) — wrap akshare's minute-bar endpoint.
  Quality may be 15-30 minute delayed.

Then set the cron schedule to fire at the end of every minute during
A-share trading hours (09:31, 09:32, ..., 11:30, 13:01, ..., 14:59).

Render free-tier limits: cron jobs can run as often as every minute
but with a 30-minute monthly compute cap on free plan. For sustained
minute-by-minute live, a paid plan ($7/month for the cron worker) is
required.

### 4. Authentication on the data source

If using JoinQuant or Tushare, set the credential as a Render secret
env var (e.g., `JQDATA_USER`, `JQDATA_PASS`, `TUSHARE_TOKEN`). The
adapter reads them at startup.

## Important caveats

- The streaming engine uses precise ETF data. Make sure
  `/Users/daweizhong/Documents/projects/ETF data core7 precise/` is the
  source. Old rounded ETF data will silently produce different results.
- The current dashboard is read-only — it does NOT submit any orders.
  This is intentional. To wire real broker execution, write a separate
  `broker_adapter.py` and call it when the engine inserts an entry/exit
  record.
- The 16-trade walk-forward is a tiny sample. Months of forward data
  are needed before treating any of these numbers as production-grade.

## Where the signal rule lives

The exact rule is in `live_signal_engine.py`'s `stream_one_day` function
under the "signal_fired" assignment. It mirrors V1.5's
`research/strategy_lab/hs300_downside_propagation_v1_5/BEST_STRATEGY.json`.

If you change the rule, change it in the BEST_STRATEGY.json first,
then mirror the change in `stream_one_day`. Then re-run the audit
script to confirm bit-exactness against the new batch run.
