#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask dashboard for HS300 V1.5 paper trading.

Reads paper_trading.db and serves a single-page dashboard plus JSON APIs.

Routes:
  GET /                       — main dashboard (HTML)
  GET /api/trades.json        — all trades, newest first
  GET /api/signals.json       — recent signals, newest first
  GET /api/equity.json        — daily equity curve (cumulative net bps)
  GET /api/status.json        — engine run log
  GET /api/today.json         — today's signal/trade summary
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("PAPER_TRADING_DB", REPO_ROOT / "paper_trading" / "paper_trading.db"))

app = Flask(__name__)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- API ----------------

@app.route("/api/trades.json")
def api_trades():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, date, signal_ts, entry_ts, exit_ts, entry_minute_idx, exit_minute_idx, "
        "side, entry_px, exit_px, gross_bps, net_bps, status FROM trades ORDER BY signal_ts DESC LIMIT 200"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/signals.json")
def api_signals():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, ts, date, minute_idx, state, DirScore_5, DirScore_10, Exhaustion_10, "
        "IPG_10, Bucket_Penetration_10, fired, skipped_reason FROM signals "
        "WHERE fired=1 OR state IN ('EMERGING_DOWN','CONFIRMED_DOWN') "
        "ORDER BY ts DESC LIMIT 200"
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/equity.json")
def api_equity():
    """Cumulative net bps across all closed trades, ordered chronologically."""
    conn = get_db()
    rows = conn.execute(
        "SELECT date, signal_ts, net_bps FROM trades WHERE status='CLOSED' ORDER BY signal_ts ASC"
    ).fetchall()
    conn.close()
    cum = 0.0
    out = []
    for r in rows:
        cum += float(r["net_bps"] or 0.0)
        out.append({"signal_ts": r["signal_ts"], "date": r["date"], "net_bps": float(r["net_bps"] or 0), "cum_bps": cum})
    return jsonify(out)


@app.route("/api/status.json")
def api_status():
    conn = get_db()
    n_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    n_fired = conn.execute("SELECT COUNT(*) FROM signals WHERE fired=1").fetchone()[0]
    n_trades_total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    n_open = conn.execute("SELECT COUNT(*) FROM trades WHERE status='OPEN'").fetchone()[0]
    n_closed = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED'").fetchone()[0]
    wins = conn.execute("SELECT COUNT(*) FROM trades WHERE status='CLOSED' AND net_bps > 0").fetchone()[0]
    total_bps = conn.execute("SELECT COALESCE(SUM(net_bps),0) FROM trades WHERE status='CLOSED'").fetchone()[0]
    avg_bps = conn.execute("SELECT COALESCE(AVG(net_bps),0) FROM trades WHERE status='CLOSED'").fetchone()[0]
    last_run = conn.execute("SELECT ts, date, mode, message FROM run_log ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return jsonify({
        "n_signals": n_signals,
        "n_fired": n_fired,
        "n_trades_total": n_trades_total,
        "n_open": n_open,
        "n_closed": n_closed,
        "wins": wins,
        "win_rate": (wins / n_closed) if n_closed else None,
        "total_net_bps": total_bps,
        "avg_net_bps": avg_bps,
        "last_run": dict(last_run) if last_run else None,
    })


@app.route("/api/today.json")
def api_today():
    conn = get_db()
    today = conn.execute("SELECT MAX(date) FROM signals").fetchone()[0]
    if not today:
        conn.close()
        return jsonify({"date": None, "signals_today": 0, "fired_today": 0})
    sigs = conn.execute("SELECT * FROM signals WHERE date=? ORDER BY ts ASC", (today,)).fetchall()
    trs = conn.execute("SELECT * FROM trades WHERE date=? ORDER BY signal_ts ASC", (today,)).fetchall()
    conn.close()
    return jsonify({
        "date": today,
        "signals_today": len(sigs),
        "fired_today": sum(1 for s in sigs if s["fired"]),
        "trades_today": [dict(t) for t in trs],
        "fired_signals": [dict(s) for s in sigs if s["fired"]],
    })


# ---------------- HTML ----------------

TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>HS300 V1.5 Paper Trading</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #0f1419; color: #d4d4d4; margin: 0; padding: 20px; }
  h1 { color: #ffffff; margin-bottom: 4px; }
  .subtitle { color: #888; margin-bottom: 20px; font-size: 13px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .card { background: #1a2027; border: 1px solid #2a3038; border-radius: 6px; padding: 16px; }
  .card h2 { font-size: 14px; color: #888; margin: 0 0 10px 0; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat { font-size: 28px; font-weight: 600; }
  .stat-row { display: flex; gap: 24px; margin-bottom: 12px; }
  .stat-item { flex: 1; }
  .stat-label { font-size: 11px; color: #888; text-transform: uppercase; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a3038; color: #888; font-weight: 500; text-transform: uppercase; font-size: 11px; }
  td { padding: 8px 10px; border-bottom: 1px solid #1f2530; }
  tr:hover td { background: #1f2530; }
  .pos { color: #4ec9b0; }
  .neg { color: #f48771; }
  .open { color: #dcdcaa; }
  .closed { color: #9cdcfe; }
  pre { font-size: 11px; color: #888; white-space: pre-wrap; }
  #equity-chart { width: 100%; height: 300px; }
  .full { grid-column: 1 / -1; }
  .small { font-size: 11px; color: #888; }
  .big-pos { color: #4ec9b0; font-weight: 600; }
  .big-neg { color: #f48771; font-weight: 600; }
</style>
</head>
<body>
<h1>HS300 V1.5 Paper Trading</h1>
<div class="subtitle">短期下跌传导策略 · 实盘模拟 · 自动刷新 30 秒</div>

<div class="stat-row">
  <div class="card stat-item">
    <div class="stat-label">已平仓交易</div>
    <div class="stat" id="n-closed">—</div>
    <div class="small">持仓中: <span id="n-open">—</span></div>
  </div>
  <div class="card stat-item">
    <div class="stat-label">胜率</div>
    <div class="stat" id="win-rate">—</div>
    <div class="small">赢: <span id="wins">—</span></div>
  </div>
  <div class="card stat-item">
    <div class="stat-label">累计 Net Bps</div>
    <div class="stat" id="total-bps">—</div>
    <div class="small">平均: <span id="avg-bps">—</span> /笔</div>
  </div>
  <div class="card stat-item">
    <div class="stat-label">最近运行</div>
    <div class="stat" style="font-size: 14px;" id="last-run">—</div>
  </div>
</div>

<div class="grid">
  <div class="card full">
    <h2>累计权益曲线 (Net Bps)</h2>
    <canvas id="equity-chart"></canvas>
  </div>

  <div class="card">
    <h2>已平仓交易（最近 30 笔）</h2>
    <div style="max-height: 500px; overflow-y: auto;">
      <table>
        <thead>
          <tr><th>日期</th><th>入场</th><th>出场</th><th>方向</th><th>Net Bps</th></tr>
        </thead>
        <tbody id="trades-tbody"></tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>触发的信号（最近 30 个）</h2>
    <div style="max-height: 500px; overflow-y: auto;">
      <table>
        <thead>
          <tr><th>时间</th><th>State</th><th>Dir5/10</th><th>IPG10</th><th>BP10</th></tr>
        </thead>
        <tbody id="signals-tbody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="card full" style="margin-top: 20px;">
  <h2>持仓中</h2>
  <table>
    <thead>
      <tr><th>日期</th><th>信号时刻</th><th>入场时刻</th><th>入场价</th><th>方向</th><th>预计出场</th></tr>
    </thead>
    <tbody id="open-tbody"></tbody>
  </table>
</div>

<script>
async function fetchJson(url) {
  const r = await fetch(url);
  return await r.json();
}

function fmtBps(b) {
  if (b == null) return '—';
  const cls = b >= 0 ? 'pos' : 'neg';
  const sign = b >= 0 ? '+' : '';
  return `<span class="${cls}">${sign}${b.toFixed(2)}</span>`;
}

function fmt(n, d=2) {
  if (n == null) return '—';
  return Number(n).toFixed(d);
}

async function refresh() {
  const status = await fetchJson('/api/status.json');
  document.getElementById('n-closed').textContent = status.n_closed;
  document.getElementById('n-open').textContent = status.n_open;
  document.getElementById('win-rate').textContent = status.win_rate != null ? (status.win_rate*100).toFixed(1) + '%' : '—';
  document.getElementById('wins').textContent = status.wins;
  const bps = status.total_net_bps;
  document.getElementById('total-bps').innerHTML = fmtBps(bps);
  document.getElementById('avg-bps').innerHTML = fmtBps(status.avg_net_bps);
  if (status.last_run) {
    document.getElementById('last-run').textContent = `${status.last_run.date} · ${status.last_run.mode}`;
  }

  const trades = await fetchJson('/api/trades.json');
  const tbody = document.getElementById('trades-tbody');
  const otbody = document.getElementById('open-tbody');
  tbody.innerHTML = '';
  otbody.innerHTML = '';
  let closed_count = 0;
  for (const t of trades) {
    if (t.status === 'CLOSED' && closed_count < 30) {
      tbody.insertAdjacentHTML('beforeend',
        `<tr><td>${t.date}</td><td>${t.entry_ts ? t.entry_ts.split('T')[1] : '—'}</td>` +
        `<td>${t.exit_ts ? t.exit_ts.split('T')[1] : '—'}</td>` +
        `<td>${t.side}</td><td>${fmtBps(t.net_bps)}</td></tr>`);
      closed_count++;
    } else if (t.status === 'OPEN') {
      otbody.insertAdjacentHTML('beforeend',
        `<tr><td>${t.date}</td><td>${t.signal_ts ? t.signal_ts.split('T')[1] : '—'}</td>` +
        `<td>${t.entry_ts ? t.entry_ts.split('T')[1] : '—'}</td>` +
        `<td>${fmt(t.entry_px, 4)}</td><td>${t.side}</td><td>min ${t.exit_minute_idx}</td></tr>`);
    }
  }

  const sigs = await fetchJson('/api/signals.json');
  const sbody = document.getElementById('signals-tbody');
  sbody.innerHTML = '';
  for (const s of sigs.slice(0, 30)) {
    const tag = s.fired ? '<span class="pos">✓</span>' : '';
    sbody.insertAdjacentHTML('beforeend',
      `<tr><td>${s.ts ? s.ts.replace('T', ' ') : '—'} ${tag}</td>` +
      `<td>${s.state}</td>` +
      `<td>${fmt(s.DirScore_5)}/${fmt(s.DirScore_10)}</td>` +
      `<td>${fmt(s.IPG_10)}</td>` +
      `<td>${fmt(s.Bucket_Penetration_10, 3)}</td></tr>`);
  }

  drawEquityCurve();
}

async function drawEquityCurve() {
  const eq = await fetchJson('/api/equity.json');
  const c = document.getElementById('equity-chart');
  const ctx = c.getContext('2d');
  c.width = c.clientWidth * window.devicePixelRatio;
  c.height = c.clientHeight * window.devicePixelRatio;
  ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  ctx.fillStyle = '#0f1419';
  ctx.fillRect(0, 0, c.clientWidth, c.clientHeight);
  if (eq.length === 0) return;
  const w = c.clientWidth;
  const h = c.clientHeight;
  const padding = 40;
  const xs = eq.map((_, i) => i);
  const ys = eq.map(d => d.cum_bps);
  const ymin = Math.min(0, ...ys);
  const ymax = Math.max(...ys);
  const yrange = (ymax - ymin) || 1;
  const xscale = i => padding + (i / (eq.length - 1 || 1)) * (w - 2 * padding);
  const yscale = y => h - padding - ((y - ymin) / yrange) * (h - 2 * padding);

  // zero line
  ctx.strokeStyle = '#444'; ctx.lineWidth = 1;
  ctx.beginPath();
  const yzero = yscale(0);
  ctx.moveTo(padding, yzero); ctx.lineTo(w - padding, yzero); ctx.stroke();

  // line
  ctx.strokeStyle = '#4ec9b0'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < eq.length; i++) {
    const x = xscale(i), y = yscale(ys[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // axis labels
  ctx.fillStyle = '#888'; ctx.font = '11px sans-serif';
  ctx.fillText(ymax.toFixed(0) + ' bps', 4, 14);
  ctx.fillText(ymin.toFixed(0) + ' bps', 4, h - padding + 4);
  ctx.fillText(`${eq.length} trades`, w - 100, h - 6);
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE)


def maybe_start_scheduler():
    """If RUN_SCHEDULER=1, spawn the akshare-driven background scheduler.
    On Render this will be enabled to refresh signals every 5 min during
    market hours."""
    if os.environ.get("RUN_SCHEDULER", "0") in ("1", "true", "yes"):
        try:
            from scheduler import start_in_background
            start_in_background(DB_PATH)
        except Exception as exc:
            print(f"[dashboard] failed to start scheduler: {exc}")


# Start scheduler on import (so it works under gunicorn too)
maybe_start_scheduler()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
