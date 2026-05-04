#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
import uuid
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
JQ_HELPER_DIR = PROJECT_ROOT / "Research" / "strategy_cowork"
if str(JQ_HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(JQ_HELPER_DIR))

from joinquant_download_daily_via_backtest import (  # noqa: E402
    CHUNK_PREFIX,
    META_PREFIX,
    fetch_all_logs,
    load_form_defaults,
    make_session,
    reconstruct_csv_text,
    submit_backtest,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download daily HS300 constituents/weights from JoinQuant.")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--index-code", default="000300.XSHG")
    p.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "smart1" / "research" / "strategy_lab" / "data" / "hs300_daily_weights"),
    )
    p.add_argument("--poll-seconds", type=float, default=3.0)
    p.add_argument("--timeout-seconds", type=float, default=1800.0)
    return p.parse_args()


def build_strategy_code(job_id: str, index_code: str) -> str:
    cfg = {"job_id": job_id, "index_code": index_code, "chunk_chars": 3500}
    cfg_json = json.dumps(cfg, ensure_ascii=True, separators=(",", ":"))
    return f"""
from jqdata import *
import base64
import gzip
import json
import pandas as pd

CONFIG = json.loads({cfg_json!r})
META_PREFIX = {META_PREFIX!r}
CHUNK_PREFIX = {CHUNK_PREFIX!r}


def initialize(context):
    set_benchmark(CONFIG['index_code'])
    log.set_level('order', 'error')


def emit_csv(day_str, csv_text):
    raw = gzip.compress(csv_text.encode('utf-8'))
    encoded = base64.b64encode(raw).decode('ascii')
    chunk_chars = int(CONFIG['chunk_chars'])
    total_parts = (len(encoded) + chunk_chars - 1) // chunk_chars
    for part_no in range(total_parts):
        payload = encoded[part_no * chunk_chars:(part_no + 1) * chunk_chars]
        log.info('%s|%s|%s|0|%s|%s|%s' % (
            CHUNK_PREFIX, CONFIG['job_id'], day_str, part_no, total_parts, payload
        ))


def after_trading_end(context):
    day_str = context.current_dt.strftime('%Y-%m-%d')
    try:
        wdf = get_index_weights(CONFIG['index_code'], date=day_str)
        source = 'index_weights'
    except Exception as exc:
        stocks = get_index_stocks(CONFIG['index_code'], date=day_str)
        wdf = pd.DataFrame({{'code': stocks, 'weight': [1.0 / len(stocks)] * len(stocks)}})
        source = 'equal_fallback'

    if wdf is None or len(wdf) == 0:
        stocks = get_index_stocks(CONFIG['index_code'], date=day_str)
        wdf = pd.DataFrame({{'code': stocks, 'weight': [1.0 / len(stocks)] * len(stocks)}})
        source = 'equal_fallback_empty'

    out = wdf.copy()
    if 'code' not in out.columns:
        if 'con_code' in out.columns:
            out = out.rename(columns={{'con_code': 'code'}})
        else:
            out = out.reset_index().rename(columns={{'index': 'code'}})
    if 'weight_pct' not in out.columns:
        if 'weight' in out.columns:
            out['weight_pct'] = out['weight'].astype(float)
        else:
            out['weight_pct'] = 0.0
    if out['weight_pct'].max() <= 1.0:
        out['weight_pct'] = out['weight_pct'] * 100.0
    out['date'] = day_str
    out['index_code'] = CONFIG['index_code']
    out['source'] = source
    cols = ['date', 'index_code', 'code', 'weight_pct', 'source']
    for c in ['display_name', 'name']:
        if c in out.columns:
            cols.append(c)
    out = out[cols].sort_values('weight_pct', ascending=False).reset_index(drop=True)
    log.info('%s|%s|%s|daily_weights|%s|%s|1' % (
        META_PREFIX, CONFIG['job_id'], day_str, CONFIG['index_code'], len(out)
    ))
    emit_csv(day_str, out.to_csv(index=False))
""".strip()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sess = make_session()
    defaults = load_form_defaults(sess)
    job_id = uuid.uuid4().hex[:16]
    code = build_strategy_code(job_id, args.index_code)
    build = submit_backtest(
        sess,
        defaults,
        code,
        args.start_date,
        args.end_date,
        "daily",
        f"codex_hs300_weights_{job_id[:8]}",
    )
    logs = fetch_all_logs(sess, build["backtestId"], args.poll_seconds, args.timeout_seconds)
    csv_text = reconstruct_csv_text(logs, job_id)

    all_path = out_dir / f"hs300_weights_{args.start_date}_{args.end_date}.csv"
    all_path.write_text(csv_text, encoding="utf-8")
    df = pd.read_csv(all_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for day, sub in df.groupby("date", sort=True):
        path = daily_dir / f"{day}.csv"
        sub.to_csv(path, index=False)
        written.append({"date": day, "rows": int(len(sub)), "path": str(path)})

    summary = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "index_code": args.index_code,
        "rows": int(len(df)),
        "days": int(df["date"].nunique()),
        "sources": {str(k): int(v) for k, v in df["source"].value_counts().items()},
        "csv_path": str(all_path),
        "daily_dir": str(daily_dir),
        "written": written,
        "backtest_detail_url": f"https://www.joinquant.com/algorithm/backtest/detail?backtestId={build['backtestId']}",
    }
    summary_path = out_dir / f"summary_{args.start_date}_{args.end_date}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
