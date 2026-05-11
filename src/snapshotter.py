"""
Polls Deribit + Kalshi public REST APIs at a fixed interval and writes
raw snapshots to disk for offline analysis and backtesting.

Output layout:
    data/snapshots/YYYY-MM-DD/HHMMSS.json.gz   (full raw payload, gzipped JSON)
    data/snapshots/snapshotter.log             (heartbeat + errors)

Run once (smoke test):
    .venv/bin/python -m src.snapshotter --once

Run persistent:
    .venv/bin/python -m src.snapshotter --interval 60 --out data/snapshots
    nohup .venv/bin/python -m src.snapshotter --interval 60 > /dev/null 2>&1 &

Design notes:
    * Loss-less by construction: stores raw API responses verbatim.
      Schema decisions are deferred to the offline pipeline that reads
      these files. Whatever new field Kalshi or Deribit add tomorrow is
      already captured today.
    * Each sub-fetch is independent. A failure on one endpoint does not
      drop the whole snapshot.
    * Idempotent file naming (HHMMSS within day dir). Re-running within
      the same second overwrites; otherwise appends.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
HTTP_TIMEOUT = 20

log = logging.getLogger("snapshotter")

_stop = False


def _on_signal(signum, _frame):
    global _stop
    log.info("signal %s received, will stop after current snapshot", signum)
    _stop = True


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _get(session: requests.Session, url: str, params: dict | None = None) -> dict:
    r = session.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_deribit(session: requests.Session) -> dict:
    out: dict = {}
    endpoints = [
        ("instruments_usdc", f"{DERIBIT_BASE}/public/get_instruments",
         {"currency": "USDC", "kind": "option", "expired": "false"}),
        ("book_summary_usdc", f"{DERIBIT_BASE}/public/get_book_summary_by_currency",
         {"currency": "USDC", "kind": "option"}),
        ("instruments_btc", f"{DERIBIT_BASE}/public/get_instruments",
         {"currency": "BTC", "kind": "option", "expired": "false"}),
        ("book_summary_btc", f"{DERIBIT_BASE}/public/get_book_summary_by_currency",
         {"currency": "BTC", "kind": "option"}),
        ("index_btc_usdc", f"{DERIBIT_BASE}/public/get_index_price",
         {"index_name": "btc_usdc"}),
        ("index_btc_usd", f"{DERIBIT_BASE}/public/get_index_price",
         {"index_name": "btc_usd"}),
    ]
    for key, url, params in endpoints:
        try:
            out[key] = _get(session, url, params).get("result")
        except Exception as e:
            log.warning("deribit %s fetch failed: %s", key, e)
            out[key] = {"_error": str(e)}
    return out


def fetch_kalshi(session: requests.Session) -> dict:
    out: dict = {}
    try:
        markets = []
        cursor = None
        while True:
            params = {"limit": 1000, "status": "open", "series_ticker": "KXBTC"}
            if cursor:
                params["cursor"] = cursor
            page = _get(session, f"{KALSHI_BASE}/markets", params)
            markets.extend(page.get("markets") or [])
            cursor = page.get("cursor")
            if not cursor:
                break
        out["markets_open"] = markets
    except Exception as e:
        log.warning("kalshi markets fetch failed: %s", e)
        out["markets_open"] = {"_error": str(e)}

    try:
        events = _get(
            session,
            f"{KALSHI_BASE}/events",
            {"limit": 200, "series_ticker": "KXBTC", "status": "open"},
        ).get("events", [])
        out["events_open"] = events
    except Exception as e:
        log.warning("kalshi events fetch failed: %s", e)
        out["events_open"] = {"_error": str(e)}

    return out


def take_snapshot(out_dir: Path) -> Path:
    now = _utcnow()
    payload: dict = {
        "snapshot_ts_utc": now.isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
    }

    session = requests.Session()
    session.headers.update({"User-Agent": "cross-market-arb-snapshotter/0.1"})

    payload["deribit"] = fetch_deribit(session)
    payload["kalshi"] = fetch_kalshi(session)

    day_dir = out_dir / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / (now.strftime("%H%M%S") + ".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    deribit = payload.get("deribit") or {}
    kalshi = payload.get("kalshi") or {}
    n_inst = len(deribit.get("instruments_usdc") or []) if isinstance(deribit.get("instruments_usdc"), list) else 0
    n_mkts = len(kalshi.get("markets_open") or []) if isinstance(kalshi.get("markets_open"), list) else 0
    n_evts = len(kalshi.get("events_open") or []) if isinstance(kalshi.get("events_open"), list) else 0
    sz_kb = path.stat().st_size / 1024.0
    log.info(
        "snapshot %s | deribit_usdc_inst=%d | kalshi_mkts=%d | kalshi_evts=%d | %.1f KB",
        path.name, n_inst, n_mkts, n_evts, sz_kb,
    )
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Poll Deribit + Kalshi and snapshot to disk")
    p.add_argument("--interval", type=int, default=60, help="seconds between snapshots")
    p.add_argument("--out", type=Path, default=Path("data/snapshots"))
    p.add_argument("--once", action="store_true", help="take a single snapshot and exit")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.out / "snapshotter.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.Formatter.converter = time.gmtime

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info(
        "snapshotter starting | interval=%ds | out=%s | once=%s",
        args.interval, args.out, args.once,
    )

    if args.once:
        take_snapshot(args.out)
        return

    while not _stop:
        t0 = time.monotonic()
        try:
            take_snapshot(args.out)
        except Exception:
            log.exception("snapshot iteration failed")
        elapsed = time.monotonic() - t0
        sleep_for = max(args.interval - elapsed, 1.0)
        deadline = time.monotonic() + sleep_for
        while not _stop and time.monotonic() < deadline:
            time.sleep(0.5)

    log.info("snapshotter stopped cleanly")


if __name__ == "__main__":
    main()
