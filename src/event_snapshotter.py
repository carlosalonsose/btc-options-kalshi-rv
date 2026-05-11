"""
Event-driven REST snapshotter.

This collector polls Deribit + Kalshi at a configurable cadence, but only writes
a snapshot when the quote state changes. It keeps the same raw payload schema as
src.snapshotter, so existing loaders/backtests can consume the output directory.

It is "event-driven" at the persistence layer: REST APIs are still polled, but
disk writes are triggered by bid/ask/size changes instead of by wall-clock time.
For true exchange event timing, the next step would be WebSocket order books.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from src.snapshotter import fetch_deribit, fetch_kalshi

log = logging.getLogger("event_snapshotter")

_stop = False


KALSHI_QUOTE_FIELDS = (
    "ticker",
    "event_ticker",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "no_bid_dollars",
    "no_ask_dollars",
    "yes_bid_size_fp",
    "yes_ask_size_fp",
    "volume_fp",
    "volume_24h_fp",
    "open_interest_fp",
    "status",
)

DERIBIT_QUOTE_FIELDS = (
    "instrument_name",
    "bid_price",
    "ask_price",
    "bid_iv",
    "ask_iv",
)

DERIBIT_INSTRUMENT_FIELDS = (
    "instrument_name",
    "expiration_timestamp",
    "strike",
    "option_type",
)


def _on_signal(signum, _frame) -> None:
    global _stop
    log.info("signal %s received, will stop after current poll", signum)
    _stop = True


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def canonical_quote_state(payload: dict[str, Any], watch: str = "both") -> dict[str, Any]:
    """Return the minimal quote state used to decide whether to write a snapshot.

    The state intentionally ignores mark prices, underlying prices, and other
    fast-moving diagnostics that would cause nearly every poll to be written.
    """
    state: dict[str, Any] = {}

    if watch in ("kalshi", "both"):
        kalshi = payload.get("kalshi") or {}
        markets = kalshi.get("markets_open") or []
        events = kalshi.get("events_open") or []
        state["kalshi"] = {
            "markets": _project_sorted(markets, KALSHI_QUOTE_FIELDS, "ticker"),
            "events": _project_sorted(events, ("event_ticker", "status"), "event_ticker"),
        }

    if watch in ("deribit", "both"):
        deribit = payload.get("deribit") or {}
        state["deribit"] = {
            "book_summary_usdc": _project_sorted(
                deribit.get("book_summary_usdc") or [],
                DERIBIT_QUOTE_FIELDS,
                "instrument_name",
            ),
            "book_summary_btc": _project_sorted(
                deribit.get("book_summary_btc") or [],
                DERIBIT_QUOTE_FIELDS,
                "instrument_name",
            ),
            "instruments_usdc": _project_sorted(
                deribit.get("instruments_usdc") or [],
                DERIBIT_INSTRUMENT_FIELDS,
                "instrument_name",
            ),
            "instruments_btc": _project_sorted(
                deribit.get("instruments_btc") or [],
                DERIBIT_INSTRUMENT_FIELDS,
                "instrument_name",
            ),
        }

    return state


def state_hash(state: dict[str, Any]) -> str:
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def side_hashes(state: dict[str, Any]) -> dict[str, str]:
    return {side: state_hash({side: value}) for side, value in state.items()}


def changed_sides(prev: dict[str, str] | None, current: dict[str, str]) -> list[str]:
    if prev is None:
        return sorted(current)
    return sorted(side for side, digest in current.items() if prev.get(side) != digest)


def take_event_snapshot(
    out_dir: Path,
    session: requests.Session,
    watch: str,
    prev_hash: str | None,
    prev_side_hashes: dict[str, str] | None,
    last_write_monotonic: float | None,
    heartbeat_seconds: float,
    min_write_interval: float,
) -> tuple[str, dict[str, str], bool, str, Path | None]:
    """Poll APIs once and write only if the quote state changed or heartbeat is due."""
    now = _utcnow()
    payload: dict[str, Any] = {
        "snapshot_ts_utc": now.isoformat().replace("+00:00", "Z"),
        "schema_version": 1,
    }
    payload["deribit"] = fetch_deribit(session)
    payload["kalshi"] = fetch_kalshi(session)

    state = canonical_quote_state(payload, watch=watch)
    digest = state_hash(state)
    sides = side_hashes(state)
    side_changes = changed_sides(prev_side_hashes, sides)

    monotonic_now = time.monotonic()
    first = prev_hash is None
    changed = digest != prev_hash
    heartbeat_due = (
        heartbeat_seconds > 0
        and last_write_monotonic is not None
        and monotonic_now - last_write_monotonic >= heartbeat_seconds
    )
    min_interval_ok = (
        last_write_monotonic is None
        or monotonic_now - last_write_monotonic >= min_write_interval
    )

    should_write = first or heartbeat_due or (changed and min_interval_ok)
    trigger = "first" if first else "heartbeat" if heartbeat_due else "change" if changed else "unchanged"

    path = None
    if should_write:
        payload["event_snapshotter"] = {
            "capture_mode": "event_driven_rest_diff",
            "watch": watch,
            "trigger": trigger,
            "state_hash": digest,
            "previous_state_hash": prev_hash,
            "changed_sides": side_changes if trigger != "heartbeat" else [],
            "heartbeat_seconds": heartbeat_seconds,
            "min_write_interval_seconds": min_write_interval,
        }
        path = write_snapshot(out_dir, payload, now)

    return digest, sides, should_write, trigger, path


def write_snapshot(out_dir: Path, payload: dict[str, Any], now: datetime) -> Path:
    day_dir = out_dir / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / (now.strftime("%H%M%S_%f") + ".json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    return path


def _project_sorted(rows: Any, fields: tuple[str, ...], sort_key: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({field: row.get(field) for field in fields if field in row})
    return sorted(out, key=lambda row: str(row.get(sort_key, "")))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Poll Deribit + Kalshi and write only quote-state changes")
    p.add_argument("--poll-interval", type=float, default=10.0, help="seconds between REST polls")
    p.add_argument("--out", type=Path, default=Path("data/event_snapshots"))
    p.add_argument("--watch", choices=("kalshi", "deribit", "both"), default="both")
    p.add_argument("--heartbeat-seconds", type=float, default=300.0,
                   help="write a heartbeat snapshot even if quotes did not change; 0 disables")
    p.add_argument("--min-write-interval", type=float, default=1.0,
                   help="minimum seconds between writes when quotes change")
    p.add_argument("--once", action="store_true", help="poll once and write one snapshot")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(args.out / "event_snapshotter.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.Formatter.converter = time.gmtime

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    session = requests.Session()
    session.headers.update({"User-Agent": "cross-market-arb-event-snapshotter/0.1"})

    log.info(
        "event snapshotter starting | poll_interval=%.3fs | watch=%s | out=%s | once=%s",
        args.poll_interval,
        args.watch,
        args.out,
        args.once,
    )

    prev_hash: str | None = None
    prev_side_hashes: dict[str, str] | None = None
    last_write_monotonic: float | None = None

    while not _stop:
        t0 = time.monotonic()
        try:
            digest, sides, wrote, trigger, path = take_event_snapshot(
                out_dir=args.out,
                session=session,
                watch=args.watch,
                prev_hash=prev_hash,
                prev_side_hashes=prev_side_hashes,
                last_write_monotonic=last_write_monotonic,
                heartbeat_seconds=args.heartbeat_seconds,
                min_write_interval=args.min_write_interval,
            )
            if wrote:
                last_write_monotonic = time.monotonic()
                log.info("wrote %s | trigger=%s", path, trigger)
            else:
                log.info("skipped write | trigger=%s", trigger)
            prev_hash = digest
            prev_side_hashes = sides
        except Exception:
            log.exception("event snapshot iteration failed")

        if args.once:
            break

        elapsed = time.monotonic() - t0
        sleep_for = max(args.poll_interval - elapsed, 0.1)
        deadline = time.monotonic() + sleep_for
        while not _stop and time.monotonic() < deadline:
            time.sleep(min(0.5, max(deadline - time.monotonic(), 0.0)))

    log.info("event snapshotter stopped cleanly")


if __name__ == "__main__":
    main()
