#!/bin/bash
# Starts the event-driven snapshotter in the background.
# It polls often, but only writes snapshots when quote state changes.

cd "$(dirname "$0")" || exit 1

pkill -f "src.event_snapshotter" 2>/dev/null
sleep 1

POLL_INTERVAL="${POLL_INTERVAL:-10}"
WATCH="${WATCH:-both}"
OUT_DIR="${OUT_DIR:-data/event_snapshots}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-300}"

nohup .venv/bin/python -m src.event_snapshotter \
    --poll-interval "$POLL_INTERVAL" \
    --watch "$WATCH" \
    --out "$OUT_DIR" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" \
    >/dev/null 2>&1 &

PID=$!
sleep 2

if pgrep -f "src.event_snapshotter" >/dev/null; then
    echo "OK: event snapshotter running (PID $PID)"
    echo "Out: $(pwd)/$OUT_DIR"
    echo "Log: $(pwd)/$OUT_DIR/event_snapshotter.log"
else
    echo "FAILED: event snapshotter did not start. Check $OUT_DIR/event_snapshotter.log"
    exit 1
fi
