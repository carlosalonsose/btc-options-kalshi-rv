#!/bin/bash
# Lanza la grabadora (snapshotter) en segundo plano.
# Si ya estaba corriendo, la para primero para no duplicar.

cd "$(dirname "$0")" || exit 1

pkill -f "src.snapshotter" 2>/dev/null
sleep 1

nohup .venv/bin/python -m src.snapshotter --interval 60 --out data/snapshots >/dev/null 2>&1 &
PID=$!
sleep 2

if pgrep -f "src.snapshotter" >/dev/null; then
    echo "OK: snapshotter corriendo (PID $PID)"
    echo "Log: $(pwd)/data/snapshots/snapshotter.log"
else
    echo "FALLO: la grabadora no arrancó. Revisa data/snapshots/snapshotter.log"
    exit 1
fi
