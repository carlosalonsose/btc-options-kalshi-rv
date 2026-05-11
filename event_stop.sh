#!/bin/bash
# Stops the event-driven snapshotter if it is running.

if pkill -f "src.event_snapshotter" 2>/dev/null; then
    echo "OK: event snapshotter stopped"
else
    echo "no event snapshotter instance was running"
fi
