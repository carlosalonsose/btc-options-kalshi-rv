#!/bin/bash
# Para la grabadora si está corriendo.

if pkill -f "src.snapshotter" 2>/dev/null; then
    echo "OK: snapshotter parada"
else
    echo "no había ninguna instancia corriendo"
fi
