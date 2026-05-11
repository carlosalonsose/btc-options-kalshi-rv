#!/bin/bash
# Re-corre el backtest offline. Pasa los args al modulo, ej:
#   bash backtest.sh                  # default stride=5
#   bash backtest.sh --stride 1       # cada snapshot
#   bash backtest.sh --refresh        # re-fetch settled de Kalshi
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m src.runner.backtest "$@"
