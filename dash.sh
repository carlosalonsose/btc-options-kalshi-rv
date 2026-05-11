#!/bin/bash
# Lanza el dashboard Tkinter.
# Necesita display (no funciona en SSH sin -X / no funciona headless).
cd "$(dirname "$0")" || exit 1
exec .venv/bin/python -m src.ui.dashboard
