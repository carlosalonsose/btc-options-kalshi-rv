#!/bin/bash
# Lanza el dashboard Tkinter.
# Necesita display (no funciona en SSH sin -X / no funciona headless).
# Requiere Tk moderno: brew install python-tk@3.9
cd "$(dirname "$0")" || exit 1
export TK_SILENCE_DEPRECATION=1
exec .venv/bin/python -m src.ui.dashboard
