#!/bin/bash
# Lanza la app Streamlit de analiticas. Abre browser en localhost:8501.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/streamlit run src/ui/streamlit_app.py --server.headless false
