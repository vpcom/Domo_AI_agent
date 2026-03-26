#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

exec "$ROOT/.venv/bin/python" -m streamlit run "$ROOT/app/streamlit_app.py" --server.port 8051
