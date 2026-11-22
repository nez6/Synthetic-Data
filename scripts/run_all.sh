#!/usr/bin/env bash
# Start the API (with the HL7 MLLP listener) and the Streamlit dashboard together.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
API_PORT=${API_PORT:-8000}
UI_PORT=${UI_PORT:-8501}

"$PY" -m uvicorn src.app:app --port "$API_PORT" &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

echo "API:          http://localhost:$API_PORT/docs"
echo "HL7 listener: 127.0.0.1:2575 (MLLP)"
echo "Dashboard:    http://localhost:$UI_PORT"
"$PY" -m streamlit run interface/frontend.py --server.port "$UI_PORT"
