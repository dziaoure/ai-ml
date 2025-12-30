#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_app.sh <demo-folder>/app/app.py>
# Example:
#   ./run_app.sh health-risk-scoring/app/app.py

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-streamlit-app.py>"
  exit 1
fi

APP_PATH="$1"

if [[ ! -d ".venv" ]]; then
  echo "Error: .venv not found at repo root. Create it with:"
  echo "  python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.apps.txt"
  exit 1
fi

# Activate venv
# shellcheck disable=SC1091
source ".venv/bin/activate"

# Sanity checks
VENV_PY="$(python -c 'import sys; print(sys.executable)')"
echo "Using python: $VENV_PY"

# Ensure streamlit module is importable in this venv
python -c "import streamlit; print('streamlit:', streamlit.__version__)"

if [[ ! -f "$APP_PATH" ]]; then
  echo "Error: app file not found: $APP_PATH"
  exit 1
fi

echo "Running: $APP_PATH"
exec python -m streamlit run "$APP_PATH"
