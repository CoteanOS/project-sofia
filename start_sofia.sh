#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./warm_models.sh

exec "$(dirname "$0")/.venv/bin/uvicorn" server:app --host 127.0.0.1 --port 8000
