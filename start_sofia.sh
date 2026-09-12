#!/usr/bin/env bash
set -euo pipefail

cd ~/code/sofia

./warm_models.sh

exec ~/code/sofia/.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8000
