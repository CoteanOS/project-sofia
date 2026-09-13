#!/usr/bin/env bash
set -euo pipefail

echo "Warming sofia-worker..."
curl -s http://127.0.0.1:11434/api/generate \
  -d '{
    "model": "sofia-worker",
    "prompt": "hi",
    "stream": false,
    "keep_alive": "5m",
    "options": {
      "num_ctx": 8192
    }
  }' > /dev/null

echo "Warming sofia-router..."
curl -s http://127.0.0.1:11434/api/generate \
  -d '{
    "model": "sofia-router",
    "prompt": "Reply only with: assistant",
    "stream": false,
    "keep_alive": "5m",
    "options": {
      "num_ctx": 1024,
      "temperature": 0
    }
  }' > /dev/null

echo
ollama ps
