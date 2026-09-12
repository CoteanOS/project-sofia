#!/usr/bin/env bash
set -euo pipefail

echo "Warming gpt-oss:20b..."
curl -s http://127.0.0.1:11434/api/generate \
  -d '{
    "model": "gpt-oss:20b",
    "prompt": "hi",
    "stream": false,
    "keep_alive": "5m",
    "options": {
      "num_ctx": 8192
    }
  }' > /dev/null

echo "Warming qwen3:4b-instruct..."
curl -s http://127.0.0.1:11434/api/generate \
  -d '{
    "model": "qwen3:4b-instruct",
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
