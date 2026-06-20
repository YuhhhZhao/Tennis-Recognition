#!/usr/bin/env bash
set -euo pipefail

if command -v tegrastats >/dev/null 2>&1; then
  seconds="${JETSON_MONITOR_SECONDS:-5}"
  echo "Running tegrastats for ${seconds}s."
  if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout "${seconds}s" tegrastats
    status="$?"
    set -e
    if [ "$status" -eq 124 ] || [ "$status" -eq 130 ]; then
      echo "tegrastats sample complete."
      exit 0
    fi
    exit "$status"
  fi
  tegrastats &
  pid="$!"
  sleep "$seconds"
  kill "$pid" >/dev/null 2>&1 || true
  wait "$pid" >/dev/null 2>&1 || true
  echo "tegrastats sample complete."
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "No tegrastats or nvidia-smi command found; Jetson monitoring is unavailable."
fi
