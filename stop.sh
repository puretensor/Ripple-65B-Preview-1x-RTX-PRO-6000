#!/usr/bin/env bash
# Stop the server ./start.sh started (its whole process group). Weights stay in MODELS_DIR.
cd "$(dirname "$0")"
[ -f .server.pid ] || { echo "not running (no .server.pid)"; exit 0; }
PID=$(cat .server.pid)
if kill -0 "$PID" 2>/dev/null; then
  kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID"
  for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$PID" 2>/dev/null || true
  echo "stopped ripple (pid $PID)"
else
  echo "ripple was not running"
fi
rm -f .server.pid
