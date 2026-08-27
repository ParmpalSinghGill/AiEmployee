#!/usr/bin/env bash
# Stop the background AI Employee server started by start_server.sh.

# Re-exec under bash if launched with sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PID_FILE="data/server.pid"

notify() {
  command -v notify-send >/dev/null && notify-send "AI Employee" "$1" || echo "$1"
}

if [ ! -f "$PID_FILE" ]; then
  notify "Server is not running (no ${PID_FILE})."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  # Give it a few seconds to shut down cleanly, then force-kill.
  for _ in $(seq 1 10); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$PID" 2>/dev/null || true
  notify "Server stopped."
else
  notify "Server was not running — cleaned up stale PID file."
fi
rm -f "$PID_FILE"
