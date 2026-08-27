#!/usr/bin/env bash
# Start the AI Employee server in the background and open the console.
# Double-click this file (or the AIEmployee.desktop launcher) — it needs no terminal.
# If the server is already running, it asks whether to restart it.
# Logs: data/server.log   PID: data/server.pid   Stop it with: ./stop_server.sh

# The script uses bash features — re-exec under bash if launched with sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PORT="${AI_EMPLOYEE_PORT:-8000}"
PID_FILE="data/server.pid"
LOG_FILE="data/server.log"
URL="http://localhost:${PORT}/ui"
mkdir -p data

notify() {  # best-effort desktop notification; falls back to stdout
  command -v notify-send >/dev/null && notify-send "AI Employee" "$1" || echo "$1"
}

confirm() {  # GUI yes/no when double-clicked, terminal prompt otherwise
  if command -v zenity >/dev/null; then
    zenity --question --title "AI Employee" --text "$1" --width 300 2>/dev/null
  elif command -v kdialog >/dev/null; then
    kdialog --title "AI Employee" --yesno "$1"
  elif [ -t 0 ]; then
    read -r -p "$1 [y/N] " reply && [[ "$reply" =~ ^[Yy] ]]
  else
    return 1  # no way to ask — treat as "no"
  fi
}

stop_running() {
  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
}

# Already running? Ask before restarting; otherwise just open the console.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  if confirm $'The AI Employee server is already running.\n\nRestart it? (running tasks will be interrupted)'; then
    notify "Restarting the server…"
    stop_running
  else
    notify "Keeping the running server — opening ${URL}"
    xdg-open "$URL" >/dev/null 2>&1 || true
    exit 0
  fi
fi
rm -f "$PID_FILE"

# Activate the conda env (AIEmployee) without needing an interactive shell.
for conda_sh in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
  if [ -f "$conda_sh" ]; then
    # shellcheck source=/dev/null
    source "$conda_sh"
    conda activate AIEmployee
    break
  fi
done

nohup python -m ai_employee serve --port "$PORT" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

# Wait for the server to come up (first start can take a few seconds).
for _ in $(seq 1 30); do
  if curl -fsS "http://localhost:${PORT}/ui" >/dev/null 2>&1; then
    notify "Server started — opening ${URL}"
    xdg-open "$URL" >/dev/null 2>&1 || true
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    notify "Server failed to start — see ${LOG_FILE}"
    rm -f "$PID_FILE"
    exit 1
  fi
  sleep 1
done
notify "Server is starting slowly — check ${LOG_FILE}, console at ${URL}"
