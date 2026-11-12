#!/usr/bin/env bash
# Start the AI Employee server in the background and open the console.
# Double-click this file (or the AIEmployee.desktop launcher) — it needs no terminal.
# If the server is already running, it asks whether to restart it.
# Logs: data/server.log   PID: data/server.pid   Stop it with: ./stop_server.sh
#
# Observability: also starts the Docker stack in docker-compose.observability.yml
# (OpenTelemetry Collector, Prometheus, Loki, Tempo, Grafana) and runs the app with
# OTEL_ENABLED=true, then prints the links to every dashboard.
#   OBSERVABILITY=0 ./start_server.sh   # skip the Docker stack (plain local logs only)

# The script uses bash features — re-exec under bash if launched with sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

PORT="${AI_EMPLOYEE_PORT:-8000}"
PID_FILE="data/server.pid"
LOG_FILE="data/server.log"
URL="http://localhost:${PORT}/ui"
OBSERVABILITY="${OBSERVABILITY:-1}"
COMPOSE_FILE="docker-compose.observability.yml"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
LOKI_PORT="${LOKI_PORT:-3100}"
TEMPO_PORT="${TEMPO_PORT:-3200}"
OTLP_HTTP_PORT="${OTLP_HTTP_PORT:-4318}"
GRAFANA="http://localhost:${GRAFANA_PORT}"
STACK_UP=0
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

# ── Observability stack (Docker) ─────────────────────────────────────────────
start_observability() {
  [ "$OBSERVABILITY" = "0" ] && return 0
  if ! command -v docker >/dev/null || ! docker info >/dev/null 2>&1; then
    echo "⚠  Docker is not available — skipping the observability stack (logs stay in ${LOG_FILE})."
    return 0
  fi
  echo "▶ Starting observability containers (Collector, Prometheus, Loki, Tempo, Grafana)…"
  if ! docker compose -f "$COMPOSE_FILE" up -d; then
    echo "⚠  Could not start the observability stack — continuing without it."
    return 0
  fi
  # Wait (up to ~60s) for Grafana and the collector to accept connections.
  for _ in $(seq 1 30); do
    if curl -fsS "${GRAFANA}/api/health" >/dev/null 2>&1 \
      && curl -sS -o /dev/null "http://localhost:${OTLP_HTTP_PORT}/v1/traces" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  STACK_UP=1
  # The app exports traces/metrics/logs over OTLP to the collector.
  export OTEL_ENABLED=true
  export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:${OTLP_HTTP_PORT}"
}

print_links() {
  local explore_logs explore_traces
  explore_logs="${GRAFANA}/explore?schemaVersion=1&orgId=1&panes=%7B%22a%22%3A%7B%22datasource%22%3A%22loki%22%2C%22queries%22%3A%5B%7B%22refId%22%3A%22A%22%2C%22datasource%22%3A%7B%22type%22%3A%22loki%22%2C%22uid%22%3A%22loki%22%7D%2C%22expr%22%3A%22%7Bservice_name%3D%5C%22ai-employee%5C%22%7D%22%7D%5D%2C%22range%22%3A%7B%22from%22%3A%22now-1h%22%2C%22to%22%3A%22now%22%7D%7D%7D"
  explore_traces="${GRAFANA}/explore?schemaVersion=1&orgId=1&panes=%7B%22a%22%3A%7B%22datasource%22%3A%22tempo%22%2C%22queries%22%3A%5B%7B%22refId%22%3A%22A%22%2C%22datasource%22%3A%7B%22type%22%3A%22tempo%22%2C%22uid%22%3A%22tempo%22%7D%2C%22queryType%22%3A%22traceql%22%2C%22query%22%3A%22%7Bresource.service.name%3D%5C%22ai-employee%5C%22%7D%22%7D%5D%2C%22range%22%3A%7B%22from%22%3A%22now-1h%22%2C%22to%22%3A%22now%22%7D%7D%7D"
  echo
  echo "════════════════════════════════════════════════════════════════════════"
  echo " AI Employee is running"
  echo "════════════════════════════════════════════════════════════════════════"
  echo "  Console (web UI) ........ ${URL}"
  echo "  API docs ................ http://localhost:${PORT}/docs"
  echo "  Server log file ......... $(pwd)/${LOG_FILE}"
  if [ "$STACK_UP" = "1" ]; then
    echo
    echo "  Logging & monitoring — open these links:"
    echo "  Grafana dashboard ....... ${GRAFANA}/d/ai-employee/ai-employee"
    echo "  Logs (Loki in Grafana) .. ${explore_logs}"
    echo "  Traces (Tempo in Grafana) ${explore_traces}"
    echo "  Prometheus (metrics) .... http://localhost:${PROMETHEUS_PORT}/graph"
    echo "  Prometheus targets ...... http://localhost:${PROMETHEUS_PORT}/targets"
    echo "  Loki status ............. http://localhost:${LOKI_PORT}/ready"
    echo "  Tempo status ............ http://localhost:${TEMPO_PORT}/ready"
    echo "  OTLP endpoint (app→) .... http://localhost:${OTLP_HTTP_PORT}"
    echo "  Grafana login (if asked)  admin / admin"
    echo
    echo "  Containers:"
    docker compose -f "$COMPOSE_FILE" ps --format '    {{.Service}}  {{.State}}  {{.Ports}}' 2>/dev/null || true
  else
    echo "  (Observability stack not running — start Docker, or run without OBSERVABILITY=0.)"
  fi
  echo "════════════════════════════════════════════════════════════════════════"
}

start_observability

# Already running? Ask before restarting; otherwise just open the console.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  if confirm $'The AI Employee server is already running.\n\nRestart it? (running tasks will be interrupted)'; then
    notify "Restarting the server…"
    stop_running
  else
    notify "Keeping the running server — opening ${URL}"
    print_links
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
    print_links
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
print_links
