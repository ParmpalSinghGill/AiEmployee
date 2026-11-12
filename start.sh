#!/usr/bin/env bash
# Shortcut for ./start_server.sh: starts the observability containers
# (Collector, Prometheus, Loki, Tempo, Grafana), the AI Employee server, and
# prints every link (console, Grafana dashboard, logs, traces, metrics).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
exec "$(dirname "$(readlink -f "$0")")/start_server.sh" "$@"
