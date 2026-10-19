#!/usr/bin/env bash
# Run the AI Employee CLI inside its conda env (AIEmployee), from any shell.
#
#   ./aiemployee.sh resources purge
#   ./aiemployee.sh resources links
#   ./aiemployee.sh run "Plan a product launch"
#   ./aiemployee.sh serve
#
# Without this, `python -m ai_employee ...` must be run from the activated env —
# using base Anaconda Python gives "No module named ai_employee".

if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

for conda_sh in "$HOME/anaconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
  if [ -f "$conda_sh" ]; then
    # shellcheck source=/dev/null
    source "$conda_sh"
    conda activate AIEmployee
    break
  fi
done

if ! python -c "import ai_employee" >/dev/null 2>&1; then
  echo "error: the 'AIEmployee' conda env (with ai_employee installed) was not found." >&2
  echo "       create it with:  make env && conda activate AIEmployee && make dev" >&2
  exit 1
fi

exec python -m ai_employee "$@"
