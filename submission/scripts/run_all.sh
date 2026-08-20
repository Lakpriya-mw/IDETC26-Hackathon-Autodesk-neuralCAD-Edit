#!/usr/bin/env bash
# End-to-end run for lw_solution (macOS / Linux).
#
#   cd /path/to/IDETC_Hackathon_Source
#   bash lw_solution_v2/scripts/run_all.sh            # everything
#   LIMIT=3 bash lw_solution_v2/scripts/run_all.sh    # smoke test
#
# On headless Linux, prefix the harness/postprocess calls with `xvfb-run -a`
# so offscreen rendering works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG="${SCRIPT_DIR}/../config/lw_config.json"

export PYTHONPATH="${REPO_ROOT}"

LIMIT="${LIMIT:-0}"
SKIP_HARNESS="${SKIP_HARNESS:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

echo "repo root : ${REPO_ROOT}"
echo "config    : ${CONFIG}"

echo -e "\n=== 0. self-test ==="
python "${SCRIPT_DIR}/selftest.py" --config "${CONFIG}"

if [ "${SKIP_HARNESS}" != "1" ]; then
  echo -e "\n=== 1. run the agentic harness ==="
  if [ "${LIMIT}" != "0" ]; then
    python "${SCRIPT_DIR}/run_harness.py" --config "${CONFIG}" --limit "${LIMIT}"
  else
    python "${SCRIPT_DIR}/run_harness.py" --config "${CONFIG}"
  fi

  echo -e "\n=== 2. backfill any missing .stl / views ==="
  python "${SCRIPT_DIR}/postprocess.py"
fi

if [ "${SKIP_EVAL}" != "1" ]; then
  echo -e "\n=== 3. ingest + score with the organisers' pipeline ==="
  python "${SCRIPT_DIR}/evaluate.py" --config "${CONFIG}"
fi

echo -e "\nDone. Open src/notebooks/leaderboard.ipynb to view the results."
