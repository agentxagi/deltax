#!/bin/bash
# Parse run.sh output into RTABench results JSON.
# Usage: ./parse-results.sh < run_output.txt > results/pg_deltax.json
#
# Marker-aware: each query is delimited by "=== NAME ===" in run.sh's output.
# When a marker is seen, the current query's timings are flushed (padded
# to 3 with `null`), and a new query begins. This way a partial run
# (errors, timeouts) still produces a well-formed JSON with the right
# number of entries.

set -euo pipefail

MACHINE="${1:-c6a.4xlarge}"
LOAD_TIME="${LOAD_TIME:-0}"
DATA_SIZE="${DATA_SIZE:-0}"

results=()
timings=()
have_marker=0

flush() {
    while [ "${#timings[@]}" -lt 3 ]; do
        timings+=("null")
    done
    results+=("[${timings[0]}, ${timings[1]}, ${timings[2]}]")
    timings=()
}

while IFS= read -r line; do
    if [[ "$line" =~ ^===\ (.+)\ ===$ ]]; then
        if [ "$have_marker" -eq 1 ]; then
            flush
        fi
        have_marker=1
    elif [[ "$line" =~ Time:\ ([0-9.]+)\ ms ]]; then
        secs=$(echo "${BASH_REMATCH[1]} / 1000" | bc -l | xargs printf '%.3f')
        timings+=("$secs")
    elif [[ "$line" =~ psql:\ error ]] || [[ "$line" == "QUERY_ERROR" ]]; then
        timings+=("null")
    fi
done

if [ "$have_marker" -eq 1 ]; then
    flush
fi

# Emit JSON
cat <<EOF
{
    "system": "pg_deltax",
    "date": "$(date +%Y-%m-%d)",
    "machine": "${MACHINE}",
    "cluster_size": 1,
    "proprietary": "no",
    "hardware": "cpu",
    "tuned": "no",
    "tags": ["Rust", "PostgreSQL compatible", "column-oriented", "time-series", "real-time-analytics", "lukewarm-cold-run"],
    "load_time": ${LOAD_TIME},
    "data_size": ${DATA_SIZE},
    "result": [
EOF

n=${#results[@]}
for ((q = 0; q < n; q++)); do
    comma=","
    [ $((q + 1)) -eq "$n" ] && comma=""
    echo "        ${results[$q]}${comma}"
done

echo "    ]"
echo "}"
