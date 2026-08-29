#!/bin/bash
# Run all RTABench queries 3x each, dropping OS caches between queries.
# Output is consumed by parse-results.sh — the "=== NAME ===" markers
# pair filenames with timings so partial-failure runs still produce
# well-formed JSON.

set -u

TRIES=3
DIR="$(cd "$(dirname "$0")" && pwd)"

for file in "$DIR/queries"/*.sql; do
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    name="$(basename "$file" .sql)"
    echo "=== $name ==="

    query="$(cat "$file")"
    for i in $(seq 1 $TRIES); do
        # `work_mem=50MB` matches the TimescaleDB RTABench harness for an
        # apples-to-apples comparison.
        # ON_ERROR_STOP=on so psql exits non-zero on PG ERROR. Without it,
        # `\timing` still prints "Time: X.XX ms" (time-to-error) and the
        # parser would accept failed queries as fast successes — this
        # silently produced a bogus result set once (json_extract metadata
        # mismatch errored every order_events query; all 3 runs recorded
        # the time-to-error). Same guard as clickbench/run.sh.
        out=$(sudo -u postgres psql test --no-psqlrc --tuples-only -v ON_ERROR_STOP=on \
            --command "\timing on" \
            --command "SET work_mem = '50MB'" \
            --command "$query" 2>&1)
        if [ $? -ne 0 ]; then
            echo "QUERY_ERROR"
        else
            echo "$out" | grep -P 'Time' | tail -n1
        fi
    done
done
