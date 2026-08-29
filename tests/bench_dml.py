"""Local Docker DML benchmark — pg_deltax compressed-partition write latency.

RTABench and ClickBench are read-only. This measures the *write* side that
DML-on-compressed (P1 INSERT / P2 decompose UPDATE·DELETE / P2.5 tombstone
DELETE) adds, and tracks it over time.

Self-contained: generates a synthetic time-series table (no external data)
loaded into two twins with identical rows — `dml_bench` (pg_deltax,
compressed) and `dml_bench_plain` (plain PostgreSQL). Every op is reported
as an absolute warm latency AND as a ratio to native Postgres — the ratio is
the product requirement ("the user should almost not notice it's not normal
Postgres by wait time").

Each op is measured on a pristine compressed partition and undone with
ROLLBACK between iterations, so numbers don't accumulate and stay comparable
release-to-release (the decompose/tombstone/heap-insert work all happens
before COMMIT, so ROLLBACK isolates the extension's per-op cost from fsync
noise — same for both twins, so the ratio is fair).

A second phase measures the read-after-write cliff: a GROUP BY aggregate on
pristine data (DeltaXAgg pushdown) vs after one committed INSERT (pushdown
disabled → row path) vs after deltax_compact_partition() (restored) — the
"does DML slow the happy path" number nothing else tracks.

Results + history archive land in tests/.bench_results/ via save_bench_results.
A pinned baseline (dml_baseline.json) drives a soft regression gate; re-bless
with DML_BLESS=1.

Makefile:
  make bench-dml            # measure, print report, archive to history
  make bench-dml-keep       # + keep the container/db
  make bench-dml-bless      # record current warm times as the baseline
"""

from __future__ import annotations

import os
import statistics
import time
import uuid

import psycopg
import pytest

from clickbench_data import (
    BENCH_RESULTS_DIR,
    save_bench_results,
    _get_git_commit_short,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DML_ROWS = int(os.environ.get("DML_ROWS", "300000"))  # ~10 segments over ~4 partitions
SEGMENT_SIZE = int(os.environ.get("DML_SEGMENT_SIZE", "30000"))
WARMUP_RUNS = 1
TIMED_RUNS = 5
# One row per second from this base. 300k rows ≈ 3.47 days → daily partitions
# give ~4 partitions, each ~3 segments (exercises multi-segment + multi-partition).
# One row/sec from BASE_TS. mock_now == BASE_TS so deltax_create_table's
# future partitions start at the data and cover it (data must land in real
# partitions, not the default — we INSERT then compress, we don't run the
# drain worker). 300k rows ≈ 3.5 days → ~4 daily partitions, each ~3 segments.
BASE_TS = "2024-01-01 00:00:00+00"
MOCK_NOW = "2024-01-01 00:00:00+00"
PARTITION_INTERVAL = "1 day"

BASELINE_PATH = BENCH_RESULTS_DIR / "dml_baseline.json"
BENCH_BLESS = os.environ.get("DML_BLESS", "") not in ("", "0", "false")
TIME_RATIO_THRESHOLD = float(os.environ.get("DML_TIME_RATIO", "3"))
TIME_FLOOR_MS = float(os.environ.get("DML_TIME_FLOOR_MS", "1"))

SCHEMA = """
CREATE TABLE {name} (
    ts          timestamptz NOT NULL,
    device_id   int         NOT NULL,
    region      text        NOT NULL,
    temperature float8      NOT NULL,
    payload     text
);
"""
COLS = "ts, device_id, region, temperature, payload"


# ---------------------------------------------------------------------------
# Fixture — generate + load twins, compress the pg_deltax one.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def dml_db(pg_container):
    from conftest import HOST_PORT, PG_PASSWORD, PG_USER, _admin_conn

    persist = bool(os.environ.get("BENCH_PERSIST"))
    db_name = "bench_dml" if persist else f"bench_dml_{uuid.uuid4().hex[:8]}"
    admin = _admin_conn()
    if not admin.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
    ).fetchone():
        admin.execute(f'CREATE DATABASE "{db_name}"')
    admin.close()

    conn = psycopg.connect(
        host="localhost", port=HOST_PORT, user=PG_USER,
        password=PG_PASSWORD, dbname=db_name,
    )
    conn.execute("SET jit = off")
    conn.execute("CREATE EXTENSION IF NOT EXISTS pg_deltax")
    conn.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
    conn.commit()

    already = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE tablename = 'dml_bench')"
    ).fetchone()[0]
    if not already:
        _load(conn)
    else:
        print(f"\n  Reusing existing dml_bench ({DML_ROWS:,} rows target)")

    # Pristine: every dml_bench partition fully compressed, empty heap, no
    # tombstones. Scoped to this deltatable so unrelated tables in a reused
    # persist DB don't trip it.
    loose = conn.execute(
        "SELECT count(*) FROM deltax.deltax_partition "
        "WHERE (has_loose_rows OR has_tombstones) AND deltatable_id = "
        "(SELECT id FROM deltax.deltax_deltatable WHERE table_name = 'dml_bench')"
    ).fetchone()[0]
    assert loose == 0, f"{loose} dml_bench partitions already dirty before bench"

    yield conn

    conn.close()
    if os.environ.get("KEEP_CONTAINER") or persist:
        print(f"\n  Keeping database {db_name} for reuse")
    else:
        admin = _admin_conn()
        admin.execute(f'DROP DATABASE "{db_name}"')
        admin.close()


def _load(conn):
    print(f"\n  Generating {DML_ROWS:,} synthetic rows ...")
    for name in ("dml_bench", "dml_bench_plain"):
        conn.execute(SCHEMA.format(name=name))
    conn.execute(
        "SELECT deltax.deltax_create_table('dml_bench', 'ts', "
        f"'{PARTITION_INTERVAL}'::interval, 40)"
    )
    conn.execute(
        "SELECT deltax.deltax_enable_compression('dml_bench', "
        f"order_by => ARRAY['ts'], segment_size => {SEGMENT_SIZE})"
    )
    conn.commit()

    # Deterministic generator: one row/sec; device_id spreads 0..999; region
    # is one of 8; payload ~48 bytes so rows aren't trivially narrow.
    gen = f"""
        SELECT
            '{BASE_TS}'::timestamptz + (g || ' seconds')::interval,
            g % 1000,
            'region-' || (g % 8),
            (g % 100)::float8 + 0.5,
            'payload-' || md5(g::text)
        FROM generate_series(0, {DML_ROWS - 1}) g
    """
    for name in ("dml_bench_plain", "dml_bench"):
        t0 = time.monotonic()
        conn.execute(f"INSERT INTO {name} ({COLS}) {gen}")
        conn.commit()
        print(f"    loaded {name} in {time.monotonic()-t0:.1f}s")

    # Guard: with mock_now == BASE_TS the data must land in real partitions,
    # not the default (we don't run the drain worker).
    default_rows = conn.execute(
        "SELECT count(*) FROM ONLY dml_bench_default"
    ).fetchone()[0] if conn.execute(
        "SELECT to_regclass('dml_bench_default') IS NOT NULL"
    ).fetchone()[0] else 0
    assert default_rows == 0, (
        f"{default_rows} rows in the default partition — mock_now/interval "
        "don't cover the data span")

    # Compress only the partitions that actually hold data (the 40 pre-created
    # future partitions past the ~4-day span are empty — skip them).
    end_ts = conn.execute(
        f"SELECT '{BASE_TS}'::timestamptz + (%s || ' seconds')::interval",
        (DML_ROWS,),
    ).fetchone()[0]
    parts = [r[0] for r in conn.execute(
        "SELECT partition_name FROM deltax.deltax_partition_info('dml_bench') "
        "WHERE NOT is_compressed AND partition_name NOT LIKE '%%default%%' "
        "AND range_start < %s AND range_end > %s::timestamptz",
        (end_ts, BASE_TS),
    ).fetchall()]
    for p in parts:
        conn.execute("SELECT deltax.deltax_compress_partition(%s)", (p,))
    conn.commit()
    n_seg = conn.execute(
        "SELECT count(*) FROM deltax.deltax_partition WHERE is_compressed"
    ).fetchone()[0]
    print(f"    compressed {n_seg} partitions ({len(parts)} with data)")


# ---------------------------------------------------------------------------
# Measurement — warm, rollback-isolated, median of TIMED_RUNS.
# ---------------------------------------------------------------------------

def measure(conn, op, *, fetch=False) -> tuple[float, int]:
    def once():
        with conn.cursor() as cur:
            t0 = time.monotonic()
            op(cur)
            if fetch:
                cur.fetchall()
            dt = (time.monotonic() - t0) * 1000.0
            rc = cur.rowcount
        conn.rollback()
        return dt, rc

    for _ in range(WARMUP_RUNS):
        once()
    times, rc = [], 0
    for _ in range(TIMED_RUNS):
        dt, rc = once()
        times.append(dt)
    return statistics.median(times), rc


def sql_op(sql: str):
    return lambda cur: cur.execute(sql)


# ---------------------------------------------------------------------------
# Targets + the op matrix.
# ---------------------------------------------------------------------------

def derive_targets(conn) -> dict:
    # A ts in the middle of the data → a single interior segment.
    mid_ts = conn.execute(
        f"SELECT '{BASE_TS}'::timestamptz + (%s || ' seconds')::interval",
        (DML_ROWS // 2,),
    ).fetchone()[0]
    # Earliest compressed partition — its range_end is a clean retention cutoff.
    earliest_part, part_end = conn.execute(
        "SELECT partition_name, range_end FROM deltax.deltax_partition_info('dml_bench') "
        "WHERE is_compressed ORDER BY range_start LIMIT 1"
    ).fetchone()
    return {
        "point_ts": mid_ts,
        "earliest_part": earliest_part,
        "retention_cutoff": part_end,
        "insert_ts": conn.execute("SELECT min(ts) FROM dml_bench_plain").fetchone()[0],
        "point_rows": conn.execute(
            "SELECT count(*) FROM dml_bench_plain WHERE ts = %s", (mid_ts,)
        ).fetchone()[0],
        "retention_rows": conn.execute(
            "SELECT count(*) FROM dml_bench_plain WHERE ts < %s", (part_end,)
        ).fetchone()[0],
    }


def sample_rows(conn, n: int) -> list[tuple]:
    return conn.execute(
        f"SELECT {COLS} FROM dml_bench_plain ORDER BY ts LIMIT %s", (n,)
    ).fetchall()


def build_ops(conn, tgt: dict) -> list[dict]:
    PT = tgt["point_ts"]
    CUT = tgt["retention_cutoff"]
    TS = tgt["insert_ts"]
    batch = sample_rows(conn, 1000)

    def one_row(table):
        return (f"INSERT INTO {table} ({COLS}) SELECT {COLS} FROM dml_bench_plain "
                f"WHERE ts = '{TS}'::timestamptz LIMIT 1")

    def batch_insert(table):
        return lambda cur: cur.executemany(
            f"INSERT INTO {table} ({COLS}) VALUES (%s,%s,%s,%s,%s)", batch)

    def copy_insert(table):
        def op(cur):
            with cur.copy(f"COPY {table} ({COLS}) FROM STDIN") as cp:
                for r in batch:
                    cp.write_row(r)
        return op

    return [
        {"label": "INSERT single row", "path": "loose-region heap insert",
         "dx": sql_op(one_row("dml_bench")), "plain": sql_op(one_row("dml_bench_plain"))},
        {"label": "INSERT batch (1000)", "path": "loose-region, executemany",
         "dx": batch_insert("dml_bench"), "plain": batch_insert("dml_bench_plain")},
        {"label": "COPY (1000)", "path": "loose-region, COPY",
         "dx": copy_insert("dml_bench"), "plain": copy_insert("dml_bench_plain")},
        {"label": f"DELETE point (ts=mid, {tgt['point_rows']} row)",
         "path": "tombstone fast path",
         "dx": sql_op(f"DELETE FROM dml_bench WHERE ts = '{PT}'::timestamptz"),
         "plain": sql_op(f"DELETE FROM dml_bench_plain WHERE ts = '{PT}'::timestamptz")},
        {"label": f"DELETE retention (< partition end, {tgt['retention_rows']} rows)",
         "path": "whole-segment drop (partition-aligned)",
         "dx": sql_op(f"DELETE FROM dml_bench WHERE ts < '{CUT}'::timestamptz"),
         "plain": sql_op(f"DELETE FROM dml_bench_plain WHERE ts < '{CUT}'::timestamptz")},
        {"label": "DELETE ... RETURNING (ts=mid)",
         "path": "forces decompose (rows observed)", "fetch": True,
         "dx": sql_op(f"DELETE FROM dml_bench WHERE ts = '{PT}'::timestamptz RETURNING *"),
         "plain": sql_op(f"DELETE FROM dml_bench_plain WHERE ts = '{PT}'::timestamptz RETURNING *")},
        {"label": "UPDATE point (ts=mid)", "path": "decompose-on-write",
         "dx": sql_op(f"UPDATE dml_bench SET temperature = temperature + 1 WHERE ts = '{PT}'::timestamptz"),
         "plain": sql_op(f"UPDATE dml_bench_plain SET temperature = temperature + 1 WHERE ts = '{PT}'::timestamptz")},
        {"label": f"UPDATE retention range (< partition end)", "path": "decompose-on-write",
         "dx": sql_op(f"UPDATE dml_bench SET temperature = temperature + 1 WHERE ts < '{CUT}'::timestamptz"),
         "plain": sql_op(f"UPDATE dml_bench_plain SET temperature = temperature + 1 WHERE ts < '{CUT}'::timestamptz")},
    ]


# ---------------------------------------------------------------------------
# Read-after-write cliff.
# ---------------------------------------------------------------------------

CLIFF_AGG = "SELECT region, count(*) FROM dml_bench GROUP BY region"


def _time_query(conn, sql: str) -> float:
    for _ in range(WARMUP_RUNS):
        conn.execute(sql).fetchall(); conn.rollback()
    ts = []
    for _ in range(TIMED_RUNS):
        t0 = time.monotonic(); conn.execute(sql).fetchall()
        ts.append((time.monotonic() - t0) * 1000.0); conn.rollback()
    return statistics.median(ts)


def measure_read_cliff(conn, tgt: dict) -> dict:
    part, TS = tgt["earliest_part"], tgt["insert_ts"]
    pristine = _time_query(conn, CLIFF_AGG)

    conn.execute(
        f"INSERT INTO dml_bench ({COLS}) SELECT {COLS} FROM dml_bench_plain "
        f"WHERE ts = '{TS}'::timestamptz LIMIT 1"
    )
    conn.commit()
    dirty = _time_query(conn, CLIFF_AGG)

    conn.execute("SELECT deltax.deltax_compact_partition(%s)", (part,))
    conn.commit()
    compacted = _time_query(conn, CLIFF_AGG)
    return {
        "pristine_ms": pristine,
        "after_1_insert_ms": dirty,
        "after_compaction_ms": compacted,
        "cliff_ratio": (dirty / pristine) if pristine else None,
    }


# ---------------------------------------------------------------------------
# Baseline regression gate.
# ---------------------------------------------------------------------------

def load_baseline():
    if not BASELINE_PATH.exists():
        return None
    try:
        import json
        with open(BASELINE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def save_baseline(report):
    import json
    BENCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump({"rows": DML_ROWS, "commit": _get_git_commit_short(),
                   "deltax_ms": {k: v["deltax_ms"] for k, v in report.items()}}, f, indent=2)
    print(f"  Blessed DML baseline → {BASELINE_PATH}")


# ---------------------------------------------------------------------------
# The test.
# ---------------------------------------------------------------------------

class TestDmlBench:
    def test_bench(self, dml_db):
        conn = dml_db
        tgt = derive_targets(conn)
        ops = build_ops(conn, tgt)

        print("\n\n=== DML latency: pg_deltax (compressed) vs plain PostgreSQL ===")
        print(f"{'operation':<48}{'deltax':>11}{'plain':>11}{'ratio':>8}  path")
        report = {}
        for op in ops:
            dx_ms, dx_rc = measure(conn, op["dx"], fetch=op.get("fetch", False))
            pl_ms, _ = measure(conn, op["plain"], fetch=op.get("fetch", False))
            ratio = dx_ms / pl_ms if pl_ms else float("inf")
            report[op["label"]] = {"deltax_ms": dx_ms, "plain_ms": pl_ms,
                                   "ratio": ratio, "path": op["path"], "rowcount": dx_rc}
            print(f"{op['label']:<48}{dx_ms:>9.2f}ms{pl_ms:>9.2f}ms{ratio:>7.1f}x  {op['path']}")

        print("\n=== Read-after-write cliff (GROUP BY agg pushdown) ===")
        cliff = measure_read_cliff(conn, tgt)
        print(f"  pristine (DeltaXAgg pushdown): {cliff['pristine_ms']:.2f} ms")
        print(f"  after 1 INSERT (row path):     {cliff['after_1_insert_ms']:.2f} ms"
              f"  ({cliff['cliff_ratio']:.1f}x slower)")
        print(f"  after compaction (restored):   {cliff['after_compaction_ms']:.2f} ms")

        baseline = load_baseline()
        violations = []
        if BENCH_BLESS:
            save_baseline(report)
        elif baseline and baseline.get("rows") == DML_ROWS:
            print(f"\n=== vs baseline (commit {baseline.get('commit', '?')}) ===")
            base = baseline.get("deltax_ms", {})
            for label, r in report.items():
                b = base.get(label)
                if not b or b <= 0:
                    continue
                ratio = r["deltax_ms"] / b
                flag = ("  ⚠" if (TIME_RATIO_THRESHOLD > 0 and b >= TIME_FLOOR_MS
                                  and ratio > TIME_RATIO_THRESHOLD) else "")
                if b >= TIME_FLOOR_MS or flag:
                    print(f"  {label:<48}{b:>8.2f} → {r['deltax_ms']:>8.2f} ms {ratio:>5.2f}x{flag}")
                if flag:
                    violations.append(label)
        else:
            print("\n  No comparable DML baseline — record one with `make bench-dml-bless`.")

        save_bench_results("dml_pg_deltax", {
            "rows": DML_ROWS,
            "targets": {k: str(v) for k, v in tgt.items()},
            "operations": report,
            "read_cliff": cliff,
            "regression_violations": violations,
            "commit": _get_git_commit_short(),
        })

        assert cliff["after_compaction_ms"] <= cliff["after_1_insert_ms"] * 1.5 + 1.0, (
            f"compaction did not restore the aggregate fast path: {cliff}")
        assert TIME_RATIO_THRESHOLD <= 0 or not violations, (
            f"{len(violations)} DML op(s) regressed >{TIME_RATIO_THRESHOLD:g}x vs baseline: "
            f"{violations}. Re-bless with `make bench-dml-bless` or override DML_TIME_RATIO.")
