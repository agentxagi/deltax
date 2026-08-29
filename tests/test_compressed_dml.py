"""Integration tests for P2 transparent UPDATE/DELETE on compressed
partitions (decompose-on-write, dev/docs/COMPRESSED_DML.md §5).

Mechanism under test: the ExecutorStart interceptor locates candidate
segments via the read path's pruning machinery (conservative superset),
decomposes ONLY those segments back into ordinary heap rows (meta + sidecar
rows deleted, rows restored into the partition heap), and lets PostgreSQL's
normal UPDATE/DELETE run over heap tuples. MVCC/rollback come free. A DELETE
whose predicate provably covers a whole segment drops it directly without
materializing rows (§5.4) — and still reports the logical row count.

Correctness is asserted by comparing against an identical plain-PostgreSQL
twin table after every step (same pattern as test_compressed_insert.py).
"""

import threading
import time

import psycopg
import pytest

MOCK_NOW = "2025-01-15 12:00:00+00"
BASE_TS = "2025-01-15 00:00:00+00"
DAY2_TS = "2025-01-16 00:00:00+00"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_tables(conn, segment_by=True, days=1):
    """Create a deltax table + a plain twin, load identical data, compress.

    Data layout (per day): 4 devices x 200 points, one point per minute.
    `val` ranges are disjoint per device (device d: d*1000 .. d*1000+199 on
    day 0; +10000 per extra day), so equality quals on `val` are prunable to
    a single segment via colstats minmax.
    """
    conn.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
    for name in ("events", "events_plain"):
        conn.execute(f"""
            CREATE TABLE {name} (
                ts TIMESTAMPTZ NOT NULL,
                device_id TEXT NOT NULL,
                val INT,
                temperature DOUBLE PRECISION
            )
        """)
    conn.execute(
        "SELECT deltax.deltax_create_table('events', 'ts', '1 day'::interval)"
    )
    seg = "ARRAY['device_id']" if segment_by else "ARRAY[]::text[]"
    conn.execute(
        f"SELECT deltax.deltax_enable_compression('events', "
        f"segment_by => {seg}, order_by => ARRAY['ts'])"
    )
    conn.commit()

    rows = []
    for day in range(days):
        for d in range(4):
            for p in range(200):
                rows.append(
                    f"('{BASE_TS}'::timestamptz + interval '{day} days' "
                    f"+ interval '{p} minutes', "
                    f"'device-{d}', {day * 10000 + d * 1000 + p}, "
                    f"{20.0 + d}::float8 + {p}::float8 / 4)"
                )
    for name in ("events", "events_plain"):
        conn.execute(
            f"INSERT INTO {name} (ts, device_id, val, temperature) VALUES "
            + ", ".join(rows)
        )
    conn.commit()


def compress_all(conn, table="events"):
    parts = [
        r[0]
        for r in conn.execute(
            f"SELECT partition_name FROM deltax.deltax_partition_info('{table}') "
            "WHERE partition_name NOT LIKE '%default%'"
        ).fetchall()
    ]
    compressed = []
    for p in parts:
        result = conn.execute(
            f"SELECT deltax.deltax_compress_partition('{p}')"
        ).fetchone()[0]
        if "Compressed" in result:
            compressed.append(p)
    conn.commit()
    assert compressed, "expected at least one compressed partition"
    return compressed


def data_partition(conn, table="events"):
    """The first compressed partition (holds BASE_TS)."""
    return conn.execute(
        f"SELECT partition_name FROM deltax.deltax_partition_info('{table}') "
        "WHERE is_compressed ORDER BY partition_name LIMIT 1"
    ).fetchone()[0]


def segment_count(conn, part):
    return conn.execute(
        f'SELECT count(*) FROM _deltax_compressed."{part.split(".")[-1]}_meta" '
        "WHERE _segment_id > 0"
    ).fetchone()[0]


def dml_flags(conn, part):
    """(has_loose_rows, has_tombstones) from the catalog for a partition."""
    return conn.execute(
        "SELECT has_loose_rows, has_tombstones FROM deltax.deltax_partition "
        f"WHERE table_name = '{part.split('.')[-1]}'"
    ).fetchone()


class TestDmlGateFlags:
    """The `has_loose_rows` / `has_tombstones` catalog flags drive the
    plan/exec fast-path gates (replacing per-partition physical heap probes).
    They must track DML transactionally and be cleared by compaction."""

    def test_flag_lifecycle(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        # Fresh compression: both flags false.
        assert dml_flags(db, part) == (False, False)

        # A transparent INSERT into the compressed partition sets
        # has_loose_rows in the same transaction (BEFORE-INSERT trigger).
        db.execute(
            f"INSERT INTO events (ts, device_id, val, temperature) VALUES "
            f"('{BASE_TS}'::timestamptz + interval '5 minutes', 'device-9', 99999, 1.0)"
        )
        assert dml_flags(db, part)[0] is True, "INSERT must set has_loose_rows"
        db.commit()
        assert dml_flags(db, part)[0] is True

        # Compaction folds the loose row into a segment and clears the flag.
        db.execute(f"SELECT deltax.deltax_compact_partition('{part}')")
        db.commit()
        assert dml_flags(db, part) == (False, False), "compaction must clear has_loose_rows"

        # A tombstone-eligible DELETE sets has_tombstones.
        db.execute("DELETE FROM events WHERE val = 2050")
        assert dml_flags(db, part)[1] is True, "tombstone DELETE must set has_tombstones"
        db.commit()

        # Compaction rewrites the tombstoned segment and clears the flag.
        db.execute(f"SELECT deltax.deltax_compact_partition('{part}')")
        db.commit()
        assert dml_flags(db, part) == (False, False), "compaction must clear has_tombstones"

    def test_insert_rollback_leaves_flag_consistent(self, db):
        """A rolled-back INSERT must not leave has_loose_rows stuck true
        (the flag UPDATE rolls back with the insert; the per-txn memo is
        cleared on abort so a later insert re-sets it)."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        db.execute(
            f"INSERT INTO events (ts, device_id, val, temperature) VALUES "
            f"('{BASE_TS}'::timestamptz + interval '6 minutes', 'device-9', 88888, 1.0)"
        )
        assert dml_flags(db, part)[0] is True
        db.rollback()
        db.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
        db.commit()
        assert dml_flags(db, part) == (False, False), "rolled-back INSERT must not set the flag"

        # A subsequent committed insert still sets it (memo was cleared).
        db.execute(
            f"INSERT INTO events (ts, device_id, val, temperature) VALUES "
            f"('{BASE_TS}'::timestamptz + interval '7 minutes', 'device-9', 77777, 1.0)"
        )
        db.commit()
        assert dml_flags(db, part)[0] is True


def sidecar_segment_ids(conn, part, sidecar):
    name = part.split(".")[-1]
    return {
        r[0]
        for r in conn.execute(
            f'SELECT DISTINCT _segment_id FROM _deltax_compressed."{name}_{sidecar}" '
            "WHERE _segment_id > 0"
        ).fetchall()
    }


def assert_tables_match(conn):
    q = "SELECT ts, device_id, val, temperature FROM {} ORDER BY ts, device_id, val"
    got = conn.execute(q.format("events")).fetchall()
    want = conn.execute(q.format("events_plain")).fetchall()
    assert got == want


QUERIES = [
    ("count_star", "SELECT count(*) FROM {}"),
    ("min_max_sum_val", "SELECT min(val), max(val), sum(val), count(val) FROM {}"),
    ("group_by_device",
     "SELECT device_id, count(*), sum(val) FROM {} GROUP BY device_id ORDER BY device_id"),
    ("point_lookup", "SELECT count(*) FROM {} WHERE val = 1005"),
    # Equality on the time column exercises the per-segment bloom probe's
    # PG-epoch -> Unix-epoch encoding (regression: probes used to hash the
    # raw datum and bloom-rejected every segment).
    ("point_ts", "SELECT count(*) FROM {} WHERE ts = '2025-01-15 01:00:00+00'"),
    ("topn", "SELECT ts, device_id, val FROM {} ORDER BY ts DESC, device_id, val LIMIT 7"),
]


def assert_queries_match(conn):
    for desc, q in QUERIES:
        got = conn.execute(q.format("events")).fetchall()
        want = conn.execute(q.format("events_plain")).fetchall()
        assert got == want, f"{desc}: deltax={got} plain={want}"


def second_connection(conn, **kwargs):
    """Open an independent connection to the same per-test database."""
    info = conn.info
    return psycopg.connect(
        host=info.host,
        port=info.port,
        user=info.user,
        password="postgres",
        dbname=info.dbname,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Twin-equality: every DML statement applied to both tables
# ---------------------------------------------------------------------------

class TestCompressedDmlTwin:
    @pytest.mark.parametrize(
        "stmt",
        [
            # Time-range UPDATE through the parent (prunable on order-by col)
            "UPDATE {} SET temperature = temperature + 100 "
            f"WHERE ts >= '{BASE_TS}'::timestamptz + interval '30 minutes' "
            f"AND ts < '{BASE_TS}'::timestamptz + interval '40 minutes'",
            # Point UPDATE on a non-segby column (colstats-prunable)
            "UPDATE {} SET val = -1 WHERE val = 2050",
            # segment_by equality UPDATE
            "UPDATE {} SET temperature = 0 WHERE device_id = 'device-3'",
            # Multi-qual DELETE
            "DELETE FROM {} WHERE device_id = 'device-1' AND val % 7 = 0",
            # Time-range DELETE
            "DELETE FROM {} WHERE ts < "
            f"'{BASE_TS}'::timestamptz + interval '15 minutes'",
            # Unqualified DELETE (whole-partition retention pattern)
            "DELETE FROM {}",
        ],
        ids=[
            "update_time_range",
            "update_point_val",
            "update_segby_eq",
            "delete_multi_qual",
            "delete_time_range",
            "delete_all",
        ],
    )
    def test_dml_matches_plain_twin(self, db, stmt):
        setup_tables(db)
        compress_all(db)

        got = db.execute(stmt.format("events")).rowcount
        want = db.execute(stmt.format("events_plain")).rowcount
        db.commit()
        assert got == want, f"rowcount: deltax={got} plain={want}"

        assert_tables_match(db)
        assert_queries_match(db)

    def test_dml_without_segment_by(self, db):
        setup_tables(db, segment_by=False)
        compress_all(db)

        for stmt in (
            "UPDATE {} SET temperature = -5 WHERE val = 1010",
            "DELETE FROM {} WHERE device_id = 'device-2'",
        ):
            got = db.execute(stmt.format("events")).rowcount
            want = db.execute(stmt.format("events_plain")).rowcount
            assert got == want
        db.commit()
        assert_tables_match(db)
        assert_queries_match(db)

    def test_update_returning(self, db):
        setup_tables(db)
        compress_all(db)
        q = ("UPDATE {} SET temperature = -1 WHERE val = 1003 "
             "RETURNING ts, device_id, val, temperature")
        got = sorted(db.execute(q.format("events")).fetchall())
        want = sorted(db.execute(q.format("events_plain")).fetchall())
        db.commit()
        assert got == want and len(got) == 1
        assert_tables_match(db)

    def test_delete_returning_whole_partition(self, db):
        # RETURNING disables the whole-segment drop fast path: rows must be
        # decomposed so the executor can emit them.
        setup_tables(db)
        compress_all(db)
        q = "DELETE FROM {} RETURNING device_id, val"
        got = sorted(db.execute(q.format("events")).fetchall())
        want = sorted(db.execute(q.format("events_plain")).fetchall())
        db.commit()
        assert got == want and len(got) == 800
        assert_tables_match(db)

    def test_time_range_delete_whole_drops_not_decompose(self, db):
        """A time-range DELETE that fully covers a partition must take the
        whole-segment-drop fast path, NOT decompose every segment.

        Regression: the time column's colstats `_nonnull_count` is a zero
        placeholder (its real stats live in _meta), so classify_segment_quals
        read 0 non-null rows and returned Ambiguous — defeating the drop, so
        `ts < boundary` retention deletes decomposed/tombstoned every row
        (~80x slower). The discriminator here is `has_loose_rows`: whole-drop
        never inserts into the heap (flag stays false); decompose restores
        rows first (flag flips true)."""
        # No segment_by → segments are time-ordered, so a partition-boundary
        # predicate cleanly covers whole segments. days=2 → two partitions.
        setup_tables(db, segment_by=False, days=2)
        compress_all(db)
        day2 = "2025-01-16 00:00:00+00"
        first_part = db.execute(
            "SELECT partition_name FROM deltax.deltax_partition_info('events') "
            "WHERE is_compressed ORDER BY range_start LIMIT 1"
        ).fetchone()[0]
        segs_before = segment_count(db, first_part)
        assert segs_before > 0

        got = db.execute(f"DELETE FROM events WHERE ts < '{day2}'::timestamptz").rowcount
        want = db.execute(
            f"DELETE FROM events_plain WHERE ts < '{day2}'::timestamptz"
        ).rowcount
        assert got == want > 0
        # Whole-drop: the earliest partition's segments are gone and NOTHING
        # was decomposed into the heap (flag false ⇒ no heap inserts happened).
        assert segment_count(db, first_part) == 0, "earliest partition not dropped"
        assert dml_flags(db, first_part) == (False, False), (
            "time-range retention DELETE decomposed instead of whole-dropping"
        )
        db.commit()
        assert_tables_match(db)

    def test_update_on_loose_rows_only(self, db):
        # An UPDATE matching only loose (P1-inserted) rows must not be
        # blocked and must leave segments alone when pruning rules them out.
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        for name in ("events", "events_plain"):
            db.execute(
                f"INSERT INTO {name} (ts, device_id, val, temperature) VALUES "
                f"('{BASE_TS}'::timestamptz + interval '6 hours', 'device-late', 90001, 1.0)"
            )
        db.commit()
        segs_before = segment_count(db, part)
        got = db.execute("UPDATE events SET temperature = 2 WHERE val = 90001").rowcount
        want = db.execute(
            "UPDATE events_plain SET temperature = 2 WHERE val = 90001"
        ).rowcount
        db.commit()
        assert got == want == 1
        # val=90001 is outside every segment's colstats range → no decompose.
        assert segment_count(db, part) == segs_before
        assert_tables_match(db)

    def test_cross_partition_update(self, db):
        # Moving a row across partitions = delete + routed insert; the
        # insert lands in the target's loose region (P1).
        setup_tables(db, days=2)
        compress_all(db)
        stmt = ("UPDATE {} SET ts = ts + interval '1 day' "
                "WHERE device_id = 'device-0' AND val = 17")
        got = db.execute(stmt.format("events")).rowcount
        want = db.execute(stmt.format("events_plain")).rowcount
        db.commit()
        assert got == want == 1
        assert_tables_match(db)
        assert_queries_match(db)

    def test_merge_when_matched(self, db):
        # MERGE targets are intercepted like UPDATE/DELETE (conservative
        # decompose: join quals aren't pushable, so candidates = the
        # segments the source rows could live in).
        setup_tables(db)
        compress_all(db)
        merge = (
            "MERGE INTO {} t USING (VALUES (1003, -7.5), (2104, -8.5)) "
            "AS s(val, new_temp) ON t.val = s.val "
            "WHEN MATCHED THEN UPDATE SET temperature = s.new_temp"
        )
        got = db.execute(merge.format("events")).rowcount
        want = db.execute(merge.format("events_plain")).rowcount
        db.commit()
        assert got == want == 2
        assert_tables_match(db)


# ---------------------------------------------------------------------------
# Decompose granularity / fast paths
# ---------------------------------------------------------------------------

class TestDecomposeGranularity:
    def test_targeted_update_decomposes_single_segment(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        segs_before = segment_count(db, part)
        assert segs_before == 4  # one per device (200 rows < segment_size)

        # val = 2050 exists only in device-2's segment (disjoint ranges).
        cur = db.execute("UPDATE events SET temperature = -3 WHERE val = 2050")
        assert cur.rowcount == 1
        db.execute("UPDATE events_plain SET temperature = -3 WHERE val = 2050")
        db.commit()

        # Exactly one segment was decomposed; its sidecar rows went with it.
        assert segment_count(db, part) == segs_before - 1
        for sidecar in ("colstats", "blobs"):
            assert len(sidecar_segment_ids(db, part, sidecar)) == segs_before - 1

        # Its 200 rows (199 untouched + 1 updated) now live in the heap:
        # total visible rows minus rows still accounted to segments. (A
        # direct `FROM ONLY` count goes through the union scan, so probe
        # via segment metadata + physical size.)
        seg_rows = db.execute(
            f'SELECT COALESCE(sum(_row_count), 0) FROM _deltax_compressed."{part.split(".")[-1]}_meta" '
            "WHERE _segment_id > 0"
        ).fetchone()[0]
        total = db.execute("SELECT count(*) FROM events").fetchone()[0]
        assert total - seg_rows == 200
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] > 0

        # The partition still EXPLAINs/queries through the union scan.
        plan = "\n".join(
            r[0] for r in db.execute("EXPLAIN SELECT * FROM events").fetchall()
        )
        assert "DeltaX" in plan
        assert_tables_match(db)
        assert_queries_match(db)

    def test_segby_update_decomposes_single_segment(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)

        db.execute("UPDATE events SET temperature = 9 WHERE device_id = 'device-0'")
        db.execute("UPDATE events_plain SET temperature = 9 WHERE device_id = 'device-0'")
        db.commit()
        assert segment_count(db, part) == segs_before - 1
        assert_tables_match(db)

    def test_delete_whole_partition_uses_direct_drop(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        assert segment_count(db, part) == 4

        cur = db.execute(f"DELETE FROM {part}")
        # Command tag reports the LOGICAL row count even though the
        # segments were dropped without materializing rows.
        assert cur.rowcount == 800
        db.execute("DELETE FROM events_plain")
        db.commit()

        assert segment_count(db, part) == 0
        # Fast-path proof: rows were never written to the heap.
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        for sidecar in ("colstats", "blobs"):
            assert sidecar_segment_ids(db, part, sidecar) == set()
        assert_tables_match(db)
        assert_queries_match(db)

    def test_delete_time_range_covering_one_segment(self, db):
        # Without segment_by, segments split by row count; a time predicate
        # covering one whole segment should drop it directly.
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        # device-0's segment spans minutes 0..199. Delete exactly that
        # device via a covering predicate is segby — use the val range
        # instead: [0, 199] covers device-0's whole segment.
        cur = db.execute("DELETE FROM events WHERE val >= 0 AND val <= 199")
        assert cur.rowcount == 200
        db.execute("DELETE FROM events_plain WHERE val >= 0 AND val <= 199")
        db.commit()

        assert segment_count(db, part) == 3
        # Direct drop: no rows materialized.
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        assert_tables_match(db)
        assert_queries_match(db)

    def test_catalog_counters_updated(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        name = part.split(".")[-1]

        rc_before, size_before = db.execute(
            "SELECT row_count, compressed_size FROM deltax.deltax_partition "
            f"WHERE table_name = '{name}'"
        ).fetchone()
        assert rc_before == 800

        db.execute("UPDATE events SET temperature = 1 WHERE device_id = 'device-2'")
        db.commit()

        rc_after, size_after = db.execute(
            "SELECT row_count, compressed_size FROM deltax.deltax_partition "
            f"WHERE table_name = '{name}'"
        ).fetchone()
        assert rc_after == 600  # one 200-row segment left the compressed set
        assert size_after < size_before


# ---------------------------------------------------------------------------
# Transactionality
# ---------------------------------------------------------------------------

class TestDmlTransactionality:
    def test_rollback_mid_update_leaves_everything_intact(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)
        blobs_before = sidecar_segment_ids(db, part, "blobs")

        cur = db.execute("UPDATE events SET temperature = -99 WHERE val = 1005")
        assert cur.rowcount == 1
        db.rollback()
        db.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
        db.commit()

        # Segment + sidecars restored by MVCC; heap insert rolled back.
        assert segment_count(db, part) == segs_before
        assert sidecar_segment_ids(db, part, "blobs") == blobs_before
        assert db.execute(
            "SELECT count(*) FROM events WHERE temperature = -99"
        ).fetchone()[0] == 0
        assert_tables_match(db)
        assert_queries_match(db)

    def test_rollback_whole_partition_delete(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        cur = db.execute(f"DELETE FROM {part}")
        assert cur.rowcount == 800
        db.rollback()
        db.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
        db.commit()

        assert segment_count(db, part) == 4
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 800
        assert_tables_match(db)

    def test_update_then_compaction_recompresses(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        db.execute("UPDATE events SET temperature = 55 WHERE val = 3100")
        db.execute("UPDATE events_plain SET temperature = 55 WHERE val = 3100")
        db.commit()
        # The decomposed segment's rows are loose in the partition heap.
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] > 0
        assert segment_count(db, part) == 3

        result = db.execute(
            f"SELECT deltax.deltax_compact_partition('{part}')"
        ).fetchone()[0]
        db.commit()
        assert "Compacted" in result

        # Loose region empty again; rows live in fresh segments.
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        assert segment_count(db, part) == 4
        assert_tables_match(db)
        assert_queries_match(db)

    def test_compaction_never_reuses_decomposed_segment_ids(self, db):
        # P2 invariant: shared caches are keyed (companion_oid, segment_id);
        # decompose records a high-water mark so compaction can't recycle a
        # deleted id. Decompose the max-id segment, recompact, verify the
        # new ids are strictly greater.
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        name = part.split(".")[-1]

        max_id_before = db.execute(
            f'SELECT max(_segment_id) FROM _deltax_compressed."{name}_meta"'
        ).fetchone()[0]
        # device-3 holds vals 3000..3199 and is the last-flushed segment.
        db.execute("UPDATE events SET temperature = 7 WHERE device_id = 'device-3'")
        db.execute("UPDATE events_plain SET temperature = 7 WHERE device_id = 'device-3'")
        db.commit()
        assert db.execute(
            f'SELECT count(*) FROM _deltax_compressed."{name}_meta" '
            f"WHERE _segment_id = {max_id_before}"
        ).fetchone()[0] == 0

        db.execute(f"SELECT deltax.deltax_compact_partition('{part}')")
        db.commit()
        new_ids = [
            r[0]
            for r in db.execute(
                f'SELECT _segment_id FROM _deltax_compressed."{name}_meta" '
                f"WHERE _segment_id >= {max_id_before}"
            ).fetchall()
        ]
        assert new_ids and all(i > max_id_before for i in new_ids)
        assert_tables_match(db)
        assert_queries_match(db)


# ---------------------------------------------------------------------------
# P2.5 tombstone DELETE fast layer (dev/docs/COMPRESSED_DML.md §P2.5)
# ---------------------------------------------------------------------------

def tombstone_rows(conn, part):
    """All visible tombstone rows for a partition, as {(segment_id, offset)}."""
    name = part.split(".")[-1]
    exists = conn.execute(
        f"SELECT to_regclass('_deltax_compressed.\"{name}_tombstones\"') IS NOT NULL"
    ).fetchone()[0]
    if not exists:
        return set()
    return set(
        conn.execute(
            f'SELECT _segment_id, _row_offset FROM _deltax_compressed."{name}_tombstones"'
        ).fetchall()
    )


EXTRA_QUERIES = QUERIES + [
    ("select_star", "SELECT ts, device_id, val, temperature FROM {} ORDER BY ts, device_id, val"),
    ("min_max_ts", "SELECT min(ts), max(ts) FROM {}"),
    ("avg_temp", "SELECT avg(temperature)::numeric(20,6) FROM {}"),
    ("count_where_time",
     f"SELECT count(*) FROM {{}} WHERE ts >= '{BASE_TS}'::timestamptz"),
    ("groupby_having",
     "SELECT device_id, min(val), max(val) FROM {} GROUP BY device_id "
     "HAVING count(*) > 0 ORDER BY device_id"),
    ("topn_asc", "SELECT ts, device_id, val FROM {} ORDER BY ts ASC, device_id LIMIT 5"),
]


def assert_extra_queries_match(conn):
    for desc, q in EXTRA_QUERIES:
        got = conn.execute(q.format("events")).fetchall()
        want = conn.execute(q.format("events_plain")).fetchall()
        assert got == want, f"{desc}: deltax={got} plain={want}"


class TestTombstoneDelete:
    def test_single_row_delete_uses_tombstone_fast_path(self, db):
        """The headline P2.5 mechanism assertion: a single-row DELETE with an
        exactly-evaluable predicate leaves every segment intact (no
        decompose), writes nothing to the partition heap, and records exactly
        one tombstone row — while the command tag and all reads stay exact."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)
        blobs_before = sidecar_segment_ids(db, part, "blobs")
        rc_before = db.execute(
            "SELECT row_count FROM deltax.deltax_partition "
            f"WHERE table_name = '{part.split('.')[-1]}'"
        ).fetchone()[0]

        cur = db.execute("DELETE FROM events WHERE val = 1005")
        assert cur.rowcount == 1
        db.execute("DELETE FROM events_plain WHERE val = 1005")
        db.commit()

        # Mechanism: no segment was decomposed or dropped, no sidecar rows
        # were touched, and the partition heap is still physically empty.
        assert segment_count(db, part) == segs_before
        assert sidecar_segment_ids(db, part, "blobs") == blobs_before
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        # Exactly one tombstone row, in exactly one segment.
        tombs = tombstone_rows(db, part)
        assert len(tombs) == 1
        # Catalog row_count tracks live rows in segments — decremented.
        rc_after = db.execute(
            "SELECT row_count FROM deltax.deltax_partition "
            f"WHERE table_name = '{part.split('.')[-1]}'"
        ).fetchone()[0]
        assert rc_after == rc_before - 1

        assert_tables_match(db)
        assert_extra_queries_match(db)

    def test_reads_with_tombstones_every_query_shape(self, db):
        """Tombstone the extremum rows (min ts, max ts, max val) plus one
        mid-segment row per device, then verify every query shape against
        the twin — count/minmax/agg/groupby/point/topn/SELECT *."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)

        # max(val)=3199, min(ts) row of device-0 (val=0), max(ts) row
        # (p=199 of every device) — extremum deletions are exactly what
        # breaks naive metadata answers.
        stmts = [
            "DELETE FROM {} WHERE val = 3199",                  # max(val), max(ts) of d3
            "DELETE FROM {} WHERE val = 0",                     # min(val), min(ts) of d0
            "DELETE FROM {} WHERE val IN (1100, 2100, 150)",    # mid-segment, IN list
            # ts+segby point — NB: a literal timestamp; `ts = const +
            # interval` would not be plan-time folded (the + operator is
            # STABLE) and correctly falls back to decompose.
            "DELETE FROM {} WHERE ts = '2025-01-15 03:19:00+00'::timestamptz "
            "AND device_id = 'device-1'",
        ]
        for stmt in stmts:
            got = db.execute(stmt.format("events")).rowcount
            want = db.execute(stmt.format("events_plain")).rowcount
            assert got == want, f"rowcount mismatch for {stmt}"
        db.commit()

        # All of the above were tombstone-eligible: segments intact.
        assert segment_count(db, part) == segs_before
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        assert len(tombstone_rows(db, part)) == 6

        assert_tables_match(db)
        assert_extra_queries_match(db)
        # Point lookup on a tombstoned value: zero rows.
        assert db.execute("SELECT count(*) FROM events WHERE val = 3199").fetchone()[0] == 0
        assert db.execute("SELECT * FROM events WHERE val = 0").fetchall() == []

    def test_tombstone_delete_rollback(self, db):
        """Tombstones are ordinary heap rows: rollback restores everything,
        including the catalog row_count decrement."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        name = part.split(".")[-1]
        rc_before = db.execute(
            f"SELECT row_count FROM deltax.deltax_partition WHERE table_name = '{name}'"
        ).fetchone()[0]

        cur = db.execute("DELETE FROM events WHERE val = 2050")
        assert cur.rowcount == 1
        # Inside the transaction the row is gone.
        assert db.execute("SELECT count(*) FROM events WHERE val = 2050").fetchone()[0] == 0
        db.rollback()
        db.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
        db.commit()

        assert tombstone_rows(db, part) == set()
        assert db.execute("SELECT count(*) FROM events WHERE val = 2050").fetchone()[0] == 1
        rc_after = db.execute(
            f"SELECT row_count FROM deltax.deltax_partition WHERE table_name = '{name}'"
        ).fetchone()[0]
        assert rc_after == rc_before
        assert_tables_match(db)
        assert_extra_queries_match(db)

    def test_compaction_rewrites_tombstoned_segments(self, db):
        """deltax_compact_partition() physically rewrites tombstone-bearing
        segments (decompose minus dead rows + recompress) and TRUNCATEs the
        tombstones table back to the zero-block steady state."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        name = part.split(".")[-1]

        for stmt in (
            "DELETE FROM {} WHERE val = 1005",
            "DELETE FROM {} WHERE val IN (2000, 2001, 2002)",
        ):
            db.execute(stmt.format("events"))
            db.execute(stmt.format("events_plain"))
        db.commit()
        assert len(tombstone_rows(db, part)) == 4
        assert segment_count(db, part) == 4

        db.execute(f"SELECT deltax.deltax_compact_partition('{part}')")
        db.commit()

        # Tombstones consumed; table truncated back to zero blocks so the
        # steady-state gate (nblocks probe) reports clean.
        assert tombstone_rows(db, part) == set()
        assert db.execute(
            f"SELECT pg_relation_size('_deltax_compressed.\"{name}_tombstones\"')"
        ).fetchone()[0] == 0
        # Rows live in fresh segments again; loose region empty.
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        seg_rows = db.execute(
            f'SELECT COALESCE(sum(_row_count), 0) FROM _deltax_compressed."{name}_meta" '
            "WHERE _segment_id > 0"
        ).fetchone()[0]
        assert seg_rows == 800 - 4
        assert_tables_match(db)
        assert_extra_queries_match(db)
        # Pushdowns re-engage after compaction: metadata answers stay right.
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 796

    def test_worker_style_auto_compaction_covers_tombstones(self, db):
        """A fully tombstoned segment is removed outright by compaction even
        when there are no loose rows to fold (the worker triggers on a
        non-empty tombstones table alone)."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        # Tombstone an entire segment's rows via a non-AllPass-provable
        # predicate? val range [0,199] covers device-0 wholly and is
        # provable → whole-drop. Use modulo-free range minus one row, then
        # the last row separately, so both statements take the tombstone
        # path (not AllPass).
        db.execute("DELETE FROM events WHERE val >= 0 AND val <= 198")
        db.execute("DELETE FROM events WHERE val = 199")
        db.execute("DELETE FROM events_plain WHERE val >= 0 AND val <= 199")
        db.commit()
        assert segment_count(db, part) in (3, 4)  # whole-drop may have fired

        result = db.execute(
            f"SELECT deltax.deltax_compact_partition('{part}')"
        ).fetchone()[0]
        db.commit()
        assert "tombstoned segment" in result or "Compacted" in result
        assert tombstone_rows(db, part) == set()
        assert_tables_match(db)
        assert_extra_queries_match(db)

    def test_mixed_tombstones_and_heap_tail(self, db):
        """Loose (P1-inserted) rows + tombstoned segment rows in the same
        partition: one DELETE statement removes one of each; every read
        unions heap tail and filters tombstones correctly."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)

        for name in ("events", "events_plain"):
            db.execute(
                f"INSERT INTO {name} (ts, device_id, val, temperature) VALUES "
                f"('{BASE_TS}'::timestamptz + interval '6 hours', 'device-0', 1005, 1.0), "
                f"('{BASE_TS}'::timestamptz + interval '7 hours', 'device-9', 4242, 2.0)"
            )
        db.commit()

        # val=1005 matches one compressed row AND one loose heap row: the
        # tombstone path handles the segment side, the planned heap DELETE
        # handles the loose row — the command tag must report both.
        got = db.execute("DELETE FROM events WHERE val = 1005").rowcount
        want = db.execute("DELETE FROM events_plain WHERE val = 1005").rowcount
        db.commit()
        assert got == want == 2
        assert segment_count(db, part) == segs_before
        assert len(tombstone_rows(db, part)) == 1

        assert_tables_match(db)
        assert_extra_queries_match(db)

        # Compaction folds the remaining loose row AND rewrites the
        # tombstoned segment in one pass.
        db.execute(f"SELECT deltax.deltax_compact_partition('{part}')")
        db.commit()
        assert tombstone_rows(db, part) == set()
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        assert_tables_match(db)
        assert_extra_queries_match(db)

    def test_delete_returning_falls_back_to_decompose(self, db):
        """RETURNING must observe the deleted rows — tombstones can't serve
        it, so the statement decomposes (P2) and stays correct."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)

        q = "DELETE FROM {} WHERE val = 1005 RETURNING device_id, val"
        got = db.execute(q.format("events")).fetchall()
        want = db.execute(q.format("events_plain")).fetchall()
        db.commit()
        assert got == want == [("device-1", 1005)]
        # Mechanism: decompose, not tombstone.
        assert segment_count(db, part) == segs_before - 1
        assert tombstone_rows(db, part) == set()
        assert_tables_match(db)

    def test_delete_unextractable_qual_falls_back_to_decompose(self, db):
        """A predicate the batch machinery can't fully evaluate (val % 7)
        must decompose — never guess."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)

        got = db.execute(
            "DELETE FROM events WHERE device_id = 'device-1' AND val % 7 = 0"
        ).rowcount
        want = db.execute(
            "DELETE FROM events_plain WHERE device_id = 'device-1' AND val % 7 = 0"
        ).rowcount
        db.commit()
        assert got == want
        assert tombstone_rows(db, part) == set()
        assert_tables_match(db)
        assert_queries_match(db)

    def test_repeat_delete_same_segment_accumulates_tombstones(self, db):
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        segs_before = segment_count(db, part)

        for v in (1001, 1002, 1003):
            got = db.execute(f"DELETE FROM events WHERE val = {v}").rowcount
            want = db.execute(f"DELETE FROM events_plain WHERE val = {v}").rowcount
            assert got == want == 1
        # Idempotence: deleting an already-tombstoned row deletes nothing.
        assert db.execute("DELETE FROM events WHERE val = 1002").rowcount == 0
        db.commit()

        assert segment_count(db, part) == segs_before
        assert len(tombstone_rows(db, part)) == 3
        assert_tables_match(db)
        assert_extra_queries_match(db)

    def test_single_row_delete_latency(self, db):
        """P2.5 requirement: tombstone DELETE beats decompose by >10x on the
        single-row case. Same partition, same segment shape; decompose is
        forced via RETURNING (which disables the tombstone fast path)."""
        db.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
        for name in ("big", "big_plain"):
            db.execute(
                f"CREATE TABLE {name} (ts TIMESTAMPTZ NOT NULL, val INT, payload TEXT)"
            )
        db.execute("SELECT deltax.deltax_create_table('big', 'ts', '1 day'::interval)")
        db.execute(
            "SELECT deltax.deltax_enable_compression('big', "
            "segment_by => ARRAY[]::text[], order_by => ARRAY['ts'])"
        )
        db.commit()
        for name in ("big", "big_plain"):
            db.execute(
                f"INSERT INTO {name} (ts, val, payload) "
                f"SELECT '{BASE_TS}'::timestamptz + (i || ' seconds')::interval, "
                "i, 'payload-' || i FROM generate_series(0, 59999) AS i"
            )
        db.commit()
        compress_all(db, table="big")
        part = data_partition(db, table="big")
        segs_before = segment_count(db, part)

        # Tombstone path.
        t0 = time.perf_counter()
        assert db.execute("DELETE FROM big WHERE val = 12345").rowcount == 1
        db.commit()
        tombstone_s = time.perf_counter() - t0
        assert segment_count(db, part) == segs_before  # fast path proof

        # Decompose path (RETURNING disables tombstones) on a row in a
        # different segment so the previous statement's work doesn't help.
        t0 = time.perf_counter()
        rows = db.execute(
            "DELETE FROM big WHERE val = 45678 RETURNING val"
        ).fetchall()
        db.commit()
        decompose_s = time.perf_counter() - t0
        assert rows == [(45678,)]
        assert segment_count(db, part) < segs_before  # decompose proof

        # Generous margin for CI noise; locally the ratio is far higher.
        assert decompose_s > 10 * tombstone_s, (
            f"tombstone={tombstone_s * 1000:.1f}ms "
            f"decompose={decompose_s * 1000:.1f}ms"
        )

        db.execute("DELETE FROM big_plain WHERE val IN (12345, 45678)")
        db.commit()
        q = "SELECT count(*), min(val), max(val), sum(val::int8) FROM {}"
        assert db.execute(q.format("big")).fetchone() == \
            db.execute(q.format("big_plain")).fetchone()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestDmlConcurrency:
    def test_concurrent_reader_sees_consistent_data(self, db):
        """A reader concurrent with a decomposing UPDATE sees either the
        full pre-state or the full post-state — never a partial mix.

        The plan/exec DML gates read the `has_loose_rows` / `has_tombstones`
        catalog flags rather than physically probing the partition heap, so
        an analytic reader no longer takes an ACCESS SHARE lock on the
        partition during planning and therefore does NOT block behind the
        decompose's ACCESS EXCLUSIVE lock (standard PostgreSQL MVCC: readers
        don't block writers). Because the writer's transaction is still open
        when the reader runs, the reader's snapshot — and the committed
        catalog flags it reads — both reflect the pre-state, so it reads a
        consistent pre-image. The invariant under test is "pre OR post,
        never torn"; the flag's MVCC atomicity with the decomposed rows is
        what guarantees the two never disagree.
        """
        setup_tables(db)
        compress_all(db)

        expected_before = db.execute(
            "SELECT count(*), sum(val), "
            "count(*) FILTER (WHERE temperature = -42) FROM events"
        ).fetchone()
        results = {}

        def reader():
            with second_connection(db) as rconn:
                rows = rconn.execute(
                    "SELECT count(*), sum(val), "
                    "count(*) FILTER (WHERE temperature = -42) FROM events"
                ).fetchone()
                results["row"] = rows

        db.execute("UPDATE events SET temperature = -42 WHERE device_id = 'device-1'")
        db.execute("UPDATE events_plain SET temperature = -42 WHERE device_id = 'device-1'")
        expected_after = db.execute(
            "SELECT count(*), sum(val), "
            "count(*) FILTER (WHERE temperature = -42) FROM events_plain"
        ).fetchone()

        # The reader runs while the writing transaction is still open. It
        # does not block (MVCC read), and must observe a self-consistent
        # snapshot — never a partial mix of decomposed and still-segmented
        # rows.
        t = threading.Thread(target=reader)
        t.start()
        t.join(timeout=30)
        assert not t.is_alive(), "reader never finished"
        assert results["row"] in (expected_before, expected_after), (
            f"reader saw a torn state: {results['row']} "
            f"(pre={expected_before}, post={expected_after})"
        )
        # Deterministically the pre-image, since the writer is uncommitted.
        assert results["row"] == expected_before
        db.commit()
        assert_tables_match(db)

    def test_repeatable_read_reader_keeps_pre_state(self, db):
        """A REPEATABLE READ snapshot taken before the decompose commits
        keeps seeing the segment (MVCC: meta/sidecar rows are ordinary
        tuples; the decomposed heap rows are invisible to the old
        snapshot)."""
        setup_tables(db)
        compress_all(db)

        pre = db.execute(
            "SELECT count(*), sum(val), min(temperature) FROM events"
        ).fetchone()

        with second_connection(db) as rconn:
            rconn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            # Materialize the snapshot before the writer touches anything.
            # Use the twin table: snapshots are database-wide, but touching
            # `events` here would leave an ACCESS SHARE lock on the
            # partition that the writer's decompose lock (ACCESS EXCLUSIVE)
            # would block on — an honest-to-goodness lock conflict, not the
            # MVCC behavior under test.
            assert rconn.execute("SELECT count(*) FROM events_plain").fetchone()[0] == pre[0]

            db.execute("UPDATE events SET temperature = -77 WHERE device_id = 'device-2'")
            db.commit()

            got = rconn.execute(
                "SELECT count(*), sum(val), min(temperature) FROM events"
            ).fetchone()
            rconn.commit()
            assert got == pre

        # A fresh snapshot sees the post-state.
        assert db.execute(
            "SELECT count(*) FROM events WHERE temperature = -77"
        ).fetchone()[0] == 200


# ---------------------------------------------------------------------------
# Observers of deleted rows: CTEs, RETURNING, transition tables
# ---------------------------------------------------------------------------

class TestDeleteObservers:
    def test_cte_delete_returning_yields_every_row(self, db):
        """A data-modifying CTE's DELETE ... RETURNING feeds the outer query,
        so the whole-segment-drop / tombstone fast paths must be disabled
        even though the TOP-level statement (a SELECT) has no RETURNING —
        every deleted row must be materialized and flow through the CTE."""
        setup_tables(db)
        compress_all(db)

        q = ("WITH d AS (DELETE FROM {} WHERE device_id = 'device-1' "
             "RETURNING val) SELECT count(*), sum(val) FROM d")
        got = db.execute(q.format("events")).fetchone()
        want = db.execute(q.format("events_plain")).fetchone()
        db.commit()
        assert got == want == (200, sum(range(1000, 1200)))
        assert_tables_match(db)
        assert_queries_match(db)

    def test_cte_delete_without_returning_keeps_outer_tag(self, db):
        """`WITH d AS (DELETE ...) SELECT 1`: the rows are deleted, but the
        outer SELECT's command tag must not absorb the CTE's row count
        (PostgreSQL reports SELECT 1 regardless of CTE-deleted rows)."""
        setup_tables(db)
        compress_all(db)

        cur = db.execute("WITH d AS (DELETE FROM events) SELECT 1")
        assert cur.fetchall() == [(1,)]
        assert cur.statusmessage == "SELECT 1"
        db.execute("WITH d AS (DELETE FROM events_plain) SELECT 1")
        db.commit()
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 0
        assert_tables_match(db)

    def test_delete_with_transition_table_sees_every_row(self, db):
        """An AFTER DELETE statement trigger with REFERENCING OLD TABLE must
        observe every deleted row — the fast paths (whole-drop, tombstone)
        never materialize rows, so they must be disabled."""
        setup_tables(db)
        compress_all(db)
        part = data_partition(db)
        db.execute("""
            CREATE TABLE delete_audit (n BIGINT, sum_val BIGINT);
            CREATE FUNCTION audit_deletes() RETURNS trigger
            LANGUAGE plpgsql AS $$
            BEGIN
                INSERT INTO delete_audit
                SELECT count(*), COALESCE(sum(val), 0) FROM old_rows;
                RETURN NULL;
            END $$;
            CREATE TRIGGER events_delete_audit
                AFTER DELETE ON events
                REFERENCING OLD TABLE AS old_rows
                FOR EACH STATEMENT EXECUTE FUNCTION audit_deletes();
        """)
        db.commit()

        # Covers one whole segment (would be a whole-drop candidate) plus a
        # tombstone-eligible single row in another segment.
        cur = db.execute("DELETE FROM events WHERE val <= 199 OR val = 2050")
        assert cur.rowcount == 201
        db.execute("DELETE FROM events_plain WHERE val <= 199 OR val = 2050")
        db.commit()

        assert db.execute(
            "SELECT n, sum_val FROM delete_audit"
        ).fetchall() == [(201, sum(range(200)) + 2050)]
        # Mechanism: the transition table forced decompose for both segments.
        assert tombstone_rows(db, part) == set()
        assert_tables_match(db)
        assert_queries_match(db)
