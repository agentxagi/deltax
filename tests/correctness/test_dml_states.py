"""Correctness against the oracle with DML-mutated physical state.

The rest of the correctness harness runs its oracle-compared query battery
against a freshly-compressed (pristine) table. But DML changes the physical
representation a read path sees — loose rows in the partition heap, tombstones
subtracted on the fly, decomposed segments, post-compaction segments — and
that is exactly where a scan / aggregate / Top-N path's DML-awareness could
silently diverge from PostgreSQL (a tombstone not subtracted, a heap-tail row
missed, whole-segment-drop dropping too much).

This suite closes that gap by mutating BOTH twins identically — so the plain
PostgreSQL table stays the oracle — and then re-running the curated query
battery against each mutated state. Any read path that mishandles the mutated
state fails the oracle comparison.

Each state is independent (built fresh from the pristine pair) and asserts the
physical state it claims to exercise actually exists, so the coverage can't
silently degrade into "everything decomposed back to a plain heap".
"""

import pytest

from .datasets import BASE_TS, MOCK_NOW, create_tiny_events_pair
from .harness import run_query_case
from .querygen import curated_smoke_cases

pytestmark = pytest.mark.extended


def _apply_both(conn, plain, deltax, stmt):
    """Run a `{table}`-placeholder statement against both twins."""
    conn.execute(stmt.format(table=plain))
    conn.execute(stmt.format(table=deltax))


# New rows whose ts falls inside the (already-compressed) day-0 partition, with
# ids well above the loaded range so they don't collide. Same column
# expressions as the loader, shifted, so both twins get identical rows.
_LOOSE_INSERT = f"""
    INSERT INTO {{table}} (ts, id, device_id, kind, val, metric)
    SELECT
        '{BASE_TS}'::timestamptz + (i * interval '1 minute'),
        i,
        CASE WHEN i % 11 = 0 THEN NULL ELSE i % 5 END,
        CASE WHEN i % 3 = 0 THEN 'alpha' WHEN i % 3 = 1 THEN 'beta' ELSE 'gamma' END,
        CASE WHEN i % 17 = 0 THEN NULL ELSE (i % 23) - 11 END,
        CASE WHEN i % 19 = 0 THEN NULL ELSE (i::float8 / 10.0) END
    FROM generate_series(200, 229) AS g(i)
"""


def _state_pristine(conn, plain, deltax):
    pass  # baseline: confirms the battery + oracle agree with no DML


def _state_loose_inserts(conn, plain, deltax):
    _apply_both(conn, plain, deltax, _LOOSE_INSERT)


def _state_tombstone_deletes(conn, plain, deltax):
    # Scattered, batch-evaluable point predicate → tombstone fast path
    # (segments stay intact, rows filtered by offset at read time).
    _apply_both(conn, plain, deltax, "DELETE FROM {table} WHERE val = 5")
    _apply_both(conn, plain, deltax, "DELETE FROM {table} WHERE id = 37")


def _state_decompose_updates(conn, plain, deltax):
    # UPDATEs decompose the covering segments into heap rows, then apply.
    _apply_both(conn, plain, deltax, "UPDATE {table} SET metric = metric + 1 WHERE id = 40")
    _apply_both(conn, plain, deltax, "UPDATE {table} SET kind = 'delta' WHERE device_id = 3")


def _state_mixed(conn, plain, deltax):
    # The realistic messy state: loose rows + tombstones + decomposed segments
    # all present at once.
    _apply_both(conn, plain, deltax, _LOOSE_INSERT)
    _apply_both(conn, plain, deltax, "DELETE FROM {table} WHERE val = 7")
    _apply_both(conn, plain, deltax, "UPDATE {table} SET metric = metric - 2 WHERE id = 22")


def _state_compacted(conn, plain, deltax):
    # Same mutations, then compact the deltax side back into fresh segments.
    # Compaction is deltax-internal and content-preserving, so the plain twin
    # is untouched and stays the oracle; this validates post-compaction reads.
    _state_mixed(conn, plain, deltax)
    conn.commit()
    parts = conn.execute(
        "SELECT partition_name FROM deltax.deltax_partition_info(%s) WHERE is_compressed",
        (deltax,),
    ).fetchall()
    for (part,) in parts:
        conn.execute("SELECT deltax.deltax_compact_partition(%s)", (part,))


DML_STATES = {
    "pristine": _state_pristine,
    "loose_inserts": _state_loose_inserts,
    "tombstone_deletes": _state_tombstone_deletes,
    "decompose_updates": _state_decompose_updates,
    "mixed": _state_mixed,
    "compacted": _state_compacted,
}


def _physical_flags(conn, deltax):
    """(has_loose_rows, has_tombstones) across the deltax table's partitions.

    `deltax_partition.table_name` holds partition names, so scope by the
    deltatable (the parent table name) via `deltatable_id`.
    """
    return conn.execute(
        "SELECT bool_or(p.has_loose_rows), bool_or(p.has_tombstones) "
        "FROM deltax.deltax_partition p "
        "JOIN deltax.deltax_deltatable d ON d.id = p.deltatable_id "
        "WHERE d.table_name = %s",
        (deltax,),
    ).fetchone()


# The physical state each scenario must actually reach, so the coverage can't
# silently collapse to "plain heap" (loose, tombstoned).
_EXPECTED_FLAGS = {
    "pristine": (False, False),
    "loose_inserts": (True, False),
    "tombstone_deletes": (False, True),
    "decompose_updates": (True, False),  # decompose leaves restored rows loose
    "mixed": (True, True),
    "compacted": (False, False),  # compaction restores pristine fast-path state
}


@pytest.mark.parametrize("state_name", list(DML_STATES), ids=list(DML_STATES))
def test_curated_queries_correct_under_dml_state(db, state_name):
    db.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
    plain, deltax = create_tiny_events_pair(db)

    DML_STATES[state_name](db, plain, deltax)
    db.commit()

    # Confirm the mutation actually produced the physical state it claims to,
    # so a regression that makes DML silently decompose everything (or no-op)
    # can't hollow out this coverage.
    assert _physical_flags(db, deltax) == _EXPECTED_FLAGS[state_name], (
        f"state {state_name!r} did not reach its expected physical shape"
    )

    # Re-run the full curated battery against both twins. Both got identical
    # DML, so the plain table remains the oracle.
    failures = []
    for case in curated_smoke_cases():
        result = run_query_case(db, case, plain_table=plain, deltax_table=deltax)
        if not result.comparison.ok:
            failures.append(f"{case.name}: {result.comparison.detail}")
    assert not failures, (
        f"state={state_name}: {len(failures)} query(ies) diverged from the "
        f"oracle:\n" + "\n".join(failures[:10])
    )
