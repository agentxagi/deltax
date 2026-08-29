"""DML-on-compressed × columnar-format coverage.

The other DML test files (test_compressed_insert / test_compressed_dml) use a
narrow schema (ts, text, int, float8), so they exercise only a few codecs.
This file guards the interaction the columnar format is most likely to break
as codecs change: a DML operation that round-trips a value through
compression must reproduce it exactly, for EVERY type/codec.

Approach: a wide table with one column per codec-relevant type (+ NULLs and
edge values), loaded across several segments, with a byte-identical
plain-PostgreSQL twin. Every test applies DML to both and asserts full-row
equality — and the decompose/decompress paths (which reconstruct rows from
the compressed blobs) are exercised on real multi-column, multi-row segments.

If you change a codec or the on-disk format and any DML round-trip regresses,
one of these fails.
"""

from __future__ import annotations

MOCK_NOW = "2025-01-15 12:00:00+00"
BASE_TS = "2025-01-15 00:00:00+00"
NROWS = 2000
SEGMENT_SIZE = 500  # → NROWS / SEGMENT_SIZE = 4 segments in the one partition

# Columns chosen to hit distinct codecs / format paths:
#   i2/i4/i8    integer delta-varint + frame-of-reference bitpacking
#   f4/f8       Gorilla float
#   t_hi        high-cardinality text → LZ4
#   t_lo        low-cardinality text → Dictionary / DictionaryLz4
#   vc / bc     varchar / char(n) (bpchar trailing-space semantics)
#   flag        BOOLEAN bitmap
#   d           DATE encoding
#   js          JSONB (no json_extract — see test_compressed_insert for that)
#   konst       CONSTANT column (single distinct value → Constant codec)
# Nullable columns (n_i4, n_txt, n_f8, n_flag) carry NULLs to exercise the
# null-bitmap extract/reinsert on decompress.
COLS = ("id, ts, i2, i4, i8, f4, f8, t_hi, t_lo, vc, bc, flag, d, "
        "js, konst, n_i4, n_txt, n_f8, n_flag")

SCHEMA = """
CREATE TABLE {name} (
    id     bigint      NOT NULL,
    ts     timestamptz NOT NULL,
    i2     smallint,
    i4     integer,
    i8     bigint,
    f4     real,
    f8     double precision,
    t_hi   text,
    t_lo   text,
    vc     varchar(20),
    bc     char(6),
    flag   boolean,
    d      date,
    js     jsonb,
    konst  integer,
    n_i4   integer,
    n_txt  text,
    n_f8   double precision,
    n_flag boolean
)
"""

# Deterministic generator, run identically into both twins so the compared
# rows are byte-identical by construction. Edge values are seeded via the
# arithmetic (negative i2 near its bounds, large i8, etc.); NULLs land on a
# fixed residue so every codec sees some.
GEN = f"""
    SELECT g,
           '{BASE_TS}'::timestamptz + (g || ' seconds')::interval,
           (g % 65536 - 32768)::smallint,
           (g - 1000) * 7,
           g::bigint * 1000000007,
           ((g % 200) - 100)::real / 3,
           g::float8 / 7 - 500,
           't-' || md5(g::text),
           'grp-' || (g % 8),
           ('vc-' || (g % 5))::varchar(20),
           ('c' || (g % 4))::char(6),
           (g % 2 = 0),
           '2025-01-15'::date + (g % 90),
           jsonb_build_object('k', g % 10, 'tag', 'v' || (g % 3)),
           42,
           CASE WHEN g % 17 = 0 THEN NULL ELSE g * 3 END,
           CASE WHEN g % 11 = 0 THEN NULL ELSE 'n-' || (g % 6) END,
           CASE WHEN g % 13 = 0 THEN NULL ELSE g::float8 / 2 END,
           CASE WHEN g % 7 = 0 THEN NULL ELSE (g % 3 = 0) END
    FROM generate_series(1, {NROWS}) g
"""

# All columns, deterministically ordered, for full-row twin comparison.
_ALL = ("id, ts, i2, i4, i8, f4, f8, t_hi, t_lo, vc, bc, flag, d, "
        "js::text, konst, n_i4, n_txt, n_f8, n_flag")


def _part(conn):
    return conn.execute(
        "SELECT partition_name FROM deltax.deltax_partition_info('wide') "
        "WHERE is_compressed ORDER BY range_start LIMIT 1"
    ).fetchone()[0]


def _segcount(conn, part):
    return conn.execute(
        f'SELECT count(*) FROM _deltax_compressed."{part.split(".")[-1]}_meta" '
        "WHERE _segment_id > 0"
    ).fetchone()[0]


def setup_wide(conn):
    """Both twins, identical data, `wide` compressed into ~4 segments."""
    conn.execute(f"SET pg_deltax.mock_now = '{MOCK_NOW}'")
    for name in ("wide", "wide_plain"):
        conn.execute(SCHEMA.format(name=name))
    conn.execute("SELECT deltax.deltax_create_table('wide', 'ts', '1 day'::interval)")
    conn.execute(
        "SELECT deltax.deltax_enable_compression('wide', order_by => ARRAY['ts'], "
        f"segment_size => {SEGMENT_SIZE})"
    )
    conn.commit()
    for name in ("wide", "wide_plain"):
        conn.execute(f"INSERT INTO {name} ({COLS}) {GEN}")
    conn.commit()
    part = conn.execute(
        "SELECT partition_name FROM deltax.deltax_partition_info('wide') "
        "WHERE range_start <= %s::timestamptz AND range_end > %s::timestamptz",
        (BASE_TS, BASE_TS),
    ).fetchone()[0]
    result = conn.execute(f"SELECT deltax.deltax_compress_partition('{part}')").fetchone()[0]
    conn.commit()
    assert "Compressed" in result, result
    assert _segcount(conn, part) == NROWS // SEGMENT_SIZE
    return part


def assert_wide_match(conn, where=""):
    """Every column of every row must match the plain twin."""
    q = f"SELECT {_ALL} FROM {{}} {where} ORDER BY id"
    got = conn.execute(q.format("wide")).fetchall()
    want = conn.execute(q.format("wide_plain")).fetchall()
    assert got == want, (
        "row mismatch between compressed and plain twin"
        + (f" (first diff: {next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), '?')})"
           if len(got) == len(want) else f" (len {len(got)} vs {len(want)})")
    )


def _apply_both(conn, stmt_template):
    """Run `stmt` (with a `{}` table placeholder) against both twins."""
    dx = conn.execute(stmt_template.format("wide")).rowcount
    pl = conn.execute(stmt_template.format("wide_plain")).rowcount
    assert dx == pl, f"rowcount mismatch: deltax={dx} plain={pl} for {stmt_template}"
    return dx


class TestDmlAllTypes:
    def test_compression_preserves_all_types(self, db):
        """Baseline: compression itself round-trips every type (before any
        DML), so a later DML failure is attributable to DML, not compression."""
        setup_wide(db)
        assert_wide_match(db)

    def test_insert_loose_rows_all_types(self, db):
        """Loose rows (P1) of every type — including NULLs and edge values —
        read back identically through the segments ∪ heap-tail union."""
        setup_wide(db)
        # Insert new rows (ids above the compressed range) into both twins.
        gen = GEN.replace("generate_series(1, %d)" % NROWS,
                           "generate_series(%d, %d)" % (NROWS + 1, NROWS + 50))
        for name in ("wide", "wide_plain"):
            db.execute(f"INSERT INTO {name} ({COLS}) {gen}")
        db.commit()
        assert_wide_match(db)
        # Also read only the loose tail explicitly.
        assert_wide_match(db, where=f"WHERE id > {NROWS}")

    def test_update_decomposes_segment_all_types(self, db):
        """A targeted UPDATE decomposes one multi-column segment back to heap
        rows; every reconstructed column must equal the twin. This is the
        core codec round-trip guard for decompose-on-write."""
        part = setup_wide(db)
        segs_before = _segcount(db, part)
        # id=750 lives in the 2nd segment (ids 501..1000). Update a column
        # of every type-family so the whole row is rewritten from the heap.
        stmt = ("UPDATE {} SET i4 = i4 + 1, f8 = f8 + 0.5, t_hi = t_hi || '!', "
                "flag = NOT flag, n_i4 = COALESCE(n_i4, 0) + 1 "
                "WHERE id = 750")
        assert _apply_both(db, stmt) == 1
        db.commit()
        assert _segcount(db, part) == segs_before - 1, "one segment should decompose"
        assert_wide_match(db)

    def test_range_update_decomposes_multiple_segments(self, db):
        part = setup_wide(db)
        # Spans segments 1..3 (ids 1..1500) → decomposes 3 segments.
        stmt = "UPDATE {} SET i8 = i8 + 1 WHERE id <= 1500"
        assert _apply_both(db, stmt) == 1500
        db.commit()
        assert _segcount(db, part) <= 1
        assert_wide_match(db)

    def test_delete_tombstone_all_types(self, db):
        """Point DELETEs tombstone rows scattered across segments; the
        remaining rows of every type still read back exactly."""
        part = setup_wide(db)
        segs_before = _segcount(db, part)
        for id_ in (3, 501, 999, 1500, 1999):
            assert _apply_both(db, f"DELETE FROM {{}} WHERE id = {id_}") == 1
        db.commit()
        # Tombstoned (segments intact) — not decomposed.
        assert _segcount(db, part) == segs_before
        assert_wide_match(db)
        assert db.execute("SELECT count(*) FROM wide").fetchone()[0] == NROWS - 5

    def test_compact_roundtrip_all_types(self, db):
        """Mixed DML (loose insert + tombstone + decompose-update) then
        compaction rewrites everything back into fresh segments. The data
        must survive two trips through the codecs."""
        part = setup_wide(db)
        # loose insert
        gen = GEN.replace("generate_series(1, %d)" % NROWS,
                          "generate_series(%d, %d)" % (NROWS + 1, NROWS + 20))
        for name in ("wide", "wide_plain"):
            db.execute(f"INSERT INTO {name} ({COLS}) {gen}")
        # tombstone delete
        _apply_both(db, "DELETE FROM {} WHERE id = 250")
        # decompose update
        _apply_both(db, "UPDATE {} SET f4 = f4 + 1 WHERE id = 1750")
        db.commit()
        assert_wide_match(db)

        db.execute(f"SELECT deltax.deltax_compact_partition('{part}')")
        db.commit()
        # Back to pristine segments, no loose rows / tombstones, data intact.
        assert db.execute(
            "SELECT has_loose_rows, has_tombstones FROM deltax.deltax_partition "
            f"WHERE table_name = '{part.split('.')[-1]}'"
        ).fetchone() == (False, False)
        assert db.execute(f"SELECT pg_relation_size('{part}')").fetchone()[0] == 0
        assert_wide_match(db)

    def test_decompress_after_dml_matches_twin(self, db):
        """The ultimate round-trip: compress → DML → full decompress must
        reproduce the plain twin exactly. Reconstructs every row of every
        type from the blobs, independent of the scan-time union path."""
        part = setup_wide(db)
        gen = GEN.replace("generate_series(1, %d)" % NROWS,
                          "generate_series(%d, %d)" % (NROWS + 1, NROWS + 10))
        for name in ("wide", "wide_plain"):
            db.execute(f"INSERT INTO {name} ({COLS}) {gen}")
        _apply_both(db, "UPDATE {} SET t_lo = t_lo || 'x' WHERE id = 1234")
        _apply_both(db, "DELETE FROM {} WHERE id = 1500")
        db.commit()

        db.execute(f"SELECT deltax.deltax_decompress_partition('{part}')")
        db.commit()
        # Partition is now a plain heap; the companion tables are gone.
        assert not db.execute(
            "SELECT is_compressed FROM deltax.deltax_partition "
            f"WHERE table_name = '{part.split('.')[-1]}'"
        ).fetchone()[0]
        assert_wide_match(db)

    def test_null_only_and_constant_columns_survive_update(self, db):
        """A column that is constant across a segment (Constant codec) and
        the nullable columns must survive a decompose-update untouched."""
        part = setup_wide(db)
        before = db.execute(
            "SELECT konst, count(*) FILTER (WHERE n_txt IS NULL) FROM wide GROUP BY konst"
        ).fetchall()
        _apply_both(db, "UPDATE {} SET i4 = i4 WHERE id = 800")  # no-op value, forces decompose
        db.commit()
        after = db.execute(
            "SELECT konst, count(*) FILTER (WHERE n_txt IS NULL) FROM wide GROUP BY konst"
        ).fetchall()
        assert before == after
        assert_wide_match(db)


# NOTE: the wide-type matrix above deliberately EXCLUDES numeric, time, uuid,
# and bytea. Those types have no specialized codec and fall back to the
# byte/text pipeline, which currently round-trips them incorrectly on a
# compressed read (a pre-existing, non-DML compression bug — types get stored
# as their text-output form and decoded back wrong). That corruption is the
# canonical, strict=True tracking in
# `tests/correctness/test_codecs_extended.py` (numeric / time / uuid / bytea
# fallback cases); it is a base-codec issue independent of DML, so it is not
# re-tracked here to avoid duplicate xfails. When the codec fallback is fixed,
# add these types back into the matrix above to get DML round-trip coverage
# for them too.
