"""The planner hook must not break queries that are none of our business.

WHY THIS FILE EXISTS

On 2026-07-27 we found that, with pg_deltax in `shared_preload_libraries`, *any*
query shaped like `count(...)` over a `UNION ALL` of subqueries died with
`could not open relation with OID 0` — no deltax table involved, in a database
where the extension had never been created.

The cause: `subpath_has_data()` took the `relid` off the subpath's RTE and called
`table_open()` without checking it was a relation RTE. For a subquery, view, CTE,
function or VALUES that `relid` is 0, and `table_open(0)` aborts the query during
planning, so even `EXPLAIN` failed.

What it cost: ValorBrain's knowledge-graph mirror (pg_ripple) sat for months with
8% of the facts, because the triple counter used SPARQL aggregation inside
`GRAPH`, got this error, and everyone — the diagnostic code included — concluded
the defect was in pg_ripple. Blaming the third-party extension is the natural
first suspicion and the wrong one; our own hook, which runs on every query in the
database, is the right one.

The rule this file enforces: the pg_deltax planner hook does not interfere with a
query that touches no deltax table, whatever shape that query has.
"""

import pytest

pytestmark = pytest.mark.smoke


SHAPES = [
    (
        "count over UNION ALL of subqueries",
        """
        SELECT count(o) FROM (
          SELECT 1::bigint AS p, s, o, g FROM plain_a WHERE g = 7
          UNION ALL
          SELECT 1::bigint AS p, s, o, g FROM plain_b WHERE g = 7
        ) t
        """,
        50,
    ),
    (
        "count over UNION ALL of views",
        """
        SELECT count(o) FROM (
          SELECT s, o, g FROM view_a WHERE g = 7
          UNION ALL
          SELECT s, o, g FROM view_b WHERE g = 7
        ) t
        """,
        50,
    ),
    (
        "count(*) over UNION ALL of three views",
        """
        SELECT count(*) FROM (
          SELECT s FROM view_a UNION ALL SELECT s FROM view_b UNION ALL SELECT s FROM view_a
        ) t
        """,
        75,
    ),
    (
        "aggregate with GROUP BY over UNION ALL",
        """
        SELECT g, count(*) FROM (
          SELECT g FROM view_a UNION ALL SELECT g FROM view_b
        ) t GROUP BY g
        """,
        None,  # checked in its own test
    ),
    (
        "count over CTEs in UNION ALL",
        """
        WITH ua AS (SELECT s, o, g FROM plain_a),
             ub AS (SELECT s, o, g FROM plain_b)
        SELECT count(o) FROM (SELECT * FROM ua UNION ALL SELECT * FROM ub) t
        """,
        50,
    ),
    (
        "count over UNION ALL with VALUES",
        """
        SELECT count(*) FROM (
          SELECT s FROM view_a
          UNION ALL
          SELECT * FROM (VALUES (1::bigint), (2::bigint)) AS v(s)
        ) t
        """,
        27,
    ),
    (
        "min/max over UNION ALL of views",
        "SELECT min(o), max(o) FROM (SELECT o FROM view_a UNION ALL SELECT o FROM view_b) t",
        None,
    ),
]


@pytest.fixture()
def plain_tables(db):
    """Ordinary tables and views, nothing deltax about them."""
    with db.cursor() as cur:
        cur.execute("DROP VIEW IF EXISTS view_a, view_b CASCADE")
        cur.execute("DROP TABLE IF EXISTS plain_a, plain_b, tomb_a, tomb_b CASCADE")
        for name in ("plain_a", "plain_b"):
            cur.execute(
                f"CREATE TABLE {name} (s bigint, o bigint, g bigint, note text)"
            )
            cur.execute(
                f"INSERT INTO {name} SELECT i, i * 10, 7, 'x' FROM generate_series(1, 25) i"
            )
        for name in ("tomb_a", "tomb_b"):
            cur.execute(f"CREATE TABLE {name} (s bigint, o bigint, g bigint)")
        # Views shaped the way pg_ripple shapes its own (DISTINCT ON + LEFT JOIN
        # over an inner UNION ALL), which is where the defect showed in production.
        for view, base, tomb in (("view_a", "plain_a", "tomb_a"), ("view_b", "plain_b", "tomb_b")):
            cur.execute(
                f"""
                CREATE VIEW {view} AS
                  SELECT DISTINCT ON (s, o, g) s, o, g FROM (
                    SELECT m.s, m.o, m.g FROM {base} m
                      LEFT JOIN {tomb} t ON m.s = t.s AND m.o = t.o AND m.g = t.g
                     WHERE t.s IS NULL
                  ) x
                """
            )
        db.commit()
        yield
        cur.execute("DROP VIEW IF EXISTS view_a, view_b CASCADE")
        cur.execute("DROP TABLE IF EXISTS plain_a, plain_b, tomb_a, tomb_b CASCADE")
        db.commit()


@pytest.mark.parametrize("label,sql,expected", SHAPES, ids=[s[0] for s in SHAPES])
def test_shape_plans_and_runs(db, plain_tables, label, sql, expected):
    """The query plans and runs. Before the fix, planning alone already failed."""
    with db.cursor() as cur:
        # EXPLAIN on its own: the defect was in planning, so this is the half that
        # failed first.
        cur.execute(f"EXPLAIN {sql}")
        assert cur.fetchall(), f"{label}: EXPLAIN returned no plan"

        cur.execute(sql)
        rows = cur.fetchall()
        assert rows, f"{label}: query returned no row"
        if expected is not None:
            assert rows[0][0] == expected, f"{label}: expected {expected}, got {rows[0][0]}"


def test_group_by_over_union_counts_right(db, plain_tables):
    """GROUP BY over a union: one row per group, with the right count."""
    with db.cursor() as cur:
        cur.execute(
            "SELECT g, count(*) FROM (SELECT g FROM view_a UNION ALL SELECT g FROM view_b) t GROUP BY g"
        )
        rows = cur.fetchall()
        assert rows == [(7, 50)], f"expected [(7, 50)], got {rows}"


def test_minmax_over_union(db, plain_tables):
    with db.cursor() as cur:
        cur.execute(
            "SELECT min(o), max(o) FROM (SELECT o FROM view_a UNION ALL SELECT o FROM view_b) t"
        )
        assert cur.fetchone() == (10, 250)
