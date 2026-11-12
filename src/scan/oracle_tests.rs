//! End-to-end regressions for scans over compressed partitions, checked
//! against a plain heap copy of the same rows (the oracle).
//!
//! FB-0256 (ValorBrain, 2026-10-22): on `usage_events` (segment_by
//! `tenant_id`), `WHERE tenant_id = $t AND operation = 'memory_used'`
//! returned fewer rows from the compressed partitions than the same filter
//! applied after the scan — no error, just missing rows. The shape that
//! triggers it is common for a multi-tenant table: most tenants use a handful
//! of `operation` values, one tenant uses more than
//! `VALBITMAP_MAX_DISTINCT` (32). That tenant's segment writes no valbitmap
//! row and contributes nothing to the partition-level `column_valmap`, so the
//! values only it uses are absent from the valmap.
//!
//! Three defects returned wrong counts for that shape:
//!
//! 1. Valbitmap pruning treated "constant absent from the valmap" as "no
//!    segment of the partition can match" and dropped every segment without
//!    reading the bitmap rows — including the overflowed segment, the only one
//!    that holds the constant.
//! 2. The valmap reader decoded the catalog JSON byte by byte, so a value
//!    with a non-ASCII character or a JSON escape other than `\"` / `\\`
//!    never matched its own constant: a valmap miss, and every segment with
//!    a bitmap row — the ones holding the value — was pruned.
//! 3. `count(*) FILTER (WHERE ...)` as the only aggregate took the
//!    `DeltaXCount` fast path, which answers from catalog/segment row counts
//!    and ignores the FILTER (it returned every row of the tenant).
//! 4. Bloom probes hashed the PostgreSQL-epoch datum of a timestamp/date
//!    constant while the filters were built from Unix-epoch values, so every
//!    segment was bloom-rejected: `created_at = <exact ts>` and `IN (...)`
//!    returned zero rows. The colstats min/max check for timestamp/date
//!    IN-lists had the same epoch mix-up.
//! 5. The time column's `_min_`/`_max_` from the meta table went into
//!    `ColMinMax` in the PostgreSQL epoch, while its readers decode the
//!    Unix-epoch colstats encoding: the metadata-only `min()`/`max()` of the
//!    time column came back 30 years early.
//!
//! FB-0275 (independent review, 2026-10-31):
//!
//! 6. `col op ANY(array)` was pushed down as `col IN (...)` whatever the
//!    operator: `id < ANY(ARRAY[100, 1200])` kept only the rows equal to an
//!    element. The DeltaXAgg planner gate also accepted IN-lists the executor
//!    does not push (a NULL element, an empty or a two-dimensional array),
//!    and the WHERE was dropped.
//! 7. A constant of another type was read as if it had the column's type:
//!    `created_at < DATE '2025-01-16'` took a day count for microseconds and
//!    pruned every segment (0 rows instead of 1100), a `smallint` compared
//!    with 65539 used the truncated constant 3, a `float8` constant was read
//!    as a `float4` and a `name` as `text`.

#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    /// Operations of the tenants whose segment stays under the valbitmap cap,
    /// two per tenant, so the partition valmap for `operation` is exactly
    /// these six values. The last four need more than byte-copying to come
    /// back from the catalog JSON: a non-ASCII character, a newline, a
    /// backslash and double quotes.
    const SMALL_OPS_SQL: &str = r#"ARRAY['GET /documents', 'POST /documents',
        'exportação', E'multi\nline', E'C:\\temp', 'say "hi"']"#;

    fn run(sql: &str) {
        Spi::run(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }

    fn one_i64(sql: &str) -> i64 {
        Spi::get_one::<i64>(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .unwrap_or_else(|| panic!("{sql}: NULL"))
    }

    fn one_text(sql: &str) -> Option<String> {
        Spi::get_one::<String>(sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    fn one_bool(sql: &str) -> bool {
        Spi::get_one::<bool>(sql)
            .unwrap_or_else(|e| panic!("{sql}: {e}"))
            .unwrap_or_else(|| panic!("{sql}: NULL"))
    }

    fn explain(sql: &str) -> String {
        let query = format!("EXPLAIN (COSTS OFF) {sql}");
        Spi::connect(|client| {
            let rows = client
                .select(&query, None, &[])
                .unwrap_or_else(|e| panic!("{query}: {e}"));
            let mut lines = Vec::new();
            for row in rows {
                lines.push(row.get::<String>(1).unwrap().unwrap_or_default());
            }
            lines.join("\n")
        })
    }

    /// One compressed partition shaped like `usage_events_p20260705`:
    /// segment_by `tenant_id`, one segment per tenant. Tenant `big` has 40
    /// distinct operations (over the cap → no valbitmap row) and is the only
    /// one with `memory_used` / `upsert_keyed_fact`. `fb0256_ref` keeps the
    /// same rows in a plain heap table.
    fn setup() {
        run("SET LOCAL pg_deltax.mock_now = '2025-01-15 12:00:00+00'");
        run("CREATE TABLE fb0256_events (
                 id         bigint      NOT NULL,
                 tenant_id  text        NOT NULL,
                 created_at timestamptz NOT NULL,
                 channel    text,
                 operation  text,
                 logged_on  date,
                 kind       smallint,
                 score      real
             )");
        run("SELECT deltax.deltax_create_table('fb0256_events', 'created_at', '1 day'::interval)");
        run("INSERT INTO fb0256_events
             SELECT i, 'big',
                    '2025-01-15 06:00:00+00'::timestamptz + i * interval '1 second',
                    CASE WHEN (i / 40) % 2 = 0 THEN 'mcp' ELSE 'rest' END,
                    CASE i % 40 WHEN 0 THEN 'memory_used'
                                WHEN 1 THEN 'upsert_keyed_fact'
                                ELSE 'op' || lpad((i % 40)::text, 2, '0') END,
                    DATE '2025-01-15', i % 5, i / 10.0
               FROM generate_series(1, 800) i");
        run(&format!(
            "INSERT INTO fb0256_events
             SELECT 1000 + i, 'small-' || (i % 3),
                    '2025-01-15 06:00:00+00'::timestamptz + i * interval '1 second',
                    'rest', ({SMALL_OPS_SQL})[1 + 2 * (i % 3) + (i / 3) % 2],
                    DATE '2025-01-15', i % 5, i / 10.0
               FROM generate_series(1, 300) i"
        ));
        run("CREATE TABLE fb0256_ref AS SELECT * FROM fb0256_events");
        run("SELECT deltax.deltax_enable_compression('fb0256_events',
                 segment_by => ARRAY['tenant_id'],
                 order_by => ARRAY['tenant_id', 'created_at'])");
        // The rows' day has to be over before its partition may be compressed.
        run("SET LOCAL pg_deltax.mock_now = '2025-01-20 00:00:00+00'");
        run("DO $$
             DECLARE p text; n bigint;
             BEGIN
                 FOR p IN SELECT partition_name FROM deltax.deltax_partition_info('fb0256_events')
                 LOOP
                     EXECUTE format('SELECT count(*) FROM %I', p) INTO n;
                     IF n > 0 THEN PERFORM deltax.deltax_compress_partition(p); END IF;
                 END LOOP;
             END $$");

        // Every row lives in the compressed partition, and the valmap has the
        // shape that triggered FB-0256: present for `operation`, built from
        // the small tenants only.
        assert_eq!(
            one_i64(
                "SELECT coalesce(sum(row_count), 0)::int8 FROM deltax.deltax_partition
                  WHERE table_name LIKE 'fb0256_events%' AND is_compressed"
            ),
            1100,
            "fixture: all rows must sit in one compressed partition"
        );
        let valmap_ok = format!(
            "SELECT column_valmap->'operation' @> to_jsonb({SMALL_OPS_SQL})
                    AND jsonb_array_length(column_valmap->'operation') = 6
                    AND NOT column_valmap->'operation' ? 'memory_used'
               FROM deltax.deltax_partition
              WHERE table_name LIKE 'fb0256_events%' AND is_compressed"
        );
        assert!(
            one_bool(&valmap_ok),
            "fixture: valmap for operation must hold exactly the small tenants' values, got {:?}",
            one_text(
                "SELECT column_valmap::text FROM deltax.deltax_partition
                  WHERE table_name LIKE 'fb0256_events%' AND is_compressed"
            )
        );
    }

    /// `count(*)` with `pred` on the compressed table equals the oracle.
    fn assert_count_matches(pred: &str) -> i64 {
        let got = one_i64(&format!("SELECT count(*) FROM fb0256_events WHERE {pred}"));
        let want = one_i64(&format!("SELECT count(*) FROM fb0256_ref WHERE {pred}"));
        assert_eq!(got, want, "count(*) WHERE {pred}");
        got
    }

    /// `count(id)` and the ids with `pred` on the compressed table equal the
    /// oracle. The count goes through the aggregate paths that accept the
    /// WHERE (DeltaXAgg, the metadata-only DeltaXMinMax), the ids through the
    /// row path (DeltaXAppend). `id` is NOT NULL, so `count(id)` is the row
    /// count; `count(*)` over a plain DeltaX row scan projects an empty target
    /// list, which trips an assertion of cassert builds of PostgreSQL 18 (a
    /// separate defect).
    fn assert_rows_match(pred: &str) -> i64 {
        let count = |t: &str| one_i64(&format!("SELECT count(id) FROM {t} WHERE {pred}"));
        let got = count("fb0256_events");
        assert_eq!(got, count("fb0256_ref"), "count(id) WHERE {pred}");
        let ids = |t: &str| {
            one_text(&format!(
                "SELECT string_agg(id::text, ',' ORDER BY id) FROM (
                     SELECT id FROM {t} WHERE {pred} ORDER BY id LIMIT 2000) s"
            ))
        };
        assert_eq!(ids("fb0256_events"), ids("fb0256_ref"), "rows WHERE {pred}");
        got
    }

    /// `count(id)` with `pred` is answered by DeltaXAgg: the qual is still
    /// pushed down.
    fn assert_pushed_to_agg(pred: &str) {
        let plan = explain(&format!("SELECT count(id) FROM fb0256_events WHERE {pred}"));
        assert!(
            plan.contains("DeltaXAgg"),
            "WHERE {pred}: expected DeltaXAgg, got:\n{plan}"
        );
    }

    #[pg_test]
    fn valmap_miss_keeps_overflowed_segment() {
        setup();

        let shapes = [
            // The reporter's queries.
            "tenant_id = 'big' AND operation = 'memory_used'",
            "tenant_id = 'big' AND channel = 'mcp' AND operation = 'upsert_keyed_fact'",
            // Same constant without the segment-by filter.
            "operation = 'memory_used'",
            // Controls: a value only the small tenants use (valmap hit) and a
            // value no tenant uses (valmap miss, zero rows is right).
            "operation = 'GET /documents'",
            "tenant_id = 'big' AND operation = 'GET /documents'",
            "operation = 'never-used'",
        ];
        for pred in shapes {
            assert_count_matches(pred);
        }
        assert_eq!(
            assert_count_matches("tenant_id = 'big' AND operation = 'memory_used'"),
            20
        );

        // The count above runs through DeltaXAgg; the row path (DeltaXAppend /
        // DeltaXDecompress) goes through the same segment pruning.
        let rows_sql = |table: &str| {
            format!(
                "SELECT string_agg(id::text, ',' ORDER BY id) FROM (
                     SELECT id FROM {table}
                      WHERE tenant_id = 'big' AND operation = 'memory_used'
                      ORDER BY id LIMIT 1000) s"
            )
        };
        assert_eq!(
            one_text(&rows_sql("fb0256_events")),
            one_text(&rows_sql("fb0256_ref")),
            "rows WHERE tenant_id = 'big' AND operation = 'memory_used'"
        );

        let plan = explain("SELECT count(*) FROM fb0256_events WHERE operation = 'memory_used'");
        assert!(
            plan.contains("DeltaX"),
            "expected a DeltaX scan, got:\n{plan}"
        );
    }

    #[pg_test]
    fn valmap_values_needing_json_unescape_still_match() {
        setup();

        // Each value lives in exactly one small tenant's segment, which has a
        // bitmap row with the value's bit set.
        for pred in [
            "operation = 'exportação'",
            r"operation = E'multi\nline'",
            r"operation = E'C:\\temp'",
            r#"operation = 'say "hi"'"#,
        ] {
            assert_eq!(assert_count_matches(pred), 50, "count(*) WHERE {pred}");
        }
    }

    #[pg_test]
    fn temporal_equality_and_in_list_match() {
        setup();

        // Rows sit at 06:00:00 + i seconds (i from 1): one big-tenant row and
        // one small-tenant row per second for the first 300 seconds.
        assert_eq!(
            assert_count_matches("created_at = '2025-01-15 06:00:01+00'"),
            2
        );
        assert_eq!(
            assert_count_matches(
                "created_at IN ('2025-01-15 06:00:01+00', '2025-01-15 06:05:00+00')"
            ),
            4
        );
        assert_count_matches("tenant_id = 'big' AND created_at = '2025-01-15 06:00:40+00'");
        // A date column that is not the partitioning column: bloom on `=`,
        // colstats min/max on IN.
        assert_eq!(assert_count_matches("logged_on = '2025-01-15'"), 1100);
        assert_eq!(
            assert_count_matches("logged_on IN ('2025-01-15', '2025-01-16')"),
            1100
        );
        assert_count_matches("logged_on = '2025-01-16'");
    }

    #[pg_test]
    fn temporal_min_max_from_metadata_matches() {
        setup();

        // Over the whole compressed day, DeltaXMinMax answers from the meta
        // table. The meta keeps the time column's `_min_`/`_max_` in the
        // PostgreSQL epoch, and they were decoded as Unix-epoch colstats
        // values: the result came back 30 years early (in ValorBrain
        // production, 1996-08-04 for a 2026-08-04 row of usage_events).
        let min_max = |table: &str| {
            format!(
                "SELECT m.lo::text || ' / ' || m.hi::text
                   FROM (SELECT min(created_at) AS lo, max(created_at) AS hi
                           FROM {table}
                          WHERE created_at >= '2025-01-15 00:00:00+00'
                            AND created_at < '2025-01-16 00:00:00+00') m"
            )
        };
        let plan = explain(&min_max("fb0256_events"));
        assert!(
            plan.contains("DeltaXMinMax"),
            "expected DeltaXMinMax, got:\n{plan}"
        );
        assert_eq!(
            one_text(&min_max("fb0256_events")),
            one_text(&min_max("fb0256_ref")),
            "min/max(created_at) from the compressed partition"
        );
    }

    #[pg_test]
    fn count_star_filter_skips_count_fast_path() {
        setup();

        for (filter, pred) in [
            ("operation = 'memory_used'", "tenant_id = 'big'"),
            ("operation = 'memory_used'", "true"),
            ("channel = 'mcp'", "tenant_id = 'big'"),
        ] {
            let got = one_i64(&format!(
                "SELECT count(*) FILTER (WHERE {filter}) FROM fb0256_events WHERE {pred}"
            ));
            let want = one_i64(&format!(
                "SELECT count(*) FILTER (WHERE {filter}) FROM fb0256_ref WHERE {pred}"
            ));
            assert_eq!(got, want, "count(*) FILTER (WHERE {filter}) WHERE {pred}");
        }
        // The FILTER-free count keeps its fast path.
        assert_eq!(
            one_i64("SELECT count(*) FROM fb0256_events WHERE tenant_id = 'big'"),
            800
        );
        let plan = explain("SELECT count(*) FROM fb0256_events WHERE tenant_id = 'big'");
        assert!(
            plan.contains("DeltaXCount"),
            "expected DeltaXCount, got:\n{plan}"
        );
    }

    #[pg_test]
    fn any_all_with_other_operators_match() {
        setup();

        // `op ANY (array)` holds when `op` holds for some element, `op ALL`
        // when it holds for every element. Only `= ANY` is an IN-list.
        for (pred, want) in [
            ("id < ANY(ARRAY[100, 1200]::int8[])", 999),
            ("id > ANY(ARRAY[100, 1200]::int8[])", 1000),
            ("id <> ANY(ARRAY[100, 1200]::int8[])", 1100),
            (
                "tenant_id = 'big' AND id < ANY(ARRAY[100, 1200]::int8[])",
                800,
            ),
            ("id < ALL(ARRAY[100, 1200]::int8[])", 99),
            ("id > ALL(ARRAY[100, 1200]::int8[])", 100),
            ("id <> ALL(ARRAY[100, 1200]::int8[])", 1098),
            ("operation <> ANY(ARRAY['memory_used'])", 1080),
            ("operation <> ALL(ARRAY['memory_used', 'op02'])", 1060),
            (
                "created_at < ANY(ARRAY['2025-01-15 06:01:00+00',
                                        '2025-01-15 06:02:00+00']::timestamptz[])",
                238,
            ),
            (
                "created_at > ALL(ARRAY['2025-01-15 06:01:00+00',
                                        '2025-01-15 06:02:00+00']::timestamptz[])",
                860,
            ),
            ("logged_on <> ANY(ARRAY['2025-01-16']::date[])", 1100),
            // IN-lists the executor does not push (a NULL element, an empty
            // or a two-dimensional array) must not reach DeltaXAgg either.
            ("id IN (5, NULL)", 1),
            ("operation IN ('memory_used', NULL)", 20),
            ("id = ANY('{}'::int8[])", 0),
            ("id = ANY('{{5, 1005}, {6, 1006}}'::int8[])", 4),
            // Controls: plain IN-lists.
            ("id IN (5, 1005)", 2),
            ("operation IN ('memory_used', 'GET /documents')", 70),
        ] {
            assert_eq!(assert_rows_match(pred), want, "count(id) WHERE {pred}");
        }
        // Text order depends on the collation; the oracle uses the same one.
        for pred in [
            "operation < ANY(ARRAY['b', 'op10'])",
            "operation > ANY(ARRAY['op30'])",
            "operation >= ALL(ARRAY['op10', 'op20'])",
        ] {
            assert_rows_match(pred);
        }
        // `= ANY` keeps its pushdown.
        assert_pushed_to_agg("id IN (5, 1005)");
        assert_pushed_to_agg("operation IN ('memory_used', 'GET /documents')");
    }

    #[pg_test]
    fn temporal_constants_of_another_type_match() {
        setup();
        // Cross-type temporal comparisons convert in the session time zone;
        // outside UTC the raw datums of the two types are hours apart.
        run("SET LOCAL timezone = 'America/Sao_Paulo'");

        for (pred, want) in [
            // timestamptz column, date and timestamp constants. Rows sit at
            // 2025-01-15 06:00:01+00 .. 06:13:20+00.
            ("created_at < DATE '2025-01-16'", 1100),
            ("created_at >= DATE '2025-01-15'", 1100),
            ("created_at < DATE '2025-01-15'", 0),
            ("created_at < TIMESTAMP '2025-01-15 03:05:00'", 598),
            ("created_at = TIMESTAMP '2025-01-15 03:00:01'", 2),
            // date column, timestamptz and timestamp constants.
            ("logged_on < TIMESTAMPTZ '2025-01-16 00:00:00+00'", 1100),
            ("logged_on >= TIMESTAMPTZ '2025-01-15 12:00:00+00'", 0),
            ("logged_on = TIMESTAMP '2025-01-15 00:00:00'", 1100),
            // Controls: constants of the column's own type.
            ("created_at < '2025-01-15 06:05:00+00'", 598),
            ("logged_on = DATE '2025-01-15'", 1100),
        ] {
            assert_eq!(assert_rows_match(pred), want, "count(id) WHERE {pred}");
        }
        assert_pushed_to_agg("created_at < '2025-01-15 06:05:00+00' AND tenant_id = 'big'");
    }

    #[pg_test]
    fn numeric_constants_of_another_type_match() {
        setup();

        for (pred, want) in [
            // smallint column (values 0..4), int4 constants outside its range:
            // truncated to int16 they became 3, 1 and 1.
            ("kind < 65539", 1100),
            ("kind <> 65537", 1100),
            ("kind > -65535", 1100),
            ("kind = 65537", 0),
            // real column (i / 10), float8 constants.
            ("score > 1", 1080),
            ("score < 50.5", 804),
            ("score = 0.5", 2),
            // Controls: integer constants that fit the column, of the same or
            // of another integer type, stay pushed down.
            ("id = 5", 1),
            ("id < 100", 99),
            ("id = ANY(ARRAY[5, 1005])", 2),
            ("kind = 1", 220),
            ("kind IN (1, 2)", 440),
        ] {
            assert_eq!(assert_rows_match(pred), want, "count(id) WHERE {pred}");
        }
        for pred in [
            "id = 5",
            "id = ANY(ARRAY[5, 1005])",
            "kind = 1",
            "kind IN (1, 2)",
        ] {
            assert_pushed_to_agg(pred);
        }
    }

    #[pg_test]
    fn text_constants_of_another_type_match() {
        setup();

        for (pred, want) in [
            // `text = name` is a cross-type operator; a `name` is no varlena.
            ("operation = 'memory_used'::name", 20),
            ("operation = ANY(ARRAY['memory_used']::name[])", 20),
            // segment_by: the executor applies the segment filter only to a
            // constant of the column's type, so the metadata-only count must
            // not take this one either (it would count every segment).
            ("tenant_id = 'big'::name", 800),
        ] {
            assert_eq!(assert_rows_match(pred), want, "count(id) WHERE {pred}");
        }
    }
}
