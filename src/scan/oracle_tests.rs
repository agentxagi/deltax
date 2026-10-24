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
                 operation  text
             )");
        run("SELECT deltax.deltax_create_table('fb0256_events', 'created_at', '1 day'::interval)");
        run("INSERT INTO fb0256_events
             SELECT i, 'big',
                    '2025-01-15 06:00:00+00'::timestamptz + i * interval '1 second',
                    CASE WHEN (i / 40) % 2 = 0 THEN 'mcp' ELSE 'rest' END,
                    CASE i % 40 WHEN 0 THEN 'memory_used'
                                WHEN 1 THEN 'upsert_keyed_fact'
                                ELSE 'op' || lpad((i % 40)::text, 2, '0') END
               FROM generate_series(1, 800) i");
        run(&format!(
            "INSERT INTO fb0256_events
             SELECT 1000 + i, 'small-' || (i % 3),
                    '2025-01-15 06:00:00+00'::timestamptz + i * interval '1 second',
                    'rest', ({SMALL_OPS_SQL})[1 + 2 * (i % 3) + (i / 3) % 2]
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
}
