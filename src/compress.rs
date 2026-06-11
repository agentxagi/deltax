use pgrx::prelude::*;
use pgrx::spi::SpiClient;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use cardinality_estimator::CardinalityEstimator;

use crate::USE_LZ4;
use crate::catalog;
use crate::compression::{self, CompressedColumn, CompressionType};

/// Microseconds between Unix epoch (1970-01-01) and PG epoch (2000-01-01).
pub(crate) const PG_EPOCH_OFFSET_USEC: i64 = 946_684_800_000_000;
/// Days between Unix epoch (1970-01-01) and PG epoch (2000-01-01).
pub(crate) const PG_EPOCH_OFFSET_DAYS: i64 = 10_957;

/// Column metadata from information_schema, plus any synthetic columns
/// introduced by `json_extract` configuration. Extracted columns sit at the
/// end of the slice and carry `extracted = Some(_)`; all other paths through
/// this struct ignore them or special-case them based on that flag.
#[derive(Debug, Clone)]
pub(crate) struct ColumnMeta {
    pub(crate) name: String,
    pub(crate) data_type: String,
    pub(crate) is_segment_by: bool,
    pub(crate) is_time_column: bool,
    /// `Some` for synthetic columns produced by JSON-path extraction at COPY
    /// time. `None` for physical columns of the parent table.
    pub(crate) extracted: Option<ExtractSpec>,
}

/// One JSON-path extraction directive — extract `path` from JSONB column
/// `src_column`, store as a synthetic columnar column named `target_name` of
/// `target_kind`. Built by `parse_extract_specs` from the user-supplied JSONB.
#[derive(Debug, Clone)]
pub(crate) struct ExtractSpec {
    pub(crate) src_column: String,
    #[allow(dead_code)] // consumed by COPY-time extraction in step 3
    pub(crate) path: Vec<String>,
    pub(crate) target_name: String,
    #[allow(dead_code)] // consumed by COPY-time extraction in step 3
    pub(crate) target_kind: ColumnKind,
    /// User-provided PG type alias (e.g. "text", "bigint"). Kept verbatim so
    /// it can be echoed back through the column-metadata pipeline alongside
    /// physical columns, and so EXPLAIN can show the original type alias.
    pub(crate) target_type: String,
}

/// Validate and parse the `json_extract` JSONB blob into a list of specs.
/// Errors are emitted via `pgrx::error!` so they surface as PG ERRORs from
/// `deltax_enable_compression`.
pub(crate) fn parse_extract_specs(value: &serde_json::Value) -> Vec<ExtractSpec> {
    let arr = value.as_array().unwrap_or_else(|| {
        pgrx::error!(
            "pg_deltax: json_extract must be a JSON array of {{src,path,name,type}} objects"
        )
    });

    let mut specs: Vec<ExtractSpec> = Vec::with_capacity(arr.len());
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (i, entry) in arr.iter().enumerate() {
        let obj = entry.as_object().unwrap_or_else(|| {
            pgrx::error!(
                "pg_deltax: json_extract[{}] must be an object with src/path/name/type",
                i
            )
        });

        let src_column = obj
            .get("src")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| {
                pgrx::error!(
                    "pg_deltax: json_extract[{}].src must be a string column name",
                    i
                )
            })
            .to_string();

        let path_value = obj
            .get("path")
            .unwrap_or_else(|| pgrx::error!("pg_deltax: json_extract[{}].path is required", i));
        let path_arr = path_value.as_array().unwrap_or_else(|| {
            pgrx::error!(
                "pg_deltax: json_extract[{}].path must be a JSON array of strings",
                i
            )
        });
        if path_arr.is_empty() {
            pgrx::error!("pg_deltax: json_extract[{}].path must not be empty", i);
        }
        let path: Vec<String> = path_arr
            .iter()
            .enumerate()
            .map(|(j, v)| {
                v.as_str()
                    .unwrap_or_else(|| {
                        pgrx::error!(
                            "pg_deltax: json_extract[{}].path[{}] must be a string (array indices not yet supported)",
                            i, j
                        )
                    })
                    .to_string()
            })
            .collect();

        let target_name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| pgrx::error!("pg_deltax: json_extract[{}].name must be a string", i))
            .to_string();
        if !is_valid_identifier(&target_name) {
            pgrx::error!(
                "pg_deltax: json_extract[{}].name {:?} is not a valid SQL identifier",
                i,
                target_name
            );
        }
        if !seen_names.insert(target_name.clone()) {
            pgrx::error!(
                "pg_deltax: json_extract has duplicate target name {:?}",
                target_name
            );
        }

        let target_type = obj
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| pgrx::error!("pg_deltax: json_extract[{}].type must be a string", i))
            .to_string();
        let target_kind = classify_column(&target_type, false);
        if matches!(target_kind, ColumnKind::Jsonb) {
            // jsonb-extracted-as-jsonb adds no value; reject for clarity.
            pgrx::error!(
                "pg_deltax: json_extract[{}].type=jsonb is not supported (use the source jsonb column directly)",
                i
            );
        }
        // Unknown type names fall through to `Text` in classify_column. Keep
        // the user's spelling in `target_type` but warn on obvious typos by
        // requiring the recognized ones explicitly.
        if !is_recognized_extract_type(&target_type) {
            pgrx::error!(
                "pg_deltax: json_extract[{}].type {:?} is not recognized (expected one of: text, varchar, char, smallint, integer, bigint, real, double precision, boolean, timestamp, timestamp with time zone, date)",
                i,
                target_type
            );
        }

        specs.push(ExtractSpec {
            src_column,
            path,
            target_name,
            target_kind,
            target_type,
        });
    }

    specs
}

/// Per-source-column extraction targets. Built once per COPY (or compress)
/// pass and threaded through the parser as `Option<&ColumnExtractTargets>`
/// alongside each physical column. `targets[k] = (idx_in_typed_cols, spec)`
/// means "after parsing this row's source jsonb column, extract spec.path
/// and write the leaf into typed_cols[idx_in_typed_cols]".
#[derive(Debug, Clone, Default)]
pub(crate) struct ColumnExtractTargets {
    pub(crate) targets: Vec<(usize, ExtractSpec)>,
}

/// Build `Vec<Option<ColumnExtractTargets>>` indexed by physical column index
/// (matching the physical column ordinal in the parent table). Extracted
/// columns sit beyond the last physical column in `columns`; we look them up
/// by name.
pub(crate) fn build_extract_targets_per_column(
    columns: &[ColumnMeta],
) -> Vec<Option<ColumnExtractTargets>> {
    let physical_count = columns
        .iter()
        .position(|c| c.extracted.is_some())
        .unwrap_or(columns.len());

    let mut per_col: Vec<Option<ColumnExtractTargets>> =
        (0..physical_count).map(|_| None).collect();

    for (target_idx, col) in columns.iter().enumerate() {
        let Some(spec) = col.extracted.as_ref() else {
            continue;
        };
        let src_idx = match columns
            .iter()
            .take(physical_count)
            .position(|c| c.name == spec.src_column)
        {
            Some(idx) => idx,
            None => {
                pgrx::error!(
                    "pg_deltax: json_extract spec for {:?}: src column {:?} not found in physical columns",
                    spec.target_name,
                    spec.src_column
                )
            }
        };
        per_col[src_idx]
            .get_or_insert_with(ColumnExtractTargets::default)
            .targets
            .push((target_idx, spec.clone()));
    }

    per_col
}

/// Driven by the per-row caller: unescape the raw COPY field, run NULL check,
/// and apply extraction targets. NULL source -> NULL for every target.
/// Same NULL-on-error contract as `apply_extract_targets`.
pub(crate) fn extract_from_raw_field(
    raw: &[u8],
    null_string: &[u8],
    targets: &ColumnExtractTargets,
    typed_cols: &mut [TypedColumn],
) {
    if raw == null_string {
        for (idx, _) in &targets.targets {
            push_typed_null(&mut typed_cols[*idx]);
        }
        return;
    }
    let unescaped = crate::copyparse::unescape_field_always(raw);
    apply_extract_targets(&unescaped, targets, typed_cols);
}

/// Same as `extract_from_raw_field` but for an already-unescaped &str field
/// (legacy/STDIN path: PG hands us decoded `Option<&str>` directly).
pub(crate) fn extract_from_str_field(
    field: Option<&str>,
    targets: &ColumnExtractTargets,
    typed_cols: &mut [TypedColumn],
) {
    let Some(text) = field else {
        for (idx, _) in &targets.targets {
            push_typed_null(&mut typed_cols[*idx]);
        }
        return;
    };
    apply_extract_targets(text, targets, typed_cols);
}

/// Apply a JSON extraction context to a row's just-parsed source-column text.
/// `json_text` is the unescaped UTF-8 JSON for this row's source jsonb column;
/// for each spec in `targets`, descend `spec.path` and push a coerced leaf
/// into `typed_cols[target_idx]`. Missing paths and type mismatches yield NULL.
/// Malformed JSON yields NULL for every target (we never abort the COPY here —
/// the source jsonb's own conversion via `jsonb_in` will surface a real error
/// if the row truly isn't valid JSON).
pub(crate) fn apply_extract_targets(
    json_text: &str,
    targets: &ColumnExtractTargets,
    typed_cols: &mut [TypedColumn],
) {
    let value: serde_json::Value = match serde_json::from_str(json_text) {
        Ok(v) => v,
        Err(_) => {
            for (idx, _) in &targets.targets {
                push_typed_null(&mut typed_cols[*idx]);
            }
            return;
        }
    };
    for (idx, spec) in &targets.targets {
        let leaf = descend_json_path(&value, &spec.path);
        push_extracted_leaf(leaf, spec.target_kind, &mut typed_cols[*idx]);
    }
}

/// Walk a JSON Value down a sequence of object-key steps. Returns `None` if
/// any step is missing or the intermediate value isn't an object.
fn descend_json_path<'a>(
    root: &'a serde_json::Value,
    path: &[String],
) -> Option<&'a serde_json::Value> {
    let mut cursor = root;
    for step in path {
        cursor = cursor.as_object()?.get(step)?;
    }
    Some(cursor)
}

/// Coerce a JSON leaf value to `kind` and push to the typed column. NULL on
/// type mismatch — the user opted into a target type, so we don't try to
/// stringify numbers etc. silently.
fn push_extracted_leaf(
    leaf: Option<&serde_json::Value>,
    kind: ColumnKind,
    typed_col: &mut TypedColumn,
) {
    let leaf = match leaf {
        Some(v) if !v.is_null() => v,
        _ => {
            push_typed_null(typed_col);
            return;
        }
    };
    match (kind, typed_col, leaf) {
        // Text: accept strings; numbers/bools/etc. stringify via to_string()
        (ColumnKind::Text, TypedColumn::Text(vec), serde_json::Value::String(s)) => {
            vec.push(Some(s.clone()));
        }
        (ColumnKind::Int16, TypedColumn::Int16(vec), serde_json::Value::Number(n)) => {
            vec.push(n.as_i64().and_then(|x| i16::try_from(x).ok()));
        }
        (ColumnKind::Int32, TypedColumn::Int32(vec), serde_json::Value::Number(n)) => {
            vec.push(n.as_i64().and_then(|x| i32::try_from(x).ok()));
        }
        (ColumnKind::Int64, TypedColumn::Int64(vec), serde_json::Value::Number(n)) => {
            vec.push(n.as_i64());
        }
        (ColumnKind::Int16, TypedColumn::Int16(vec), serde_json::Value::String(s)) => {
            vec.push(s.parse::<i16>().ok());
        }
        (ColumnKind::Int32, TypedColumn::Int32(vec), serde_json::Value::String(s)) => {
            vec.push(s.parse::<i32>().ok());
        }
        (ColumnKind::Int64, TypedColumn::Int64(vec), serde_json::Value::String(s)) => {
            vec.push(s.parse::<i64>().ok());
        }
        (ColumnKind::Float32, TypedColumn::Float32(vec), serde_json::Value::Number(n)) => {
            vec.push(n.as_f64().map(|x| x as f32));
        }
        (ColumnKind::Float64, TypedColumn::Float64(vec), serde_json::Value::Number(n)) => {
            vec.push(n.as_f64());
        }
        (ColumnKind::Bool, TypedColumn::Bool(vec), serde_json::Value::Bool(b)) => {
            vec.push(Some(*b));
        }
        (
            ColumnKind::Timestamp | ColumnKind::TimestampTz,
            TypedColumn::Int64(vec),
            serde_json::Value::String(s),
        ) => {
            // PG-format timestamp text. Best-effort parse; NULL on miss.
            // Wrap parse_timestamp_to_usec which currently doesn't return Result —
            // catch panics from malformed inputs and treat as NULL.
            let parsed = std::panic::catch_unwind(|| crate::timeparse::parse_timestamp_to_usec(s));
            vec.push(parsed.ok());
        }
        (ColumnKind::Date, TypedColumn::Int64(vec), serde_json::Value::String(s)) => {
            let parsed = std::panic::catch_unwind(|| crate::timeparse::parse_timestamp_to_usec(s));
            vec.push(parsed.ok());
        }
        // Anything else: type mismatch -> NULL.
        (_, typed_col, _) => {
            push_typed_null(typed_col);
        }
    }
}

fn is_valid_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_recognized_extract_type(t: &str) -> bool {
    let l = t.to_lowercase();
    matches!(
        l.as_str(),
        "text"
            | "varchar"
            | "char"
            | "smallint"
            | "int2"
            | "integer"
            | "int4"
            | "bigint"
            | "int8"
            | "real"
            | "float4"
            | "double precision"
            | "float8"
            | "boolean"
            | "bool"
            | "timestamp"
            | "timestamp without time zone"
            | "timestamp with time zone"
            | "timestamptz"
            | "date"
    )
}

// ============================================================================
// SQL-callable functions
// ============================================================================

/// Enable compression on a deltax deltatable.
///
/// ```sql
/// SELECT deltax_enable_compression('metrics',
///     segment_by => ARRAY['device_id'],
///     order_by => ARRAY['ts']);
/// ```
///
/// `json_extract` (optional) is a JSON array of `{src, path, name, type}`
/// objects describing JSON paths to extract from JSONB columns at COPY time
/// into extra columnar columns. Example:
///
/// ```sql
/// SELECT deltax_enable_compression('bluesky',
///     order_by => ARRAY['ts'],
///     json_extract => '[{"src":"data","path":["commit","collection"],
///                        "name":"x_collection","type":"text"}]'::jsonb);
/// ```
#[pg_extern]
fn deltax_enable_compression(
    relation: &str,
    segment_by: default!(Vec<String>, "ARRAY[]::text[]"),
    order_by: default!(Vec<String>, "ARRAY[]::text[]"),
    segment_size: default!(i32, "30000"),
    json_extract: default!(Option<pgrx::datum::JsonB>, "NULL"),
) -> String {
    maybe_warn_lz4();
    Spi::connect_mut(|client| {
        let (schema, table) = crate::partition::resolve_relation(client, relation);
        let ht = catalog::get_deltatable(client, &schema, &table)
            .expect("failed to query deltatable")
            .unwrap_or_else(|| {
                pgrx::error!(
                    "pg_deltax: table {}.{} is not a deltax table",
                    schema,
                    table
                )
            });

        // Validate segment_by columns exist
        for col in &segment_by {
            let exists = client
                .select(
                    "SELECT 1 FROM information_schema.columns
                     WHERE table_schema = $1 AND table_name = $2 AND column_name::text = $3",
                    None,
                    &[
                        schema.as_str().into(),
                        table.as_str().into(),
                        col.as_str().into(),
                    ],
                )
                .expect("failed to check column");
            if exists.is_empty() {
                pgrx::error!(
                    "pg_deltax: segment_by column '{}' not found in {}.{}",
                    col,
                    schema,
                    table
                );
            }
        }

        // If order_by is empty, default to the time column
        let effective_order_by = if order_by.is_empty() {
            vec![ht.time_column.clone()]
        } else {
            order_by
        };

        let effective_segment_size = if segment_size <= 0 {
            30000
        } else {
            segment_size
        };

        // Validate json_extract specs (if any) before persisting. Each spec's
        // src column must exist in the parent table and be jsonb. The names
        // must not collide with any physical column.
        let extract_summary = if let Some(ref jx) = json_extract {
            let specs = parse_extract_specs(&jx.0);
            for spec in &specs {
                let row = client
                    .select(
                        "SELECT data_type FROM information_schema.columns
                         WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
                        None,
                        &[
                            schema.as_str().into(),
                            table.as_str().into(),
                            spec.src_column.as_str().into(),
                        ],
                    )
                    .expect("failed to check src column");
                let dt: Option<String> = row
                    .first()
                    .get_one::<String>()
                    .expect("failed to read data_type");
                match dt {
                    None => pgrx::error!(
                        "pg_deltax: json_extract src column '{}' not found in {}.{}",
                        spec.src_column,
                        schema,
                        table
                    ),
                    Some(t) if t.to_lowercase() != "jsonb" => pgrx::error!(
                        "pg_deltax: json_extract src column '{}' must be jsonb (is {})",
                        spec.src_column,
                        t
                    ),
                    Some(_) => {}
                }
                // target_name must not collide with a physical column
                let collision = client
                    .select(
                        "SELECT 1 FROM information_schema.columns
                         WHERE table_schema = $1 AND table_name = $2 AND column_name = $3",
                        None,
                        &[
                            schema.as_str().into(),
                            table.as_str().into(),
                            spec.target_name.as_str().into(),
                        ],
                    )
                    .expect("failed to check name collision");
                if !collision.is_empty() {
                    pgrx::error!(
                        "pg_deltax: json_extract name '{}' collides with an existing column in {}.{}",
                        spec.target_name,
                        schema,
                        table
                    );
                }
            }
            format!(", json_extract: {} path(s)", specs.len())
        } else {
            String::new()
        };

        catalog::update_deltatable_compression(
            client,
            ht.id,
            &segment_by,
            &effective_order_by,
            effective_segment_size,
            json_extract,
        )
        .expect("failed to update compression settings");

        format!(
            "Compression enabled on {}.{} (segment_by: {:?}, order_by: {:?}, segment_size: {}{})",
            schema, table, segment_by, effective_order_by, effective_segment_size, extract_summary
        )
    })
}

/// Set the automatic compression policy for a deltatable.
#[pg_extern]
fn deltax_set_compression_policy(relation: &str, compress_after: pgrx::datum::Interval) -> String {
    Spi::connect_mut(|client| {
        let (schema, table) = crate::partition::resolve_relation(client, relation);
        let ht = catalog::get_deltatable(client, &schema, &table)
            .expect("failed to query deltatable")
            .unwrap_or_else(|| {
                pgrx::error!(
                    "pg_deltax: table {}.{} is not a deltax table",
                    schema,
                    table
                )
            });

        if ht.segment_by.is_empty() && ht.order_by.is_empty() {
            pgrx::error!("pg_deltax: enable compression first with deltax_enable_compression()");
        }

        catalog::set_compress_after(client, ht.id, &compress_after)
            .expect("failed to set compression policy");

        format!(
            "Compression policy set on {}.{}: compress_after = {}",
            schema, table, compress_after
        )
    })
}

/// Compress a single partition.
#[pg_extern]
fn deltax_compress_partition(partition: &str) -> String {
    Spi::connect_mut(|client| {
        let msg = compress_partition_impl(client, partition);
        // Refresh the parent-relation merged statistics so a manual
        // compress leaves the planner-visible stats current (with
        // pg_deltax.flatten_partitions the planner reads ONLY the parent's
        // pg_statistic rows — children aren't expanded). Deliberately in
        // the wrapper, NOT in compress_partition_impl: the maintenance
        // pass compresses many partitions in one transaction while holding
        // their AccessExclusive locks, and refreshing the parent's catalog
        // rows between partition lock acquisitions adds lock edges (and a
        // wide interleave window) against concurrent DDL like DROP TABLE
        // CASCADE — a real deadlock seen in CI. The pass refreshes once
        // per table after auto-compress instead (worker.rs).
        if msg.starts_with("Compressed") {
            refresh_parent_stats_for_partition(client, partition);
        }
        msg
    })
}

/// Best-effort parent-stats refresh for the deltatable owning `partition`.
/// Failures warn rather than error — the partition is compressed and
/// queryable either way, just with staler planner estimates.
fn refresh_parent_stats_for_partition(client: &mut SpiClient, partition: &str) {
    let (schema, part_table) = crate::partition::resolve_relation(client, partition);
    let ht = catalog::get_partition_by_name(client, &schema, &part_table)
        .ok()
        .flatten()
        .and_then(|p| {
            catalog::get_deltatable_by_id(client, p.deltatable_id)
                .ok()
                .flatten()
        });
    let Some(ht) = ht else { return };
    if let Err(e) = crate::stats::write_table_stats(client, &ht.schema_name, &ht.table_name) {
        pgrx::warning!(
            "pg_deltax: failed to refresh parent stats for {}.{}: {}. \
             Run deltax_analyze_table('{}.{}') to retry.",
            ht.schema_name,
            ht.table_name,
            e,
            ht.schema_name,
            ht.table_name,
        );
    }
}

/// Compress every "sealed" uncompressed partition of a deltax-managed table
/// in one call. A partition is sealed when its `range_end` is in the past
/// (with respect to `pg_deltax.mock_now` when set, otherwise `now()`); this
/// avoids racing with writers on the current partition. With `older_than`,
/// the threshold becomes `now() - older_than` so users can demand an extra
/// retention buffer.
///
/// Returns one row per partition the call touched, with the same status
/// string `deltax_compress_partition` produces. An empty result set means
/// nothing was eligible — not an error.
#[pg_extern]
fn deltax_compress_all_partitions(
    relation: &str,
    older_than: default!(Option<pgrx::datum::Interval>, "NULL"),
) -> TableIterator<'static, (name!(partition_name, String), name!(result, String))> {
    maybe_warn_lz4();

    let rows = Spi::connect_mut(|client| {
        let (schema, table) = crate::partition::resolve_relation(client, relation);
        let ht = catalog::get_deltatable(client, &schema, &table)
            .expect("failed to query deltatable")
            .unwrap_or_else(|| {
                pgrx::error!(
                    "pg_deltax: table {}.{} is not a deltax table",
                    schema,
                    table
                )
            });

        // Threshold is now() (mock-aware) minus older_than. interval_to_usec
        // rejects month-based intervals, matching the rest of the codebase.
        let now = crate::partition::now_usec();
        let threshold_usec = match &older_than {
            Some(i) => now - crate::partition::interval_to_usec(i),
            None => now,
        };
        let threshold_tstz = crate::partition::usec_to_tstz(threshold_usec);

        let result = client
            .select(
                "SELECT schema_name, table_name FROM deltax.deltax_partition
                 WHERE deltatable_id = $1
                   AND NOT is_compressed
                   AND range_end <= $2::timestamptz
                 ORDER BY range_start",
                None,
                &[ht.id.into(), threshold_tstz.into()],
            )
            .expect("failed to query eligible partitions");

        // Materialize the partition list first; `compress_partition_impl`
        // needs `&mut SpiClient` and would conflict with the live iterator.
        let mut targets: Vec<(String, String)> = Vec::new();
        for row in result {
            let sch: String = row
                .get_datum_by_ordinal(1)
                .unwrap()
                .value::<String>()
                .unwrap()
                .unwrap();
            let name: String = row
                .get_datum_by_ordinal(2)
                .unwrap()
                .value::<String>()
                .unwrap()
                .unwrap();
            targets.push((sch, name));
        }

        let mut out: Vec<(String, String)> = Vec::with_capacity(targets.len());
        let mut any_compressed = false;
        for (sch, name) in targets {
            // `compress_partition_impl` re-resolves the name via
            // `resolve_relation`, which expects an unquoted `schema.table`
            // (or bare) form. The deltax-generated partition names are
            // simple identifiers, so this is safe.
            let qualified = format!("{}.{}", sch, name);
            let msg = compress_partition_impl(client, &qualified);
            any_compressed |= msg.starts_with("Compressed");
            out.push((name, msg));
        }
        // One parent-stats refresh for the whole batch — see the
        // deltax_compress_partition wrapper for why this must not run
        // per-partition inside compress_partition_impl.
        if any_compressed
            && let Err(e) = crate::stats::write_table_stats(client, &ht.schema_name, &ht.table_name)
        {
            pgrx::warning!(
                "pg_deltax: failed to refresh parent stats for {}.{}: {}",
                ht.schema_name,
                ht.table_name,
                e,
            );
        }
        out
    });

    TableIterator::new(rows)
}

/// Decompress a single partition.
#[pg_extern]
fn deltax_decompress_partition(partition: &str) -> String {
    Spi::connect_mut(|client| decompress_partition_impl(client, partition))
}

/// Refresh `pg_class.reltuples` and `pg_statistic` for a compressed
/// partition from the existing `_colstats` data. Used to (re-)populate
/// planner stats on partitions that were compressed before the
/// stats-population path shipped, or after an accidental `ANALYZE` on
/// a compressed partition.
#[pg_extern]
fn deltax_analyze_partition(partition: &str) -> String {
    Spi::connect_mut(|client| analyze_partition_impl(client, partition))
}

/// Refresh stats on every compressed partition of a deltax-managed
/// table. Equivalent to calling `deltax_analyze_partition` on each
/// partition returned by `deltax_partition_info(relation)`.
#[pg_extern]
fn deltax_analyze_table(relation: &str) -> String {
    Spi::connect_mut(|client| analyze_table_impl(client, relation))
}

/// Show compression statistics for a deltatable.
#[pg_extern]
#[allow(clippy::type_complexity)]
fn deltax_compression_stats(
    relation: &str,
) -> TableIterator<
    'static,
    (
        name!(partition_name, String),
        name!(is_compressed, bool),
        name!(raw_size, Option<i64>),
        name!(compressed_size, Option<i64>),
        name!(compression_ratio, Option<f64>),
        name!(row_count, Option<i64>),
    ),
> {
    let rows = Spi::connect(|client| {
        let (schema, table) = crate::partition::resolve_relation(client, relation);
        let ht = catalog::get_deltatable(client, &schema, &table)
            .expect("failed to query deltatable")
            .unwrap_or_else(|| {
                pgrx::error!(
                    "pg_deltax: table {}.{} is not a deltax table",
                    schema,
                    table
                )
            });

        let result = client
            .select(
                "SELECT table_name, is_compressed, raw_size, compressed_size, row_count
                 FROM deltax.deltax_partition
                 WHERE deltatable_id = $1
                 ORDER BY range_start",
                None,
                &[ht.id.into()],
            )
            .expect("failed to query partitions");

        let mut rows = Vec::new();
        for row in result {
            let name: String = row
                .get_datum_by_ordinal(1)
                .unwrap()
                .value::<String>()
                .unwrap()
                .unwrap();
            let compressed: bool = row
                .get_datum_by_ordinal(2)
                .unwrap()
                .value::<bool>()
                .unwrap()
                .unwrap_or(false);
            let raw: Option<i64> = row.get_datum_by_ordinal(3).unwrap().value::<i64>().unwrap();
            let comp: Option<i64> = row.get_datum_by_ordinal(4).unwrap().value::<i64>().unwrap();
            let count: Option<i64> = row.get_datum_by_ordinal(5).unwrap().value::<i64>().unwrap();
            let ratio = match (raw, comp) {
                (Some(r), Some(c)) if c > 0 => Some(r as f64 / c as f64),
                _ => None,
            };
            rows.push((name, compressed, raw, comp, ratio, count));
        }
        rows
    });

    TableIterator::new(rows)
}

/// Return the total on-disk size of a deltatable in bytes.
///
/// For compressed partitions, uses the stored `compressed_size` from the catalog.
/// For uncompressed partitions, uses `pg_total_relation_size`.
#[pg_extern]
fn deltax_table_size(relation: &str) -> i64 {
    Spi::connect(|client| {
        let (schema, table) = crate::partition::resolve_relation(client, relation);
        let ht = catalog::get_deltatable(client, &schema, &table)
            .expect("failed to query deltatable")
            .unwrap_or_else(|| {
                pgrx::error!(
                    "pg_deltax: table {}.{} is not a deltax table",
                    schema,
                    table
                )
            });

        let result = client
            .select(
                "SELECT table_name, is_compressed
                 FROM deltax.deltax_partition
                 WHERE deltatable_id = $1",
                None,
                &[ht.id.into()],
            )
            .expect("failed to query partitions");

        let companion_schema = "_deltax_compressed";
        let mut total: i64 = 0;
        for row in result {
            let part_name: String = row
                .get_datum_by_ordinal(1)
                .unwrap()
                .value::<String>()
                .unwrap()
                .unwrap();
            let compressed: bool = row
                .get_datum_by_ordinal(2)
                .unwrap()
                .value::<bool>()
                .unwrap()
                .unwrap_or(false);

            if compressed {
                // Measure live size of companion tables
                for suffix in &["meta", "blobs", "blooms"] {
                    let fqn = format!("\"{}\".\"{}_{}\"", companion_schema, part_name, suffix);
                    total += estimate_raw_size(client, &fqn);
                }
            } else {
                let fqn = crate::partition::fqn(&schema, &part_name);
                total += estimate_raw_size(client, &fqn);
            }
        }
        total
    })
}

// ============================================================================
// Internal implementation
// ============================================================================

/// Scoped pin of the GUCs that affect the `::text` renderings compression
/// stores for fallback-typed columns (`classify_column` → `ColumnKind::Text`
/// with a non-text attribute type: interval, date/timestamp inside arrays or
/// composites, float arrays, ...). The stored text must parse back to the
/// same value under ANY reader session GUCs, so rendering pins:
///
/// - `IntervalStyle = postgres`: unit-labeled output that `interval_in`
///   parses identically under every IntervalStyle (sql_standard renderings
///   like '1-2' are re-interpreted differently per style — the pg_dump
///   caveat, which pins the same value);
/// - `DateStyle = ISO, MDY`: ISO date/timestamp text is unambiguous under
///   every DateStyle (a 'DMY'-rendered '02/03/2025' inside e.g. a
///   timestamptz[] would silently swap day/month for an 'MDY' reader);
/// - `extra_float_digits = 1`: shortest-precise float rendering; a session
///   with a negative setting would render floats inside arrays/composites
///   lossily — permanent data loss, not just a read-side issue.
///
/// Uses the standard PG nest-level save/restore idiom (`NewGUCNestLevel` +
/// `GUC_ACTION_SAVE` + `AtEOXact_GUC`), so the previous values come back at
/// scope exit instead of leaking a `SET LOCAL` into the caller's transaction.
/// On error, the transaction abort path restores them.
struct RenderGucGuard {
    nest_level: std::ffi::c_int,
}

impl RenderGucGuard {
    fn pin() -> Self {
        let nest_level = unsafe { pgrx::pg_sys::NewGUCNestLevel() };
        for (name, value) in [
            ("intervalstyle", "postgres"),
            ("datestyle", "ISO, MDY"),
            ("extra_float_digits", "1"),
        ] {
            let name_c = std::ffi::CString::new(name).unwrap();
            let value_c = std::ffi::CString::new(value).unwrap();
            unsafe {
                pgrx::pg_sys::set_config_option(
                    name_c.as_ptr(),
                    value_c.as_ptr(),
                    pgrx::pg_sys::GucContext::PGC_USERSET,
                    pgrx::pg_sys::GucSource::PGC_S_SESSION,
                    pgrx::pg_sys::GucAction::GUC_ACTION_SAVE,
                    true,
                    pgrx::pg_sys::PGERROR as std::ffi::c_int,
                    false,
                );
            }
        }
        Self { nest_level }
    }
}

impl Drop for RenderGucGuard {
    fn drop(&mut self) {
        // On panic/error the transaction abort path runs AtEOXact_GUC itself;
        // calling it here with isCommit=true would be wrong mid-abort.
        if !std::thread::panicking() {
            unsafe { pgrx::pg_sys::AtEOXact_GUC(true, self.nest_level) };
        }
    }
}

fn compress_partition_impl(client: &mut SpiClient, partition: &str) -> String {
    // Pin rendering GUCs for the whole pass: covers the streaming `::text`
    // SELECT and the segment-by stats/valcounts queries.
    let _render_gucs = RenderGucGuard::pin();
    // 1. Look up partition in catalog
    let (schema, part_table) = crate::partition::resolve_relation(client, partition);
    let part_info = catalog::get_partition_by_name(client, &schema, &part_table)
        .expect("failed to query partition")
        .unwrap_or_else(|| {
            pgrx::error!(
                "pg_deltax: partition {}.{} not found in catalog",
                schema,
                part_table
            )
        });

    if part_info.is_compressed {
        return format!("Partition {}.{} is already compressed", schema, part_table);
    }

    // Open month partition must stay heap — compressed partitions are read-only.
    let range_end_usec = part_info.range_end.into_inner() + PG_EPOCH_OFFSET_USEC;
    if range_end_usec > crate::partition::now_usec() {
        pgrx::error!(
            "pg_deltax: refusing to compress open partition {}.{} (range_end {} > now())",
            schema,
            part_table,
            part_info.range_end
        );
    }

    // 2. Get deltatable info (compression settings)
    let ht = catalog::get_deltatable_by_id(client, part_info.deltatable_id)
        .expect("failed to query deltatable")
        .unwrap();

    if ht.order_by.is_empty() && ht.segment_by.is_empty() {
        pgrx::error!(
            "pg_deltax: compression not enabled on {}.{}. Call deltax_enable_compression() first.",
            ht.schema_name,
            ht.table_name
        );
    }

    // 3. Get column metadata
    let columns = get_column_metadata(
        client,
        &schema,
        &part_table,
        &ht.segment_by,
        &ht.time_column,
        ht.json_extract.as_ref(),
    );
    if columns.is_empty() {
        pgrx::error!("pg_deltax: no columns found for {}.{}", schema, part_table);
    }

    // 4. Estimate row count from pg_class stats (instant, no scan).
    // Used for skipping empty partitions.
    // reltuples: 0 = empty (known), -1 = unknown (no ANALYZE yet), >0 = estimated count.
    let part_fqn = crate::partition::fqn(&schema, &part_table);
    let reltuples = client
        .select(
            &format!(
                "SELECT reltuples::int8 FROM pg_class WHERE oid = '{}'::regclass",
                part_fqn
            ),
            None,
            &[],
        )
        .expect("failed to get reltuples")
        .first()
        .get_one::<i64>()
        .unwrap()
        .unwrap_or(-1);

    // reltuples = 0 means PG knows the partition is empty (e.g. freshly created).
    // Skip compression — creating the companion table would confuse the scan hook.
    if reltuples == 0 {
        return format!(
            "Partition {}.{} has no rows to compress",
            schema, part_table
        );
    }

    // 5. Build companion table DDL: meta (thin) + colstats (wide) + blobs + blooms
    let ddl = build_companion_ddl(&part_table, &columns);
    // NOTE: table creation is deferred until we confirm data exists.
    // Creating it early would cause the scan hook to intercept queries on the partition
    // (it checks for meta table existence, not is_compressed in the catalog).

    // 6. Read and compress data per segment
    let raw_size = estimate_raw_size(client, &part_fqn);

    let segment_size = ht.segment_size as usize;

    let (
        total_compressed_size,
        row_count,
        partition_hll,
        mut column_valmap,
        mut column_valcounts,
        partition_topvals,
    ) = compress_partition_streaming(
        client,
        &part_fqn,
        &ddl,
        &columns,
        &ht.order_by,
        &ht.segment_by,
        segment_size,
    );

    // Empty partition — clean up tables and return
    if row_count == 0 {
        client
            .update(&format!("DROP TABLE IF EXISTS {}", ddl.meta_fqn), None, &[])
            .expect("failed to drop empty meta table");
        client
            .update(
                &format!("DROP TABLE IF EXISTS {}", ddl.colstats_fqn),
                None,
                &[],
            )
            .expect("failed to drop empty colstats table");
        client
            .update(
                &format!("DROP TABLE IF EXISTS {}", ddl.blobs_fqn),
                None,
                &[],
            )
            .expect("failed to drop empty blobs table");
        client
            .update(
                &format!("DROP TABLE IF EXISTS {}", ddl.blooms_fqn),
                None,
                &[],
            )
            .expect("failed to drop empty blooms table");
        client
            .update(
                &format!("DROP TABLE IF EXISTS {}", ddl.text_lengths_fqn),
                None,
                &[],
            )
            .expect("failed to drop empty text_lengths table");
        client
            .update(
                &format!("DROP TABLE IF EXISTS {}", ddl.valbitmap_fqn),
                None,
                &[],
            )
            .expect("failed to drop empty valbitmap table");
        return format!(
            "Partition {}.{} has no rows to compress",
            schema, part_table
        );
    }

    // 8. Truncate original partition (stays attached to parent)
    client
        .update(&format!("TRUNCATE {}", part_fqn), None, &[])
        .expect("failed to truncate partition");

    // 9. Update catalog
    catalog::mark_partition_compressed(
        client,
        part_info.id,
        total_compressed_size,
        raw_size,
        row_count,
    )
    .expect("failed to update catalog");
    catalog::install_compressed_dml_trigger(client, &schema, &part_table)
        .expect("failed to install compressed partition DML trigger");

    // P2.5: eagerly create the (empty) `_tombstones` companion so the
    // OWNER/GRANT cascade covers it and tombstone DELETEs by non-owner
    // roles don't need CREATE on the companion schema. Zero blocks ⇒ zero
    // cost on every read path.
    client
        .update(&ddl_if_not_exists(&ddl.tombstones_ddl), None, &[])
        .expect("failed to create tombstones table");

    // Persist per-column ndistinct from the partition-level HLL merge
    // (strictly more accurate than the old MAX-over-segments approach,
    // especially for time-clustered high-cardinality keys like order_id
    // where per-segment HLL sees only a fraction of global distinct
    // values). The resulting JSONB map feeds both `scan::cost` planner
    // estimates and the `pg_statistic.stadistinct` write below.
    let nd_col_names: Vec<&str> = columns
        .iter()
        .filter(|c| !c.is_segment_by)
        .map(|c| c.name.as_str())
        .collect();
    let mut col_ndistinct: std::collections::HashMap<String, i64> = nd_col_names
        .iter()
        .zip(partition_hll.iter())
        .map(|(name, hll)| ((*name).to_string(), hll.estimate() as i64))
        .collect();
    // Fold segment-by columns (absent from the HLL/valbitmap) into the stat
    // maps from the meta table's exact (segment value, _row_count) so the
    // child + parent pg_statistic machinery covers them.
    augment_segment_by_stats(
        client,
        &ddl.meta_fqn,
        &columns,
        &mut col_ndistinct,
        &mut column_valmap,
        &mut column_valcounts,
    );
    catalog::update_partition_column_ndistinct_from_map(client, part_info.id, &col_ndistinct)
        .expect("failed to update partition column_ndistinct");

    // Persist the per-column HLL sketches so the table-wide distinct count can
    // be computed by merging them across partitions (see write_table_stats).
    if let Some(hll_json) = serialize_partition_hll(&nd_col_names, &partition_hll) {
        catalog::update_partition_column_hll(client, part_info.id, &hll_json)
            .expect("failed to update partition column_hll");
    }

    // Persist the partition-level value→bit_idx maps for low-card text
    // columns. Empty map is fine — the read path treats a missing entry
    // for a column as "no bitmap available, fall back to bloom/batch
    // filtering".
    catalog::update_partition_column_valmap(client, part_info.id, &column_valmap)
        .expect("failed to update partition column_valmap");

    // Persist the summed per-value occurrence counts for those same low-card
    // columns. `stats.rs` divides by the partition row count to write real
    // `most_common_freqs` (skewed enums no longer get a flat 1/ndistinct).
    catalog::update_partition_column_valcounts(client, part_info.id, &column_valcounts)
        .expect("failed to update partition column_valcounts");

    // For high-cardinality text columns (not covered by the <=32 complete
    // valmap), persist a partial MCV: the heavy hitters from the top-value
    // summary that are notably more common than uniform. `stats.rs` writes
    // these as an MCV while keeping the real HLL `stadistinct`, so PG estimates
    // hot values from the MCV and the long tail from the remainder.
    let column_mcv = select_partial_mcv(
        &partition_topvals,
        &nd_col_names,
        &col_ndistinct,
        &column_valmap,
        row_count,
    );
    catalog::update_partition_column_mcv(client, part_info.id, &column_mcv)
        .expect("failed to update partition column_mcv");

    // Aggregate per-segment colstats into a partition-level {col_name: [min,max]}
    // map. Read path uses this to skip partitions whose [min, max] range
    // doesn't cover the const in `WHERE col = const` queries — cuts the
    // 60µs/partition setup cost for non-matching partitions on wide scans.
    catalog::update_partition_column_minmax(client, part_info.id, &ddl.colstats_fqn, &columns)
        .expect("failed to update partition column_minmax");

    // Snapshot the physical-column shape so the scan path can decode this
    // partition's blobs even if the parent's pg_attribute changes later
    // (e.g. ADD COLUMN with a default). See dev/docs/SCHEMA_CHANGES.md.
    let cc_json = catalog::snapshot_compressed_columns(
        client,
        &ht.schema_name,
        &ht.table_name,
        &ht.segment_by,
    )
    .expect("failed to snapshot compressed_columns");
    catalog::update_partition_compressed_columns(client, part_info.id, &cc_json)
        .expect("failed to update partition compressed_columns");

    // Populate pg_class.reltuples + pg_statistic for the compressed
    // child partition so PG's built-in selectivity functions stop
    // falling back to defaults (0.005 equality-sel, ~2.5e-5 text-eq).
    // Failure here is WARNING, not fatal — the partition is still
    // queryable with pessimistic estimates.
    let part_rel_oid: pg_sys::Oid = client
        .select(&format!("SELECT '{}'::regclass::oid", part_fqn), None, &[])
        .expect("failed to resolve partition oid")
        .first()
        .get_one::<pg_sys::Oid>()
        .ok()
        .flatten()
        .unwrap_or(pg_sys::InvalidOid);
    if part_rel_oid != pg_sys::InvalidOid
        && let Err(e) = crate::stats::write_partition_stats(
            client,
            part_rel_oid,
            &col_ndistinct,
            row_count,
            &ddl.colstats_fqn,
            &columns,
        )
    {
        pgrx::warning!(
            "pg_deltax: failed to update pg_statistic for {}: {}. \
             Run deltax_analyze_partition('{}') to retry.",
            part_fqn,
            e,
            part_fqn,
        );
    }

    // Disable autovacuum so user-triggered ANALYZE (including the
    // autovacuum launcher) doesn't sample this empty-heap partition
    // and wipe the pg_statistic rows we just wrote. The ProcessUtility
    // hook (src/copy.rs) also filters explicit `ANALYZE <part>` calls
    // as a belt-and-suspenders safeguard.
    let _ = client.update(
        &format!("ALTER TABLE {} SET (autovacuum_enabled = off)", part_fqn),
        None,
        &[],
    );

    crate::scan::invalidate_compressed_cache();

    format!(
        "Compressed {}.{}: {} rows, ratio {:.1}x",
        schema,
        part_table,
        row_count,
        if total_compressed_size > 0 {
            raw_size as f64 / total_compressed_size as f64
        } else {
            0.0
        }
    )
}

// ============================================================================
// Typed column storage — avoids text round-trip for numeric/boolean columns
// ============================================================================

/// Classifies how to read a column from SPI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ColumnKind {
    Text,        // text, varchar, char — read as String
    Int16,       // smallint/int2
    Int32,       // integer/int4
    Int64,       // bigint/int8
    Float32,     // real/float4
    Float64,     // double precision/float8
    Bool,        // boolean/bool
    Timestamp,   // timestamp without time zone — read as pgrx::Timestamp → i64 usec
    TimestampTz, // timestamp with time zone — read as pgrx::TimestampWithTimeZone → i64 usec
    Date,        // date — read as pgrx::Date → i64 usec
    Time,        // time without time zone — read as pgrx::Time → i64 usec since midnight
    Jsonb,       // jsonb — stored as the binary varlena form produced by jsonb_in
    Uuid,        // uuid — stored as the raw 16 bytes (fixed-length, by-reference)
    Bytea,       // bytea — stored as the raw varlena payload bytes
    Inet,        // inet/cidr — stored as the raw varlena payload (inet_struct bytes)
}

/// Column data stored in native types.
///
/// `Bytes` holds opaque byte blobs (used for jsonb, where the stored payload is
/// PG's binary jsonb varlena, which is not UTF-8 and therefore cannot fit in a
/// Rust `String`). On-disk compression, sort, and ndistinct paths treat it
/// identically to `Text` — both are variable-length byte sequences — but the
/// decompression-to-Datum path skips UTF-8 validation and hands the raw bytes
/// back to PG wrapped in a varlena header tagged with `JSONBOID`.
#[derive(Debug, PartialEq)]
pub(crate) enum TypedColumn {
    Text(Vec<Option<String>>),
    Bytes(Vec<Option<Vec<u8>>>),
    Int16(Vec<Option<i16>>),
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Float32(Vec<Option<f32>>),
    Float64(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
}

impl TypedColumn {
    /// Split off elements from index `at` onward, returning them as a new TypedColumn.
    /// `self` retains elements `0..at`.
    pub(crate) fn split_off(&mut self, at: usize) -> Self {
        match self {
            TypedColumn::Text(v) => TypedColumn::Text(v.split_off(at)),
            TypedColumn::Bytes(v) => TypedColumn::Bytes(v.split_off(at)),
            TypedColumn::Int16(v) => TypedColumn::Int16(v.split_off(at)),
            TypedColumn::Int32(v) => TypedColumn::Int32(v.split_off(at)),
            TypedColumn::Int64(v) => TypedColumn::Int64(v.split_off(at)),
            TypedColumn::Float32(v) => TypedColumn::Float32(v.split_off(at)),
            TypedColumn::Float64(v) => TypedColumn::Float64(v.split_off(at)),
            TypedColumn::Bool(v) => TypedColumn::Bool(v.split_off(at)),
        }
    }

    pub(crate) fn extend(&mut self, other: Self) {
        match (self, other) {
            (TypedColumn::Text(a), TypedColumn::Text(b)) => a.extend(b),
            (TypedColumn::Bytes(a), TypedColumn::Bytes(b)) => a.extend(b),
            (TypedColumn::Int16(a), TypedColumn::Int16(b)) => a.extend(b),
            (TypedColumn::Int32(a), TypedColumn::Int32(b)) => a.extend(b),
            (TypedColumn::Int64(a), TypedColumn::Int64(b)) => a.extend(b),
            (TypedColumn::Float32(a), TypedColumn::Float32(b)) => a.extend(b),
            (TypedColumn::Float64(a), TypedColumn::Float64(b)) => a.extend(b),
            (TypedColumn::Bool(a), TypedColumn::Bool(b)) => a.extend(b),
            _ => panic!("TypedColumn::extend: mismatched variants"),
        }
    }

    /// Push a single row from `src` at index `idx` into `self`.
    pub(crate) fn push_from(&mut self, src: &Self, idx: usize) {
        match (self, src) {
            (TypedColumn::Text(dst), TypedColumn::Text(s)) => dst.push(s[idx].clone()),
            (TypedColumn::Bytes(dst), TypedColumn::Bytes(s)) => dst.push(s[idx].clone()),
            (TypedColumn::Int16(dst), TypedColumn::Int16(s)) => dst.push(s[idx]),
            (TypedColumn::Int32(dst), TypedColumn::Int32(s)) => dst.push(s[idx]),
            (TypedColumn::Int64(dst), TypedColumn::Int64(s)) => dst.push(s[idx]),
            (TypedColumn::Float32(dst), TypedColumn::Float32(s)) => dst.push(s[idx]),
            (TypedColumn::Float64(dst), TypedColumn::Float64(s)) => dst.push(s[idx]),
            (TypedColumn::Bool(dst), TypedColumn::Bool(s)) => dst.push(s[idx]),
            _ => panic!("TypedColumn::push_from: mismatched variants"),
        }
    }
}

pub(crate) fn classify_column(data_type: &str, is_segment_by: bool) -> ColumnKind {
    if is_segment_by {
        return ColumnKind::Text; // segment_by always read as text for SQL literals
    }
    let dt = data_type.to_lowercase();
    if dt == "smallint" || dt == "int2" {
        ColumnKind::Int16
    } else if dt == "integer" || dt == "int4" {
        ColumnKind::Int32
    } else if dt == "bigint" || dt == "int8" {
        ColumnKind::Int64
    } else if dt == "double precision" || dt == "float8" {
        ColumnKind::Float64
    } else if dt == "real" || dt == "float4" {
        ColumnKind::Float32
    } else if dt == "boolean" || dt == "bool" {
        ColumnKind::Bool
    } else if dt == "timestamp with time zone" {
        ColumnKind::TimestampTz
    } else if dt.contains("timestamp") {
        ColumnKind::Timestamp
    } else if dt == "date" {
        ColumnKind::Date
    } else if dt == "jsonb" {
        ColumnKind::Jsonb
    } else if crate::FORCE_TEXT_FALLBACK.get() {
        // TESTING: force the graduated types back onto the legacy ::text
        // codecs so mixed-generation reads can be exercised with current code.
        ColumnKind::Text
    } else if dt == "time" || dt == "time without time zone" {
        // timetz stays on the text fallback (carries a zone offset).
        ColumnKind::Time
    } else if dt == "uuid" {
        ColumnKind::Uuid
    } else if dt == "bytea" {
        ColumnKind::Bytea
    } else if dt == "inet" || dt == "cidr" {
        ColumnKind::Inet
    } else {
        ColumnKind::Text
    }
}

pub(crate) fn new_typed_column(kind: ColumnKind) -> TypedColumn {
    match kind {
        ColumnKind::Text => TypedColumn::Text(Vec::new()),
        ColumnKind::Jsonb => TypedColumn::Bytes(Vec::new()),
        ColumnKind::Int16 => TypedColumn::Int16(Vec::new()),
        ColumnKind::Int32 => TypedColumn::Int32(Vec::new()),
        ColumnKind::Int64 => TypedColumn::Int64(Vec::new()),
        ColumnKind::Float32 => TypedColumn::Float32(Vec::new()),
        ColumnKind::Float64 => TypedColumn::Float64(Vec::new()),
        ColumnKind::Bool => TypedColumn::Bool(Vec::new()),
        ColumnKind::Timestamp | ColumnKind::TimestampTz | ColumnKind::Date | ColumnKind::Time => {
            TypedColumn::Int64(Vec::new())
        }
        ColumnKind::Uuid | ColumnKind::Bytea | ColumnKind::Inet => TypedColumn::Bytes(Vec::new()),
    }
}

/// Worker-thread variant of `new_typed_column`. Identical except that JSONB
/// columns are accumulated as `Text` instead of `Bytes`, because converting
/// JSON text to the binary jsonb varlena requires `jsonb_in`, which calls
/// PG memory-context and function-manager APIs that are not safe to invoke
/// from a non-backend thread. The merge phase converts Text → Bytes on the
/// main thread before the data reaches the partition buffer.
pub(crate) fn new_worker_typed_column(kind: ColumnKind) -> TypedColumn {
    match kind {
        ColumnKind::Jsonb => TypedColumn::Text(Vec::new()),
        _ => new_typed_column(kind),
    }
}

/// Create empty TypedColumn vectors for all columns based on their ColumnKind.
pub(crate) fn init_typed_columns(columns: &[ColumnMeta], kinds: &[ColumnKind]) -> Vec<TypedColumn> {
    columns
        .iter()
        .zip(kinds.iter())
        .map(|(_, kind)| new_typed_column(*kind))
        .collect()
}

/// Extract one SPI row into typed column accumulators using native datum access.
/// Segment_by columns are skipped (their TypedColumn slots remain empty).
/// Push a NULL into any TypedColumn variant.
fn push_typed_null(col: &mut TypedColumn) {
    match col {
        TypedColumn::Int16(v) => v.push(None),
        TypedColumn::Int32(v) => v.push(None),
        TypedColumn::Int64(v) => v.push(None),
        TypedColumn::Float32(v) => v.push(None),
        TypedColumn::Float64(v) => v.push(None),
        TypedColumn::Bool(v) => v.push(None),
        TypedColumn::Text(v) => v.push(None),
        TypedColumn::Bytes(v) => v.push(None),
    }
}

fn append_row_to_columns(
    row: &pgrx::spi::SpiHeapTupleData,
    columns: &[ColumnMeta],
    kinds: &[ColumnKind],
    typed_cols: &mut [TypedColumn],
) {
    for (i, (col, kind)) in columns.iter().zip(kinds.iter()).enumerate() {
        if col.is_segment_by {
            continue;
        }
        // Synthetic extracted columns have no SPI ordinal — they must be
        // populated from the source jsonb in their own pass. Until that's
        // wired up for the SPI-fetch (post-INSERT) compression path, push
        // NULL placeholders so per-segment row counts stay aligned.
        if col.extracted.is_some() {
            push_typed_null(&mut typed_cols[i]);
            continue;
        }
        let ordinal = i + 1; // SPI ordinals are 1-based
        match kind {
            ColumnKind::Int16 => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<i16>()
                    .unwrap();
                if let TypedColumn::Int16(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Int32 => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<i32>()
                    .unwrap();
                if let TypedColumn::Int32(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Int64 => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<i64>()
                    .unwrap();
                if let TypedColumn::Int64(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Float32 => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<f32>()
                    .unwrap();
                if let TypedColumn::Float32(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Float64 => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<f64>()
                    .unwrap();
                if let TypedColumn::Float64(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Bool => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<bool>()
                    .unwrap();
                if let TypedColumn::Bool(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Timestamp => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<pgrx::datum::Timestamp>()
                    .unwrap();
                if let TypedColumn::Int64(vec) = &mut typed_cols[i] {
                    // Convert PG-epoch usec to Unix-epoch usec
                    vec.push(v.map(|ts| ts.into_inner() + PG_EPOCH_OFFSET_USEC));
                }
            }
            ColumnKind::TimestampTz => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<pgrx::datum::TimestampWithTimeZone>()
                    .unwrap();
                if let TypedColumn::Int64(vec) = &mut typed_cols[i] {
                    // Convert PG-epoch usec to Unix-epoch usec
                    vec.push(v.map(|ts| ts.into_inner() + PG_EPOCH_OFFSET_USEC));
                }
            }
            ColumnKind::Date => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<pgrx::datum::Date>()
                    .unwrap();
                if let TypedColumn::Int64(vec) = &mut typed_cols[i] {
                    // Convert PG-epoch days to Unix-epoch usec
                    vec.push(v.map(|d| {
                        ((d.into_inner() as i64) + PG_EPOCH_OFFSET_DAYS) * 86_400_000_000
                    }));
                }
            }
            ColumnKind::Time => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<pgrx::datum::Time>()
                    .unwrap();
                if let TypedColumn::Int64(vec) = &mut typed_cols[i] {
                    // TimeADT is already microseconds since midnight
                    vec.push(v.map(|t| t.into_inner()));
                }
            }
            ColumnKind::Text => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<String>()
                    .unwrap();
                if let TypedColumn::Text(vec) = &mut typed_cols[i] {
                    vec.push(v);
                }
            }
            ColumnKind::Jsonb => {
                // Classic compression path (post-INSERT). SPI returns native
                // jsonb Datums, so we read the on-disk binary varlena payload
                // directly via `JsonbRaw` — no jsonb_out/serde_json/jsonb_in
                // round-trip (which would be lossy for high-precision numbers
                // and leak a jsonb_in datum per row). The bytes are the same
                // canonical jsonb container the COPY path produces via
                // `jsonb_text_to_binary`, so the scan path is unaffected.
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<JsonbRaw>()
                    .unwrap();
                if let TypedColumn::Bytes(vec) = &mut typed_cols[i] {
                    vec.push(v.map(|j| j.0));
                }
            }
            ColumnKind::Uuid => {
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<UuidRaw>()
                    .unwrap();
                if let TypedColumn::Bytes(vec) = &mut typed_cols[i] {
                    vec.push(v.map(|u| u.0.to_vec()));
                }
            }
            ColumnKind::Bytea | ColumnKind::Inet => {
                // Same raw-varlena-payload read as jsonb: detoast and copy
                // the bytes after the varlena header.
                let v = row
                    .get_datum_by_ordinal(ordinal)
                    .unwrap()
                    .value::<JsonbRaw>()
                    .unwrap();
                if let TypedColumn::Bytes(vec) = &mut typed_cols[i] {
                    vec.push(v.map(|j| j.0));
                }
            }
        }
    }
}

/// A fixed-length pass-by-reference Datum (uuid: 16 bytes) read verbatim.
pub(crate) struct UuidRaw(pub [u8; 16]);

impl FromDatum for UuidRaw {
    unsafe fn from_polymorphic_datum(
        datum: pgrx::pg_sys::Datum,
        is_null: bool,
        _typoid: pgrx::pg_sys::Oid,
    ) -> Option<Self> {
        if is_null {
            return None;
        }
        unsafe {
            let ptr = datum.cast_mut_ptr::<u8>();
            let mut bytes = [0u8; 16];
            std::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), 16);
            Some(UuidRaw(bytes))
        }
    }
}

thread_local! {
    /// Reusable scratch memory context for per-row `jsonb_in` calls.
    /// `jsonb_in` leaves its parse-tree allocations in `CurrentMemoryContext`
    /// and doesn't free them — for a 181M-row rtabench load that's tens of GB
    /// of leaked parse nodes. We switch to this scratch context for each call
    /// and `MemoryContextReset` after, which reclaims everything cheaply.
    static JSONB_SCRATCH_CTX: std::cell::Cell<pgrx::pg_sys::MemoryContext> =
        const { std::cell::Cell::new(std::ptr::null_mut()) };
}

/// A `jsonb` Datum read as its raw on-disk binary varlena payload (everything
/// after the varlena header), with no parse/serialize round-trip.
///
/// Reading a `jsonb` column via pgrx's `JsonB` would route the value through
/// `jsonb_out` → `serde_json::Value` → `jsonb_in`, which is both expensive and
/// lossy: pgrx's `serde_json` has no `arbitrary_precision`, so numbers outside
/// i64/u64/f64 range (high-precision decimals, large integers) get rounded.
/// We instead detoast and copy the binary container verbatim — the same bytes
/// `jsonb_text_to_binary` produces, so the scan path reconstructs identically.
pub(crate) struct JsonbRaw(pub Vec<u8>);

impl FromDatum for JsonbRaw {
    unsafe fn from_polymorphic_datum(
        datum: pgrx::pg_sys::Datum,
        is_null: bool,
        _typoid: pgrx::pg_sys::Oid,
    ) -> Option<Self> {
        if is_null {
            return None;
        }
        unsafe {
            let varlena = datum.cast_mut_ptr::<pgrx::pg_sys::varlena>();
            let detoasted = pgrx::pg_sys::pg_detoast_datum(varlena);
            let total_len = pgrx::varsize_any_exhdr(detoasted);
            let data_ptr = pgrx::vardata_any(detoasted).cast::<u8>();
            let bytes = std::slice::from_raw_parts(data_ptr, total_len).to_vec();
            // pg_detoast_datum allocates a copy in CurrentMemoryContext only
            // when the datum was actually toasted; free it so a long compress
            // loop doesn't accumulate detoasted copies.
            if detoasted != varlena {
                pgrx::pg_sys::pfree(detoasted.cast());
            }
            Some(JsonbRaw(bytes))
        }
    }
}

impl IntoDatum for JsonbRaw {
    fn into_datum(self) -> Option<pgrx::pg_sys::Datum> {
        // Read-only helper: only the FromDatum side is used (via SpiHeapTupleDataEntry::value).
        // Required by the IntoDatum bound on `value::<T>()` and the binary-coercibility check.
        unreachable!("JsonbRaw is read-only and must not be converted back into a Datum")
    }

    fn type_oid() -> pgrx::pg_sys::Oid {
        pgrx::pg_sys::JSONBOID
    }

    fn is_compatible_with(other: pgrx::pg_sys::Oid) -> bool {
        // JsonbRaw is a generic "detoasted varlena payload" reader — the
        // bytea/inet native codecs reuse it to capture their raw payload
        // bytes (see the Bytea/Inet extraction arms).
        matches!(
            other,
            pgrx::pg_sys::JSONBOID
                | pgrx::pg_sys::BYTEAOID
                | pgrx::pg_sys::INETOID
                | pgrx::pg_sys::CIDROID
        )
    }
}

impl IntoDatum for UuidRaw {
    fn into_datum(self) -> Option<pgrx::pg_sys::Datum> {
        // Read-only helper, same contract as JsonbRaw.
        unreachable!("UuidRaw is read-only and must not be converted back into a Datum")
    }

    fn type_oid() -> pgrx::pg_sys::Oid {
        pgrx::pg_sys::UUIDOID
    }
}

/// Convert canonical JSON text to the binary jsonb varlena payload
/// (everything after the varlena header) by calling PG's `jsonb_in`.
/// Caller stores the returned bytes verbatim; to reconstruct a Datum, wrap
/// them in a fresh varlena header. See `byte_slices_to_jsonb_datums_arena`
/// in datum_utils.
pub(crate) unsafe fn jsonb_text_to_binary(text: &str) -> Vec<u8> {
    unsafe {
        let scratch = jsonb_scratch_ctx();

        let old = pgrx::pg_sys::MemoryContextSwitchTo(scratch);

        let c_text = std::ffi::CString::new(text).expect("jsonb text contains null byte");
        let mut typinput: pgrx::pg_sys::Oid = pgrx::pg_sys::InvalidOid;
        let mut typioparam: pgrx::pg_sys::Oid = pgrx::pg_sys::InvalidOid;
        pgrx::pg_sys::getTypeInputInfo(pgrx::pg_sys::JSONBOID, &mut typinput, &mut typioparam);
        let datum =
            pgrx::pg_sys::OidInputFunctionCall(typinput, c_text.as_ptr() as *mut _, typioparam, -1);
        let varlena = datum.cast_mut_ptr::<pgrx::pg_sys::varlena>();
        let detoasted = pgrx::pg_sys::pg_detoast_datum(varlena);
        let total_len = pgrx::varsize_any_exhdr(detoasted);
        let data_ptr = pgrx::vardata_any(detoasted).cast::<u8>();
        // Copy into Rust heap before resetting the scratch context.
        let bytes = std::slice::from_raw_parts(data_ptr, total_len).to_vec();

        pgrx::pg_sys::MemoryContextSwitchTo(old);
        pgrx::pg_sys::MemoryContextReset(scratch);

        bytes
    }
}

/// Lazily create (once per backend) and return the scratch memory context
/// used by the per-row jsonb conversions above/below.
unsafe fn jsonb_scratch_ctx() -> pgrx::pg_sys::MemoryContext {
    unsafe {
        JSONB_SCRATCH_CTX.with(|c| {
            let p = c.get();
            if p.is_null() {
                let new_ctx = pgrx::pg_sys::AllocSetContextCreateInternal(
                    pgrx::pg_sys::TopMemoryContext,
                    c"pg_deltax_jsonb_scratch".as_ptr(),
                    pgrx::pg_sys::ALLOCSET_SMALL_MINSIZE as usize,
                    pgrx::pg_sys::ALLOCSET_SMALL_INITSIZE as usize,
                    pgrx::pg_sys::ALLOCSET_SMALL_MAXSIZE as usize,
                );
                c.set(new_ctx);
                new_ctx
            } else {
                p
            }
        })
    }
}

/// A type's output function resolved once, for converting stored binary
/// payloads (jsonb container bytes, uuid bytes, bytea/inet varlena payloads)
/// back to their canonical text inside per-value loops. Hoists the
/// `getTypeOutputInfo` syscache lookup and the `fmgr_info` resolution out of
/// the loop. Must only be used on the main backend thread.
pub(crate) struct TypeOutFn {
    finfo: pgrx::pg_sys::FmgrInfo,
}

impl TypeOutFn {
    pub(crate) unsafe fn resolve(type_oid: pgrx::pg_sys::Oid) -> Self {
        unsafe {
            let mut typoutput: pgrx::pg_sys::Oid = pgrx::pg_sys::InvalidOid;
            let mut typisvarlena: bool = false;
            pgrx::pg_sys::getTypeOutputInfo(type_oid, &mut typoutput, &mut typisvarlena);
            let mut finfo = pgrx::pg_sys::FmgrInfo::default();
            pgrx::pg_sys::fmgr_info(typoutput, &mut finfo);
            Self { finfo }
        }
    }

    unsafe fn call_out(&mut self, datum: pgrx::pg_sys::Datum) -> String {
        unsafe {
            let cstr = pgrx::pg_sys::OutputFunctionCall(&mut self.finfo, datum);
            let text = std::ffi::CStr::from_ptr(cstr)
                .to_str()
                .expect("type output function produced invalid UTF-8")
                .to_string();
            pgrx::pg_sys::pfree(cstr.cast());
            text
        }
    }

    /// Wrap `payload` (the bytes after the varlena header) in a fresh varlena
    /// header and call the type's output function. For jsonb this is the
    /// exact inverse of `jsonb_text_to_binary`.
    pub(crate) unsafe fn render(&mut self, payload: &[u8]) -> String {
        unsafe {
            let total_len = pgrx::pg_sys::VARHDRSZ + payload.len();
            let varlena = pgrx::pg_sys::palloc(total_len) as *mut pgrx::pg_sys::varlena;
            pgrx::set_varsize_4b(varlena, total_len as i32);
            std::ptr::copy_nonoverlapping(
                payload.as_ptr(),
                (varlena as *mut u8).add(pgrx::pg_sys::VARHDRSZ),
                payload.len(),
            );
            let text = self.call_out(pgrx::pg_sys::Datum::from(varlena as usize));
            pgrx::pg_sys::pfree(varlena.cast());
            text
        }
    }

    /// Render a fixed-length by-reference value (uuid: 16 bytes) — the datum
    /// is a bare pointer to the bytes, no varlena header.
    pub(crate) unsafe fn render_fixed(&mut self, bytes: &[u8]) -> String {
        unsafe {
            let buf = pgrx::pg_sys::palloc(bytes.len()) as *mut u8;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());
            let text = self.call_out(pgrx::pg_sys::Datum::from(buf as usize));
            pgrx::pg_sys::pfree(buf.cast());
            text
        }
    }
}

/// Sort typed columns in-place by the given order_by column indices.
/// Computes a permutation from the sort keys, then reorders all columns by that permutation.
pub(crate) fn sort_typed_columns(
    typed_cols: &mut [TypedColumn],
    order_col_indices: &[usize],
    num_rows: usize,
) {
    if order_col_indices.is_empty() || num_rows <= 1 {
        return;
    }

    // Build sort permutation using indices
    let mut perm: Vec<usize> = (0..num_rows).collect();
    perm.sort_by(|&a, &b| {
        for &col_idx in order_col_indices {
            let cmp = match &typed_cols[col_idx] {
                TypedColumn::Int16(v) => v[a].cmp(&v[b]),
                TypedColumn::Int32(v) => v[a].cmp(&v[b]),
                TypedColumn::Int64(v) => v[a].cmp(&v[b]),
                TypedColumn::Float32(v) => {
                    let fa = v[a].map(|f| f.to_bits());
                    let fb = v[b].map(|f| f.to_bits());
                    fa.cmp(&fb)
                }
                TypedColumn::Float64(v) => {
                    let fa = v[a].map(|f| f.to_bits());
                    let fb = v[b].map(|f| f.to_bits());
                    fa.cmp(&fb)
                }
                TypedColumn::Bool(v) => v[a].cmp(&v[b]),
                TypedColumn::Text(v) => v[a].cmp(&v[b]),
                TypedColumn::Bytes(v) => v[a].cmp(&v[b]),
            };
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });

    // Apply permutation to all columns
    for tc in typed_cols.iter_mut() {
        match tc {
            TypedColumn::Int16(v) => apply_permutation(v, &perm),
            TypedColumn::Int32(v) => apply_permutation(v, &perm),
            TypedColumn::Int64(v) => apply_permutation(v, &perm),
            TypedColumn::Float32(v) => apply_permutation(v, &perm),
            TypedColumn::Float64(v) => apply_permutation(v, &perm),
            TypedColumn::Bool(v) => apply_permutation(v, &perm),
            TypedColumn::Text(v) => apply_permutation(v, &perm),
            TypedColumn::Bytes(v) => apply_permutation(v, &perm),
        }
    }
}

/// Reorder a Vec according to a permutation, returning a new Vec.
fn apply_permutation<T: Clone>(v: &mut Vec<T>, perm: &[usize]) {
    let reordered: Vec<T> = perm.iter().map(|&i| v[i].clone()).collect();
    *v = reordered;
}

/// A single row for the normalized colstats table.
pub(crate) struct ColstatsRow {
    pub(crate) col_idx: i16,
    pub(crate) segment_id: i32,
    pub(crate) min_val: Option<i64>,
    pub(crate) max_val: Option<i64>,
    pub(crate) sum_val: Option<String>, // NUMERIC as string
    pub(crate) nonnull_count: i32,
    pub(crate) nonzero_count: i32,
    pub(crate) ndistinct: i64,
}

/// Per-segment `(value, occurrence_count)` list for one low-cardinality text
/// column, sorted by value. The values feed the valbitmap (presence) + the
/// catalog `column_valmap`; the counts are summed across segments into
/// `column_valcounts`, which lets `stats.rs` write *real* `most_common_freqs`
/// instead of a uniform `1/ndistinct` (e.g. `event_type='Approved'` is 41%,
/// not 11% — see PLANNER_STATS.md P1).
pub(crate) type SegValueCounts = Vec<(String, u32)>;

/// Partition-level summed per-value occurrence counts, keyed by user column
/// name: `{col_name: [(value, count), ...]}`. Persisted as
/// `deltax_partition.column_valcounts` and read by `stats.rs` to write real
/// `pg_statistic.most_common_freqs`.
pub(crate) type ColumnValcounts = std::collections::HashMap<String, Vec<(String, i64)>>;

/// Return type for flush_segment_metadata: (compressed_size, column blobs,
/// per-column bloom entries, colstats rows, per-text-column length sidecars,
/// per-text-column value+count lists for valbitmap/valcounts).
/// Each bloom entry is (col_idx, num_hashes, bloom_bytes).
/// Each text-length entry is (col_idx, length_blob).
/// Each valbitmap entry is (col_idx, sorted (value, count) list) — only for text
/// columns with ≤ `VALBITMAP_MAX_DISTINCT` distinct values in this segment.
pub(crate) type FlushResult = (
    i64,
    Vec<(u16, Vec<u8>)>,
    Vec<(u16, u8, Vec<u8>)>,
    Vec<ColstatsRow>,
    Vec<(u16, Vec<u8>)>,
    Vec<(u16, SegValueCounts)>,
);

/// Cap on distinct values for the per-segment value-presence bitmap. Each
/// segment's bitmap is one bit per distinct partition-level value, so 32
/// values fit in 4 bytes. Columns whose partition-level distinct count
/// exceeds this cap are dropped from valbitmap entirely (no entry written).
pub(crate) const VALBITMAP_MAX_DISTINCT: usize = 32;

/// Cap on distinct values tracked per text column for the partial-MCV
/// (heavy-hitter) summary. Generous enough to hold every distinct value of a
/// real categorical column exactly; for higher-cardinality columns we stop
/// admitting new values once full — a genuinely hot value appears early so it
/// is captured, and the significance filter at finalize drops the cold tail.
pub(crate) const MCV_MAX_DISTINCT: usize = 2048;

/// Per-non-segment-by-column exact value→count maps (text columns only) — the
/// partition-level heavy-hitter summary feeding the partial MCV for skewed
/// high-cardinality text columns. Same per-non-seg-col shape as `partition_hll`.
pub(crate) type TopVals = Vec<std::collections::HashMap<String, i64>>;

/// Fold one segment's text values into the partition-level top-value summary
/// `acc` (indexed by non-segment-by column position, like `partition_hll`).
/// Once a column's map reaches `MCV_MAX_DISTINCT` new values are dropped but
/// existing counts keep accumulating. Cheap: O(1) per value.
pub(crate) fn merge_segment_topvals(
    typed_cols: &[TypedColumn],
    columns: &[ColumnMeta],
    acc: &mut TopVals,
) {
    // Lazily size to one map per non-segment-by column (the COPY path starts
    // with an empty Vec); num_nonseg is constant for a partition so this only
    // fires once.
    let num_nonseg = columns.iter().filter(|c| !c.is_segment_by).count();
    if acc.len() != num_nonseg {
        *acc = (0..num_nonseg)
            .map(|_| std::collections::HashMap::new())
            .collect();
    }
    let mut nonseg = 0usize;
    for (i, col) in columns.iter().enumerate() {
        if col.is_segment_by {
            continue;
        }
        if nonseg < acc.len()
            && let TypedColumn::Text(vals) = &typed_cols[i]
        {
            let m = &mut acc[nonseg];
            for v in vals.iter().flatten() {
                if let Some(c) = m.get_mut(v) {
                    *c += 1;
                } else if m.len() < MCV_MAX_DISTINCT {
                    m.insert(v.clone(), 1);
                }
            }
        }
        nonseg += 1;
    }
}

/// Compress accumulated typed column data and INSERT metadata into the meta table.
/// Returns (compressed_size, column blobs, bloom entries, colstats rows) — blobs and colstats
/// are NOT inserted, they are returned for column-major buffering by the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_segment_metadata(
    client: &mut SpiClient,
    meta_fqn: &str,
    _colstats_fqn: &str,
    columns: &[ColumnMeta],
    typed_cols: &[TypedColumn],
    segment_by_values: &[Option<String>],
    ndistinct_values: &[i64],
    row_count: u32,
    segment_id: i32,
) -> FlushResult {
    // Returns (compressed_size, blobs, bloom_entries)
    // Compress each non-segment column, collect blobs for caller
    let mut blobs: Vec<(u16, Vec<u8>)> = Vec::new(); // (col_idx, compressed_data)
    let mut col_minmax: std::collections::HashMap<String, (Option<String>, Option<String>)> =
        std::collections::HashMap::new();
    let mut total_size: i64 = 0;

    let mut col_sums: std::collections::HashMap<String, (Option<String>, i64, i64)> =
        std::collections::HashMap::new();

    // Per-text-column length sidecars (col_idx, length_blob).
    let mut text_length_blobs: Vec<(u16, Vec<u8>)> = Vec::new();

    let mut col_idx: u16 = 0;
    for (i, col) in columns.iter().enumerate() {
        if col.is_segment_by {
            continue;
        }
        let compressed = compress_typed_column(&typed_cols[i], &col.data_type);
        if supports_minmax(&col.data_type) {
            let (min_val, max_val) = compute_typed_minmax(&typed_cols[i], &col.data_type);
            col_minmax.insert(col.name.clone(), (min_val, max_val));
        }
        if supports_sum(&col.data_type) {
            col_sums.insert(col.name.clone(), compute_typed_sum(&typed_cols[i]));
        }
        // Build length sidecar for text columns. The main blob already contains
        // the string bodies; the sidecar lets queries that only need
        // length(col)/col='' skip detoasting the main blob.
        if is_text_data_type(&col.data_type.to_lowercase())
            && let TypedColumn::Text(vals) = &typed_cols[i]
        {
            text_length_blobs.push((col_idx, compress_text_lengths(vals)));
        }
        total_size += compressed.len() as i64;
        blobs.push((col_idx, compressed));
        col_idx += 1;
    }

    // Build INSERT for thin meta table: segment_id, segment_by, time min/max, row_count
    let mut meta_cols = Vec::new();
    let mut meta_vals = Vec::new();

    meta_cols.push("_segment_id".to_string());
    meta_vals.push(segment_id.to_string());

    // Segment-by columns
    let mut seg_idx = 0;
    for col in columns {
        if col.is_segment_by {
            meta_cols.push(format!("\"{}\"", col.name));
            if seg_idx < segment_by_values.len() {
                match &segment_by_values[seg_idx] {
                    Some(v) => meta_vals.push(format!("'{}'", v.replace('\'', "''"))),
                    None => meta_vals.push("NULL".to_string()),
                }
                seg_idx += 1;
            }
        }
    }

    // Time column min/max only
    for col in columns {
        if col.is_time_column && !col.is_segment_by && supports_minmax(&col.data_type) {
            meta_cols.push(format!("\"_min_{}\"", col.name));
            meta_cols.push(format!("\"_max_{}\"", col.name));
            match col_minmax.get(&col.name) {
                Some((Some(min_val), Some(max_val))) => {
                    meta_vals.push(format_minmax_for_insert(min_val, &col.data_type));
                    meta_vals.push(format_minmax_for_insert(max_val, &col.data_type));
                }
                _ => {
                    meta_vals.push("NULL".to_string());
                    meta_vals.push("NULL".to_string());
                }
            }
        }
    }

    meta_cols.push("_row_count".to_string());
    meta_vals.push(row_count.to_string());

    let meta_sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        meta_fqn,
        meta_cols.join(", "),
        meta_vals.join(", ")
    );
    client
        .update(&meta_sql, None, &[])
        .expect("failed to insert segment metadata");

    // Build normalized colstats rows: one per non-segment-by column
    // Rows are returned to the caller for column-major buffering (sorted by col_idx, segment_id).
    let mut cs_rows: Vec<ColstatsRow> = Vec::new();
    let mut col_idx_counter: i16 = 0;
    let mut nd_idx = 0;
    for (i, col) in columns.iter().enumerate() {
        if col.is_segment_by {
            continue;
        }
        let (min_enc, max_enc) = compute_minmax_encoded_i64(&typed_cols[i], &col.data_type);

        let (sum_val, nonnull, nonzero) = if supports_sum(&col.data_type) {
            let (s, nn, nz) = col_sums.get(&col.name).cloned().unwrap_or((None, 0, 0));
            (s, nn as i32, nz as i32)
        } else {
            (None, 0, 0)
        };

        let nd = if nd_idx < ndistinct_values.len() {
            ndistinct_values[nd_idx]
        } else {
            0
        };
        nd_idx += 1;

        cs_rows.push(ColstatsRow {
            col_idx: col_idx_counter,
            segment_id,
            min_val: min_enc,
            max_val: max_enc,
            sum_val,
            nonnull_count: nonnull,
            nonzero_count: nonzero,
            ndistinct: nd,
        });
        col_idx_counter += 1;
    }

    // Compute per-column bloom filters (if enabled via GUC) — stored separately
    let bloom_entries = if crate::BLOOM_FILTERS.get() {
        compute_segment_blooms(typed_cols, columns, ndistinct_values)
    } else {
        Vec::new()
    };

    // Per-segment distinct-value sets for low-cardinality text columns. The
    // bitmap itself is encoded later (in `compress_partition_streaming`)
    // once the partition-level value→bit_idx map is finalized.
    let valbitmap_value_sets = compute_segment_valbitmap_values(typed_cols, columns);

    (
        total_size,
        blobs,
        bloom_entries,
        cs_rows,
        text_length_blobs,
        valbitmap_value_sets,
    )
}

/// Collect per-segment distinct text values for low-cardinality columns.
/// Returns one `(col_idx, sorted_values)` entry per text column whose
/// distinct count in this segment is ≤ `VALBITMAP_MAX_DISTINCT`. Columns
/// that overflow the cap are simply omitted — the partition-level finalize
/// pass treats a missing entry as "give up on bitmap for this column".
pub(crate) fn compute_segment_valbitmap_values(
    typed_cols: &[TypedColumn],
    columns: &[ColumnMeta],
) -> Vec<(u16, SegValueCounts)> {
    let mut entries: Vec<(u16, SegValueCounts)> = Vec::new();
    let mut col_idx: u16 = 0;
    for (i, col) in columns.iter().enumerate() {
        if col.is_segment_by {
            continue;
        }
        if let TypedColumn::Text(vals) = &typed_cols[i] {
            // Count occurrences per distinct value. As soon as the distinct
            // count would exceed the cap we know this column can't get a
            // bitmap, so we bail and skip counting the rest.
            let mut counts: std::collections::BTreeMap<String, u32> =
                std::collections::BTreeMap::new();
            let mut overflow = false;
            for v in vals.iter().flatten() {
                if counts.len() >= VALBITMAP_MAX_DISTINCT && !counts.contains_key(v) {
                    overflow = true;
                    break;
                }
                *counts.entry(v.clone()).or_insert(0) += 1;
            }
            if !overflow {
                // BTreeMap iteration is already sorted by value.
                entries.push((col_idx, counts.into_iter().collect()));
            }
        }
        col_idx += 1;
    }
    entries
}

/// Slice a TypedColumn to a sub-range [start..end).
/// Empty columns (e.g. segment_by placeholders) are returned as-is.
fn slice_typed_column(tc: &TypedColumn, start: usize, end: usize) -> TypedColumn {
    match tc {
        TypedColumn::Text(v) if v.is_empty() => TypedColumn::Text(Vec::new()),
        TypedColumn::Text(v) => TypedColumn::Text(v[start..end].to_vec()),
        TypedColumn::Int16(v) if v.is_empty() => TypedColumn::Int16(Vec::new()),
        TypedColumn::Int16(v) => TypedColumn::Int16(v[start..end].to_vec()),
        TypedColumn::Int32(v) if v.is_empty() => TypedColumn::Int32(Vec::new()),
        TypedColumn::Int32(v) => TypedColumn::Int32(v[start..end].to_vec()),
        TypedColumn::Int64(v) if v.is_empty() => TypedColumn::Int64(Vec::new()),
        TypedColumn::Int64(v) => TypedColumn::Int64(v[start..end].to_vec()),
        TypedColumn::Float32(v) if v.is_empty() => TypedColumn::Float32(Vec::new()),
        TypedColumn::Float32(v) => TypedColumn::Float32(v[start..end].to_vec()),
        TypedColumn::Float64(v) if v.is_empty() => TypedColumn::Float64(Vec::new()),
        TypedColumn::Float64(v) => TypedColumn::Float64(v[start..end].to_vec()),
        TypedColumn::Bool(v) if v.is_empty() => TypedColumn::Bool(Vec::new()),
        TypedColumn::Bool(v) => TypedColumn::Bool(v[start..end].to_vec()),
        TypedColumn::Bytes(v) if v.is_empty() => TypedColumn::Bytes(Vec::new()),
        TypedColumn::Bytes(v) => TypedColumn::Bytes(v[start..end].to_vec()),
    }
}

/// Hash a value and return the hash for HLL insertion.
fn hash_for_hll<T: Hash>(val: &T) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    val.hash(&mut hasher);
    hasher.finish()
}

/// Merge per-segment HLL sketches into a partition-level accumulator (one
/// sketch per non-segment-by column). The accumulator is lazily sized on the
/// first call; subsequent calls union each column's sketch. Used by both load
/// paths to build the per-partition sketch persisted as `column_hll`.
pub(crate) fn accumulate_partition_hll(
    acc: &mut Vec<CardinalityEstimator<u64>>,
    sketches: &[CardinalityEstimator<u64>],
) {
    if acc.len() != sketches.len() {
        *acc = (0..sketches.len())
            .map(|_| CardinalityEstimator::<u64>::new())
            .collect();
    }
    for (a, s) in acc.iter_mut().zip(sketches) {
        a.merge(s);
    }
}

/// Merge a per-segment top-value summary into the partition-level accumulator
/// (both indexed by non-segment-by column position). Used by the parallel COPY
/// path, where segments are summarized off-thread and merged on the main thread.
/// Capped at `MCV_MAX_DISTINCT` per column.
pub(crate) fn merge_topvals_into(acc: &mut TopVals, seg: TopVals) {
    if acc.len() != seg.len() {
        *acc = seg;
        return;
    }
    for (a, s) in acc.iter_mut().zip(seg) {
        for (v, c) in s {
            if let Some(x) = a.get_mut(&v) {
                *x += c;
            } else if a.len() < MCV_MAX_DISTINCT {
                a.insert(v, c);
            }
        }
    }
}

/// Serialize a partition's per-column HLL sketches to a JSON object string
/// `{col_name: <sketch>}` for storage in `deltax_partition.column_hll`.
/// `stats::write_table_stats` deserializes and merges these across partitions
/// to get an accurate table-wide distinct count for join/range estimation.
pub(crate) fn serialize_partition_hll(
    col_names: &[&str],
    hll: &[CardinalityEstimator<u64>],
) -> Option<String> {
    if hll.is_empty() {
        return None;
    }
    let mut map = serde_json::Map::new();
    for (name, sketch) in col_names.iter().zip(hll.iter()) {
        if let Ok(v) = serde_json::to_value(sketch) {
            map.insert((*name).to_string(), v);
        }
    }
    if map.is_empty() {
        return None;
    }
    Some(serde_json::Value::Object(map).to_string())
}

/// Compute per-segment ndistinct using HyperLogLog estimators.
/// Returns (per-non-segment-by-column estimates, per-non-segment-by-column HLL sketches).
/// The sketches can be merged across segments to compute a partition-level
/// cardinality estimate (used by `src/stats.rs` to populate `pg_statistic`).
pub(crate) fn compute_segment_ndistinct(
    typed_cols: &[TypedColumn],
    columns: &[ColumnMeta],
) -> (Vec<i64>, Vec<CardinalityEstimator<u64>>) {
    let mut estimates = Vec::new();
    let mut sketches = Vec::new();
    for (i, col) in columns.iter().enumerate() {
        if col.is_segment_by {
            continue;
        }
        let mut hll = CardinalityEstimator::<u64>::new();
        match &typed_cols[i] {
            TypedColumn::Int16(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(x));
                }
            }
            TypedColumn::Int32(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(x));
                }
            }
            TypedColumn::Int64(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(x));
                }
            }
            TypedColumn::Float32(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(&x.to_bits()));
                }
            }
            TypedColumn::Float64(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(&x.to_bits()));
                }
            }
            TypedColumn::Bool(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(x));
                }
            }
            TypedColumn::Text(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(x));
                }
            }
            TypedColumn::Bytes(v) => {
                for x in v.iter().flatten() {
                    hll.insert_hash(hash_for_hll(x));
                }
            }
        }
        estimates.push(hll.estimate() as i64);
        sketches.push(hll);
    }
    (estimates, sketches)
}

/// Compute per-column bloom filters for a segment.
/// Returns one (col_idx, num_hashes, bloom_bytes) entry per column that got a bloom,
/// or empty if no columns qualify. Only builds bloom filters for numeric/date/timestamp
/// columns with ndistinct > 0.
pub(crate) fn compute_segment_blooms(
    typed_cols: &[TypedColumn],
    columns: &[ColumnMeta],
    ndistinct_values: &[i64],
) -> Vec<(u16, u8, Vec<u8>)> {
    use crate::bloom::{BloomFilter, hash_datum_i64};

    let mut entries: Vec<(u16, u8, Vec<u8>)> = Vec::new();
    let mut nd_idx: usize = 0;
    let mut col_idx: u16 = 0;

    for (i, col) in columns.iter().enumerate() {
        if col.is_segment_by {
            continue;
        }
        let nd = if nd_idx < ndistinct_values.len() {
            ndistinct_values[nd_idx]
        } else {
            0
        };
        nd_idx += 1;

        if !supports_minmax(&col.data_type) || nd <= 0 {
            col_idx += 1;
            continue;
        }

        let mut bf = BloomFilter::for_ndistinct(nd as usize);
        match &typed_cols[i] {
            TypedColumn::Int16(v) => {
                for x in v.iter().flatten() {
                    bf.insert(hash_datum_i64(*x as i64));
                }
            }
            TypedColumn::Int32(v) => {
                for x in v.iter().flatten() {
                    bf.insert(hash_datum_i64(*x as i64));
                }
            }
            TypedColumn::Int64(v) => {
                for x in v.iter().flatten() {
                    bf.insert(hash_datum_i64(*x));
                }
            }
            TypedColumn::Float32(v) => {
                for x in v.iter().flatten() {
                    bf.insert(hash_datum_i64(x.to_bits() as i64));
                }
            }
            TypedColumn::Float64(v) => {
                for x in v.iter().flatten() {
                    bf.insert(hash_datum_i64(x.to_bits() as i64));
                }
            }
            _ => {
                col_idx += 1;
                continue;
            }
        }

        entries.push((col_idx, bf.num_hashes(), bf.as_bytes().to_vec()));
        col_idx += 1;
    }

    entries
}

/// Flush typed column data, splitting into segment_size chunks if needed.
/// Returns compressed_size. Blobs and blooms are buffered for batch insertion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn flush_with_splitting(
    client: &mut SpiClient,
    meta_fqn: &str,
    colstats_fqn: &str,
    columns: &[ColumnMeta],
    typed_cols: &[TypedColumn],
    seg_values: &[Option<String>],
    total_rows: usize,
    segment_size: usize,
    next_segment_id: &mut i32,
    blob_buffer: &mut Vec<(u16, i32, Vec<u8>)>,
    bloom_buffer: &mut Vec<(u16, i32, u8, Vec<u8>)>,
    colstats_buffer: &mut Vec<ColstatsRow>,
    text_length_buffer: &mut Vec<(u16, i32, Vec<u8>)>,
    valbitmap_value_buffer: &mut Vec<(u16, i32, SegValueCounts)>,
    partition_hll: &mut [CardinalityEstimator<u64>],
    partition_topvals: &mut TopVals,
) -> i64 {
    let mut total_size = 0i64;
    let mut offset = 0;
    while offset < total_rows {
        let chunk_end = (offset + segment_size).min(total_rows);
        let chunk_rows = (chunk_end - offset) as u32;
        let seg_id = *next_segment_id;
        *next_segment_id += 1;
        if offset == 0 && chunk_end == total_rows {
            let (ndistinct, sketches) = compute_segment_ndistinct(typed_cols, columns);
            for (dst, src) in partition_hll.iter_mut().zip(sketches.iter()) {
                dst.merge(src);
            }
            merge_segment_topvals(typed_cols, columns, partition_topvals);
            let (size, blobs, bloom_entries, cs_rows, length_blobs, vb_values) =
                flush_segment_metadata(
                    client,
                    meta_fqn,
                    colstats_fqn,
                    columns,
                    typed_cols,
                    seg_values,
                    &ndistinct,
                    chunk_rows,
                    seg_id,
                );
            total_size += size;
            for (col_idx, blob) in blobs {
                blob_buffer.push((col_idx, seg_id, blob));
            }
            for (col_idx, num_hashes, bytes) in bloom_entries {
                bloom_buffer.push((col_idx, seg_id, num_hashes, bytes));
            }
            for (col_idx, blob) in length_blobs {
                text_length_buffer.push((col_idx, seg_id, blob));
            }
            for (col_idx, vals) in vb_values {
                valbitmap_value_buffer.push((col_idx, seg_id, vals));
            }
            colstats_buffer.extend(cs_rows);
        } else {
            let chunk_cols: Vec<TypedColumn> = typed_cols
                .iter()
                .map(|tc| slice_typed_column(tc, offset, chunk_end))
                .collect();
            let (ndistinct, sketches) = compute_segment_ndistinct(&chunk_cols, columns);
            for (dst, src) in partition_hll.iter_mut().zip(sketches.iter()) {
                dst.merge(src);
            }
            merge_segment_topvals(&chunk_cols, columns, partition_topvals);
            let (size, blobs, bloom_entries, cs_rows, length_blobs, vb_values) =
                flush_segment_metadata(
                    client,
                    meta_fqn,
                    colstats_fqn,
                    columns,
                    &chunk_cols,
                    seg_values,
                    &ndistinct,
                    chunk_rows,
                    seg_id,
                );
            total_size += size;
            for (col_idx, blob) in blobs {
                blob_buffer.push((col_idx, seg_id, blob));
            }
            for (col_idx, num_hashes, bytes) in bloom_entries {
                bloom_buffer.push((col_idx, seg_id, num_hashes, bytes));
            }
            for (col_idx, blob) in length_blobs {
                text_length_buffer.push((col_idx, seg_id, blob));
            }
            for (col_idx, vals) in vb_values {
                valbitmap_value_buffer.push((col_idx, seg_id, vals));
            }
            colstats_buffer.extend(cs_rows);
        }
        offset = chunk_end;
    }
    total_size
}

/// DDL info for all companion tables of a compressed partition.
pub(crate) struct CompanionDdl {
    pub(crate) meta_fqn: String,
    pub(crate) colstats_fqn: String,
    pub(crate) blobs_fqn: String,
    pub(crate) blooms_fqn: String,
    pub(crate) text_lengths_fqn: String,
    pub(crate) valbitmap_fqn: String,
    pub(crate) meta_ddl: String,
    pub(crate) colstats_ddl: String,
    pub(crate) blobs_ddl: String,
    pub(crate) blooms_ddl: String,
    pub(crate) text_lengths_ddl: String,
    pub(crate) valbitmap_ddl: String,
    pub(crate) tombstones_ddl: String,
}

/// Cached probe result for whether the running PostgreSQL was built with
/// `--with-lz4`. lz4 support is a postmaster compile-time property, so one
/// probe per backend is sufficient.
static LZ4_SUPPORTED: OnceLock<bool> = OnceLock::new();

/// Set to true after we've emitted the one-shot `use_lz4=on but PG lacks
/// lz4` WARNING for the current backend, so we don't spam users that enable
/// compression on multiple tables.
static LZ4_WARNED: AtomicBool = AtomicBool::new(false);

/// Detect whether the running PostgreSQL was built with `--with-lz4`.
/// Cheap probe: `default_toast_compression` is an enum GUC whose accepted
/// values include `lz4` only when the server has lz4 linked in.
pub(crate) fn lz4_supported() -> bool {
    *LZ4_SUPPORTED.get_or_init(|| {
        Spi::get_one::<bool>(
            "SELECT enumvals @> ARRAY['lz4'] \
             FROM pg_settings WHERE name = 'default_toast_compression'",
        )
        .ok()
        .flatten()
        .unwrap_or(false)
    })
}

/// Pure logic for [`lz4_clause`]: emit the lz4 attribute only when the user
/// has opted in (`use_lz4=on`) *and* the server supports it. Split out so
/// unit tests can cover all four combinations without depending on the
/// cached probe or the live GUC.
fn compute_lz4_clause(use_lz4: bool, supported: bool) -> &'static str {
    if use_lz4 && supported {
        " COMPRESSION lz4"
    } else {
        ""
    }
}

/// Returns `" COMPRESSION lz4"` (with a leading space) when the running PG
/// supports lz4 and the `pg_deltax.use_lz4` GUC is on; otherwise `""`. Used
/// at the seven companion-table DDL sites.
pub(crate) fn lz4_clause() -> &'static str {
    compute_lz4_clause(USE_LZ4.get(), lz4_supported())
}

/// Emit a one-shot WARNING per backend when `use_lz4=on` was requested
/// but the running PG wasn't built with lz4 support. Called from
/// `deltax_enable_compression`.
pub(crate) fn maybe_warn_lz4() {
    if !USE_LZ4.get() || lz4_supported() {
        return;
    }
    if LZ4_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    pgrx::warning!(
        "pg_deltax: PostgreSQL was not built with --with-lz4; \
        compressed-blob storage might be larger and cold reads slower."
    )
}

/// Build DDL for companion tables (meta, colstats, blobs, blooms) for a partition.
///
/// The meta table is thin: only segment_id, segment_by cols, time column min/max,
/// and row_count. All other per-column stats (min/max for non-time columns,
/// sum/count, ndistinct) go into the colstats table.
pub(crate) fn build_companion_ddl(part_table: &str, columns: &[ColumnMeta]) -> CompanionDdl {
    let companion_schema = "_deltax_compressed";
    let meta_fqn = format!("\"{}\".\"{}_meta\"", companion_schema, part_table);
    let colstats_fqn = format!("\"{}\".\"{}_colstats\"", companion_schema, part_table);
    let blobs_fqn = format!("\"{}\".\"{}_blobs\"", companion_schema, part_table);
    let blooms_fqn = format!("\"{}\".\"{}_blooms\"", companion_schema, part_table);
    let text_lengths_fqn = format!("\"{}\".\"{}_text_lengths\"", companion_schema, part_table);
    let valbitmap_fqn = format!("\"{}\".\"{}_valbitmap\"", companion_schema, part_table);

    // Thin meta table: segment_id, segment_by cols, time column min/max, row_count
    let mut meta_cols = Vec::new();
    meta_cols.push("_segment_id INT PRIMARY KEY".to_string());
    for col in columns {
        if col.is_segment_by {
            meta_cols.push(format!("\"{}\" {}", col.name, col.data_type));
        }
    }
    for col in columns {
        if col.is_time_column && !col.is_segment_by && supports_minmax(&col.data_type) {
            meta_cols.push(format!("\"_min_{}\" {}", col.name, col.data_type));
            meta_cols.push(format!("\"_max_{}\" {}", col.name, col.data_type));
        }
    }
    meta_cols.push("_row_count INT".to_string());

    let meta_ddl = format!("CREATE TABLE {} ({})", meta_fqn, meta_cols.join(", "));

    // Normalized colstats table: fixed 8-column schema
    let colstats_ddl = format!(
        "CREATE TABLE {} (\
         _col_idx SMALLINT NOT NULL, \
         _segment_id INT NOT NULL, \
         _min INT8, \
         _max INT8, \
         _sum NUMERIC, \
         _nonnull_count INT, \
         _nonzero_count INT, \
         _ndistinct INT, \
         PRIMARY KEY (_col_idx, _segment_id))",
        colstats_fqn
    );

    let lz4 = lz4_clause();

    let blobs_ddl = format!(
        "CREATE TABLE {} (_col_idx SMALLINT NOT NULL, _segment_id INT NOT NULL, _data BYTEA{}, PRIMARY KEY (_col_idx, _segment_id))",
        blobs_fqn, lz4
    );

    let blooms_ddl = format!(
        "CREATE TABLE {} (_col_idx SMALLINT NOT NULL, _segment_id INT NOT NULL, _num_hashes SMALLINT NOT NULL, _data BYTEA{} NOT NULL, PRIMARY KEY (_col_idx, _segment_id))",
        blooms_fqn, lz4
    );

    // Per-text-column per-segment length sidecar: compact u32 array, LZ4-compressed.
    // Used when a query only needs length(col)/col=''/col<>'' — lets the scan skip
    // detoasting the (typically large) main text blob.
    let text_lengths_ddl = format!(
        "CREATE TABLE {} (_col_idx SMALLINT NOT NULL, _segment_id INT NOT NULL, _data BYTEA{} NOT NULL, PRIMARY KEY (_col_idx, _segment_id))",
        text_lengths_fqn, lz4
    );

    // Per-segment value-presence bitmap for low-cardinality (≤32) text columns.
    // One bit per distinct partition-level value (mapping persisted in
    // `deltax.deltax_partition.column_valmap`). Lets `WHERE col = const` queries skip
    // segments where the constant's bit is clear, with no false positives.
    let valbitmap_ddl = format!(
        "CREATE TABLE {} (_col_idx SMALLINT NOT NULL, _segment_id INT NOT NULL, _bits BYTEA{} NOT NULL, PRIMARY KEY (_col_idx, _segment_id))",
        valbitmap_fqn, lz4
    );

    // P2.5 tombstones: one row per logically deleted segment row
    // (`dev/docs/COMPRESSED_DML.md` P2.5). Ordinary heap rows ⇒ MVCC,
    // rollback and replication are native. Created (empty) at compress time
    // so the GRANT/OWNER cascade covers it; zero blocks = zero read-side
    // cost. The PK doubles as the indexed existence probe.
    let tombstones_fqn = format!("\"{}\".\"{}_tombstones\"", companion_schema, part_table);
    let tombstones_ddl = format!(
        "CREATE TABLE {} (_segment_id INT NOT NULL, _row_offset INT NOT NULL, PRIMARY KEY (_segment_id, _row_offset))",
        tombstones_fqn
    );

    CompanionDdl {
        meta_fqn,
        colstats_fqn,
        blobs_fqn,
        blooms_fqn,
        text_lengths_fqn,
        valbitmap_fqn,
        meta_ddl,
        colstats_ddl,
        blobs_ddl,
        blooms_ddl,
        text_lengths_ddl,
        valbitmap_ddl,
        tombstones_ddl,
    }
}

/// Compress a partition using cursor-based streaming.
/// Reads native PG datums directly — no text round-trip for numeric/timestamp types.
/// Handles both segment_by and non-segment_by partitions (boundary detection is
/// guarded by `if !seg_col_indices.is_empty()` and naturally skipped when empty).
/// Returns (compressed_size, row_count). ndistinct is tracked per-segment via HLL
/// and stored in the meta table. Blobs are buffered and inserted column-major
/// into the blobs table after all segments are processed.
/// Returns (total_compressed_size, total_rows, partition_hll_per_nonseg_col,
/// finalized_valbitmap_value_map). The valbitmap map shape is
/// `{column_name: [val0, val1, ...]}` where the array index is the bit
/// position in each segment's bitmap; absent columns means "no bitmap"
/// (e.g. > 32 distinct values across the partition or non-text type).
#[allow(clippy::type_complexity)]
fn compress_partition_streaming(
    client: &mut SpiClient,
    part_fqn: &str,
    ddl: &CompanionDdl,
    columns: &[ColumnMeta],
    order_by: &[String],
    segment_by: &[String],
    segment_size: usize,
) -> (
    i64,
    i64,
    Vec<CardinalityEstimator<u64>>,
    std::collections::HashMap<String, Vec<String>>,
    ColumnValcounts,
    TopVals,
) {
    let batch_size = segment_size;

    // Classify columns for native datum extraction
    let kinds: Vec<ColumnKind> = columns
        .iter()
        .map(|c| classify_column(&c.data_type, c.is_segment_by))
        .collect();

    // Build SELECT list: segment_by and text-classified cols cast to ::text,
    // others as native types. The ::text cast is needed for CHAR/VARCHAR
    // columns which have different OIDs than text. jsonb must stay native:
    // `append_row_to_columns` reads it as a raw binary varlena via `JsonbRaw`
    // (no jsonb_out/jsonb_in round-trip, which would be lossy — see #27).
    let select_cols = columns
        .iter()
        .zip(kinds.iter())
        .map(|(c, kind)| {
            if c.is_segment_by || matches!(kind, ColumnKind::Text) {
                format!("\"{}\"::text", c.name)
            } else {
                format!("\"{}\"", c.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    // Build ORDER BY: only needed when segment_by is non-empty (for boundary detection).
    // When segment_by is empty, we skip the SQL ORDER BY to avoid a full-partition sort
    // and instead sort each segment in Rust before flushing.
    let order_clause = if !segment_by.is_empty() {
        let mut order_parts = Vec::new();
        for s in segment_by {
            order_parts.push(format!("\"{}\"", s));
        }
        for o in order_by {
            order_parts.push(format!("\"{}\"", o));
        }
        format!(" ORDER BY {}", order_parts.join(", "))
    } else {
        String::new()
    };

    // Resolve order_by column indices for Rust-side sorting (used when no SQL ORDER BY)
    let order_col_indices: Vec<usize> = order_by
        .iter()
        .filter_map(|name| columns.iter().position(|c| c.name == *name))
        .collect();

    // Segment_by column indices (for boundary detection)
    let seg_col_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_segment_by)
        .map(|(i, _)| i)
        .collect();

    // DECLARE CURSOR
    let cursor_sql = format!(
        "DECLARE comp_cursor CURSOR FOR SELECT {} FROM {}{}",
        select_cols, part_fqn, order_clause
    );
    client
        .update(&cursor_sql, None, &[])
        .expect("failed to declare cursor");

    let fetch_sql = format!("FETCH {} FROM comp_cursor", batch_size);

    let mut typed_cols = init_typed_columns(columns, &kinds);
    let mut current_seg_values: Vec<Option<String>> = Vec::new();
    let mut rows_in_segment: usize = 0;
    let mut total_compressed_size: i64 = 0;
    let mut total_rows: i64 = 0;
    let mut tables_created = false;
    let mut next_segment_id: i32 = 1;
    let mut blob_buffer: Vec<(u16, i32, Vec<u8>)> = Vec::new(); // (col_idx, segment_id, blob)
    let mut bloom_buffer: Vec<(u16, i32, u8, Vec<u8>)> = Vec::new(); // (col_idx, segment_id, num_hashes, bloom_bytes)
    let mut colstats_buffer: Vec<ColstatsRow> = Vec::new();
    let mut text_length_buffer: Vec<(u16, i32, Vec<u8>)> = Vec::new(); // (col_idx, segment_id, length_blob)
    // (col_idx, segment_id, sorted distinct values). Encoded into per-segment
    // bitmaps after the streaming loop, once partition-level value lists are
    // finalized.
    let mut valbitmap_value_buffer: Vec<(u16, i32, SegValueCounts)> = Vec::new();

    // Partition-level HLL sketches, one per non-segment-by column (matches
    // the order `compute_segment_ndistinct` returns). Each per-segment HLL
    // gets merged in below; the final merged estimates feed the
    // `pg_statistic.stadistinct` write.
    let num_nonseg_cols = columns.iter().filter(|c| !c.is_segment_by).count();
    let mut partition_hll: Vec<CardinalityEstimator<u64>> = (0..num_nonseg_cols)
        .map(|_| CardinalityEstimator::<u64>::new())
        .collect();
    // Partition-level heavy-hitter summary per non-seg col (text only) for the
    // partial MCV; fed at each segment flush, like `partition_hll`.
    let mut partition_topvals: TopVals = (0..num_nonseg_cols)
        .map(|_| std::collections::HashMap::new())
        .collect();

    loop {
        let result = client
            .select(&fetch_sql, None, &[])
            .expect("failed to fetch from cursor");
        let fetched = result.len();
        if fetched == 0 {
            break;
        }
        // Save the tuptable pointer so we can free it after consuming all rows.
        // pgrx doesn't free SPI tuple tables until SPI_finish(), which causes
        // unbounded memory growth when fetching millions of rows via cursor.
        let tuptable_to_free = unsafe { pg_sys::SPI_tuptable };

        for row in result {
            // Check segment_by boundary
            if !seg_col_indices.is_empty() {
                let row_seg_values: Vec<Option<String>> = seg_col_indices
                    .iter()
                    .map(|&i| {
                        row.get_datum_by_ordinal(i + 1)
                            .unwrap()
                            .value::<String>()
                            .unwrap()
                    })
                    .collect();

                if current_seg_values.is_empty() {
                    current_seg_values = row_seg_values;
                } else if row_seg_values != current_seg_values {
                    // Segment boundary — flush accumulated data
                    if rows_in_segment > 0 {
                        if !tables_created {
                            client
                                .update(&ddl.meta_ddl, None, &[])
                                .expect("failed to create meta table");
                            client
                                .update(&ddl.colstats_ddl, None, &[])
                                .expect("failed to create colstats table");
                            tables_created = true;
                        }
                        total_compressed_size += flush_with_splitting(
                            client,
                            &ddl.meta_fqn,
                            &ddl.colstats_fqn,
                            columns,
                            &typed_cols,
                            &current_seg_values,
                            rows_in_segment,
                            segment_size,
                            &mut next_segment_id,
                            &mut blob_buffer,
                            &mut bloom_buffer,
                            &mut colstats_buffer,
                            &mut text_length_buffer,
                            &mut valbitmap_value_buffer,
                            &mut partition_hll,
                            &mut partition_topvals,
                        );
                        typed_cols = init_typed_columns(columns, &kinds);
                        rows_in_segment = 0;
                    }
                    current_seg_values = row_seg_values;
                }
            }

            append_row_to_columns(&row, columns, &kinds, &mut typed_cols);
            rows_in_segment += 1;
            total_rows += 1;

            // Check segment_size limit
            if rows_in_segment >= segment_size {
                if !tables_created {
                    client
                        .update(&ddl.meta_ddl, None, &[])
                        .expect("failed to create meta table");
                    client
                        .update(&ddl.colstats_ddl, None, &[])
                        .expect("failed to create colstats table");
                    tables_created = true;
                }
                // Sort in Rust when no SQL ORDER BY (non-segment_by path)
                if seg_col_indices.is_empty() {
                    sort_typed_columns(&mut typed_cols, &order_col_indices, rows_in_segment);
                }
                let seg_id = next_segment_id;
                next_segment_id += 1;
                let (ndistinct, sketches) = compute_segment_ndistinct(&typed_cols, columns);
                for (dst, src) in partition_hll.iter_mut().zip(sketches.iter()) {
                    dst.merge(src);
                }
                merge_segment_topvals(&typed_cols, columns, &mut partition_topvals);
                let (size, blobs, bloom_entries, cs_rows, length_blobs, vb_values) =
                    flush_segment_metadata(
                        client,
                        &ddl.meta_fqn,
                        &ddl.colstats_fqn,
                        columns,
                        &typed_cols,
                        &current_seg_values,
                        &ndistinct,
                        rows_in_segment as u32,
                        seg_id,
                    );
                total_compressed_size += size;
                for (col_idx, blob) in blobs {
                    blob_buffer.push((col_idx, seg_id, blob));
                }
                for (col_idx, num_hashes, bytes) in bloom_entries {
                    bloom_buffer.push((col_idx, seg_id, num_hashes, bytes));
                }
                for (col_idx, blob) in length_blobs {
                    text_length_buffer.push((col_idx, seg_id, blob));
                }
                for (col_idx, vals) in vb_values {
                    valbitmap_value_buffer.push((col_idx, seg_id, vals));
                }
                colstats_buffer.extend(cs_rows);
                typed_cols = init_typed_columns(columns, &kinds);
                rows_in_segment = 0;
            }
        }

        // Free the SPI tuple table from this batch to prevent unbounded memory growth.
        // Safe because we've fully consumed all rows and extracted values into owned Rust types.
        if !tuptable_to_free.is_null() {
            unsafe { pg_sys::SPI_freetuptable(tuptable_to_free) };
        }

        if fetched < batch_size {
            break;
        }
    }

    // Flush remaining
    if rows_in_segment > 0 {
        if !tables_created {
            client
                .update(&ddl.meta_ddl, None, &[])
                .expect("failed to create meta table");
            client
                .update(&ddl.colstats_ddl, None, &[])
                .expect("failed to create colstats table");
        }
        if seg_col_indices.is_empty() {
            sort_typed_columns(&mut typed_cols, &order_col_indices, rows_in_segment);
        }
        total_compressed_size += flush_with_splitting(
            client,
            &ddl.meta_fqn,
            &ddl.colstats_fqn,
            columns,
            &typed_cols,
            &current_seg_values,
            rows_in_segment,
            segment_size,
            &mut next_segment_id,
            &mut blob_buffer,
            &mut bloom_buffer,
            &mut colstats_buffer,
            &mut text_length_buffer,
            &mut valbitmap_value_buffer,
            &mut partition_hll,
            &mut partition_topvals,
        );
    }

    client
        .update("CLOSE comp_cursor", None, &[])
        .expect("failed to close cursor");

    // Flush colstats column-major: sort by (col_idx, segment_id) so heap pages
    // are naturally clustered for index scans by _col_idx.
    if !colstats_buffer.is_empty() {
        colstats_buffer.sort_by_key(|r| (r.col_idx, r.segment_id));

        // Batch insert for efficiency
        let batch_size = 100;
        for chunk in colstats_buffer.chunks(batch_size) {
            let values: Vec<String> = chunk
                .iter()
                .map(|r| {
                    let min_str = r.min_val.map_or("NULL".to_string(), |v| v.to_string());
                    let max_str = r.max_val.map_or("NULL".to_string(), |v| v.to_string());
                    let sum_str = r.sum_val.as_deref().unwrap_or("NULL");
                    format!(
                        "({}, {}, {}, {}, {}, {}, {}, {})",
                        r.col_idx,
                        r.segment_id,
                        min_str,
                        max_str,
                        sum_str,
                        r.nonnull_count,
                        r.nonzero_count,
                        r.ndistinct
                    )
                })
                .collect();
            let sql = format!(
                "INSERT INTO {} (_col_idx, _segment_id, _min, _max, _sum, _nonnull_count, _nonzero_count, _ndistinct) VALUES {}",
                ddl.colstats_fqn,
                values.join(", ")
            );
            client
                .update(&sql, None, &[])
                .expect("failed to insert colstats batch");
        }
    }

    // Flush blobs column-major into the blobs table
    if !blob_buffer.is_empty() {
        client
            .update(&ddl.blobs_ddl, None, &[])
            .expect("failed to create blobs table");

        // Sort by (col_idx, segment_id) for column-major insertion order
        blob_buffer.sort_by_key(|&(col_idx, seg_id, _)| (col_idx, seg_id));

        for (col_idx, seg_id, blob) in blob_buffer {
            use pgrx::datum::DatumWithOid;
            let insert_sql = format!(
                "INSERT INTO {} (_col_idx, _segment_id, _data) VALUES ($1, $2, $3)",
                ddl.blobs_fqn
            );
            let args: Vec<DatumWithOid> = vec![
                (col_idx as i16).into(),
                seg_id.into(),
                DatumWithOid::from(blob),
            ];
            client
                .update(&insert_sql, None, &args)
                .expect("failed to insert blob");
        }

        // Flush bloom filters into separate blooms table
        if !bloom_buffer.is_empty() {
            client
                .update(&ddl.blooms_ddl, None, &[])
                .expect("failed to create blooms table");

            // Sort by (col_idx, segment_id) for column-major insertion order
            bloom_buffer.sort_by_key(|&(col_idx, seg_id, _, _)| (col_idx, seg_id));

            for (col_idx, seg_id, num_hashes, bloom_bytes) in bloom_buffer {
                use pgrx::datum::DatumWithOid;
                let insert_sql = format!(
                    "INSERT INTO {} (_col_idx, _segment_id, _num_hashes, _data) VALUES ($1, $2, $3, $4)",
                    ddl.blooms_fqn
                );
                let args: Vec<DatumWithOid> = vec![
                    (col_idx as i16).into(),
                    seg_id.into(),
                    (num_hashes as i16).into(),
                    DatumWithOid::from(bloom_bytes),
                ];
                client
                    .update(&insert_sql, None, &args)
                    .expect("failed to insert bloom data");
            }
            client
                .update(&format!("ANALYZE {}", ddl.blooms_fqn), None, &[])
                .expect("failed to analyze blooms table");
        }

        // Flush text-length sidecars into the text_lengths table
        if !text_length_buffer.is_empty() {
            client
                .update(&ddl.text_lengths_ddl, None, &[])
                .expect("failed to create text_lengths table");

            // Sort by (col_idx, segment_id) for column-major insertion order
            text_length_buffer.sort_by_key(|&(col_idx, seg_id, _)| (col_idx, seg_id));

            for (col_idx, seg_id, blob) in text_length_buffer {
                use pgrx::datum::DatumWithOid;
                let insert_sql = format!(
                    "INSERT INTO {} (_col_idx, _segment_id, _data) VALUES ($1, $2, $3)",
                    ddl.text_lengths_fqn
                );
                let args: Vec<DatumWithOid> = vec![
                    (col_idx as i16).into(),
                    seg_id.into(),
                    DatumWithOid::from(blob),
                ];
                client
                    .update(&insert_sql, None, &args)
                    .expect("failed to insert text length sidecar");
            }
            client
                .update(&format!("ANALYZE {}", ddl.text_lengths_fqn), None, &[])
                .expect("failed to analyze text_lengths table");
        }

        // Add a btree index on `(_col_idx, _min, _max)` for point-lookup
        // pruning. Lets `WHERE col = N` queries skip directly to the
        // segments whose [_min,_max] range covers N — start at the smallest
        // `_min`, iterate while `_min <= N`, post-filter `_max >= N`. Mirrors
        // what TimescaleDB does on its compressed chunks (their index is
        // similar but explicit min/max columns vs our normalized colstats
        // table). PG auto-names the index (truncates to 63 bytes if needed).
        client
            .update(
                &format!(
                    "CREATE INDEX ON {} (_col_idx, _min, _max)",
                    ddl.colstats_fqn
                ),
                None,
                &[],
            )
            .expect("failed to create colstats minmax index");

        // ANALYZE meta, colstats, and blobs tables for planner statistics
        client
            .update(&format!("ANALYZE {}", ddl.meta_fqn), None, &[])
            .expect("failed to analyze meta table");
        client
            .update(&format!("ANALYZE {}", ddl.colstats_fqn), None, &[])
            .expect("failed to analyze colstats table");
        client
            .update(&format!("ANALYZE {}", ddl.blobs_fqn), None, &[])
            .expect("failed to analyze blobs table");
    }

    // Finalize per-segment value bitmaps. For each col_idx, take the union
    // of per-segment value sets — if it's still ≤ VALBITMAP_MAX_DISTINCT we
    // keep it; otherwise we drop the column from valbitmap entirely. Then
    // encode each segment's bitmap against the finalized partition map and
    // bulk-insert into the valbitmap table. The partition map itself is
    // returned to the caller for catalog persistence.
    let (column_valmap, column_valcounts) =
        finalize_and_insert_valbitmaps(client, ddl, columns, valbitmap_value_buffer);

    (
        total_compressed_size,
        total_rows,
        partition_hll,
        column_valmap,
        column_valcounts,
        partition_topvals,
    )
}

/// Build partition-level value→bit_idx maps from per-segment value sets,
/// encode each segment's bitmap, bulk-insert into the valbitmap table.
/// Returns the partition-level value map (column name → sorted distinct values,
/// for `column_valmap`) plus the summed per-value occurrence counts (column
/// name → (value, count) list, for `column_valcounts` → real MCV frequencies).
fn finalize_and_insert_valbitmaps(
    client: &mut SpiClient,
    ddl: &CompanionDdl,
    columns: &[ColumnMeta],
    value_buffer: Vec<(u16, i32, SegValueCounts)>,
) -> (
    std::collections::HashMap<String, Vec<String>>,
    ColumnValcounts,
) {
    use std::collections::{BTreeMap, HashMap};

    if value_buffer.is_empty() {
        return (HashMap::new(), HashMap::new());
    }

    // Aggregate per-col_idx: distinct value set (for the bitmap/valmap) and the
    // summed occurrence counts (for valcounts). Stop accumulating into a column
    // as soon as it crosses VALBITMAP_MAX_DISTINCT (we'll drop it anyway).
    let mut count_by_col: HashMap<u16, BTreeMap<String, i64>> = HashMap::new();
    let mut overflow_cols: std::collections::HashSet<u16> = std::collections::HashSet::new();
    for (col_idx, _seg_id, vals) in &value_buffer {
        if overflow_cols.contains(col_idx) {
            continue;
        }
        let entry = count_by_col.entry(*col_idx).or_default();
        for (v, c) in vals {
            if entry.len() >= VALBITMAP_MAX_DISTINCT && !entry.contains_key(v) {
                overflow_cols.insert(*col_idx);
                count_by_col.remove(col_idx);
                break;
            }
            *entry.entry(v.clone()).or_insert(0) += *c as i64;
        }
    }

    if count_by_col.is_empty() {
        return (HashMap::new(), HashMap::new());
    }

    // Finalize per-column sorted value list + value→bit_idx index. The
    // BTreeMap keys are already sorted, matching the prior sorted-set order.
    let mut finalized: HashMap<u16, (Vec<String>, HashMap<String, u8>)> = HashMap::new();
    for (col_idx, counts) in &count_by_col {
        let sorted: Vec<String> = counts.keys().cloned().collect();
        let mut idx: HashMap<String, u8> = HashMap::new();
        for (i, v) in sorted.iter().enumerate() {
            idx.insert(v.clone(), i as u8);
        }
        finalized.insert(*col_idx, (sorted, idx));
    }

    // Map non-segment-by col_idx → user column name for the catalog payload.
    let col_idx_to_name: HashMap<u16, String> = {
        let mut m = HashMap::new();
        let mut idx: u16 = 0;
        for col in columns {
            if col.is_segment_by {
                continue;
            }
            m.insert(idx, col.name.clone());
            idx += 1;
        }
        m
    };

    // Encode + bulk-insert per-segment bitmaps. n_bytes = ceil(ndistinct/8).
    client
        .update(&ddl.valbitmap_ddl, None, &[])
        .expect("failed to create valbitmap table");

    let mut entries: Vec<(u16, i32, Vec<u8>)> = Vec::with_capacity(value_buffer.len());
    for (col_idx, seg_id, vals) in value_buffer {
        let Some((_, idx_map)) = finalized.get(&col_idx) else {
            // Column overflowed at partition level — skip.
            continue;
        };
        let n_bits = idx_map.len();
        let n_bytes = n_bits.div_ceil(8);
        let mut bits: Vec<u8> = vec![0; n_bytes];
        for (v, _c) in &vals {
            if let Some(&bit_idx) = idx_map.get(v) {
                bits[(bit_idx / 8) as usize] |= 1u8 << (bit_idx % 8);
            }
        }
        entries.push((col_idx, seg_id, bits));
    }

    // Sort by (col_idx, seg_id) for column-major insertion order.
    entries.sort_by_key(|&(col_idx, seg_id, _)| (col_idx, seg_id));
    for (col_idx, seg_id, bits) in entries {
        use pgrx::datum::DatumWithOid;
        let insert_sql = format!(
            "INSERT INTO {} (_col_idx, _segment_id, _bits) VALUES ($1, $2, $3)",
            ddl.valbitmap_fqn
        );
        let args: Vec<DatumWithOid> = vec![
            (col_idx as i16).into(),
            seg_id.into(),
            DatumWithOid::from(bits),
        ];
        client
            .update(&insert_sql, None, &args)
            .expect("failed to insert valbitmap row");
    }
    client
        .update(&format!("ANALYZE {}", ddl.valbitmap_fqn), None, &[])
        .expect("failed to analyze valbitmap table");

    // Build the catalog payloads keyed by user column name: the sorted value
    // list (valmap) and the summed per-value counts (valcounts).
    let mut valmap: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();
    for (col_idx, (vals, _)) in finalized {
        if let Some(name) = col_idx_to_name.get(&col_idx) {
            valmap.insert(name.clone(), vals);
        }
    }
    let mut valcounts: ColumnValcounts = std::collections::HashMap::new();
    for (col_idx, counts) in count_by_col {
        if let Some(name) = col_idx_to_name.get(&col_idx) {
            valcounts.insert(name.clone(), counts.into_iter().collect());
        }
    }
    (valmap, valcounts)
}

/// Compress a typed column directly, bypassing string parsing.
pub(crate) fn compress_typed_column(data: &TypedColumn, data_type: &str) -> Vec<u8> {
    match data {
        TypedColumn::Int16(values) => {
            let (non_null, null_bitmap) = compression::extract_nulls(values);
            let ints: Vec<i32> = non_null.iter().map(|&v| v as i32).collect();
            let (type_tag, encoded) = compression::bitpacked::best_encoding_i32(&ints);
            CompressedColumn {
                type_tag,
                row_count: values.len() as u32,
                null_bitmap,
                data: encoded,
            }
            .to_bytes()
        }
        TypedColumn::Int32(values) => {
            let (non_null, null_bitmap) = compression::extract_nulls(values);
            let (type_tag, encoded) = compression::bitpacked::best_encoding_i32(&non_null);
            CompressedColumn {
                type_tag,
                row_count: values.len() as u32,
                null_bitmap,
                data: encoded,
            }
            .to_bytes()
        }
        TypedColumn::Int64(values) => {
            let dt = data_type.to_lowercase();
            let (non_null, null_bitmap) = compression::extract_nulls(values);
            if dt.contains("timestamp") || dt == "date" {
                // Timestamp/date: use Gorilla timestamp encoding
                let data = compression::gorilla::encode_timestamps(&non_null);
                CompressedColumn {
                    type_tag: CompressionType::Gorilla,
                    row_count: values.len() as u32,
                    null_bitmap,
                    data,
                }
                .to_bytes()
            } else {
                // Integer: try Constant, FOR, DeltaVarint — pick smallest
                let (type_tag, encoded) = compression::bitpacked::best_encoding_i64(&non_null);
                CompressedColumn {
                    type_tag,
                    row_count: values.len() as u32,
                    null_bitmap,
                    data: encoded,
                }
                .to_bytes()
            }
        }
        TypedColumn::Float64(values) => {
            let (non_null, null_bitmap) = compression::extract_nulls(values);
            let encoded = compression::gorilla::encode_floats(&non_null);
            CompressedColumn {
                type_tag: CompressionType::Gorilla,
                row_count: values.len() as u32,
                null_bitmap,
                data: encoded,
            }
            .to_bytes()
        }
        TypedColumn::Float32(values) => {
            let (non_null, null_bitmap) = compression::extract_nulls(values);
            let encoded = compression::gorilla::encode_floats_f32(&non_null);
            CompressedColumn {
                type_tag: CompressionType::Gorilla,
                row_count: values.len() as u32,
                null_bitmap,
                data: encoded,
            }
            .to_bytes()
        }
        TypedColumn::Bool(values) => {
            let (non_null, null_bitmap) = compression::extract_nulls(values);
            let encoded = compression::boolean::encode(&non_null);
            CompressedColumn {
                type_tag: CompressionType::BooleanBitmap,
                row_count: values.len() as u32,
                null_bitmap,
                data: encoded,
            }
            .to_bytes()
        }
        TypedColumn::Text(values) => {
            let dt = data_type.to_lowercase();
            if (dt.starts_with("numeric") || dt.starts_with("decimal"))
                && !crate::FORCE_TEXT_FALLBACK.get()
            {
                // Try the scaled-i64 numeric codec; per-blob text fallback for
                // NaN/Infinity, mixed dscale, or mantissas beyond i64.
                if let Some(blob) = compress_numeric_scaled(values) {
                    return blob;
                }
            }
            // Delegate to existing string-based compression
            compress_column_values(values, data_type, "")
        }
        TypedColumn::Bytes(values) => {
            let dt = data_type.to_lowercase();
            if dt == "jsonb" {
                // jsonb varlena payloads: treat as opaque byte blobs, reuse
                // the same variable-length compression pipeline as text. Keeps
                // the legacy text-family tags for on-disk compatibility —
                // jsonb readers disambiguate by type oid, not by tag.
                compress_byte_values(values)
            } else {
                // uuid/bytea/inet binary payloads: same pipeline, but with
                // the Binary* tags so readers know these blobs hold raw bytes
                // and not the legacy `::text` renderings.
                compress_binary_values(values)
            }
        }
    }
}

/// Encode f64 to i64 in a way that preserves numeric order under signed i64
/// comparison. This is used by colstats min/max pruning.
pub(crate) fn encode_f64_to_i64(v: f64) -> i64 {
    const SIGN: u64 = 1u64 << 63;
    let bits = v.to_bits();
    let unsigned_key = if bits & SIGN != 0 { !bits } else { bits ^ SIGN };
    (unsigned_key ^ SIGN) as i64
}

/// Decode order-preserving i64 back to f64.
pub(crate) fn decode_i64_to_f64(enc: i64) -> f64 {
    const SIGN: u64 = 1u64 << 63;
    let unsigned_key = (enc as u64) ^ SIGN;
    let bits = if unsigned_key & SIGN != 0 {
        unsigned_key ^ SIGN
    } else {
        !unsigned_key
    };
    f64::from_bits(bits)
}

/// Encode f32 to i64 in a way that preserves numeric order under signed i64
/// comparison (via 32-bit transform, then sign-extend).
pub(crate) fn encode_f32_to_i64(v: f32) -> i64 {
    const SIGN: u32 = 1u32 << 31;
    let bits = v.to_bits();
    let unsigned_key = if bits & SIGN != 0 { !bits } else { bits ^ SIGN };
    ((unsigned_key ^ SIGN) as i32) as i64
}

/// Decode order-preserving i64 back to f32.
pub(crate) fn decode_i64_to_f32(enc: i64) -> f32 {
    const SIGN: u32 = 1u32 << 31;
    let unsigned_key = ((enc as i32) as u32) ^ SIGN;
    let bits = if unsigned_key & SIGN != 0 {
        unsigned_key ^ SIGN
    } else {
        !unsigned_key
    };
    f32::from_bits(bits)
}

/// Reduce `values` to the `(min, max)` of `encode(v)` over non-null entries,
/// producing the order-preserving i64 pair stored in the colstats table.
fn minmax_encoded_via<T: Copy, F: Fn(T) -> i64>(
    values: &[Option<T>],
    encode: F,
) -> (Option<i64>, Option<i64>) {
    let mut min_v: Option<i64> = None;
    let mut max_v: Option<i64> = None;
    for v in values.iter().flatten() {
        let e = encode(*v);
        min_v = Some(min_v.map_or(e, |cur| cur.min(e)));
        max_v = Some(max_v.map_or(e, |cur| cur.max(e)));
    }
    (min_v, max_v)
}

/// Compute min/max encoded as order-preserving i64, for use in normalized colstats table.
/// Returns None for types without minmax support.
pub(crate) fn compute_minmax_encoded_i64(
    data: &TypedColumn,
    data_type: &str,
) -> (Option<i64>, Option<i64>) {
    if !supports_minmax(data_type) {
        return (None, None);
    }
    match data {
        TypedColumn::Int16(values) => minmax_encoded_via(values, |v| v as i64),
        TypedColumn::Int32(values) => minmax_encoded_via(values, |v| v as i64),
        // int64, timestamp, timestamptz, date — identity (already i64).
        TypedColumn::Int64(values) => minmax_encoded_via(values, |v| v),
        TypedColumn::Float64(values) => minmax_encoded_via(values, encode_f64_to_i64),
        TypedColumn::Float32(values) => minmax_encoded_via(values, encode_f32_to_i64),
        _ => (None, None), // Text, Bool — no minmax
    }
}

/// Reduce a `Vec<Option<T: Ord>>` to its `(min, max)` over non-null entries.
fn minmax_ord<T: Copy + Ord>(values: &[Option<T>]) -> (Option<T>, Option<T>) {
    let mut min_v: Option<T> = None;
    let mut max_v: Option<T> = None;
    for v in values.iter().flatten() {
        min_v = Some(min_v.map_or(*v, |cur| cur.min(*v)));
        max_v = Some(max_v.map_or(*v, |cur| cur.max(*v)));
    }
    (min_v, max_v)
}

/// Float counterpart of `minmax_ord`. Uses `<`/`>` comparisons directly so
/// NaN tracks like the prior implementation (first non-NaN wins; subsequent
/// NaN comparisons are false and never update the running min/max).
fn minmax_float<T: Copy + PartialOrd>(values: &[Option<T>]) -> (Option<T>, Option<T>) {
    let mut min_v: Option<T> = None;
    let mut max_v: Option<T> = None;
    for v in values.iter().flatten() {
        min_v = Some(min_v.map_or(*v, |cur| if *v < cur { *v } else { cur }));
        max_v = Some(max_v.map_or(*v, |cur| if *v > cur { *v } else { cur }));
    }
    (min_v, max_v)
}

/// Compute min/max for typed columns, returning string representations for SQL INSERT.
pub(crate) fn compute_typed_minmax(
    data: &TypedColumn,
    data_type: &str,
) -> (Option<String>, Option<String>) {
    match data {
        TypedColumn::Int16(values) => {
            let (min_v, max_v) = minmax_ord(values);
            (min_v.map(|v| v.to_string()), max_v.map(|v| v.to_string()))
        }
        TypedColumn::Int32(values) => {
            let (min_v, max_v) = minmax_ord(values);
            (min_v.map(|v| v.to_string()), max_v.map(|v| v.to_string()))
        }
        TypedColumn::Int64(values) => {
            let (min_v, max_v) = minmax_ord(values);
            let dt = data_type.to_lowercase();
            if dt.contains("timestamp") {
                (
                    min_v.map(usec_to_timestamp_string),
                    max_v.map(usec_to_timestamp_string),
                )
            } else if dt == "date" {
                (
                    min_v.map(crate::timeparse::usec_to_date_string),
                    max_v.map(crate::timeparse::usec_to_date_string),
                )
            } else {
                (min_v.map(|v| v.to_string()), max_v.map(|v| v.to_string()))
            }
        }
        TypedColumn::Float64(values) => {
            let (min_v, max_v) = minmax_float(values);
            (min_v.map(|v| v.to_string()), max_v.map(|v| v.to_string()))
        }
        TypedColumn::Float32(values) => {
            let (min_v, max_v) = minmax_float(values);
            (min_v.map(|v| v.to_string()), max_v.map(|v| v.to_string()))
        }
        TypedColumn::Text(values) => compute_column_minmax(values, data_type),
        TypedColumn::Bytes(_) => (None, None), // jsonb has no meaningful minmax
        TypedColumn::Bool(_) => (None, None),  // booleans don't support minmax
    }
}

/// Compress a column's values based on the PostgreSQL data type.
/// Only used for Text columns now — numeric/timestamp types go through compress_typed_column.
fn compress_column_values(values: &[Option<String>], _data_type: &str, _col_name: &str) -> Vec<u8> {
    // Only used for Text columns now — numeric/timestamp types go through compress_typed_column
    let (non_null, null_bitmap) = compression::extract_nulls(values);
    let refs: Vec<&str> = non_null.iter().map(|s| s.as_str()).collect();

    if compression::dictionary::should_use_dictionary(&refs) {
        let dict_encoded = compression::dictionary::encode(&refs);
        let lz4_encoded = compression::dictionary::encode_lz4(&refs);
        let (tag, encoded) = if lz4_encoded.len() < dict_encoded.len() {
            (CompressionType::DictionaryLz4, lz4_encoded)
        } else {
            (CompressionType::Dictionary, dict_encoded)
        };
        CompressedColumn {
            type_tag: tag,
            row_count: values.len() as u32,
            null_bitmap,
            data: encoded,
        }
        .to_bytes()
    } else {
        let encoded = compression::lz4::encode_blocked(&refs, compression::lz4::DEFAULT_BLOCK_SIZE);
        CompressedColumn {
            type_tag: CompressionType::Lz4Blocked,
            row_count: values.len() as u32,
            null_bitmap,
            data: encoded,
        }
        .to_bytes()
    }
}

/// Compress opaque byte blobs (used for jsonb column payloads). Mirrors the
/// text pipeline: try dictionary encoding for low-cardinality data, else
/// Lz4Blocked. `&[u8]` fits transparently into the existing string-oriented
/// codecs via `std::str::from_utf8_unchecked` — the codecs only ever treat
/// their input as byte slices (length-prefixed blocks / dictionary indexing),
/// so passing non-UTF-8 jsonb varlena bytes is safe as long as we don't try
/// to iterate chars.
fn compress_byte_values(values: &[Option<Vec<u8>>]) -> Vec<u8> {
    // Convert Option<Vec<u8>> → Option<String> via an unsafe wrapper so we can
    // reuse the existing codecs. The String is never read as a valid UTF-8
    // string (compressors only inspect bytes / lengths); it is immediately
    // dropped after compression.
    let as_strings: Vec<Option<String>> = values
        .iter()
        .map(|opt| {
            opt.as_ref()
                .map(|bytes| unsafe { String::from_utf8_unchecked(bytes.clone()) })
        })
        .collect();
    compress_column_values(&as_strings, "jsonb", "")
}

/// Compress raw binary payloads (uuid/bytea/inet) through the byte pipeline,
/// then remap the codec tag to its Binary* variant. The blob layout is
/// otherwise identical — only the tag byte differs, marking the payloads as
/// raw bytes so the read side never routes them through the UTF-8 text or
/// input-function paths.
/// Encode a segment of `numeric_out` renderings as scaled-i64 mantissas
/// (`CompressionType::NumericScaled`): data = [dscale][inner integer tag]
/// [inner encoding]. Returns `None` when the segment doesn't qualify
/// (see `compression::numeric_scaled::to_uniform_scaled`).
fn compress_numeric_scaled(values: &[Option<String>]) -> Option<Vec<u8>> {
    let (scaled, dscale) = compression::numeric_scaled::to_uniform_scaled(values)?;
    let (non_null, null_bitmap) = compression::extract_nulls(&scaled);
    let (inner_tag, encoded) = compression::bitpacked::best_encoding_i64(&non_null);
    let mut data = Vec::with_capacity(2 + encoded.len());
    data.push(dscale);
    data.push(inner_tag as u8);
    data.extend_from_slice(&encoded);
    Some(
        CompressedColumn {
            type_tag: CompressionType::NumericScaled,
            row_count: values.len() as u32,
            null_bitmap,
            data,
        }
        .to_bytes(),
    )
}

fn compress_binary_values(values: &[Option<Vec<u8>>]) -> Vec<u8> {
    let mut blob = compress_byte_values(values);
    if let Some(tag) = blob.first_mut() {
        *tag = CompressionType::from_u8(*tag).to_binary_variant() as u8;
    }
    blob
}

/// Get column metadata for a table.
pub(crate) fn get_column_metadata(
    client: &SpiClient,
    schema: &str,
    table: &str,
    segment_by: &[String],
    time_column: &str,
    json_extract: Option<&serde_json::Value>,
) -> Vec<ColumnMeta> {
    let result = client
        .select(
            "SELECT column_name::text, data_type::text
             FROM information_schema.columns
             WHERE table_schema = $1 AND table_name = $2
             ORDER BY ordinal_position",
            None,
            &[schema.into(), table.into()],
        )
        .expect("failed to get columns");

    let mut columns = Vec::new();
    for row in result {
        let name: String = row
            .get_datum_by_ordinal(1)
            .unwrap()
            .value::<String>()
            .unwrap()
            .unwrap();
        let data_type: String = row
            .get_datum_by_ordinal(2)
            .unwrap()
            .value::<String>()
            .unwrap()
            .unwrap();
        let is_segment = segment_by.contains(&name);
        let is_time = name == time_column;
        columns.push(ColumnMeta {
            name,
            data_type,
            is_segment_by: is_segment,
            is_time_column: is_time,
            extracted: None,
        });
    }

    // Append synthetic columns from json_extract. The `_col_idx` slots in
    // companion tables are assigned in iteration order over non-segment-by
    // columns, so extracted columns naturally land after physical columns
    // without disturbing existing partitions.
    if let Some(jx) = json_extract {
        let mode = crate::get_json_extract_mode();
        if mode != crate::JsonExtractMode::None {
            let specs = parse_extract_specs(jx);
            for spec in specs {
                columns.push(ColumnMeta {
                    name: spec.target_name.clone(),
                    data_type: spec.target_type.clone(),
                    is_segment_by: false,
                    is_time_column: false,
                    extracted: Some(spec),
                });
            }
        }
    }

    columns
}

/// Estimate raw table size in bytes.
fn estimate_raw_size(client: &SpiClient, table_fqn: &str) -> i64 {
    client
        .select(
            &format!(
                "SELECT pg_total_relation_size('{}'::regclass)::int8",
                table_fqn
            ),
            None,
            &[],
        )
        .expect("failed to get table size")
        .first()
        .get_one::<i64>()
        .unwrap()
        .unwrap_or(0)
}

// ============================================================================
// Decompression
// ============================================================================

/// Decode one segment's column blobs and INSERT the restored rows back into
/// the partition heap. Shared by full decompression
/// (`decompress_partition_inner`) and per-segment decompose-on-write
/// (`decompose_segments_for_dml`). Reads the segment's blobs from the
/// `_blobs` companion — so it must run BEFORE the caller deletes the
/// segment's blob rows.
///
/// `non_seg_cols` is the positional `(col_idx, name, data_type)` mapping for
/// non-segment-by physical columns (synthetic json-extract columns live at
/// higher `_col_idx` values and are intentionally not restored — the
/// original JSONB column is physical and lands in the heap row).
///
/// `skip_offsets` (P2.5 tombstones) lists 0-based row offsets that are
/// logically deleted and must NOT be rematerialized — every restore path
/// (full decompress, decompose-on-write, compaction rewrite) honors it.
/// Returns the number of rows actually inserted (live rows).
#[allow(clippy::too_many_arguments)]
fn restore_segment_rows(
    client: &mut SpiClient,
    part_fqn: &str,
    blobs_fqn: &str,
    columns: &[ColumnMeta],
    non_seg_cols: &[(u16, String, String)],
    segment_id: i32,
    segment_by_vals: &[Option<String>],
    row_count: i32,
    skip_offsets: Option<&std::collections::HashSet<i32>>,
) -> i64 {
    // Read blobs for this segment from the blobs table
    let blob_query = format!(
        "SELECT _col_idx, _data FROM {} WHERE _segment_id = $1 ORDER BY _col_idx",
        blobs_fqn
    );
    let blob_rows = client
        .select(&blob_query, None, &[segment_id.into()])
        .expect("failed to read blobs");

    let mut blob_map: std::collections::HashMap<u16, Vec<u8>> = std::collections::HashMap::new();
    for brow in blob_rows {
        let ci: i16 = brow
            .get_datum_by_ordinal(1)
            .unwrap()
            .value::<i16>()
            .unwrap()
            .unwrap_or(0);
        let data: Option<Vec<u8>> = brow
            .get_datum_by_ordinal(2)
            .unwrap()
            .value::<Vec<u8>>()
            .unwrap();
        blob_map.insert(ci as u16, data.unwrap_or_default());
    }

    // Decompress all columns
    let mut decompressed_cols: Vec<(String, Vec<Option<String>>)> = Vec::new();

    // Segment-by columns: repeat the value for every row
    let mut seg_idx = 0;
    for col in columns {
        if col.is_segment_by {
            let val = &segment_by_vals[seg_idx];
            let repeated: Vec<Option<String>> = (0..row_count).map(|_| val.clone()).collect();
            decompressed_cols.push((col.name.clone(), repeated));
            seg_idx += 1;
        }
    }

    // Compressed columns: decompress from blob_map
    for (ci, name, data_type) in non_seg_cols {
        let blob = blob_map.get(ci).cloned().unwrap_or_default();
        let values = if data_type.eq_ignore_ascii_case("jsonb") {
            // jsonb blobs hold binary jsonb varlena payloads, not UTF-8 text:
            // decode byte-safe, then convert back to JSON text via jsonb_out
            // so the value can travel through the INSERT literal below.
            let mut jsonb_out = unsafe { TypeOutFn::resolve(pgrx::pg_sys::JSONBOID) };
            decompress_column_byte_values(&blob)
                .into_iter()
                .map(|opt| opt.map(|payload| unsafe { jsonb_out.render(&payload) }))
                .collect()
        } else {
            decompress_column_values(&blob, data_type)
        };
        decompressed_cols.push((name.clone(), values));
    }

    // Sort columns back to original order
    let mut ordered_cols: Vec<(String, Vec<Option<String>>)> = Vec::new();
    for col in columns {
        for dc in &decompressed_cols {
            if dc.0 == col.name {
                ordered_cols.push(dc.clone());
                break;
            }
        }
    }

    // INSERT rows back into partition
    let col_names: String = ordered_cols
        .iter()
        .map(|(name, _)| format!("\"{}\"", name))
        .collect::<Vec<_>>()
        .join(", ");

    const BATCH_SIZE: usize = 1000;
    let mut batch_start = 0;
    let mut inserted: i64 = 0;
    while batch_start < row_count as usize {
        let batch_end = (batch_start + BATCH_SIZE).min(row_count as usize);

        let mut all_row_values = Vec::with_capacity(batch_end - batch_start);
        for row_idx in batch_start..batch_end {
            if let Some(skip) = skip_offsets
                && skip.contains(&(row_idx as i32))
            {
                continue; // tombstoned — logically deleted, do not resurrect
            }
            let vals: Vec<String> = ordered_cols
                .iter()
                .enumerate()
                .map(|(col_idx, (_, values))| {
                    let col_meta = &columns[col_idx];
                    match &values[row_idx] {
                        None => "NULL".to_string(),
                        Some(v) => format_value_for_insert(v, &col_meta.data_type),
                    }
                })
                .collect();
            all_row_values.push(format!("({})", vals.join(", ")));
        }
        if !all_row_values.is_empty() {
            inserted += all_row_values.len() as i64;
            let insert_sql = format!(
                "INSERT INTO {} ({}) VALUES {}",
                part_fqn,
                col_names,
                all_row_values.join(", ")
            );
            client
                .update(&insert_sql, None, &[])
                .expect("failed to insert decompressed rows");
        }

        batch_start = batch_end;
    }

    inserted
}

/// Read tombstoned row offsets (P2.5) for the given segments — or all
/// segments when `only_ids` is `None` — as `segment_id → set of offsets`.
/// Returns an empty map when the tombstones table doesn't exist (data
/// compressed before P2.5) or holds no matching rows.
fn load_tombstone_offsets_sql(
    client: &SpiClient,
    tombstones_fqn: &str,
    only_ids: Option<&[i32]>,
) -> std::collections::HashMap<i32, std::collections::HashSet<i32>> {
    let mut map: std::collections::HashMap<i32, std::collections::HashSet<i32>> =
        std::collections::HashMap::new();
    if !relation_exists(client, tombstones_fqn) {
        return map;
    }
    let where_clause = match only_ids {
        Some([]) => return map,
        Some(ids) => format!(
            " WHERE _segment_id IN ({})",
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        None => String::new(),
    };
    let rows = client
        .select(
            &format!(
                "SELECT _segment_id, _row_offset FROM {}{}",
                tombstones_fqn, where_clause
            ),
            None,
            &[],
        )
        .expect("failed to read tombstones");
    for row in rows {
        let sid: Option<i32> = row.get(1).ok().flatten();
        let off: Option<i32> = row.get(2).ok().flatten();
        if let (Some(sid), Some(off)) = (sid, off) {
            map.entry(sid).or_default().insert(off);
        }
    }
    map
}

fn decompress_partition_impl(client: &mut SpiClient, partition: &str) -> String {
    // Bypass the DML-on-compressed check for the INSERT we are about to do
    crate::scan::set_dml_bypass(true);
    let result = decompress_partition_inner(client, partition);
    crate::scan::set_dml_bypass(false);
    result
}

fn decompress_partition_inner(client: &mut SpiClient, partition: &str) -> String {
    // 1. Look up partition
    let (schema, part_table) = crate::partition::resolve_relation(client, partition);
    let part_info = catalog::get_partition_by_name(client, &schema, &part_table)
        .expect("failed to query partition")
        .unwrap_or_else(|| {
            pgrx::error!(
                "pg_deltax: partition {}.{} not found in catalog",
                schema,
                part_table
            )
        });

    if !part_info.is_compressed {
        return format!("Partition {}.{} is not compressed", schema, part_table);
    }

    let ht = catalog::get_deltatable_by_id(client, part_info.deltatable_id)
        .expect("failed to query deltatable")
        .unwrap();

    // 2. Get column metadata (from the parent table, since partition is truncated)
    // Decompression repopulates the parent table's physical columns only —
    // the synthetic json_extract columns live solely in the companion blobs
    // and don't need to be reconstructed.
    let columns = get_column_metadata(
        client,
        &ht.schema_name,
        &ht.table_name,
        &ht.segment_by,
        &ht.time_column,
        None,
    );

    let companion_schema = "_deltax_compressed";
    let meta_fqn = format!("\"{}\".\"{}_meta\"", companion_schema, part_table);
    let colstats_fqn = format!("\"{}\".\"{}_colstats\"", companion_schema, part_table);
    let blobs_fqn = format!("\"{}\".\"{}_blobs\"", companion_schema, part_table);
    let blooms_fqn = format!("\"{}\".\"{}_blooms\"", companion_schema, part_table);
    let text_lengths_fqn = format!("\"{}\".\"{}_text_lengths\"", companion_schema, part_table);
    let valbitmap_fqn = format!("\"{}\".\"{}_valbitmap\"", companion_schema, part_table);
    let tombstones_fqn = format!("\"{}\".\"{}_tombstones\"", companion_schema, part_table);
    let part_fqn = crate::partition::fqn(&schema, &part_table);
    catalog::drop_compressed_dml_trigger(client, &schema, &part_table)
        .expect("failed to drop compressed partition DML trigger");

    // P2.5: tombstoned rows are logically deleted — they must not be
    // rematerialized by the full decompress.
    let tombstones = load_tombstone_offsets_sql(client, &tombstones_fqn, None);

    // 3. Read compressed segments from meta + blobs tables

    // Build col_idx mapping: non-segment-by columns in ordinal order
    let mut non_seg_cols: Vec<(u16, String, String)> = Vec::new(); // (col_idx, name, data_type)
    let mut col_idx: u16 = 0;
    for col in &columns {
        if !col.is_segment_by {
            non_seg_cols.push((col_idx, col.name.clone(), col.data_type.clone()));
            col_idx += 1;
        }
    }

    // Read meta table: segment_by cols, _segment_id, _row_count
    let mut meta_select_cols = vec!["_segment_id".to_string()];
    for col in &columns {
        if col.is_segment_by {
            meta_select_cols.push(format!("\"{}\"::text", col.name));
        }
    }
    meta_select_cols.push("_row_count".to_string());

    let meta_query = format!(
        "SELECT {} FROM {} ORDER BY _segment_id",
        meta_select_cols.join(", "),
        meta_fqn
    );
    let meta_rows = client
        .select(&meta_query, None, &[])
        .expect("failed to read meta table");

    // Collect all segment metadata
    struct SegMeta {
        segment_id: i32,
        segment_by_vals: Vec<Option<String>>,
        row_count: i32,
    }
    let mut seg_metas: Vec<SegMeta> = Vec::new();
    for row in meta_rows {
        let mut col_ordinal: usize = 1;
        let segment_id: i32 = row
            .get_datum_by_ordinal(col_ordinal)
            .unwrap()
            .value::<i32>()
            .unwrap()
            .unwrap_or(0);
        col_ordinal += 1;

        let mut segment_by_vals: Vec<Option<String>> = Vec::new();
        for col in &columns {
            if col.is_segment_by {
                let val: Option<String> = row
                    .get_datum_by_ordinal(col_ordinal)
                    .unwrap()
                    .value::<String>()
                    .unwrap();
                segment_by_vals.push(val);
                col_ordinal += 1;
            }
        }

        let row_count: i32 = row
            .get_datum_by_ordinal(col_ordinal)
            .unwrap()
            .value::<i32>()
            .unwrap()
            .unwrap_or(0);
        seg_metas.push(SegMeta {
            segment_id,
            segment_by_vals,
            row_count,
        });
    }

    let mut total_rows_restored = 0i64;

    for seg_meta in &seg_metas {
        if seg_meta.row_count == 0 {
            continue;
        }
        total_rows_restored += restore_segment_rows(
            client,
            &part_fqn,
            &blobs_fqn,
            &columns,
            &non_seg_cols,
            seg_meta.segment_id,
            &seg_meta.segment_by_vals,
            seg_meta.row_count,
            tombstones.get(&seg_meta.segment_id),
        );
    }

    // 4. Drop meta + colstats + blobs + blooms + text_lengths + valbitmap tables
    client
        .update(&format!("DROP TABLE IF EXISTS {}", blobs_fqn), None, &[])
        .expect("failed to drop blobs table");
    client
        .update(&format!("DROP TABLE IF EXISTS {}", blooms_fqn), None, &[])
        .expect("failed to drop blooms table");
    client
        .update(
            &format!("DROP TABLE IF EXISTS {}", text_lengths_fqn),
            None,
            &[],
        )
        .expect("failed to drop text_lengths table");
    client
        .update(
            &format!("DROP TABLE IF EXISTS {}", valbitmap_fqn),
            None,
            &[],
        )
        .expect("failed to drop valbitmap table");
    client
        .update(&format!("DROP TABLE IF EXISTS {}", colstats_fqn), None, &[])
        .expect("failed to drop colstats table");
    client
        .update(
            &format!("DROP TABLE IF EXISTS {}", tombstones_fqn),
            None,
            &[],
        )
        .expect("failed to drop tombstones table");
    client
        .update(&format!("DROP TABLE IF EXISTS {}", meta_fqn), None, &[])
        .expect("failed to drop meta table");

    // 5. Update catalog
    catalog::mark_partition_decompressed(client, part_info.id).expect("failed to update catalog");

    crate::scan::invalidate_compressed_cache();

    format!(
        "Decompressed {}.{}: {} rows restored",
        schema, part_table, total_rows_restored
    )
}

/// Decompress a column blob back to string representations.
fn decompress_column_values(blob: &[u8], data_type: &str) -> Vec<Option<String>> {
    if blob.is_empty() {
        return Vec::new();
    }

    let cc = CompressedColumn::from_bytes(blob);
    let total_count = cc.row_count as usize;
    let dt = data_type.to_lowercase();

    // jsonb blobs store PG's BINARY jsonb varlena payload (not UTF-8 text — see
    // `jsonb_text_to_binary` / `JsonbRaw`), so they must be decoded via the
    // byte-level codec entry points and converted back to canonical JSON text
    // through jsonb's output function. Routing them through the UTF-8-validating
    // text decoders below panics with "invalid UTF-8 in dictionary" /
    // "invalid UTF-8 in LZ4 data".
    if dt == "jsonb" {
        let non_null_count = count_non_null(&cc.null_bitmap, total_count);
        let mut jsonb_out = unsafe { TypeOutFn::resolve(pgrx::pg_sys::JSONBOID) };
        let strings: Vec<String> = match cc.type_tag {
            CompressionType::Dictionary => {
                compression::dictionary::decode_to_byte_slices(&cc.data, non_null_count)
                    .iter()
                    .map(|b| unsafe { jsonb_out.render(b) })
                    .collect()
            }
            CompressionType::DictionaryLz4 => {
                let normalized = compression::dictionary::normalize_lz4(&cc.data);
                compression::dictionary::decode_to_byte_slices(&normalized, non_null_count)
                    .iter()
                    .map(|b| unsafe { jsonb_out.render(b) })
                    .collect()
            }
            CompressionType::Lz4 => {
                let (buf, ranges) = compression::lz4::decode_to_ranges(&cc.data, non_null_count);
                ranges
                    .iter()
                    .map(|&(off, len)| unsafe { jsonb_out.render(&buf[off..off + len]) })
                    .collect()
            }
            CompressionType::Lz4Blocked => {
                let (buf, ranges) =
                    compression::lz4::decode_to_ranges_blocked(&cc.data, non_null_count, None);
                ranges
                    .iter()
                    .map(|&(off, len)| unsafe { jsonb_out.render(&buf[off..off + len]) })
                    .collect()
            }
            other => pgrx::error!(
                "pg_deltax: unexpected compression type {:?} for jsonb column",
                other
            ),
        };
        return compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count);
    }

    // Binary-tag blobs (uuid/bytea/inet native codecs) hold the type's raw
    // binary payload; render back to canonical text via the type's output
    // function for the re-INSERT.
    if matches!(
        cc.type_tag,
        CompressionType::BinaryDictionary
            | CompressionType::BinaryDictionaryLz4
            | CompressionType::BinaryLz4Blocked
    ) {
        let (type_oid, is_fixed_len) = match dt.as_str() {
            "uuid" => (pgrx::pg_sys::UUIDOID, true),
            "bytea" => (pgrx::pg_sys::BYTEAOID, false),
            "inet" => (pgrx::pg_sys::INETOID, false),
            "cidr" => (pgrx::pg_sys::CIDROID, false),
            other => pgrx::error!(
                "pg_deltax: unexpected binary-tag blob for column type {}",
                other
            ),
        };
        let non_null_count = count_non_null(&cc.null_bitmap, total_count);
        let mut out_fn = unsafe { TypeOutFn::resolve(type_oid) };
        let mut render = |b: &[u8]| -> String {
            unsafe {
                if is_fixed_len {
                    out_fn.render_fixed(b)
                } else {
                    out_fn.render(b)
                }
            }
        };
        let strings: Vec<String> = match cc.type_tag {
            CompressionType::BinaryDictionary => {
                compression::dictionary::decode_to_byte_slices(&cc.data, non_null_count)
                    .iter()
                    .map(|b| render(b))
                    .collect()
            }
            CompressionType::BinaryDictionaryLz4 => {
                let normalized = compression::dictionary::normalize_lz4(&cc.data);
                compression::dictionary::decode_to_byte_slices(&normalized, non_null_count)
                    .iter()
                    .map(|b| render(b))
                    .collect()
            }
            CompressionType::BinaryLz4Blocked => {
                let (buf, ranges) =
                    compression::lz4::decode_to_ranges_blocked(&cc.data, non_null_count, None);
                ranges
                    .iter()
                    .map(|&(off, len)| render(&buf[off..off + len]))
                    .collect()
            }
            _ => unreachable!(),
        };
        return compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count);
    }

    // NumericScaled blobs: decode the scaled-i64 mantissas and render the
    // exact numeric_out text back (pure Rust inverse of the write side).
    if cc.type_tag == CompressionType::NumericScaled {
        let non_null_count = count_non_null(&cc.null_bitmap, total_count);
        let dscale = cc.data[0];
        let inner_tag = CompressionType::from_u8(cc.data[1]);
        let payload = &cc.data[2..];
        let ints: Vec<i64> = match inner_tag {
            CompressionType::DeltaVarint => {
                compression::integer::decode_i64(payload, non_null_count)
            }
            CompressionType::Constant => {
                compression::bitpacked::decode_constant_i64(payload, non_null_count)
            }
            CompressionType::ForBitpacked => {
                compression::bitpacked::decode_for_i64(payload, non_null_count)
            }
            other => pgrx::error!(
                "pg_deltax: unexpected inner tag {:?} in NumericScaled blob",
                other
            ),
        };
        let strings: Vec<String> = ints
            .iter()
            .map(|&m| compression::numeric_scaled::format_numeric_scaled(m, dscale))
            .collect();
        return compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count);
    }

    // time columns: integer-tag blobs (native codec) hold microseconds since
    // midnight and must be rendered as 'HH:MM:SS[.ffffff]' for the re-INSERT.
    // Legacy text-tag blobs hold the time's text rendering already and fall
    // through to the generic dictionary/LZ4 arms below.
    if (dt == "time" || dt == "time without time zone")
        && matches!(
            cc.type_tag,
            CompressionType::DeltaVarint
                | CompressionType::Constant
                | CompressionType::ForBitpacked
        )
    {
        let non_null_count = count_non_null(&cc.null_bitmap, total_count);
        let usecs: Vec<i64> = match cc.type_tag {
            CompressionType::DeltaVarint => {
                compression::integer::decode_i64(&cc.data, non_null_count)
            }
            CompressionType::Constant => {
                compression::bitpacked::decode_constant_i64(&cc.data, non_null_count)
            }
            CompressionType::ForBitpacked => {
                compression::bitpacked::decode_for_i64(&cc.data, non_null_count)
            }
            _ => unreachable!(),
        };
        let strings: Vec<String> = usecs
            .iter()
            .map(|&u| crate::timeparse::usec_to_time_string(u))
            .collect();
        return compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count);
    }

    match cc.type_tag {
        CompressionType::Gorilla => {
            if dt.contains("timestamp") || dt == "date" {
                let timestamps = compression::gorilla::decode_timestamps(
                    &cc.data,
                    count_non_null(&cc.null_bitmap, total_count),
                );
                let strings: Vec<String> = if dt == "date" {
                    timestamps
                        .iter()
                        .map(|&usec| crate::timeparse::usec_to_date_string(usec))
                        .collect()
                } else {
                    timestamps
                        .iter()
                        .map(|&usec| usec_to_timestamp_string(usec))
                        .collect()
                };
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else if dt == "real" || dt == "float4" {
                let floats = compression::gorilla::decode_floats_f32(
                    &cc.data,
                    count_non_null(&cc.null_bitmap, total_count),
                );
                let strings: Vec<String> = floats.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else {
                let floats = compression::gorilla::decode_floats(
                    &cc.data,
                    count_non_null(&cc.null_bitmap, total_count),
                );
                let strings: Vec<String> = floats.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            }
        }
        CompressionType::DeltaVarint => {
            if dt == "smallint" || dt == "int2" {
                // Decode as i32 and downcast to i16
                let ints = compression::integer::decode_i32(
                    &cc.data,
                    count_non_null(&cc.null_bitmap, total_count),
                );
                let strings: Vec<String> = ints.iter().map(|v| (*v as i16).to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else if dt == "integer" || dt == "int4" {
                let ints = compression::integer::decode_i32(
                    &cc.data,
                    count_non_null(&cc.null_bitmap, total_count),
                );
                let strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else {
                let ints = compression::integer::decode_i64(
                    &cc.data,
                    count_non_null(&cc.null_bitmap, total_count),
                );
                let strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            }
        }
        CompressionType::Dictionary => {
            let strings = compression::dictionary::decode(
                &cc.data,
                count_non_null(&cc.null_bitmap, total_count),
            );
            compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
        }
        CompressionType::DictionaryLz4 => {
            let normalized = compression::dictionary::normalize_lz4(&cc.data);
            let strings = compression::dictionary::decode(
                &normalized,
                count_non_null(&cc.null_bitmap, total_count),
            );
            compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
        }
        CompressionType::Lz4 => {
            let strings =
                compression::lz4::decode(&cc.data, count_non_null(&cc.null_bitmap, total_count));
            compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
        }
        CompressionType::Lz4Blocked => {
            let strings = compression::lz4::decode_blocked(
                &cc.data,
                count_non_null(&cc.null_bitmap, total_count),
            );
            compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
        }
        CompressionType::BooleanBitmap => {
            let bools = compression::boolean::decode(
                &cc.data,
                count_non_null(&cc.null_bitmap, total_count),
            );
            let strings: Vec<String> = bools
                .iter()
                .map(|&b| if b { "t".to_string() } else { "f".to_string() })
                .collect();
            compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
        }
        // Handled by the early-return branches above for the types that
        // write these tags; anything else is corrupt.
        CompressionType::BinaryDictionary
        | CompressionType::BinaryDictionaryLz4
        | CompressionType::BinaryLz4Blocked
        | CompressionType::NumericScaled => pgrx::error!(
            "pg_deltax: unexpected {:?} blob for column type {}",
            cc.type_tag,
            dt
        ),
        CompressionType::Constant => {
            let non_null_count = count_non_null(&cc.null_bitmap, total_count);
            if dt == "smallint" || dt == "int2" {
                let ints = compression::bitpacked::decode_constant_i32(&cc.data, non_null_count);
                let strings: Vec<String> = ints.iter().map(|v| (*v as i16).to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else if dt == "integer" || dt == "int4" {
                let ints = compression::bitpacked::decode_constant_i32(&cc.data, non_null_count);
                let strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else {
                let ints = compression::bitpacked::decode_constant_i64(&cc.data, non_null_count);
                let strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            }
        }
        CompressionType::ForBitpacked => {
            let non_null_count = count_non_null(&cc.null_bitmap, total_count);
            if dt == "smallint" || dt == "int2" {
                let ints = compression::bitpacked::decode_for_i32(&cc.data, non_null_count);
                let strings: Vec<String> = ints.iter().map(|v| (*v as i16).to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else if dt == "integer" || dt == "int4" {
                let ints = compression::bitpacked::decode_for_i32(&cc.data, non_null_count);
                let strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            } else {
                let ints = compression::bitpacked::decode_for_i64(&cc.data, non_null_count);
                let strings: Vec<String> = ints.iter().map(|v| v.to_string()).collect();
                compression::reinsert_nulls(&strings, &cc.null_bitmap, total_count)
            }
        }
    }
}

/// Byte-level counterpart of `decompress_column_values` for jsonb columns,
/// whose stored values are binary jsonb varlena payloads (NOT UTF-8 text —
/// see `compress_byte_values`). Decodes through the byte-safe codec variants
/// so no UTF-8 validation runs on the payload bytes.
fn decompress_column_byte_values(blob: &[u8]) -> Vec<Option<Vec<u8>>> {
    if blob.is_empty() {
        return Vec::new();
    }

    let cc = CompressedColumn::from_bytes(blob);
    let total_count = cc.row_count as usize;
    let non_null_count = count_non_null(&cc.null_bitmap, total_count);

    let bytes: Vec<Vec<u8>> = match cc.type_tag {
        CompressionType::Dictionary => {
            compression::dictionary::decode_to_byte_slices(&cc.data, non_null_count)
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect()
        }
        CompressionType::DictionaryLz4 => {
            let normalized = compression::dictionary::normalize_lz4(&cc.data);
            compression::dictionary::decode_to_byte_slices(&normalized, non_null_count)
                .into_iter()
                .map(<[u8]>::to_vec)
                .collect()
        }
        CompressionType::Lz4 => {
            let (buf, ranges) = compression::lz4::decode_to_ranges(&cc.data, non_null_count);
            ranges
                .iter()
                .map(|&(off, len)| buf[off..off + len].to_vec())
                .collect()
        }
        CompressionType::Lz4Blocked => {
            let (buf, ranges) =
                compression::lz4::decode_to_ranges_blocked(&cc.data, non_null_count, None);
            ranges
                .iter()
                .map(|&(off, len)| buf[off..off + len].to_vec())
                .collect()
        }
        other => pgrx::error!(
            "pg_deltax: unexpected compression tag {:?} for a jsonb column blob",
            other
        ),
    };

    compression::reinsert_nulls(&bytes, &cc.null_bitmap, total_count)
}

/// Count non-null values given a null bitmap and total count.
fn count_non_null(null_bitmap: &[u8], total_count: usize) -> usize {
    if null_bitmap.is_empty() {
        return total_count;
    }
    let null_count: usize = (0..total_count)
        .filter(|&i| (null_bitmap[i / 8] >> (i % 8)) & 1 == 1)
        .count();
    total_count - null_count
}

fn usec_to_timestamp_string(usec: i64) -> String {
    crate::timeparse::usec_to_timestamp_string(usec)
}

fn format_value_for_insert(value: &str, data_type: &str) -> String {
    let dt = data_type.to_lowercase();
    if dt.contains("timestamp") {
        format!("'{}'::timestamptz", value.replace('\'', "''"))
    } else if dt == "date" {
        format!("'{}'::date", value.replace('\'', "''"))
    } else if dt == "boolean" || dt == "bool" {
        if value == "t" || value == "true" || value == "1" {
            "true".to_string()
        } else {
            "false".to_string()
        }
    } else if dt == "integer"
        || dt == "int4"
        || dt == "bigint"
        || dt == "int8"
        || dt == "smallint"
        || dt == "int2"
        || dt == "double precision"
        || dt == "float8"
        || dt == "real"
        || dt == "float4"
    {
        value.to_string()
    } else {
        format!("'{}'", value.replace('\'', "''"))
    }
}

/// Check if a column type supports min/max metadata.
pub(crate) fn supports_minmax(data_type: &str) -> bool {
    let dt = data_type.to_lowercase();
    dt.contains("timestamp")
        || dt == "date"
        || dt == "time"
        || dt == "time without time zone"
        || dt == "integer"
        || dt == "int4"
        || dt == "bigint"
        || dt == "int8"
        || dt == "smallint"
        || dt == "int2"
        || dt == "double precision"
        || dt == "float8"
        || dt == "real"
        || dt == "float4"
}

/// Check if a column type supports sum metadata. Numeric types get SUM(col);
/// text types get SUM(length(col)) + nonempty_count (both go through the same
/// `_sum`/`_nonnull_count`/`_nonzero_count` colstats slots — the interpretation
/// at read time is driven by column type).
pub(crate) fn supports_sum(data_type: &str) -> bool {
    let dt = data_type.to_lowercase();
    dt == "integer"
        || dt == "int4"
        || dt == "bigint"
        || dt == "int8"
        || dt == "smallint"
        || dt == "int2"
        || dt == "double precision"
        || dt == "float8"
        || dt == "real"
        || dt == "float4"
        || is_text_data_type(&dt)
}

/// Fold segment-by columns into the partition's catalog stat maps so the child
/// and parent `pg_statistic` machinery covers them. PG otherwise defaults
/// `WHERE segkey = X` to 0.005, and the segment key is the column users filter
/// and join on most. Segment-by values are stored as the partition's
/// `segment_values` (not in the blob/HLL), but the meta table holds the exact
/// `(segment value, _row_count)` per segment, so `SUM(_row_count) GROUP BY
/// segval` is the exact per-value frequency. This populates `col_ndistinct` for
/// every segment-by column; for text segment keys with no more than
/// `VALBITMAP_MAX_DISTINCT` distinct values it also populates `valmap` and
/// `valcounts`, yielding an MCV with real frequencies. Non-text keys get
/// stadistinct only, since the MCV slot is text-only (see `stats::is_text_type`).
pub(crate) fn augment_segment_by_stats(
    client: &mut SpiClient,
    meta_fqn: &str,
    columns: &[ColumnMeta],
    col_ndistinct: &mut std::collections::HashMap<String, i64>,
    valmap: &mut std::collections::HashMap<String, Vec<String>>,
    valcounts: &mut ColumnValcounts,
) {
    for col in columns {
        if !col.is_segment_by {
            continue;
        }
        let ident = col.name.replace('"', "\"\"");
        let query = format!(
            "SELECT \"{ident}\"::text AS v, SUM(_row_count)::int8 AS c \
             FROM {meta_fqn} WHERE \"{ident}\" IS NOT NULL GROUP BY 1 ORDER BY 1"
        );
        let Ok(rows) = client.select(&query, None, &[]) else {
            continue;
        };
        let mut pairs: Vec<(String, i64)> = Vec::new();
        for row in rows {
            let v: Option<String> = row.get(1).ok().flatten();
            let c: i64 = row.get(2).ok().flatten().unwrap_or(0);
            if let Some(v) = v {
                pairs.push((v, c));
            }
        }
        if pairs.is_empty() {
            continue;
        }
        // Exact distinct count → accurate equality selectivity (1/ndistinct).
        col_ndistinct.insert(col.name.clone(), pairs.len() as i64);
        // Text, low-card → also an MCV with exact frequencies + absent-value ~0.
        if is_text_data_type(&col.data_type.to_lowercase()) && pairs.len() <= VALBITMAP_MAX_DISTINCT
        {
            valmap.insert(
                col.name.clone(),
                pairs.iter().map(|(v, _)| v.clone()).collect(),
            );
            valcounts.insert(col.name.clone(), pairs);
        }
    }
}

/// Select the partial MCV for high-cardinality text columns from the
/// partition-level top-value summary. Skips columns already covered by the
/// `<=32` complete valmap (those get an exact MCV). Keeps values notably more
/// common than uniform — `freq > 1.25 / ndistinct`, PG's MCV admission
/// heuristic — and, when at least one such heavy hitter exists, the top-N by
/// count (>= 2, capped at PG's default statistics target of 100). Counts are
/// exact for the monitored values; `stats.rs` pairs them with the real HLL
/// `stadistinct` so the long tail is still estimated.
pub(crate) fn select_partial_mcv(
    topvals: &TopVals,
    nd_col_names: &[&str],
    col_ndistinct: &std::collections::HashMap<String, i64>,
    valmap: &std::collections::HashMap<String, Vec<String>>,
    row_count: i64,
) -> ColumnValcounts {
    const MAX_MCV: usize = 100;
    let mut out: ColumnValcounts = std::collections::HashMap::new();
    if row_count <= 0 {
        return out;
    }
    for (i, &name) in nd_col_names.iter().enumerate() {
        if valmap.contains_key(name) {
            continue; // complete MCV already written from the valmap
        }
        let Some(map) = topvals.get(i) else { continue };
        if map.len() < 2 {
            continue;
        }
        let ndistinct = col_ndistinct
            .get(name)
            .copied()
            .unwrap_or(map.len() as i64)
            .max(1);
        let threshold = 1.25 / ndistinct as f64;
        let n_sig = map
            .values()
            .filter(|&&c| (c as f64 / row_count as f64) > threshold)
            .count();
        if n_sig == 0 {
            continue; // near-uniform: no meaningful MCV
        }
        let mut sorted: Vec<(String, i64)> = map.iter().map(|(v, &c)| (v.clone(), c)).collect();
        // Highest count first; value as a stable tiebreak.
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        // Keep the heavy hitters; ensure at least 2 (the MCV slot needs >=2),
        // capped at the statistics target.
        let keep = n_sig.max(2).min(sorted.len()).min(MAX_MCV);
        sorted.truncate(keep);
        out.insert(name.to_string(), sorted);
    }
    out
}

/// True for PostgreSQL text-family types.
pub(crate) fn is_text_data_type(dt: &str) -> bool {
    dt == "text"
        || dt == "varchar"
        || dt.starts_with("varchar(")
        || dt == "character varying"
        || dt.starts_with("character varying(")
        || dt == "char"
        || dt.starts_with("char(")
        || dt == "character"
        || dt.starts_with("character(")
        || dt == "bpchar"
}

/// Walk an integer column, accumulating sum (as i128 to avoid overflow on
/// large segments), non-null count, and nonzero count. Returns the same
/// `(sum_str, nonnull, nonzero)` shape `compute_typed_sum` expects.
fn sum_int_column<T: Copy + Into<i128> + PartialEq + Default>(
    v: &[Option<T>],
) -> (Option<String>, i64, i64) {
    let zero = T::default();
    let mut sum: i128 = 0;
    let mut count: i64 = 0;
    let mut nonzero: i64 = 0;
    for val in v.iter().flatten() {
        sum += (*val).into();
        count += 1;
        if *val != zero {
            nonzero += 1;
        }
    }
    if count > 0 {
        (Some(sum.to_string()), count, nonzero)
    } else {
        (None, 0, 0)
    }
}

/// Float counterpart of `sum_int_column`. Accumulates in f64 (sufficient for
/// SUM(real)/SUM(double) over a segment); formats with `{:.17e}` to round-trip
/// exactly.
fn sum_float_column<T: Copy + Into<f64> + PartialEq + Default>(
    v: &[Option<T>],
) -> (Option<String>, i64, i64) {
    let zero = T::default();
    let mut sum: f64 = 0.0;
    let mut count: i64 = 0;
    let mut nonzero: i64 = 0;
    for val in v.iter().flatten() {
        sum += (*val).into();
        count += 1;
        if *val != zero {
            nonzero += 1;
        }
    }
    if count > 0 {
        (Some(format!("{:.17e}", sum)), count, nonzero)
    } else {
        (None, 0, 0)
    }
}

/// Compute sum, non-null count, and nonzero count for a typed column.
/// Returns (sum_as_string, nonnull_count, nonzero_count). Uses i128 for integer sums to avoid overflow.
pub(crate) fn compute_typed_sum(data: &TypedColumn) -> (Option<String>, i64, i64) {
    match data {
        TypedColumn::Int16(v) => sum_int_column(v),
        TypedColumn::Int32(v) => sum_int_column(v),
        TypedColumn::Int64(v) => sum_int_column(v),
        TypedColumn::Float32(v) => sum_float_column(v),
        TypedColumn::Float64(v) => sum_float_column(v),
        TypedColumn::Bool(_) => (None, 0, 0),
        TypedColumn::Text(v) => {
            // For text columns we store in _sum the sum of length(value) over
            // non-null rows (character count — same semantics as PostgreSQL's
            // `length(text)`); _nonnull_count counts non-null rows;
            // _nonzero_count counts rows with a non-empty string. These power
            // the length-sidecar metadata fast path without affecting numeric
            // SUM() resolution (the numeric fast path gates on type_oid).
            let mut sum: i128 = 0;
            let mut nonnull: i64 = 0;
            let mut nonempty: i64 = 0;
            for val in v.iter().flatten() {
                nonnull += 1;
                let chars = val.chars().count() as i128;
                sum += chars;
                if chars > 0 {
                    nonempty += 1;
                }
            }
            if nonnull > 0 {
                (Some(sum.to_string()), nonnull, nonempty)
            } else {
                (None, 0, 0)
            }
        }
        TypedColumn::Bytes(_) => {
            // jsonb columns — no meaningful numeric SUM / length sidecar.
            (None, 0, 0)
        }
    }
}

/// Compress a text column's per-row length array into a sidecar blob.
///
/// Wire format mirrors CompressedColumn: [type_tag=Lz4][row_count][has_nulls]
/// [null_bitmap?][lz4_flex::compress_prepend_size(u32 array of non-null lengths)].
///
/// Lengths are stored as *character* counts (same semantics as PostgreSQL's
/// `length(text)`), not byte counts, so the sidecar can directly serve
/// `length(col)` expressions.
///
/// This blob is a fraction of the main text blob (URL avg ~50 bytes per value,
/// length fits in 2 bytes; LZ4 shrinks further because neighbouring URLs on the
/// same site have similar lengths). Used by queries that only need length(col)
/// or col <> ''.
pub(crate) fn compress_text_lengths(values: &[Option<String>]) -> Vec<u8> {
    let (non_null, null_bitmap) = compression::extract_nulls(values);
    let mut u32_bytes = Vec::with_capacity(non_null.len() * 4);
    for s in &non_null {
        let chars = s.chars().count() as u32;
        u32_bytes.extend_from_slice(&chars.to_le_bytes());
    }
    let compressed = lz4_flex::compress_prepend_size(&u32_bytes);
    CompressedColumn {
        type_tag: CompressionType::Lz4,
        row_count: values.len() as u32,
        null_bitmap,
        data: compressed,
    }
    .to_bytes()
}

/// Compute the min and max of a column's string values using type-aware comparison.
fn compute_column_minmax(
    values: &[Option<String>],
    data_type: &str,
) -> (Option<String>, Option<String>) {
    let mut min_val: Option<&str> = None;
    let mut max_val: Option<&str> = None;

    for val in values.iter().flatten() {
        let v = val.as_str();
        min_val = Some(match min_val {
            None => v,
            Some(cur) => {
                if compare_values(v, cur, data_type) == std::cmp::Ordering::Less {
                    v
                } else {
                    cur
                }
            }
        });
        max_val = Some(match max_val {
            None => v,
            Some(cur) => {
                if compare_values(v, cur, data_type) == std::cmp::Ordering::Greater {
                    v
                } else {
                    cur
                }
            }
        });
    }

    (
        min_val.map(|s| s.to_string()),
        max_val.map(|s| s.to_string()),
    )
}

/// Type-aware comparison of string-encoded values.
fn compare_values(a: &str, b: &str, data_type: &str) -> std::cmp::Ordering {
    let dt = data_type.to_lowercase();
    if dt.contains("timestamp") || dt == "date" {
        // ISO format sorts lexicographically
        a.cmp(b)
    } else if dt == "double precision" || dt == "float8" || dt == "real" || dt == "float4" {
        let fa: f64 = a.parse().unwrap_or(0.0);
        let fb: f64 = b.parse().unwrap_or(0.0);
        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
    } else {
        // Integer types
        let ia: i64 = a.parse().unwrap_or(0);
        let ib: i64 = b.parse().unwrap_or(0);
        ia.cmp(&ib)
    }
}

/// Format a min/max value for SQL INSERT based on the column type.
pub(crate) fn format_minmax_for_insert(val: &str, data_type: &str) -> String {
    let dt = data_type.to_lowercase();
    if dt.contains("timestamp") {
        format!("'{}'::timestamptz", val.replace('\'', "''"))
    } else if dt == "date" {
        format!("'{}'::date", val.replace('\'', "''"))
    } else {
        // Numeric types — use the value directly
        val.to_string()
    }
}

/// Public function used by the background worker for auto-compression.
pub fn auto_compress_partitions(client: &mut SpiClient<'_>, ht: &catalog::DeltatableInfo) -> i32 {
    let compress_after = match &ht.compress_after {
        Some(interval) => interval,
        None => return 0,
    };

    if ht.order_by.is_empty() && ht.segment_by.is_empty() {
        return 0;
    }

    // Find partitions eligible for compression:
    // range_end < now() - compress_after AND NOT is_compressed
    let eligible = client
        .select(
            "SELECT table_name FROM deltax.deltax_partition
             WHERE deltatable_id = $1 AND is_compressed = false
               AND range_end < now() - $2::interval",
            None,
            &[ht.id.into(), (*compress_after).into()],
        )
        .expect("failed to query eligible partitions");

    let mut partition_names: Vec<String> = Vec::new();
    for row in eligible {
        let name: String = row
            .get_datum_by_ordinal(1)
            .unwrap()
            .value::<String>()
            .unwrap()
            .unwrap();
        partition_names.push(name);
    }

    let mut compressed = 0;
    for name in &partition_names {
        compress_partition_impl(client, name);
        compressed += 1;
    }

    compressed
}

// ============================================================================
// P1 compaction: fold loose heap rows (transparent INSERTs into compressed
// partitions) into new segments appended to the existing companion tables.
// See dev/docs/COMPRESSED_DML.md §4.5.
// ============================================================================

// ============================================================================
// P2 decompose-on-write (COMPRESSED_DML.md §5): per-segment decompose for
// transparent UPDATE/DELETE on compressed partitions.
// ============================================================================

/// Outcome of `decompose_segments_for_dml`. Counts reflect only segments
/// actually claimed by THIS transaction (the delete-first protocol skips
/// segments a concurrent transaction decomposed first).
pub(crate) struct DmlDecomposeStats {
    /// Rows restored into the partition heap from decomposed segments.
    pub(crate) restored_rows: i64,
    pub(crate) decomposed_segments: usize,
    /// Segments removed by the whole-segment DELETE fast path (§5.4) —
    /// their rows were never materialized.
    pub(crate) dropped_segments: usize,
    /// Logical rows removed by whole-segment drops (command-tag credit).
    pub(crate) dropped_rows: i64,
}

/// Decompose the given segments of a compressed partition back into ordinary
/// heap rows (`decompose_ids`), and/or drop them outright (`drop_ids`,
/// whole-segment DELETE fast path). Runs inside the user's transaction from
/// the ExecutorStart interceptor, BEFORE the UPDATE/DELETE executes.
///
/// Mechanics per segment: delete its `_meta` row (claim), decode its blobs
/// into rows inserted into the partition heap (skipped for `drop_ids`),
/// then delete its sidecar rows in `_colstats`/`_blobs`/`_blooms`/
/// `_text_lengths`/`_valbitmap`. Everything is plain WAL-logged heap DML in
/// one transaction, so MVCC, rollback and crash recovery need no
/// extension-side machinery (§5.3).
///
/// Locking & concurrency (§5.5): takes the same ACCESS EXCLUSIVE lock on the
/// partition that compaction takes, held to end of transaction. Because
/// every user DML already holds ROW EXCLUSIVE from planning, two concurrent
/// DMLs on the same compressed partition cannot interleave here — one
/// blocks (or the deadlock detector cancels one of them; the survivor is
/// always correct). The meta-row delete is the serialization point proper:
/// delete-first wins; a claim that deletes 0 rows means a concurrent
/// transaction already decomposed that segment and it is skipped. Under
/// REPEATABLE READ/SERIALIZABLE a lost claim surfaces as a serialization
/// error, matching vanilla PostgreSQL semantics. Readers that only hold
/// ACCESS SHARE block for the duration of the writing transaction — the
/// honest cost of decompose-on-write; reads either see the segment intact
/// (pre-commit snapshot) or the heap rows (post-commit), never both/neither.
///
/// Sidecar/metadata effects (§6):
/// - Catalog `column_minmax` is NOT recomputed: a range covering removed
///   values merely over-covers, which can only disable a pruning
///   opportunity (segments scanned unnecessarily) — never produce a wrong
///   answer. The same rule applies to partition-level bloom sentinels
///   (`_segment_id = -1`, PERF #47 — not on this branch) when they land.
/// - `column_valmap`/HLL/MCV/pg_statistic go stale (planner-only; refreshed
///   by the next compaction or recompress).
/// - Catalog `row_count`/`compressed_size` are decremented.
/// - `max_segment_id` is raised to the current meta maximum BEFORE deleting,
///   so a later compaction can never reuse a decomposed segment's id —
///   the shared blob/decompressed caches and the backend-local colstats
///   cache are keyed by `(companion_oid, segment_id, ...)` and id reuse
///   within one companion-table lifetime would poison them.
pub(crate) fn decompose_segments_for_dml(
    partition_oid: pg_sys::Oid,
    decompose_ids: &[i32],
    drop_ids: &[i32],
) -> DmlDecomposeStats {
    let mut stats = DmlDecomposeStats {
        restored_rows: 0,
        decomposed_segments: 0,
        dropped_segments: 0,
        dropped_rows: 0,
    };
    if decompose_ids.is_empty() && drop_ids.is_empty() {
        return stats;
    }

    // Internal DML below (companion deletes + heap inserts) must bypass the
    // ExecutorStart interception and plan against the plain heap. Guard
    // resets the flag on every exit path, including errors.
    let _bypass = DmlBypassGuard::new();
    // Tag the restored-row heap inserts + companion deletes with the
    // deltax_internal replication origin so origin=none logical subscribers
    // filter them (the user's own UPDATE/DELETE runs AFTER this returns, with
    // the origin reset, so it replicates normally). See COMPRESSED_DML.md §7.
    let _origin = InternalOriginGuard::new();

    let (schema, part_table) = partition_names_for_oid(partition_oid);

    Spi::connect_mut(|client| {
        let Some(part_info) = catalog::get_partition_by_name(client, &schema, &part_table)
            .expect("failed to query partition")
        else {
            return;
        };
        if !part_info.is_compressed {
            // Raced with a full decompress — the rows are already plain
            // heap tuples and the DML needs no help.
            return;
        }
        let ht = catalog::get_deltatable_by_id(client, part_info.deltatable_id)
            .expect("failed to query deltatable")
            .unwrap();

        // Decompose restores the parent table's physical columns only —
        // synthetic json_extract columns live solely in the companion blobs
        // (the physical JSONB column carries the data into the heap row).
        let columns = get_column_metadata(
            client,
            &ht.schema_name,
            &ht.table_name,
            &ht.segment_by,
            &ht.time_column,
            None,
        );

        // Schema-drift guard (same rule as compaction): the positional
        // col_idx mapping below assumes the column shape is unchanged since
        // compression. Decomposing with a drifted shape would restore
        // garbage — refuse and point at the full-rebuild path.
        let cc_now = catalog::snapshot_compressed_columns(
            client,
            &ht.schema_name,
            &ht.table_name,
            &ht.segment_by,
        )
        .expect("failed to snapshot column shape");
        let cc_now_val: serde_json::Value =
            serde_json::from_str(&cc_now).unwrap_or(serde_json::Value::Null);
        let shape_matches =
            matches!(&part_info.compressed_columns, Some(stored) if *stored == cc_now_val);
        if !shape_matches {
            pgrx::error!(
                "pg_deltax: cannot UPDATE/DELETE in compressed partition {}.{}: its column shape changed since compression; run deltax_decompress_partition() + deltax_compress_partition() first",
                schema,
                part_table
            );
        }

        let companion_schema = "_deltax_compressed";
        let meta_fqn = format!("\"{}\".\"{}_meta\"", companion_schema, part_table);
        let colstats_fqn = format!("\"{}\".\"{}_colstats\"", companion_schema, part_table);
        let blobs_fqn = format!("\"{}\".\"{}_blobs\"", companion_schema, part_table);
        let blooms_fqn = format!("\"{}\".\"{}_blooms\"", companion_schema, part_table);
        let text_lengths_fqn = format!("\"{}\".\"{}_text_lengths\"", companion_schema, part_table);
        let valbitmap_fqn = format!("\"{}\".\"{}_valbitmap\"", companion_schema, part_table);
        let tombstones_fqn = format!("\"{}\".\"{}_tombstones\"", companion_schema, part_table);
        let part_fqn = crate::partition::fqn(&schema, &part_table);

        // Serialize against compaction and concurrent decompose (§5.5).
        client
            .update(
                &format!("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE", part_fqn),
                None,
                &[],
            )
            .expect("failed to lock partition for decompose");

        // Raise the segment-id high-water mark BEFORE any deletion so
        // compaction can never reuse the ids we are about to remove.
        client
            .update(
                &format!(
                    "UPDATE deltax.deltax_partition
                     SET max_segment_id = GREATEST(COALESCE(max_segment_id, 0),
                         (SELECT COALESCE(MAX(_segment_id), 0) FROM {}))
                     WHERE id = $1",
                    meta_fqn
                ),
                None,
                &[part_info.id.into()],
            )
            .expect("failed to bump segment-id high-water mark");

        let id_list = |ids: &[i32]| {
            ids.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };

        // P2.5: tombstoned rows of the affected segments are already
        // logically deleted — they must neither be rematerialized by the
        // restore nor counted by the whole-drop's command-tag credit.
        let all_target_ids: Vec<i32> = drop_ids
            .iter()
            .chain(decompose_ids.iter())
            .copied()
            .collect();
        let tombstones = load_tombstone_offsets_sql(client, &tombstones_fqn, Some(&all_target_ids));

        let mut removed_ids: Vec<i32> = Vec::new();

        // Whole-segment DELETE fast path: claim + account, no row restore.
        if !drop_ids.is_empty() {
            let rows = client
                .update(
                    &format!(
                        "DELETE FROM {} WHERE _segment_id IN ({}) RETURNING _segment_id, _row_count",
                        meta_fqn,
                        id_list(drop_ids)
                    ),
                    None,
                    &[],
                )
                .expect("failed to claim segments for whole-segment delete");
            for row in rows {
                let sid: i32 = row
                    .get_datum_by_ordinal(1)
                    .unwrap()
                    .value::<i32>()
                    .unwrap()
                    .unwrap_or(0);
                let rc: i32 = row
                    .get_datum_by_ordinal(2)
                    .unwrap()
                    .value::<i32>()
                    .unwrap()
                    .unwrap_or(0);
                let tomb_count = tombstones.get(&sid).map_or(0, |s| s.len() as i32);
                removed_ids.push(sid);
                stats.dropped_segments += 1;
                stats.dropped_rows += (rc - tomb_count).max(0) as i64;
            }
        }

        // Decompose path: claim, then decode + restore each segment's rows.
        if !decompose_ids.is_empty() {
            // Positional col_idx mapping (same rule as compression /
            // decompress): non-segment-by physical columns in ordinal order.
            let mut non_seg_cols: Vec<(u16, String, String)> = Vec::new();
            let mut col_idx: u16 = 0;
            for col in &columns {
                if !col.is_segment_by {
                    non_seg_cols.push((col_idx, col.name.clone(), col.data_type.clone()));
                    col_idx += 1;
                }
            }

            let mut returning_cols = vec!["_segment_id".to_string()];
            for col in &columns {
                if col.is_segment_by {
                    returning_cols.push(format!("\"{}\"::text", col.name));
                }
            }
            returning_cols.push("_row_count".to_string());

            struct ClaimedSegment {
                segment_id: i32,
                segment_by_vals: Vec<Option<String>>,
                row_count: i32,
            }
            let mut claimed: Vec<ClaimedSegment> = Vec::new();
            {
                let rows = client
                    .update(
                        &format!(
                            "DELETE FROM {} WHERE _segment_id IN ({}) RETURNING {}",
                            meta_fqn,
                            id_list(decompose_ids),
                            returning_cols.join(", ")
                        ),
                        None,
                        &[],
                    )
                    .expect("failed to claim segments for decompose");
                for row in rows {
                    let mut ordinal = 1;
                    let segment_id: i32 = row
                        .get_datum_by_ordinal(ordinal)
                        .unwrap()
                        .value::<i32>()
                        .unwrap()
                        .unwrap_or(0);
                    ordinal += 1;
                    let mut segment_by_vals: Vec<Option<String>> = Vec::new();
                    for col in &columns {
                        if col.is_segment_by {
                            let val: Option<String> = row
                                .get_datum_by_ordinal(ordinal)
                                .unwrap()
                                .value::<String>()
                                .unwrap();
                            segment_by_vals.push(val);
                            ordinal += 1;
                        }
                    }
                    let row_count: i32 = row
                        .get_datum_by_ordinal(ordinal)
                        .unwrap()
                        .value::<i32>()
                        .unwrap()
                        .unwrap_or(0);
                    claimed.push(ClaimedSegment {
                        segment_id,
                        segment_by_vals,
                        row_count,
                    });
                }
            }

            for seg in &claimed {
                if seg.row_count > 0 {
                    stats.restored_rows += restore_segment_rows(
                        client,
                        &part_fqn,
                        &blobs_fqn,
                        &columns,
                        &non_seg_cols,
                        seg.segment_id,
                        &seg.segment_by_vals,
                        seg.row_count,
                        tombstones.get(&seg.segment_id),
                    );
                }
                removed_ids.push(seg.segment_id);
                stats.decomposed_segments += 1;
            }
        }

        if removed_ids.is_empty() {
            return;
        }

        // Compressed bytes being removed (planner-side counter only) —
        // measured before the blob rows are deleted.
        let removed_bytes: i64 = client
            .select(
                &format!(
                    "SELECT COALESCE(SUM(octet_length(_data)), 0)::bigint FROM {} WHERE _segment_id IN ({})",
                    blobs_fqn,
                    id_list(&removed_ids)
                ),
                None,
                &[],
            )
            .ok()
            .and_then(|r| r.first().get_one::<i64>().ok().flatten())
            .unwrap_or(0);

        // Sidecar rows go with the segment (colstats/blooms/bitmaps for
        // LIVE segments stay exact — the architecture's invariant). Rows
        // with negative `_segment_id` (reserved for partition-level
        // sentinels, PERF #47) are untouched by construction: claimed ids
        // are positive. Tombstone rows are consumed with their segment
        // (the restore above skipped them).
        for fqn in [
            &colstats_fqn,
            &blobs_fqn,
            &blooms_fqn,
            &text_lengths_fqn,
            &valbitmap_fqn,
            &tombstones_fqn,
        ] {
            if relation_exists(client, fqn) {
                client
                    .update(
                        &format!(
                            "DELETE FROM {} WHERE _segment_id IN ({})",
                            fqn,
                            id_list(&removed_ids)
                        ),
                        None,
                        &[],
                    )
                    .expect("failed to delete segment sidecar rows");
            }
        }

        catalog::bump_partition_decompose(
            client,
            part_info.id,
            stats.restored_rows + stats.dropped_rows,
            removed_bytes,
        )
        .expect("failed to update partition counters after decompose");

        // Restored rows are loose rows in the partition heap. The insert
        // trigger skips its flag UPDATE under DML_BYPASS (active here), so
        // set `has_loose_rows` explicitly on this client. Only when rows
        // were actually restored — a pure whole-segment DELETE (drop_ids,
        // restored_rows == 0) adds no loose rows.
        if stats.restored_rows > 0 {
            catalog::set_partition_has_loose_rows(client, part_info.id, true)
                .expect("failed to set has_loose_rows flag after decompose");
        }
    });

    // Let other backends drop cached row-count/plan state for this
    // partition (mirrors compaction; the flag flip above is what the gates
    // read, and this invalidation drops their cached flag map).
    unsafe { pg_sys::CacheInvalidateRelcacheByRelid(partition_oid) };

    stats
}

/// Resolve a partition OID to `(schema, table)` names, erroring out on
/// failure (shared by the P2 decompose and P2.5 tombstone write paths).
fn partition_names_for_oid(partition_oid: pg_sys::Oid) -> (String, String) {
    unsafe {
        let name_ptr = pg_sys::get_rel_name(partition_oid);
        let ns_ptr = pg_sys::get_namespace_name(pg_sys::get_rel_namespace(partition_oid));
        if name_ptr.is_null() || ns_ptr.is_null() {
            pgrx::error!(
                "pg_deltax: cannot resolve partition OID {} for compressed DML",
                u32::from(partition_oid)
            );
        }
        (
            std::ffi::CStr::from_ptr(ns_ptr)
                .to_string_lossy()
                .into_owned(),
            std::ffi::CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// P2.5 tombstone DELETE, step 1: lock the candidate segments' `_meta` rows
/// `FOR UPDATE` (ordered, to avoid ABBA between concurrent tombstone
/// writers) and return the ids still alive. The row locks are held to end
/// of transaction, so a concurrent decompose/compaction of those segments
/// blocks until we commit — and then sees our tombstones (its restore
/// excludes them). Candidates whose meta row is already gone (concurrently
/// decomposed) are simply absent from the result: their rows are ordinary
/// heap tuples now and the planned DELETE handles whatever its snapshot
/// sees — the same "0 rows claimed → skip" rule P2 uses. Returns an empty
/// vec when the partition is no longer compressed (raced a full decompress).
pub(crate) fn claim_segments_for_tombstone(
    partition_oid: pg_sys::Oid,
    segment_ids: &[i32],
) -> Vec<i32> {
    if segment_ids.is_empty() {
        return Vec::new();
    }
    let _bypass = DmlBypassGuard::new();
    let (_schema, part_table) = partition_names_for_oid(partition_oid);
    let meta_fqn = format!("\"_deltax_compressed\".\"{}_meta\"", part_table);

    Spi::connect_mut(|client| {
        if !relation_exists(client, &meta_fqn) {
            return Vec::new();
        }
        let id_list = segment_ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        // Goes through `update` (read_only = false): SELECT ... FOR UPDATE
        // is not allowed on SPI's read-only path.
        let rows = client
            .update(
                &format!(
                    "SELECT _segment_id FROM {} WHERE _segment_id IN ({}) ORDER BY _segment_id FOR UPDATE",
                    meta_fqn, id_list
                ),
                None,
                &[],
            )
            .expect("failed to lock segments for tombstone DELETE");
        let mut survivors = Vec::new();
        for row in rows {
            if let Some(sid) = row.get::<i32>(1).ok().flatten() {
                survivors.push(sid);
            }
        }
        survivors
    })
}

/// P2.5 tombstone DELETE, step 2: record the exactly-matching rows as
/// `(segment_id, row_offset)` tombstones. Ordinary heap inserts in the
/// user's transaction ⇒ MVCC, rollback and replication are native; readers
/// see the segment unfiltered until commit and exactly filtered after.
///
/// `ON CONFLICT DO NOTHING` deduplicates rows that are already tombstoned
/// (by us under a previous statement, or by a concurrent committed
/// transaction — matching READ COMMITTED's "row already deleted → skip");
/// only rows actually inserted count toward the command tag and the catalog
/// `row_count` decrement (catalog counts stay exact: they track LIVE rows
/// stored in segments, which is what `DeltaXCount`'s catalog path serves).
///
/// Returns the number of rows logically deleted.
pub(crate) fn insert_dml_tombstones(
    partition_oid: pg_sys::Oid,
    targets: &[(i32, Vec<u32>)],
) -> i64 {
    if targets.is_empty() {
        return 0;
    }
    let _bypass = DmlBypassGuard::new();
    let (schema, part_table) = partition_names_for_oid(partition_oid);
    let tombstones_fqn = format!("\"_deltax_compressed\".\"{}_tombstones\"", part_table);

    let inserted = Spi::connect_mut(|client| {
        let Some(part_info) = catalog::get_partition_by_name(client, &schema, &part_table)
            .expect("failed to query partition")
        else {
            return 0i64;
        };
        if !part_info.is_compressed {
            return 0i64;
        }
        // Lazy creation for partitions compressed before P2.5 (newer
        // compressions create it eagerly so the GRANT cascade covers it).
        if !relation_exists(client, &tombstones_fqn) {
            client
                .update(
                    &format!(
                        "CREATE TABLE IF NOT EXISTS {} (_segment_id INT NOT NULL, _row_offset INT NOT NULL, PRIMARY KEY (_segment_id, _row_offset))",
                        tombstones_fqn
                    ),
                    None,
                    &[],
                )
                .expect("failed to create tombstones table");
        }

        let mut inserted: i64 = 0;
        for (segment_id, offsets) in targets {
            if offsets.is_empty() {
                continue;
            }
            let offs = offsets
                .iter()
                .map(|o| o.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let rows = client
                .update(
                    &format!(
                        "INSERT INTO {} (_segment_id, _row_offset)
                         SELECT {}, o FROM unnest(ARRAY[{}]::int4[]) AS o
                         ON CONFLICT DO NOTHING
                         RETURNING 1",
                        tombstones_fqn, segment_id, offs
                    ),
                    None,
                    &[],
                )
                .expect("failed to insert tombstones");
            inserted += rows.len() as i64;
        }

        if inserted > 0 {
            // Catalog row_count tracks live segment rows — keep it exact so
            // the DeltaXCount catalog shortcut and planner stats stay right.
            catalog::bump_partition_decompose(client, part_info.id, inserted, 0)
                .expect("failed to update partition counters after tombstone DELETE");
            // Flip the P2.5 gate flag in the same transaction (MVCC-atomic
            // with the tombstone rows): the plan/exec gates read this
            // instead of probing the _tombstones table.
            catalog::set_partition_has_tombstones(client, part_info.id, true)
                .expect("failed to set has_tombstones flag");
        }
        inserted
    });

    if inserted > 0 {
        // Invalidate cached plans for this partition: plans built before the
        // first tombstone may carry DeltaXAgg / DeltaXMinMax fast paths that
        // are gated off while tombstones exist (their exec-time stale guards
        // would otherwise error). Same mechanism P1 uses for the heap-tail
        // empty→non-empty transition.
        unsafe { pg_sys::CacheInvalidateRelcacheByRelid(partition_oid) };
    }
    inserted
}

/// Compact the loose heap rows of a compressed partition into new segments.
///
/// Loose rows are INSERTs that arrived after the partition was compressed
/// (P1 transparent DML). They live in the partition heap and are unioned
/// with the segment data at scan time; compaction moves them into proper
/// segments so the metadata-only fast paths re-engage.
#[pg_extern]
fn deltax_compact_partition(partition: &str) -> String {
    Spi::connect_mut(|client| compact_partition_impl(client, partition))
}

/// RAII guard for the backend-local DML/planner bypass flag. While held,
/// the set_rel_pathlist / upper-paths hooks plan queries against the plain
/// heap (no DeltaX custom scans) and the DML rejection (trigger + executor
/// hook) is disabled. Drop-restore (not drop-reset) so guards NEST safely —
/// compaction holds one while calling `decompose_segments_for_dml`, which
/// takes its own — and an error inside compaction can't leave the backend
/// planning every later query without DeltaX paths.
struct DmlBypassGuard {
    prev: bool,
}

impl DmlBypassGuard {
    fn new() -> Self {
        let prev = crate::scan::dml_bypass_active();
        crate::scan::set_dml_bypass(true);
        DmlBypassGuard { prev }
    }
}

impl Drop for DmlBypassGuard {
    fn drop(&mut self) {
        crate::scan::set_dml_bypass(self.prev);
    }
}

/// Backend-local cache of the `deltax_internal` replication-origin id.
/// `InvalidRepOriginId` (0) means "not found" — the guard then no-ops.
fn internal_origin_id() -> pg_sys::RepOriginId {
    thread_local! {
        static ID: std::cell::Cell<pg_sys::RepOriginId> =
            const { std::cell::Cell::new(pg_sys::InvalidRepOriginId as pg_sys::RepOriginId) };
    }
    ID.with(|c| {
        let cached = c.get();
        if cached != pg_sys::InvalidRepOriginId as pg_sys::RepOriginId {
            return cached;
        }
        // Look up by name at the C level (no SQL-function ACL; the origin is
        // created in the extension SQL). `missing_ok = true` returns
        // InvalidRepOriginId rather than erroring when absent.
        let id = unsafe { pg_sys::replorigin_by_name(c"deltax_internal".as_ptr(), true) };
        if id != pg_sys::InvalidRepOriginId as pg_sys::RepOriginId {
            c.set(id);
        }
        id
    })
}

/// RAII guard that tags all WAL written in its scope with the
/// `deltax_internal` replication origin, so logical subscribers created
/// `WITH (origin = none)` filter these internal maintenance changes
/// (decompose-on-write's restored-row inserts, compaction's segment folds and
/// loose-row removal) while still receiving the user's own DML. Subscribers
/// using the default `origin = any` receive the tagged changes normally, so
/// tagging is safe for both replication topologies (see COMPRESSED_DML.md §7).
///
/// Sets the backend-local `replorigin_session_origin` directly rather than via
/// `replorigin_session_setup`: we only need per-record WAL tagging, not the
/// exclusive progress slot session_setup reserves (which would serialize
/// concurrent internal writers). Nests safely (save/restore).
struct InternalOriginGuard {
    prev: pg_sys::RepOriginId,
}

impl InternalOriginGuard {
    fn new() -> Self {
        let prev = unsafe { pg_sys::replorigin_session_origin };
        // If the origin is absent, id is Invalid → this is a no-op (records
        // stay untagged), preserving today's behavior.
        unsafe { pg_sys::replorigin_session_origin = internal_origin_id() };
        InternalOriginGuard { prev }
    }
}

impl Drop for InternalOriginGuard {
    fn drop(&mut self) {
        unsafe { pg_sys::replorigin_session_origin = self.prev };
    }
}

/// Rewrite a companion `CREATE TABLE` DDL into `CREATE TABLE IF NOT EXISTS`.
/// Compaction may need to create sidecar tables that the original
/// compression skipped (e.g. `_blooms` when the GUC was off, `_text_lengths`
/// when there were no text columns yet).
pub(crate) fn ddl_if_not_exists(ddl: &str) -> String {
    ddl.replacen("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ", 1)
}

/// True when the relation named by `fqn` (a quoted, schema-qualified name)
/// exists.
fn relation_exists(client: &SpiClient, fqn: &str) -> bool {
    client
        .select("SELECT to_regclass($1) IS NOT NULL", Some(1), &[fqn.into()])
        .ok()
        .and_then(|r| r.first().get_one::<bool>().ok().flatten())
        .unwrap_or(false)
}

pub(crate) fn compact_partition_impl(client: &mut SpiClient, partition: &str) -> String {
    let (schema, part_table) = crate::partition::resolve_relation(client, partition);
    let part_info = catalog::get_partition_by_name(client, &schema, &part_table)
        .expect("failed to query partition")
        .unwrap_or_else(|| {
            pgrx::error!(
                "pg_deltax: partition {}.{} not found in catalog",
                schema,
                part_table
            )
        });

    if !part_info.is_compressed {
        return format!(
            "Partition {}.{} is not compressed; nothing to compact",
            schema, part_table
        );
    }

    let ht = catalog::get_deltatable_by_id(client, part_info.deltatable_id)
        .expect("failed to query deltatable")
        .unwrap();

    let part_fqn = crate::partition::fqn(&schema, &part_table);

    // Everything below reads the partition through plain heap plans (the
    // custom scans would union the segments back in — double-counting) and
    // deletes the compacted rows past the DML rejection. The guard resets
    // the flag on every exit path, including errors.
    let _bypass = DmlBypassGuard::new();
    // Compaction is pure internal churn (fold loose rows into fresh segments,
    // rewrite tombstoned segments, delete/truncate the loose region) — none of
    // it changes the table's logical content. Tag its WAL so origin=none
    // logical subscribers ignore it. See COMPRESSED_DML.md §7.
    let _origin = InternalOriginGuard::new();

    // Serialize against concurrent writers: the read → flush → delete cycle
    // below requires a stable visible row set. The lock is acquired BEFORE
    // the row-reading snapshot, so in READ COMMITTED every committed loose
    // row is visible and TRUNCATE at the end cannot destroy unseen data.
    client
        .update(
            &format!("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE", part_fqn),
            None,
            &[],
        )
        .expect("failed to lock partition for compaction");

    // Re-read the catalog row now that the lock is held: a concurrent
    // decompress could have flipped is_compressed between the unlocked
    // check above and the lock grant — compacting then would treat the
    // fully-restored heap as "loose rows" and truncate real data. The
    // re-read also pins compressed_columns for the shape guard below.
    let part_info = catalog::get_partition_by_name(client, &schema, &part_table)
        .expect("failed to re-query partition")
        .unwrap_or_else(|| {
            pgrx::error!(
                "pg_deltax: partition {}.{} disappeared from catalog during compaction",
                schema,
                part_table
            )
        });
    if !part_info.is_compressed {
        return format!(
            "Partition {}.{} is not compressed; nothing to compact",
            schema, part_table
        );
    }

    // P2.5: physically rewrite tombstone-bearing segments. Decompose them
    // (P2 machinery — the restore skips tombstoned rows and deletes the
    // segments' tombstone rows), then let the normal compaction below fold
    // the restored live rows into fresh, pristine segments. Finally restore
    // the zero-block fast gate by TRUNCATE-ing the now-empty tombstones
    // table (also clears dead pages left by rolled-back tombstone DML).
    let tombstones_fqn = format!("\"_deltax_compressed\".\"{}_tombstones\"", part_table);
    let mut rewritten_segments = 0usize;
    if relation_exists(client, &tombstones_fqn) {
        let mut tomb_ids: Vec<i32> = Vec::new();
        for row in client
            .select(
                &format!("SELECT DISTINCT _segment_id FROM {}", tombstones_fqn),
                None,
                &[],
            )
            .expect("failed to read tombstoned segment ids")
        {
            if let Some(sid) = row.get::<i32>(1).ok().flatten() {
                tomb_ids.push(sid);
            }
        }
        if !tomb_ids.is_empty() {
            let part_oid: pg_sys::Oid = client
                .select(
                    "SELECT to_regclass($1)::oid",
                    Some(1),
                    &[part_fqn.as_str().into()],
                )
                .ok()
                .and_then(|r| r.first().get_one::<pg_sys::Oid>().ok().flatten())
                .unwrap_or(pg_sys::InvalidOid);
            if part_oid != pg_sys::InvalidOid {
                let stats = decompose_segments_for_dml(part_oid, &tomb_ids, &[]);
                rewritten_segments = stats.decomposed_segments;
            }
        }
        let any_left: bool = client
            .select(
                &format!("SELECT EXISTS (SELECT 1 FROM {})", tombstones_fqn),
                Some(1),
                &[],
            )
            .ok()
            .and_then(|r| r.first().get_one::<bool>().ok().flatten())
            .unwrap_or(true);
        if !any_left {
            client
                .update(&format!("TRUNCATE {}", tombstones_fqn), None, &[])
                .expect("failed to truncate tombstones table");
            // All tombstones folded away — clear the P2.5 gate flag so
            // MIN/MAX/agg pushdowns re-engage for this partition.
            catalog::set_partition_has_tombstones(client, part_info.id, false)
                .expect("failed to clear has_tombstones flag after compaction");
        }
    }

    let loose_rows: i64 = client
        .select(
            &format!("SELECT count(*) FROM ONLY {}", part_fqn),
            None,
            &[],
        )
        .expect("failed to count loose rows")
        .first()
        .get_one::<i64>()
        .ok()
        .flatten()
        .unwrap_or(0);
    if loose_rows == 0 {
        // A prior REPEATABLE READ compaction deletes (rather than truncates)
        // the rows it compacted, and autovacuum is disabled on compressed
        // partitions — so dead rows would keep the heap at nonzero blocks
        // forever, pinning every scan on the slower heap-tail path and
        // re-locking here each worker cycle. With the AEL held and zero
        // visible rows, truncating away the dead tuples is the same MVCC
        // tradeoff as the compress-time TRUNCATE.
        let read_committed =
            unsafe { pg_sys::XactIsoLevel } < pg_sys::XACT_REPEATABLE_READ as std::ffi::c_int;
        let heap_bytes: i64 = client
            .select(
                &format!("SELECT pg_relation_size('{}'::regclass)", part_fqn),
                Some(1),
                &[],
            )
            .ok()
            .and_then(|r| r.first().get_one::<i64>().ok().flatten())
            .unwrap_or(0);
        if read_committed && heap_bytes > 0 {
            client
                .update(&format!("TRUNCATE ONLY {}", part_fqn), None, &[])
                .expect("failed to truncate dead loose-row region");
            crate::scan::invalidate_compressed_cache();
        }
        // No visible loose rows remain (dead pages from a rolled-back or
        // RR-deleted insert don't count) — clear the stale P1 gate flag.
        catalog::set_partition_has_loose_rows(client, part_info.id, false)
            .expect("failed to clear has_loose_rows flag");
        if rewritten_segments > 0 {
            // Everything in the tombstoned segments was deleted — the
            // decompose removed them outright and there is nothing to fold
            // (decompose_segments_for_dml already fired the relcache
            // invalidation for this partition).
            return format!(
                "Partition {}.{}: rewrote {} tombstoned segment(s); no loose rows to compact",
                schema, part_table, rewritten_segments
            );
        }
        return format!(
            "Partition {}.{} has no loose rows to compact",
            schema, part_table
        );
    }

    // Schema-drift guard: appending segments assumes the companion layout
    // (col_idx assignment) still matches the live table shape. Refuse when
    // the parent's column shape changed since compression — the safe path
    // is a full decompress + recompress.
    let cc_now = catalog::snapshot_compressed_columns(
        client,
        &ht.schema_name,
        &ht.table_name,
        &ht.segment_by,
    )
    .expect("failed to snapshot column shape");
    let cc_now_val: serde_json::Value =
        serde_json::from_str(&cc_now).unwrap_or(serde_json::Value::Null);
    let shape_matches =
        matches!(&part_info.compressed_columns, Some(stored) if *stored == cc_now_val);
    if !shape_matches {
        pgrx::error!(
            "pg_deltax: partition {}.{} has loose rows but its column shape changed since compression; run deltax_decompress_partition() + deltax_compress_partition() instead",
            schema,
            part_table
        );
    }

    let columns = get_column_metadata(
        client,
        &schema,
        &part_table,
        &ht.segment_by,
        &ht.time_column,
        ht.json_extract.as_ref(),
    );
    if columns.is_empty() {
        pgrx::error!("pg_deltax: no columns found for {}.{}", schema, part_table);
    }
    // Synthetic json_extract columns don't exist in the partition heap, so
    // the loose-row cursor below cannot read them (and the companion layout
    // expects them). Bail with the supported remedy instead of a raw SQL
    // error from the cursor.
    if columns.iter().any(|c| c.extracted.is_some()) {
        pgrx::error!(
            "pg_deltax: partition {}.{} uses json_extract; compacting loose rows is not supported — run deltax_decompress_partition() + deltax_compress_partition() instead",
            schema,
            part_table
        );
    }

    let ddl = build_companion_ddl(&part_table, &columns);
    if !relation_exists(client, &ddl.meta_fqn) {
        pgrx::error!(
            "pg_deltax: companion meta table missing for compressed partition {}.{}",
            schema,
            part_table
        );
    }

    // New segment ids continue after the existing maximum (sentinel rows in
    // _blooms use -1 and never appear in _meta). The catalog high-water
    // mark guards against id REUSE: decompose-on-write (P2) deletes meta
    // rows — possibly the current maximum — after recording it in
    // `max_segment_id`. Reusing such an id would poison the shared
    // blob/decompressed caches and the backend-local colstats cache, which
    // are keyed by `(companion_oid, segment_id, ...)` and assume ids are
    // never recycled within a companion table's lifetime.
    let max_segment_id: i32 = client
        .select(
            &format!(
                "SELECT GREATEST(COALESCE(MAX(_segment_id), 0),
                        (SELECT COALESCE(max_segment_id, 0)
                           FROM deltax.deltax_partition WHERE id = {}))
                 FROM {}",
                part_info.id, ddl.meta_fqn
            ),
            None,
            &[],
        )
        .expect("failed to read max segment id")
        .first()
        .get_one::<i32>()
        .ok()
        .flatten()
        .unwrap_or(0);
    let mut next_segment_id: i32 = max_segment_id + 1;
    let first_new_segment_id = next_segment_id;

    let segment_size = ht.segment_size as usize;
    let kinds: Vec<ColumnKind> = columns
        .iter()
        .map(|c| classify_column(&c.data_type, c.is_segment_by))
        .collect();

    // segment_by and text cols cast to ::text; jsonb stays native for the
    // `JsonbRaw` read (see the matching SELECT builder in
    // compress_partition_impl for why).
    let select_cols = columns
        .iter()
        .zip(kinds.iter())
        .map(|(c, kind)| {
            if c.is_segment_by || matches!(kind, ColumnKind::Text) {
                format!("\"{}\"::text", c.name)
            } else {
                format!("\"{}\"", c.name)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    let order_clause = if !ht.segment_by.is_empty() {
        let mut order_parts = Vec::new();
        for s in &ht.segment_by {
            order_parts.push(format!("\"{}\"", s));
        }
        for o in &ht.order_by {
            order_parts.push(format!("\"{}\"", o));
        }
        format!(" ORDER BY {}", order_parts.join(", "))
    } else {
        String::new()
    };

    let order_col_indices: Vec<usize> = ht
        .order_by
        .iter()
        .filter_map(|name| columns.iter().position(|c| c.name == *name))
        .collect();
    let seg_col_indices: Vec<usize> = columns
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_segment_by)
        .map(|(i, _)| i)
        .collect();

    let cursor_sql = format!(
        "DECLARE compact_cursor CURSOR FOR SELECT {} FROM ONLY {}{}",
        select_cols, part_fqn, order_clause
    );
    client
        .update(&cursor_sql, None, &[])
        .expect("failed to declare compaction cursor");
    let batch_size = segment_size;
    let fetch_sql = format!("FETCH {} FROM compact_cursor", batch_size);

    let mut typed_cols = init_typed_columns(&columns, &kinds);
    let mut current_seg_values: Vec<Option<String>> = Vec::new();
    let mut rows_in_segment: usize = 0;
    let mut total_compressed_size: i64 = 0;
    let mut total_rows: i64 = 0;
    let mut blob_buffer: Vec<(u16, i32, Vec<u8>)> = Vec::new();
    let mut bloom_buffer: Vec<(u16, i32, u8, Vec<u8>)> = Vec::new();
    let mut colstats_buffer: Vec<ColstatsRow> = Vec::new();
    let mut text_length_buffer: Vec<(u16, i32, Vec<u8>)> = Vec::new();
    let mut valbitmap_value_buffer: Vec<(u16, i32, SegValueCounts)> = Vec::new();
    let num_nonseg_cols = columns.iter().filter(|c| !c.is_segment_by).count();
    // Batch-only HLL — merged into the persisted partition sketches below.
    let mut batch_hll: Vec<CardinalityEstimator<u64>> = (0..num_nonseg_cols)
        .map(|_| CardinalityEstimator::<u64>::new())
        .collect();
    let mut batch_topvals: TopVals = (0..num_nonseg_cols)
        .map(|_| std::collections::HashMap::new())
        .collect();

    loop {
        let result = client
            .select(&fetch_sql, None, &[])
            .expect("failed to fetch from compaction cursor");
        let fetched = result.len();
        if fetched == 0 {
            break;
        }
        let tuptable_to_free = unsafe { pg_sys::SPI_tuptable };

        for row in result {
            if !seg_col_indices.is_empty() {
                let row_seg_values: Vec<Option<String>> = seg_col_indices
                    .iter()
                    .map(|&i| {
                        row.get_datum_by_ordinal(i + 1)
                            .unwrap()
                            .value::<String>()
                            .unwrap()
                    })
                    .collect();
                if current_seg_values.is_empty() {
                    current_seg_values = row_seg_values;
                } else if row_seg_values != current_seg_values {
                    if rows_in_segment > 0 {
                        total_compressed_size += flush_with_splitting(
                            client,
                            &ddl.meta_fqn,
                            &ddl.colstats_fqn,
                            &columns,
                            &typed_cols,
                            &current_seg_values,
                            rows_in_segment,
                            segment_size,
                            &mut next_segment_id,
                            &mut blob_buffer,
                            &mut bloom_buffer,
                            &mut colstats_buffer,
                            &mut text_length_buffer,
                            &mut valbitmap_value_buffer,
                            &mut batch_hll,
                            &mut batch_topvals,
                        );
                        typed_cols = init_typed_columns(&columns, &kinds);
                        rows_in_segment = 0;
                    }
                    current_seg_values = row_seg_values;
                }
            }

            append_row_to_columns(&row, &columns, &kinds, &mut typed_cols);
            rows_in_segment += 1;
            total_rows += 1;

            if rows_in_segment >= segment_size {
                if seg_col_indices.is_empty() {
                    sort_typed_columns(&mut typed_cols, &order_col_indices, rows_in_segment);
                }
                total_compressed_size += flush_with_splitting(
                    client,
                    &ddl.meta_fqn,
                    &ddl.colstats_fqn,
                    &columns,
                    &typed_cols,
                    &current_seg_values,
                    rows_in_segment,
                    segment_size,
                    &mut next_segment_id,
                    &mut blob_buffer,
                    &mut bloom_buffer,
                    &mut colstats_buffer,
                    &mut text_length_buffer,
                    &mut valbitmap_value_buffer,
                    &mut batch_hll,
                    &mut batch_topvals,
                );
                typed_cols = init_typed_columns(&columns, &kinds);
                rows_in_segment = 0;
            }
        }

        if !tuptable_to_free.is_null() {
            unsafe { pg_sys::SPI_freetuptable(tuptable_to_free) };
        }
        if fetched < batch_size {
            break;
        }
    }

    if rows_in_segment > 0 {
        if seg_col_indices.is_empty() {
            sort_typed_columns(&mut typed_cols, &order_col_indices, rows_in_segment);
        }
        total_compressed_size += flush_with_splitting(
            client,
            &ddl.meta_fqn,
            &ddl.colstats_fqn,
            &columns,
            &typed_cols,
            &current_seg_values,
            rows_in_segment,
            segment_size,
            &mut next_segment_id,
            &mut blob_buffer,
            &mut bloom_buffer,
            &mut colstats_buffer,
            &mut text_length_buffer,
            &mut valbitmap_value_buffer,
            &mut batch_hll,
            &mut batch_topvals,
        );
    }

    client
        .update("CLOSE compact_cursor", None, &[])
        .expect("failed to close compaction cursor");

    if total_rows == 0 {
        return format!(
            "Partition {}.{} has no loose rows to compact",
            schema, part_table
        );
    }

    // ---- Append sidecar rows to the TOAST-backed companion tables. ----
    if !colstats_buffer.is_empty() {
        colstats_buffer.sort_by_key(|r| (r.col_idx, r.segment_id));
        for chunk in colstats_buffer.chunks(100) {
            let values: Vec<String> = chunk
                .iter()
                .map(|r| {
                    let min_str = r.min_val.map_or("NULL".to_string(), |v| v.to_string());
                    let max_str = r.max_val.map_or("NULL".to_string(), |v| v.to_string());
                    let sum_str = r.sum_val.as_deref().unwrap_or("NULL");
                    format!(
                        "({}, {}, {}, {}, {}, {}, {}, {})",
                        r.col_idx,
                        r.segment_id,
                        min_str,
                        max_str,
                        sum_str,
                        r.nonnull_count,
                        r.nonzero_count,
                        r.ndistinct
                    )
                })
                .collect();
            let sql = format!(
                "INSERT INTO {} (_col_idx, _segment_id, _min, _max, _sum, _nonnull_count, _nonzero_count, _ndistinct) VALUES {}",
                ddl.colstats_fqn,
                values.join(", ")
            );
            client
                .update(&sql, None, &[])
                .expect("failed to insert compaction colstats");
        }
    }

    if !blob_buffer.is_empty() {
        client
            .update(&ddl_if_not_exists(&ddl.blobs_ddl), None, &[])
            .expect("failed to ensure blobs table");
        blob_buffer.sort_by_key(|&(col_idx, seg_id, _)| (col_idx, seg_id));
        for (col_idx, seg_id, blob) in blob_buffer {
            use pgrx::datum::DatumWithOid;
            let insert_sql = format!(
                "INSERT INTO {} (_col_idx, _segment_id, _data) VALUES ($1, $2, $3)",
                ddl.blobs_fqn
            );
            let args: Vec<DatumWithOid> = vec![
                (col_idx as i16).into(),
                seg_id.into(),
                DatumWithOid::from(blob),
            ];
            client
                .update(&insert_sql, None, &args)
                .expect("failed to insert compaction blob");
        }
    }

    if !bloom_buffer.is_empty() {
        client
            .update(&ddl_if_not_exists(&ddl.blooms_ddl), None, &[])
            .expect("failed to ensure blooms table");
        bloom_buffer.sort_by_key(|&(col_idx, seg_id, _, _)| (col_idx, seg_id));
        for (col_idx, seg_id, num_hashes, bloom_bytes) in bloom_buffer {
            use pgrx::datum::DatumWithOid;
            let insert_sql = format!(
                "INSERT INTO {} (_col_idx, _segment_id, _num_hashes, _data) VALUES ($1, $2, $3, $4)",
                ddl.blooms_fqn
            );
            let args: Vec<DatumWithOid> = vec![
                (col_idx as i16).into(),
                seg_id.into(),
                (num_hashes as i16).into(),
                DatumWithOid::from(bloom_bytes),
            ];
            client
                .update(&insert_sql, None, &args)
                .expect("failed to insert compaction bloom");
        }
    }

    if !text_length_buffer.is_empty() {
        client
            .update(&ddl_if_not_exists(&ddl.text_lengths_ddl), None, &[])
            .expect("failed to ensure text_lengths table");
        text_length_buffer.sort_by_key(|&(col_idx, seg_id, _)| (col_idx, seg_id));
        for (col_idx, seg_id, blob) in text_length_buffer {
            use pgrx::datum::DatumWithOid;
            let insert_sql = format!(
                "INSERT INTO {} (_col_idx, _segment_id, _data) VALUES ($1, $2, $3)",
                ddl.text_lengths_fqn
            );
            let args: Vec<DatumWithOid> = vec![
                (col_idx as i16).into(),
                seg_id.into(),
                DatumWithOid::from(blob),
            ];
            client
                .update(&insert_sql, None, &args)
                .expect("failed to insert compaction text length sidecar");
        }
    }

    // NOTE for partition-level bloom sentinels (PERF #47, not yet on this
    // branch): compaction appends segments to a live companion table, so if
    // `_segment_id = -1` sentinel rows ever exist they MUST be fold-merged
    // with the batch's value hashes or deleted here — a sentinel that
    // under-covers any live segment produces wrong query results.

    // ---- Valbitmap: keep every existing column's bitmap regime exact, or
    // drop it (absence is safe — segments just stop being pruned on that
    // column). ----
    let new_segment_ids: Vec<i32> = (first_new_segment_id..next_segment_id).collect();
    append_valbitmaps_for_compaction(
        client,
        part_info.id,
        &ddl.valbitmap_fqn,
        &columns,
        &valbitmap_value_buffer,
        &new_segment_ids,
    );

    // ---- Catalog refresh ----
    catalog::bump_partition_compaction(client, part_info.id, total_rows, total_compressed_size)
        .expect("failed to bump partition counters");

    // Partition-level [min,max] map (Phase -1 pruning) — recomputed from the
    // full colstats table, which now includes the new segments. CORRECTNESS-
    // critical: new values may extend the old ranges.
    catalog::update_partition_column_minmax(client, part_info.id, &ddl.colstats_fqn, &columns)
        .expect("failed to refresh partition column_minmax");

    // ndistinct/HLL: merge the batch sketches into the persisted ones
    // (planner-only; best-effort).
    merge_batch_hll_into_catalog(client, part_info.id, &columns, &batch_hll);

    // ---- Remove the compacted loose rows. READ COMMITTED: TRUNCATE is safe
    // (the AEL was taken before the reading snapshot, so the visible set is
    // complete) and restores the zero-block state that re-enables every
    // metadata fast path. Under REPEATABLE READ / SERIALIZABLE the reading
    // snapshot may predate the lock, so DELETE only what we actually read
    // (rows invisible to the snapshot survive for a later compaction). ----
    let read_committed =
        unsafe { pg_sys::XactIsoLevel } < pg_sys::XACT_REPEATABLE_READ as std::ffi::c_int;
    if read_committed {
        client
            .update(&format!("TRUNCATE ONLY {}", part_fqn), None, &[])
            .expect("failed to truncate compacted loose rows");
    } else {
        client
            .update(&format!("DELETE FROM ONLY {}", part_fqn), None, &[])
            .expect("failed to delete compacted loose rows");
    }
    // Every loose row visible to this compaction is now folded into a
    // segment; clear the P1 gate flag so plan/exec fast paths re-engage.
    // (A concurrent inserter's rows aren't visible here and its own commit
    // re-sets the flag.) MVCC-atomic with the row removal above.
    catalog::set_partition_has_loose_rows(client, part_info.id, false)
        .expect("failed to clear has_loose_rows flag after compaction");

    // ANALYZE the touched companion tables and refresh pg_statistic /
    // reltuples for the partition (best-effort).
    let _ = client.update(&format!("ANALYZE {}", ddl.meta_fqn), None, &[]);
    let _ = client.update(&format!("ANALYZE {}", ddl.colstats_fqn), None, &[]);
    let msg = analyze_partition_impl_split(client, &schema, &part_table);
    pgrx::debug1!("pg_deltax compact stats refresh: {}", msg);

    // Make the change visible to cached plans in every backend: TRUNCATE
    // already swaps the relfilenode (relcache inval), but the DELETE path
    // needs an explicit invalidation; sending it twice is harmless.
    let part_oid: pg_sys::Oid = client
        .select(&format!("SELECT '{}'::regclass::oid", part_fqn), None, &[])
        .ok()
        .and_then(|r| r.first().get_one::<pg_sys::Oid>().ok().flatten())
        .unwrap_or(pg_sys::InvalidOid);
    if part_oid != pg_sys::InvalidOid {
        unsafe { pg_sys::CacheInvalidateRelcacheByRelid(part_oid) };
    }
    crate::scan::invalidate_compressed_cache();

    format!(
        "Compacted {}.{}: {} loose rows into {} new segment(s)",
        schema,
        part_table,
        total_rows,
        next_segment_id - first_new_segment_id,
    )
}

/// Keep the per-segment value-presence bitmaps exact across a compaction.
///
/// For every column present in the catalog `column_valmap`:
///   - if every new segment produced a complete value set and the union of
///     old + new values still fits the bitmap budget, append the novel
///     values to the END of the valmap (existing bitmaps stay valid — the
///     scan probes positions linearly and missing trailing bits read as 0)
///     and insert a bitmap row for each new segment;
///   - otherwise DROP the column's bitmap regime entirely (delete its
///     valbitmap rows + valmap/valcounts entries). Absence is safe; an
///     incomplete valmap is not (`prune_all` would wrongly prune segments
///     containing a new value).
fn append_valbitmaps_for_compaction(
    client: &mut SpiClient,
    partition_id: i32,
    valbitmap_fqn: &str,
    columns: &[ColumnMeta],
    value_buffer: &[(u16, i32, SegValueCounts)],
    new_segment_ids: &[i32],
) {
    use std::collections::HashMap;

    // Load the current valmap / valcounts from the catalog.
    let (valmap_text, valcounts_text): (Option<String>, Option<String>) = {
        let row = client
            .select(
                "SELECT column_valmap::text, column_valcounts::text
                   FROM deltax.deltax_partition WHERE id = $1",
                Some(1),
                &[partition_id.into()],
            )
            .expect("failed to read partition valmap");
        let first = row.first();
        (first.get(1).ok().flatten(), first.get(2).ok().flatten())
    };
    let mut valmap: HashMap<String, Vec<String>> = valmap_text
        .as_deref()
        .and_then(|t| serde_json::from_str(t).ok())
        .unwrap_or_default();
    if valmap.is_empty() {
        // No bitmap regime on any column — nothing to maintain.
        return;
    }
    let mut valcounts: ColumnValcounts = valcounts_text
        .as_deref()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
        .map(|v| {
            let mut out: ColumnValcounts = HashMap::new();
            if let serde_json::Value::Object(cols) = v {
                for (name, counts) in cols {
                    if let serde_json::Value::Object(m) = counts {
                        let list: Vec<(String, i64)> = m
                            .into_iter()
                            .filter_map(|(k, c)| c.as_i64().map(|n| (k, n)))
                            .collect();
                        out.insert(name, list);
                    }
                }
            }
            out
        })
        .unwrap_or_default();

    // col_idx ↔ name mapping (non-segment-by enumeration order).
    let mut idx_to_name: HashMap<u16, &str> = HashMap::new();
    let mut name_to_idx: HashMap<&str, u16> = HashMap::new();
    {
        let mut ci: u16 = 0;
        for col in columns {
            if col.is_segment_by {
                continue;
            }
            idx_to_name.insert(ci, col.name.as_str());
            name_to_idx.insert(col.name.as_str(), ci);
            ci += 1;
        }
    }

    let mut changed = false;
    let col_names: Vec<String> = valmap.keys().cloned().collect();
    for col_name in col_names {
        let Some(&col_idx) = name_to_idx.get(col_name.as_str()) else {
            // Column no longer exists in the layout (shouldn't happen — the
            // shape guard upstream rejects drift). Drop defensively.
            drop_valbitmap_column(
                client,
                valbitmap_fqn,
                None,
                &mut valmap,
                &mut valcounts,
                &col_name,
            );
            changed = true;
            continue;
        };

        // Collect this column's per-new-segment value sets.
        let entries: Vec<&(u16, i32, SegValueCounts)> = value_buffer
            .iter()
            .filter(|(ci, _, _)| *ci == col_idx)
            .collect();

        // Completeness: every new segment must have produced a value set
        // (compute_segment_valbitmap_values omits segments whose local
        // distinct count overflowed — their values are unknown).
        if entries.len() != new_segment_ids.len() {
            drop_valbitmap_column(
                client,
                valbitmap_fqn,
                Some(col_idx),
                &mut valmap,
                &mut valcounts,
                &col_name,
            );
            changed = true;
            continue;
        }

        // Union check + novel-value collection.
        let existing = valmap.get(&col_name).cloned().unwrap_or_default();
        let existing_set: std::collections::HashSet<&str> =
            existing.iter().map(|s| s.as_str()).collect();
        let mut novel: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (_, _, vals) in &entries {
            for (v, _) in vals.iter() {
                if !existing_set.contains(v.as_str()) {
                    novel.insert(v.clone());
                }
            }
        }
        if existing.len() + novel.len() > VALBITMAP_MAX_DISTINCT {
            drop_valbitmap_column(
                client,
                valbitmap_fqn,
                Some(col_idx),
                &mut valmap,
                &mut valcounts,
                &col_name,
            );
            changed = true;
            continue;
        }

        // Append novel values (sorted among themselves) to the END of the
        // valmap — existing bitmaps stay valid.
        let mut updated = existing.clone();
        updated.extend(novel.iter().cloned());
        let bit_idx: HashMap<&str, u8> = updated
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_str(), i as u8))
            .collect();

        // Ensure the bitmap table exists (it might not when the original
        // compress produced no bitmap rows but the catalog valmap exists —
        // defensive; normally both exist together).
        let n_bytes = updated.len().div_ceil(8);
        for (_, seg_id, vals) in &entries {
            let mut bits: Vec<u8> = vec![0; n_bytes.max(1)];
            for (v, _) in vals.iter() {
                if let Some(&bi) = bit_idx.get(v.as_str()) {
                    bits[(bi / 8) as usize] |= 1u8 << (bi % 8);
                }
            }
            use pgrx::datum::DatumWithOid;
            let args: Vec<DatumWithOid> = vec![
                (col_idx as i16).into(),
                (*seg_id).into(),
                DatumWithOid::from(bits),
            ];
            client
                .update(
                    &format!(
                        "INSERT INTO {} (_col_idx, _segment_id, _bits) VALUES ($1, $2, $3)",
                        valbitmap_fqn
                    ),
                    None,
                    &args,
                )
                .expect("failed to insert compaction valbitmap row");
        }

        // Merge counts into valcounts (stats-only).
        if !novel.is_empty() || !entries.is_empty() {
            let counts = valcounts.entry(col_name.clone()).or_default();
            for (_, _, vals) in &entries {
                for (v, c) in vals.iter() {
                    if let Some(slot) = counts.iter_mut().find(|(k, _)| k == v) {
                        slot.1 += *c as i64;
                    } else {
                        counts.push((v.clone(), *c as i64));
                    }
                }
            }
        }
        if !novel.is_empty() {
            valmap.insert(col_name.clone(), updated);
        }
        changed = true;
    }

    if changed {
        catalog::update_partition_column_valmap(client, partition_id, &valmap)
            .expect("failed to update partition column_valmap");
        catalog::update_partition_column_valcounts(client, partition_id, &valcounts)
            .expect("failed to update partition column_valcounts");
    }
}

/// Remove one column's bitmap regime: its valbitmap rows (when the col_idx
/// is known and the table exists) plus the valmap/valcounts entries.
fn drop_valbitmap_column(
    client: &mut SpiClient,
    valbitmap_fqn: &str,
    col_idx: Option<u16>,
    valmap: &mut std::collections::HashMap<String, Vec<String>>,
    valcounts: &mut ColumnValcounts,
    col_name: &str,
) {
    if let Some(ci) = col_idx
        && relation_exists(client, valbitmap_fqn)
    {
        client
            .update(
                &format!("DELETE FROM {} WHERE _col_idx = $1", valbitmap_fqn),
                None,
                &[(ci as i16).into()],
            )
            .expect("failed to delete valbitmap rows");
    }
    valmap.remove(col_name);
    valcounts.remove(col_name);
}

/// Merge the compaction batch's per-column HLL sketches into the persisted
/// partition sketches and refresh `column_ndistinct`. Planner-only data —
/// every failure path silently keeps the (slightly stale) existing values.
fn merge_batch_hll_into_catalog(
    client: &mut SpiClient,
    partition_id: i32,
    columns: &[ColumnMeta],
    batch_hll: &[CardinalityEstimator<u64>],
) {
    let stored: Option<String> = client
        .select(
            "SELECT column_hll::text FROM deltax.deltax_partition WHERE id = $1",
            Some(1),
            &[partition_id.into()],
        )
        .ok()
        .and_then(|r| r.first().get_one::<String>().ok().flatten());
    let Some(stored) = stored else { return };
    let Ok(serde_json::Value::Object(map)) = serde_json::from_str::<serde_json::Value>(&stored)
    else {
        return;
    };

    let nd_col_names: Vec<&str> = columns
        .iter()
        .filter(|c| !c.is_segment_by)
        .map(|c| c.name.as_str())
        .collect();

    let mut merged: std::collections::HashMap<String, CardinalityEstimator<u64>> =
        std::collections::HashMap::new();
    for (name, val) in map {
        if let Ok(sketch) = serde_json::from_value::<CardinalityEstimator<u64>>(val) {
            merged.insert(name, sketch);
        }
    }
    let mut nd_updates: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for (name, sketch) in nd_col_names.iter().zip(batch_hll.iter()) {
        if let Some(existing) = merged.get_mut(*name) {
            existing.merge(sketch);
            nd_updates.insert((*name).to_string(), existing.estimate() as i64);
        }
    }
    if nd_updates.is_empty() {
        return;
    }

    // Refresh column_ndistinct: existing values for untouched columns,
    // merged estimates for the rest.
    let existing_nd: std::collections::HashMap<String, i64> = client
        .select(
            "SELECT column_ndistinct::text FROM deltax.deltax_partition WHERE id = $1",
            Some(1),
            &[partition_id.into()],
        )
        .ok()
        .and_then(|r| r.first().get_one::<String>().ok().flatten())
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    let mut nd_final = existing_nd;
    for (k, v) in nd_updates {
        nd_final.insert(k, v);
    }
    let _ = catalog::update_partition_column_ndistinct_from_map(client, partition_id, &nd_final);

    // Re-serialize the merged sketches.
    let names: Vec<&str> = merged.keys().map(|s| s.as_str()).collect();
    let sketches: Vec<CardinalityEstimator<u64>> = names
        .iter()
        .map(|n| merged.get(*n).cloned().unwrap())
        .collect();
    if let Some(json) = serialize_partition_hll(&names, &sketches) {
        let _ = catalog::update_partition_column_hll(client, partition_id, &json);
    }
}

/// Worker entry point: compact every compressed partition of `ht` whose heap
/// has accumulated loose rows. Returns the number of partitions compacted.
/// Each partition runs inside a subtransaction (`PgTryBuilder`) so a lock
/// timeout or transient failure on one partition doesn't abort the worker's
/// whole maintenance cycle.
pub fn auto_compact_partitions(client: &mut SpiClient<'_>, ht: &catalog::DeltatableInfo) -> i32 {
    // json_extract deltatables can't be compacted (synthetic columns have no
    // heap presence — compact_partition_impl rejects them); skip instead of
    // error-logging every worker cycle.
    if ht.json_extract.is_some() && crate::get_json_extract_mode() != crate::JsonExtractMode::None {
        return 0;
    }

    // Candidates: loose heap rows (P1 INSERTs / P2 decomposed rows) OR a
    // non-empty `_tombstones` companion (P2.5 — compaction physically
    // rewrites tombstoned segments and TRUNCATEs the table back to the
    // zero-block steady state).
    let candidates = client
        .select(
            "SELECT p.schema_name, p.table_name
               FROM deltax.deltax_partition p
              WHERE p.deltatable_id = $1
                AND p.is_compressed
                AND (COALESCE(
                        pg_catalog.pg_relation_size(
                            pg_catalog.to_regclass(
                                pg_catalog.format('%I.%I', p.schema_name, p.table_name))),
                        0) > 0
                  OR COALESCE(
                        pg_catalog.pg_relation_size(
                            pg_catalog.to_regclass(
                                pg_catalog.format('_deltax_compressed.%I',
                                                  p.table_name || '_tombstones'))),
                        0) > 0)",
            None,
            &[ht.id.into()],
        )
        .expect("failed to query compaction candidates");

    let mut targets: Vec<(String, String)> = Vec::new();
    for row in candidates {
        let sch: Option<String> = row.get(1).ok().flatten();
        let name: Option<String> = row.get(2).ok().flatten();
        if let (Some(sch), Some(name)) = (sch, name) {
            targets.push((sch, name));
        }
    }
    if targets.is_empty() {
        return 0;
    }

    // Don't let the AccessExclusive lock stall the worker behind long
    // queries; skip locked partitions until the next cycle.
    let _ = client.update("SET LOCAL lock_timeout = '5s'", None, &[]);

    let mut compacted = 0;
    for (sch, name) in &targets {
        let qualified = format!("{}.{}", sch, name);
        // `&mut SpiClient` is not UnwindSafe; PgTryBuilder runs the closure
        // in a subtransaction and rolls it back on error, which is exactly
        // the recovery the SPI stack needs — assert the boundary.
        let client_cell = std::panic::AssertUnwindSafe(&mut *client);
        let ok = pgrx::PgTryBuilder::new(move || {
            let client_cell = client_cell;
            let msg = compact_partition_impl(client_cell.0, &qualified);
            pgrx::log!("pg_deltax: {}", msg);
            true
        })
        .catch_others(|_e| {
            pgrx::log!(
                "pg_deltax: compaction of a partition was skipped (lock timeout or error); will retry next cycle"
            );
            false
        })
        .execute();
        if ok {
            compacted += 1;
        }
    }

    let _ = client.update("SET LOCAL lock_timeout = 0", None, &[]);

    compacted
}

/// Re-populate pg_class.reltuples + pg_statistic for an already-compressed
/// partition. `stats::analyze_partition_from_catalog` reads the
/// authoritative per-column distinct counts persisted at compression time
/// in `deltax.deltax_partition.column_ndistinct` (merged-HLL), so a
/// standalone refresh produces the same stats the compression path would.
pub(crate) fn analyze_partition_impl(client: &mut SpiClient, partition: &str) -> String {
    let (schema, part_table) = crate::partition::resolve_relation(client, partition);
    analyze_partition_impl_split(client, &schema, &part_table)
}

/// Same as `analyze_partition_impl` but takes (schema, table) separately.
/// Callers inside an already-open SPI connection should use this variant —
/// `resolve_relation` opens a nested `Spi::get_one_with_args` which has
/// been observed to confuse the outer connection's tuptable cursor
/// (pgrx SPI iterator returns `InvalidPosition` after the nested call
/// pops its frame).
pub(crate) fn analyze_partition_impl_split(
    client: &mut SpiClient,
    schema: &str,
    part_table: &str,
) -> String {
    let schema = schema.to_string();
    let part_table = part_table.to_string();
    let part_info = match catalog::get_partition_by_name(client, &schema, &part_table) {
        Ok(Some(p)) => p,
        Ok(None) => return format!("Partition {}.{} not found in catalog", schema, part_table),
        Err(e) => return format!("Failed to query partition: {}", e),
    };

    if !part_info.is_compressed {
        return format!(
            "Partition {}.{} is not compressed; nothing to analyze",
            schema, part_table
        );
    }

    let ht = match catalog::get_deltatable_by_id(client, part_info.deltatable_id) {
        Ok(Some(h)) => h,
        _ => {
            return format!("Failed to look up deltatable for {}.{}", schema, part_table);
        }
    };

    // ANALYZE writes to pg_statistic, which is keyed on pg_attribute attnos —
    // synthetic extracted columns have no pg_attribute entry, so they're
    // omitted here.
    let columns = get_column_metadata(
        client,
        &schema,
        &part_table,
        &ht.segment_by,
        &ht.time_column,
        None,
    );
    let part_fqn = crate::partition::fqn(&schema, &part_table);
    let ddl = build_companion_ddl(&part_table, &columns);

    let part_rel_oid: pg_sys::Oid = client
        .select(&format!("SELECT '{}'::regclass::oid", part_fqn), None, &[])
        .ok()
        .and_then(|r| r.first().get_one::<pg_sys::Oid>().ok().flatten())
        .unwrap_or(pg_sys::InvalidOid);

    let row_count: i64 = client
        .select(
            "SELECT row_count FROM deltax.deltax_partition WHERE id = $1",
            None,
            &[part_info.id.into()],
        )
        .ok()
        .and_then(|r| r.first().get_one::<i64>().ok().flatten())
        .unwrap_or(0);
    if part_rel_oid == pg_sys::InvalidOid || row_count <= 0 {
        return format!(
            "Partition {}.{} has no usable stats (row_count={})",
            schema, part_table, row_count,
        );
    }

    if let Err(e) = crate::stats::analyze_partition_from_catalog(
        client,
        part_rel_oid,
        &ddl.colstats_fqn,
        &columns,
        row_count,
    ) {
        return format!("Failed to update pg_statistic for {}: {}", part_fqn, e);
    }

    // Keep autovacuum disabled so a future ANALYZE doesn't clobber what
    // we just wrote. Safe to re-set even if already off.
    let _ = client.update(
        &format!("ALTER TABLE {} SET (autovacuum_enabled = off)", part_fqn),
        None,
        &[],
    );

    crate::scan::invalidate_compressed_cache();

    format!("Refreshed stats for {} ({} rows)", part_fqn, row_count)
}

pub(crate) fn analyze_table_impl(client: &mut SpiClient, relation: &str) -> String {
    let (schema, table) = crate::partition::resolve_relation(client, relation);
    let query = "SELECT schema_name, table_name FROM deltax.deltax_partition \
                 WHERE schema_name = $1 AND is_compressed = true AND deltatable_id = (\
                     SELECT id FROM deltax.deltax_deltatable WHERE schema_name = $1 AND table_name = $2\
                 ) \
                 ORDER BY range_start";
    let rows = match client.select(query, None, &[schema.clone().into(), table.clone().into()]) {
        Ok(r) => r,
        Err(e) => return format!("Failed to list partitions: {}", e),
    };

    let mut partitions: Vec<(String, String)> = Vec::new();
    for row in rows {
        let s: Option<String> = row
            .get_datum_by_ordinal(1)
            .ok()
            .and_then(|d| d.value().ok().flatten());
        let t: Option<String> = row
            .get_datum_by_ordinal(2)
            .ok()
            .and_then(|d| d.value().ok().flatten());
        if let (Some(s), Some(t)) = (s, t) {
            partitions.push((s, t));
        }
    }

    if partitions.is_empty() {
        return format!("No compressed partitions found for {}.{}", schema, table);
    }

    let mut n_ok = 0;
    let mut n_err = 0;
    for (s, t) in &partitions {
        // Use the split variant — invoking `analyze_partition_impl` inside
        // this loop would call `resolve_relation` → nested
        // `Spi::get_one_with_args`, which confuses the outer cursor.
        let result = analyze_partition_impl_split(client, s, t);
        if result.starts_with("Failed") || result.starts_with("Partition") {
            n_err += 1;
            pgrx::warning!("deltax_analyze_table: {}", result);
        } else {
            n_ok += 1;
        }
    }

    // Merge the per-partition stats onto the parent relation so the planner
    // has table-wide distinct counts / histograms for join and range
    // estimation (the partitions are scanned through a single DeltaXAppend).
    if let Err(e) = crate::stats::write_table_stats(client, &schema, &table) {
        pgrx::warning!(
            "deltax_analyze_table: failed to write parent stats for {}.{}: {}",
            schema,
            table,
            e
        );
    }

    format!(
        "deltax_analyze_table({}.{}): refreshed {} partition(s), {} failed",
        schema, table, n_ok, n_err,
    )
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;

    /// End-to-end P2 decompose-on-write inside the pgrx harness (this also
    /// runs against PG 18 via `make test PG_MAJOR=18`, covering the
    /// version-gated ExecutorRun hook): an UPDATE on a compressed partition
    /// decomposes only the candidate segment; an unqualified DELETE drops
    /// the remaining segments directly.
    #[pg_test]
    fn test_decompose_on_write_update_delete() {
        Spi::run("SET pg_deltax.mock_now = '2025-01-15 12:00:00+00'").unwrap();
        Spi::run("CREATE TABLE dml_e2e (ts timestamptz NOT NULL, device text NOT NULL, val int)")
            .unwrap();
        Spi::get_one::<String>(
            "SELECT deltax.deltax_create_table('dml_e2e', 'ts', '1 day'::interval)",
        )
        .unwrap();
        Spi::get_one::<String>(
            "SELECT deltax.deltax_enable_compression('dml_e2e', \
             segment_by => ARRAY['device'], order_by => ARRAY['ts'])",
        )
        .unwrap();
        // 3 devices x 50 points; disjoint val ranges per device so equality
        // quals are colstats-prunable to a single segment.
        Spi::run(
            "INSERT INTO dml_e2e \
             SELECT '2025-01-15'::timestamptz + (p || ' minutes')::interval, \
                    'dev-' || d, d * 1000 + p \
             FROM generate_series(0, 2) d, generate_series(0, 49) p",
        )
        .unwrap();

        let part: String = Spi::get_one(
            "SELECT partition_name FROM deltax.deltax_partition_info('dml_e2e') \
             WHERE range_start <= '2025-01-15'::timestamptz \
               AND range_end > '2025-01-15'::timestamptz",
        )
        .unwrap()
        .unwrap();
        let result: String = Spi::get_one(&format!(
            "SELECT deltax.deltax_compress_partition('{part}')"
        ))
        .unwrap()
        .unwrap();
        assert!(result.contains("Compressed"), "compress failed: {result}");

        let part_table = part.rsplit('.').next().unwrap();
        let meta = format!("_deltax_compressed.\"{part_table}_meta\"");
        let seg_count = || -> i64 {
            Spi::get_one(&format!(
                "SELECT count(*) FROM {meta} WHERE _segment_id > 0"
            ))
            .unwrap()
            .unwrap()
        };
        assert_eq!(seg_count(), 3, "expected one segment per device");

        // Targeted UPDATE: val = 1010 lives only in dev-1's segment.
        Spi::run("UPDATE dml_e2e SET val = -1 WHERE val = 1010").unwrap();
        assert_eq!(seg_count(), 2, "exactly one segment decomposed");
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_e2e"),
            Ok(Some(150)),
            "row count unchanged by UPDATE"
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_e2e WHERE val = -1"),
            Ok(Some(1)),
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_e2e WHERE val = 1010"),
            Ok(Some(0)),
        );
        // Sidecars went with the decomposed segment.
        assert_eq!(
            Spi::get_one::<i64>(&format!(
                "SELECT count(DISTINCT _segment_id) \
                 FROM _deltax_compressed.\"{part_table}_colstats\" WHERE _segment_id > 0"
            )),
            Ok(Some(2)),
        );

        // Unqualified DELETE drops the remaining segments + heap rows.
        Spi::run("DELETE FROM dml_e2e").unwrap();
        assert_eq!(seg_count(), 0);
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_e2e"),
            Ok(Some(0)),
        );
    }

    /// jsonb columns store binary jsonb varlena payloads in segment blobs
    /// (not UTF-8 text). Restoring rows to the heap — decompose-on-write for
    /// UPDATE and full deltax_decompress_partition() — must go through the
    /// byte-safe decode + jsonb_out path, not the text codec (which would
    /// panic with "invalid UTF-8 in LZ4 data").
    #[pg_test]
    fn test_decompose_and_decompress_roundtrip_jsonb() {
        Spi::run("SET pg_deltax.mock_now = '2025-01-15 12:00:00+00'").unwrap();
        Spi::run("CREATE TABLE dml_jsonb (ts timestamptz NOT NULL, val int, payload jsonb)")
            .unwrap();
        Spi::get_one::<String>(
            "SELECT deltax.deltax_create_table('dml_jsonb', 'ts', '1 day'::interval)",
        )
        .unwrap();
        Spi::get_one::<String>(
            "SELECT deltax.deltax_enable_compression('dml_jsonb', \
             segment_by => ARRAY[]::text[], order_by => ARRAY['ts'])",
        )
        .unwrap();
        // Distinct payloads per row defeat dictionary encoding so the blob
        // takes the Lz4Blocked path; a NULL exercises the null bitmap.
        Spi::run(
            "INSERT INTO dml_jsonb \
             SELECT '2025-01-15'::timestamptz + (p || ' minutes')::interval, p, \
                    CASE WHEN p = 7 THEN NULL \
                         ELSE jsonb_build_object('k', p, 'tag', 'v-' || p) END \
             FROM generate_series(0, 49) p",
        )
        .unwrap();

        let part: String = Spi::get_one(
            "SELECT partition_name FROM deltax.deltax_partition_info('dml_jsonb') \
             WHERE range_start <= '2025-01-15'::timestamptz \
               AND range_end > '2025-01-15'::timestamptz",
        )
        .unwrap()
        .unwrap();
        let result: String = Spi::get_one(&format!(
            "SELECT deltax.deltax_compress_partition('{part}')"
        ))
        .unwrap()
        .unwrap();
        assert!(result.contains("Compressed"), "compress failed: {result}");

        // UPDATE decomposes the segment; the restored heap rows must carry
        // the original jsonb payloads.
        Spi::run("UPDATE dml_jsonb SET val = -1 WHERE val = 10").unwrap();
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_jsonb"),
            Ok(Some(50)),
        );
        assert_eq!(
            Spi::get_one::<String>(
                "SELECT payload->>'tag' FROM dml_jsonb WHERE payload->>'k' = '10'"
            ),
            Ok(Some("v-10".to_string())),
        );
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_jsonb WHERE payload IS NULL"),
            Ok(Some(1)),
        );

        // Compact (folds the decomposed heap rows back into segments — this
        // exercises the post-INSERT jsonb compression path too), then full
        // decompress: the same restore path, all rows.
        Spi::get_one::<String>(&format!("SELECT deltax.deltax_compact_partition('{part}')"))
            .unwrap();
        Spi::get_one::<String>(&format!(
            "SELECT deltax.deltax_decompress_partition('{part}')"
        ))
        .unwrap();
        assert_eq!(
            Spi::get_one::<i64>("SELECT count(*) FROM dml_jsonb"),
            Ok(Some(50)),
        );
        assert_eq!(
            Spi::get_one::<i64>(
                "SELECT count(*) FROM dml_jsonb \
                 WHERE payload IS NOT NULL AND payload->>'tag' = 'v-' || (payload->>'k')"
            ),
            Ok(Some(49)),
        );
    }

    #[test]
    fn select_partial_mcv_keeps_heavy_hitters() {
        use std::collections::HashMap;
        // One high-card column "c": A=5000 (50%), B=3000 (30%), + 98 cold values
        // at 20 each. ndistinct=100, row_count=10000.
        let mut m: HashMap<String, i64> = HashMap::new();
        m.insert("A".to_string(), 5000);
        m.insert("B".to_string(), 3000);
        for i in 0..98 {
            m.insert(format!("cold{i}"), 20);
        }
        let topvals: TopVals = vec![m];
        let nd: HashMap<String, i64> = [("c".to_string(), 100)].into_iter().collect();
        let valmap: HashMap<String, Vec<String>> = HashMap::new();
        let out = select_partial_mcv(&topvals, &["c"], &nd, &valmap, 10000);
        let mcv = out.get("c").expect("expected a partial MCV for c");
        // Only A and B clear the 1.25/ndistinct (=1.25%) admission threshold;
        // cold values (0.2%) are dropped.
        let kept: std::collections::HashSet<&str> = mcv.iter().map(|(v, _)| v.as_str()).collect();
        assert_eq!(kept, ["A", "B"].into_iter().collect());
    }

    #[test]
    fn select_partial_mcv_skips_uniform_and_covered() {
        use std::collections::HashMap;
        // Near-uniform high-card column → no value clears the threshold → no MCV.
        let mut uniform: HashMap<String, i64> = HashMap::new();
        for i in 0..100 {
            uniform.insert(format!("v{i}"), 100);
        }
        let topvals: TopVals = vec![uniform.clone()];
        let nd: HashMap<String, i64> = [("c".to_string(), 100)].into_iter().collect();
        let empty: HashMap<String, Vec<String>> = HashMap::new();
        assert!(select_partial_mcv(&topvals, &["c"], &nd, &empty, 10000).is_empty());

        // Columns already covered by the complete valmap are skipped.
        let covered: HashMap<String, Vec<String>> = [("c".to_string(), vec!["x".to_string()])]
            .into_iter()
            .collect();
        let mut skewed: HashMap<String, i64> = HashMap::new();
        skewed.insert("A".to_string(), 9000);
        skewed.insert("B".to_string(), 1000);
        assert!(select_partial_mcv(&vec![skewed], &["c"], &nd, &covered, 10000).is_empty());
    }

    #[test]
    fn test_split_off_int64() {
        let mut col = TypedColumn::Int64(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let tail = col.split_off(3);
        assert_eq!(col, TypedColumn::Int64(vec![Some(1), Some(2), Some(3)]));
        assert_eq!(tail, TypedColumn::Int64(vec![Some(4), Some(5)]));
    }

    #[test]
    fn test_split_off_text_with_nulls() {
        let mut col = TypedColumn::Text(vec![
            Some("a".into()),
            None,
            Some("c".into()),
            Some("d".into()),
        ]);
        let tail = col.split_off(2);
        assert_eq!(col, TypedColumn::Text(vec![Some("a".into()), None]));
        assert_eq!(
            tail,
            TypedColumn::Text(vec![Some("c".into()), Some("d".into())])
        );
    }

    #[test]
    fn test_split_off_at_zero() {
        let mut col = TypedColumn::Bool(vec![Some(true), Some(false)]);
        let tail = col.split_off(0);
        assert_eq!(col, TypedColumn::Bool(vec![]));
        assert_eq!(tail, TypedColumn::Bool(vec![Some(true), Some(false)]));
    }

    #[test]
    fn test_split_off_at_end() {
        let mut col = TypedColumn::Int32(vec![Some(1), Some(2)]);
        let tail = col.split_off(2);
        assert_eq!(col, TypedColumn::Int32(vec![Some(1), Some(2)]));
        assert_eq!(tail, TypedColumn::Int32(vec![]));
    }

    #[test]
    fn test_extend_int64() {
        let mut a = TypedColumn::Int64(vec![Some(1), Some(2)]);
        let b = TypedColumn::Int64(vec![Some(3), None]);
        a.extend(b);
        assert_eq!(a, TypedColumn::Int64(vec![Some(1), Some(2), Some(3), None]));
    }

    #[test]
    fn test_extend_empty() {
        let mut a = TypedColumn::Float32(vec![]);
        let b = TypedColumn::Float32(vec![Some(1.0)]);
        a.extend(b);
        assert_eq!(a, TypedColumn::Float32(vec![Some(1.0)]));
    }

    #[test]
    #[should_panic(expected = "mismatched variants")]
    fn test_extend_mismatched() {
        let mut a = TypedColumn::Int32(vec![]);
        let b = TypedColumn::Int64(vec![]);
        a.extend(b);
    }

    #[test]
    fn test_push_from_int64() {
        let src = TypedColumn::Int64(vec![Some(10), Some(20), None]);
        let mut dst = TypedColumn::Int64(vec![]);
        dst.push_from(&src, 1);
        dst.push_from(&src, 2);
        assert_eq!(dst, TypedColumn::Int64(vec![Some(20), None]));
    }

    #[test]
    fn test_push_from_text() {
        let src = TypedColumn::Text(vec![Some("hello".into()), None, Some("world".into())]);
        let mut dst = TypedColumn::Text(vec![]);
        dst.push_from(&src, 0);
        dst.push_from(&src, 1);
        assert_eq!(dst, TypedColumn::Text(vec![Some("hello".into()), None]));
    }

    #[test]
    #[should_panic(expected = "mismatched variants")]
    fn test_push_from_mismatched() {
        let src = TypedColumn::Int32(vec![Some(1)]);
        let mut dst = TypedColumn::Int64(vec![]);
        dst.push_from(&src, 0);
    }

    #[test]
    fn test_split_off_all_variants() {
        // Ensure split_off works for every TypedColumn variant
        let mut f64_col = TypedColumn::Float64(vec![Some(1.0), Some(2.0)]);
        let tail = f64_col.split_off(1);
        assert_eq!(f64_col, TypedColumn::Float64(vec![Some(1.0)]));
        assert_eq!(tail, TypedColumn::Float64(vec![Some(2.0)]));

        let mut f32_col = TypedColumn::Float32(vec![Some(1.0), Some(2.0)]);
        let tail = f32_col.split_off(1);
        assert_eq!(f32_col, TypedColumn::Float32(vec![Some(1.0)]));
        assert_eq!(tail, TypedColumn::Float32(vec![Some(2.0)]));

        let mut i16_col = TypedColumn::Int16(vec![Some(1), Some(2)]);
        let tail = i16_col.split_off(1);
        assert_eq!(i16_col, TypedColumn::Int16(vec![Some(1)]));
        assert_eq!(tail, TypedColumn::Int16(vec![Some(2)]));
    }

    #[test]
    fn test_float64_minmax_encoding_preserves_signed_order() {
        let values = [
            f64::NEG_INFINITY,
            -100.0,
            -2.5,
            -0.0,
            0.0,
            3.25,
            100.0,
            f64::INFINITY,
        ];
        for pair in values.windows(2) {
            assert!(
                encode_f64_to_i64(pair[0]) < encode_f64_to_i64(pair[1]),
                "{} should encode below {}",
                pair[0],
                pair[1]
            );
        }
        for value in values {
            assert_eq!(
                decode_i64_to_f64(encode_f64_to_i64(value)).to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn test_float32_minmax_encoding_preserves_signed_order() {
        let values = [
            f32::NEG_INFINITY,
            -100.0,
            -2.5,
            -0.0,
            0.0,
            3.25,
            100.0,
            f32::INFINITY,
        ];
        for pair in values.windows(2) {
            assert!(
                encode_f32_to_i64(pair[0]) < encode_f32_to_i64(pair[1]),
                "{} should encode below {}",
                pair[0],
                pair[1]
            );
        }
        for value in values {
            assert_eq!(
                decode_i64_to_f32(encode_f32_to_i64(value)).to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn compute_minmax_encoded_i64_handles_each_numeric_kind() {
        // Int16 widens to i64.
        let col = TypedColumn::Int16(vec![Some(-3), None, Some(5), Some(-1)]);
        assert_eq!(
            compute_minmax_encoded_i64(&col, "smallint"),
            (Some(-3), Some(5))
        );

        // Int32 widens to i64.
        let col = TypedColumn::Int32(vec![Some(100), Some(-200), Some(0)]);
        assert_eq!(
            compute_minmax_encoded_i64(&col, "integer"),
            (Some(-200), Some(100))
        );

        // Int64 is identity-encoded (matches timestamp/date encoding too).
        let col = TypedColumn::Int64(vec![Some(1_000), Some(7_000), None]);
        assert_eq!(
            compute_minmax_encoded_i64(&col, "bigint"),
            (Some(1_000), Some(7_000))
        );

        // Float kinds use the order-preserving i64 encoding; -100 < -2.5 < 3.25.
        let col = TypedColumn::Float64(vec![Some(-100.0), Some(3.25), Some(-2.5)]);
        let (min, max) = compute_minmax_encoded_i64(&col, "double precision");
        assert_eq!(min, Some(encode_f64_to_i64(-100.0)));
        assert_eq!(max, Some(encode_f64_to_i64(3.25)));

        let col = TypedColumn::Float32(vec![Some(1.5f32), Some(-0.5)]);
        let (min, max) = compute_minmax_encoded_i64(&col, "real");
        assert_eq!(min, Some(encode_f32_to_i64(-0.5)));
        assert_eq!(max, Some(encode_f32_to_i64(1.5)));

        // All-null column yields (None, None).
        let col = TypedColumn::Int32(vec![None, None]);
        assert_eq!(compute_minmax_encoded_i64(&col, "integer"), (None, None));
    }

    #[test]
    fn compute_minmax_encoded_i64_returns_none_for_unsupported_types() {
        let col = TypedColumn::Text(vec![Some("hello".into())]);
        assert_eq!(compute_minmax_encoded_i64(&col, "text"), (None, None));
        let col = TypedColumn::Bool(vec![Some(true)]);
        assert_eq!(compute_minmax_encoded_i64(&col, "boolean"), (None, None));
        // Even though the column has integers, an unsupported data_type bails out.
        let col = TypedColumn::Int32(vec![Some(1)]);
        assert_eq!(compute_minmax_encoded_i64(&col, "uuid"), (None, None));
    }

    #[test]
    fn compute_typed_sum_integer_branches() {
        // Three non-null values, two non-zero: sum=10+(-3)+0 = 7, count=3, nonzero=2.
        let col = TypedColumn::Int32(vec![Some(10), Some(-3), Some(0), None]);
        assert_eq!(compute_typed_sum(&col), (Some("7".to_string()), 3, 2));

        // Empty (or all-null) → None.
        let col = TypedColumn::Int64(vec![None, None]);
        assert_eq!(compute_typed_sum(&col), (None, 0, 0));

        // Int16 widens to i128 — sum of three i16::MAX must not overflow.
        let col = TypedColumn::Int16(vec![Some(i16::MAX), Some(i16::MAX), Some(i16::MAX)]);
        let (sum, count, nonzero) = compute_typed_sum(&col);
        assert_eq!(sum, Some((3 * i16::MAX as i128).to_string()));
        assert_eq!((count, nonzero), (3, 3));
    }

    #[test]
    fn compute_typed_sum_float_branches() {
        let col = TypedColumn::Float64(vec![Some(1.0), Some(2.5), Some(0.0), None]);
        let (sum, count, nonzero) = compute_typed_sum(&col);
        assert!(sum.unwrap().starts_with("3."));
        assert_eq!((count, nonzero), (3, 2));

        // Float32 widens to f64 internally; result string uses {:.17e}.
        let col = TypedColumn::Float32(vec![Some(1.5)]);
        let (sum, count, nonzero) = compute_typed_sum(&col);
        assert!(sum.is_some());
        assert_eq!((count, nonzero), (1, 1));
    }

    #[test]
    fn compute_typed_sum_text_returns_char_count_sum() {
        // Char count, not byte count: "héllo" is 5 characters (é = 2 bytes).
        let col = TypedColumn::Text(vec![
            Some("héllo".into()),
            Some("".into()),
            None,
            Some("ab".into()),
        ]);
        let (sum, nonnull, nonempty) = compute_typed_sum(&col);
        assert_eq!(sum, Some("7".to_string()));
        assert_eq!(nonnull, 3); // 3 non-null rows
        assert_eq!(nonempty, 2); // 2 non-empty strings
    }

    #[test]
    fn compute_typed_sum_bool_and_bytes_have_no_sum() {
        let col = TypedColumn::Bool(vec![Some(true), Some(false)]);
        assert_eq!(compute_typed_sum(&col), (None, 0, 0));
        let col = TypedColumn::Bytes(vec![Some(vec![1, 2]), None]);
        assert_eq!(compute_typed_sum(&col), (None, 0, 0));
    }

    #[test]
    fn supports_minmax_matrix() {
        for ty in [
            "smallint",
            "int2",
            "integer",
            "int4",
            "bigint",
            "int8",
            "real",
            "float4",
            "double precision",
            "float8",
            "timestamp",
            "timestamp without time zone",
            "timestamp with time zone",
            "date",
        ] {
            assert!(supports_minmax(ty), "expected {} to support minmax", ty);
        }
        // Uppercase echoes from PG catalogs work too.
        assert!(supports_minmax("INTEGER"));
        for ty in ["text", "varchar", "boolean", "jsonb", "uuid"] {
            assert!(
                !supports_minmax(ty),
                "expected {} to NOT support minmax",
                ty
            );
        }
    }

    #[test]
    fn supports_sum_matrix() {
        // Numeric and text accepted; bool/jsonb rejected.
        for ty in [
            "integer",
            "bigint",
            "real",
            "float8",
            "text",
            "varchar(100)",
            "char(8)",
        ] {
            assert!(supports_sum(ty), "expected {} to support sum", ty);
        }
        for ty in ["boolean", "jsonb", "date", "timestamp"] {
            assert!(!supports_sum(ty), "expected {} to NOT support sum", ty);
        }
    }

    #[test]
    fn is_text_data_type_matrix() {
        for ty in [
            "text",
            "varchar",
            "varchar(100)",
            "char",
            "char(8)",
            "character",
            "character(1)",
            "character varying",
            "character varying(64)",
            "bpchar",
        ] {
            assert!(is_text_data_type(ty), "expected {} to be text", ty);
        }
        for ty in ["integer", "bigint", "boolean", "jsonb", "timestamp", "date"] {
            assert!(!is_text_data_type(ty), "expected {} to NOT be text", ty);
        }
    }

    #[test]
    fn is_valid_identifier_accepts_legal_names() {
        // Letters / underscore start; alphanumeric or underscore body.
        for ok in ["x", "_y", "ColumnName", "snake_case", "a_1", "Z9"] {
            assert!(is_valid_identifier(ok), "{} should be valid", ok);
        }
        for bad in [
            "",
            "1col",
            "-name",
            "col-name",
            "with space",
            "col$1",
            "café",
        ] {
            assert!(!is_valid_identifier(bad), "{} should be invalid", bad);
        }
    }

    #[test]
    fn is_recognized_extract_type_matrix() {
        for ok in [
            "text",
            "TEXT",
            "varchar",
            "char",
            "smallint",
            "int2",
            "integer",
            "int4",
            "bigint",
            "int8",
            "real",
            "float4",
            "double precision",
            "float8",
            "boolean",
            "bool",
            "timestamp",
            "timestamp without time zone",
            "timestamp with time zone",
            "timestamptz",
            "date",
        ] {
            assert!(
                is_recognized_extract_type(ok),
                "{} should be recognized",
                ok
            );
        }
        // Jsonb is intentionally rejected at parse time (see parse_extract_specs).
        for bad in ["jsonb", "uuid", "numeric", "interval", "money"] {
            assert!(
                !is_recognized_extract_type(bad),
                "{} should NOT be recognized",
                bad
            );
        }
    }

    #[test]
    fn classify_column_segment_by_is_text() {
        // Any column with is_segment_by = true is forced to Text so the SQL
        // literal round-trip works uniformly.
        assert!(matches!(classify_column("integer", true), ColumnKind::Text));
        assert!(matches!(
            classify_column("timestamp", true),
            ColumnKind::Text
        ));
    }

    #[test]
    fn classify_column_maps_pg_aliases() {
        assert!(matches!(
            classify_column("smallint", false),
            ColumnKind::Int16
        ));
        assert!(matches!(classify_column("int2", false), ColumnKind::Int16));
        assert!(matches!(
            classify_column("integer", false),
            ColumnKind::Int32
        ));
        assert!(matches!(classify_column("int4", false), ColumnKind::Int32));
        assert!(matches!(
            classify_column("bigint", false),
            ColumnKind::Int64
        ));
        assert!(matches!(
            classify_column("real", false),
            ColumnKind::Float32
        ));
        assert!(matches!(
            classify_column("double precision", false),
            ColumnKind::Float64
        ));
        assert!(matches!(
            classify_column("boolean", false),
            ColumnKind::Bool
        ));
        assert!(matches!(
            classify_column("timestamp", false),
            ColumnKind::Timestamp
        ));
        assert!(matches!(
            classify_column("timestamp with time zone", false),
            ColumnKind::TimestampTz
        ));
        assert!(matches!(classify_column("date", false), ColumnKind::Date));
        assert!(matches!(classify_column("jsonb", false), ColumnKind::Jsonb));
        assert!(matches!(
            classify_column("time without time zone", false),
            ColumnKind::Time
        ));
        assert!(matches!(classify_column("uuid", false), ColumnKind::Uuid));
        assert!(matches!(classify_column("bytea", false), ColumnKind::Bytea));
        assert!(matches!(classify_column("inet", false), ColumnKind::Inet));
        assert!(matches!(classify_column("cidr", false), ColumnKind::Inet));
        // Unknown types default to Text (no error — caller doesn't see this).
        assert!(matches!(
            classify_column("interval", false),
            ColumnKind::Text
        ));
        assert!(matches!(classify_column("timetz", false), ColumnKind::Text));
        // segment_by columns are always read as text for SQL literals.
        assert!(matches!(classify_column("uuid", true), ColumnKind::Text));
    }

    #[test]
    fn test_compute_lz4_clause_all_combinations() {
        // Only the (use_lz4=on AND supported) case yields the attribute.
        assert_eq!(compute_lz4_clause(true, true), " COMPRESSION lz4");
        assert_eq!(compute_lz4_clause(true, false), "");
        assert_eq!(compute_lz4_clause(false, true), "");
        assert_eq!(compute_lz4_clause(false, false), "");
    }
}
