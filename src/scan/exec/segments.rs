use pgrx::pg_guard;
use pgrx::pg_sys;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::batch_qual::{BatchCompareOp, BatchQual, LikeStrategy, sql_like_match};
use super::datum_utils::tupdesc_get_attr;
use crate::compress::{encode_f32_to_i64, encode_f64_to_i64};
use crate::compression;

/// `index_beginscan` mudou de assinatura duas vezes seguidas:
///   PG17  (heap, index, snapshot, nkeys, norderbys)
///   PG18  + `IndexScanInstrumentation *instrument` antes de nkeys
///   PG19  + `uint32 flags` no fim (bitmask de ScanOptions)
///
/// Todas as varreduras internas do deltax são de leitura comum, sem
/// instrumentação de EXPLAIN e sem dica de relação somente-leitura, então
/// passamos NULL e `SO_NONE` — o mesmo que o core passa quando não há hint.
/// Concentrar aqui evita repetir três blocos `cfg` em sete pontos de chamada.
#[inline]
unsafe fn index_beginscan_compat(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    unsafe {
        #[cfg(feature = "pg17")]
        {
            pg_sys::index_beginscan(heap, index, snapshot, nkeys, norderbys)
        }
        #[cfg(all(feature = "pg18_plus", not(feature = "pg19")))]
        {
            pg_sys::index_beginscan(
                heap,
                index,
                snapshot,
                std::ptr::null_mut(),
                nkeys,
                norderbys,
            )
        }
        #[cfg(feature = "pg19")]
        {
            pg_sys::index_beginscan(
                heap,
                index,
                snapshot,
                std::ptr::null_mut(),
                nkeys,
                norderbys,
                pg_sys::ScanOptions::SO_NONE as u32,
            )
        }
    }
}

/// Cached colstats row for a single segment: min/max/sum/counts.
struct CachedColStatsRow {
    min_encoded: i64,
    max_encoded: i64,
    min_null: bool,
    max_null: bool,
    sum_i128: Option<i128>, // Integer sums (e.g. "42000" parses to i128)
    sum_f64: Option<f64>,   // Float sums (e.g. "123.5" parses to f64 but not i128)
    sum_null: bool,
    nonnull_count: i64,
    nonzero_count: i64,
}

/// Cached colstats for a (colstats_oid, col_idx) pair: segment_id → row data.
struct CachedColStats {
    rows: HashMap<i32, CachedColStatsRow>,
}

thread_local! {
    /// Cache: (colstats_oid, col_idx) → CachedColStats.
    /// Invalidated via invalidate_colstats_cache() on compress/decompress.
    static COLSTATS_CACHE: RefCell<HashMap<(pg_sys::Oid, i16), CachedColStats>> =
        RefCell::new(HashMap::new());
}

pub(in crate::scan) fn invalidate_colstats_cache() {
    COLSTATS_CACHE.with(|c| c.borrow_mut().clear());
}

/// Which dict check to perform in `segment_skippable_by_dict`.
#[derive(Clone, Copy, PartialEq)]
enum DictCheck {
    Eq,
    Ne,
    Like,
    NotLike,
}

/// Filter for pruning segments based on min/max metadata in the normalized colstats table.
/// Built from batch quals with orderable types (int, float, timestamp, date).
pub(super) struct MinMaxFilter {
    pub(super) col_idx: i16,       // _col_idx in normalized colstats
    pub(super) op: BatchCompareOp, // Eq, Lt, Le, Gt, Ge, InList
    pub(super) const_i64: i64,     // pre-encoded constant
    pub(super) in_list_i64: Option<Vec<i64>>,
}

/// Check whether a segment might contain rows matching the filter using encoded i64 min/max.
/// Returns `true` if the segment should be kept (may match), `false` if it can be skipped.
pub(super) fn segment_passes_minmax_filter(f: &MinMaxFilter, seg_min: i64, seg_max: i64) -> bool {
    match f.op {
        BatchCompareOp::InList => {
            if let Some(ref values) = f.in_list_i64 {
                values.iter().any(|&v| v >= seg_min && v <= seg_max)
            } else {
                true
            }
        }
        _ => {
            let c = f.const_i64;
            match f.op {
                BatchCompareOp::Eq => seg_min <= c && seg_max >= c,
                BatchCompareOp::Ne => !(seg_min == c && seg_max == c),
                BatchCompareOp::Lt => seg_min < c,
                BatchCompareOp::Le => seg_min <= c,
                BatchCompareOp::Gt => seg_max > c,
                BatchCompareOp::Ge => seg_max >= c,
                _ => true, // Like, NotLike — can't prune
            }
        }
    }
}

/// Look up the per-partition btree index on `(_col_idx, _min, _max)` and
/// compute the set of segment_ids whose stored [_min, _max] range covers
/// every queried equality value. Returns `None` if the index isn't present
/// (older partition compressed before the index was added) or the table
/// can't be opened — caller falls back to the regular colstats scan.
///
/// `filters` is the list of `(col_idx, value_i64)` equality predicates,
/// already encoded with the same `encode_datum_to_i64` rule used to populate
/// `_min` / `_max` at compression time.
unsafe fn lookup_segments_by_minmax_index(
    colstats_oid: pg_sys::Oid,
    filters: &[(i16, i64)],
) -> Option<std::collections::HashSet<i32>> {
    if filters.is_empty() {
        return None;
    }
    unsafe {
        let cs_rel = pg_sys::table_open(colstats_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        // Find the btree on (_col_idx, _min, _max). Skip the PK
        // (`indisprimary == true`, on (_col_idx, _segment_id)).
        let mut minmax_idx_oid = pg_sys::InvalidOid;
        let index_list = pg_sys::RelationGetIndexList(cs_rel);
        if !index_list.is_null() {
            let n = (*index_list).length;
            for i in 0..n {
                let idx_oid = (*(*index_list).elements.add(i as usize)).oid_value;
                let idx_rel =
                    pg_sys::index_open(idx_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                let info = (*idx_rel).rd_index;
                let is_target = if !info.is_null() {
                    let is_primary = (*info).indisprimary;
                    let nkeys = (*info).indnkeyatts as usize;
                    // Read the indkey attribute numbers; key 1 = _col_idx (1),
                    // key 2 = _min (3), key 3 = _max (4) — values are 1-based
                    // attnums on the colstats table.
                    if !is_primary && nkeys >= 3 {
                        let indkey = (*info).indkey.values.as_ptr();
                        *indkey.add(0) == 1 && *indkey.add(1) == 3 && *indkey.add(2) == 4
                    } else {
                        false
                    }
                } else {
                    false
                };
                pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                if is_target {
                    minmax_idx_oid = idx_oid;
                    break;
                }
            }
            pg_sys::list_free(index_list);
        }

        if minmax_idx_oid == pg_sys::InvalidOid {
            pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            return None;
        }

        // Find the _segment_id and _max attribute positions on the heap.
        let cs_tupdesc = (*cs_rel).rd_att;
        let cs_natts = (*cs_tupdesc).natts as usize;
        let mut sid_att: Option<usize> = None;
        let mut max_att: Option<usize> = None;
        for i in 0..cs_natts {
            let att = &*tupdesc_get_attr(cs_tupdesc, i);
            if att.attisdropped {
                continue;
            }
            let name = std::ffi::CStr::from_ptr(att.attname.data.as_ptr()).to_string_lossy();
            if name == "_segment_id" {
                sid_att = Some(i);
            } else if name == "_max" {
                max_att = Some(i);
            }
        }
        let (Some(sid_att), Some(max_att)) = (sid_att, max_att) else {
            pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            return None;
        };

        let snapshot = pg_sys::GetActiveSnapshot();
        let idx_rel =
            pg_sys::index_open(minmax_idx_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let slot = pg_sys::table_slot_create(cs_rel, std::ptr::null_mut());

        // Per-filter candidate set; intersect across filters at the end.
        let mut combined: Option<std::collections::HashSet<i32>> = None;

        for &(col_idx, value) in filters {
            let mut skey = [pg_sys::ScanKeyData::default(); 2];
            // _col_idx = col_idx
            pg_sys::ScanKeyInit(
                &mut skey[0],
                1,
                pg_sys::BTEqualStrategyNumber as u16,
                pg_sys::F_INT2EQ.into(),
                pg_sys::Datum::from(col_idx),
            );
            // _min <= value (BTLessEqualStrategyNumber on attnum 2 = _min)
            pg_sys::ScanKeyInit(
                &mut skey[1],
                2,
                pg_sys::BTLessEqualStrategyNumber as u16,
                pg_sys::F_INT8LE.into(),
                pg_sys::Datum::from(value as usize),
            );

            let scan = index_beginscan_compat(cs_rel, idx_rel, snapshot, 2, 0);
            pg_sys::index_rescan(scan, skey.as_mut_ptr(), 2, std::ptr::null_mut(), 0);

            let mut this: std::collections::HashSet<i32> = std::collections::HashSet::new();
            loop {
                if !pg_sys::index_getnext_slot(
                    scan,
                    pg_sys::ScanDirection::ForwardScanDirection,
                    slot,
                ) {
                    break;
                }
                pg_sys::slot_getallattrs(slot);
                let tts_values = (*slot).tts_values;
                let tts_isnull = (*slot).tts_isnull;
                if *tts_isnull.add(sid_att) || *tts_isnull.add(max_att) {
                    continue;
                }
                // Post-filter: _max >= value.
                let max_v = (*tts_values.add(max_att)).value() as i64;
                if max_v < value {
                    continue;
                }
                let seg_id = (*tts_values.add(sid_att)).value() as i32;
                this.insert(seg_id);
            }
            pg_sys::index_endscan(scan);

            combined = Some(match combined.take() {
                None => this,
                Some(prev) => prev.intersection(&this).copied().collect(),
            });
            // Early-exit if intersection is already empty.
            if combined.as_ref().is_some_and(|s| s.is_empty()) {
                break;
            }
        }

        pg_sys::ExecDropSingleTupleTableSlot(slot);
        pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        combined
    }
}

/// Convert a batch-qual constant (raw PG datum bits) into the i64 domain the
/// bloom BUILD side hashed (`compress.rs` hashes `TypedColumn` values, which
/// store timestamps/dates as Unix-epoch microseconds and floats as raw bit
/// patterns). Probing with the raw PG-epoch datum would never match the
/// build-side hashes — every segment would be falsely bloom-rejected and
/// `col = const` on a timestamp/date column would return zero rows.
fn bloom_probe_encode(datum: pg_sys::Datum, type_oid: pg_sys::Oid) -> i64 {
    match type_oid {
        pg_sys::TIMESTAMPOID | pg_sys::TIMESTAMPTZOID => {
            // PG-epoch µs → Unix-epoch µs (matches TypedColumn::Int64 build).
            (datum.value() as i64) + crate::compress::PG_EPOCH_OFFSET_USEC
        }
        pg_sys::DATEOID => {
            // PG-epoch days (int32 datum) → Unix-epoch µs.
            ((datum.value() as i32 as i64) + crate::compress::PG_EPOCH_OFFSET_DAYS) * 86_400_000_000
        }
        // f32 bit pattern occupies the low 32 datum bits (build hashes
        // `x.to_bits() as i64`).
        pg_sys::FLOAT4OID => (datum.value() as u32) as i64,
        // Integers and f64 bit patterns are identity.
        _ => datum.value() as i64,
    }
}

/// Resolve `{partition}_<suffix>` (where the partition name is derived
/// from `meta_oid` by stripping the `_meta` suffix) to a relation OID in
/// the same namespace as `meta_oid`. Returns `InvalidOid` when the table
/// doesn't exist (e.g. data compressed before a sidecar feature shipped)
/// or `meta_oid` doesn't resolve.
///
/// Used by `fetch_segment_blobs`, `load_text_length_sidecars`, and the
/// colstats-table lookup inside `load_segments_heap`.
unsafe fn sibling_table_oid(meta_oid: pg_sys::Oid, suffix: &str) -> pg_sys::Oid {
    unsafe {
        let meta_name_ptr = pg_sys::get_rel_name(meta_oid);
        if meta_name_ptr.is_null() {
            return pg_sys::InvalidOid;
        }
        let meta_name = std::ffi::CStr::from_ptr(meta_name_ptr)
            .to_string_lossy()
            .into_owned();
        let meta_ns_oid = pg_sys::get_rel_namespace(meta_oid);
        let partition_name = meta_name.strip_suffix("_meta").unwrap_or(&meta_name);
        let sibling_name = format!("{}{}", partition_name, suffix);
        let cname = match std::ffi::CString::new(sibling_name) {
            Ok(s) => s,
            Err(_) => return pg_sys::InvalidOid,
        };
        pg_sys::get_relname_relid(cname.as_ptr(), meta_ns_oid)
    }
}

/// P2.5 tombstones: cheap conservative gate — true when the partition's
/// `_tombstones` companion exists and has at least one heap block. Mirrors
/// `relation_heap_is_empty`'s "physical truth" approach: a freshly created
/// (or `TRUNCATE`d by compaction) tombstone table has zero blocks, so the
/// steady-state cost is one syscache probe + one relcache nblocks read.
/// May report `true` for a table whose rows were all deleted but not yet
/// truncated (conservative — callers then load the exact map and find it
/// empty).
pub(crate) unsafe fn companion_may_have_tombstones(meta_oid: pg_sys::Oid) -> bool {
    unsafe {
        let tomb_oid = sibling_table_oid(meta_oid, "_tombstones");
        tomb_oid != pg_sys::InvalidOid && !crate::scan::relation_heap_is_empty(tomb_oid)
    }
}

/// Exact tombstone map for one companion: `segment_id → sorted row offsets`,
/// read with a plain heap scan under the active snapshot (the same snapshot
/// the `_meta` scan uses, so a transaction that deletes a segment's meta row
/// and its tombstone rows together is seen atomically). `None` means "no
/// tombstones visible" — the zero-cost steady state (gated by
/// `companion_may_have_tombstones`, which is one syscache + nblocks probe).
pub(crate) unsafe fn load_tombstone_map(
    meta_oid: pg_sys::Oid,
) -> Option<HashMap<i32, std::sync::Arc<Vec<u32>>>> {
    unsafe {
        let tomb_oid = sibling_table_oid(meta_oid, "_tombstones");
        if tomb_oid == pg_sys::InvalidOid {
            return None;
        }
        let rel = pg_sys::table_open(tomb_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        if pg_sys::RelationGetNumberOfBlocksInFork(rel, pg_sys::ForkNumber::MAIN_FORKNUM) == 0 {
            pg_sys::table_close(rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            return None;
        }
        let tupdesc = (*rel).rd_att;
        let snapshot = pg_sys::GetActiveSnapshot();
        let flags: u32 = pg_sys::ScanOptions::SO_TYPE_SEQSCAN
            | pg_sys::ScanOptions::SO_ALLOW_STRAT
            | pg_sys::ScanOptions::SO_ALLOW_SYNC
            | pg_sys::ScanOptions::SO_ALLOW_PAGEMODE;
        let scan = (*(*rel).rd_tableam).scan_begin.unwrap()(
            rel,
            snapshot,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            flags,
        );
        let mut raw: HashMap<i32, Vec<u32>> = HashMap::new();
        loop {
            let tuple = pg_sys::heap_getnext(scan, pg_sys::ScanDirection::ForwardScanDirection);
            if tuple.is_null() {
                break;
            }
            // Layout: (_segment_id int4, _row_offset int4) — attnos 1, 2.
            let mut isnull = false;
            let sid_datum = pg_sys::heap_getattr(tuple, 1, tupdesc, &mut isnull);
            if isnull {
                continue;
            }
            let sid = sid_datum.value() as i32;
            let off_datum = pg_sys::heap_getattr(tuple, 2, tupdesc, &mut isnull);
            if isnull {
                continue;
            }
            raw.entry(sid).or_default().push(off_datum.value() as u32);
        }
        if let Some(end) = (*(*rel).rd_tableam).scan_end {
            end(scan);
        }
        pg_sys::table_close(rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        if raw.is_empty() {
            return None;
        }
        let mut map = HashMap::with_capacity(raw.len());
        for (sid, mut offs) in raw {
            offs.sort_unstable();
            offs.dedup();
            map.insert(sid, std::sync::Arc::new(offs));
        }
        Some(map)
    }
}

/// Attach per-segment tombstone offsets to freshly loaded segments. No-op
/// (and near-zero cost) when the partition's `has_tombstones` catalog flag
/// is false — avoids opening the `_tombstones` table on the steady-state
/// read path (see `hook::DML_FLAGS`). When the flag is set, the exact map
/// is loaded under the scan's active snapshot.
pub(super) unsafe fn attach_tombstones(meta_oid: pg_sys::Oid, segments: &mut [SegmentData]) {
    if segments.is_empty() {
        return;
    }
    unsafe {
        if !crate::scan::hook::companion_has_tombstones_flag(meta_oid) {
            return;
        }
        if let Some(map) = load_tombstone_map(meta_oid) {
            for seg in segments.iter_mut() {
                seg.tombstones = map.get(&seg.segment_id).cloned();
            }
        }
    }
}

thread_local! {
    /// Backend-local cache: companion (`_meta`) table OID → partition heap
    /// OID. Reverse of `check_compressed_partition`. Cleared wholesale by
    /// `metadata_relcache_callback` like `METADATA_CACHE`.
    static PARTITION_OID_CACHE: RefCell<HashMap<pg_sys::Oid, pg_sys::Oid>> =
        RefCell::new(HashMap::new());
}

/// Resolve a companion (`_meta`) table OID back to the partition heap OID,
/// via the `deltax.deltax_partition` catalog (the partition lives in the
/// user's schema, not in `_deltax_compressed`, so `sibling_table_oid` can't
/// be used). Returns `InvalidOid` when the partition can't be found.
/// Cached per-backend; only successful lookups are cached.
pub(crate) fn partition_oid_for_companion(companion_oid: pg_sys::Oid) -> pg_sys::Oid {
    if let Some(oid) = PARTITION_OID_CACHE.with(|c| c.borrow().get(&companion_oid).copied()) {
        return oid;
    }
    ensure_metadata_callback_registered();
    let partition_name = unsafe {
        let name_ptr = pg_sys::get_rel_name(companion_oid);
        if name_ptr.is_null() {
            return pg_sys::InvalidOid;
        }
        let meta_name = std::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned();
        match meta_name.strip_suffix("_meta") {
            Some(p) => p.to_string(),
            None => return pg_sys::InvalidOid,
        }
    };
    // `is_compressed` disambiguates same-named partitions across schemas:
    // companions live in the single `_deltax_compressed` namespace, so at
    // most one partition of a given name can be compressed (and it is the
    // one this companion belongs to).
    let oid = pgrx::Spi::connect(|client| {
        client
            .select(
                "SELECT c.oid
                   FROM pg_class c
                   JOIN pg_namespace n ON n.oid = c.relnamespace
                   JOIN deltax.deltax_partition p
                     ON p.schema_name = n.nspname AND p.table_name = c.relname
                  WHERE p.table_name = $1 AND p.is_compressed",
                Some(1),
                &[partition_name.into()],
            )
            .ok()
            .and_then(|r| r.first().get_one::<pg_sys::Oid>().ok().flatten())
            .unwrap_or(pg_sys::InvalidOid)
    });
    if oid != pg_sys::InvalidOid {
        PARTITION_OID_CACHE.with(|c| c.borrow_mut().insert(companion_oid, oid));
    }
    oid
}

/// Locate the primary-key btree index on a heap relation by walking
/// `RelationGetIndexList` and matching on `indisprimary`. Returns
/// `InvalidOid` if the relation has no PK (e.g. blobs/blooms tables in
/// the middle of a direct-backfill load — PK is added in
/// `finalize_partition`).
unsafe fn primary_key_index_oid(rel: pg_sys::Relation) -> pg_sys::Oid {
    unsafe {
        let mut pk_oid = pg_sys::InvalidOid;
        let index_list = pg_sys::RelationGetIndexList(rel);
        if !index_list.is_null() {
            let n = (*index_list).length;
            for i in 0..n {
                let idx_oid = (*(*index_list).elements.add(i as usize)).oid_value;
                let idx_rel =
                    pg_sys::index_open(idx_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                let is_primary =
                    !(*idx_rel).rd_index.is_null() && (*(*idx_rel).rd_index).indisprimary;
                pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                if is_primary {
                    pk_oid = idx_oid;
                    break;
                }
            }
            pg_sys::list_free(index_list);
        }
        pk_oid
    }
}

/// Detoast a varlena pointer and copy its body into a freshly-allocated
/// `Vec<u8>`. Releases the detoasted buffer with `pfree` only when
/// `pg_detoast_datum` actually allocated (`detoasted != input`).
///
/// # Safety
/// `varlena_ptr` must point at a valid PG varlena (e.g. a `BYTEA` slot
/// datum) and the surrounding scan must hold a snapshot that pins the
/// TOAST chunks.
unsafe fn detoast_varlena_to_vec(varlena_ptr: *mut pg_sys::varlena) -> Vec<u8> {
    unsafe {
        let detoasted = pg_sys::pg_detoast_datum(varlena_ptr);
        let len = pgrx::varsize_any_exhdr(detoasted);
        let data = pgrx::vardata_any(detoasted);
        #[allow(clippy::unnecessary_cast)]
        let bytes = std::slice::from_raw_parts(data as *const u8, len).to_vec();
        if detoasted != varlena_ptr {
            pg_sys::pfree(detoasted as *mut _);
        }
        bytes
    }
}

/// Fast path for point lookups on a NOT NULL numeric column whose min/max
/// stats live in `_colstats`: return candidate segments directly from the
/// `(_col_idx, _min, _max)` btree without scanning the partition `_meta` heap.
unsafe fn lookup_point_segments_by_minmax_index(
    colstats_oid: pg_sys::Oid,
    col_idx: i16,
    value: i64,
    num_blob_cols: usize,
) -> Option<Vec<SegmentData>> {
    unsafe {
        let cs_rel = pg_sys::table_open(colstats_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        let mut minmax_idx_oid = pg_sys::InvalidOid;
        let index_list = pg_sys::RelationGetIndexList(cs_rel);
        if !index_list.is_null() {
            let n = (*index_list).length;
            for i in 0..n {
                let idx_oid = (*(*index_list).elements.add(i as usize)).oid_value;
                let idx_rel =
                    pg_sys::index_open(idx_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                let info = (*idx_rel).rd_index;
                let is_target = if !info.is_null() {
                    let is_primary = (*info).indisprimary;
                    let nkeys = (*info).indnkeyatts as usize;
                    if !is_primary && nkeys >= 3 {
                        let indkey = (*info).indkey.values.as_ptr();
                        *indkey.add(0) == 1 && *indkey.add(1) == 3 && *indkey.add(2) == 4
                    } else {
                        false
                    }
                } else {
                    false
                };
                pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                if is_target {
                    minmax_idx_oid = idx_oid;
                    break;
                }
            }
            pg_sys::list_free(index_list);
        }

        if minmax_idx_oid == pg_sys::InvalidOid {
            pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            return None;
        }

        let cs_tupdesc = (*cs_rel).rd_att;
        let cs_natts = (*cs_tupdesc).natts as usize;
        let mut sid_att: Option<usize> = None;
        let mut max_att: Option<usize> = None;
        let mut nonnull_att: Option<usize> = None;
        for i in 0..cs_natts {
            let att = &*tupdesc_get_attr(cs_tupdesc, i);
            if att.attisdropped {
                continue;
            }
            let name = std::ffi::CStr::from_ptr(att.attname.data.as_ptr()).to_string_lossy();
            match name.as_ref() {
                "_segment_id" => sid_att = Some(i),
                "_max" => max_att = Some(i),
                "_nonnull_count" => nonnull_att = Some(i),
                _ => {}
            }
        }
        let (Some(sid_att), Some(max_att), Some(nonnull_att)) = (sid_att, max_att, nonnull_att)
        else {
            pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            return None;
        };

        let snapshot = pg_sys::GetActiveSnapshot();
        let idx_rel =
            pg_sys::index_open(minmax_idx_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let slot = pg_sys::table_slot_create(cs_rel, std::ptr::null_mut());

        let mut skey = [pg_sys::ScanKeyData::default(); 2];
        pg_sys::ScanKeyInit(
            &mut skey[0],
            1,
            pg_sys::BTEqualStrategyNumber as u16,
            pg_sys::F_INT2EQ.into(),
            pg_sys::Datum::from(col_idx),
        );
        pg_sys::ScanKeyInit(
            &mut skey[1],
            2,
            pg_sys::BTLessEqualStrategyNumber as u16,
            pg_sys::F_INT8LE.into(),
            pg_sys::Datum::from(value as usize),
        );

        let scan = index_beginscan_compat(cs_rel, idx_rel, snapshot, 2, 0);
        pg_sys::index_rescan(scan, skey.as_mut_ptr(), 2, std::ptr::null_mut(), 0);

        let mut segments = Vec::new();
        loop {
            if !pg_sys::index_getnext_slot(scan, pg_sys::ScanDirection::ForwardScanDirection, slot)
            {
                break;
            }
            pg_sys::slot_getallattrs(slot);
            let tts_values = (*slot).tts_values;
            let tts_isnull = (*slot).tts_isnull;
            if *tts_isnull.add(sid_att) || *tts_isnull.add(max_att) || *tts_isnull.add(nonnull_att)
            {
                continue;
            }
            let max_v = (*tts_values.add(max_att)).value() as i64;
            if max_v < value {
                continue;
            }
            let segment_id = (*tts_values.add(sid_att)).value() as i32;
            let row_count = (*tts_values.add(nonnull_att)).value() as i32;
            let mut compressed_blobs: Vec<BlobBytes> = Vec::with_capacity(num_blob_cols);
            compressed_blobs.resize_with(num_blob_cols, BlobBytes::default);
            segments.push(SegmentData {
                companion_oid: pg_sys::InvalidOid,
                segment_id,
                segment_values: Vec::new(),
                compressed_blobs,
                text_length_blobs: vec![Vec::new(); num_blob_cols],
                row_count,
                min_time: None,
                max_time: None,
                col_minmax: HashMap::new(),
                col_sums: HashMap::new(),
                toast_pointers: vec![Vec::new(); num_blob_cols],
                cached_blob_pins: Vec::new(),
                tombstones: None,
            });
        }

        pg_sys::index_endscan(scan);
        pg_sys::ExecDropSingleTupleTableSlot(slot);
        pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        Some(segments)
    }
}

unsafe fn reltuples_as_u64(rel_oid: pg_sys::Oid) -> Option<u64> {
    unsafe {
        let tuple = pg_sys::SearchSysCache1(
            pg_sys::SysCacheIdentifier::RELOID as i32,
            pg_sys::ObjectIdGetDatum(rel_oid),
        );
        if tuple.is_null() {
            return None;
        }
        let rel_form = pg_sys::GETSTRUCT(tuple) as pg_sys::Form_pg_class;
        let reltuples = (*rel_form).reltuples;
        pg_sys::ReleaseSysCache(tuple);
        if reltuples > 0.0 {
            Some(reltuples.round() as u64)
        } else {
            None
        }
    }
}

/// Encode a pg_sys::Datum to i64 for the given type OID, matching the order-preserving
/// encoding used in the colstats table.
///
/// Timestamps and dates are stored in the colstats table as Unix-epoch microseconds
/// (matching the internal TypedColumn representation), so we must convert from PG's
/// native representation (PG-epoch) when encoding filter constants.
fn encode_datum_to_i64(datum: pg_sys::Datum, type_oid: pg_sys::Oid) -> Option<i64> {
    match type_oid {
        pg_sys::INT2OID | pg_sys::INT4OID | pg_sys::INT8OID => Some(datum.value() as i64),
        pg_sys::TIMESTAMPOID | pg_sys::TIMESTAMPTZOID => {
            // PG stores as PG-epoch microseconds; colstats stores as Unix-epoch microseconds
            let pg_epoch_usec = datum.value() as i64;
            Some(pg_epoch_usec + crate::compress::PG_EPOCH_OFFSET_USEC)
        }
        pg_sys::DATEOID => {
            // PG stores as PG-epoch days (int32); colstats stores as Unix-epoch microseconds
            let pg_epoch_days = datum.value() as i32 as i64;
            Some((pg_epoch_days + crate::compress::PG_EPOCH_OFFSET_DAYS) * 86_400_000_000)
        }
        pg_sys::TIMEOID => {
            // TimeADT is microseconds since midnight; colstats stores the same
            // (no epoch offset), so identity encoding.
            Some(datum.value() as i64)
        }
        pg_sys::FLOAT4OID => {
            let v = f32::from_bits(datum.value() as u32);
            Some(encode_f32_to_i64(v))
        }
        pg_sys::FLOAT8OID => {
            let v = f64::from_bits(datum.value() as u64);
            Some(encode_f64_to_i64(v))
        }
        _ => None,
    }
}

/// Returns Some(true) if all rows provably satisfy the qual,
/// Some(false) if no rows satisfy (already pruned by load_segments_heap),
/// None if ambiguous (must decompress).
pub(super) fn segment_all_rows_pass(
    cm: &ColMinMax,
    op: BatchCompareOp,
    const_datum: pg_sys::Datum,
) -> Option<bool> {
    if cm.min_null || cm.max_null {
        return None;
    }

    // Encode the constant datum to i64 for comparison with stored encoded values
    let c = encode_datum_to_i64(const_datum, cm.type_oid)?;
    let seg_min = cm.min_encoded;
    let seg_max = cm.max_encoded;

    match op {
        BatchCompareOp::Eq => {
            if seg_min == c && seg_max == c {
                Some(true)
            } else if seg_max < c || seg_min > c {
                Some(false)
            } else {
                None
            }
        }
        BatchCompareOp::Ne => {
            if seg_min > c || seg_max < c {
                Some(true)
            } else if seg_min == c && seg_max == c {
                Some(false)
            } else {
                None
            }
        }
        BatchCompareOp::Gt => {
            if seg_min > c {
                Some(true)
            } else if seg_max <= c {
                Some(false)
            } else {
                None
            }
        }
        BatchCompareOp::Ge => {
            if seg_min >= c {
                Some(true)
            } else if seg_max < c {
                Some(false)
            } else {
                None
            }
        }
        BatchCompareOp::Lt => {
            if seg_max < c {
                Some(true)
            } else if seg_min >= c {
                Some(false)
            } else {
                None
            }
        }
        BatchCompareOp::Le => {
            if seg_max <= c {
                Some(true)
            } else if seg_min > c {
                Some(false)
            } else {
                None
            }
        }
        BatchCompareOp::InList | BatchCompareOp::Like | BatchCompareOp::NotLike => None,
    }
}

/// Result of classifying whether all rows in a segment satisfy all quals.
pub(super) enum SegmentQualResult {
    /// Metadata proves all rows satisfy all quals and no NULLs in qual columns.
    AllPass,
    /// Metadata proves NO rows satisfy the quals (e.g. nonzero_count == 0 with Ne 0).
    NonePass,
    /// Cannot determine from metadata — must decompress.
    Ambiguous,
}

/// Returns true if the datum is zero for the given numeric type OID.
pub(super) fn is_zero_const(datum: pg_sys::Datum, type_oid: pg_sys::Oid) -> bool {
    match type_oid {
        pg_sys::INT2OID => datum.value() as i16 == 0,
        pg_sys::INT4OID => datum.value() as i32 == 0,
        pg_sys::INT8OID => datum.value() as i64 == 0,
        pg_sys::FLOAT4OID => f32::from_bits(datum.value() as u32) == 0.0,
        pg_sys::FLOAT8OID => f64::from_bits(datum.value() as u64) == 0.0,
        _ => false,
    }
}

/// Classify a segment by **only** the numeric subset of `batch_quals` —
/// useful in the mixed text+numeric path where text quals have no
/// `col_minmax` metadata to consult. Returns `Ambiguous` if no numeric
/// quals are present (caller should fall through to per-row eval).
///
/// `NonePass` is sound: a numeric qual that rejects every row in the
/// segment also rejects the same rows under any text qual, so the
/// segment can be skipped entirely. `AllPass` here means the numeric
/// quals pass for every row — text quals may still filter; the caller
/// uses this to skip the per-row numeric `evaluate_batch_quals` step
/// while keeping the text qual application.
pub(super) fn classify_segment_quals_numeric(
    seg: &SegmentData,
    batch_quals: &[BatchQual],
    col_names: &[String],
) -> SegmentQualResult {
    use super::batch_qual::is_batch_comparable_type;
    let mut any_numeric = false;
    let mut any_nonepass = false;
    let mut any_ambiguous = false;
    for bq in batch_quals {
        if !is_batch_comparable_type(bq.type_oid) {
            continue; // skip text quals — handled per-row by caller
        }
        any_numeric = true;
        let col_name = &col_names[bq.col_idx];
        let cm = match seg.col_minmax.get(col_name) {
            Some(cm) => cm,
            None => {
                any_ambiguous = true;
                continue;
            }
        };
        match segment_all_rows_pass(cm, bq.op, bq.const_datum) {
            Some(true) => {}
            Some(false) => {
                any_nonepass = true;
            }
            None => {
                any_ambiguous = true;
            }
        }
    }
    if !any_numeric {
        return SegmentQualResult::Ambiguous;
    }
    if any_nonepass {
        return SegmentQualResult::NonePass;
    }
    if any_ambiguous {
        return SegmentQualResult::Ambiguous;
    }
    // All numeric quals pass via minmax. Check NULLs in the numeric
    // qual columns: minmax covers only non-NULL values.
    for bq in batch_quals {
        if !is_batch_comparable_type(bq.type_oid) {
            continue;
        }
        let col_name = &col_names[bq.col_idx];
        match seg.col_sums.get(col_name) {
            Some(cs) => {
                if cs.nonnull_count < seg.row_count as i64 {
                    return SegmentQualResult::Ambiguous;
                }
            }
            None => return SegmentQualResult::Ambiguous,
        }
    }
    SegmentQualResult::AllPass
}

/// Classify a segment: can we prove all rows pass all batch quals using metadata?
///
/// `time_column` is the partition/time column name: it is non-null by
/// construction in a compressed (non-default) partition — NULL partition
/// keys route to the default partition — so its NULL-safety check below is
/// satisfied without a `col_sums` entry. Compression writes the time
/// column's colstats row with a ZERO `_nonnull_count` placeholder (its
/// authoritative min/max live in `_meta`, populated into `col_minmax`
/// separately), so consulting `col_sums` for it would spuriously read 0
/// non-null rows and return `Ambiguous`, defeating whole-segment DELETE and
/// the metadata fast paths on time-range predicates.
pub(super) fn classify_segment_quals(
    seg: &SegmentData,
    batch_quals: &[BatchQual],
    col_names: &[String],
    time_column: &str,
) -> SegmentQualResult {
    let mut any_nonepass = false;
    for bq in batch_quals {
        let col_name = &col_names[bq.col_idx];
        let cm = match seg.col_minmax.get(col_name) {
            Some(cm) => cm,
            None => return SegmentQualResult::Ambiguous,
        };
        match segment_all_rows_pass(cm, bq.op, bq.const_datum) {
            Some(true) => {} // this qual is satisfied for all rows
            Some(false) => return SegmentQualResult::NonePass,
            None => {
                // minmax couldn't resolve — try nonzero_count for Ne/Eq with 0
                if is_zero_const(bq.const_datum, bq.type_oid)
                    && let Some(cs) = seg.col_sums.get(col_name)
                    && cs.nonzero_count >= 0
                    && cs.nonnull_count == seg.row_count as i64
                {
                    match bq.op {
                        BatchCompareOp::Ne if cs.nonzero_count == 0 => {
                            // All values are zero → Ne 0 passes for no rows
                            any_nonepass = true;
                            continue;
                        }
                        BatchCompareOp::Eq if cs.nonzero_count == cs.nonnull_count => {
                            // All values are nonzero → Eq 0 passes for no rows
                            any_nonepass = true;
                            continue;
                        }
                        _ => {}
                    }
                }
                return SegmentQualResult::Ambiguous;
            }
        }
    }
    if any_nonepass {
        return SegmentQualResult::NonePass;
    }
    // All quals passed via minmax. Now check for NULLs in qual columns:
    // min/max covers only non-NULL values, so if NULLs exist, we can't trust row_count.
    for bq in batch_quals {
        let col_name = &col_names[bq.col_idx];
        // The time column is non-null by construction here (see fn doc); its
        // colstats nonnull_count is a zero placeholder, so skip the check.
        if col_name == time_column {
            continue;
        }
        match seg.col_sums.get(col_name) {
            Some(cs) => {
                if cs.nonnull_count < seg.row_count as i64 {
                    return SegmentQualResult::Ambiguous;
                }
            }
            None => return SegmentQualResult::Ambiguous,
        }
    }
    SegmentQualResult::AllPass
}

/// Per-column min/max metadata from the companion table, stored as order-preserving i64 encodings.
pub(super) struct ColMinMax {
    pub(super) min_encoded: i64,
    pub(super) max_encoded: i64,
    pub(super) min_null: bool,
    pub(super) max_null: bool,
    pub(super) type_oid: pg_sys::Oid,
}

/// Per-column sum metadata from the companion table.
pub(super) struct ColSum {
    pub(super) sum_datum: pg_sys::Datum,
    pub(super) sum_null: bool,
    pub(super) sum_i128: Option<i128>, // Cached/pre-converted integer sum
    pub(super) sum_f64: Option<f64>,   // Cached/pre-converted float sum (when i128 parse fails)
    pub(super) nonnull_count: i64,
    pub(super) nonzero_count: i64, // -1 = unavailable (column missing in older meta tables)
    pub(super) type_oid: pg_sys::Oid, // NUMERICOID or FLOAT8OID
}

/// Check whether a segment can be skipped based on dictionary pruning for LIKE quals.
///
/// For each LIKE/NOT LIKE batch qual, finds the corresponding compressed blob and
/// checks if it's dictionary-encoded. If so, tests dictionary entries against the
/// Check whether a segment can be skipped based on dictionary pruning for text quals.
///
/// For each LIKE/NOT LIKE/Eq/Ne batch qual on dict-encoded text columns, finds the
/// corresponding compressed blob and checks dictionary entries:
/// - **Like**: skip if NO dict entry matches the pattern (no row can match)
/// - **NotLike**: skip if ALL dict entries match the pattern (every row is filtered)
/// - **Eq**: skip if NO dict entry equals the constant (no row can match)
/// - **Ne**: skip if ALL dict entries equal the constant (every row is filtered)
///
/// Returns `true` if the segment should be skipped.
pub(super) fn segment_skippable_by_dict(
    batch_quals: &[BatchQual],
    blob_idx_map: &[Option<u16>],
    compressed_blobs: &[BlobBytes],
) -> bool {
    for bq in batch_quals {
        // Determine which operation we're checking
        let check = match (&bq.op, &bq.like_strategy) {
            (BatchCompareOp::Like, Some(_)) => DictCheck::Like,
            (BatchCompareOp::NotLike, Some(_)) => DictCheck::NotLike,
            (BatchCompareOp::Eq, _) if bq.text_const.is_some() => DictCheck::Eq,
            (BatchCompareOp::Ne, _) if bq.text_const.is_some() => DictCheck::Ne,
            _ => continue,
        };

        // Look up the persisted `_col_idx` for the queried column. None
        // means either segment_by (no blob to check) or ADD-COLUMN-after-
        // compression (no blob exists, so dict pruning can't help — fall
        // through to qual eval which handles the synthesized value).
        let Some(blob_idx) = blob_idx_map.get(bq.col_idx).copied().flatten() else {
            continue;
        };
        let blob_idx = blob_idx as usize;

        let blob = &compressed_blobs[blob_idx];
        if blob.len() < 6 {
            continue;
        }

        // Check if dictionary-encoded
        let type_tag = compression::CompressionType::from_u8(blob[0]);
        let is_dict = matches!(
            type_tag,
            compression::CompressionType::Dictionary | compression::CompressionType::DictionaryLz4
        );
        if !is_dict {
            continue;
        }

        // Parse the compressed column header to get the data portion
        let cc = compression::CompressedColumnRef::from_bytes(blob);

        // Ne can only prune a segment whose dictionary is a single entry
        // equal to the constant. dict_size lives in the uncompressed header
        // prefix (same layout for Dictionary and DictionaryLz4), so read it
        // directly instead of decompressing every segment's dictionary just
        // to conclude "can't prune" (this decompression on the scan leader
        // dominated ClickBench Q25's `SearchPhrase <> ''`).
        if check == DictCheck::Ne {
            if cc.data.len() < 4 {
                continue;
            }
            let dict_size = u32::from_le_bytes(cc.data[0..4].try_into().unwrap());
            if dict_size > 1 {
                continue;
            }
        }

        // Normalize DictionaryLz4 → Dictionary format for header parsing
        let norm_buf;
        let dict_data = if type_tag == compression::CompressionType::DictionaryLz4 {
            norm_buf = compression::dictionary::normalize_lz4(cc.data);
            &norm_buf[..]
        } else {
            cc.data
        };

        // Check dictionary entries against the predicate
        let any_match = compression::dictionary::any_entry_matches(dict_data, |text| match check {
            DictCheck::Eq => text == bq.text_const.as_ref().unwrap().as_str(),
            DictCheck::Ne => text != bq.text_const.as_ref().unwrap().as_str(),
            DictCheck::Like | DictCheck::NotLike => {
                let strategy = bq.like_strategy.as_ref().unwrap();
                let matched = match strategy {
                    LikeStrategy::Contains(s) => text.contains(s.as_str()),
                    LikeStrategy::StartsWith(s) => text.starts_with(s.as_str()),
                    LikeStrategy::EndsWith(s) => text.ends_with(s.as_str()),
                    LikeStrategy::Exact(s) => text == s.as_str(),
                    LikeStrategy::General(p) => sql_like_match(text, p),
                };
                if check == DictCheck::NotLike {
                    !matched
                } else {
                    matched
                }
            }
        });

        if !any_match {
            return true; // No rows can match — skip segment
        }
    }

    false
}

/// One per-column compressed blob stored in `SegmentData`. Lets cache
/// hits skip the `to_vec()` copy: instead of materialising the cached
/// bytes into a backend-heap `Vec<u8>`, `Cached` keeps a raw pointer into
/// the DSA-backed `BlobCachePin` allocation. The corresponding pin lives
/// in `SegmentData::cached_blob_pins`, which is declared AFTER
/// `compressed_blobs` so Rust drops `compressed_blobs` first — the raw
/// pointers go out of scope before the pins release the entry.
///
/// `Deref<Target = [u8]>` so existing consumer code that takes `&[u8]`
/// keeps working without changes.
pub(crate) enum BlobBytes {
    Owned(Vec<u8>),
    /// Borrowed bytes from the blob cache. Valid for the lifetime of
    /// the surrounding `SegmentData` (i.e. until the matching pin in
    /// `cached_blob_pins` drops).
    Cached {
        data: *const u8,
        len: u32,
    },
}

impl Default for BlobBytes {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}

impl std::ops::Deref for BlobBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            BlobBytes::Owned(v) => v.as_slice(),
            BlobBytes::Cached { data, len } => unsafe {
                std::slice::from_raw_parts(*data, *len as usize)
            },
        }
    }
}

// SAFETY: Cached's raw pointer references DSA shared memory whose
// lifetime is guaranteed by the matching `BlobCachePin` in the same
// `SegmentData`. The pin uses atomic pin_count to keep the entry
// resident across all readers and prevents eviction. The bytes are
// only ever read, never written.
unsafe impl Send for BlobBytes {}
unsafe impl Sync for BlobBytes {}

pub(in crate::scan) struct SegmentData {
    /// Source companion-table OID. Populated by the caller after
    /// `load_segments_heap` returns; used by `fetch_segment_blobs` to re-open
    /// the right `_blobs` table when blobs are materialised on-claim.
    pub(super) companion_oid: pg_sys::Oid,
    /// Companion-table segment id (used to fetch sidecar/bloom data after
    /// the main load).
    pub(super) segment_id: i32,
    pub(super) segment_values: Vec<Option<String>>,
    pub(super) compressed_blobs: Vec<BlobBytes>,
    /// Per-text-column length sidecar blobs (parallel to compressed_blobs).
    /// Non-empty when the planner has marked a text column as sidecar-only;
    /// holds the compressed u32-per-row length array instead of the main blob.
    pub(super) text_length_blobs: Vec<Vec<u8>>,
    pub(super) row_count: i32,
    pub(super) min_time: Option<i64>,
    pub(super) max_time: Option<i64>,
    /// Per-column min/max (column name → ColMinMax).
    pub(super) col_minmax: HashMap<String, ColMinMax>,
    /// Per-column sum metadata (column name → ColSum).
    pub(super) col_sums: HashMap<String, ColSum>,
    /// Deferred TOAST pointer copies for lazy detoasting (Top-N only).
    /// Parallel to compressed_blobs: non-empty means "not yet detoasted, call
    /// detoast_lazy_blobs() to materialize". Empty means already detoasted or
    /// not needed.
    pub(super) toast_pointers: Vec<Vec<u8>>,
    /// Pins for blobs served from the shared blob cache. Holding these pins
    /// guarantees the underlying DSA-backed bytes outlive every read of
    /// `compressed_blobs` (including parallel-worker reads, since detoast
    /// runs on the leader before worker dispatch and segments are owned by
    /// the leader's `DecompressState`). Released automatically on drop.
    pub(super) cached_blob_pins: Vec<crate::blob_cache::BlobCachePin>,
    /// P2.5 tombstones: sorted 0-based row offsets that are logically
    /// deleted from this segment (rows in the per-partition `_tombstones`
    /// companion visible under the scan snapshot). `None` = no tombstones —
    /// the steady state. Decode paths AND these into the selection vector;
    /// metadata-only consumers must use `live_row_count()` or bail.
    pub(super) tombstones: Option<std::sync::Arc<Vec<u32>>>,
}

impl SegmentData {
    /// Logical (live) row count: physical `row_count` minus tombstoned rows.
    pub(super) fn live_row_count(&self) -> i32 {
        match &self.tombstones {
            Some(t) => self.row_count - (t.len() as i32),
            None => self.row_count,
        }
    }

    /// Build a seed selection vector of length `n` (`n` may be truncated
    /// below `row_count` by Top-N cutoffs) that is `true` everywhere except
    /// at tombstoned offsets. `None` when the segment has no tombstones —
    /// callers keep their existing "no pre-selection" fast path.
    pub(super) fn tombstone_preselection(&self, n: usize) -> Option<Vec<bool>> {
        let tombs = self.tombstones.as_ref()?;
        let mut sel = vec![true; n];
        for &off in tombs.iter() {
            if (off as usize) < n {
                sel[off as usize] = false;
            }
        }
        Some(sel)
    }
}

// SAFETY: SegmentData is shared across threads only via immutable references
// during parallel aggregation. The pg_sys::Datum fields in ColMinMax/ColSum
// are not accessed on worker threads (only compressed_blobs, segment_values,
// row_count, and time bounds are used). All accessed fields are safe Rust types.
unsafe impl Send for SegmentData {}
unsafe impl Sync for SegmentData {}

/// Metadata returned by the SPI metadata query.
#[derive(Clone)]
pub(super) struct MetadataInfo {
    pub(super) col_names: Vec<String>,
    pub(super) col_types: Vec<pg_sys::Oid>,
    pub(super) col_typmods: Vec<i32>,
    pub(super) col_not_null: Vec<bool>,
    pub(super) segment_by: Vec<String>,
    pub(super) order_by: Vec<String>,
    pub(super) time_column: String,
    /// Parallel to `col_names`. `Some(i)` = read this column's data from
    /// `_blobs`/`_colstats`/etc. at `_col_idx = i`. `None` means either
    /// the column is `segment_by` (lives in `_meta`, not in the blob path)
    /// or the column was added to the parent after this partition was
    /// compressed (decompress synthesizes the value via
    /// `pg_sys::getmissingattr`). For legacy partitions whose
    /// `compressed_columns` descriptor is NULL, this is identical to the
    /// positional mapping the scan path used to compute locally.
    /// See `dev/docs/SCHEMA_CHANGES.md`.
    pub(super) blob_idx: Vec<Option<u16>>,
    /// Parallel to `col_names`. Current `pg_attribute.attnum` for physical
    /// columns, or `0` for json-extract synthetic columns (which have no
    /// pg_attribute row). Used inside `load_metadata` to call
    /// `pg_sys::getmissingattr` for columns added after compression;
    /// kept on `MetadataInfo` for future descriptor-shape resolution
    /// that needs to join by attnum rather than name.
    pub(super) attnums: Vec<i32>,
    /// Parallel to `col_names`. `Some((datum, is_null))` = pre-computed
    /// missing value via `pg_sys::getmissingattr` for a column that was
    /// added to the parent after this partition was compressed. `None`
    /// means either the column has a blob OR is segment_by (read from
    /// `_meta`). Decompress overlays these onto the output slot.
    /// SAFETY: For pass-by-reference types the Datum points into the
    /// partition relation's tupdesc and is only valid while the relation
    /// remains open (PG's relcache keeps it live for the query duration).
    pub(super) missing_values: Vec<Option<(pg_sys::Datum, bool)>>,
}

impl MetadataInfo {
    /// Number of PHYSICAL columns (those present in the partition heap).
    /// json-extract synthetic columns are appended to `col_names` with an
    /// `attnum` of 0 (no pg_attribute row); every real column — including
    /// segment_by — has a positive attnum. The partition heap's `natts`
    /// equals this count, so heap-tail layout guards compare against it
    /// rather than `col_names.len()` (which over-counts by the synthetic
    /// columns and would false-fire on any json_extract table with loose
    /// rows).
    pub(super) fn physical_col_count(&self) -> usize {
        self.attnums.iter().filter(|&&a| a != 0).count()
    }
}

// SAFETY: `MetadataInfo` is shared across threads only during parallel
// aggregation (see `agg::metadata::accumulate_segment_decompressed`), and
// only via immutable reference. The `Datum` values in `missing_values`
// point into the partition's relcache descriptor whose lifetime exceeds
// the scoped-thread join boundary (relcache entries pinned for the
// query duration). No worker writes to these fields.
unsafe impl Send for MetadataInfo {}
unsafe impl Sync for MetadataInfo {}

thread_local! {
    /// Backend-local cache: companion (`_meta`) table OID → MetadataInfo.
    /// Populated lazily by `load_metadata_cached`. Cleared wholesale on any
    /// relcache invalidation (see `metadata_relcache_callback`) — that
    /// covers ALTER TABLE on the parent (which invalidates the Datums in
    /// `missing_values`) plus partition compress/decompress.
    static METADATA_CACHE: RefCell<HashMap<pg_sys::Oid, MetadataInfo>> =
        RefCell::new(HashMap::new());
    /// Tracks whether we've registered the relcache callback yet in this
    /// backend. PG's `CacheRegisterRelcacheCallback` is one-shot — register
    /// once on the first cache miss.
    static METADATA_CB_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

#[pg_guard]
unsafe extern "C-unwind" fn metadata_relcache_callback(_arg: pg_sys::Datum, _relid: pg_sys::Oid) {
    // Conservative: wipe the whole cache on any relcache invalidation.
    // The cache is tiny (≤ #partitions per backend) so re-populating is cheap,
    // and this avoids tracking dependencies between MetadataInfo entries and
    // every catalog row they read (parent table pg_attribute, deltax catalog).
    METADATA_CACHE.with(|c| c.borrow_mut().clear());
    PARTITION_OID_CACHE.with(|c| c.borrow_mut().clear());
}

fn ensure_metadata_callback_registered() {
    METADATA_CB_REGISTERED.with(|c| {
        if !c.get() {
            unsafe {
                pg_sys::CacheRegisterRelcacheCallback(
                    Some(metadata_relcache_callback),
                    pg_sys::Datum::from(0u32),
                );
            }
            c.set(true);
        }
    });
}

/// Cached variant of `load_metadata` keyed on the companion (`_meta`) table
/// OID. On miss, derives the companion name and runs the SPI queries; on
/// hit, returns a clone of the cached `MetadataInfo` (which all 5 executor
/// call sites consume by value).
pub(super) fn load_metadata_cached(companion_oid: pg_sys::Oid) -> MetadataInfo {
    if let Some(cached) = METADATA_CACHE.with(|c| c.borrow().get(&companion_oid).cloned()) {
        return cached;
    }
    ensure_metadata_callback_registered();
    let companion_name = unsafe {
        let name_ptr = pg_sys::get_rel_name(companion_oid);
        if name_ptr.is_null() {
            pgrx::error!(
                "pg_deltax: companion table not found for OID {}",
                u32::from(companion_oid)
            );
        }
        std::ffi::CStr::from_ptr(name_ptr)
            .to_string_lossy()
            .into_owned()
    };
    let meta = pgrx::Spi::connect(|client| load_metadata(client, &companion_name));
    METADATA_CACHE.with(|c| c.borrow_mut().insert(companion_oid, meta.clone()));
    meta
}

/// Load metadata (column names, types, segment_by) from catalog via SPI.
/// `companion_name` is the meta table name (e.g. "<partition>_meta"). The `_meta`
/// suffix is stripped to find the partition in the catalog.
pub(super) fn load_metadata(
    client: &pgrx::spi::SpiClient<'_>,
    companion_name: &str,
) -> MetadataInfo {
    // Strip _meta suffix to get the partition name for catalog lookup
    let partition_name = companion_name
        .strip_suffix("_meta")
        .unwrap_or(companion_name);

    // Get the partition's deltatable info
    let mut ht_result = client
        .select(
            "SELECT h.segment_by, h.order_by, h.time_column, h.schema_name, h.table_name,
                    h.json_extract, p.compressed_columns
             FROM deltax.deltax_partition p
             JOIN deltax.deltax_deltatable h ON h.id = p.deltatable_id
             WHERE p.table_name = $1 AND p.is_compressed = true",
            None,
            &[partition_name.into()],
        )
        .expect("failed to query partition info");

    let ht_row = ht_result.next().unwrap_or_else(|| {
        pgrx::error!(
            "pg_deltax: no compressed partition info found for {}",
            companion_name
        );
    });

    let segment_by: Vec<String> = ht_row
        .get_datum_by_ordinal(1)
        .unwrap()
        .value::<Vec<String>>()
        .unwrap()
        .unwrap_or_default();
    let order_by: Vec<String> = ht_row
        .get_datum_by_ordinal(2)
        .unwrap()
        .value::<Vec<String>>()
        .unwrap()
        .unwrap_or_default();
    let time_column: String = ht_row
        .get_datum_by_ordinal(3)
        .unwrap()
        .value::<String>()
        .unwrap()
        .unwrap();
    let parent_schema: String = ht_row
        .get_datum_by_ordinal(4)
        .unwrap()
        .value::<String>()
        .unwrap()
        .unwrap();
    let parent_table: String = ht_row
        .get_datum_by_ordinal(5)
        .unwrap()
        .value::<String>()
        .unwrap()
        .unwrap();
    let json_extract: Option<serde_json::Value> = ht_row
        .get_datum_by_ordinal(6)
        .unwrap()
        .value::<pgrx::datum::JsonB>()
        .unwrap()
        .map(|j| j.0);
    let compressed_columns: Option<serde_json::Value> = ht_row
        .get_datum_by_ordinal(7)
        .unwrap()
        .value::<pgrx::datum::JsonB>()
        .unwrap()
        .map(|j| j.0);

    // Get column info via a direct relcache walk on the parent table —
    // avoids the 4-table `pg_attribute ⋈ pg_type ⋈ pg_class ⋈ pg_namespace`
    // SPI + name→OID translation that this used to do (~0.8 ms saved per
    // miss on the fresh-connection path the bench measures). `attnum` is
    // returned so the decompress path can call `pg_sys::getmissingattr` for
    // columns added after compression (see SCHEMA_CHANGES.md).
    let ParentColumnLayout {
        names: mut col_names,
        types: mut col_types,
        typmods: mut col_typmods,
        not_null: mut col_not_null,
        attnums: mut col_attnums,
    } = load_parent_columns(&parent_schema, &parent_table);
    // Descriptor-compat: `compressed_columns.type_oid` was persisted as i64.
    let col_atttypids: Vec<i64> = col_types.iter().map(|oid| u32::from(*oid) as i64).collect();

    // Resolve each physical column's `_col_idx` against the persisted
    // descriptor. None descriptor → legacy positional mapping (today's
    // behavior): non-segment_by columns counted in attnum order starting
    // at 0. Present descriptor → look up by attnum and use the historical
    // `compressed_col_idx`. Mismatch on type/typmod is a defensive error.
    let mut blob_idx: Vec<Option<u16>> = Vec::with_capacity(col_names.len());
    let descriptor_entries: Option<Vec<DescriptorEntry>> = compressed_columns
        .as_ref()
        .and_then(parse_compressed_columns);
    if let Some(entries) = descriptor_entries.as_ref() {
        for i in 0..col_names.len() {
            let attnum = col_attnums[i];
            match entries.iter().find(|e| e.attnum == attnum) {
                Some(entry) => {
                    if entry.dropped {
                        blob_idx.push(None);
                        continue;
                    }
                    if entry.type_oid != col_atttypids[i] || entry.typmod != col_typmods[i] {
                        pgrx::error!(
                            "pg_deltax: column {:?} on partition {} has type/typmod \
                             ({}, {}) that differs from compressed snapshot ({}, {}); \
                             decompress and recompress the partition to apply the change",
                            col_names[i],
                            companion_name,
                            col_atttypids[i],
                            col_typmods[i],
                            entry.type_oid,
                            entry.typmod,
                        );
                    }
                    blob_idx.push(entry.compressed_col_idx);
                }
                None => {
                    // Column added after this partition was compressed —
                    // decompress synthesizes via getmissingattr.
                    blob_idx.push(None);
                }
            }
        }
    } else {
        // Legacy partition (no descriptor): identity mapping from
        // segment_by names, identical to the historical local
        // computation in run_segments_scan.
        let mut next_idx: u16 = 0;
        for name in &col_names {
            if segment_by.contains(name) {
                blob_idx.push(None);
            } else {
                blob_idx.push(Some(next_idx));
                next_idx += 1;
            }
        }
    }

    // Append synthetic columns from json_extract (in spec order). These map
    // 1-to-1 with the extracted ColumnMeta entries that were appended at
    // compress time, so their `_col_idx` slots are physical_count_at_compress + i.
    // The executor uses col_names/col_types indexed by `_col_idx`, so they
    // need to be visible here too. json_extract gating against partitions
    // compressed before the feature was added is handled separately in
    // scan::path::is_json_extract_safe_for_rel.
    if let Some(jx) = json_extract {
        let specs = crate::compress::parse_extract_specs(&jx);
        // For json-extract columns the historical `_col_idx` was computed at
        // compress time as `non_segment_by_physical_count + i`. Recover that
        // count from the descriptor when present (more accurate when columns
        // have been added since), and fall back to live count otherwise.
        let non_segment_by_physical_count: u16 = if let Some(entries) = descriptor_entries.as_ref()
        {
            entries
                .iter()
                .filter(|e| !e.dropped && e.compressed_col_idx.is_some())
                .count() as u16
        } else {
            blob_idx.iter().filter(|b| b.is_some()).count() as u16
        };
        for (i, spec) in specs.iter().enumerate() {
            col_names.push(spec.target_name.clone());
            col_types.push(crate::scan::json_extract::kind_to_type_oid(
                spec.target_kind,
            ));
            col_typmods.push(-1);
            col_not_null.push(false);
            col_attnums.push(0); // synthetic — no pg_attribute row
            blob_idx.push(Some(non_segment_by_physical_count + i as u16));
        }
    }

    // Compute missing values for columns that exist in current pg_attribute
    // but have no entry in the persisted descriptor (added after this
    // partition was compressed). PG's fast-default machinery populates
    // `pg_attribute.attmissingval` for those — we call `getmissingattr` on
    // the partition's own tupdesc (partitions inherit from parent so the
    // missingval is the same). Only populated when a descriptor was found;
    // legacy partitions (descriptor IS NULL) treat all attnums as present
    // in the legacy positional mapping, so no synthesis is needed.
    let mut missing_values: Vec<Option<(pg_sys::Datum, bool)>> = vec![None; col_names.len()];
    if descriptor_entries.is_some() {
        // PG sets `pg_attribute.attmissingval` on each leaf partition (not
        // the parent — the parent has no heap and pg_attribute.atthasmissing
        // stays false). Use the partition's own OID so `getmissingattr`
        // reads the per-partition default that PG populated when ALTER
        // TABLE ADD COLUMN propagated to this leaf.
        let part_fqn = format!("{}.{}", parent_schema, partition_name);
        let part_rel_oid: pg_sys::Oid = client
            .select(&format!("SELECT '{}'::regclass::oid", part_fqn), None, &[])
            .ok()
            .and_then(|r| r.first().get_one::<pg_sys::Oid>().ok().flatten())
            .unwrap_or(pg_sys::InvalidOid);
        if part_rel_oid != pg_sys::InvalidOid {
            for i in 0..col_names.len() {
                let attnum = col_attnums[i];
                if attnum <= 0 {
                    continue; // synthetic json-extract column
                }
                if blob_idx[i].is_some() {
                    continue; // has a blob, no synthesis needed
                }
                if segment_by.contains(&col_names[i]) {
                    continue; // segment_by values come from _meta
                }
                // Descriptor present + no blob_idx + not segment_by =
                // column added after this partition was compressed.
                let (datum, is_null) = unsafe {
                    crate::scan::exec::datum_utils::missing_attr_for_relation(part_rel_oid, attnum)
                };
                missing_values[i] = Some((datum, is_null));
            }
        }
    }

    debug_assert_eq!(col_names.len(), col_types.len());
    debug_assert_eq!(col_names.len(), col_typmods.len());
    debug_assert_eq!(col_names.len(), col_not_null.len());
    debug_assert_eq!(col_names.len(), col_attnums.len());
    debug_assert_eq!(col_names.len(), blob_idx.len());
    debug_assert_eq!(col_names.len(), missing_values.len());

    MetadataInfo {
        col_names,
        col_types,
        col_typmods,
        col_not_null,
        segment_by,
        order_by,
        time_column,
        blob_idx,
        attnums: col_attnums,
        missing_values,
    }
}

/// Parallel-vector view of one parent table's physical columns. Returned
/// by `load_parent_columns` in attnum order, skipping dropped attributes.
struct ParentColumnLayout {
    names: Vec<String>,
    types: Vec<pg_sys::Oid>,
    typmods: Vec<i32>,
    not_null: Vec<bool>,
    attnums: Vec<i32>,
}

/// Walk the parent table's relcache `TupleDesc` and return its physical
/// column layout. Replaces the 4-table `pg_attribute ⋈ pg_type ⋈ pg_class ⋈
/// pg_namespace` SPI that the executor used to do here — saves ~0.8 ms per
/// `load_metadata` miss on fresh-connection paths.
///
/// Errors out if the schema or relation can't be resolved (e.g. the parent
/// has been dropped between planning and execution).
fn load_parent_columns(parent_schema: &str, parent_table: &str) -> ParentColumnLayout {
    let schema_cstr =
        std::ffi::CString::new(parent_schema).expect("pg_deltax: parent schema name contained NUL");
    let table_cstr =
        std::ffi::CString::new(parent_table).expect("pg_deltax: parent table name contained NUL");
    unsafe {
        let ns_oid = pg_sys::get_namespace_oid(schema_cstr.as_ptr(), false);
        let parent_oid = pg_sys::get_relname_relid(table_cstr.as_ptr(), ns_oid);
        if parent_oid == pg_sys::InvalidOid {
            pgrx::error!(
                "pg_deltax: parent relation {}.{} not found",
                parent_schema,
                parent_table
            );
        }
        // The parent is already locked by the surrounding query (planner +
        // executor take their own lock); re-acquiring AccessShareLock here
        // just bumps the local lock count.
        let rel = pg_sys::table_open(parent_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let tupdesc = (*rel).rd_att;
        let natts = (*tupdesc).natts as usize;

        let mut col_names = Vec::with_capacity(natts);
        let mut col_types = Vec::with_capacity(natts);
        let mut col_typmods = Vec::with_capacity(natts);
        let mut col_not_null = Vec::with_capacity(natts);
        let mut col_attnums = Vec::with_capacity(natts);
        for i in 0..natts {
            let att = &*tupdesc_get_attr(tupdesc, i);
            if att.attisdropped {
                continue;
            }
            let name = std::ffi::CStr::from_ptr(att.attname.data.as_ptr())
                .to_string_lossy()
                .into_owned();
            col_names.push(name);
            col_types.push(att.atttypid);
            col_typmods.push(att.atttypmod);
            col_not_null.push(att.attnotnull);
            col_attnums.push(att.attnum as i32);
        }
        pg_sys::table_close(rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        ParentColumnLayout {
            names: col_names,
            types: col_types,
            typmods: col_typmods,
            not_null: col_not_null,
            attnums: col_attnums,
        }
    }
}

/// Parsed view of one entry in the `compressed_columns` JSONB descriptor.
/// Field semantics match the JSON shape produced by
/// `catalog::snapshot_compressed_columns`. See `dev/docs/SCHEMA_CHANGES.md`.
struct DescriptorEntry {
    attnum: i32,
    type_oid: i64,
    typmod: i32,
    /// Historical `_col_idx` for non-segment_by columns. `None` for segment_by
    /// (they live in `_meta`, not in the blob path).
    compressed_col_idx: Option<u16>,
    dropped: bool,
}

/// Parse the `compressed_columns` JSONB array into a list of entries.
/// Returns `None` if the JSON shape is not the expected array-of-objects —
/// callers fall back to the legacy positional mapping in that case so a
/// corrupted descriptor doesn't take a partition offline.
fn parse_compressed_columns(value: &serde_json::Value) -> Option<Vec<DescriptorEntry>> {
    let arr = value.as_array()?;
    let mut out: Vec<DescriptorEntry> = Vec::with_capacity(arr.len());
    for entry in arr {
        let obj = entry.as_object()?;
        let attnum = obj.get("attnum")?.as_i64()? as i32;
        let type_oid = obj.get("type_oid")?.as_i64()?;
        let typmod = obj.get("typmod")?.as_i64()? as i32;
        let compressed_col_idx = match obj.get("compressed_col_idx") {
            Some(v) if v.is_null() => None,
            Some(v) => {
                let n = v.as_i64()?;
                if !(0..=i64::from(u16::MAX)).contains(&n) {
                    return None;
                }
                Some(n as u16)
            }
            None => None,
        };
        let dropped = obj
            .get("dropped")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        out.push(DescriptorEntry {
            attnum,
            type_oid,
            typmod,
            compressed_col_idx,
            dropped,
        });
    }
    Some(out)
}

/// Per-phase shared-buffer counters captured from `pgBufferUsage` deltas,
/// so EXPLAIN (ANALYZE, BUFFERS) can distinguish where I/O happened
/// (meta vs bloom pruning vs blob detoast) — custom-scan work runs in
/// BeginCustomScan, outside PG's own node-level instrumentation.
#[derive(Default, Clone, Copy)]
pub(crate) struct ScanBufferStats {
    pub(crate) meta_hit: i64,
    pub(crate) meta_read: i64,
    pub(crate) bloom_hit: i64,
    pub(crate) bloom_read: i64,
    pub(crate) blob_hit: i64,
    pub(crate) blob_read: i64,
}

impl ScanBufferStats {
    fn accumulate(&mut self, other: &ScanBufferStats) {
        self.meta_hit += other.meta_hit;
        self.meta_read += other.meta_read;
        self.bloom_hit += other.bloom_hit;
        self.bloom_read += other.bloom_read;
        self.blob_hit += other.blob_hit;
        self.blob_read += other.blob_read;
    }
}

// Thread-local accumulator for buffer stats produced by `load_segments_heap`.
// Callers reset via `reset_scan_buf_stats()` at the start of a BeginCustomScan
// callback and read the accumulated value via `take_scan_buf_stats()` before
// stashing it in their state struct. This avoids threading a parameter through
// the many agg/count/minmax fast-path helpers that construct state.
thread_local! {
    static LAST_SCAN_BUF_STATS: std::cell::Cell<ScanBufferStats> =
        const { std::cell::Cell::new(ScanBufferStats {
            meta_hit: 0, meta_read: 0,
            bloom_hit: 0, bloom_read: 0,
            blob_hit: 0, blob_read: 0,
        }) };
}

pub(crate) fn reset_scan_buf_stats() {
    LAST_SCAN_BUF_STATS.with(|c| c.set(ScanBufferStats::default()));
}

pub(crate) fn take_scan_buf_stats() -> ScanBufferStats {
    LAST_SCAN_BUF_STATS.with(|c| c.replace(ScanBufferStats::default()))
}

fn accumulate_scan_buf_stats(delta: &ScanBufferStats) {
    LAST_SCAN_BUF_STATS.with(|c| {
        let mut cur = c.get();
        cur.accumulate(delta);
        c.set(cur);
    });
}

/// Snapshot `(shared_blks_hit, shared_blks_read)` from the global
/// `pgBufferUsage` counter. Used to compute per-phase deltas in
/// `load_segments_heap`.
#[inline]
unsafe fn shared_buf_snapshot() -> (i64, i64) {
    unsafe {
        let bu = std::ptr::addr_of!(pg_sys::pgBufferUsage);
        ((*bu).shared_blks_hit, (*bu).shared_blks_read)
    }
}

/// An equality/IN-list bloom probe: the hashes of the constant(s) compared
/// against column `col_idx` of the companion tables.
struct BloomCheck {
    col_idx: u16,
    hashes: Vec<u64>,
}

/// Probe the partition's `<partition>_blooms` table for the given checks and
/// return the ids (subset of `surviving_ids`) of segments whose bloom filter
/// proves the queried constant(s) absent. Segments without a bloom row are
/// never returned — their values are not covered (e.g. blooms were disabled
/// at compress time), so absence of a row says nothing.
unsafe fn bloom_pruned_segment_ids(
    meta_oid: pg_sys::Oid,
    bloom_checks: &[BloomCheck],
    surviving_ids: &std::collections::HashSet<i32>,
) -> std::collections::HashSet<i32> {
    unsafe {
        let mut bloom_pruned_ids: std::collections::HashSet<i32> = std::collections::HashSet::new();
        if bloom_checks.is_empty() || surviving_ids.is_empty() {
            return bloom_pruned_ids;
        }

        let meta_name_ptr = pg_sys::get_rel_name(meta_oid);
        if meta_name_ptr.is_null() {
            return bloom_pruned_ids;
        }
        let meta_name_str = std::ffi::CStr::from_ptr(meta_name_ptr)
            .to_string_lossy()
            .into_owned();
        let meta_ns_oid = pg_sys::get_rel_namespace(meta_oid);
        let partition_name = meta_name_str
            .strip_suffix("_meta")
            .unwrap_or(&meta_name_str);
        let blooms_name = format!("{}_blooms", partition_name);
        let blooms_cname = std::ffi::CString::new(blooms_name).unwrap();
        let blooms_oid = pg_sys::get_relname_relid(blooms_cname.as_ptr(), meta_ns_oid);
        if blooms_oid == pg_sys::InvalidOid {
            return bloom_pruned_ids;
        }

        let blooms_rel =
            pg_sys::table_open(blooms_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let blooms_tupdesc = (*blooms_rel).rd_att;
        let blooms_natts = (*blooms_tupdesc).natts as usize;

        // Locate attnos for _segment_id, _num_hashes, _data once from the tupdesc
        let mut seg_id_att: Option<usize> = None;
        let mut num_hashes_att: Option<usize> = None;
        let mut data_att: Option<usize> = None;
        for i in 0..blooms_natts {
            let attr = &*tupdesc_get_attr(blooms_tupdesc, i);
            let name = std::ffi::CStr::from_ptr(attr.attname.data.as_ptr()).to_string_lossy();
            if name == "_segment_id" {
                seg_id_att = Some(i);
            } else if name == "_num_hashes" {
                num_hashes_att = Some(i);
            } else if name == "_data" {
                data_att = Some(i);
            }
        }

        let pk_index_oid = primary_key_index_oid(blooms_rel);

        if let (Some(sid_att), Some(nh_att), Some(dat_att), true) = (
            seg_id_att,
            num_hashes_att,
            data_att,
            pk_index_oid != pg_sys::InvalidOid,
        ) {
            let snapshot = pg_sys::GetActiveSnapshot();
            let idx_rel =
                pg_sys::index_open(pk_index_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

            for bc in bloom_checks {
                // Set up scan key: _col_idx = col_idx (SMALLINT equality)
                let mut skey = [pg_sys::ScanKeyData::default()];
                pg_sys::ScanKeyInit(
                    &mut skey[0],
                    1, // attnum 1 = _col_idx
                    pg_sys::BTEqualStrategyNumber as u16,
                    pg_sys::F_INT2EQ.into(),
                    pg_sys::Datum::from(bc.col_idx as i16),
                );

                let scan = index_beginscan_compat(blooms_rel, idx_rel, snapshot, 1, 0);
                pg_sys::index_rescan(scan, skey.as_mut_ptr(), 1, std::ptr::null_mut(), 0);

                let slot = pg_sys::table_slot_create(blooms_rel, std::ptr::null_mut());

                loop {
                    if !pg_sys::index_getnext_slot(
                        scan,
                        pg_sys::ScanDirection::ForwardScanDirection,
                        slot,
                    ) {
                        break;
                    }

                    pg_sys::slot_getallattrs(slot);
                    let tts_values = (*slot).tts_values;
                    let tts_isnull = (*slot).tts_isnull;

                    if *tts_isnull.add(sid_att)
                        || *tts_isnull.add(nh_att)
                        || *tts_isnull.add(dat_att)
                    {
                        continue;
                    }
                    let seg_id = (*tts_values.add(sid_att)).value() as i32;

                    if !surviving_ids.contains(&seg_id) {
                        continue;
                    }

                    let num_hashes = (*tts_values.add(nh_att)).value() as u8;

                    // Detoast bloom data
                    let varlena_ptr = (*tts_values.add(dat_att)).cast_mut_ptr::<pg_sys::varlena>();
                    let detoasted = pg_sys::pg_detoast_datum(varlena_ptr);
                    let data_ptr = pgrx::vardata_any(detoasted);
                    let data_len = pgrx::varsize_any_exhdr(detoasted);
                    #[allow(clippy::unnecessary_cast)]
                    let bloom_bytes = std::slice::from_raw_parts(data_ptr as *const u8, data_len);

                    let bf = crate::bloom::BloomFilter::from_bytes(bloom_bytes, num_hashes);
                    let any_match = bc.hashes.iter().any(|&h| bf.might_contain(h));

                    if detoasted != varlena_ptr {
                        pg_sys::pfree(detoasted as *mut _);
                    }

                    if !any_match {
                        bloom_pruned_ids.insert(seg_id);
                    }
                }

                pg_sys::ExecDropSingleTupleTableSlot(slot);
                pg_sys::index_endscan(scan);
            }

            pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        }

        pg_sys::table_close(blooms_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        bloom_pruned_ids
    }
}

/// Load segment data via two-phase scan: meta table (no TOAST) then blob table
/// (column-major, sequential TOAST I/O per column).
///
/// Phase 1: Heap-scan the meta table to extract segment_by values, row counts,
/// min/max, sums, and apply pruning. Zero TOAST I/O (no BYTEA columns).
///
/// Phase 2: Index-scan the blob table for each needed column, reading only
/// surviving segments. TOAST chunks are contiguous per column for sequential I/O.
///
/// When `lazy_cols` is provided, columns marked true are stored as TOAST pointer
/// copies (~18 bytes each) instead of being fully detoasted. Call
/// `detoast_lazy_blobs()` later to materialize them on demand.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn load_segments_heap(
    meta_oid: pg_sys::Oid,
    col_names: &[String],
    segment_by: &[String],
    needed_cols: &[bool],
    time_column: &str,
    load_minmax: bool,
    segment_by_filters: &[(usize, String)],
    time_min: Option<i64>,
    time_max: Option<i64>,
    lazy_cols: Option<&[bool]>,
    batch_quals: &[BatchQual],
    needed_stats_cols: &[String],
    col_types: &[pg_sys::Oid],
    col_not_null: &[bool],
    needed_minmax_cols: &[String],
    // Parallel to `col_names`. `Some(i)` = read this column from the
    // blob/colstats path at `_col_idx = i`. `None` = segment_by (live in
    // `_meta`) OR a column added to the parent after this partition was
    // compressed (no blob — decompress synthesizes via getmissingattr).
    blob_idx: &[Option<u16>],
    // `skip_blob_load = true` skips Phase 2 entirely. Callers that fetch blobs
    // on-claim via `fetch_segment_blobs` pass true — compressed_blobs and
    // toast_pointers stay empty at return.
    skip_blob_load: bool,
) -> (Vec<SegmentData>, u64, u64, u64, u64, u64) {
    // Returns: (segments, total_skipped, minmax_skipped, bloom_skipped,
    // valbitmap_skipped, detoast_us). Segment-level pruning counters are
    // additive: `total_skipped` = sum of every reason we dropped a segment.
    // Buffer stats are accumulated into a thread-local via `accumulate_scan_buf_stats`;
    // callers read them with `take_scan_buf_stats()` after all companion OIDs are processed.
    unsafe {
        let mut buf_stats = ScanBufferStats::default();
        let (t0_hit, t0_read) = shared_buf_snapshot();

        // ================================================================
        // Phase -1: Partition-level minmax pruning. Uses the partition's
        // `column_minmax` JSONB on `deltax.deltax_partition` (populated at
        // compress time) to skip partitions whose [min, max] range doesn't
        // cover any of the equality consts. Avoids the ~60µs per-partition
        // colstats open + index probe for the bulk of partitions on wide
        // scans that filter on a non-time, non-segment-by column (e.g.
        // `WHERE order_id = 700` over 123 monthly partitions — 119 of them
        // get ruled out here).
        //
        // We only consult column_minmax when batch_quals has eligible
        // equality predicates. Bulk-loaded across the deltatable on first
        // miss (one SPI for all partitions, cached backend-local).
        if !batch_quals.is_empty() && segment_by.is_empty() {
            let eligible_eq: Vec<(usize, i64)> = batch_quals
                .iter()
                .filter_map(|bq| {
                    if bq.op != BatchCompareOp::Eq {
                        return None;
                    }
                    let col_name = &col_names[bq.col_idx];
                    let is_orderable = matches!(
                        bq.type_oid,
                        pg_sys::INT2OID
                            | pg_sys::INT4OID
                            | pg_sys::INT8OID
                            | pg_sys::FLOAT4OID
                            | pg_sys::FLOAT8OID
                            | pg_sys::DATEOID
                            | pg_sys::TIMESTAMPOID
                            | pg_sys::TIMESTAMPTZOID
                    );
                    if !is_orderable
                        || col_name == time_column
                        || !col_not_null.get(bq.col_idx).copied().unwrap_or(false)
                    {
                        return None;
                    }
                    let value = encode_datum_to_i64(bq.const_datum, bq.type_oid)?;
                    Some((bq.col_idx, value))
                })
                .collect();

            if !eligible_eq.is_empty()
                && let Some(part_minmax) = crate::scan::cost::get_partition_column_minmax(meta_oid)
            {
                let can_match = eligible_eq.iter().all(|(col_idx, value)| {
                    let col_name = &col_names[*col_idx];
                    match part_minmax.get(col_name) {
                        Some(&(pmin, pmax)) => *value >= pmin && *value <= pmax,
                        None => true, // no entry → can't prune
                    }
                });
                if !can_match {
                    let total_segments = reltuples_as_u64(meta_oid).unwrap_or_else(|| {
                        crate::scan::cost::get_segment_count(meta_oid).max(0) as u64
                    });
                    let (t1_hit, t1_read) = shared_buf_snapshot();
                    buf_stats.meta_hit = t1_hit - t0_hit;
                    buf_stats.meta_read = t1_read - t0_read;
                    accumulate_scan_buf_stats(&buf_stats);
                    return (Vec::new(), total_segments, total_segments, 0, 0, 0);
                }
            }
        }

        // ================================================================
        // Phase 0: Colstats-index prefilter — done BEFORE opening meta so
        // partitions whose colstats minmax rules out every segment can
        // return without paying the meta-table open + tupdesc walk +
        // attno HashMap construction. For queries like Q12/Q13 (no time
        // predicate, equality on a non-segment-by column) this is what
        // most partitions do — the prefilter immediately rules them out.
        // ================================================================

        // The authoritative `col_names[i] → _col_idx` mapping is the
        // `blob_idx` parameter (built from the partition's
        // `compressed_columns` descriptor in `load_metadata`). For legacy
        // partitions with no descriptor, `load_metadata` populates `blob_idx`
        // from the same positional rule we used to compute locally here, so
        // behavior is unchanged for them. See `dev/docs/SCHEMA_CHANGES.md`.
        debug_assert_eq!(col_names.len(), blob_idx.len());
        let col_idx_map: &[Option<u16>] = blob_idx;
        let num_blob_cols: usize = col_idx_map.iter().filter(|b| b.is_some()).count();

        // Phase 0a: skip-meta fast path — `count(*)`-style scans (or any
        // caller that passes `skip_blob_load=true` with a single point qual)
        // can be answered entirely from the colstats minmax index.
        let point_lookup_filter = if segment_by.is_empty()
            && segment_by_filters.is_empty()
            && time_min.is_none()
            && time_max.is_none()
            && !load_minmax
            && needed_stats_cols.is_empty()
            && needed_minmax_cols.is_empty()
            && skip_blob_load
            && batch_quals.len() == 1
        {
            let bq = &batch_quals[0];
            let col_name = &col_names[bq.col_idx];
            let is_orderable = matches!(
                bq.type_oid,
                pg_sys::INT2OID
                    | pg_sys::INT4OID
                    | pg_sys::INT8OID
                    | pg_sys::FLOAT4OID
                    | pg_sys::FLOAT8OID
                    | pg_sys::DATEOID
                    | pg_sys::TIMESTAMPOID
                    | pg_sys::TIMESTAMPTZOID
            );
            if bq.op == BatchCompareOp::Eq
                && is_orderable
                && col_name != time_column
                && !segment_by.contains(col_name)
                && col_not_null.get(bq.col_idx).copied().unwrap_or(false)
                && let Some(ci) = col_idx_map[bq.col_idx]
                && let Some(value) = encode_datum_to_i64(bq.const_datum, bq.type_oid)
            {
                Some((ci as i16, value))
            } else {
                None
            }
        } else {
            None
        };

        if let Some((filter_col_idx, filter_value)) = point_lookup_filter {
            let colstats_oid = sibling_table_oid(meta_oid, "_colstats");
            if colstats_oid != pg_sys::InvalidOid
                && let Some(mut point_segments) = lookup_point_segments_by_minmax_index(
                    colstats_oid,
                    filter_col_idx,
                    filter_value,
                    num_blob_cols,
                )
            {
                for seg in &mut point_segments {
                    seg.companion_oid = meta_oid;
                }
                attach_tombstones(meta_oid, &mut point_segments);
                let total_segments = reltuples_as_u64(meta_oid).unwrap_or_else(|| {
                    crate::scan::cost::get_segment_count(meta_oid).max(0) as u64
                });

                let (t1_hit, t1_read) = shared_buf_snapshot();
                buf_stats.meta_hit = t1_hit - t0_hit;
                buf_stats.meta_read = t1_read - t0_read;

                // Bloom-prune the minmax survivors: for high-cardinality
                // columns whose per-segment [min, max] spans most of the
                // domain, the minmax index barely prunes and the blooms do
                // the real work (e.g. Q19 `WHERE UserID = <const>`).
                let mut bloom_skipped: u64 = 0;
                if !point_segments.is_empty() {
                    let bq = &batch_quals[0];
                    // Hash the constant exactly as compress-time bloom
                    // insertion does: `bloom_probe_encode` converts the raw
                    // PG datum into the build domain (Unix-epoch µs for
                    // timestamps/dates, bit pattern for floats, identity for
                    // ints — see `compute_segment_blooms` and the epoch
                    // regression tests).
                    let val_i64 = bloom_probe_encode(bq.const_datum, bq.type_oid);
                    let checks = [BloomCheck {
                        col_idx: filter_col_idx as u16,
                        hashes: vec![crate::bloom::hash_datum_i64(val_i64)],
                    }];
                    let surviving: std::collections::HashSet<i32> =
                        point_segments.iter().map(|s| s.segment_id).collect();
                    let pruned = bloom_pruned_segment_ids(meta_oid, &checks, &surviving);
                    if !pruned.is_empty() {
                        bloom_skipped = pruned.len() as u64;
                        point_segments.retain(|s| !pruned.contains(&s.segment_id));
                    }
                    let (tb_hit, tb_read) = shared_buf_snapshot();
                    buf_stats.bloom_hit = tb_hit - t1_hit;
                    buf_stats.bloom_read = tb_read - t1_read;
                }
                accumulate_scan_buf_stats(&buf_stats);

                let kept = point_segments.len() as u64;
                let skipped = total_segments.saturating_sub(kept);
                let minmax_skipped = skipped.saturating_sub(bloom_skipped);

                return (point_segments, skipped, minmax_skipped, bloom_skipped, 0, 0);
            }
        }

        // Phase 0b: general point-lookup prefilter. Probes the colstats
        // `(_col_idx, _min, _max)` btree to get the candidate segment_ids
        // whose min/max range covers each equality const. If this returns
        // an empty set, the partition contributes zero rows and we can skip
        // the meta open entirely.
        let mut point_prefilter_cols: std::collections::HashSet<i16> =
            std::collections::HashSet::new();
        let point_prefilter = if segment_by.is_empty() {
            let filters: Vec<(i16, i64)> = batch_quals
                .iter()
                .filter_map(|bq| {
                    if bq.op != BatchCompareOp::Eq {
                        return None;
                    }
                    let col_name = &col_names[bq.col_idx];
                    let is_orderable = matches!(
                        bq.type_oid,
                        pg_sys::INT2OID
                            | pg_sys::INT4OID
                            | pg_sys::INT8OID
                            | pg_sys::FLOAT4OID
                            | pg_sys::FLOAT8OID
                            | pg_sys::DATEOID
                            | pg_sys::TIMESTAMPOID
                            | pg_sys::TIMESTAMPTZOID
                    );
                    if !is_orderable
                        || col_name == time_column
                        || segment_by.contains(col_name)
                        || !col_not_null.get(bq.col_idx).copied().unwrap_or(false)
                    {
                        return None;
                    }
                    let ci = col_idx_map[bq.col_idx]?;
                    let value = encode_datum_to_i64(bq.const_datum, bq.type_oid)?;
                    Some((ci as i16, value))
                })
                .collect();

            if filters.is_empty() {
                None
            } else {
                let colstats_oid = sibling_table_oid(meta_oid, "_colstats");
                if colstats_oid == pg_sys::InvalidOid {
                    None
                } else {
                    let candidates = lookup_segments_by_minmax_index(colstats_oid, &filters);
                    if candidates.is_some() {
                        point_prefilter_cols.extend(filters.iter().map(|(ci, _)| *ci));
                    }
                    candidates
                }
            }
        } else {
            None
        };

        if point_prefilter.as_ref().is_some_and(|ids| ids.is_empty()) {
            let total_segments = reltuples_as_u64(meta_oid)
                .unwrap_or_else(|| crate::scan::cost::get_segment_count(meta_oid).max(0) as u64);
            let (t1_hit, t1_read) = shared_buf_snapshot();
            buf_stats.meta_hit = t1_hit - t0_hit;
            buf_stats.meta_read = t1_read - t0_read;
            accumulate_scan_buf_stats(&buf_stats);

            return (Vec::new(), total_segments, total_segments, 0, 0, 0);
        }

        // ================================================================
        // Phase 1: Scan meta table — no TOAST I/O
        // ================================================================
        let rel = pg_sys::table_open(meta_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let tupdesc = (*rel).rd_att;
        let natts = (*tupdesc).natts as usize;

        // Build column-name-to-attno mapping from meta TupleDesc
        let mut attno_map: HashMap<String, usize> = HashMap::new();
        let mut att_type_oids: HashMap<String, pg_sys::Oid> = HashMap::new();
        for i in 0..natts {
            let att = &*tupdesc_get_attr(tupdesc, i);
            if att.attisdropped {
                continue;
            }
            let name = std::ffi::CStr::from_ptr(att.attname.data.as_ptr())
                .to_string_lossy()
                .into_owned();
            att_type_oids.insert(name.clone(), att.atttypid);
            attno_map.insert(name, i);
        }

        // Locate attribute indices for segment_by columns and _row_count
        let mut segment_by_attnos: Vec<(usize, pg_sys::Oid)> = Vec::new();
        for name in col_names {
            if segment_by.contains(name)
                && let Some(&attno) = attno_map.get(name.as_str())
            {
                let type_oid = att_type_oids[name.as_str()];
                segment_by_attnos.push((attno, type_oid));
            }
        }

        let row_count_attno = attno_map.get("_row_count").copied();
        let segment_id_attno = attno_map.get("_segment_id").copied();

        let min_time_name = format!("_min_{}", time_column);
        let max_time_name = format!("_max_{}", time_column);
        let min_time_attno = attno_map.get(min_time_name.as_str()).copied();
        let max_time_attno = attno_map.get(max_time_name.as_str()).copied();

        // Discover per-column min/max columns
        let mut minmax_col_attnos: Vec<(String, usize, usize, pg_sys::Oid)> = Vec::new();
        if load_minmax {
            for col_name in col_names {
                if segment_by.contains(col_name) {
                    continue;
                }
                let min_name = format!("_min_{}", col_name);
                let max_name = format!("_max_{}", col_name);
                if let (Some(&min_att), Some(&max_att)) = (
                    attno_map.get(min_name.as_str()),
                    attno_map.get(max_name.as_str()),
                ) {
                    let type_oid = att_type_oids
                        .get(min_name.as_str())
                        .copied()
                        .unwrap_or(pg_sys::InvalidOid);
                    minmax_col_attnos.push((col_name.clone(), min_att, max_att, type_oid));
                }
            }
        }

        // Discover per-column sum/nonnull_count/nonzero_count columns
        let load_sums = !needed_stats_cols.is_empty();
        let mut sum_col_attnos: Vec<(String, usize, usize, Option<usize>, pg_sys::Oid)> =
            Vec::new();
        if load_sums {
            for col_name in col_names {
                if segment_by.contains(col_name) {
                    continue;
                }
                let sum_name = format!("_sum_{}", col_name);
                let nonnull_name = format!("_nonnull_count_{}", col_name);
                let nonzero_name = format!("_nonzero_count_{}", col_name);
                if let (Some(&sum_att), Some(&nn_att)) = (
                    attno_map.get(sum_name.as_str()),
                    attno_map.get(nonnull_name.as_str()),
                ) {
                    let nz_att = attno_map.get(nonzero_name.as_str()).copied();
                    let type_oid = att_type_oids
                        .get(sum_name.as_str())
                        .copied()
                        .unwrap_or(pg_sys::InvalidOid);
                    sum_col_attnos.push((col_name.clone(), sum_att, nn_att, nz_att, type_oid));
                }
            }
        }

        // Begin meta table scan
        let snapshot = pg_sys::GetActiveSnapshot();
        let flags: u32 = pg_sys::ScanOptions::SO_TYPE_SEQSCAN
            | pg_sys::ScanOptions::SO_ALLOW_STRAT
            | pg_sys::ScanOptions::SO_ALLOW_SYNC
            | pg_sys::ScanOptions::SO_ALLOW_PAGEMODE;
        let scan = (*(*rel).rd_tableam).scan_begin.unwrap()(
            rel,
            snapshot,
            0,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            flags,
        );

        // Surviving segment metadata: (index_in_segments_vec, segment_id)
        let mut segments: Vec<SegmentData> = Vec::new();
        let mut surviving_segment_ids: Vec<i32> = Vec::new();
        let mut segments_skipped: u64 = 0;
        let mut segments_minmax_skipped: u64 = 0;
        let mut heap_getnext_us: u64 = 0;
        let mut deform_us: u64 = 0;
        let mut values = vec![pg_sys::Datum::from(0); natts];
        let mut nulls = vec![true; natts];

        // Build bloom filter checks from batch quals (Eq and InList on numeric types)
        // Build valbitmap checks from batch quals (text Eq on low-card columns
        // whose partition-level value list is in `column_valmap`). Each check
        // carries the bit indices the segment must contain at least one of.
        // An empty `wanted_bits` means the queried constant is absent from
        // the partition-level valmap — every segment that wrote a bitmap row
        // fails the check and gets pruned. Segments WITHOUT a bitmap row
        // (they overflowed VALBITMAP_MAX_DISTINCT at compress time, so the
        // valmap doesn't cover their values) must never be pruned: the
        // constant may well live only in them.
        struct ValbitmapCheck {
            col_idx: u16,
            wanted_bits: Vec<u8>,
        }
        let mut bloom_checks: Vec<BloomCheck> = Vec::new();
        let mut valbitmap_checks: Vec<ValbitmapCheck> = Vec::new();
        let valmap = crate::scan::cost::get_column_valmap(meta_oid);
        for bq in batch_quals {
            match bq.op {
                BatchCompareOp::Eq | BatchCompareOp::InList => {}
                _ => continue,
            }
            let col_name = &col_names[bq.col_idx];
            if segment_by.contains(col_name) {
                continue;
            }
            let ci = match col_idx_map[bq.col_idx] {
                Some(ci) => ci,
                None => continue,
            };

            // Numeric / temporal types → bloom (existing path).
            let is_numeric_type = matches!(
                bq.type_oid,
                pg_sys::INT2OID
                    | pg_sys::INT4OID
                    | pg_sys::INT8OID
                    | pg_sys::FLOAT4OID
                    | pg_sys::FLOAT8OID
                    | pg_sys::DATEOID
                    | pg_sys::TIMESTAMPOID
                    | pg_sys::TIMESTAMPTZOID
            );
            if is_numeric_type {
                let hashes = if bq.op == BatchCompareOp::InList {
                    if let Some(ref vals) = bq.in_list_i64 {
                        vals.iter()
                            .map(|&v| {
                                crate::bloom::hash_datum_i64(bloom_probe_encode(
                                    pg_sys::Datum::from(v as usize),
                                    bq.type_oid,
                                ))
                            })
                            .collect()
                    } else {
                        continue;
                    }
                } else {
                    vec![crate::bloom::hash_datum_i64(bloom_probe_encode(
                        bq.const_datum,
                        bq.type_oid,
                    ))]
                };
                bloom_checks.push(BloomCheck {
                    col_idx: ci,
                    hashes,
                });
                continue;
            }

            // Text Eq on a column with a partition-level valmap → exact bitmap
            // pruning. InList not yet supported for valbitmap (would need
            // text_const_list on BatchQual; not in the struct today).
            if bq.op == BatchCompareOp::Eq
                && super::batch_qual::is_text_type(bq.type_oid)
                && let Some(ref needle) = bq.text_const
                && let Some(values) = valmap.get(col_name)
            {
                // A valmap miss leaves `wanted_bits` empty: no segment that
                // contributed to the valmap can match. Segments that
                // overflowed the per-segment distinct cap contributed
                // nothing (and have no bitmap row), so presence is still
                // re-checked per segment below.
                let wanted_bits: Vec<u8> = values
                    .iter()
                    .position(|v| v == needle)
                    .map(|idx| vec![idx as u8])
                    .unwrap_or_default();
                valbitmap_checks.push(ValbitmapCheck {
                    col_idx: ci,
                    wanted_bits,
                });
            }
        }
        let mut segments_bloom_skipped: u64 = 0;
        let mut segments_valbitmap_skipped: u64 = 0;

        loop {
            let getnext_start = std::time::Instant::now();
            let tuple = pg_sys::heap_getnext(scan, pg_sys::ScanDirection::ForwardScanDirection);
            heap_getnext_us += getnext_start.elapsed().as_micros() as u64;
            if tuple.is_null() {
                break;
            }

            let deform_start = std::time::Instant::now();
            pg_sys::heap_deform_tuple(tuple, tupdesc, values.as_mut_ptr(), nulls.as_mut_ptr());
            deform_us += deform_start.elapsed().as_micros() as u64;

            // Extract _segment_id
            let segment_id = match segment_id_attno {
                Some(attno) if !nulls[attno] => values[attno].value() as i32,
                _ => 0,
            };
            if let Some(ref candidate_ids) = point_prefilter
                && !candidate_ids.contains(&segment_id)
            {
                segments_skipped += 1;
                segments_minmax_skipped += 1;
                continue;
            }

            // Extract segment_by values
            let mut segment_values: Vec<Option<String>> = Vec::new();
            for &(attno, type_oid) in &segment_by_attnos {
                if !nulls[attno] {
                    let mut typoutput: pg_sys::Oid = pg_sys::InvalidOid;
                    let mut typisvarlena: bool = false;
                    pg_sys::getTypeOutputInfo(type_oid, &mut typoutput, &mut typisvarlena);
                    let cstr = pg_sys::OidOutputFunctionCall(typoutput, values[attno]);
                    let s = std::ffi::CStr::from_ptr(cstr)
                        .to_string_lossy()
                        .into_owned();
                    pg_sys::pfree(cstr as *mut _);
                    segment_values.push(Some(s));
                } else {
                    segment_values.push(None);
                }
            }

            let row_count = match row_count_attno {
                Some(attno) if !nulls[attno] => values[attno].value() as i32,
                _ => 0,
            };

            let seg_min_time = match min_time_attno {
                Some(attno) if !nulls[attno] => Some(values[attno].value() as i64),
                _ => None,
            };
            let seg_max_time = match max_time_attno {
                Some(attno) if !nulls[attno] => Some(values[attno].value() as i64),
                _ => None,
            };

            // --- Pruning (same logic as before, zero TOAST I/O) ---

            if !segment_by_filters.is_empty() {
                let mut skip = false;
                for &(seg_val_idx, ref filter_val) in segment_by_filters {
                    match &segment_values.get(seg_val_idx).and_then(|v| v.as_ref()) {
                        Some(val) if *val == filter_val => {}
                        _ => {
                            skip = true;
                            break;
                        }
                    }
                }
                if skip {
                    segments_skipped += 1;
                    continue;
                }
            }

            // time_max is exclusive (canonical half-open bounds).
            if let (Some(s_min), Some(s_max)) = (seg_min_time, seg_max_time)
                && (time_min.is_some_and(|qmin| s_max < qmin)
                    || time_max.is_some_and(|qmax| s_min >= qmax))
            {
                segments_skipped += 1;
                continue;
            }

            // --- Segment survived time/segment_by pruning ---

            // Extract per-column min/max. The meta table stores the time
            // column's `_min_`/`_max_` in its NATIVE type (PG-epoch datum),
            // but every `ColMinMax` consumer (`decode_encoded_to_datum`,
            // `decode_encoded_to_pg_i64`, `segment_all_rows_pass`, ...)
            // expects the order-preserving colstats encoding (Unix-epoch
            // microseconds for timestamps/dates) — so convert here with the
            // same `encode_datum_to_i64` rule used to populate colstats.
            let mut col_minmax = HashMap::new();
            for (col_name, min_att, max_att, type_oid) in &minmax_col_attnos {
                let min_null = nulls[*min_att];
                let max_null = nulls[*max_att];
                let encode_meta = |d: pg_sys::Datum| -> i64 {
                    encode_datum_to_i64(d, *type_oid).unwrap_or_else(|| d.value() as i64)
                };
                let min_enc = if min_null {
                    0i64
                } else {
                    encode_meta(values[*min_att])
                };
                let max_enc = if max_null {
                    0i64
                } else {
                    encode_meta(values[*max_att])
                };
                col_minmax.insert(
                    col_name.clone(),
                    ColMinMax {
                        min_encoded: min_enc,
                        max_encoded: max_enc,
                        min_null,
                        max_null,
                        type_oid: *type_oid,
                    },
                );
            }

            // Also populate time column minmax when requested by caller
            // (e.g. DeltaXMinMax on the time column) — avoids colstats scan.
            // Must encode PG-epoch datum → Unix-epoch i64 to match colstats encoding.
            if needed_minmax_cols.iter().any(|n| n == time_column)
                && !col_minmax.contains_key(time_column)
                && let (Some(min_att), Some(max_att)) = (min_time_attno, max_time_attno)
            {
                let min_null = nulls[min_att];
                let max_null = nulls[max_att];
                let time_type_oid = att_type_oids
                    .get(format!("_min_{}", time_column).as_str())
                    .copied()
                    .unwrap_or(pg_sys::TIMESTAMPTZOID);
                let encode_time = |raw: i64| -> i64 {
                    match time_type_oid {
                        pg_sys::DATEOID => {
                            // raw is PG-epoch days → Unix-epoch microseconds
                            (raw + crate::compress::PG_EPOCH_OFFSET_DAYS) * 86_400_000_000
                        }
                        pg_sys::TIMESTAMPOID | pg_sys::TIMESTAMPTZOID => {
                            // raw is PG-epoch usec → Unix-epoch usec
                            raw + crate::compress::PG_EPOCH_OFFSET_USEC
                        }
                        _ => raw,
                    }
                };
                let min_enc = if min_null {
                    0i64
                } else {
                    encode_time(values[min_att].value() as i64)
                };
                let max_enc = if max_null {
                    0i64
                } else {
                    encode_time(values[max_att].value() as i64)
                };
                col_minmax.insert(
                    time_column.to_string(),
                    ColMinMax {
                        min_encoded: min_enc,
                        max_encoded: max_enc,
                        min_null,
                        max_null,
                        type_oid: time_type_oid,
                    },
                );
            }

            // Extract per-column sum/nonnull_count/nonzero_count
            let mut col_sums = HashMap::new();
            for (col_name, sum_att, nn_att, nz_att, type_oid) in &sum_col_attnos {
                let sum_null = nulls[*sum_att];
                let sum_datum = if sum_null {
                    pg_sys::Datum::from(0usize)
                } else {
                    values[*sum_att]
                };
                let nonnull_count = if nulls[*nn_att] {
                    0i64
                } else {
                    values[*nn_att].value() as i64
                };
                let nonzero_count = match nz_att {
                    Some(att) => {
                        if nulls[*att] {
                            -1i64
                        } else {
                            values[*att].value() as i64
                        }
                    }
                    None => -1i64, // column missing in older meta tables
                };
                col_sums.insert(
                    col_name.clone(),
                    ColSum {
                        sum_datum,
                        sum_null,
                        sum_i128: None,
                        sum_f64: None,
                        nonnull_count,
                        nonzero_count,
                        type_oid: *type_oid,
                    },
                );
            }

            // Pre-allocate empty blob slots — will be filled in Phase 2.
            // resize_with avoids requiring BlobBytes: Clone (it isn't).
            let mut compressed_blobs: Vec<BlobBytes> = Vec::with_capacity(num_blob_cols);
            compressed_blobs.resize_with(num_blob_cols, BlobBytes::default);
            let text_length_blobs: Vec<Vec<u8>> = vec![Vec::new(); num_blob_cols];
            let toast_pointers: Vec<Vec<u8>> = vec![Vec::new(); num_blob_cols];

            surviving_segment_ids.push(segment_id);
            segments.push(SegmentData {
                companion_oid: meta_oid,
                segment_id,
                segment_values,
                compressed_blobs,
                text_length_blobs,
                row_count,
                min_time: seg_min_time,
                max_time: seg_max_time,
                col_minmax,
                col_sums,
                toast_pointers,
                cached_blob_pins: Vec::new(),
                tombstones: None,
            });
        }

        // End meta scan
        (*(*rel).rd_tableam).scan_end.unwrap()(scan);
        pg_sys::table_close(rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        let (t1_hit, t1_read) = shared_buf_snapshot();
        buf_stats.meta_hit = t1_hit - t0_hit;
        buf_stats.meta_read = t1_read - t0_read;

        // ================================================================
        // Phase 1b: Scan normalized colstats table for per-column stats
        // Only opened when we need non-time column stats and have surviving segments.
        // ================================================================
        // Build set of column names that already have minmax in the meta table.
        // Always include the time column — its min/max is loaded from meta
        // regardless of `load_minmax`.
        let mut meta_minmax_names: std::collections::HashSet<&str> = minmax_col_attnos
            .iter()
            .map(|(name, ..)| name.as_str())
            .collect();
        if min_time_attno.is_some() && max_time_attno.is_some() {
            meta_minmax_names.insert(time_column);
        }

        let need_colstats = !segments.is_empty()
            && (
                // Need sum data that's not in meta?
                (load_sums && sum_col_attnos.is_empty())
            // Caller needs minmax for specific columns not already in meta?
            || (!needed_minmax_cols.is_empty()
                && needed_minmax_cols.iter().any(|n| !meta_minmax_names.contains(n.as_str())))
            // Have batch quals on non-time orderable columns not covered by meta?
            || batch_quals.iter().any(|bq| {
                !matches!(bq.op, BatchCompareOp::Like | BatchCompareOp::NotLike)
                && matches!(bq.type_oid,
                    pg_sys::INT2OID | pg_sys::INT4OID | pg_sys::INT8OID
                    | pg_sys::FLOAT4OID | pg_sys::FLOAT8OID
                    | pg_sys::DATEOID | pg_sys::TIMESTAMPOID | pg_sys::TIMESTAMPTZOID)
                && {
                    let col_name = &col_names[bq.col_idx];
                    let min_name = format!("_min_{}", col_name);
                    !attno_map.contains_key(min_name.as_str())
                }
            })
            );

        if need_colstats {
            let colstats_oid = sibling_table_oid(meta_oid, "_colstats");

            if colstats_oid != pg_sys::InvalidOid {
                // Build col_idx -> (column_name, original_type_oid) mapping
                // (non-segment-by columns, 0-based, same order as blob table)
                let mut idx_to_col: Vec<(String, pg_sys::Oid)> = Vec::new();
                let mut col_to_idx: std::collections::HashMap<&str, usize> =
                    std::collections::HashMap::new();
                for (i, name) in col_names.iter().enumerate() {
                    if !segment_by.contains(name) {
                        let ci = idx_to_col.len();
                        idx_to_col.push((name.clone(), col_types[i]));
                        col_to_idx.insert(name.as_str(), ci);
                    }
                }

                // Build surviving segment_id -> index mapping
                let mut seg_id_to_idx: HashMap<i32, usize> = HashMap::new();
                for (idx, &sid) in surviving_segment_ids.iter().enumerate() {
                    seg_id_to_idx.insert(sid, idx);
                }

                // Build minmax filters for colstats (batch quals on non-time orderable columns)
                let mut cs_minmax_filters: Vec<MinMaxFilter> = Vec::new();
                for bq in batch_quals {
                    match bq.op {
                        BatchCompareOp::Like | BatchCompareOp::NotLike => continue,
                        _ => {}
                    }
                    let col_name = &col_names[bq.col_idx];
                    let min_name = format!("_min_{}", col_name);
                    if attno_map.contains_key(min_name.as_str()) {
                        continue;
                    } // already in meta
                    if segment_by.contains(col_name) {
                        continue;
                    }

                    let ci = match col_idx_map[bq.col_idx] {
                        Some(ci) => ci as i16,
                        None => continue,
                    };
                    if point_prefilter_cols.contains(&ci) {
                        continue;
                    }

                    let const_i64 = match encode_datum_to_i64(bq.const_datum, bq.type_oid) {
                        Some(v) => v,
                        None => continue,
                    };

                    // `in_list_i64` holds raw PG datum bits — re-encode each
                    // value into the colstats domain with the same rule as
                    // `const_i64` above (timestamps/dates: PG epoch → Unix
                    // epoch; floats: order-preserving bits; ints: identity).
                    let encoded_in_list = match bq.in_list_i64.as_ref() {
                        None => None,
                        Some(vals) => match vals
                            .iter()
                            .map(|&v| {
                                encode_datum_to_i64(pg_sys::Datum::from(v as usize), bq.type_oid)
                            })
                            .collect::<Option<Vec<i64>>>()
                        {
                            Some(enc) => Some(enc),
                            // Unencodable element — skip minmax pruning for
                            // this qual rather than risk a wrong prune.
                            None => continue,
                        },
                    };

                    cs_minmax_filters.push(MinMaxFilter {
                        col_idx: ci,
                        op: bq.op,
                        const_i64,
                        in_list_i64: encoded_in_list,
                    });
                }

                // Collect the set of _col_idx values we actually need:
                // - minmax filter columns (from batch quals)
                // - columns caller needs minmax for (needed_minmax_cols)
                // - columns caller needs stats for (needed_stats_cols)
                let mut needed_col_idxs: std::collections::HashSet<i16> =
                    std::collections::HashSet::new();
                for f in &cs_minmax_filters {
                    needed_col_idxs.insert(f.col_idx);
                }
                for name in needed_minmax_cols {
                    if let Some(&ci) = col_to_idx.get(name.as_str()) {
                        needed_col_idxs.insert(ci as i16);
                    }
                }
                for name in needed_stats_cols {
                    if let Some(&ci) = col_to_idx.get(name.as_str()) {
                        needed_col_idxs.insert(ci as i16);
                    }
                }

                let mut cs_pruned_ids: std::collections::HashSet<i32> =
                    std::collections::HashSet::new();

                // Check colstats cache — populate segments from cached data and
                // remove fully-cached col_idxs so we skip scanning them.
                COLSTATS_CACHE.with(|cache| {
                    let cache = cache.borrow();
                    let mut cached_idxs: Vec<i16> = Vec::new();
                    for &ci in &needed_col_idxs {
                        if let Some(cached) = cache.get(&(colstats_oid, ci)) {
                            let (ref col_name, orig_type_oid) = idx_to_col[ci as usize];
                            let mut all_found = true;
                            for (&sid, &seg_idx) in &seg_id_to_idx {
                                if let Some(row) = cached.rows.get(&sid) {
                                    // Apply minmax filters from cache
                                    if !cs_minmax_filters.is_empty()
                                        && !cs_pruned_ids.contains(&sid)
                                    {
                                        let mut skip = false;
                                        for f in &cs_minmax_filters {
                                            if f.col_idx == ci
                                                && !row.min_null
                                                && !row.max_null
                                                && !segment_passes_minmax_filter(
                                                    f,
                                                    row.min_encoded,
                                                    row.max_encoded,
                                                )
                                            {
                                                skip = true;
                                                break;
                                            }
                                        }
                                        if skip {
                                            cs_pruned_ids.insert(sid);
                                            segments_minmax_skipped += 1;
                                            continue;
                                        }
                                    }
                                    if load_minmax {
                                        segments[seg_idx].col_minmax.insert(
                                            col_name.clone(),
                                            ColMinMax {
                                                min_encoded: row.min_encoded,
                                                max_encoded: row.max_encoded,
                                                min_null: row.min_null,
                                                max_null: row.max_null,
                                                type_oid: orig_type_oid,
                                            },
                                        );
                                    }
                                    if load_sums {
                                        segments[seg_idx].col_sums.insert(
                                            col_name.clone(),
                                            ColSum {
                                                sum_datum: pg_sys::Datum::from(0usize),
                                                sum_null: row.sum_null,
                                                sum_i128: row.sum_i128,
                                                sum_f64: row.sum_f64,
                                                nonnull_count: row.nonnull_count,
                                                nonzero_count: row.nonzero_count,
                                                type_oid: pg_sys::NUMERICOID,
                                            },
                                        );
                                    }
                                } else {
                                    all_found = false;
                                }
                            }
                            if all_found {
                                cached_idxs.push(ci);
                            }
                        }
                    }
                    for ci in cached_idxs {
                        needed_col_idxs.remove(&ci);
                    }
                });

                // If all needed col_idxs were served from cache, skip opening colstats table
                COLSTATS_CACHE.with(|cache| {
                    let cache = cache.borrow();
                    let cache_size = cache.len();
                    let has_oid = cache.keys().any(|&(oid, _)| oid == colstats_oid);
                    pgrx::log!(
                        "colstats_cache: oid={:?} remaining_uncached={} cache_entries={} has_oid={}",
                        colstats_oid,
                        needed_col_idxs.len(),
                        cache_size,
                        has_oid,
                    );
                });

                // Indexed minmax pruning: when every column we need from
                // colstats is the target of an equality minmax filter (the
                // common point-lookup shape), use the per-partition btree on
                // `(_col_idx, _min, _max)` to compute the surviving seg_ids
                // directly. Skips iterating ~all colstats rows on the slow
                // PK-scan path (heap_scan: ~30 ms → ~1 ms for queries like
                // `WHERE order_id = N`). Mirrors TimescaleDB's
                // `compress_hyper_*__ts_meta_min_*__ts_meta_max_*__t_idx`.
                let eq_filter_cols: Vec<(i16, i64)> = cs_minmax_filters
                    .iter()
                    .filter(|f| matches!(f.op, BatchCompareOp::Eq))
                    .map(|f| (f.col_idx, f.const_i64))
                    .collect();
                let all_needed_are_eq_filters = !eq_filter_cols.is_empty()
                    && needed_col_idxs.len() == eq_filter_cols.len()
                    && eq_filter_cols
                        .iter()
                        .all(|(ci, _)| needed_col_idxs.contains(ci));
                if all_needed_are_eq_filters
                    && let Some(survivors) =
                        lookup_segments_by_minmax_index(colstats_oid, &eq_filter_cols)
                {
                    // Mark every seg_id NOT in the survivor set as pruned.
                    for &sid in &surviving_segment_ids {
                        if !cs_pruned_ids.contains(&sid) && !survivors.contains(&sid) {
                            cs_pruned_ids.insert(sid);
                            segments_minmax_skipped += 1;
                        }
                    }
                    // Bypass the colstats heap scan — we already have every
                    // seg_id we need, and the caller didn't ask for cached
                    // min/max or sums (load_minmax false + empty
                    // needed_minmax_cols / needed_stats_cols is implied by
                    // needed_col_idxs == filter cols).
                    needed_col_idxs.clear();
                }

                if needed_col_idxs.is_empty() {
                    // Remove colstats-pruned segments
                    if !cs_pruned_ids.is_empty() {
                        let mut i = 0;
                        while i < segments.len() {
                            if cs_pruned_ids.contains(&surviving_segment_ids[i]) {
                                segments.swap_remove(i);
                                surviving_segment_ids.swap_remove(i);
                                segments_skipped += 1;
                            } else {
                                i += 1;
                            }
                        }
                    }
                } else {
                    // Open normalized colstats table and locate fixed columns
                    let cs_rel = pg_sys::table_open(
                        colstats_oid,
                        pg_sys::AccessShareLock as pg_sys::LOCKMODE,
                    );
                    let cs_tupdesc = (*cs_rel).rd_att;
                    let cs_natts = (*cs_tupdesc).natts as usize;

                    let mut cs_col_idx_att: Option<usize> = None;
                    let mut cs_seg_id_att: Option<usize> = None;
                    let mut cs_min_att: Option<usize> = None;
                    let mut cs_max_att: Option<usize> = None;
                    let mut cs_sum_att: Option<usize> = None;
                    let mut cs_nonnull_att: Option<usize> = None;
                    let mut cs_nonzero_att: Option<usize> = None;
                    let mut cs_ndistinct_att: Option<usize> = None;
                    for i in 0..cs_natts {
                        let att = &*tupdesc_get_attr(cs_tupdesc, i);
                        if att.attisdropped {
                            continue;
                        }
                        let name =
                            std::ffi::CStr::from_ptr(att.attname.data.as_ptr()).to_string_lossy();
                        match name.as_ref() {
                            "_col_idx" => cs_col_idx_att = Some(i),
                            "_segment_id" => cs_seg_id_att = Some(i),
                            "_min" => cs_min_att = Some(i),
                            "_max" => cs_max_att = Some(i),
                            "_sum" => cs_sum_att = Some(i),
                            "_nonnull_count" => cs_nonnull_att = Some(i),
                            "_nonzero_count" => cs_nonzero_att = Some(i),
                            "_ndistinct" => cs_ndistinct_att = Some(i),
                            _ => {}
                        }
                    }

                    // Decide: index scan (few columns) vs seq scan (many columns).
                    // Index scan reads only needed col_idx rows via PK (_col_idx, _segment_id).
                    // Threshold: use index scan if < 50% of columns needed.
                    let use_index_scan = needed_col_idxs.len() < idx_to_col.len() / 2 + 1
                        || needed_col_idxs.len() <= 4;

                    // Find PK index OID for index scan path
                    let pk_index_oid = if use_index_scan {
                        primary_key_index_oid(cs_rel)
                    } else {
                        pg_sys::InvalidOid
                    };

                    // Accumulate raw colstats rows into a per-(col_idx, segment_id) map
                    // for cache population, independent of pruning decisions.
                    let mut cs_raw_rows: HashMap<(i16, i32), CachedColStatsRow> = HashMap::new();

                    // Helper closure: process one colstats row from slot values/nulls
                    macro_rules! process_colstats_row {
                        ($vals:expr, $nls:expr, $ci_att:expr, $sid_att:expr,
                     $min_att:expr, $max_att:expr, $sum_att:expr, $nn_att:expr, $nz_att:expr) => {{
                            let seg_id = if !$nls[$sid_att] {
                                $vals[$sid_att].value() as i32
                            } else {
                                continue;
                            };
                            let seg_idx = match seg_id_to_idx.get(&seg_id) {
                                Some(&idx) => idx,
                                None => continue,
                            };

                            let col_idx_val = if !$nls[$ci_att] {
                                $vals[$ci_att].value() as i16
                            } else {
                                continue;
                            };
                            if col_idx_val < 0 || col_idx_val as usize >= idx_to_col.len() {
                                continue;
                            }
                            let (ref col_name, orig_type_oid) = idx_to_col[col_idx_val as usize];

                            let min_null = $nls[$min_att];
                            let max_null = $nls[$max_att];
                            let min_enc = if min_null {
                                0i64
                            } else {
                                $vals[$min_att].value() as i64
                            };
                            let max_enc = if max_null {
                                0i64
                            } else {
                                $vals[$max_att].value() as i64
                            };

                            // Extract sum data for both segment population and cache
                            let sum_null = $nls[$sum_att];
                            let sum_datum = if sum_null {
                                pg_sys::Datum::from(0usize)
                            } else {
                                $vals[$sum_att]
                            };
                            let nonnull_count = if $nls[$nn_att] {
                                0i64
                            } else {
                                $vals[$nn_att].value() as i64
                            };
                            let nonzero_count = if $nls[$nz_att] {
                                -1i64
                            } else {
                                $vals[$nz_att].value() as i64
                            };

                            // Convert NUMERIC sum to i128/f64 at scan time for caching
                            let (sum_i128, sum_f64): (Option<i128>, Option<f64>) = if sum_null {
                                (None, None)
                            } else {
                                let cstr = pg_sys::OidOutputFunctionCall(
                                    pg_sys::Oid::from(1702u32), // numeric_out
                                    sum_datum,
                                );
                                let s = std::ffi::CStr::from_ptr(cstr).to_string_lossy();
                                let i = s.parse::<i128>().ok();
                                let f = if i.is_none() {
                                    s.parse::<f64>().ok()
                                } else {
                                    None
                                };
                                pg_sys::pfree(cstr as *mut _);
                                (i, f)
                            };

                            // Store raw row for cache population (before pruning)
                            cs_raw_rows.insert(
                                (col_idx_val, seg_id),
                                CachedColStatsRow {
                                    min_encoded: min_enc,
                                    max_encoded: max_enc,
                                    min_null,
                                    max_null,
                                    sum_i128,
                                    sum_f64,
                                    sum_null,
                                    nonnull_count,
                                    nonzero_count,
                                },
                            );

                            // Apply pruning
                            if cs_pruned_ids.contains(&seg_id) {
                                continue;
                            }

                            if !cs_minmax_filters.is_empty() {
                                let mut skip = false;
                                for f in &cs_minmax_filters {
                                    if f.col_idx == col_idx_val
                                        && !min_null
                                        && !max_null
                                        && !segment_passes_minmax_filter(f, min_enc, max_enc)
                                    {
                                        skip = true;
                                        break;
                                    }
                                }
                                if skip {
                                    cs_pruned_ids.insert(seg_id);
                                    segments_minmax_skipped += 1;
                                    continue;
                                }
                            }

                            if load_minmax {
                                segments[seg_idx].col_minmax.insert(
                                    col_name.clone(),
                                    ColMinMax {
                                        min_encoded: min_enc,
                                        max_encoded: max_enc,
                                        min_null,
                                        max_null,
                                        type_oid: orig_type_oid,
                                    },
                                );
                            }

                            if load_sums {
                                let sum_type_oid = if !sum_null {
                                    let sum_attr = &*tupdesc_get_attr(cs_tupdesc, $sum_att);
                                    sum_attr.atttypid
                                } else {
                                    pg_sys::NUMERICOID
                                };
                                segments[seg_idx].col_sums.insert(
                                    col_name.clone(),
                                    ColSum {
                                        sum_datum,
                                        sum_null,
                                        sum_i128,
                                        sum_f64,
                                        nonnull_count,
                                        nonzero_count,
                                        type_oid: sum_type_oid,
                                    },
                                );
                            }
                        }};
                    }

                    if let (
                        Some(ci_att),
                        Some(sid_att),
                        Some(min_att),
                        Some(max_att),
                        Some(sum_att),
                        Some(nn_att),
                        Some(nz_att),
                        Some(_nd_att),
                    ) = (
                        cs_col_idx_att,
                        cs_seg_id_att,
                        cs_min_att,
                        cs_max_att,
                        cs_sum_att,
                        cs_nonnull_att,
                        cs_nonzero_att,
                        cs_ndistinct_att,
                    ) {
                        if use_index_scan && pk_index_oid != pg_sys::InvalidOid {
                            // Index scan path: one scan per needed col_idx
                            let cs_snapshot = pg_sys::GetActiveSnapshot();
                            let idx_rel = pg_sys::index_open(
                                pk_index_oid,
                                pg_sys::AccessShareLock as pg_sys::LOCKMODE,
                            );
                            let slot = pg_sys::table_slot_create(cs_rel, std::ptr::null_mut());

                            for &col_idx_val in &needed_col_idxs {
                                let mut skey = [pg_sys::ScanKeyData::default()];
                                pg_sys::ScanKeyInit(
                                    &mut skey[0],
                                    1, // attnum 1 = _col_idx (first column in PK)
                                    pg_sys::BTEqualStrategyNumber as u16,
                                    pg_sys::F_INT2EQ.into(),
                                    pg_sys::Datum::from(col_idx_val),
                                );

                                let scan =
                                    index_beginscan_compat(cs_rel, idx_rel, cs_snapshot, 1, 0);
                                pg_sys::index_rescan(
                                    scan,
                                    skey.as_mut_ptr(),
                                    1,
                                    std::ptr::null_mut(),
                                    0,
                                );

                                loop {
                                    if !pg_sys::index_getnext_slot(
                                        scan,
                                        pg_sys::ScanDirection::ForwardScanDirection,
                                        slot,
                                    ) {
                                        break;
                                    }
                                    pg_sys::slot_getallattrs(slot);
                                    let tts_values =
                                        std::slice::from_raw_parts((*slot).tts_values, cs_natts);
                                    let tts_isnull =
                                        std::slice::from_raw_parts((*slot).tts_isnull, cs_natts);

                                    process_colstats_row!(
                                        tts_values, tts_isnull, ci_att, sid_att, min_att, max_att,
                                        sum_att, nn_att, nz_att
                                    );
                                }

                                pg_sys::index_endscan(scan);
                            }

                            pg_sys::ExecDropSingleTupleTableSlot(slot);
                            pg_sys::index_close(
                                idx_rel,
                                pg_sys::AccessShareLock as pg_sys::LOCKMODE,
                            );
                        } else {
                            // Seq scan path: scan all rows, filter by needed col_idx
                            let cs_snapshot = pg_sys::GetActiveSnapshot();
                            let cs_flags: u32 = pg_sys::ScanOptions::SO_TYPE_SEQSCAN
                                | pg_sys::ScanOptions::SO_ALLOW_STRAT
                                | pg_sys::ScanOptions::SO_ALLOW_SYNC
                                | pg_sys::ScanOptions::SO_ALLOW_PAGEMODE;
                            let cs_scan = (*(*cs_rel).rd_tableam).scan_begin.unwrap()(
                                cs_rel,
                                cs_snapshot,
                                0,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                                cs_flags,
                            );

                            let mut cs_values = vec![pg_sys::Datum::from(0); cs_natts];
                            let mut cs_nulls = vec![true; cs_natts];

                            loop {
                                let tuple = pg_sys::heap_getnext(
                                    cs_scan,
                                    pg_sys::ScanDirection::ForwardScanDirection,
                                );
                                if tuple.is_null() {
                                    break;
                                }

                                pg_sys::heap_deform_tuple(
                                    tuple,
                                    cs_tupdesc,
                                    cs_values.as_mut_ptr(),
                                    cs_nulls.as_mut_ptr(),
                                );

                                // Skip columns we don't need in seq scan path
                                if !needed_col_idxs.is_empty() {
                                    let ci = if !cs_nulls[ci_att] {
                                        cs_values[ci_att].value() as i16
                                    } else {
                                        continue;
                                    };
                                    if !needed_col_idxs.contains(&ci) {
                                        continue;
                                    }
                                }

                                process_colstats_row!(
                                    cs_values, cs_nulls, ci_att, sid_att, min_att, max_att,
                                    sum_att, nn_att, nz_att
                                );
                            }

                            (*(*cs_rel).rd_tableam).scan_end.unwrap()(cs_scan);
                        }
                    }

                    pg_sys::table_close(cs_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

                    // Populate colstats cache from the raw rows collected during scan.
                    // Uses cs_raw_rows which includes all segments (even pruned ones).
                    COLSTATS_CACHE.with(|cache| {
                        let mut cache = cache.borrow_mut();
                        for ((ci, sid), row) in cs_raw_rows.drain() {
                            let entry =
                                cache
                                    .entry((colstats_oid, ci))
                                    .or_insert_with(|| CachedColStats {
                                        rows: HashMap::new(),
                                    });
                            entry.rows.insert(sid, row);
                        }
                    });

                    // Remove colstats-pruned segments
                    if !cs_pruned_ids.is_empty() {
                        let mut i = 0;
                        while i < segments.len() {
                            if cs_pruned_ids.contains(&surviving_segment_ids[i]) {
                                segments.swap_remove(i);
                                surviving_segment_ids.swap_remove(i);
                                segments_skipped += 1;
                            } else {
                                i += 1;
                            }
                        }
                    }
                } // end else (uncached col_idxs scan)
            }
        }

        let (t1b_hit, t1b_read) = shared_buf_snapshot();
        buf_stats.meta_hit += t1b_hit - t1_hit;
        buf_stats.meta_read += t1b_read - t1_read;

        // ================================================================
        // Bloom phase: probe per-segment blooms to prune surviving segments.
        // ================================================================
        if !bloom_checks.is_empty() && !segments.is_empty() {
            let surviving: std::collections::HashSet<i32> =
                surviving_segment_ids.iter().copied().collect();
            let bloom_pruned_ids = bloom_pruned_segment_ids(meta_oid, &bloom_checks, &surviving);

            // Remove bloom-pruned segments (segments and surviving_segment_ids are parallel)
            if !bloom_pruned_ids.is_empty() {
                let before = segments.len();
                let mut i = 0;
                while i < segments.len() {
                    if bloom_pruned_ids.contains(&surviving_segment_ids[i]) {
                        segments.swap_remove(i);
                        surviving_segment_ids.swap_remove(i);
                    } else {
                        i += 1;
                    }
                }
                let pruned = before - segments.len();
                segments_skipped += pruned as u64;
                segments_bloom_skipped += pruned as u64;
            }
        }

        let (t2_hit, t2_read) = shared_buf_snapshot();
        buf_stats.bloom_hit = t2_hit - t1b_hit;
        buf_stats.bloom_read = t2_read - t1b_read;

        // ----------------------------------------------------------------
        // Segment pruning via per-segment value-presence bitmap (text Eq).
        // Mirrors the bloom block above: open `<partition>_valbitmap` by
        // PK on `(_col_idx, _segment_id)`, fetch `_bits`, test the bit
        // recorded in `valmap` for the queried constant. Exact (no false
        // positives), so a clear bit guarantees the segment can be skipped.
        //
        // Presence is decided strictly per segment: only segments that wrote
        // a bitmap row for the column are candidates for pruning. A segment
        // with no row exceeded VALBITMAP_MAX_DISTINCT at compress time — its
        // values are not covered by the partition valmap, so neither a clear
        // bit nor a valmap miss (empty `wanted_bits`) says anything about it.
        // ----------------------------------------------------------------
        if !valbitmap_checks.is_empty() && !segments.is_empty() {
            let meta_name_ptr = pg_sys::get_rel_name(meta_oid);
            let meta_name_str = std::ffi::CStr::from_ptr(meta_name_ptr)
                .to_string_lossy()
                .into_owned();
            let meta_ns_oid = pg_sys::get_rel_namespace(meta_oid);
            let partition_name = meta_name_str
                .strip_suffix("_meta")
                .unwrap_or(&meta_name_str);
            let valbitmap_name = format!("{}_valbitmap", partition_name);
            let valbitmap_cname = std::ffi::CString::new(valbitmap_name).unwrap();
            let valbitmap_oid = pg_sys::get_relname_relid(valbitmap_cname.as_ptr(), meta_ns_oid);

            if valbitmap_oid != pg_sys::InvalidOid {
                let mut seg_id_to_idx: HashMap<i32, usize> = HashMap::new();
                for (idx, &sid) in surviving_segment_ids.iter().enumerate() {
                    seg_id_to_idx.insert(sid, idx);
                }

                let vb_rel =
                    pg_sys::table_open(valbitmap_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                let vb_tupdesc = (*vb_rel).rd_att;
                let vb_natts = (*vb_tupdesc).natts as usize;

                let mut seg_id_att: Option<usize> = None;
                let mut bits_att: Option<usize> = None;
                for i in 0..vb_natts {
                    let attr = &*tupdesc_get_attr(vb_tupdesc, i);
                    let name =
                        std::ffi::CStr::from_ptr(attr.attname.data.as_ptr()).to_string_lossy();
                    if name == "_segment_id" {
                        seg_id_att = Some(i);
                    } else if name == "_bits" {
                        bits_att = Some(i);
                    }
                }

                let pk_index_oid = primary_key_index_oid(vb_rel);

                if let (Some(sid_att), Some(bits_a), true) =
                    (seg_id_att, bits_att, pk_index_oid != pg_sys::InvalidOid)
                {
                    let snapshot = pg_sys::GetActiveSnapshot();
                    let idx_rel = pg_sys::index_open(
                        pk_index_oid,
                        pg_sys::AccessShareLock as pg_sys::LOCKMODE,
                    );
                    let mut vb_pruned_ids: std::collections::HashSet<i32> =
                        std::collections::HashSet::new();

                    // Valmap-miss checks carry an empty `wanted_bits`,
                    // so every segment with a bitmap row fails `passes`
                    // below and is pruned; segments without a row
                    // survive.
                    for vc in &valbitmap_checks {
                        let mut skey = [pg_sys::ScanKeyData::default()];
                        pg_sys::ScanKeyInit(
                            &mut skey[0],
                            1, // attnum 1 = _col_idx
                            pg_sys::BTEqualStrategyNumber as u16,
                            pg_sys::F_INT2EQ.into(),
                            pg_sys::Datum::from(vc.col_idx as i16),
                        );

                        let scan = index_beginscan_compat(vb_rel, idx_rel, snapshot, 1, 0);
                        pg_sys::index_rescan(scan, skey.as_mut_ptr(), 1, std::ptr::null_mut(), 0);

                        let slot = pg_sys::table_slot_create(vb_rel, std::ptr::null_mut());

                        loop {
                            if !pg_sys::index_getnext_slot(
                                scan,
                                pg_sys::ScanDirection::ForwardScanDirection,
                                slot,
                            ) {
                                break;
                            }
                            pg_sys::slot_getallattrs(slot);
                            let tts_values = (*slot).tts_values;
                            let tts_isnull = (*slot).tts_isnull;
                            if *tts_isnull.add(sid_att) || *tts_isnull.add(bits_a) {
                                continue;
                            }
                            let seg_id = (*tts_values.add(sid_att)).value() as i32;
                            if !seg_id_to_idx.contains_key(&seg_id) {
                                continue;
                            }

                            let varlena_ptr =
                                (*tts_values.add(bits_a)).cast_mut_ptr::<pg_sys::varlena>();
                            let detoasted = pg_sys::pg_detoast_datum(varlena_ptr);
                            let data_ptr = pgrx::vardata_any(detoasted);
                            let data_len = pgrx::varsize_any_exhdr(detoasted);
                            #[allow(clippy::unnecessary_cast)]
                            let bits = std::slice::from_raw_parts(data_ptr as *const u8, data_len);

                            // A segment passes if any wanted bit is set.
                            let passes = vc.wanted_bits.iter().any(|&bi| {
                                let byte = (bi / 8) as usize;
                                let mask = 1u8 << (bi % 8);
                                byte < bits.len() && (bits[byte] & mask) != 0
                            });

                            if detoasted != varlena_ptr {
                                pg_sys::pfree(detoasted as *mut _);
                            }

                            if !passes {
                                vb_pruned_ids.insert(seg_id);
                            }
                        }

                        pg_sys::ExecDropSingleTupleTableSlot(slot);
                        pg_sys::index_endscan(scan);
                    }

                    pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

                    if !vb_pruned_ids.is_empty() {
                        let before = segments.len();
                        let mut i = 0;
                        while i < segments.len() {
                            if vb_pruned_ids.contains(&surviving_segment_ids[i]) {
                                segments.swap_remove(i);
                                surviving_segment_ids.swap_remove(i);
                            } else {
                                i += 1;
                            }
                        }
                        let pruned = before - segments.len();
                        segments_skipped += pruned as u64;
                        segments_valbitmap_skipped += pruned as u64;
                    }
                }

                pg_sys::table_close(vb_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            }
        }

        pgrx::log!(
            "load_segments_heap phase1: segments={} skipped={} (minmax={} bloom={} valbitmap={}) heap_getnext={:.1}ms deform={:.1}ms",
            segments.len(),
            segments_skipped,
            segments_minmax_skipped,
            segments_bloom_skipped,
            segments_valbitmap_skipped,
            heap_getnext_us as f64 / 1000.0,
            deform_us as f64 / 1000.0,
        );

        // ================================================================
        // Phase 2: Scan blob table — sequential TOAST I/O per column
        // ================================================================
        let mut detoast_us: u64 = 0;

        // Check if any blobs are needed
        let any_blobs_needed = col_names
            .iter()
            .enumerate()
            .any(|(i, name)| !segment_by.contains(name) && needed_cols[i]);

        if !segments.is_empty() && any_blobs_needed && !skip_blob_load {
            // Derive blob table OID from meta table name
            let meta_name_ptr = pg_sys::get_rel_name(meta_oid);
            let meta_name = std::ffi::CStr::from_ptr(meta_name_ptr)
                .to_string_lossy()
                .into_owned();
            let meta_ns_oid = pg_sys::get_rel_namespace(meta_oid);

            // Strip "_meta" suffix to get partition name, then add "_blobs"
            let partition_name = meta_name.strip_suffix("_meta").unwrap_or(&meta_name);
            let blobs_name = format!("{}_blobs", partition_name);
            let blobs_cname = std::ffi::CString::new(blobs_name).unwrap();
            let blob_oid = pg_sys::get_relname_relid(blobs_cname.as_ptr(), meta_ns_oid);

            if blob_oid != pg_sys::InvalidOid {
                // Build surviving segment_id → segment index mapping
                let mut seg_id_to_idx: HashMap<i32, usize> = HashMap::new();
                for (idx, &sid) in surviving_segment_ids.iter().enumerate() {
                    seg_id_to_idx.insert(sid, idx);
                }

                // Determine which col_idx values we need. `col_idx_map[i] = None`
                // for segment_by columns AND for columns added to the parent
                // after this partition was compressed — both have no blob to
                // fetch (the latter get synthesized later via getmissingattr).
                let mut needed_col_indices: Vec<(u16, usize)> = Vec::new(); // (col_idx, blob_slot_idx)
                for i in 0..col_names.len() {
                    let Some(ci) = col_idx_map[i] else {
                        continue;
                    };
                    if needed_cols[i] {
                        needed_col_indices.push((ci, ci as usize));
                    }
                }

                // Open blob table + its PK index
                let blob_rel =
                    pg_sys::table_open(blob_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                let blob_tupdesc = (*blob_rel).rd_att;

                // Find PK index OID — first index that is primary
                let pk_index_oid = primary_key_index_oid(blob_rel);

                let detoast_start = std::time::Instant::now();

                if pk_index_oid != pg_sys::InvalidOid {
                    let idx_rel = pg_sys::index_open(
                        pk_index_oid,
                        pg_sys::AccessShareLock as pg_sys::LOCKMODE,
                    );

                    for &(col_idx, blob_slot) in &needed_col_indices {
                        let is_lazy = lazy_cols.is_some_and(|lc| {
                            // Find the original col_names index for this col_idx
                            col_names.iter().enumerate().any(|(i, name)| {
                                !segment_by.contains(name)
                                    && col_idx_map[i] == Some(col_idx)
                                    && i < lc.len()
                                    && lc[i]
                            })
                        });

                        // Set up scan key: _col_idx = col_idx (SMALLINT equality)
                        let mut skey = [pg_sys::ScanKeyData::default()];
                        pg_sys::ScanKeyInit(
                            &mut skey[0],
                            1, // attnum 1 = _col_idx
                            pg_sys::BTEqualStrategyNumber as u16,
                            pg_sys::F_INT2EQ.into(),
                            pg_sys::Datum::from(col_idx as i16),
                        );

                        let scan = index_beginscan_compat(blob_rel, idx_rel, snapshot, 1, 0);
                        pg_sys::index_rescan(scan, skey.as_mut_ptr(), 1, std::ptr::null_mut(), 0);

                        // Allocate slot for tuple extraction
                        let slot = pg_sys::table_slot_create(blob_rel, std::ptr::null_mut());

                        loop {
                            if !pg_sys::index_getnext_slot(
                                scan,
                                pg_sys::ScanDirection::ForwardScanDirection,
                                slot,
                            ) {
                                break;
                            }

                            // Extract _segment_id (attnum 2) and _data (attnum 3)
                            let mut blob_values = [pg_sys::Datum::from(0); 3];
                            let mut blob_nulls = [true; 3];
                            pg_sys::slot_getallattrs(slot);
                            let tts_values = (*slot).tts_values;
                            let tts_isnull = (*slot).tts_isnull;
                            for j in 0..3usize {
                                blob_values[j] = *tts_values.add(j);
                                blob_nulls[j] = *tts_isnull.add(j);
                            }

                            if blob_nulls[1] {
                                continue; // no segment_id — skip
                            }
                            let seg_id = blob_values[1].value() as i32;

                            // Check if this segment survived pruning
                            let seg_idx = match seg_id_to_idx.get(&seg_id) {
                                Some(&idx) => idx,
                                None => continue, // pruned — skip without detoasting
                            };

                            if blob_nulls[2] {
                                // null blob — leave empty
                                continue;
                            }

                            if is_lazy {
                                // Lazy: copy just the TOAST pointer
                                let varlena_ptr = blob_values[2].cast_mut_ptr::<pg_sys::varlena>();
                                let ptr_size = pgrx::varsize_any(varlena_ptr);
                                let mut ptr_copy = vec![0u8; ptr_size];
                                std::ptr::copy_nonoverlapping(
                                    varlena_ptr as *const u8,
                                    ptr_copy.as_mut_ptr(),
                                    ptr_size,
                                );
                                segments[seg_idx].toast_pointers[blob_slot] = ptr_copy;
                            } else {
                                // Eager path: try the cache, fall back to detoast.
                                let cache_key = crate::blob_cache::BlobCacheKey::new(
                                    meta_oid, seg_id, blob_slot,
                                );
                                if let Some(pin) = crate::blob_cache::get_pinned(&cache_key) {
                                    let s = pin.as_slice();
                                    segments[seg_idx].compressed_blobs[blob_slot] =
                                        BlobBytes::Cached {
                                            data: s.as_ptr(),
                                            len: s.len() as u32,
                                        };
                                    segments[seg_idx].cached_blob_pins.push(pin);
                                } else {
                                    let varlena_ptr: *mut pg_sys::varlena =
                                        blob_values[2].cast_mut_ptr();
                                    let bytes = detoast_varlena_to_vec(varlena_ptr);
                                    crate::blob_cache::insert(&cache_key, &bytes);
                                    segments[seg_idx].compressed_blobs[blob_slot] =
                                        BlobBytes::Owned(bytes);
                                }
                            }
                        }

                        pg_sys::ExecDropSingleTupleTableSlot(slot);
                        pg_sys::index_endscan(scan);
                    }

                    pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
                } else {
                    // Fallback: sequential scan of blob table (no PK index found)
                    let blob_flags: u32 = pg_sys::ScanOptions::SO_TYPE_SEQSCAN
                        | pg_sys::ScanOptions::SO_ALLOW_STRAT
                        | pg_sys::ScanOptions::SO_ALLOW_SYNC
                        | pg_sys::ScanOptions::SO_ALLOW_PAGEMODE;
                    let blob_scan = (*(*blob_rel).rd_tableam).scan_begin.unwrap()(
                        blob_rel,
                        snapshot,
                        0,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        blob_flags,
                    );

                    let blob_natts = (*blob_tupdesc).natts as usize;
                    let mut bv = vec![pg_sys::Datum::from(0); blob_natts];
                    let mut bn = vec![true; blob_natts];

                    // Build set of needed col indices for fast lookup
                    let needed_set: std::collections::HashSet<u16> =
                        needed_col_indices.iter().map(|&(ci, _)| ci).collect();

                    loop {
                        let tuple = pg_sys::heap_getnext(
                            blob_scan,
                            pg_sys::ScanDirection::ForwardScanDirection,
                        );
                        if tuple.is_null() {
                            break;
                        }
                        pg_sys::heap_deform_tuple(
                            tuple,
                            blob_tupdesc,
                            bv.as_mut_ptr(),
                            bn.as_mut_ptr(),
                        );

                        if bn[0] || bn[1] {
                            continue;
                        }
                        let ci = bv[0].value() as u16;
                        let seg_id = bv[1].value() as i32;

                        if !needed_set.contains(&ci) {
                            continue;
                        }
                        let seg_idx = match seg_id_to_idx.get(&seg_id) {
                            Some(&idx) => idx,
                            None => continue,
                        };
                        if bn[2] {
                            continue;
                        }

                        let blob_slot = ci as usize;
                        let is_lazy = lazy_cols.is_some_and(|lc| {
                            col_names.iter().enumerate().any(|(i, name)| {
                                !segment_by.contains(name)
                                    && col_idx_map[i] == Some(ci)
                                    && i < lc.len()
                                    && lc[i]
                            })
                        });

                        if is_lazy {
                            let varlena_ptr = bv[2].cast_mut_ptr::<pg_sys::varlena>();
                            let ptr_size = pgrx::varsize_any(varlena_ptr);
                            let mut ptr_copy = vec![0u8; ptr_size];
                            std::ptr::copy_nonoverlapping(
                                varlena_ptr as *const u8,
                                ptr_copy.as_mut_ptr(),
                                ptr_size,
                            );
                            segments[seg_idx].toast_pointers[blob_slot] = ptr_copy;
                        } else {
                            let cache_key =
                                crate::blob_cache::BlobCacheKey::new(meta_oid, seg_id, blob_slot);
                            if let Some(pin) = crate::blob_cache::get_pinned(&cache_key) {
                                let s = pin.as_slice();
                                segments[seg_idx].compressed_blobs[blob_slot] = BlobBytes::Cached {
                                    data: s.as_ptr(),
                                    len: s.len() as u32,
                                };
                                segments[seg_idx].cached_blob_pins.push(pin);
                            } else {
                                let varlena_ptr: *mut pg_sys::varlena = bv[2].cast_mut_ptr();
                                let bytes = detoast_varlena_to_vec(varlena_ptr);
                                crate::blob_cache::insert(&cache_key, &bytes);
                                segments[seg_idx].compressed_blobs[blob_slot] =
                                    BlobBytes::Owned(bytes);
                            }
                        }
                    }

                    (*(*blob_rel).rd_tableam).scan_end.unwrap()(blob_scan);
                }

                detoast_us = detoast_start.elapsed().as_micros() as u64;

                pg_sys::table_close(blob_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            }
        }

        let (t3_hit, t3_read) = shared_buf_snapshot();
        buf_stats.blob_hit = t3_hit - t2_hit;
        buf_stats.blob_read = t3_read - t2_read;

        // P2.5: attach tombstone offsets so every decode path can filter
        // logically deleted rows and metadata consumers can adjust/bail.
        attach_tombstones(meta_oid, &mut segments);

        accumulate_scan_buf_stats(&buf_stats);

        pgrx::log!(
            "load_segments_heap phase2: segments={} skipped={} detoast={:.1}ms",
            segments.len(),
            segments_skipped,
            detoast_us as f64 / 1000.0,
        );

        (
            segments,
            segments_skipped,
            segments_minmax_skipped,
            segments_bloom_skipped,
            segments_valbitmap_skipped,
            detoast_us,
        )
    }
}

/// Load text-length sidecar blobs for the columns marked sidecar-only, writing
/// them into each segment's `text_length_blobs[blob_slot]`. Returns the elapsed
/// detoast time in microseconds.
///
/// Uses an index scan on the `<partition>_text_lengths` PK (same pattern as the
/// main blob loader). Silently no-ops when the table doesn't exist (old data
/// compressed before the sidecar was introduced).
pub(super) unsafe fn load_text_length_sidecars(
    meta_oid: pg_sys::Oid,
    col_names: &[String],
    sidecar_cols: &[bool],
    // Persisted `_col_idx` map — see `load_segments_heap`'s `blob_idx`
    // parameter. None entries (segment_by or ADD-COLUMN-after-compression)
    // have no sidecar to load.
    blob_idx: &[Option<u16>],
    segments: &mut [SegmentData],
) -> u64 {
    if segments.is_empty() || !sidecar_cols.iter().any(|&s| s) {
        return 0;
    }

    unsafe {
        let tl_oid = sibling_table_oid(meta_oid, "_text_lengths");
        if tl_oid == pg_sys::InvalidOid {
            // Data compressed before the sidecar feature — no sidecar to load.
            return 0;
        }

        debug_assert_eq!(col_names.len(), blob_idx.len());
        let col_idx_map: &[Option<u16>] = blob_idx;

        // Determine which col_idx values we need sidecars for
        let mut needed_col_idxs: Vec<u16> = Vec::new();
        for (i, &is_sidecar) in sidecar_cols.iter().enumerate() {
            if is_sidecar && let Some(ci) = col_idx_map[i] {
                needed_col_idxs.push(ci);
            }
        }
        if needed_col_idxs.is_empty() {
            return 0;
        }

        // Build segment_id -> index-in-segments map
        let mut seg_id_to_idx: HashMap<i32, usize> = HashMap::new();
        for (idx, seg) in segments.iter().enumerate() {
            seg_id_to_idx.insert(seg.segment_id, idx);
        }

        let t_start = std::time::Instant::now();

        let rel = pg_sys::table_open(tl_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let pk_index_oid = primary_key_index_oid(rel);
        let snapshot = pg_sys::GetActiveSnapshot();

        if pk_index_oid != pg_sys::InvalidOid {
            let idx_rel =
                pg_sys::index_open(pk_index_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

            for &col_idx in &needed_col_idxs {
                let mut skey = [pg_sys::ScanKeyData::default()];
                pg_sys::ScanKeyInit(
                    &mut skey[0],
                    1, // _col_idx
                    pg_sys::BTEqualStrategyNumber as u16,
                    pg_sys::F_INT2EQ.into(),
                    pg_sys::Datum::from(col_idx as i16),
                );

                let scan = index_beginscan_compat(rel, idx_rel, snapshot, 1, 0);
                pg_sys::index_rescan(scan, skey.as_mut_ptr(), 1, std::ptr::null_mut(), 0);

                let slot = pg_sys::table_slot_create(rel, std::ptr::null_mut());

                loop {
                    if !pg_sys::index_getnext_slot(
                        scan,
                        pg_sys::ScanDirection::ForwardScanDirection,
                        slot,
                    ) {
                        break;
                    }
                    pg_sys::slot_getallattrs(slot);
                    let tts_values = (*slot).tts_values;
                    let tts_isnull = (*slot).tts_isnull;

                    // attnum 2 = _segment_id, attnum 3 = _data
                    if *tts_isnull.add(1) || *tts_isnull.add(2) {
                        continue;
                    }
                    let seg_id = (*tts_values.add(1)).value() as i32;
                    let seg_idx = match seg_id_to_idx.get(&seg_id) {
                        Some(&i) => i,
                        None => continue, // pruned
                    };

                    let varlena_ptr: *mut pg_sys::varlena = (*tts_values.add(2)).cast_mut_ptr();
                    let bytes = detoast_varlena_to_vec(varlena_ptr);
                    segments[seg_idx].text_length_blobs[col_idx as usize] = bytes;
                }

                pg_sys::ExecDropSingleTupleTableSlot(slot);
                pg_sys::index_endscan(scan);
            }

            pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        }

        pg_sys::table_close(rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);

        t_start.elapsed().as_micros() as u64
    }
}

/// Fetch compressed blobs for a single segment via `(_col_idx, _segment_id)`
/// PK lookup on the companion `_blobs` table. Populates
/// `seg.compressed_blobs[col_idx]` for every non-segment-by column marked in
/// `needed_cols`, detoasting each value in place. Idempotent (skips columns
/// already populated).
///
/// Called on-claim from `load_next_segment` (parallel and serial paths after
/// §5.7 DSM sharing): instead of the leader eagerly detoasting every
/// segment's blobs in `load_segments_heap`, each claimant fetches only the
/// blobs for segments it actually processes — so blob I/O is parallelised
/// across workers.
pub(super) unsafe fn fetch_segment_blobs(
    companion_oid: pg_sys::Oid,
    segment_id: i32,
    col_names: &[String],
    needed_cols: &[bool],
    // Persisted `_col_idx` map — see `load_segments_heap`'s `blob_idx`
    // parameter. None entries (segment_by or ADD-COLUMN-after-compression)
    // have no blob to fetch.
    blob_idx: &[Option<u16>],
    seg: &mut SegmentData,
) -> u64 {
    let t_start = std::time::Instant::now();
    unsafe {
        debug_assert_eq!(col_names.len(), blob_idx.len());
        // Pre-size blob slots if empty (first fetch). `num_blob_cols` is the
        // count of distinct `_col_idx` slots referenced by the descriptor.
        let num_blob_cols = blob_idx.iter().filter(|b| b.is_some()).count();
        if seg.compressed_blobs.is_empty() {
            seg.compressed_blobs = Vec::with_capacity(num_blob_cols);
            seg.compressed_blobs
                .resize_with(num_blob_cols, BlobBytes::default);
        }

        let blob_oid = sibling_table_oid(companion_oid, "_blobs");
        if blob_oid == pg_sys::InvalidOid {
            return t_start.elapsed().as_micros() as u64;
        }

        let col_idx_map: &[Option<u16>] = blob_idx;

        let blob_rel = pg_sys::table_open(blob_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let pk_index_oid = primary_key_index_oid(blob_rel);

        if pk_index_oid == pg_sys::InvalidOid {
            pg_sys::table_close(blob_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
            return t_start.elapsed().as_micros() as u64;
        }

        let idx_rel = pg_sys::index_open(pk_index_oid, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        let snapshot = pg_sys::GetActiveSnapshot();

        for i in 0..col_names.len() {
            if !needed_cols[i] {
                continue;
            }
            let col_idx = match col_idx_map[i] {
                Some(ci) => ci,
                None => continue,
            };
            let blob_slot = col_idx as usize;
            if !seg.compressed_blobs[blob_slot].is_empty() {
                continue; // already fetched
            }

            // Cache fast path: skip the index lookup + heap I/O + detoast
            // entirely if this (companion, segment, col) blob is already
            // in the shared blob cache. The pin keeps the DSA bytes alive
            // for the lifetime of the segment.
            let cache_key =
                crate::blob_cache::BlobCacheKey::new(companion_oid, segment_id, blob_slot);
            if let Some(pin) = crate::blob_cache::get_pinned(&cache_key) {
                let s = pin.as_slice();
                seg.compressed_blobs[blob_slot] = BlobBytes::Cached {
                    data: s.as_ptr(),
                    len: s.len() as u32,
                };
                seg.cached_blob_pins.push(pin);
                continue;
            }

            // Two-column PK scankey: (_col_idx = ci, _segment_id = seg_id).
            let mut skeys = [pg_sys::ScanKeyData::default(); 2];
            pg_sys::ScanKeyInit(
                &mut skeys[0],
                1,
                pg_sys::BTEqualStrategyNumber as u16,
                pg_sys::F_INT2EQ.into(),
                pg_sys::Datum::from(col_idx as i16),
            );
            pg_sys::ScanKeyInit(
                &mut skeys[1],
                2,
                pg_sys::BTEqualStrategyNumber as u16,
                pg_sys::F_INT4EQ.into(),
                pg_sys::Datum::from(segment_id),
            );

            let scan = index_beginscan_compat(blob_rel, idx_rel, snapshot, 2, 0);
            pg_sys::index_rescan(scan, skeys.as_mut_ptr(), 2, std::ptr::null_mut(), 0);

            let slot = pg_sys::table_slot_create(blob_rel, std::ptr::null_mut());

            if pg_sys::index_getnext_slot(scan, pg_sys::ScanDirection::ForwardScanDirection, slot) {
                pg_sys::slot_getallattrs(slot);
                let tts_isnull = (*slot).tts_isnull;
                let tts_values = (*slot).tts_values;
                let data_null = *tts_isnull.add(2);
                if !data_null {
                    let data_datum = *tts_values.add(2);
                    let varlena_ptr: *mut pg_sys::varlena = data_datum.cast_mut_ptr();
                    let bytes = detoast_varlena_to_vec(varlena_ptr);
                    crate::blob_cache::insert(&cache_key, &bytes);
                    seg.compressed_blobs[blob_slot] = BlobBytes::Owned(bytes);
                }
            }

            pg_sys::ExecDropSingleTupleTableSlot(slot);
            pg_sys::index_endscan(scan);
        }

        pg_sys::index_close(idx_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
        pg_sys::table_close(blob_rel, pg_sys::AccessShareLock as pg_sys::LOCKMODE);
    }
    t_start.elapsed().as_micros() as u64
}

/// Materialize a single blob slot from the cache (on hit) or via
/// `pg_detoast_datum` (on miss). On miss, the freshly-detoasted bytes
/// are also inserted into the cache best-effort.
///
/// Returns `(bytes_served_from_cache, hit)`. `hit` is true when the
/// blob came from the cache; the caller can use this to bump per-query
/// stats counters.
unsafe fn detoast_blob_slot(seg: &mut SegmentData, bi: usize) -> (u64, bool) {
    unsafe {
        let key = crate::blob_cache::BlobCacheKey::new(seg.companion_oid, seg.segment_id, bi);
        if let Some(pin) = crate::blob_cache::get_pinned(&key) {
            let slice = pin.as_slice();
            let len = slice.len() as u64;
            // Borrow directly from the pin — no memcpy. The pin lives
            // in cached_blob_pins until SegmentData drops, and
            // compressed_blobs is declared before cached_blob_pins so
            // the BlobBytes::Cached pointers drop first.
            seg.compressed_blobs[bi] = BlobBytes::Cached {
                data: slice.as_ptr(),
                len: slice.len() as u32,
            };
            seg.toast_pointers[bi].clear();
            seg.cached_blob_pins.push(pin);
            return (len, true);
        }

        let ptr = seg.toast_pointers[bi].as_ptr() as *mut pg_sys::varlena;
        let bytes = detoast_varlena_to_vec(ptr);
        crate::blob_cache::insert(&key, &bytes);
        seg.compressed_blobs[bi] = BlobBytes::Owned(bytes);
        seg.toast_pointers[bi].clear();
        (0, false)
    }
}

/// Materialize deferred TOAST pointers for a segment.
///
/// For each blob index that has a non-empty toast_pointer, calls pg_detoast_datum
/// on the stored pointer copy and replaces the empty compressed_blob with the
/// detoasted data. Clears the toast_pointer after detoasting.
///
/// Returns the [`DetoastLazyStats`] aggregated over all blobs that were
/// materialised on this call.
pub(super) unsafe fn detoast_lazy_blobs(seg: &mut SegmentData) -> DetoastLazyStats {
    let mut stats = DetoastLazyStats::default();
    unsafe {
        for bi in 0..seg.toast_pointers.len() {
            if seg.toast_pointers[bi].is_empty() {
                continue;
            }
            let (bytes_from_cache, hit) = detoast_blob_slot(seg, bi);
            if hit {
                stats.cache_hits += 1;
                stats.cache_bytes_served += bytes_from_cache;
            } else {
                stats.cache_misses += 1;
            }
        }
    }
    stats
}

/// Materialize deferred TOAST pointers for specific blob indices only.
///
/// Like `detoast_lazy_blobs` but only processes the given blob indices,
/// leaving other blobs lazy. Used in top-N Phase 1 to detoast only
/// filter + sort column blobs while deferring Phase 2 columns.
pub(super) unsafe fn detoast_lazy_blobs_selective(
    seg: &mut SegmentData,
    blob_indices: &[usize],
) -> DetoastLazyStats {
    let mut stats = DetoastLazyStats::default();
    unsafe {
        for &bi in blob_indices {
            if bi >= seg.toast_pointers.len() || seg.toast_pointers[bi].is_empty() {
                continue;
            }
            let (bytes_from_cache, hit) = detoast_blob_slot(seg, bi);
            if hit {
                stats.cache_hits += 1;
                stats.cache_bytes_served += bytes_from_cache;
            } else {
                stats.cache_misses += 1;
            }
        }
    }
    stats
}

/// Per-call counters returned by the lazy-detoast helpers. Callers fold
/// these into their `ScanTiming` so the totals show up in EXPLAIN.
#[derive(Copy, Clone, Default, Debug)]
pub(crate) struct DetoastLazyStats {
    pub(crate) cache_hits: u64,
    pub(crate) cache_misses: u64,
    pub(crate) cache_bytes_served: u64,
}

/// Extract segment pruning filters from the plan qual (raw expression tree).
///
/// Walks OpExpr nodes looking for:
/// - Equality filters on segment_by columns (e.g. `CounterID = 62`)
/// - Range filters on the time column (e.g. `ts >= '2023-01-01'`)
///
/// Returns (segment_by_filters, time_min, time_max).
pub(super) unsafe fn extract_segment_filters(
    qual_list: *mut pg_sys::List,
    col_names: &[String],
    segment_by: &[String],
    time_column: &str,
) -> (Vec<(usize, String)>, Option<i64>, Option<i64>) {
    let mut segment_by_filters: Vec<(usize, String)> = Vec::new();
    let mut time_min: Option<i64> = None;
    let mut time_max: Option<i64> = None;

    if qual_list.is_null() {
        return (segment_by_filters, time_min, time_max);
    }

    unsafe {
        // Build segment_by column name -> segment_values index mapping
        let mut seg_val_index_map: HashMap<&str, usize> = HashMap::new();
        let mut seg_val_idx = 0;
        for name in col_names {
            if segment_by.contains(name) {
                seg_val_index_map.insert(name.as_str(), seg_val_idx);
                seg_val_idx += 1;
            }
        }

        let nquals = (*qual_list).length;
        for i in 0..nquals {
            let cell = (*qual_list).elements.add(i as usize);
            let node = (*cell).ptr_value as *const pg_sys::Node;
            if node.is_null() {
                continue;
            }

            let tag = (*node).type_;
            if tag != pg_sys::NodeTag::T_OpExpr {
                continue;
            }

            let opexpr = node as *const pg_sys::OpExpr;
            let args = (*opexpr).args;
            if args.is_null() || (*args).length != 2 {
                continue;
            }

            // Get operator name
            let opname_ptr = pg_sys::get_opname((*opexpr).opno);
            if opname_ptr.is_null() {
                continue;
            }
            let opname = std::ffi::CStr::from_ptr(opname_ptr).to_str().unwrap_or("");

            // Get the two args
            let arg0 = (*(*args).elements.add(0)).ptr_value as *const pg_sys::Node;
            let arg1 = (*(*args).elements.add(1)).ptr_value as *const pg_sys::Node;
            if arg0.is_null() || arg1.is_null() {
                continue;
            }

            // Identify Var and Const (handle both orderings)
            let (var_node, const_node, var_on_left) = if (*arg0).type_ == pg_sys::NodeTag::T_Var
                && (*arg1).type_ == pg_sys::NodeTag::T_Const
            {
                (
                    arg0 as *const pg_sys::Var,
                    arg1 as *const pg_sys::Const,
                    true,
                )
            } else if (*arg0).type_ == pg_sys::NodeTag::T_Const
                && (*arg1).type_ == pg_sys::NodeTag::T_Var
            {
                (
                    arg1 as *const pg_sys::Var,
                    arg0 as *const pg_sys::Const,
                    false,
                )
            } else {
                continue;
            };

            if (*const_node).constisnull {
                continue;
            }

            // Convert 1-based varattno to 0-based column index
            let varattno = (*var_node).varattno as i32;
            if varattno < 1 || varattno as usize > col_names.len() {
                continue;
            }
            let col_idx = (varattno - 1) as usize;
            let col_name = &col_names[col_idx];

            // Check if this is a segment_by equality filter
            if opname == "="
                && let Some(&sv_idx) = seg_val_index_map.get(col_name.as_str())
            {
                // Extract const value as string (matches how segment_values are stored)
                let mut typoutput: pg_sys::Oid = pg_sys::InvalidOid;
                let mut typisvarlena: bool = false;
                pg_sys::getTypeOutputInfo(
                    (*const_node).consttype,
                    &mut typoutput,
                    &mut typisvarlena,
                );
                let cstr = pg_sys::OidOutputFunctionCall(typoutput, (*const_node).constvalue);
                let s = std::ffi::CStr::from_ptr(cstr)
                    .to_string_lossy()
                    .into_owned();
                pg_sys::pfree(cstr as *mut _);
                segment_by_filters.push((sv_idx, s));
            }

            // Check if this is a time column range filter
            if col_name == time_column {
                let ts_val = (*const_node).constvalue.value() as i64;

                // Normalize operator direction (if Var is on right, flip the operator)
                let effective_op = if var_on_left {
                    opname
                } else {
                    match opname {
                        ">=" => "<=",
                        ">" => "<",
                        "<=" => ">=",
                        "<" => ">",
                        _ => opname,
                    }
                };

                // Normalize to canonical half-open [time_min, time_max):
                // time_min is the smallest value that matches (inclusive),
                // time_max the smallest value that does NOT match
                // (exclusive). Exact for the integer-datum time types
                // (timestamp/timestamptz µs, date days), which lets the
                // metadata-only agg paths do per-row and full-containment
                // checks with these bounds, not just conservative pruning.
                match effective_op {
                    ">=" | ">" => {
                        let lo = if effective_op == ">" {
                            ts_val.saturating_add(1)
                        } else {
                            ts_val
                        };
                        // Lower bound: take the maximum of all lower bounds
                        time_min = Some(match time_min {
                            Some(existing) => existing.max(lo),
                            None => lo,
                        });
                    }
                    "<=" | "<" => {
                        let hi = if effective_op == "<=" {
                            ts_val.saturating_add(1)
                        } else {
                            ts_val
                        };
                        // Upper bound: take the minimum of all upper bounds
                        time_max = Some(match time_max {
                            Some(existing) => existing.min(hi),
                            None => hi,
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    (segment_by_filters, time_min, time_max)
}

// ============================================================================
// P2 decompose-on-write: candidate segment location for UPDATE/DELETE
// ============================================================================

/// One candidate segment for a DML statement on a compressed partition
/// (COMPRESSED_DML.md §5.2 step 3).
pub(in crate::scan) struct DmlSegmentCandidate {
    pub(in crate::scan) segment_id: i32,
    /// Metadata PROVES every row in this segment satisfies the complete DML
    /// predicate. Only ever true when the caller passed `quals_complete`
    /// (the collected qual lists are the full predicate), every qual was
    /// batch-extractable, and `classify_segment_quals` returned `AllPass`
    /// (which also requires zero NULLs in the qual columns). Enables the
    /// whole-segment DELETE fast path (§5.4); ambiguity always degrades to
    /// decompose, never to skip.
    pub(in crate::scan) all_rows_match: bool,
    /// Loaded segment metadata (no blobs — `skip_blob_load`), kept for the
    /// P2.5 tombstone fast path's per-row predicate evaluation.
    pub(in crate::scan) seg: SegmentData,
}

/// Result of candidate location for one DML target relation.
pub(in crate::scan) struct DmlCandidateSet {
    pub(in crate::scan) candidates: Vec<DmlSegmentCandidate>,
    /// The collected qual lists are PROVABLY the complete statement
    /// predicate AND every qual node was batch-extracted. This is the same
    /// exactness contract production scans rely on (`ps.qual` is nulled when
    /// all quals are batch-handled), so per-row batch evaluation over a
    /// decoded segment yields the exact matching row set.
    pub(in crate::scan) all_quals_handled: bool,
    /// Batch quals extracted from the qual lists (column indices reference
    /// the partition's attno order = companion `col_names` order).
    pub(in crate::scan) batch_quals: Vec<BatchQual>,
}

/// Locate the segments of `companion_oid` that may contain rows matching a
/// DML statement's predicate, reusing the exact pruning pipeline the read
/// path uses (`load_segments_heap`: partition minmax, colstats minmax
/// index, per-segment blooms, valbitmaps, dictionary pruning). The result
/// is a conservative superset: a segment is excluded
/// only when metadata PROVES it holds no matching row — precisely the
/// invariant the production scan path already depends on for correctness.
///
/// `qual_lists` are plan qual lists whose Vars reference the partition's
/// attnos (scan-level quals under the ModifyTable). Quals that can't be
/// extracted simply prune nothing. `quals_complete` must only be true when
/// the caller is certain the lists together are the FULL statement predicate
/// for this relation — it gates `all_rows_match`, never candidate selection.
///
/// # Safety
/// Must run inside a transaction with an active snapshot (it scans the
/// companion tables). `qual_lists` must point to valid plan qual Lists.
pub(in crate::scan) unsafe fn dml_candidate_segments(
    companion_oid: pg_sys::Oid,
    qual_lists: &[*mut pg_sys::List],
    quals_complete: bool,
) -> DmlCandidateSet {
    let meta = load_metadata_cached(companion_oid);

    let mut batch_quals: Vec<BatchQual> = Vec::new();
    let mut handled_total: usize = 0;
    let mut nquals_total: usize = 0;
    let mut seg_filters: Vec<(usize, String)> = Vec::new();
    let mut t_min: Option<i64> = None;
    let mut t_max: Option<i64> = None;

    for &ql in qual_lists {
        if ql.is_null() {
            continue;
        }
        nquals_total += unsafe { (*ql).length } as usize;
        let (bqs, _handled) =
            unsafe { super::batch_qual::extract_batch_quals(ql, &meta.col_names, &meta.col_types) };
        // Every node `extract_batch_quals` converts pushes exactly one
        // BatchQual, so `len == nquals` ⟺ every qual node was converted.
        // We deliberately do NOT use the returned `handled` count: it skips
        // ScalarArrayOpExpr (IN lists) and bare-bool-Var nodes — a
        // conservative undercount for the scan path (which then keeps
        // `ps.qual` as a correctness net), but the DML offset evaluator
        // (`dml_matching_offsets`) evaluates every pushed shape exactly,
        // including text IN lists and multiple text quals on one column.
        handled_total += bqs.len();
        batch_quals.extend(bqs);

        let (sf, lo, hi) = unsafe {
            extract_segment_filters(ql, &meta.col_names, &meta.segment_by, &meta.time_column)
        };
        seg_filters.extend(sf);
        if let Some(v) = lo {
            t_min = Some(t_min.map_or(v, |e: i64| e.max(v)));
        }
        if let Some(v) = hi {
            t_max = Some(t_max.map_or(v, |e: i64| e.min(v)));
        }
    }

    // Ask the loader to materialize colstats min/max + sums for every qual
    // column so `classify_segment_quals` can prove AllPass for the
    // whole-segment DELETE fast path. Segment-by columns aren't in colstats
    // and just resolve to "no metadata" (Ambiguous → decompose) downstream.
    let mut qual_cols: Vec<String> = Vec::new();
    for bq in &batch_quals {
        if bq.col_idx < meta.col_names.len() {
            let name = &meta.col_names[bq.col_idx];
            if !qual_cols.contains(name) {
                qual_cols.push(name.clone());
            }
        }
    }
    let needed_cols = vec![false; meta.col_names.len()];

    let (segments, _skipped, _mm, _bloom, _vb, _us) = unsafe {
        load_segments_heap(
            companion_oid,
            &meta.col_names,
            &meta.segment_by,
            &needed_cols,
            &meta.time_column,
            true, // load_minmax: feed classify_segment_quals
            &seg_filters,
            t_min,
            t_max,
            None,
            &batch_quals,
            &qual_cols, // needed_stats_cols → sums/nonnull for NULL safety
            &meta.col_types,
            &meta.col_not_null,
            &qual_cols, // needed_minmax_cols
            &meta.blob_idx,
            true, // skip_blob_load: we only need segment identity + stats
        )
    };

    // The whole-segment fast path additionally requires that EVERY qual in
    // the (complete) predicate was batch-extracted; otherwise an unhandled
    // residual qual could exclude rows we'd be dropping wholesale.
    let all_quals_handled = quals_complete && handled_total == nquals_total;

    let candidates = segments
        .into_iter()
        .map(|seg| {
            // Whole-segment proof: AllPass means every PHYSICAL row matches;
            // tombstones only remove rows, so the proof still covers every
            // LIVE row — the drop stays valid (row accounting subtracts
            // tombstones in `decompose_segments_for_dml`).
            let all_rows_match = all_quals_handled
                && (batch_quals.is_empty()
                    || matches!(
                        classify_segment_quals(
                            &seg,
                            &batch_quals,
                            &meta.col_names,
                            &meta.time_column,
                        ),
                        SegmentQualResult::AllPass
                    ));
            DmlSegmentCandidate {
                segment_id: seg.segment_id,
                all_rows_match,
                seg,
            }
        })
        .collect();

    DmlCandidateSet {
        candidates,
        all_quals_handled,
        batch_quals,
    }
}

/// Exactness pre-check for the P2.5 tombstone DELETE fast path: every batch
/// qual must evaluate with exact SQL semantics in `dml_matching_offsets`.
/// The only known divergence in the shared evaluator is BOOL with ordering
/// operators (they degrade to equality) — reject those and fall back to
/// decompose-on-write.
pub(in crate::scan) fn batch_quals_tombstone_exact(batch_quals: &[BatchQual]) -> bool {
    batch_quals.iter().all(|bq| match bq.op {
        BatchCompareOp::Lt | BatchCompareOp::Le | BatchCompareOp::Gt | BatchCompareOp::Ge => {
            bq.type_oid != pg_sys::BOOLOID
        }
        // BPCHAR equality semantics ignore trailing spaces (bpchareq);
        // the byte-wise text evaluator doesn't — refuse char(n) IN lists.
        BatchCompareOp::InList => !(bq.in_list_text.is_some() && bq.type_oid == pg_sys::BPCHAROID),
        _ => true,
    })
}

/// SQL LIKE evaluation for a pre-compiled strategy — same semantics as the
/// production text decode filters (`decompress_text_blob_with_like_filter`).
fn like_strategy_matches(strategy: &LikeStrategy, s: &str) -> bool {
    match strategy {
        LikeStrategy::Contains(c) => s.contains(c.as_str()),
        LikeStrategy::StartsWith(c) => s.starts_with(c.as_str()),
        LikeStrategy::EndsWith(c) => s.ends_with(c.as_str()),
        LikeStrategy::Exact(c) => s == c,
        LikeStrategy::General(p) => sql_like_match(s, p),
    }
}

/// Evaluate one batch qual against a single (segment-by) value. `None` means
/// "cannot evaluate exactly here — caller must fall back to decompose".
/// NULL values fail every qual (SQL three-valued logic).
fn eval_qual_on_single_value(bq: &BatchQual, value: &Option<String>) -> Option<bool> {
    let Some(s) = value else {
        return Some(false); // NULL fails every predicate
    };
    // Text-family quals compare the string directly.
    if matches!(bq.op, BatchCompareOp::Like | BatchCompareOp::NotLike) {
        let strat = bq.like_strategy.as_ref()?;
        let m = like_strategy_matches(strat, s);
        return Some(if bq.op == BatchCompareOp::NotLike {
            !m
        } else {
            m
        });
    }
    if let Some(list) = &bq.in_list_text {
        return Some(list.iter().any(|v| v == s));
    }
    if let Some(c) = &bq.text_const {
        return Some(match bq.op {
            BatchCompareOp::Eq => s == c,
            BatchCompareOp::Ne => s != c,
            _ => return None,
        });
    }
    // Numeric/bool/in-list: parse through the type's input function, then
    // compare with the same width rules as `evaluate_batch_quals`.
    let datum = super::datum_utils::string_to_datum(s, bq.type_oid);
    let col = [(datum, false)];
    let mut sel = [true];
    if bq.op == BatchCompareOp::InList {
        let values = bq.in_list_i64.as_ref()?;
        super::batch_qual::apply_batch_filter_in_list(&col, &mut sel, values, bq.type_oid);
        return Some(sel[0]);
    }
    match bq.type_oid {
        pg_sys::INT8OID | pg_sys::TIMESTAMPOID | pg_sys::TIMESTAMPTZOID => {
            super::batch_qual::apply_batch_filter_i64(
                &col,
                &mut sel,
                bq.op,
                bq.const_datum.value() as i64,
            );
        }
        pg_sys::INT4OID | pg_sys::DATEOID => {
            super::batch_qual::apply_batch_filter_i32(
                &col,
                &mut sel,
                bq.op,
                bq.const_datum.value() as i32,
            );
        }
        pg_sys::INT2OID => {
            super::batch_qual::apply_batch_filter_i16(
                &col,
                &mut sel,
                bq.op,
                bq.const_datum.value() as i16,
            );
        }
        pg_sys::FLOAT8OID => {
            let c = f64::from_bits(bq.const_datum.value() as u64);
            super::batch_qual::apply_batch_filter_f64(&col, &mut sel, bq.op, c);
        }
        pg_sys::FLOAT4OID => {
            let c = f32::from_bits(bq.const_datum.value() as u32);
            super::batch_qual::apply_batch_filter_f32(&col, &mut sel, bq.op, c);
        }
        pg_sys::BOOLOID => {
            if !matches!(bq.op, BatchCompareOp::Eq | BatchCompareOp::Ne) {
                return None;
            }
            super::batch_qual::apply_batch_filter_bool(
                &col,
                &mut sel,
                bq.op,
                bq.const_datum.value() != 0,
            );
        }
        _ => return None,
    }
    Some(sel[0])
}

/// Evaluate one text-family batch qual over a decoded text column, ANDing
/// the result into `sel`. `None` = cannot evaluate exactly (caller falls
/// back to decompose). NULL rows fail; rows already deselected are skipped.
unsafe fn apply_text_qual_exact(
    bq: &BatchQual,
    col: &[(pg_sys::Datum, bool)],
    sel: &mut [bool],
) -> Option<()> {
    enum TextCmp<'a> {
        EqNe(&'a str, bool),
        InList(&'a [String]),
        Like(&'a LikeStrategy, bool),
    }
    let cmp = if matches!(bq.op, BatchCompareOp::Like | BatchCompareOp::NotLike) {
        TextCmp::Like(bq.like_strategy.as_ref()?, bq.op == BatchCompareOp::NotLike)
    } else if let Some(list) = &bq.in_list_text {
        TextCmp::InList(list.as_slice())
    } else if let Some(c) = &bq.text_const {
        match bq.op {
            BatchCompareOp::Eq => TextCmp::EqNe(c.as_str(), false),
            BatchCompareOp::Ne => TextCmp::EqNe(c.as_str(), true),
            _ => return None,
        }
    } else {
        return None;
    };
    for (i, &(datum, is_null)) in col.iter().enumerate() {
        if !sel[i] {
            continue;
        }
        if is_null {
            sel[i] = false;
            continue;
        }
        let varlena_ptr = datum.cast_mut_ptr::<pg_sys::varlena>();
        let len = unsafe { pgrx::varsize_any_exhdr(varlena_ptr) };
        let data = unsafe { pgrx::vardata_any(varlena_ptr) };
        #[allow(clippy::unnecessary_cast)]
        let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
        let Ok(s) = std::str::from_utf8(bytes) else {
            return None;
        };
        sel[i] = match &cmp {
            TextCmp::EqNe(c, is_ne) => (s == *c) != *is_ne,
            TextCmp::InList(list) => list.iter().any(|v| v == s),
            TextCmp::Like(strat, negate) => like_strategy_matches(strat, s) != *negate,
        };
    }
    Some(())
}

/// P2.5 tombstone DELETE fast path: compute the EXACT set of row offsets in
/// `seg` matching the statement's (complete, fully batch-extracted)
/// predicate, by decoding only the qual columns and running the same batch
/// evaluation the read path uses as its sole filter. Already-tombstoned
/// offsets may be included — the tombstone INSERT's `ON CONFLICT DO NOTHING`
/// deduplicates them and they don't count toward the command tag.
///
/// Returns `None` when exact evaluation isn't possible (caller falls back to
/// P2 decompose-on-write — never to skipping). Exactness contract: the
/// caller verified `all_quals_handled` (predicate completeness) and
/// `batch_quals_tombstone_exact`.
///
/// # Safety
/// Must run inside a transaction with an active snapshot. `seg` must belong
/// to `companion_oid` and still be live (caller holds a `FOR UPDATE` lock on
/// its `_meta` row, so its blobs are immutable while we decode).
pub(in crate::scan) unsafe fn dml_matching_offsets(
    companion_oid: pg_sys::Oid,
    seg: &mut SegmentData,
    batch_quals: &[BatchQual],
) -> Option<Vec<u32>> {
    if !batch_quals_tombstone_exact(batch_quals) {
        return None;
    }
    let meta = load_metadata_cached(companion_oid);
    let meta = &meta;
    let row_count = seg.row_count as usize;
    if row_count == 0 {
        return Some(Vec::new());
    }
    let ncols = meta.col_names.len();
    let mut needed = vec![false; ncols];
    for bq in batch_quals {
        if bq.col_idx >= ncols {
            return None;
        }
        needed[bq.col_idx] = true;
    }

    // Segment-by columns: a single value covers the whole segment.
    let mut segby_value: Vec<Option<Option<String>>> = vec![None; ncols];
    {
        let mut sv_idx = 0usize;
        for (i, name) in meta.col_names.iter().enumerate() {
            if meta.segment_by.contains(name) {
                if needed[i] {
                    segby_value[i] = Some(seg.segment_values.get(sv_idx).cloned().flatten());
                }
                sv_idx += 1;
            }
        }
    }

    // Fetch and decode blobs for the remaining qual columns.
    let mut blob_needed = vec![false; ncols];
    for i in 0..ncols {
        if needed[i] && segby_value[i].is_none() && meta.blob_idx[i].is_some() {
            blob_needed[i] = true;
        }
    }
    if blob_needed.iter().any(|&b| b) {
        unsafe {
            fetch_segment_blobs(
                companion_oid,
                seg.segment_id,
                &meta.col_names,
                &blob_needed,
                &meta.blob_idx,
                seg,
            );
        }
    }

    let mut decoded: Vec<Vec<(pg_sys::Datum, bool)>> = (0..ncols).map(|_| Vec::new()).collect();
    for i in 0..ncols {
        if !needed[i] || segby_value[i].is_some() {
            continue;
        }
        let type_oid = meta.col_types[i];
        match meta.blob_idx[i] {
            Some(ci) => {
                let blob = &seg.compressed_blobs[ci as usize];
                if blob.is_empty() {
                    // Missing blob row (shouldn't happen for a live segment)
                    // — refuse rather than guess.
                    return None;
                }
                let type_name = super::datum_utils::pg_type_name(type_oid);
                let datums = unsafe {
                    super::datum_utils::decompress_blob_to_datums(
                        blob,
                        &type_name,
                        type_oid,
                        meta.col_typmods[i],
                    )
                };
                if datums.len() != row_count {
                    return None;
                }
                decoded[i] = datums;
            }
            None => {
                // Column added to the parent after compression — synthesize
                // the fast-default value, same as the read path.
                let (d, isnull) = meta
                    .missing_values
                    .get(i)
                    .copied()
                    .flatten()
                    .unwrap_or((pg_sys::Datum::from(0), true));
                decoded[i] = (0..row_count).map(|_| (d, isnull)).collect();
            }
        }
    }

    let mut sel = vec![true; row_count];

    // 1) Segment-by quals (single-value, covers all rows).
    for bq in batch_quals {
        if let Some(v) = &segby_value[bq.col_idx]
            && !eval_qual_on_single_value(bq, v)?
        {
            return Some(Vec::new());
        }
    }

    // 2) Text-family quals on decoded columns (the shared evaluator skips
    //    text types — production handles them in decode filters; here we
    //    apply each one explicitly so multiple text quals on one column all
    //    take effect).
    for bq in batch_quals {
        if segby_value[bq.col_idx].is_some() {
            continue;
        }
        let is_textish = matches!(bq.op, BatchCompareOp::Like | BatchCompareOp::NotLike)
            || bq.text_const.is_some()
            || bq.in_list_text.is_some();
        if is_textish {
            unsafe { apply_text_qual_exact(bq, &decoded[bq.col_idx], &mut sel)? };
        } else if !super::batch_qual::is_batch_comparable_type(bq.type_oid) {
            // A handled qual we can't evaluate here — refuse.
            return None;
        }
    }

    // 3) Numeric / bool / numeric-IN quals via the production evaluator
    //    (NULL fails; same comparison semantics the scan path relies on).
    //    Segment-by columns have empty decoded slots — the evaluator skips
    //    them (handled in step 1); text quals are skipped by type (step 2).
    sel = super::batch_qual::evaluate_batch_quals(&decoded, row_count, batch_quals, sel);

    Some(
        sel.iter()
            .enumerate()
            .filter_map(|(i, &s)| if s { Some(i as u32) } else { None })
            .collect(),
    )
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use super::*;

    fn mk_tombstoned_segment(row_count: i32, tombs: &[u32]) -> SegmentData {
        SegmentData {
            companion_oid: pg_sys::InvalidOid,
            segment_id: 1,
            segment_values: Vec::new(),
            compressed_blobs: Vec::new(),
            text_length_blobs: Vec::new(),
            row_count,
            min_time: None,
            max_time: None,
            col_minmax: HashMap::new(),
            col_sums: HashMap::new(),
            toast_pointers: Vec::new(),
            cached_blob_pins: Vec::new(),
            tombstones: if tombs.is_empty() {
                None
            } else {
                Some(std::sync::Arc::new(tombs.to_vec()))
            },
        }
    }

    #[test]
    fn live_row_count_subtracts_tombstones() {
        assert_eq!(mk_tombstoned_segment(100, &[]).live_row_count(), 100);
        assert_eq!(mk_tombstoned_segment(100, &[0, 5, 99]).live_row_count(), 97);
        // Fully tombstoned segment contributes zero live rows.
        assert_eq!(mk_tombstoned_segment(2, &[0, 1]).live_row_count(), 0);
    }

    #[test]
    fn tombstone_preselection_marks_offsets_false() {
        let seg = mk_tombstoned_segment(5, &[1, 3]);
        assert_eq!(
            seg.tombstone_preselection(5),
            Some(vec![true, false, true, false, true]),
        );
        // No tombstones → None (callers keep the no-pre-selection fast path).
        assert_eq!(
            mk_tombstoned_segment(5, &[]).tombstone_preselection(5),
            None
        );
        // Truncated (Top-N cutoff): offsets beyond `n` are ignored.
        assert_eq!(
            seg.tombstone_preselection(2),
            Some(vec![true, false]),
            "offset 3 is beyond the cutoff and must not panic",
        );
    }

    #[test]
    fn batch_quals_tombstone_exact_rejects_bool_ordering() {
        let mk = |op: BatchCompareOp, oid: pg_sys::Oid| BatchQual {
            op,
            type_oid: oid,
            ..Default::default()
        };
        // Bool ordering comparisons degrade to equality in the shared
        // evaluator — must NOT be trusted for exact DELETE offsets.
        assert!(!batch_quals_tombstone_exact(&[mk(
            BatchCompareOp::Lt,
            pg_sys::BOOLOID
        )]));
        assert!(batch_quals_tombstone_exact(&[mk(
            BatchCompareOp::Eq,
            pg_sys::BOOLOID
        )]));
        assert!(batch_quals_tombstone_exact(&[
            mk(BatchCompareOp::Lt, pg_sys::INT4OID),
            mk(BatchCompareOp::Like, pg_sys::TEXTOID),
        ]));
    }

    #[test]
    fn like_strategy_matches_all_shapes() {
        assert!(like_strategy_matches(
            &LikeStrategy::Contains("oo".into()),
            "foo"
        ));
        assert!(like_strategy_matches(
            &LikeStrategy::StartsWith("fo".into()),
            "foo"
        ));
        assert!(like_strategy_matches(
            &LikeStrategy::EndsWith("oo".into()),
            "foo"
        ));
        assert!(like_strategy_matches(
            &LikeStrategy::Exact("foo".into()),
            "foo"
        ));
        assert!(!like_strategy_matches(
            &LikeStrategy::Exact("foo".into()),
            "foox"
        ));
        assert!(like_strategy_matches(
            &LikeStrategy::General("f_o%".into()),
            "foobar"
        ));
    }

    #[test]
    fn eval_qual_on_single_value_null_fails_everything() {
        let bq = BatchQual {
            op: BatchCompareOp::Eq,
            type_oid: pg_sys::TEXTOID,
            text_const: Some("x".into()),
            ..Default::default()
        };
        assert_eq!(eval_qual_on_single_value(&bq, &None), Some(false));
        // Ne also fails on NULL (SQL three-valued logic).
        let bq_ne = BatchQual {
            op: BatchCompareOp::Ne,
            ..bq.clone()
        };
        assert_eq!(eval_qual_on_single_value(&bq_ne, &None), Some(false));
    }

    #[test]
    fn eval_qual_on_single_value_text_ops() {
        let mk = |op: BatchCompareOp| BatchQual {
            op,
            type_oid: pg_sys::TEXTOID,
            text_const: Some("abc".into()),
            ..Default::default()
        };
        let v = Some("abc".to_string());
        assert_eq!(
            eval_qual_on_single_value(&mk(BatchCompareOp::Eq), &v),
            Some(true)
        );
        assert_eq!(
            eval_qual_on_single_value(&mk(BatchCompareOp::Ne), &v),
            Some(false)
        );
        let other = Some("xyz".to_string());
        assert_eq!(
            eval_qual_on_single_value(&mk(BatchCompareOp::Eq), &other),
            Some(false)
        );
        let in_q = BatchQual {
            op: BatchCompareOp::InList,
            type_oid: pg_sys::TEXTOID,
            in_list_text: Some(vec!["abc".into(), "def".into()]),
            ..Default::default()
        };
        assert_eq!(eval_qual_on_single_value(&in_q, &v), Some(true));
        assert_eq!(eval_qual_on_single_value(&in_q, &other), Some(false));
    }

    fn mk_filter(op: BatchCompareOp, c: i64) -> MinMaxFilter {
        MinMaxFilter {
            col_idx: 0,
            op,
            const_i64: c,
            in_list_i64: None,
        }
    }

    #[test]
    fn segment_passes_minmax_eq_in_range() {
        // [10, 20] contains 15 → pass; doesn't contain 5 → skip.
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Eq, 15),
            10,
            20
        ));
        assert!(!segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Eq, 5),
            10,
            20
        ));
        assert!(!segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Eq, 25),
            10,
            20
        ));
        // Edges are inclusive.
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Eq, 10),
            10,
            20
        ));
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Eq, 20),
            10,
            20
        ));
    }

    #[test]
    fn segment_passes_minmax_ne_only_skips_constant_segments() {
        // Ne can only skip when every row equals the constant (min == max == c).
        assert!(!segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Ne, 5),
            5,
            5
        ));
        // Any segment with min != max or min != c keeps the chance of a non-c row.
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Ne, 5),
            5,
            6
        ));
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Ne, 5),
            0,
            10
        ));
    }

    #[test]
    fn segment_passes_minmax_comparisons() {
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Lt, 20),
            10,
            30
        ));
        assert!(!segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Lt, 10),
            10,
            30
        ));
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Le, 10),
            10,
            30
        ));
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Gt, 20),
            10,
            30
        ));
        assert!(!segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Gt, 30),
            10,
            30
        ));
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Ge, 30),
            10,
            30
        ));
    }

    #[test]
    fn segment_passes_minmax_inlist() {
        let f = MinMaxFilter {
            col_idx: 0,
            op: BatchCompareOp::InList,
            const_i64: 0,
            in_list_i64: Some(vec![5, 15, 25]),
        };
        // 15 is in range — pass
        assert!(segment_passes_minmax_filter(&f, 10, 20));
        // No value in range — skip
        assert!(!segment_passes_minmax_filter(&f, 30, 40));
    }

    #[test]
    fn segment_passes_minmax_like_never_skips() {
        // Like and NotLike fall through to `true` — they can't prune via numeric minmax.
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::Like, 0),
            10,
            20
        ));
        assert!(segment_passes_minmax_filter(
            &mk_filter(BatchCompareOp::NotLike, 0),
            10,
            20
        ));
    }

    fn mk_cm(min: i64, max: i64, type_oid: pg_sys::Oid) -> ColMinMax {
        ColMinMax {
            min_encoded: min,
            max_encoded: max,
            min_null: false,
            max_null: false,
            type_oid,
        }
    }

    #[test]
    fn bloom_probe_encode_matches_build_domain() {
        // Integers: identity.
        assert_eq!(
            bloom_probe_encode(pg_sys::Datum::from(42i64 as usize), pg_sys::INT8OID),
            42
        );
        // Timestamps: PG-epoch usec -> Unix-epoch usec (the domain
        // TypedColumn::Int64 stores and the bloom build hashes).
        assert_eq!(
            bloom_probe_encode(pg_sys::Datum::from(0i64 as usize), pg_sys::TIMESTAMPTZOID),
            crate::compress::PG_EPOCH_OFFSET_USEC
        );
        // Dates: PG-epoch days -> Unix-epoch usec.
        assert_eq!(
            bloom_probe_encode(pg_sys::Datum::from(1i32 as usize), pg_sys::DATEOID),
            (crate::compress::PG_EPOCH_OFFSET_DAYS + 1) * 86_400_000_000
        );
        // f32: low 32 datum bits = bit pattern.
        let f = 1.5f32;
        assert_eq!(
            bloom_probe_encode(pg_sys::Datum::from(f.to_bits() as usize), pg_sys::FLOAT4OID),
            f.to_bits() as i64
        );
    }

    #[test]
    fn segment_all_rows_pass_eq_ranges() {
        let cm = mk_cm(10, 10, pg_sys::INT4OID);
        // min == max == c → all rows match
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Eq, pg_sys::Datum::from(10i32 as usize)),
            Some(true),
        );
        // outside [min, max] → no rows match
        let cm = mk_cm(10, 20, pg_sys::INT4OID);
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Eq, pg_sys::Datum::from(5i32 as usize)),
            Some(false),
        );
        // overlapping → ambiguous (must decompress)
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Eq, pg_sys::Datum::from(15i32 as usize)),
            None,
        );
    }

    #[test]
    fn segment_all_rows_pass_returns_none_on_null_bounds() {
        // Either bound null → can't decide
        let cm = ColMinMax {
            min_encoded: 0,
            max_encoded: 100,
            min_null: true,
            max_null: false,
            type_oid: pg_sys::INT4OID,
        };
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Eq, pg_sys::Datum::from(10i32 as usize)),
            None,
        );
    }

    #[test]
    fn segment_all_rows_pass_comparisons() {
        let cm = mk_cm(10, 20, pg_sys::INT4OID);
        // Lt 30: max < 30 → all rows pass
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Lt, pg_sys::Datum::from(30i32 as usize)),
            Some(true),
        );
        // Lt 10: min >= 10 → no rows pass
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Lt, pg_sys::Datum::from(10i32 as usize)),
            Some(false),
        );
        // Lt 15: partial → ambiguous
        assert_eq!(
            segment_all_rows_pass(&cm, BatchCompareOp::Lt, pg_sys::Datum::from(15i32 as usize)),
            None,
        );
    }

    #[test]
    fn is_zero_const_matches_each_numeric_type() {
        assert!(is_zero_const(
            pg_sys::Datum::from(0i16 as usize),
            pg_sys::INT2OID
        ));
        assert!(!is_zero_const(
            pg_sys::Datum::from(1i16 as usize),
            pg_sys::INT2OID
        ));
        assert!(is_zero_const(
            pg_sys::Datum::from(0i32 as usize),
            pg_sys::INT4OID
        ));
        assert!(is_zero_const(
            pg_sys::Datum::from(0i64 as usize),
            pg_sys::INT8OID
        ));
        assert!(is_zero_const(
            pg_sys::Datum::from(0.0f32.to_bits() as usize),
            pg_sys::FLOAT4OID
        ));
        assert!(is_zero_const(
            pg_sys::Datum::from(0.0f64.to_bits() as usize),
            pg_sys::FLOAT8OID
        ));
        assert!(!is_zero_const(
            pg_sys::Datum::from(1.0f64.to_bits() as usize),
            pg_sys::FLOAT8OID
        ));
        // Unsupported types always false (caller treats as "can't prove")
        assert!(!is_zero_const(
            pg_sys::Datum::from(0u64 as usize),
            pg_sys::TEXTOID
        ));
    }

    #[test]
    fn encode_datum_to_i64_identity_for_integers() {
        // INT2/4/8 round-trip the raw datum value.
        assert_eq!(
            encode_datum_to_i64(pg_sys::Datum::from(42i16 as usize), pg_sys::INT2OID),
            Some(42),
        );
        assert_eq!(
            encode_datum_to_i64(pg_sys::Datum::from(42i32 as usize), pg_sys::INT4OID),
            Some(42),
        );
        assert_eq!(
            encode_datum_to_i64(pg_sys::Datum::from(42i64 as usize), pg_sys::INT8OID),
            Some(42),
        );
    }

    #[test]
    fn encode_datum_to_i64_unsupported_returns_none() {
        // TEXT etc. — caller relies on this to fall back to per-row eval.
        assert_eq!(
            encode_datum_to_i64(pg_sys::Datum::from(0u64 as usize), pg_sys::TEXTOID),
            None,
        );
    }
}
