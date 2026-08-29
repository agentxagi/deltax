# Transparent DML on Compressed Partitions

Compressed partitions accept plain `INSERT` / `UPDATE` / `DELETE` / `MERGE`.
The user issues ordinary SQL and the extension does whatever is needed
internally, with full transactional semantics — MVCC, rollback, and crash
recovery all ride on ordinary WAL-logged heap operations, with no
extension-side state machine.

The guiding constraint is that **DML must not slow analytics on the happy
path**. A compressed partition with no pending writes reads exactly as fast
as before this feature existed; the cost of DML is paid by the writer and, at
most, by reads of the specific partitions that have uncompacted changes.

This document describes the implemented design. See `ARCHITECTURE.md` and
`COLUMNAR_STORAGE.md` for the underlying scan/segment machinery, and
`tests/bench_dml.py` (`make bench-dml`) for the write-latency benchmark that
tracks all of this over time.

## 1. The invariant everything rests on

The whole scan/aggregate architecture depends on one property, which DML
preserves:

> **A live segment's metadata is exact.** `_meta._row_count`, the colstats
> min/max/sum/counts, blooms, and valbitmaps always describe exactly the rows
> in that segment. A row is either fully in a segment or fully in the
> partition heap — never half-and-half.

Because a segment is fully described by ordinary, MVCC-visible heap tuples
(`_meta` + sidecar rows in `_colstats`/`_blobs`/`_blooms`/`_text_lengths`/
`_valbitmap`), deleting a segment's `_meta` row transactionally hides the
segment — MVCC for free. Rows logically removed by DELETE are tracked
separately (tombstones, §4.1) rather than by mutating a segment's metadata, so
the invariant holds.

## 2. Two storage tiers

DML uses the storage's existing two-tier shape:

1. **Loose-row region.** The partition heap — empty immediately after
   compression — holds newly inserted (and decomposed) rows alongside the
   segments. Every scan node unions "segments ∪ heap tail". The background
   worker periodically compacts loose rows into new segments.
2. **Segments.** The columnar, immutable-until-recompacted representation in
   the companion tables.

New/decomposed rows land in tier 1 at native heap speed. Compaction moves
them down into tier 2, restoring the pristine metadata-exact fast-path state.

## 3. INSERT

### 3.1 Write path

There is no special insert path: tuple routing puts the row in the partition
heap (the loose region). Direct inserts, parent-routed inserts, multi-row
`INSERT`, and `COPY` all work unchanged and at plain-PostgreSQL heap latency —
no decompression, no segment rewrite. Indexes defined on the partition are
maintained for loose rows normally.

`INSERT ... ON CONFLICT` (upsert) is rejected — conflict inference cannot see
rows locked inside segments — including when smuggled through a
data-modifying CTE under a top-level statement. This is the only categorically
rejected DML shape.

### 3.2 Read path: the heap tail

`DeltaXDecompress` and `DeltaXAppend` gain a **Phase 3 heap-tail scan**: after
the last segment, the partition heap is scanned under the active snapshot and
its tuples flow through the ordinary `ExecQual`/`ExecProject` path. Doing the
union *inside* the custom node (rather than planning an `Append` of custom
scan + seq scan) lets every read path reuse one mechanism and keeps EXPLAIN a
single node.

The metadata-only aggregate nodes fold the tail in at exec time:

| Node | With a heap tail |
|---|---|
| `DeltaXCount` | `sum(_meta._row_count)` + a snapshot-visible count of the heap tail |
| `DeltaXMinMax` | folds heap-tail values into the metadata-derived min/max |
| `DeltaXAgg` | bails to a plain `Agg` over the heap-tail-aware scan (it cannot ingest row-form tuples into its columnar accumulators) |
| Top-N / pathkeys | disabled at plan time (the pruned candidate set wouldn't include loose rows); the plain scan + upper Sort/Limit stays correct |

In parallel plans the heap tail is emitted by the leader only.

### 3.3 The gate: catalog flags, not physical probes

Every read path has to answer two questions cheaply, on every query: does this
partition have loose rows, and does it have tombstones? Answering them by
physically opening the partition heap and the `_tombstones` table forced a
cold relcache build per partition on a fresh backend — tens of milliseconds on
a wide (100+ partition) point query, for relations a segment-only scan never
otherwise touches.

Instead, two boolean flags on `deltax.deltax_partition` —
`has_loose_rows` and `has_tombstones` — carry the answer:

- **Maintained transactionally by the writers.** The INSERT trigger
  (`deltax_note_compressed_insert`) sets `has_loose_rows` in the same
  transaction as the first loose insert; decompose sets it when it restores
  rows; the tombstone writer sets `has_tombstones`; compaction and
  `mark_partition_compressed` clear them. MVCC makes the flags *exact*: any
  snapshot that can see the loose rows / tombstones can see the flag committed
  in the same transaction.
- **Read via one bulk SPI per backend.** `hook::ensure_dml_flags_loaded`
  loads a map keyed by *both* the partition heap OID and its `_meta` companion
  OID, so every plan-time and exec-time gate is a single hash lookup — no
  per-partition catalog round-trip and no partition-heap / `_tombstones`
  opens on the steady-state path. The false→true transition fires a relcache
  invalidation, which drops the cached map (so a later query reloads) and
  replans cached plans.

This keeps the steady-state (no pending writes) read path at zero added cost
versus a table without the DML feature, and keeps a fresh-backend point query
on a wide table at parity with the pre-DML code.

Exec-time stale-plan guards (`DeltaXAgg` / `DeltaXMinMax`) error out and force
a replan if a cached plan is executed after a partition gained loose rows or
tombstones — a correctness backstop for the window between planning and
execution.

### 3.4 Readers do not block writers

Because the gates read catalog flags rather than taking an AccessShare lock on
the partition heap, an analytic reader concurrent with a decompose-UPDATE
(which holds AccessExclusive on the partition) no longer blocks — it reads a
consistent MVCC snapshot (either the full pre-state or the full post-state,
never a torn mix), which is standard PostgreSQL readers-don't-block-writers
behavior.

## 4. UPDATE / DELETE

Two mechanisms, selected automatically per statement by the `ExecutorStart`
interceptor, which walks every `ModifyTable` node (including data-modifying
CTEs and `MERGE` targets) and locates candidate segments with the read path's
pruning pipeline (`dml_candidate_segments` → `load_segments_heap`).

### 4.1 Tombstones (fast DELETE)

A qualifying `DELETE` appends `(segment_id, row_offset)` rows to a per-partition
`_tombstones` companion instead of touching segment bytes — O(1) per affected
row, single-digit milliseconds, MVCC/rollback/replication all native. Scans
subtract tombstones from the selection vector on every decode path; segments
with zero tombstones (the steady state, gated by `has_tombstones`) are
untouched.

Eligibility (per statement): it's a `DELETE`; no `RETURNING`; no row-level
DELETE triggers or `OLD TABLE` transition relations (nothing may observe the
removed rows); the collected scan quals are provably the complete predicate
and every qual was batch-extractable (only the qual columns are decoded once
to identify matching offsets). Anything else falls back to decompose — never
to skipping rows.

`DeltaXCount` stays exact (the catalog `row_count` is decremented in the same
transaction; the meta-scan fallback counts live rows). `DeltaXMinMax` and
`DeltaXAgg` describe physical rows, so they bail to the row path while
tombstones exist (via the plan-time gate + exec-time stale guard). Compaction
folds tombstoned segments back and restores the fast paths.

### 4.2 Whole-segment DELETE (fast drop)

When the pushed-down predicate provably covers a candidate segment in full —
e.g. a time-range retention `DELETE` whose bounds contain the segment's entire
time span — the segment's `_meta` + sidecar rows are dropped outright, with no
row materialization and no tombstones. This is the dominant bulk-retention
pattern and runs in milliseconds regardless of segment size.

Coverage is proven by `classify_segment_quals` returning `AllPass` from
per-segment colstats min/max. Note that the time (partition) column is
non-null by construction in a non-default partition — NULL partition keys
route to the default partition — so its NULL-safety check is satisfied even
though its colstats `_nonnull_count` is a zero placeholder (the time column's
authoritative stats live in `_meta`).

### 4.3 Decompose-on-write (UPDATE, and DELETEs that don't qualify above)

The fallback for everything else: `UPDATE`, `DELETE ... RETURNING`, DELETEs
with row triggers or non-batchable predicates. The interceptor decompresses
just the candidate segments back into the partition heap (`DML_BYPASS`,
`decompose_segments_for_dml`), deletes their `_meta` + sidecar rows, bumps the
statement snapshot's command id, and lets the planned heap scan run the DML
normally over ordinary ctids — firing user triggers, RLS, and `RETURNING` as
usual. The worker recompacts the leftover loose rows later. Cross-partition
UPDATE works for free (PostgreSQL turns it into delete + routed insert; the
insert lands in the target's loose region).

**Chosen over a persistent delete-bitmap** deliberately. A per-segment
delete-bitmap that every read AND-s into its selection vector would tax the
read side forever and break the metadata-exactness invariant (row counts,
min/max, sums, ndistinct, blooms would all over-count). This extension derives
far more from per-segment metadata than a row store does — count/minmax/agg
pushdown, the Phase-0 fast paths, sentinel pruning, synthesized
`pg_statistic` — so a permanent read-side "but check the bitmap" tax is
exactly the wrong trade. Decompose-on-write costs more per write, but writes
to compressed partitions are, by the product's own definition (`compress_after`),
rare and cold, and the cost is localized and improvable. The tombstone fast
layer (§4.1) is a *gated* delete-marker that recovers DELETE latency without
the always-on tax: the zero-tombstone gate keeps the steady-state read cost at
one flag lookup.

### 4.4 Concurrency, MVCC, crash safety

All effects — heap inserts of decomposed rows, `_meta`/sidecar deletes,
tombstone rows, the user DML — are ordinary WAL-logged heap operations in one
transaction:

- **Abort:** every tuple version vanishes; the segment's `_meta` row was never
  visibly deleted; the blob/decomp caches stay valid (segment bytes never
  changed).
- **Concurrent readers:** see either the pre- or post-state, never both or
  neither (same-transaction atomicity).
- **Crash:** standard WAL recovery, no extension-side state machine.

Decompose takes an AccessExclusive partition lock (held to end of
transaction); concurrent DML on the same partition serializes or is cancelled
by the deadlock detector — never silent wrongness. A meta-row delete-first
protocol means a loser sees 0 rows deleted and skips (READ COMMITTED) or gets
a serialization error (REPEATABLE READ+). Tombstone DELETE takes `FOR UPDATE`
on the candidate `_meta` rows (readers never block) and inserts tombstones
`ON CONFLICT DO NOTHING`.

Segment ids are never reused within a companion's lifetime: decompose records
`max_segment_id` in the catalog before deleting `_meta` rows, and compaction
allocates above it, so the `(companion_oid, segment_id, …)`-keyed blob/decomp
caches can't be poisoned.

## 5. Compaction

The background worker's `auto_compact_partitions` step (and the SQL-callable
`deltax_compact_partition`) restores the pristine state for a partition with
loose rows or tombstones, in one transaction:

1. Read loose rows sorted by `segment_by` + `order_by`.
2. Rewrite tombstone-bearing segments (decompose minus dead rows) and fold the
   loose rows into fresh segments via the existing `flush_segment` path — full
   sidecar rows, fresh `_segment_id`s above the high-water mark.
3. `TRUNCATE` the loose region and the `_tombstones` table back to the
   zero-block steady state; clear the `has_loose_rows` / `has_tombstones`
   flags.
4. Refresh catalog counts and re-synthesize `pg_statistic`.

Compaction ends with `TRUNCATE`, which — like compress-time `TRUNCATE` — is
not MVCC-safe against a concurrent REPEATABLE READ reader whose snapshot
predates the commit (PostgreSQL's documented `TRUNCATE` caveat). Under READ
COMMITTED it is safe; under RR+ it deletes rather than truncates.

## 6. Cost profile

Measured warm, against a plain-PostgreSQL twin (`make bench-dml`, 300k-row
synthetic table):

| Operation | vs native | mechanism |
|---|---|---|
| INSERT (single / batch / COPY) | ~1.0–1.5× | loose-region heap insert |
| DELETE point | ~0.1× | tombstone (beats native — no row hunt) |
| DELETE retention (partition-aligned) | ~0.1× | whole-segment drop |
| UPDATE, `DELETE ... RETURNING` | ~15× | decompose-on-write |

The read-after-write cliff: a single uncompacted INSERT into a partition
disables `DeltaXAgg` pushdown for that partition, making a representative
aggregate ~40× slower until the worker compacts. This is the concrete "cost of
DML on the happy path" and the main argument for an aggressive compaction
cadence on write-heavy compressed partitions.

## 7. Limitations

- **`INSERT ... ON CONFLICT`** is rejected (§3.1).
- **Unique constraints** are enforced only among loose rows and
  pre-compression index entries — a new row duplicating a value already inside
  a segment is not detected (segment rows are in no index).
- **`json_extract` tables** (a synthetic column derived from a JSON path):
  INSERT and reads that don't touch the extracted path work; a query that
  selects the extracted path *while loose rows exist* errors clearly (the
  heap-tail row has no synthetic value to serve), and compaction refuses these
  tables. Clearing loose rows requires `deltax_decompress_partition` +
  `deltax_compress_partition`. These tables must be loaded via the
  `COPY … FORMAT deltax_compress_csv` extract path.
- **Physical replicas:** fully correct, nothing special needed. Every DML
  effect is ordinary WAL-logged heap/TOAST activity (loose-row inserts,
  decompose inserts + meta/sidecar deletes, tombstone rows, compaction segment
  inserts + `TRUNCATE`, whole-segment drops) — no non-WAL state — so a standby
  replays a byte-identical copy. The maintenance worker skips standbys
  (`pg_is_in_recovery()`), and DML can't originate on a read-only standby, so a
  standby never diverges.
- **Logical replicas:** internal maintenance WAL (decompose-on-write's
  restored-row inserts, compaction's segment folds + loose-row removal) is
  tagged with the `deltax_internal` replication origin, set per-record via the
  backend-local `replorigin_session_origin` around those writes (see
  `InternalOriginGuard` in `compress.rs`; the origin is created in the
  extension SQL). Two topologies:
  - **Scenario 1** — publish the user table, subscriber holds it and
    (de)compresses independently: create the subscription
    `WITH (origin = none)` (and exclude `TRUNCATE` from the publication). The
    origin filter drops the internal churn while the user's own origin-less
    `INSERT`/`UPDATE`/decompose-`DELETE` replicate normally — so a
    decompose-UPDATE no longer duplicates the whole segment on the subscriber.
  - **Scenario 2** — replicate the raw companion + heap storage: use the
    default `origin = any`; the tagged changes are received normally (the tag
    is just metadata), keeping both sides byte-identical.
  Fast-path DELETEs: the **tombstone and whole-segment-drop** paths touch only
  companion tables, so under Scenario 1 (companions excluded) they produce no
  replicated event and the subscriber would keep the row. A Scenario-1
  publisher sets `pg_deltax.replicable_deletes = on` (typically per-database)
  so DELETEs on compressed partitions decompose-on-write instead — the delete
  then runs as an ordinary heap DELETE and replicates like any row delete.
  It's off by default because the fast paths are much faster; Scenario 2 leaves
  it off and lets the tombstone rows replicate through the companion tables.

## 8. Things we still want to improve

- **Faster UPDATE.** UPDATE always decomposes (~15×). A tombstone-fast UPDATE
  must still materialize the old row versions into the heap for SET
  expressions / triggers / `RETURNING`, which costs the same full-column decode
  the decompose performs — so the clean win is bounded. Restoring only the
  matching rows (rather than whole segments) is a real but partial improvement.
- **Reduce decompose amplification with segment split.** A 1-row UPDATE
  decomposes a whole 30k-row segment. Splitting the segment and decomposing
  only the affected sub-segment would cut amplification without a new
  mechanism.
- **A single gate SPI.** `COMPRESSED_NAMESET` and `DML_FLAGS` are two
  per-backend loads over the same `deltax_partition` rows; folding them into
  one would erase the last ~1 ms of fresh-backend planning cost on the
  smallest queries.
- **`json_extract` heap-tail materialization.** Compute the synthetic column
  for heap-tail rows (from the physical payload) so DML works transparently on
  `json_extract` tables and compaction can fold them — lifting the §7
  limitation.
- **Compaction policy.** Trigger thresholds (loose rows ≥ `segment_size`, ≥ N%
  of partition row_count, or oldest loose row past a `compact_after` interval),
  a churn/hysteresis guard so steady writers aren't compacted into a stream of
  tiny segments, and opportunistic merging of undersized segments. The
  read-after-write cliff (§6) makes cadence the key knob.
- **Decompose cap GUC.** `pg_deltax.max_segments_decomposed_per_dml` to bound
  an unprunable UPDATE/DELETE that would otherwise decompose a whole partition.
- **`ON CONFLICT` / uniqueness against segments.** Bloom-assisted conflict
  checking would let upserts and cross-segment unique constraints work.
- **Partition bloom sentinels.** When the partition-level bloom sentinels land,
  compaction must fold new value hashes into (or invalidate) the affected
  sentinels; `compact_partition_impl` carries an inline note for this.
- **Cheaper replicable DELETE.** `pg_deltax.replicable_deletes` (§7) makes
  fast-path DELETEs replicate under Scenario 1 by forcing decompose-on-write —
  correct, but it pays the full decompose cost for every delete. A lighter
  option would emit a synthetic heap-level delete event for tombstoned rows so
  the fast path stays fast AND replicates; not yet done.
- **Object-storage offload alignment.** The loose region is the natural write
  buffer and tombstones are the natural merge-on-read delete layer for a future
  S3/Parquet tier. The design already keeps segment metadata local and treats
  the `_meta` row as the segment's transactional identity (an ACID manifest);
  keep it that way, and make compaction's output tier a policy decision rather
  than always-local.
