# Gate 2.2 Verdict — Multi-Request Batching on Ascend NPU

**Gate ID:** 2.2 (multi-request batching on Ascend 910B1)
**Verdict:** PASS
**Frozen at commit:** `3d6b3edf1aae971082ec6cebec8e88f481e68193`
**Branch:** `gate2-multirequest-batching`
**Tag:** `gate2-multirequest-batching`
**Date:** 2026-07-05

Gate 2.2 lifts the Ascend FIA backend and the scheduler from the
single-in-flight-request scope frozen at Gate 2.1 to concurrent
multi-request batching under `Scheduler.normal_loop` in `offline_mode=True`.
All invariants below were verified on a single Ascend 910B1 die on real
hardware. Host paths, container IDs, and credentials are recorded in
private ops notes and are deliberately omitted from this document.

---

## 1. Software and hardware

Inherits Gate 2.1's platform (see `gate2_1_multistep_verdict.md` §1):

| Component | Value |
| --- | --- |
| Accelerator | Ascend 910B1 (64 GiB HBM per die) |
| Dies used | 1 (`npu:0`) |
| Model | Qwen3-0.6B, 28 layers, `float16` |
| Attention backend | `npu_fia` (Ascend Fused Infer Attention Score) |
| Cache type | `radix` |
| Page size | 16 tokens |
| Tensor parallel size | 1 |
| CUDA graph | disabled |

---

## 2. Supported scope

Gate 2.2 signs off on the following combinations, in addition to the
Gate 2.1 single-request scope:

- **Equal-length prefill (`B >= 1`).** Every request in the prefill
  batch shares `extend_len`, `device_len`, and `cached_len`. Query is
  reshaped `[B, S, Hq, D]` BSND, `actual_seq_lengths = [S]*B`,
  `actual_seq_lengths_kv = [S+cached]*B`.
- **Ragged prefill with no cached prefix (`B >= 1`).** Every request
  has `cached_len == 0`; individual `extend_len == device_len` values
  may differ. Query is right-padded to `max_query_len` along the S axis
  and each request supplies its own `actual_seq_lengths[i]` /
  `actual_seq_lengths_kv[i]`. Padding rows read `atten_mask=None`.
- **Pure-decode batches with mixed `cached_len` (`B >= 1`).** Every
  request has `extend_len == 1`; `cached_len` (equivalently `device_len`)
  may differ per request. `query_seq_lens = [1]*B`,
  `actual_seq_lengths_kv[i]` records each request's KV length, and
  `max_query_len == 1` so the FIA `atten_mask` argument is `None`.
- **Dynamic admission during decode.** A new request that arrives while
  one or more requests are running is prefilled in the very next tick as
  a single-request prefill batch, then joins the running set for joint
  decode on the subsequent tick. Existing requests' KV pages, page-table
  rows, and radix-cache state are bit-equal across the admission event.
- **Batch grow/shrink over a request's lifetime.** The batch size seen
  by the attention backend follows the observed admission/completion
  order, e.g. `[prefill1, prefill1, decode2, prefill1, decode3, decode2,
  decode1]`. Requests finish in `max_tokens` order, and each completion
  releases its page-table row and KV slots back to the pool.
- **Per-request `max_tokens` and `stop_token_ids` under batching.** The
  membership check runs per request inside `_process_last_data` after
  each request's own `append_host`; a stop for one request does not
  alter any other request's flag.
- **Cross-request KV isolation.** All observed multi-request runs have
  pair-wise-disjoint `table_idx` sets and pair-wise-disjoint physical
  page id sets across requests active in the same batch.

---

## 3. Out of scope (explicitly deferred)

Deferred to a later gate; the FIA backend raises `NotImplementedError`
today when asked for these shapes:

- **Ragged prefill with a cached prefix in the same batch.** Any batch
  where some request has `cached_len > 0` *and* some request has
  `extend_len > 1` while the batch is not entirely equal-length is
  refused. The FIA operator supports the shape in principle, but the
  wrapper does not yet emit the split-KV layout and the scheduler does
  not currently produce this shape (prefill-first admission below).
- **Chunked prefill continuation** (splitting a single request's prefill
  across two ticks).
- **Non-greedy sampling** (`temperature > 0`, top-k, top-p).
- **Preemption / re-scheduling of an in-flight request.** The scheduler
  currently admits new prefill work only when free page-table slots
  exist; it does not evict a running request to make room.
- **Cancellation and mid-request error recovery.** A raised exception
  inside the engine is not proven to leave the allocator, page table,
  and radix tree in their pre-batch state under partial completion.
- **Server / IPC path.** All Gate 2.2 evidence runs go through
  `offline_mode=True` with an in-process driver replacing
  `offline_receive_msg` / `offline_send_result`. The ZMQ / socket
  transport is not exercised.
- **Stress / soak testing.** No large-batch or long-duration runs.
- **Tensor parallel size > 1**, multi-die setups, and CUDA-graph capture
  on Ascend all remain out of scope, per Gate 2.1.
- **Performance / latency targets.** Gate 2.2 is a correctness gate.

---

## 4. Equal-length batching contract (Gate 2.2c)

`AscendFIABackend.prepare_metadata` classifies every batch by walking
`batch.reqs` once. When every request shares `extend_len`, `device_len`,
and `cached_len` with the head request, the batch takes the
**equal-length** path:

- `FIAMetadata.query_seq_len = extend_len` (shared) and
  `kv_seq_len = device_len` (shared).
- `query_seq_lens = [extend_len]*B`,
  `kv_seq_lens = [device_len]*B`.
- `actual_seq_lengths = [extend_len]*B`,
  `actual_seq_lengths_kv = [device_len]*B`.
- `max_query_len = extend_len`; the FIA `atten_mask` is a causal mask
  built once when `max_query_len > 1` and passed as `None` when
  `max_query_len == 1` (decode).
- `block_table` is a `[B, max_pages]` `int32` tensor sliced from
  `engine.page_table`; per-batch max page count comes from the shared
  `device_len`.
- KV writes use the flat `batch.out_loc` produced by the scheduler,
  concatenating per-request slots in request order.

Evidence (`gate22i_smoke1_equal.py`): two requests of length 4 with
`max_tokens=2` produce one prefill batch (`size=2`, `query_seq_len=4`)
followed by one decode batch (`size=2`, `query_seq_len=1`,
`kv_seq_len=5`). Pages `{0}` and `{1}` are disjoint; the allocator and
page-table free list return to their baselines.

---

## 5. Ragged prefill contract (Gate 2.2f)

When at least one request in a prefill batch differs from the head in
`extend_len` and every request has `cached_len == 0`, the batch takes
the **ragged-prefill** path:

- Query is right-padded to `max_query_len = max(extend_len)` along the
  S axis so the operator sees `[B, max_query_len, Hq, D]` BSND.
- `actual_seq_lengths[i] = extend_len[i]` and
  `actual_seq_lengths_kv[i] = device_len[i] == extend_len[i]`. The
  operator uses these lengths to short-circuit padded rows.
- `atten_mask` is the same causal mask used by equal-length prefill,
  sized to `max_query_len`.
- KV writes use the flat `batch.out_loc`, sized to
  `sum(extend_len)` — padded rows do **not** contribute to KV writes.

Evidence (`gate22f_paired.py`, and the ragged tests in
`tests/misc/test_ascend_fia_backend.py`): a `B=2` prefill with
`extend_len=[4, 2]` and `cached_len=[0, 0]` runs one FIA call per
layer with `query_seq_lens=[4,2]`, `kv_seq_lens=[4,2]`, and produces
disjoint per-request KV pages. Output shape matches the flat
`sum(extend_len)`-row layout that the scheduler expects.

---

## 6. Pure-decode mixed-cached-length contract (Gate 2.2f extension)

When every request has `extend_len == 1` but individual `cached_len`
(equivalently `device_len`) values differ, the batch takes the
**pure-decode** path:

- `query_seq_lens = [1]*B`, `max_query_len = 1`, so `atten_mask` is
  passed as `None` to FIA (decode has no future to mask).
- `actual_seq_lengths = [1]*B`,
  `actual_seq_lengths_kv[i] = device_len[i]` — each request answers its
  own KV length.
- `block_table` is `[B, max_pages]` where `max_pages =
  ceil(max(device_len) / page_size)`; rows are independently sliced per
  request from `engine.page_table[req.table_idx]` and the unused
  columns of shorter requests carry the page-table's default `0` padding.
- Query and output remain flat `[B, Hq, D]` at the wrapper boundary,
  reshaped to `[B, 1, Hq, D]` BSND only inside `forward`.

Evidence (four `test_gate22f_*` rows in
`tests/misc/test_ascend_fia_backend.py`): a `B=2` decode with
`cached_len=[4, 2]`, `device_len=[5, 3]` produces
`actual_seq_lengths_kv=[5, 3]`, `atten_mask=None`, and `store_kv`
receives the batch-wide `out_loc` unchanged.

---

## 7. Rejected shape today — ragged prefill with cached prefix

Batches where *some* request has `cached_len > 0` and *some* request has
`extend_len > 1`, and the batch is not entirely equal-length, raise
`NotImplementedError` inside `prepare_metadata`. The wrapper's
`raise` is the enforcement point; the classification loop reports the
first offender (uid, `cached_len`, `extend_len`, `device_len`).

The scheduler as of this gate does not currently emit this shape:

- New requests are admitted only when there is a free page-table slot.
- `_schedule_next_batch` picks any pending prefill work before any
  decode work each tick, so a newly-arrived request always gets its own
  single-request prefill tick before joining the running decode batch.
- The only ragged-prefill batches produced by the scheduler have
  `cached_len == 0` for every request in that batch (radix hits reuse
  the retained pages via a separate code path that shares the same
  `cached_len` across the batch — verified by the equal-length
  contract).

The consequence is that Gate 2.2 does not need a mixed-cached-prefix
ragged prefill path to be correct today, but adopting chunked prefill,
speculative decoding, or continuous batching in a later gate will
require lifting this rejection.

---

## 8. Dynamic admission strategy (Gate 2.2g)

`Scheduler.normal_loop` ingests messages via `offline_receive_msg` at
the top of every tick before it calls `_schedule_next_batch`. The
observed policy under Gate 2.2:

- **Prefill-before-decode priority.** If `PrefillManager.pending_list`
  is non-empty and there is a free page-table slot, `_schedule_next_batch`
  returns a prefill batch. The still-running decode set is queued for
  the next tick.
- **Single-request admission tick.** A newly-arrived request is
  prefilled by itself on the first tick after it arrives, then joins
  the joint decode batch on the following tick.
- **Bit-equal isolation.** For every observed admission event, the
  running requests' K-cache slices over `[0 .. device_len-1]` (i.e. the
  KV that had already been written to HBM) are `torch.equal`-bit-equal
  before and after the newcomer's prefill runs. The newcomer's KV lands
  in a disjoint set of pages and its page-table row is a fresh slot.
- **No preemption.** No running request is displaced or truncated by the
  admission of a new request. The active count is bounded above by
  `max_running_req` and by page-table capacity; today the admission
  order is FIFO on `pending_list`.

Evidence (`gate22g_dynamic.py`): submit A alone, tick prefill; while A
is running, push B; next tick prefills B alone; next tick runs an
`A+B` joint decode. Verified: A's tokens match its single-request
baseline `[15087, 11, 400, 4080]`; B's tokens match its single-request
baseline `[315, 279, 7042, 304]`; A's KV over its written prefix is
bit-equal across the admission event; page-table and page sets are
disjoint; the allocator returns to baseline on shutdown.

---

## 9. Batch grow/shrink timeline (Gate 2.2h)

Three requests A (`max_tokens=5`), B (`max_tokens=4`), C (`max_tokens=2`)
were admitted at three different scheduler ticks. The scheduler produced
the following batch-size timeline through the backend:

| Step | Phase | Size | uids |
| :--: | :--: | :--: | :--: |
| 0 | prefill | 1 | `{A}` |
| 1 | prefill | 1 | `{B}` |
| 2 | decode  | 2 | `{A,B}` |
| 3 | prefill | 1 | `{C}` |
| 4 | decode  | 3 | `{A,B,C}` |
| 5 | decode  | 2 | `{A,B}` (C finished) |
| 6 | decode  | 1 | `{A}` (B finished) |

At the size-3 decode step (step 4), `query_seq_lens = [1,1,1]`,
`max_query_len = 1` (so the FIA `atten_mask` is `None`), and
`actual_seq_lengths_kv` records each request's individual KV length.
The `table_idx` sets and physical page sets are pair-wise disjoint
across all three requests over the entire timeline. Every per-request
output matches its single-request baseline verbatim.

Shrink evidence: after C completes at step 5, the batch shrinks to
`{A, B}`; after B completes at step 6, only A remains. On the final
`shutdown()`, `CacheManager.available_size`, `free_slots`, and
`TableManager._free_slots` all match their pre-run baselines and
`torch.distributed.is_initialized() == False`.

---

## 10. KV and table isolation and reclamation

The following invariants hold across every Gate 2.2 smoke and probe:

- `TableManager._free_slots` is restored to its pre-batch state after
  every request completes.
- `CacheManager.available_size` and `len(CacheManager.free_slots)` are
  restored to their pre-batch baselines after `Scheduler.shutdown()`.
- Physical page id sets are pair-wise disjoint across requests active in
  the same batch (observed on every multi-request smoke).
- `table_idx` sets are pair-wise disjoint across requests active in the
  same batch.
- The layer-0 K-cache slice over each request's written prefix is
  `torch.equal`-bit-equal before and after any concurrent request's
  prefill or decode step.
- `torch.distributed.is_initialized() == False` at teardown.
  `init_process_group` and `all_reduce` counters remain zero.

---

## 11. Common invariants (all evidence runs)

Every Gate 2.2 smoke and probe additionally satisfies the Gate 2.1
invariants:

- `Scheduler.shutdown()` returns without raising.
- No CUDA-only surface (`torch.cuda.nvtx`, `torch.cuda.stream`, etc.)
  is invoked on the NPU code path.
- Greedy sampling, `page_size = 16`, `tp_size = 1`, `cache_type = radix`,
  `attention_backend = npu_fia`.

---

## 12. Test surface locked at this gate

Locked test files (all pass on the target platform under the standard
pytest invocation; row counts recorded at gate freeze):

- `tests/misc/test_ascend_fia_backend.py` — 65 rows (equal-length,
  ragged-prefill, and pure-decode mixed-cached-length contracts).
- `tests/misc/test_scheduler_device_backend.py` — 5 rows.
- `tests/misc/test_scheduler_sync_all_ranks.py` — 4 rows.
- `tests/misc/test_scheduler_stop_tokens.py` — 10 rows (inherited from
  Gate 2.1c; still valid under batching because the check is
  per-request).
- `tests/misc/test_sampling_params_stop_tokens.py` — 7 rows.
- `tests/misc/test_pyproject_config.py` — 14 rows.

Total: **105 rows pass** under the standard invocation on the target
platform.

---

## 13. Evidence trail (commit summary)

- `3f8146c` `attention/ascend_fia: support equal-length B>=1 batches (Gate 2.2c)`
- `3d6b3ed` `attention/ascend_fia: ragged prefill + pure-decode mixed cached (Gate 2.2f)`

Intermediate probe scripts (equal-length paired, ragged-prompt paired,
dynamic admission, three-request grow/shrink) are not committed to the
repository per policy — their outputs are summarised above.

---

## 14. Next-phase gaps

The following are the identified follow-ups for the gate after 2.2.
They are not regressions of Gate 2.2; they are shapes that Gate 2.2's
scope deliberately excludes:

- **Ragged prefill with cached prefix.** Lift the `NotImplementedError`
  in `prepare_metadata` once the scheduler emits this shape (chunked
  prefill or speculative decoding).
- **Cancellation and exception paths.** Prove that a mid-batch raise
  (allocator failure, model failure, driver disconnect) restores
  `CacheManager`, `TableManager`, and radix state to their pre-batch
  baselines.
- **Server / IPC transport.** Reproduce every Gate 2.2 smoke through
  the real ZMQ / socket path, not the in-process offline driver.
- **Preemption / re-scheduling.** Admit new prefill work by evicting a
  running request when the running set is at capacity.
- **Stress / soak.** Long-duration multi-request runs (hundreds of
  concurrent requests, thousands of ticks) with allocator and radix
  accounting checked on a rolling basis.
- **Batched `stop_token_ids` audit.** The per-request check works today;
  a dedicated batched-stop test row should be added when the scheduler
  supports per-tick multi-request stop concurrency beyond what
  `test_scheduler_stop_tokens.py` covers.
