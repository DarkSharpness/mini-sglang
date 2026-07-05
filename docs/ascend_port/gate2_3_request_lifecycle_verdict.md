# Gate 2.3 Verdict — Request Lifecycle and Cancel Protocol on Ascend NPU

**Gate ID:** 2.3 (request lifecycle and cancel protocol on Ascend 910B1)
**Verdict:** PASS
**Branch:** `gate2-request-lifecycle`
**Runtime frozen at commit:** `ac1bb8e58de7987c581a71806bf1cde3122e1be0`
**Tag:** `gate2-request-lifecycle` (points to the verdict-freeze commit)
**Date:** 2026-07-05

Gate 2.3 hardens the request lifecycle in the Ascend 910B1 port along the
axes that Gate 2.2 deliberately excluded (see
`gate2_2_multirequest_verdict.md` §14 "Cancellation and exception paths"):
transactional page allocation, sampler commit atomicity, shutdown drain,
overlap-loop abort fence, and end-to-end abort acknowledgement between
the frontend, tokenizer, and scheduler. All invariants below were
verified on a single Ascend 910B1 die on real hardware or through
hermetic tests committed to this branch. Host paths, container IDs, and
credentials are recorded in private ops notes and are deliberately
omitted from this document.

---

## 1. Software and hardware

Inherits Gate 2.2's platform (see `gate2_2_multirequest_verdict.md` §1):

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
| Scheduler mode | `offline_mode=True` for in-process probes; server
  path unchanged |

---

## 2. Scope

Gate 2.3 signs off on the following behaviours, in addition to the
Gate 2.2 multi-request batching scope:

- **Transactional page allocation in `_prepare_batch`.** A failure
  raised anywhere between KV-page allocation and the sampler
  commit boundary restores the allocator (`CacheManager.available_size`
  and `CacheManager.free_slots` cardinality) and the `page_table` rows
  of unrelated requests to the pre-batch snapshot.
- **Atomic request-state commit past the sampler.** The scheduler
  writes back `input_ids`, `device_len`, and `cached_len` only after
  the sampler has returned a token successfully. A raise inside the
  sampler leaves every request's logical length and host-token buffer
  unchanged.
- **Shutdown drain.** `Scheduler.shutdown` frees every pending and
  running request before tearing down the engine, so the allocator,
  page table, and radix cache return to their pre-run baseline
  regardless of the state at shutdown time.
- **Overlap-loop abort fence.** An `AbortBackendMsg` that arrives while
  the target `uid` is inflight on device is deferred to
  `deferred_abort_uids`. The uid is yanked from the schedulable set
  immediately, its post-fence sampled token is suppressed, and its
  resources are freed exactly once after `_process_last_data` has
  drained the fence tail. Sibling requests in the same batch keep
  their sampled tokens and stay running.
- **End-to-end abort acknowledgement.** The scheduler emits an
  `AbortAckMsg` on the tokenizer channel after — and only after —
  `_free_req_resources` has run for the aborted uid. The tokenizer
  forwards it as `AbortAckReply` to the frontend, and the frontend
  drains its `abort_pending` state and unblocks any waiter. The
  acknowledgement path is a first-class message channel, never a
  cancel-stub `DetokenizeMsg(next_token=-1)`.
- **Frontend abort-pending state machine.** Between
  `abort_user(uid)` and receipt of `AbortAckReply(uid)`, the frontend
  drops any late `UserReply(uid)` that races the ack, keeps the
  request's ack-map and event-map entries alive, and cleans all four
  buckets (`ack_map`, `event_map`, `abort_pending`,
  `abort_pending_events`) idempotently on ack arrival.

---

## 3. Explicit non-scope

Gate 2.3 does NOT attempt, and does NOT attest to, the following:

- **In-process sampler recovery.** After a sampler failure, the
  scheduler propagates the exception unchanged; there is no attempt to
  restart the sampler on the same tick, re-sample from the same
  hidden states, or resume the aborted forward. Gate 2.3c's assertion
  is atomicity of the commit boundary, not recovery.
- **`model.forward` / FIA in-process recovery.** A raise from inside
  the attention or transformer stack terminates the tick. There is
  no partial retry.
- **CPU copy / event-record failures past the sampler.** Once the
  sampler has committed a token and the scheduler has advanced the
  request, a raise in the CPU-side detokenize-copy or the event-record
  path is not covered — recovery would require reverting a
  committed token, which is out of scope.
- **Full HTTP + ZMQ cross-process cancel.** The abort acknowledgement
  contract is proven with the in-process offline driver and hermetic
  tests. Cross-process transport regressions are Gate 2.4 material.
- **Server crash / restart lifecycle.** No client-side reconnect or
  request-replay semantics are asserted.
- **Tensor-parallel size > 1 lifecycle.** All evidence is single-die.
- **Long soak.** No thousand-tick, hundreds-of-request stress run is
  part of the gate.

---

## 4. Gate 2.3a — Pre-work audit

Before touching runtime code, the pre-2.3 request lifecycle was audited
against the crash / cancel / shutdown scenarios that Gate 2.2 identified
as out-of-scope. The relevant findings:

- `_prepare_batch` allocated KV pages incrementally without a rollback
  path. A raise between KV-page allocation and the sampler committing
  a token would leak pages back onto no free list, permanently
  reducing `available_size`.
- `Engine.forward_batch` wrote back `input_ids`, `device_len`,
  `cached_len` before the sampler committed. A sampler failure
  therefore left the request advanced by one token that was never
  produced, breaking the invariant that host token counts match the
  scheduler's view.
- `Scheduler.shutdown` tore down the engine without walking the
  pending or running sets. In-flight requests leaked their KV pages
  and their table-manager slots, so a re-instantiated scheduler
  observed non-baseline allocator state.
- `overlap_loop` had no fence between "abort msg observed on host"
  and "device forward for the target uid still in flight". A
  well-timed abort could free a KV page while it was being read by a
  running attention kernel.
- `AbortBackendMsg` had no acknowledgement path. The frontend used a
  timeout-based heuristic to declare the request cancelled, which
  raced natural finishes.

These findings drove the changes committed in Gate 2.3b through 2.3f.

---

## 5. Gate 2.3b — Transactional page allocation

**Runtime commit:** `d520a71` — `scheduler: transactional page allocation rollback in _prepare_batch`

Invariants asserted:

- Every KV page allocated inside `_prepare_batch` is tracked. On any
  raise between the first allocation and the successful return of the
  method, `rollback_allocation` returns the allocated pages to the
  cache manager.
- The `cache_manager`'s `available_size` and the length of
  `free_slots` after the rollback are bit-equal to the pre-batch
  snapshot.
- The `page_table` rows of requests unrelated to the failing batch
  are not touched by the rollback.
- Scratch attributes (`batch.positions`, `out_loc`, `padded_reqs`,
  `attn_metadata`) written inside the failing method are cleared, so
  no stale pointer to a freed page survives.
- A fresh request pushed after the rollback completes normally and
  the allocator returns to the same free-page cardinality once that
  request retires.

Explicitly out-of-scope for 2.3b: `table_manager.allocate` runs in the
upstream `PrefillAdder` step, not inside `_prepare_batch`. The
transaction covers only the KV-page allocations that `_prepare_batch`
itself performs.

Regression coverage: `tests/misc/test_scheduler_prepare_batch_txn.py`
— 5 tests, PASS.

Real-device evidence: fresh 910B1 probe run in this Gate 2.3g freeze.

```
pre_avail=271840   pre_free_pages=16990
post_avail=271840  post_free_pages=16990   (rollback restored allocator)
post_fu_avail=271856 post_fu_free_pages=16991 (followup req retired cleanly)
prep_err='gate2.3g probe A synthetic rejection'
first_prefix_row_pre  == first_prefix_row_post  (retained prefix untouched)
VERDICT: PASS
```

---

## 6. Gate 2.3c — Sampler commit atomicity

**Runtime commit:** `f56ce2a` — `engine: atomic request-state commit past sampler in forward_batch`

Invariants asserted:

- `Engine.forward_batch` writes back `input_ids`, `device_len`, and
  `cached_len` only after the sampler has returned. A raise inside
  the sampler leaves every request's logical state unchanged.
- The exception raised inside the sampler propagates unchanged; the
  scheduler does NOT swallow, translate, or retry it.
- The host CPU token buffer is not appended to on the failing tick.

Regression coverage:
`tests/misc/test_engine_forward_sampler_atomic.py` — 5 tests, PASS.

Real-device evidence status:

```
Sampler atomicity evidence status:
- Original Gate 2.3c 910B1 evidence: PASS
- Evidence commit: f56ce2a
- Gate 2.3g fresh re-attestation: NOT RUN
- Reason: the freeze-only probe addressed the wrong model wrapper
  (`Qwen3ForCausalLM.named_modules`), so it failed before fault
  injection.
- This is a probe-construction defect, not a runtime failure.
- No runtime or probe code was modified during Gate 2.3g.
```

The Gate 2.3c capability itself is attested by the original 910B1
evidence recorded at `f56ce2a` and by the 5-row hermetic regression
that passes on the current branch.

---

## 7. Gate 2.3d — Shutdown drain

**Runtime commit:** `4ef0c15` — `scheduler: shutdown-time drain of pending + running requests`

Invariants asserted:

- `Scheduler.shutdown` walks `prefill_manager.pending_list` and
  `decode_manager.running_reqs` before the engine is torn down.
- Every drained request has its table slot returned to
  `TableManager._free_slots` and its KV pages returned via
  `cache_manager.cache_req(finished=True)`.
- After shutdown, the allocator (`available_size`, `free_slots`
  cardinality) and the `TableManager._free_slots` multiset are
  bit-equal to the pre-run baseline.
- The drain runs even when both sets are non-empty.

Regression coverage: `tests/misc/test_scheduler_shutdown_drain.py`
— 8 tests, PASS.

Real-device evidence: fresh 910B1 probe run in this Gate 2.3g freeze.

```
Pre-shutdown: running=1  pending=1  (uids A running, B pending)
              table_free=[0,1,2,3,4,5,6]  avail=271856  free_pages=16991
Post-shutdown: running=0  pending=0
               table_free=[0,1,2,3,4,5,6,7]  avail=271872  free_pages=16992
               (identical to baseline)
VERDICT: PASS (shutdown drained pending+running to baseline)
```

---

## 8. Gate 2.3e — Overlap-loop abort fence

**Runtime commit:** `ada1688` — `scheduler: overlap-loop abort fence (Gate 2.3e)`

Invariants asserted:

- When `_process_one_msg` sees `AbortBackendMsg(uid)` while `uid` is
  in `inflight_uids`, the uid is added to `deferred_abort_uids` and
  removed from `decode_manager.running_reqs` immediately, without
  freeing any resources yet.
- `_process_last_data` emits no `DetokenizeMsg` for a deferred uid
  and does not append to that uid's host token buffer.
- `_apply_deferred_aborts` runs after `_process_last_data` completes
  and frees each deferred req exactly once (`_free_req_resources`
  called with the correct `Req` instance).
- Sibling requests in the same batch keep their sampled tokens and
  stay in `decode_manager.running_reqs`.
- Two abort msgs for the same inflight uid are idempotent — the set
  membership guarantees a single free.
- `normal_loop` is unaffected: `inflight_uids` is never populated
  under `normal_loop`, so aborts take the immediate-free branch.

Regression coverage:
`tests/misc/test_scheduler_overlap_abort_fence.py` — 7 tests, PASS.

Real-device evidence status:

```
Overlap abort fence evidence status:
- Original Gate 2.3e 910B1 evidence: PASS
- Evidence commit: ada1688
- Current regression:
  test_scheduler_overlap_abort_fence.py — 7 tests PASS
- Gate 2.3g fresh real-probe re-attestation: NOT COMPLETED
- Reason:
  the pre-Gate-2.3f probe assumed every scheduler reply was a
  DetokenizeMsg. Gate 2.3f legitimately added AbortAckMsg to the
  same reply channel, so the probe failed in post-run result parsing
  when it accessed `next_token` on an AbortAckMsg.
- The overlap fence itself reached and executed its intended runtime
  path before the parser failure.
- This is historical probe/schema drift, not an overlap-fence runtime
  failure.
- No runtime or probe code was modified during Gate 2.3g.
```

The Gate 2.3e capability itself is attested by the original 910B1
evidence recorded at `ada1688` and by the 7-row hermetic regression
that passes on the current branch. The current-session Gate 2.3f
end-to-end abort ack probe (§9) also exercises the overlap fence code
path under a deferred abort and independently confirms the fence
behaviour on device.

---

## 9. Gate 2.3f — End-to-end abort acknowledgement

**Runtime commit:** `ac1bb8e` — `scheduler+server: end-to-end abort acknowledgement (Gate 2.3f)`

Invariants asserted:

- New message types are added, both idempotent by design:
  - `AbortAckMsg(uid)` on the tokenizer channel
    (`BaseTokenizerMsg` subclass).
  - `AbortAckReply(uid)` on the frontend channel
    (`BaseFrontendMsg` subclass).
- The scheduler appends the aborted uid to `_pending_abort_acks`
  immediately after `_free_req_resources(req)` — never before. The
  queue is drained by `_flush_pending_acks()` at the end of the tick.
- `_flush_pending_acks()` is called at both `normal_loop` and
  `overlap_loop` tails, so both scheduling loops emit the ack on the
  same tick the free happened.
- The tokenizer forwards a batched `AbortAckMsg` as an
  `AbortAckReply` batch (unwrapped to a single message when there is
  exactly one entry) on the frontend channel.
- The frontend's `abort_pending` set is populated by
  `abort_user(uid)` and cleared only by receipt of an
  `AbortAckReply(uid)`. A late `UserReply(uid)` that arrives between
  those two events is silently dropped by the listen dispatch.
- Cleanup on ack is idempotent: pop from `ack_map`, `event_map`,
  `abort_pending`, `abort_pending_events`; a duplicate ack is a
  strict no-op.
- Normal end-of-stream (EOS or stop_token) is undisturbed —
  no abort machinery fires on natural finish.
- Abort semantics by request state (all covered by hermetic tests):
  - **Waiting** (in `pending_list`, not yet forwarded): request
    removed from pending, KV resources freed, ack emitted.
  - **Running** under `normal_loop` (no fence): immediate-free
    branch, ack emitted the same tick.
  - **Running** under `overlap_loop` with uid inflight: fence branch
    defers the free to after `_process_last_data`, then acks.
  - **Unknown uid** (never seen by the scheduler): the abort is a
    no-op runtime-side; no ack is emitted. The frontend's
    `abort_pending` entry, if any, waits until the frontend times
    out or the caller resolves it out-of-band.
  - **Duplicate abort for the same uid** while it is inflight:
    idempotent via `deferred_abort_uids` set membership; a single
    free, a single ack.

Regression coverage: `tests/misc/test_scheduler_abort_ack.py` —
8 tests, PASS. (Covers A: non-inflight ack after free; B: deferred
inflight ack after apply; C: waiting-list abort ack; D: sibling
survives partial abort; E: idempotent duplicate abort ack; F:
frontend drops racing `UserReply`; G: natural finish emits no ack;
H: tokenizer forwards `AbortAckMsg` to frontend as `AbortAckReply`.)

Real-device evidence: fresh 910B1 probe run in this Gate 2.3g freeze,
covering four flows on one scheduler instance:

```
flow1 (normal_loop immediate abort):
  uid_in_running=False  uid_in_pending=False
  acks=1  detoks=1 (the pre-abort prefill token, not a stale finish)

flow2 (overlap_loop deferred abort):
  acks=1  detoks=0
  timeline: free_idx=0  ackq_idx=1   (free precedes ack, invariant holds)

flow3 (frontend late-token drop + idempotent cleanup):
  late_dropped=True  frontend_cleaned=True  dup_ok=True

flow4 (follow-up on the SAME scheduler after both abort flows):
  followup_ok=True

Post-shutdown allocator: avail=271856/271856  pages=16991/16991
VERDICT: PASS
```

---

## 10. Evidence summary

| Capability | Original evidence | Gate 2.3g re-attestation |
| --- | --- | --- |
| allocation rollback | prior 910B1 PASS | fresh real-probe PASS |
| sampler state atomicity | Gate 2.3c PASS / `f56ce2a` | not rerun; probe targets wrong model wrapper |
| shutdown drain | prior 910B1 PASS | fresh real-probe PASS |
| overlap abort fence | Gate 2.3e PASS / `ada1688` | hermetic 7/7 PASS; old real probe incompatible with AbortAck schema |
| abort acknowledgement | Gate 2.3f PASS / `ac1bb8e` | fresh message-flow PASS |

Summary:

```
Fresh real probes: A, C, E PASS
Inherited real evidence: B, D
Current hermetic regressions for B and D: PASS
```

---

## 11. Test surface locked at this gate

Locked test files (all pass on the target platform under the standard
pytest invocation; row counts recorded at gate freeze):

- `tests/misc/test_scheduler_prepare_batch_txn.py` — 5 rows (Gate 2.3b).
- `tests/misc/test_engine_forward_sampler_atomic.py` — 5 rows (Gate 2.3c).
- `tests/misc/test_scheduler_shutdown_drain.py` — 8 rows (Gate 2.3d).
- `tests/misc/test_scheduler_overlap_abort_fence.py` — 7 rows (Gate 2.3e).
- `tests/misc/test_scheduler_abort_ack.py` — 8 rows (Gate 2.3f).

Gate 2.3 total: **33 rows pass**.

Inherited-and-still-passing baseline surface exercised in this freeze:

- `tests/misc/test_ascend_fia_backend.py` — 72 rows.
- `tests/misc/test_pyproject_config.py` — 11 rows.

Inherited total: 83 rows pass. Combined this-gate + baseline pytest
count exercised: 116 rows.

---

## 12. Evidence trail (commit summary)

- `d520a71` `scheduler: transactional page allocation rollback in _prepare_batch (Gate 2.3b)`
- `f56ce2a` `engine: atomic request-state commit past sampler in forward_batch (Gate 2.3c)`
- `4ef0c15` `scheduler: shutdown-time drain of pending + running requests (Gate 2.3d)`
- `ada1688` `scheduler: overlap-loop abort fence (Gate 2.3e)`
- `ac1bb8e` `scheduler+server: end-to-end abort acknowledgement (Gate 2.3f)`

Intermediate probe scripts (probe A `_prepare_batch` rollback, probe C
shutdown drain, probe E abort message flow) are not committed to the
repository per policy — their outputs are summarised in §5, §7, §9
above.

---

## 13. Freeze integrity note

```
Gate 2.3g did not alter runtime or fault-injection code.
A failed sampler re-attestation script was excluded because it did not
reach the intended injection point. The verdict relies on the previously
recorded 910B1 Gate 2.3c evidence and its committed regression tests.
```

```
Two historical real-device re-attestation scripts were excluded from
fresh PASS accounting because they failed before completing their
intended assertions:
1. Probe B targeted the wrong model wrapper before sampler injection.
2. Probe D pre-dated AbortAckMsg and assumed a DetokenizeMsg-only
   scheduler reply channel.
Neither failure indicates a runtime regression. Gate 2.3g modified
neither runtime nor fault-injection code and relies on the original
committed 910B1 evidence plus current unchanged regression tests.
```

---

## 14. Next-phase gaps

These are follow-ups after Gate 2.3. They are not regressions of Gate 2.3;
they are shapes Gate 2.3's scope deliberately excludes:

- **In-process sampler / forward recovery.** Prove — or explicitly
  decline to prove — that a raise inside the sampler or the
  transformer stack can be recovered without tearing down the
  scheduler. Today the exception unwinds the tick and requires an
  external restart.
- **CPU-side detokenize copy / event-record failure.** Once the
  sampler has committed, a raise in the CPU copy or event-record
  path is not covered; recovery would require rolling back a
  committed token.
- **Cross-process transport.** Reproduce the abort-ack contract
  across the real ZMQ / HTTP path, not just the in-process offline
  driver.
- **Server crash / restart lifecycle.** Client-side reconnect and
  request replay after a server restart are not covered.
- **Tensor-parallel size > 1.** All lifecycle proofs here are single
  die; the fence-and-ack contract on TP > 1 needs its own gate.
- **Long soak.** Hundreds of concurrent requests, thousands of
  ticks, rolling allocator and radix accounting.
- **Abort-during-prefill on a chunked request.** Chunked prefill is
  not enabled at this gate; when it lands, the abort contract needs
  a re-audit against `ChunkedReq`'s partial-commit invariants.
