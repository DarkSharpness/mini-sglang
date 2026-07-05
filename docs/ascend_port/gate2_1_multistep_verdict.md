# Gate 2.1 Verdict — Single-Request Multi-Step Scheduler on Ascend NPU

**Gate ID:** 2.1 (single-request multi-step scheduler on Ascend 910B1)
**Verdict:** PASS
**Frozen at commit:** `846c16f9bc8c9b6b09815546f03a2960f64b230c`
**Branch:** `gate2-multistep-scheduler`
**Tag:** `gate2-single-request-multistep`
**Date:** 2026-07-05

Gate 2.1 extends Gate 1 (single-request eager forward) with the full
scheduler-driven multi-step generation loop on a single Ascend 910B1 die. All
invariants below were verified on real hardware. Host paths, container IDs,
and credentials are recorded in private ops notes and are deliberately omitted
from this document.

---

## 1. Software and hardware

| Component | Version |
| --- | --- |
| Accelerator | Ascend 910B1 (64 GiB HBM per die) |
| Dies used | 1 (`npu:0`) |
| Python | 3.11.14 |
| PyTorch | 2.9.x |
| torch_npu | 2.9.0.post1+gitee7ba04 |
| apache-tvm-ffi | 0.1.4+ (declared floor; wheel `0.1.12` verified on clean install) |
| Model | Qwen3-0.6B, 28 layers, `float16` |
| Attention backend | `npu_fia` (Ascend Fused Infer Attention Score) |
| Cache type | `radix` |
| Page size | 16 tokens |
| Tensor parallel size | 1 |
| CUDA graph | disabled |

---

## 2. Supported scope

Gate 2.1 signs off on the following combinations only:

- Single-request path, greedy sampling only (`temperature=0.0`).
- Prefill + decode driven by `Scheduler.normal_loop` in `offline_mode=True`.
- Radix prefix cache with page size 16 and `ignore_eos={True,False}`.
- Per-request `max_tokens` termination and per-request `stop_token_ids`
  termination (immutable tuple, membership check happens after
  `req.append_host` so the stop token itself is retained in the output).
- Cross-page continuous generation (KV writes crossing a page boundary in
  the same request).
- Retained-prefix reuse via the Radix tree, and eviction back into the
  free-slot pool via the production `_allocate` chain.

---

## 3. Out of scope (explicitly deferred)

- Multi-request batching (concurrent prefill/decode across requests). See
  §10 for the identified follow-up gate.
- Tensor parallel size > 1 and any multi-die setup.
- Non-greedy sampling (top-k, top-p, temperature != 0).
- CUDA-graph capture on Ascend.
- Continuous-batching admission / preemption policies.
- Speculative or draft-model decoding.
- Any performance / latency target — Gate 2.1 is a correctness gate.

---

## 4. Multi-step generation evidence

Prompt `[3, 7, 11, 15]`, `max_tokens=8`, `temperature=0.0`,
`ignore_eos=True`, page size 16:

- Emitted tokens: `[15087, 11, 400, 4080, 16, 11, 15, 15087]`
- Finished pattern: exactly the last step, no earlier flag.
- One prefill batch (`positions=[0,1,2,3]`) followed by seven decode batches
  (`positions=[4]` … `[10]`).
- Distributed init counter `= 0`, `all_reduce` counter `= 0`.
- `Scheduler.shutdown()` returns cleanly; CacheManager `available_size`
  returns to its baseline.

---

## 5. Stop-token semantics

`SamplingParams.stop_token_ids: tuple[int, ...] = ()`. The tuple default is
immutable and shared across instances; a non-empty tuple activates a
membership check in `Scheduler._process_last_data` **after**
`req.append_host` runs, so the stop token itself remains in the output
sequence.

Prompt `[3, 7, 11, 15]`, `max_tokens=8`, `ignore_eos=True`,
`stop_token_ids=(11,)`:

- Emitted tokens: `[15087, 11]`
- Finished pattern: `[False, True]`
- The stop-token check is independent of `ignore_eos`. `ignore_eos` only
  silences the tokenizer EOS; it does not disable user-declared stop
  tokens. An empty tuple restores pre-Gate-2.1c behaviour verbatim.
- No third batch is produced; distributed counters remain zero and the
  allocator is fully restored on shutdown.

Test coverage: `tests/misc/test_sampling_params_stop_tokens.py` (7 rows)
plus `tests/misc/test_scheduler_stop_tokens.py` (10 rows) lock the field
contract and the scheduler behaviour rows.

---

## 6. Cross-page KV contract

Prompt `[3, 7, 11, 15]`, `max_tokens=20`, `ignore_eos=True`, page size 16
crosses one page boundary in the same request:

- Prefill fills physical page 0 slots `0..3`.
- Decode steps `1..15` fill page 0 slots `4..15` (all 28 attention layers).
- Step 16 opens page 1 slot 0 (`out_loc=16`, `positions=[16]`); the page
  table row extends from `[page0, page1, …]`.
- After step 19 the request holds `cached_len=23`, `device_len=24` on
  logical page slots `0..6` of page 1; slots `7..15` remain zero (`K=0,
  V=0` across all layers) and page 2 is untouched.
- KV writes on page 0 are frozen once the page is full — subsequent
  reuse must not rewrite the retained content (see §7).

---

## 7. Radix reuse and eviction contract

Two consecutive requests share a 16-token prefix:

**Phase A — build the retained page.** Same prompt as §6; after the
request finishes the radix root has one child with `page_id=0`,
`length=16`, `ref_count=0`, and `evictable_size=16`.

**Phase B — reuse.** Submit `prefix16 + [123]` with `max_tokens=1`. The
scheduler produces exactly one prefill batch with `cached_len=16`,
`device_len=17`, `positions=[16]`, and 28 FIA calls / 28 `store_kv` calls
for the single new token. The retained physical page 0 is bit-equal
before and after Phase B (`torch.equal` on both K and V buffers). The
retained node's `page_id=0` still points into the radix root, the new
token lands in a fresh physical slot (page 2 slot 0 in the observed run),
and `available_size` is unchanged.

**Phase C — production eviction round-trip.** Force the eviction branch
by calling `CacheManager._allocate(free_pages + evictable_pages)`. This
invokes `RadixPrefixCache.evict` and concatenates the evicted pages into
`free_slots`, then `_free` returns them. Observed transitions:

- `free_pages`: baseline → baseline-1 (retained) → 0 (post `_allocate`) → baseline (restored)
- `evictable_size`: `0 → 16 → 0` (post-eviction)
- Radix root children: `[{page_id: 0, len: 16, ref: 0}] → []` after evict.
- `available_size` returns to the pre-Gate-2.1 baseline.

Probe verdict: `PROBE_RESULT=PASS`. Distributed remains uninitialised;
`shutdown()` clean.

---

## 8. Dependency requirement — `apache-tvm-ffi`

`RadixPrefixCache._tree_walk → get_match_len → fast_compare_key` (loaded
via `tvm_ffi.cpp.load` inside `minisgl.kernel.utils.load_aot`) runs on
every prefix match on every supported device (CUDA, Ascend NPU, CPU).
Before Gate 2.1g, `apache-tvm-ffi>=0.1.4` was declared only in the
`[project.optional-dependencies].cuda` extra, and the project's install
command (both `README.md` and the container `Dockerfile`) is
`pip install -e .` without any extra. Clean Ascend installs therefore
missed the wheel and crashed with
`ModuleNotFoundError: No module named 'tvm_ffi'` on the first prefix
scheduling call.

Gate 2.1g moved the pin into base `[project].dependencies` (no
duplication with the `cuda` extra) and added three structural tests
under `tests/misc/test_pyproject_config.py`:

- `test_cross_device_required_dep_is_in_base_dependencies` — must be a
  base dep.
- `test_cross_device_required_dep_not_only_in_cuda_extra` — must not be
  duplicated in the `cuda` extra.
- `test_apache_tvm_ffi_pin_preserved` — pin string
  `apache-tvm-ffi>=0.1.4` is exact.

Clean-venv smoke: `pip install -e .` in a fresh Python 3.11 venv
auto-installs `apache-tvm-ffi==0.1.12`; `import tvm_ffi`,
`from tvm_ffi.cpp import load`, and the three-state `fast_compare_key`
smoke (identical → 5, diff-at-3 → 3, disjoint → 0) all pass.

---

## 9. Common invariants (all evidence runs)

Every smoke, probe, and test run recorded above satisfies:

- `torch.distributed.is_initialized() == False` at teardown.
- `Scheduler.shutdown()` returns without raising.
- `CacheManager.available_size` post-shutdown equals its baseline.
- No CUDA-only surface (`torch.cuda.nvtx`, `torch.cuda.stream`, etc.)
  is invoked on the NPU code path (locked by `test_sampler_no_cuda_nvtx`
  and `test_device_runtime` isolation-mode passes).

---

## 10. Known follow-up gap — multi-request batching

Gate 2.1 verifies a single in-flight request end-to-end. The next Gate
(2.2, to be scoped) must cover:

- Concurrent prefill of two or more requests in the same batch.
- Mixed prefill+decode batches (chunked prefill continuation).
- Prefix-cache reuse across two requests submitted in the same
  scheduler tick.
- Correct account of KV pages, `free_slots`, and `evictable_size` when
  requests finish in a different order from the order they were
  submitted.
- Correct interaction between per-request `stop_token_ids` and batched
  decode (the current membership check runs per request in
  `_process_last_data`; batched cases must not conflate flags).

No runtime code should need to change for Gate 2.1's single-request
scope, but the batch scheduler and cache manager will exercise code
paths not covered by this verdict.

---

## 11. Evidence trail (commit summary)

- `ff0549e` `scheduler: honour per-request stop_token_ids (Gate 2.1c)`
- `846c16f` `deps: move apache-tvm-ffi to base dependencies (Gate 2.1g)`

Intermediate probe scripts (single-page decode, cross-page decode,
retained prefix reuse, eviction round-trip, dependency smoke) are not
committed to the repository per policy — their outputs are summarised
above.

---

## 12. Test surface locked at this gate

- `tests/misc/test_sampling_params_stop_tokens.py` — 7 rows
- `tests/misc/test_scheduler_stop_tokens.py` — 10 rows
- `tests/misc/test_pyproject_config.py` — 14 rows (adds `apache-tvm-ffi`
  cross-device placement)
- `tests/misc/test_scheduler_device_backend.py` — 5 rows
- `tests/misc/test_scheduler_sync_all_ranks.py` — 4 rows
- `tests/misc/test_device.py` — 8 rows
- `tests/misc/test_device_runtime.py` — 59 rows (passing under module
  isolation; per-module `sys.modules` state is a pre-existing test
  concern, not a Gate 2.1 regression)
- `tests/misc/test_sampler_no_cuda_nvtx.py` — 4 rows

Total: **111 rows pass** under isolated invocation on the target
platform.
