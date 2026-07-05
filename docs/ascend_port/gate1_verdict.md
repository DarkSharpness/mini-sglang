# Gate 1 Verdict — Single-Request Eager Inference on Ascend NPU

**Gate ID:** 1 (single-request eager on Ascend 910B1)
**Verdict:** PASS
**Frozen at commit:** `0206c54398c4c7ab567aa9b0ef11cd1683ed69b2`
**Branch:** `gate1-engine-bootstrap`
**Tag:** `gate1-single-request-eager`
**Date:** 2026-07-05

Gate 1 covers the minimum viable path: one user request, greedy sampling,
eager (non-graph) forward, tensor parallel size 1, single Ascend NPU. All
Gate 1 invariants below were verified end-to-end on real hardware.

---

## 1. Hardware and software versions

| Component | Version |
| --- | --- |
| Accelerator | Ascend 910B1 (64 GiB HBM per die, 8 dies available) |
| Dies used by Gate 1 | 1 (`npu:0`) |
| Host OS | Ubuntu 22.04.5 LTS (jammy) |
| Python | 3.11.14 |
| PyTorch | 2.9.0+cpu (CUDA build not present) |
| torch_npu | 2.9.0.post1+gitee7ba04 |
| CANN | 8.5.1 (from `/usr/local/Ascend/cann-8.5.1`) |
| Model under test | Qwen3-0.6B, 28 transformer layers, `float16` |
| Attention backend | `npu_fia` (Ascend Fused Infer Attention Score) |
| Cache type | `radix` |
| Page size | 16 |

Full path locations, container IDs, and workstation credentials are recorded
outside the repository (private ops notes) and are deliberately omitted from
this document.

---

## 2. Supported scope

Gate 1 signs off on the following combinations only.

**In scope:**

* Single accelerator, `tp_info.size == 1`.
* Eager mode only (`cuda_graph_bs=[]`, `cuda_graph_max_bs=0`).
* One user request per invocation, batch size 1.
* Greedy sampling (`temperature=0.0`).
* Attention backend: `npu_fia`.
* KV cache layout: Ascend-native paged layout.
* `offline_mode=True` scheduler driver (no ZMQ tokenizer / detokenizer).
* Distributed runtime is initialized only when `tp_info.size > 1`. For
  Gate 1 (TP=1) no `torch.distributed` backend is brought up on any device.

**Out of scope — explicitly not covered by Gate 1:**

* TP > 1 collective execution on NPU (HCCL smoke test only asserts init;
  no forward/decode across ranks).
* Continuous batching / concurrent requests.
* CUDA-graph or ACL-graph capture on NPU.
* Non-greedy samplers (top-k / top-p / temperature).
* Cross-request radix prefix reuse.
* Long-context correctness beyond the `max_seq_len_override=1024` used in
  the smoke probes.
* Speculative decoding, chunked prefill across multiple physical batches,
  MoE routing on NPU.
* CPU-only inference correctness (only dispatch surfaces are tested).

---

## 3. Key commits (chronological, oldest first)

Portable device / distributed runtime foundations:

* `806ee1b` docs: audit Ascend port dependencies
* `5c61c7c` feat: add portable device detection
* `fd63d66` feat: add distributed backend selection
* `9171ae2` feat: add portable distributed runtime initialization
* `949a7e9` refactor: make bootstrap package imports lazy
* `f7b4578` build: make CUDA dependencies optional
* `4d9a3ad` test: add Ascend HCCL smoke test
* `b7a613a` refactor: use shared device binding in engine
* `afb315d` feat: add portable device stream runtime
* `d6f826e` refactor: use portable stream runtime in engine
* `ca5b68f` feat: add portable device event runtime
* `32eb41e` refactor: use portable events in engine forward
* `e33d9fc` feat: add portable device memory maintenance
* `51455c7` refactor: use portable memory runtime in engine
* `aaf7f15` feat: add portable free memory query
* `9024a41` refactor: use portable free memory query in engine
* `395608c` refactor: use portable memory runtime in graph runner

Ascend-native KV layout and FIA attention:

* `ba49c56` feat: add Ascend-native paged KV layout
* `d15a0ab` feat: register Ascend FIA attention backend
* `998d0b9` feat: add single-request FIA attention metadata
* `2133c38` feat: implement single-request FIA attention forward

Ascend-native model layers and dispatch:

* `4ff3f1b` feat: add Ascend RMSNorm runtime dispatch
* `f717828` feat: add Ascend rotary embedding dispatch
* `75dbda8` fix: make NVTX annotations safe on non-CUDA devices
* `bf3724a` feat: add native embedding dispatch for NPU and CPU
* `73797c7` feat: add NPU and CPU SwiGLU dispatch
* `35aeab4` fix: reshape attention QKV before normalization and RoPE
* `61a2b4e` fix: key RoPE cache by explicit device

Gate 1.11 fixes (bring the smoke probe green):

* `0c66e5d` fix: skip distributed init for NPU single-rank engine
* `df11e1d` fix: drop CUDA-only NVTX context inside Sampler.sample
* `40bc57d` fix: route Scheduler stream lifecycle through device_runtime
* `0206c54` scheduler: guard sync_all_ranks against a missing CPU process group

---

## 4. Prefill / decode invariants

Verified with a single-request scheduler-driven probe (Scheduler +
Engine + attention backend, `offline_mode=True`, driven through the same
`normal_loop` branch that `run_forever` executes).

**Input:** `input_ids = [3, 7, 11, 15]`, `SamplingParams(temperature=0.0, max_tokens=2)`
**Model:** Qwen3-0.6B, dtype `float16`.

**Two-token output (greedy):**

| Step | Batch phase | Batch size | Positions | `out_loc` | Next token | Finished |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | prefill | 1 | `[0, 1, 2, 3]` | `[0, 1, 2, 3]` | `15087` | False |
| 1 | decode | 1 | `[4]` | `[4]` | `11` | True |

Post-run request state on the tracked request:

* `req.cached_len = 5`, `req.device_len = 6`
* `req.input_ids = [3, 7, 11, 15, 15087, 11]`
* `req.can_decode = False` (max_tokens reached)
* `token_pool[table_idx, :16] = [3, 7, 11, 15, 15087, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`

Reference tokens `[15087, 11]` are the frozen Gate 1 fixture. Any future
change that shifts these values must be treated as a regression.

---

## 5. KV cache and FIA contracts

**KV write contract (Ascend-native paged layout, `page_size=16`,
28 transformer layers):**

* Baseline: `_k_buffer[:, 0, :, :16, :]` and `_v_buffer[:, 0, :, :16, :]`
  are zero before the request enters the scheduler.
* After the two-step run, per-slot layer-hit counts (out of 28 layers):

  ```
  K per-slot layer hits: [28, 28, 28, 28, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  V per-slot layer hits: [28, 28, 28, 28, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ```

  Slots 0..4 receive writes from every layer for both K and V (4 prefill
  tokens plus 1 decode token = 5 slots). Slots 5..15 remain untouched.

**FIA call contract:**

* `torch_npu.npu_fused_infer_attention_score` is invoked exactly
  `layers × batches = 28 × 2 = 56` times per single-request two-token run.
* `store_kv` is invoked the same 56 times — one KV write per FIA call.
* `attn_backend.prepare_metadata` is invoked exactly `batches = 2` times.
* `FIAMetadata.get_last_indices` is not exercised on the Gate 1 path
  (single-request prefill uses `out_loc` directly).

**Scheduler / engine call contract per run:**

* `model.forward` × 2, `sampler.sample` × 2.
* `torch.distributed.init_process_group`, `all_reduce`, and
  `destroy_process_group` — each called **0 times**. Distributed is never
  brought up for TP=1 NPU.
* `Scheduler.shutdown()` returns cleanly (`synchronize_device("npu")`,
  no-op `sync_all_ranks`, `engine.shutdown()`).

---

## 6. Regression evidence (verification container)

Each test target run standalone at HEAD `0206c54` with the repo `python/`
on `PYTHONPATH` to avoid the pre-existing cross-file test-order
contamination (`test_device_runtime.py` installs stub package objects for
its own isolation; `test_ascend_fia_backend.py` prepends `python/` as a
side-effect other tests rely on).

| Test module | Result |
| --- | --- |
| `tests/misc/test_device_runtime.py` | **59 passed** |
| `tests/misc/test_engine_tp1_nodist.py` | **7 passed** |
| `tests/misc/test_sampler_no_cuda_nvtx.py` | **4 passed** |
| `tests/misc/test_scheduler_device_backend.py` | **5 passed** |
| `tests/misc/test_scheduler_sync_all_ranks.py` | **4 passed** |
| **Total** | **79 passed / 79** |

## 7. 910B1 hardware evidence

Minimum two-token smoke rerun at HEAD `0206c54`:

* `PROBE_RESULT=PASS`
* `iter_count=2`, wall clock about 1.15 s (excluding one-time weight load).
* `reply_log[0].next_token = 15087`, `reply_log[0].finished = False`
* `reply_log[1].next_token = 11`,   `reply_log[1].finished = True`
* `dist_initialized (after run) = False`
* `shutdown_ok = True`
* Counters:
  * `model_forward = 2`
  * `sampler_sample = 2`
  * `prepare_metadata = 2`
  * `store_kv = 56`
  * `fia = 56`
  * `init_process_group = 0`
  * `all_reduce = 0`
  * `destroy_process_group = 0`

The probe script itself is a private ops artifact and is intentionally
not committed to the repository. Its inputs, expected outputs, and the
observed counters are reproduced above verbatim so this document is
self-contained evidence.

---

## 8. Known gaps not covered by Gate 1

The following remain open and should be addressed by later gates.

* **TP > 1 execution on NPU.** HCCL init is exercised by
  `test_ascend_fia_backend`'s smoke and by `test_distributed_runtime`, but
  no forward or decode has been run across ranks.
* **Continuous batching.** Only a single request is exercised. Overlap
  scheduling (`Scheduler.overlap_loop`) is untested on NPU.
* **Graph capture on NPU.** `graph_runner.pad_batch` is called on the eager
  path only; no ACL-graph capture verified.
* **Non-greedy samplers.** `Sampler.sample` has been shown to work on
  greedy on NPU; top-k / top-p / temperature paths are not verified.
* **Long context and radix reuse.** `max_seq_len_override=1024` and a
  fresh request per run — no cross-request prefix cache hit tested.
* **Cross-file test isolation.** `test_device_runtime.py` and
  `test_sampler_no_cuda_nvtx.py` fail when run in the same pytest process
  as tests that legitimately import `torch_npu` or the real `minisgl`
  package. This is a test-harness ordering issue, not a runtime bug; each
  file passes standalone. Fixing it is a documentation / harness task,
  not a runtime task.
* **Detokenizer / API server path on NPU.** Only the offline scheduler is
  driven; the ZMQ tokenizer / detokenizer processes are untested here.

---

## 9. Reproduction outline

The evidence in this document can be reproduced from HEAD
`0206c54398c4c7ab567aa9b0ef11cd1683ed69b2` on tag
`gate1-single-request-eager` by, on an Ascend 910B1 host with CANN 8.5.1
and `torch_npu 2.9.0.post1`:

1. Check out the tag; ensure `python/` is on `PYTHONPATH`.
2. Load a Qwen3-0.6B checkpoint in `float16`.
3. Build a `SchedulerConfig` with the parameters listed in section 1
   (attention backend `npu_fia`, page size 16, TP=1, `offline_mode=True`,
   cuda-graph disabled).
4. Feed `input_ids=[3, 7, 11, 15]` through a `UserMsg` with
   `SamplingParams(temperature=0.0, max_tokens=2)`.
5. Drive `Scheduler.normal_loop()` inside `scheduler.engine_stream_ctx`
   until two `DetokenizeMsg` replies land.
6. Expect the invariants in sections 4 and 5 to hold exactly.

---

## Sign-off

All Gate 1 invariants verified at HEAD `0206c54`. Gate 1 is frozen at
this commit and this document. No merge into `ascend-port` is performed
by this gate — the tag `gate1-single-request-eager` is the load-bearing
reference for downstream gates.
