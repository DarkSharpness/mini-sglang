from __future__ import annotations

import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from minisgl.benchmark.client import RawResult


# A bank of distinct, non-repeating sentences about LLM serving internals. Shuffled
# per request to build a realistic document without the degenerate self-overlap of a
# single paragraph repeated many times. Comfortably exceeds 1024 tokens in one pass.
DOCUMENT_SENTENCES: tuple[str, ...] = (
    "Paged attention stores the key-value cache in fixed-size blocks instead of one contiguous buffer.",
    "A block table maps each logical position of a sequence to the physical block that holds its tokens.",
    "Because blocks are shared through reference counts, identical prompt prefixes can reuse the same pages.",
    "Prefill processes every prompt token in a single forward pass and writes their keys and values into the cache.",
    "Decoding then advances one token at a time, reading the whole cache to attend over the growing context.",
    "The decode phase is memory bound, since each step reloads the model weights just to emit a single token.",
    "Speculative decoding attacks that bottleneck by proposing several future tokens and verifying them together.",
    "A lightweight drafter guesses a short continuation, and the target model checks all of the guesses at once.",
    "When the guesses are correct, one expensive weight read yields many committed tokens instead of only one.",
    "Prompt lookup decoding skips the draft model entirely and copies candidate tokens from earlier text.",
    "It searches the prompt and the tokens generated so far for a matching n-gram suffix.",
    "If a match is found, the tokens that followed it become the speculative draft for this step.",
    "This shines when the output repeats spans of the input, as in editing, summarizing, or retrieval tasks.",
    "Verification runs the target model over the frontier token followed by every drafted token in one batch.",
    "The scheduler compares each draft against the model's own greedy choice and keeps the longest correct prefix.",
    "Rejected draft slots are returned to the allocator so their cache pages can be reused immediately.",
    "A bonus token is always emitted from the last verified position, guaranteeing forward progress every step.",
    "Continuous batching interleaves prefill and decode work so the accelerator rarely sits idle.",
    "Requests join and leave the running batch dynamically as they arrive and finish.",
    "CUDA graphs capture a fixed sequence of kernels once and replay them to remove per-step launch overhead.",
    "Graph replay only helps when tensor shapes stay constant, which constrains how much a batch may vary.",
    "Quantizing weights to eight or four bits shrinks the dominant memory read during decode.",
    "Grouped-query attention lets several query heads share a single key-value head to cut cache size.",
    "The arithmetic intensity of decode rises with batch size until bandwidth or capacity limits take over.",
    "On an H100 the compute-to-bandwidth ridge sits far above the batch sizes typical serving reaches.",
    "Longer contexts inflate key-value traffic, which steadily erodes the benefit of larger batches.",
    "Sampling at temperature zero reduces to greedy decoding, which speculation can match token for token.",
    "Nucleus and top-k sampling instead draw from a truncated probability distribution at each step.",
    "Detokenization must handle partial multi-byte characters so streamed text never splits a code point.",
    "An eviction policy decides which cached blocks to drop when memory pressure grows.",
    "Radix trees index shared prefixes so a common system prompt is stored only once.",
    "Chunked prefill splits a very long prompt into slices that fit alongside ongoing decode work.",
    "Tensor parallelism shards each weight matrix across devices and sums partial results with an all-reduce.",
    "Pipeline parallelism instead assigns whole layers to devices and streams activations between them.",
    "The overlap scheduler hides CPU bookkeeping behind the previous step's GPU computation.",
    "A well-tuned server keeps the GPU saturated while its queues absorb bursts of incoming traffic.",
    "Latency is often reported as time to first token and time per output token separately.",
    "Throughput counts committed output tokens per second across every concurrent request.",
    "Acceptance rate measures the fraction of drafted tokens that the target model actually keeps.",
    "A high acceptance rate turns each verification pass into several tokens of real progress.",
    "When acceptance is low, verification wastes compute on tokens that are ultimately thrown away.",
    "Attention kernels may run on tensor cores or ordinary units, and the two can differ in the last bit.",
    "Such tiny numerical gaps can flip a near-tied argmax and diverge two otherwise identical runs.",
    "Deterministic benchmarks therefore fix the kernel path on both sides of any comparison.",
    "Warmup requests prime caches and compile graphs before any timed measurement begins.",
    "Fixing the output length with an ignore-end-of-sequence flag makes the throughput math exact.",
    "A fresh server per benchmark cell keeps cumulative counters from leaking between measurements.",
    "Reproducible seeds ensure that prompts and any injected noise stay identical across arms.",
    "Memory capacity, not compute, usually caps how many sequences a single device can serve.",
    "Each additional token in the context adds a fixed number of bytes to every future decode read.",
    "The allocator hands out cache blocks lazily so short requests never reserve space they will not use.",
    "Preemption can pause a long request and swap its cache out when higher-priority work arrives.",
    "If any request in a batch drafts tokens, the whole batch may take the slower verification path.",
    "That coupling is why draft hit rate and batch composition jointly determine the realized speedup.",
    "Profiling before optimizing keeps effort from landing on paths that are already cheap.",
    "The cheapest token is the one you never compute because you copied it correctly.",
    "Good serving systems make the common case fast and keep the rare case merely correct.",
)


def _inject_typos(text: str, seed: int, *, rate: int = 40) -> str:
    """Deterministically corrupt ~1/rate words by transposing two interior chars.

    Keeps the document overwhelmingly intact so the corrected output still overlaps
    the prompt heavily, while giving the model a genuine (non-degenerate) edit to do.
    """
    rng = random.Random(seed)
    words = text.split(" ")
    for i, word in enumerate(words):
        if len(word) > 3 and rng.randint(0, rate) == 0:
            j = rng.randint(1, len(word) - 3)
            words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2 :]
    return " ".join(words)


def _shuffled_document(seed: int) -> str:
    """A varied, non-repeating document: the sentence bank in a seed-shuffled order."""
    sentences = list(DOCUMENT_SENTENCES)
    random.Random(seed).shuffle(sentences)
    return " ".join(sentences)


def friendly_prompt(tokenizer: Any, target_tokens: int, request_id: int) -> str:
    """Realistic high-overlap edit task: fix typos and return the full document.

    Prompt-lookup / speculative decoding is built for edit- and RAG-style rewrites,
    where the output re-emits almost the entire source verbatim. We seed a few
    spelling mistakes into a varied (non-repeating) document and ask for the corrected
    text back, so the drafter can copy long runs from the prompt without the task being
    a degenerate "echo the input" instruction.
    """
    instruction = (
        f"Request {request_id}: the document below has a few spelling mistakes. "
        "Return the full document with those mistakes corrected, changing nothing else.\n"
        "<document>\n"
    )
    body = _inject_typos(_shuffled_document(request_id), request_id)
    text = instruction + body + "\n</document>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    # Top up with further shuffles only if the caller asked for more tokens than one
    # pass of the bank provides (i.e. input_len beyond ~1k); still no verbatim repeats.
    seed = request_id + 1
    while len(ids) < target_tokens:
        more = _inject_typos(_shuffled_document(seed), seed)
        ids += tokenizer.encode(more, add_special_tokens=False)
        seed += 1
    return tokenizer.decode(ids[:target_tokens])


def output_prompt_overlap(prompt_ids: list[int], output_ids: list[int], *, n: int = 3) -> float:
    """Fraction of length-n output windows that also occur verbatim in the prompt.

    This is the prompt-lookup premise made measurable: high overlap means the
    drafter can copy long spans out of the prompt, so acceptance should be high.
    """
    if len(output_ids) < n or len(prompt_ids) < n:
        return 0.0
    prompt_windows = {tuple(prompt_ids[i : i + n]) for i in range(len(prompt_ids) - n + 1)}
    total = len(output_ids) - n + 1
    hits = sum(tuple(output_ids[i : i + n]) in prompt_windows for i in range(total))
    return hits / total


def mean_output_prompt_overlap(
    results: list[RawResult],
    tokenizer: Any,
    *,
    n: int = 3,
) -> float | None:
    """Mean prompt-copy overlap across results that captured their output text."""
    fractions: list[float] = []
    for result in results:
        if result.output_text is None:
            continue
        prompt_ids = tokenizer.encode(result.message, add_special_tokens=False)
        output_ids = tokenizer.encode(result.output_text, add_special_tokens=False)
        fractions.append(output_prompt_overlap(prompt_ids, output_ids, n=n))
    if not fractions:
        return None
    return statistics.mean(fractions)


def summarize(results: list[RawResult], output_len: int) -> dict[str, float]:
    """Compact machine keys for the BENCH_RESULT JSON line."""
    start = min(r.tics[0] for r in results)
    end = max(r.tics[-1] for r in results)
    duration = end - start
    request_rates = [output_len / (r.tics[-1] - r.tics[0]) for r in results]
    return {
        "duration_s": duration,
        "aggregate_output_tps": len(results) * output_len / duration,
        "mean_request_output_tps": statistics.mean(request_rates),
        "p50_request_output_tps": statistics.median(request_rates),
    }


def maybe_wandb_run(config: dict[str, Any], *, enabled: bool):
    """Start a wandb run when requested and credentials exist; otherwise None."""
    if not enabled:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB: --wandb set but WANDB_API_KEY missing; skipping", flush=True)
        return None
    try:
        import wandb
    except ImportError:
        print("WANDB: package not installed; skipping", flush=True)
        return None

    project = os.environ.get("WANDB_PROJECT", "mini-sglang-spec")
    entity = os.environ.get("WANDB_ENTITY") or None
    arm = str(config.get("arm") or ("spec-on" if config.get("spec") else "spec-off"))
    model_slug = str(config.get("model", "unknown")).replace("/", "--")
    name = f"{arm}-{config.get('workload')}-bs{config.get('batch_size')}-{model_slug}"
    tags = [
        arm,
        f"workload:{config.get('workload')}",
        f"bs:{config.get('batch_size')}",
        f"model:{model_slug}",
        "overlap-on" if config.get("overlap") else "overlap-off",
        "thinking-on" if config.get("enable_thinking") else "thinking-off",
    ]
    revision = str(config.get("revision") or "").strip()
    if revision and revision not in {"unknown", "working-tree-upload"}:
        tags.append(f"rev:{revision[:12]}")
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        tags=tags,
        config=config,
        job_type="benchmark_cell",
        reinit=True,
    )


def wandb_summary_payload(
    final: dict[str, Any],
    *,
    batch_wall_s: float,
    spec_metrics: dict[str, int | float] | None,
    output_prompt_overlap: float | None = None,
) -> dict[str, float]:
    """Summary metrics for bar-chart compare (one value per cell)."""
    payload: dict[str, float] = {
        "throughput (tok/s)": float(final["aggregate_output_tps"]),
        "avg request output (tok/s)": float(final["mean_request_output_tps"]),
        "latency (s)": float(batch_wall_s),
    }
    if output_prompt_overlap is not None:
        payload["prompt_overlap"] = float(output_prompt_overlap)
    if spec_metrics is not None:
        for key in ("drafted_tokens", "accepted_tokens", "acceptance_rate"):
            if key in spec_metrics:
                payload[key] = float(spec_metrics[key])
    return payload


def latest_spec_metrics(server_log: str) -> dict[str, int | float] | None:
    """Parse the newest SPEC_METRICS JSON line from a server log."""
    if not server_log:
        return None
    path = Path(server_log)
    if not path.is_file():
        return None

    latest: dict[str, int | float] | None = None
    marker = "SPEC_METRICS "
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        try:
            payload = json.loads(line.split(marker, 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest = {
                str(key): value
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }
    return latest


def poll_spec_metrics(
    server_log: str,
    *,
    timeout_s: float = 30.0,
) -> dict[str, int | float] | None:
    """Wait briefly for a drained SPEC_METRICS line (fresh server ⇒ cell-local)."""
    deadline = time.monotonic() + timeout_s
    latest: dict[str, int | float] | None = None
    while time.monotonic() < deadline:
        latest = latest_spec_metrics(server_log)
        if latest is not None and int(latest.get("proposal_requests", 0)) > 0:
            return latest
        time.sleep(0.05)
    return latest


def log_io_artifacts(wb: Any, results: list[RawResult]) -> None:
    """Attach this cell's captured prompt/response pairs to wandb for inspection.

    Logs two things: an inline ``benchmark/io`` wandb.Table (browsable in the run
    without downloading) and a downloadable ``benchmark-io`` artifact holding
    prompts.json + responses.json (index-aligned by request_id). No-op when nothing
    was captured (capture_output off). Failures are surfaced, not swallowed.
    """
    captured = [r for r in results if r.output_text is not None]
    if not captured:
        print("WANDB: no captured output_text; skipping io artifact/table", flush=True)
        return

    import wandb

    prompts = [
        {"request_id": i, "input_len": r.input_len, "prompt": r.message}
        for i, r in enumerate(captured)
    ]
    responses = [
        {"request_id": i, "output_len": r.output_len, "response": r.output_text}
        for i, r in enumerate(captured)
    ]

    try:
        # Inline table: the fastest way to eyeball per-request prompt vs response
        # (e.g. whether spec-on output is degraded or merely different).
        table = wandb.Table(columns=["request_id", "input_len", "output_len", "prompt", "response"])
        for p, resp in zip(prompts, responses):
            table.add_data(p["request_id"], p["input_len"], resp["output_len"], p["prompt"], resp["response"])
        wb.log({"benchmark/io": table})

        # Persistent temp dir (not a context manager): wandb uploads asynchronously, so
        # the files must outlive this call until wb.finish() drains them.
        tmp = Path(tempfile.mkdtemp(prefix="bench-io-"))
        for name, rows in (("prompts.json", prompts), ("responses.json", responses)):
            (tmp / name).write_text(
                json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        artifact = wandb.Artifact(f"io-{wb.id}", type="benchmark-io")
        artifact.add_file(str(tmp / "prompts.json"))
        artifact.add_file(str(tmp / "responses.json"))
        wb.log_artifact(artifact)
        print(f"WANDB: logged io table + artifact for {len(captured)} requests", flush=True)
    except Exception as exc:  # visibility beats a silently missing artifact
        print(f"WANDB: failed to log io artifact/table: {exc!r}", flush=True)


def log_server_artifact(wb: Any, server_log: str) -> None:
    """Attach the server log as a wandb artifact."""
    if not server_log:
        return
    path = Path(server_log)
    if not path.is_file():
        print(f"WANDB: server log not found: {path}", flush=True)
        return

    import wandb

    artifact = wandb.Artifact(f"server-log-{wb.id}", type="server-log")
    artifact.add_file(str(path))
    wb.log_artifact(artifact)
