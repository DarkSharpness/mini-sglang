# OLMo3 phase 4: scheduler and API smoke

Phase 4 closes the smallest useful system-level validation surface before
publication. It intentionally combines related features into two short smoke
tests instead of creating a benchmark matrix.

## Scheduler integration

`tools/olmo3/system_smoke.py` loads the official model once with FlashInfer,
Radix Cache, a two-request limit, and a 64-token prefill budget. It submits two
identical 48-token requests together and then submits the same prefix once more.

Validated properties:

- Both initial requests produce the same first greedy token as the saved HF
  reference.
- The maximum prefill batch size is two, exercising batching.
- Three prefill forwards are required, exercising chunked prefill under the
  fixed token budget.
- The repeated request matches 47 cached prefix tokens.
- CacheManager integrity passes after all requests finish.

```text
OLMO3_SYSTEM_SMOKE=passed
prefill_calls=3
max_prefill_batch=2
radix_cached_tokens=47
```

## OpenAI-compatible API

`tools/olmo3/run_api_smoke.sh` starts a temporary `TP=2` server and always
terminates its complete process group. `tools/olmo3/api_smoke.py` validates:

- `GET /v1/models` returns the local OLMo3 model.
- Non-streaming `POST /v1/chat/completions` returns HTTP 200, `Paris`, and
  `finish_reason="stop"`.
- The streaming endpoint returns the assistant role, the same combined content,
  one final stop chunk, and one `[DONE]` marker.
- Neither configured EOS token is exposed in response text.

```text
OLMO3_API_SMOKE=passed content='Paris'
```

The API test uses the same standard-NCCL fallback as the validated TP2 path.
After shutdown both GPUs return to 1 MiB used memory and no Mini-SGLang or
Uvicorn worker remains.

## Publication fixes

The final review also made three compatibility corrections:

- Distributed Q/K RMSNorm now multiplies the local weight in FP32 and performs
  only one final cast, matching Transformers OLMo3.
- OLMo3 `TP>1` automatically disables PyNCCL because its all-reduce path does
  not support the required FP32 statistics.
- Multi-EOS loading falls back to `GenerationConfig.from_model_config` when a
  separate `generation_config.json` is unavailable, while retaining the legacy
  single `eos_token_id` attribute.

No throughput, latency, concurrency-scale, or long-context benchmark was run.
