# mini-sglang-virtual-pipelining

This README includes notes about mini-sglang and our fork on it.

## Setup


1. Create and activate environment.
```bash
uv venv --python=3.12
source .venv/bin/activate

2. Set up environment:
```python
uv pip install -e .
```


## Running Benchmark with Online Server

1. Set up a server:
```bash
python -m minisgl --model "Qwen/Qwen2.5-1.5B-Instruct" --port 1920 
```

2. Then run benchmark script:
```bash
python benchmark/benchmark.py
```

This outputs stats such as TTFT, TPOT, Latency and Throughput with mean and confidence intervals. 
The default setting acts as a simple test, with benchmark settings further controllable in the script arguments in `benchmark/benchmark.py`.
This includes `num_tokens` (number of tokens to generate), `concurrency` (number of requests to send, i.e., batch size), `num_warmups` and `num_runs`.

If benchmarking the virtual pipelining implementation, the `SKIP_PROB` constant in `python/minisgl/scheduler/decode.py` file can be changed to view behaviour under different skipping probabilities.

## Online Serving

Option A - online server:

Set up server:

```bash
python -m minisgl --model "Qwen/Qwen2.5-1.5B-Instruct" --port 1920 
```

Then send OpenAI compatible request (make sure port number is correct):

```
curl -X POST http://127.0.0.1:1920/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer EMPTY" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of Turkey?"}
    ],
    "max_tokens": 50,
    "temperature": 0.0,
    "stream": false
  }'
```

Option B - interactive chat mode (shell flag):
```bash
python -m minisgl --model "Qwen/Qwen2.5-1.5B-Instruct" --shell --port 1920
```

## Online Benchmark

Start server (see above.)

Then run 
```bash
python benchmark/benchmark.py
```

This reports stats such as TTFT, TPOT, Latency and Throughput with std. You can control the number of warmup and actual runs, as well as batch size and maximum number of tokens to generate pre-request.

Sample output:
BENCHMARK RESULTS (Runs=3, Concurrency=1, MaxTokens=256)
TTFT (Prefill) : 25.30 ms ± 2.39 ms
TPOT (Decode)  : 13.03 ms ± 0.01 ms
Latency (E2E)  : 3335.20 ms ± 3.69 ms
Throughput/Req : 76.46 tk/s ± 0.08 tk/s
Sys Throughput : 76.45 tokens/sec


## Offline benchmark

Run benchmark:

```bash
python benchmark/offline/bench.py 
```
Sample output: Total: 133966tok, Time: 73.15s, Throughput: 1831.38tok/s

## Notes

- Tensor Parallelism: Scales inference across multiple GPUs.


- Overlap Scheduling: To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in NanoFlow. This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.
