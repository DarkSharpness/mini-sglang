# mini-sglang-virtual-pipelining

This README includes notes about mini-sglang and our fork on it.

## Setup
Activate environment.
Set up environment:
```python
uv pip install -e .
```

## Online Serving

Option A - online server:

Set up server:

```python
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
```python
python -m minisgl --model "Qwen/Qwen2.5-1.5B-Instruct" --shell --port 1920
```

## Online Benchmark

Start server (see above.)

Then run 
```python
python benchmark/benchmark.py
```

This reports stats such as TTFT, TPOT, Latency and Throughput with std. You can control the number of warmup and actual runs, as well as batch size and maximum number of tokens to generate pre-request.

## Offline benchmark

Run benchmark:

```python
python benchmark/offline/bench.py 
```
Sample output: Total: 133966tok, Time: 73.15s, Throughput: 1831.38tok/s

## Notes

- Tensor Parallelism: Scales inference across multiple GPUs.


Overlap Scheduling
- To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in NanoFlow. This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.
