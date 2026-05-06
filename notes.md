


# Setup
Activate environment.
Set up environment:
```
uv pip install -e .
```


# Online Serving


Option A - interactive chat mode (shell flag):
```
python -m minisgl --model "Qwen/Qwen2.5-1.5B-Instruct" --shell --port 1920
```

Option B - online server:

Set up server:

``` python -m minisgl --model "Qwen/Qwen2.5-1.5B-Instruct" --port 1920 ```

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

# Offline benchmark

Run benchmark:

```
python benchmark/offline/bench.py 
```
Sample output: Total: 133966tok, Time: 73.15s, Throughput: 1831.38tok/s



# Docs Notes

- Tensor Parallelism: Scales inference across multiple GPUs.


Overlap Scheduling
- To further reduce CPU overhead, Mini-SGLang employs overlap scheduling, a technique proposed in NanoFlow. This approach overlaps the CPU scheduling overhead with GPU computation, improving overall system throughput.
