# MLX CPU Benchmarks

**Setup:** 
```bash
cd benchmark/sysperf/mlx
uv sync  # Installs mlx-lm (no CUDA dependencies)
```

**Platform:** Apple Silicon (M1/M2/M3) recommended

**Run:**
- `uv run benchmark.py` (from `benchmark/sysperf/mlx/` directory) - Run offline benchmark on CPU using MLX community models
- Or: `uv run --directory benchmark/sysperf/mlx benchmark.py` (from project root)

## Sample Output

```bash
======================================================================
MLX CPU Benchmark (using mlx-lm)
======================================================================
Loading model: mlx-community/Qwen2.5-0.5B-Instruct-4bit
special_tokens_map.json: 100%|██████████████████████| 613/613 [00:00<00:00, 13.0MB/s]
tokenizer_config.json: 7.31kB [00:00, 4.95MB/s]            | 0.00/613 [00:00<?, ?B/s]
model.safetensors.index.json: 44.2kB [00:00, 154MB/s]
merges.txt: 1.67MB [00:00, 23.4MB/s]
added_tokens.json: 100%|████████████████████████████| 605/605 [00:00<00:00, 13.8MB/s]
config.json: 100%|██████████████████████████████████| 783/783 [00:00<00:00, 16.0MB/s]
tokenizer.json: 7.03MB [00:00, 69.2MB/s]                   | 0.00/783 [00:00<?, ?B/s]
vocab.json: 2.78MB [00:00, 35.8MB/s]
model.safetensors: 100%|██████████████████████████| 278M/278M [00:06<00:00, 41.9MB/s]
Fetching 9 files: 100%|████████████████████████████████| 9/9 [00:07<00:00,  1.25it/s]
Model loaded successfully
Warming up...sors: 100%|██████████████████████████| 278M/278M [00:06<00:00, 48.2MB/s]
Running benchmark with 256 sequences...
  Progress: 0/256
  Progress: 50/256
  Progress: 100/256
  Progress: 150/256
  Progress: 200/256
  Progress: 250/256
======================================================================
Total: 140435tok, Time: 671.49s, Throughput: 209.14tok/s
======================================================================
```

**Note:** Uses MLX community models (mlx-lm) for CPU benchmarking on Apple Silicon. This is independent of the CUDA-based mini-sglang codebase.

