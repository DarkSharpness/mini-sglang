# System Performance Benchmarks

Compare different serving systems: GPU (Modal) and CPU (MLX) benchmarks.

## Modal GPU Benchmark

**Setup:**
```bash
uv pip install modal
modal token new
```

**HF Token (optional):** Add to `tests/sysperf/modal/.env`: `HF_TOKEN=your_token_here`

**Run:**
```bash
modal run tests/sysperf/modal/modal_app.py
```

**Stats:**
- **GPU:** A10G
- **Model:** Qwen/Qwen3-0.6B
- **Total:** 133966tok
- **Time:** 44.10s
- **Throughput:** 3037.56tok/s

**Output:**
```bash
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2025-12-23|08:06:28|core|rank=0] INFO     Free memory before loading model: 21.79 GiB
[2025-12-23|08:06:56|core|rank=0] INFO     Allocating 169913 pages for KV cache, K + V = 18.15 GiB
[2025-12-23|08:06:56|core|rank=0] INFO     Auto-selected attention backend: fi
[2025-12-23|08:06:56|core|rank=0] INFO     Free memory after initialization: 2.05 GiB
[2025-12-23|08:06:56|core|rank=0] INFO     Start capturing CUDA graphs with sizes: [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1]
[2025-12-23|08:06:56|core|rank=0] INFO     Free GPU memory before capturing CUDA graphs: 1.98 GiB

Capturing graphs: bs = 256 | avail_mem = 1.84 GiB:   0%|          | 0/35 [00:00<?, ?b
Capturing graphs: bs = 248 | avail_mem = 1.84 GiB:   0%|          | 0/35 [00:00<?, ?b
Capturing graphs: bs = 240 | avail_mem = 1.84 GiB:   0%|          | 0/35 [00:00<?, ?batch/s]
Capturing graphs: bs = 240 | avail_mem = 1.84 GiB:   9%|▊         | 3/35 [00:0
Capturing graphs: bs = 232 | avail_mem = 1.83 GiB:   9%|▊         | 3/35 [00:00<00:01
Capturing graphs: bs = 224 | avail_mem = 1.83 GiB:   9%|▊         | 3/35 [00:00<00:01
Capturing graphs: bs = 216 | avail_mem = 1.82 GiB:   9%|▊         | 3/35 [00:00<00:01
Capturing graphs: bs = 208 | avail_mem = 1.82 GiB:   9%|▊         | 3/35 [00:00<00:01, 27.94batch/s]
Capturing graphs: bs = 208 | avail_mem = 1.82 GiB:  20%|██        | 7/
Capturing graphs: bs = 200 | avail_mem = 1.81 GiB:  20%|██        | 7/35 [00:00<00:00
Capturing graphs: bs = 192 | avail_mem = 1.81 GiB:  20%|██        | 7/35 [00:00<00:00
Capturing graphs: bs = 184 | avail_mem = 1.80 GiB:  20%|██        | 7/35 [00:00<00:00
Capturing graphs: bs = 176 | avail_mem = 1.80 GiB:  20%|██        | 7/35 [00:00<00:00, 30.15batch/s]
Capturing graphs: bs = 176 | avail_mem = 1.80 GiB:  31%|███▏      | 11
Capturing graphs: bs = 168 | avail_mem = 1.79 GiB:  31%|███▏      | 11/35 [00:00<00:0
Capturing graphs: bs = 160 | avail_mem = 1.79 GiB:  31%|███▏      | 11/35 [00:00<00:0
Capturing graphs: bs = 152 | avail_mem = 1.78 GiB:  31%|███▏      | 11/35 [00:00<00:0
Capturing graphs: bs = 144 | avail_mem = 1.78 GiB:  31%|███▏      | 11/35 [00:00<00:00, 30.98batch/s]
Capturing graphs: bs = 144 | avail_mem = 1.78 GiB:  43%|████▎     | 1
Capturing graphs: bs = 136 | avail_mem = 1.78 GiB:  43%|████▎     | 15/35 [00:00<00:0
Capturing graphs: bs = 128 | avail_mem = 1.77 GiB:  43%|████▎     | 15/35 [00:00<00:0
Capturing graphs: bs = 120 | avail_mem = 1.77 GiB:  43%|████▎     | 15/35 [00:00<00:0
Capturing graphs: bs = 112 | avail_mem = 1.76 GiB:  43%|████▎     | 15/35 [00:00<00:00, 29.60batch/s]
Capturing graphs: bs = 112 | avail_mem = 1.76 GiB:  54%|█████▍    | 1
Capturing graphs: bs = 104 | avail_mem = 1.76 GiB:  54%|█████▍    | 19/35 [00:00<00:0
Capturing graphs: bs = 96  | avail_mem = 1.75 GiB:  54%|█████▍    | 19/35 [00:00<00:0
Capturing graphs: bs = 88  | avail_mem = 1.75 GiB:  54%|█████▍    | 19/35 [00:00<00:0
Capturing graphs: bs = 80  | avail_mem = 1.74 GiB:  54%|█████▍    | 19/35 [00:00<00:00, 30.28batch/s]
Capturing graphs: bs = 80  | avail_mem = 1.74 GiB:  66%|██████▌   | 2
Capturing graphs: bs = 72  | avail_mem = 1.74 GiB:  66%|██████▌   | 23/35 [00:00<00:0
Capturing graphs: bs = 64  | avail_mem = 1.73 GiB:  66%|██████▌   | 23/35 [00:00<00:0
Capturing graphs: bs = 56  | avail_mem = 1.73 GiB:  66%|██████▌   | 23/35 [00:00<00:0
Capturing graphs: bs = 48  | avail_mem = 1.72 GiB:  66%|██████▌   | 23/35 [00:00<00:00, 31.69batch/s]
Capturing graphs: bs = 48  | avail_mem = 1.72 GiB:  77%|███████▋  | 2
Capturing graphs: bs = 40  | avail_mem = 1.72 GiB:  77%|███████▋  | 27/35 [00:00<00:0
Capturing graphs: bs = 32  | avail_mem = 1.71 GiB:  77%|███████▋  | 27/35 [00:00<00:0
Capturing graphs: bs = 24  | avail_mem = 1.71 GiB:  77%|███████▋  | 27/35 [00:00<00:0
Capturing graphs: bs = 16  | avail_mem = 1.70 GiB:  77%|███████▋  | 27/35 [00:00<00:00, 32.99batch/s]
Capturing graphs: bs = 16  | avail_mem = 1.70 GiB:  89%|████████▊ | 3
Capturing graphs: bs = 8   | avail_mem = 1.70 GiB:  89%|████████▊ | 31/35 [00:00<00:0
Capturing graphs: bs = 4   | avail_mem = 1.69 GiB:  89%|████████▊ | 31/35 [00:01<00:0
Capturing graphs: bs = 2   | avail_mem = 1.69 GiB:  89%|████████▊ | 31/35 [00:01<00:0
Capturing graphs: bs = 1   | avail_mem = 1.68 GiB:  89%|████████▊ | 31/35 [00:01<00:00, 31.92batch/s]
Capturing graphs: bs = 1   | avail_mem = 1.68 GiB: 100%|██████████| 35/35 [00:01<00:00, 28.09batch/s]
Capturing graphs: bs = 1   | avail_mem = 1.68 GiB: 100%|██████████| 35/35 [00:01<00:00, 29.90batch/s]

[2025-12-23|08:09:28|core|rank=0] INFO     Free GPU memory after capturing CUDA graphs: 1.68 GiB
Total: 133966tok, Time: 44.10s, Throughput: 3037.56tok/s
```

## MLX CPU Benchmark

**Setup:**
```bash
cd tests/sysperf/mlx
uv sync
```

**Platform:** Apple Silicon (M1/M2/M3) recommended

**Run:**
```bash
uv run benchmark.py
```

**Stats:**
- **CPU:** Apple Silicon M1
- **Model:** mlx-community/Qwen3-0.6B-4bit
- **Total:** 140435tok
- **Time:** 1017.77s
- **Throughput:** 137.98tok/s

**Output:**
```bash
======================================================================
MLX CPU Benchmark (using mlx-lm)
======================================================================
Loading model: mlx-community/Qwen3-0.6B-4bit
config.json: 100%|██████████████████████████████████| 937/937 [00:00<00:00, 14.2MB/s]
added_tokens.json: 100%|████████████████████████████| 707/707 [00:00<00:00, 14.2MB/s]
model.safetensors.index.json: 49.7kB [00:00, 139MB/s]      | 0.00/707 [00:00<?, ?B/s]
tokenizer_config.json: 9.71kB [00:00, 45.1MB/s]        | 1/9 [00:00<00:02,  3.50it/s]
special_tokens_map.json: 100%|██████████████████████| 613/613 [00:00<00:00, 15.4MB/s]
merges.txt: 1.67MB [00:00, 26.0MB/s]                       | 0.00/613 [00:00<?, ?B/s]
vocab.json: 2.78MB [00:00, 36.9MB/s]                   | 3/9 [00:00<00:00,  7.73it/s]
tokenizer.json: 100%|███████████████████████████| 11.4M/11.4M [00:00<00:00, 11.8MB/s]
model.safetensors: 100%|██████████████████████████| 335M/335M [00:04<00:00, 76.8MB/s]
Fetching 9 files: 100%|████████████████████████████████| 9/9 [00:04<00:00,  1.85it/s]
Model loaded successfully███████████████████▊     | 268M/335M [00:04<00:00, 82.6MB/s]
Warming up...
Running benchmark with 256 sequences...
  Progress: 0/256
  Progress: 50/256
  Progress: 100/256
  Progress: 150/256
  Progress: 200/256
  Progress: 250/256
======================================================================
Total: 140435tok, Time: 1017.77s, Throughput: 137.98tok/s
======================================================================
```

**Note:** Both benchmarks use Qwen3-0.6B model for consistent comparison (Modal uses standard HF format, MLX uses quantized 4-bit format).
