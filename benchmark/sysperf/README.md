# System Performance Benchmarks

This directory contains different benchmarking methods for comparing performance across different platforms and frameworks.

## Available Benchmarks

- **`modal/`** - GPU benchmarks using Modal.com (NVIDIA GPUs)
- **`mlx/`** - CPU benchmarks using MLX (Apple Silicon)

## Structure

```
sysperf/
├── modal/          # Modal GPU benchmarks
│   ├── modal_app.py
│   ├── profile_prepare_replay.py
│   └── README.md
├── mlx/            # MLX CPU benchmarks
│   ├── benchmark.py
│   └── README.md
└── README.md       # This file
```

## Usage

Each subdirectory contains its own README with setup and run instructions.

