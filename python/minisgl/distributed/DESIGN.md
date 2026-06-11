# Distributed

The `distributed/` component handles tensor parallelism (TP) communication across GPU ranks. It provides a small abstraction layer that lets model layers call `all_reduce` / `all_gather` without knowing which underlying transport is used.

---

## Component Map

```
DistributedCommunicator          ← singleton used by model layers
    └── plugins: List[DistributedImpl]   ← stack; last one wins

DistributedImpl (abstract)
    ├── TorchDistributedImpl     ← default; uses torch.distributed (NCCL or gloo)
    └── PyNCCLDistributedImpl    ← custom PyNCCL via CUDA IPC, faster for TP
            └── PyNCCLCommunicator  ← C++ NCCL wrapper (kernel/)
```

---

## Transport Selection

```
Engine.__init__
    │
    ├── tp_size == 1 OR use_pynccl flag?
    │       ├── YES → init_process_group(backend="gloo")
    │       │         enable_pynccl_distributed(...)
    │       │         └── DistributedCommunicator.plugins.append(PyNCCLDistributedImpl)
    │       │
    │       └── NO  → init_process_group(backend="nccl")
    │                 (uses torch.distributed NCCL directly)
    │
    └── tp_cpu_group = gloo group  ← always used for CPU-side coordination
                                     (free memory all-reduce, sync barriers)
```

When PyNCCL is active it **replaces** torch distributed for GPU-to-GPU transfers by being pushed onto the `plugins` stack. `all_reduce` / `all_gather` always call `plugins[-1]`.

---

## Communication Primitives

```
DistributedCommunicator.all_reduce(x)
    └── plugins[-1].all_reduce(x)
            ├── TorchDistributedImpl  → dist.all_reduce(x, SUM)
            └── PyNCCLDistributedImpl → comm.all_reduce(x, "sum")

DistributedCommunicator.all_gather(x)
    └── plugins[-1].all_gather(x)
            ├── TorchDistributedImpl  → dist.all_gather_into_tensor(out, x)
            └── PyNCCLDistributedImpl → comm.all_gather(result, x)
            output shape: [world_size * x.shape[0], ...]
```

TP rank = 1: both implementations short-circuit and return `x` unchanged.

---

## DistributedInfo

```python
@dataclass
class DistributedInfo:
    rank: int
    size: int

    def is_primary(self) -> bool:
        return self.rank == 0
```

Used throughout the system to shard weights and KV heads across ranks. Accessed via `get_tp_info()` (process-local singleton set at engine init).

---

## How Model Layers Use This

```
LinearOProj.forward(x):          ← output projection (row-parallel)
    y = F.linear(x, self.weight)
    if tp_size > 1:
        y = DistributedCommunicator().all_reduce(y)
    return y

VocabParallelEmbedding.forward:  ← uses all_reduce after gather
LinearRowParallel.forward:       ← same pattern as OProj
```

Column-parallel layers (QKV, gate/up projections) shard output dim; each rank computes a slice and no communication is needed until the row-parallel all-reduce.

---

## Lifecycle

```
startup:  enable_pynccl_distributed(...)   ← append PyNCCL to plugins
shutdown: destroy_distributed()            ← clear plugins list
          torch.distributed.destroy_process_group()
```

`destroy_distributed` must be called **before** freeing NCCL resources to prevent hangs (ordering enforced in `Engine.shutdown` → `GraphRunner.destroy_cuda_graphs` first).

---

## Key Files

| File | Responsibility |
|------|---------------|
| `impl.py` | `DistributedImpl`, `TorchDistributedImpl`, `PyNCCLDistributedImpl`, `DistributedCommunicator` |
| `info.py` | `DistributedInfo`, `get_tp_info`, `set_tp_info` |
