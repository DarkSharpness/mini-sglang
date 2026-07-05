# mini-sglang-ascend 源码依赖静态审计报告

> 分支：`gate0-ascend-bootstrap`  ·  审计模式：**纯静态、源码级、不安装、不运行、不改动生产代码**
> 平台：macOS 24.3.0（无 CUDA、无 NPU；仅做源码扫描与结构分析）
> 生成物仅新增两个文件：
> - `docs/ascend_port/source_dependency_audit.md`（本文件）
> - `docs/ascend_port/source_dependency_inventory.csv`
> 不修改 `pyproject.toml`、`uv.lock`、任何 `python/` 生产代码，不做 `cuda→npu` 盲替，不 `pip install`，不 `git commit`。
> 审计工具：`rg`, `grep`, `find`, Python AST（离线阅读），无副作用。

## 1. 执行摘要 (Executive Summary)

- 项目主体是一个基于 PyTorch 的 LLM 推理引擎（Mini-SGLang 分支），控制平面/调度/模型/张量/KV 缓存代码基本 **device-agnostic**，可直接复用；对 CUDA/NCCL 的耦合集中在四条通道：
  1. **设备原语**（`torch.cuda.*` Stream/Event/CUDAGraph/内存查询/nvtx）——由 `python/minisgl/engine/engine.py:31-209`、`python/minisgl/engine/graph.py:75-140`、`python/minisgl/scheduler/scheduler.py:53-134`、`python/minisgl/utils/arch.py:12-14`、`python/minisgl/utils/torch_utils.py:24` 统治。
  2. **算子后端**（`flashinfer`、`sgl_kernel`、Triton MoE）——集中在 `python/minisgl/layers/{norm,rotary,activation}.py`、`python/minisgl/attention/{fi,fa,trtllm}.py`、`python/minisgl/moe/fused.py`、`python/minisgl/engine/sample.py:30`。
  3. **自研 PyNCCL C++/CUDA 封装**——`python/minisgl/kernel/pynccl.py`、`python/minisgl/kernel/csrc/include/minisgl/nccl227.h`、`python/minisgl/kernel/csrc/src/pynccl.cu`，直接调用 NCCL 2.27 API。
  4. **JIT 加载的 CUDA 小算子**——`python/minisgl/kernel/{index,store}.py` 借助 `apache-tvm-ffi` 编译 `.cu`。
- **可直接复用比例**（按 Python 源文件数量粗估）：约 **70–75%** 的 `.py` 文件属于 `portable` 或 `thin-adaptation`；其中 `portable` 约 55%，`thin-adaptation` 约 20%；余下的 `backend-replacement`（≈20%）与 `major-rewrite`（<5%）集中在算子/通信实现层。
- **重写必需模块**：`kernel/csrc/src/pynccl.cu` + `kernel/csrc/include/minisgl/nccl227.h` + `kernel/pynccl.py`（NCCL → HCCL），以及所有 `flashinfer.*`、`sgl_kernel.*`、Triton MoE、`kernel/*.cu` JIT 算子的 Ascend 替代实现——但**这些均不在 Gate 0 范围内**。
- **Gate 0**（最小可运行 Ascend 引导）只需插入设备抽象层、令 `torch.distributed` 后端字符串在 Ascend 上走 `hccl`、把 PyNCCL/图捕获/`flashinfer`/`sgl_kernel`/Triton 相关导入改为**可选（延迟或有条件）**、把 `pyproject.toml` 里 CUDA 类依赖挪进可选 extra——**不引入任何 NPU 算子实现**。
- License：`pyproject.toml:13` 声明 `MIT`；`LICENSE` 为 sgl-project 2026 MIT；仓库无 `NOTICE`。作为 fork 建议补充 `NOTICE`，但不属于代码合规性阻塞项，且**本次审计不修改**。
- 备注：环境变量声称工作目录为 `/Users/dev/workspace-669590f9`，实际仓库位于 `/Users/lirui/PycharmProjects/mini-sglang-ascend`；已按实际代码结构完成审计。

## 2. 仓库结构 (Repository Structure)

顶层布局（见 `ls` 输出）：
```
Dockerfile
LICENSE                             # MIT, sgl-project 2026
NOTICE                              # 不存在
README.md
assets/
benchmark/
docs/
  features.md
  structures.md
  ascend_port/                      # 本次新增目录
pyproject.toml                      # setuptools+wheel, package-dir=python
python/minisgl/                     # 主包
tests/
```

`python/minisgl/` 关键子包一览：
- `core.py`——不可变数据类 `Batch`/`Req`/`Context`/`SamplingParams`（100% portable）。
- `env.py`——环境变量集中定义（含 `PYNCCL_MAX_BUFFER_SIZE`，`python/minisgl/env.py`）。
- `engine/`——运行时控制平面：`engine.py`、`graph.py`、`sample.py`、`config.py`。
- `scheduler/`——两流重叠调度：`scheduler.py`、`cache.py`、`prefill.py`、`io.py`、`config.py`。
- `distributed/`——`impl.py`（Torch/PyNCCL 双实现）、`info.py`（rank/world 元数据）。
- `attention/`——`base.py`、`utils.py`、`fi.py`、`fa.py`、`trtllm.py`、`__init__.py`（Registry）。
- `kvcache/`——`base.py`、`naive_cache.py`、`radix_cache.py`、`mha_pool.py`、`__init__.py`（Registry）。
- `layers/`——`attention.py`、`linear.py`、`norm.py`、`rotary.py`、`activation.py`、`embedding.py`、`moe.py`、`base.py`。
- `models/`——`base.py`、`weight.py`、`llama.py`、`qwen3_moe.py`。
- `moe/`——`fused.py`（`sgl_kernel` + Triton 驱动）。
- `kernel/`——tvm-ffi 驱动的 JIT/AOT 层：`__init__.py`（**贪婪导入**）、`pynccl.py`、`index.py`、`store.py`、`tensor.py`、`radix.py`、`moe_impl.py`、`utils.py`、`triton/fused_moe.py`、`csrc/`。
- `server/`——`args.py`、`launch.py`、`http.py`。
- `utils/`——`arch.py`、`torch_utils.py`、`mp.py`、`hf.py`、`logger.py`、`registry.py`、`misc.py`。
- `benchmark/perf.py`——开发工具，基于 `CUDAGraph`。

## 3. 依赖清单 (Dependency Inventory)

来源：`pyproject.toml:22-34`。

| 依赖 | 用途 | Python 源引用 | 可移植性 | 建议动作 |
| :-- | :-- | :-- | :-- | :-- |
| `accelerate` (`pyproject.toml:22`) | 未使用 | 无 (`rg` 无 `import accelerate`) | portable（可删） | Ascend 依赖列表中删除；不动主 `pyproject.toml` |
| `msgpack` | ZMQ 编码 | `python/minisgl/utils/mp.py:5` | portable | 保留 |
| `modelscope` | 权重下载备选 | `python/minisgl/utils/hf.py` | portable | 保留 |
| `torch<2.10.0` (`pyproject.toml:25`) | 全体核心 | 全仓库 | thin-adaptation | Ascend 需与 `torch_npu` 匹配版本；解耦 CUDA build |
| `transformers>=4.56.0,<=4.57.3` | tokenizer/config | `python/minisgl/utils/hf.py` | portable | 保留 |
| `flashinfer-python>=0.5.3` (`pyproject.toml:27`) | 注意力/RMSNorm/RoPE/Activation/采样 | `python/minisgl/layers/{norm,rotary,activation}.py`, `attention/{fi,trtllm}.py`, `engine/sample.py:30` | backend-replacement | 移入 `[cuda]` extra；Ascend 提供替代 |
| `pyzmq`, `uvicorn`, `fastapi`, `prompt_toolkit`, `openai` | 服务层 | `python/minisgl/{server,utils/mp.py}` | portable | 保留 |
| `apache-tvm-ffi>=0.1.4` (`pyproject.toml:32`) | JIT/AOT 加载器 | `python/minisgl/kernel/utils.py:63,99`；`python/minisgl/kernel/{pynccl,index,store,tensor,radix}.py` | thin-adaptation | CPU AOT 部分可保留；`cuda_files` 分支需要 device-gated |
| `sgl_kernel>=0.3.17.post1` (`pyproject.toml:33`) | FlashAttention KVCache/MoE 辅助 | `python/minisgl/attention/fa.py:158`, `python/minisgl/moe/fused.py:16,71` | backend-replacement | 移入 `[cuda]` extra |
| `quack-kernels` (`pyproject.toml:34`) | 未使用 | 无 | portable（可删） | 从 Ascend 依赖列表移除 |
| `triton`（隐式，随 torch） | Fused MoE | `python/minisgl/kernel/moe_impl.py:20-21,66`, `python/minisgl/kernel/triton/fused_moe.py:1-2,5,50` | backend-replacement | 在 Ascend 上禁用；MoE 路径 Gate 0 关闭 |
| dev extras (`pyproject.toml:41-52`) | pytest/ruff/mypy 等 | 开发工具 | portable | 保留 |

**关键结论：`accelerate` 与 `quack-kernels` 是完全无源码引用的“死声明”，是当前 CUDA 依赖清单里最容易的清理项。**

## 4. CUDA 专用 API 清单 (CUDA-Specific API Inventory)

以下均通过 `rg 'torch\.cuda\.'` 及派生正则获得，仅列直接调用点：

| 位置 | 调用 | 语义 | Ascend 替代 |
| :-- | :-- | :-- | :-- |
| `python/minisgl/utils/arch.py:12` | `torch.cuda.is_available()` | 判断 CUDA 是否可用 | 设备抽象；NPU 返回 `False` for `is_arch_supported` |
| `python/minisgl/utils/arch.py:14` | `torch.cuda.get_device_capability()` | 架构探测 | 抽象层返回 `None` on NPU；上层 `is_sm90/sm100` 保持 `False` |
| `python/minisgl/utils/torch_utils.py:24` | `import torch.cuda.nvtx as nvtx` | 装饰器打点 | 抽象为 no-op 或 `torch_npu` 打点 |
| `python/minisgl/engine/engine.py:31` | `torch.cuda.is_initialized()` | 启动守卫 | 抽象层 |
| `python/minisgl/engine/engine.py:35` | `f"cuda:{rank}"` | 设备字符串 | 抽象层 → `npu:{rank}` |
| `python/minisgl/engine/engine.py:36` | `torch.cuda.set_device(...)` | 绑定设备 | `torch_npu.npu.set_device(...)` |
| `python/minisgl/engine/engine.py:38-39` | `torch.cuda.Stream(...)` + `set_stream(...)` | 工作流 | `torch_npu.npu.Stream/set_stream` |
| `python/minisgl/engine/engine.py:172-174` | `synchronize/empty_cache/reset_peak_memory_stats` | 显存管理 | `torch_npu.npu.*` |
| `python/minisgl/engine/engine.py:192` | `torch.cuda.current_stream()` | 获取当前流 | `torch_npu.npu.current_stream()` |
| `python/minisgl/engine/engine.py:204` | `torch.cuda.Event(...)` | 事件同步 | `torch_npu.npu.Event(...)`（NPU 支持） |
| `python/minisgl/engine/graph.py:75` | `torch.cuda.mem_get_info(...)` | 剩余显存 | `torch_npu.npu.mem_get_info` |
| `python/minisgl/engine/graph.py:112-114` | `synchronize/empty_cache/reset_peak_memory_stats` | 显存管理 | 同上 |
| `python/minisgl/engine/graph.py:133` | `torch.cuda.CUDAGraph()` | 图捕获 | Gate 0 直接屏蔽；Gate 1 才评估 `aclgraph` |
| `python/minisgl/engine/graph.py:140` | `torch.cuda.graph(pool=...)` | 图捕获上下文 | 同上 |
| `python/minisgl/engine/sample.py:72` | `torch.cuda.nvtx.range("Sampler")` | 打点 | 抽象层 |
| `python/minisgl/scheduler/scheduler.py:53-55` | `Stream / stream / set_stream` | 双流重叠 | 抽象层 |
| `python/minisgl/scheduler/scheduler.py:128` | `torch.cuda.current_stream()` | 双流控制 | 抽象层 |
| `python/minisgl/scheduler/scheduler.py:134` | `torch.cuda.synchronize()` | 同步 | 抽象层 |
| `python/minisgl/attention/fi.py:120` | `torch.cuda.Event(...)` | FI 后端同步 | 后端选择时才触发；Gate 0 不启用 FI |

`pin_memory` 出现在 `python/minisgl/engine/sample.py:21`、`python/minisgl/attention/fi.py:114,173,198`、`python/minisgl/attention/fa.py:76`、`python/minisgl/attention/trtllm.py:100`、`python/minisgl/scheduler/scheduler.py:238,253,264,266`、`python/minisgl/scheduler/cache.py:134-135`、`python/minisgl/scheduler/prefill.py:60,81`——`torch_npu` 通常兼容 `pin_memory=True`，需要实机验证；不合规时回退为普通张量即可，属 P2。

**验证结果：`rg 'torch\.compile'` 无命中，`rg 'jit\.script'` 无命中——不需要图编译适配。**

## 5. 算子依赖清单 (Kernel Dependency Inventory)

| 类别 | 位置 | 依赖库/文件 | Ascend 替代 |
| :-- | :-- | :-- | :-- |
| Attention prefill/decode | `python/minisgl/attention/fi.py`、`python/minisgl/attention/trtllm.py:52-53` | `flashinfer.BatchPrefillWithPagedKVCacheWrapper`、`CUDAGraphBatchDecodeWithPagedKVCacheWrapper`、`trtllm_batch_decode_with_kv_cache`、`trtllm_batch_context_with_kv_cache` | Gate 1：`torch_npu.npu_incre_flash_attention` / `torch_npu.npu_prompt_flash_attention`；Gate 0 不注册 |
| Attention（PagedKV） | `python/minisgl/attention/fa.py:158` | `sgl_kernel.flash_attn.flash_attn_with_kvcache` | Gate 1 编写 Ascend 后端类；Gate 0 用 Registry 空注册 |
| RoPE | `python/minisgl/layers/rotary.py:35` | `flashinfer.apply_rope_with_cos_sin_cache_inplace` | 纯 torch 版或 `torch_npu.npu_rotary_mul` |
| RMSNorm | `python/minisgl/layers/norm.py:10,25` | `flashinfer.rmsnorm`、`flashinfer.fused_add_rmsnorm` | `torch_npu.npu_rms_norm` / 纯 torch |
| 激活 | `python/minisgl/layers/activation.py:10,16` | `flashinfer.silu_and_mul`、`flashinfer.gelu_and_mul` | `torch_npu.npu_swiglu` / 纯 torch |
| Embedding gather | `python/minisgl/layers/embedding.py` | `minisgl.kernel.indexing`（`kernel/index.py` JIT `.cu`） | 纯 torch `index_select`/scatter 回退 |
| KV cache 写入 | `python/minisgl/kvcache/mha_pool.py` | `minisgl.kernel.store_cache`（`kernel/store.py` JIT `.cu`） | 纯 torch scatter 回退 |
| 采样 | `python/minisgl/engine/sample.py:30` | `flashinfer.sampling.top_k_top_p_sampling_from_probs` | 纯 torch top-k/top-p 采样 |
| MoE gating | `python/minisgl/moe/fused.py:16,71` | `sgl_kernel.topk_softmax`、`sgl_kernel.moe_align_block_size` | 关闭 MoE 路径（Gate 0 外） |
| MoE Fused GEMM | `python/minisgl/kernel/moe_impl.py:20-21,66`、`python/minisgl/kernel/triton/fused_moe.py:1-2,5,50` | Triton `@triton.jit` | 同上 |
| PyNCCL 集合通信 | `python/minisgl/kernel/pynccl.py:30-37` | `apache-tvm-ffi` + `pynccl.cu` + NCCL | Gate 1 编写 HCCL 版；Gate 0 通过 `--disable-pynccl` 关闭 |
| CPU 辅助 | `python/minisgl/kernel/tensor.py`、`radix.py` | AOT `.cpp` | portable，直接保留 |

**结论：Ascend 后端算子替代面广但边界清晰；只要 Gate 0 把这类模块延迟导入并保持 Registry 允许“未注册”的空状态，就能满足最小引导目标。**

## 6. 分布式 / HCCL 迁移分析 (Distributed / HCCL Migration Analysis)

- `python/minisgl/engine/engine.py:114` 使用 `torch.distributed.init_process_group(backend="gloo", ...)` 建立**控制面 CPU 组**——完全设备无关，可原样保留。
- `python/minisgl/engine/engine.py:128` 使用 `init_process_group(backend="nccl", ...)` 建立**设备通信组**——需要在 Ascend 上切换为 `"hccl"`。
- `python/minisgl/engine/engine.py:135` 使用 `new_group(backend="gloo", ...)` 建立控制子组，可保留。
- `python/minisgl/distributed/impl.py` 定义两条实现：
  - `TorchDistributedImpl`：完全基于 `torch.distributed.all_reduce/all_gather` 的接口，**backend-agnostic**，只要 PyTorch 侧注册了 `hccl` 后端即可复用。
  - `PyNCCLDistributedImpl`：直接调用 `minisgl.kernel.pynccl`，属自研 NCCL 封装，与 HCCL 完全不兼容。
- `python/minisgl/distributed/impl.py:73-90` `enable_pynccl_distributed` 工厂根据 `use_pynccl` 选择实现——Gate 0 强制走 Torch 分支；Gate 1 才实现 HCCL 版 `PyHCCLDistributedImpl`。
- `python/minisgl/kernel/pynccl.py:30` 通过 `apache-tvm-ffi` 的 `load_aot` 编译 `csrc/src/pynccl.cu`，并对 `-lnccl` 硬依赖；`kernel/pynccl.py:37` 注册 `tvm_ffi` 对象 `minisgl.NCCLWrapper`。
- `python/minisgl/kernel/csrc/include/minisgl/nccl227.h` 与 `python/minisgl/kernel/csrc/src/pynccl.cu` 直接使用 NCCL 2.27 API（`ncclCommInitRank`、`ncclAllReduce`、`ncclAllGather`、`ncclCommWindowRegister`），需要**整体重写**为 HCCL 版本；属 P0 但**不在 Gate 0 内**。
- `python/minisgl/env.py:PYNCCL_MAX_BUFFER_SIZE`、`python/minisgl/server/args.py:124-129` 的 `--disable-pynccl` 是天然的降级开关；Gate 0 直接令其默认 `True` on Ascend。

**Gate 0 迁移策略：完全绕开 PyNCCL，只保留 `torch.distributed` 的字符串抽象；将 `backend="nccl"` 通过一个新的 `select_device_backend()` 函数在 NPU 上切换为 `"hccl"`。**

## 7. 注意力 / KV 后端分析 (Attention & KV Backend Analysis)

- 注意力后端通过 `python/minisgl/attention/__init__.py` 的 Registry 分发（key: `"trtllm"`, `"fi"`, `"fa"`）。
- `python/minisgl/attention/fi.py`：FlashInfer 全家桶，含 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`；也调用 `torch.cuda.Event`（120 行）。属 backend-replacement，但 Gate 0 不需要注册 Ascend 版本，只需容许 Registry 找不到时给出人类可读错误。
- `python/minisgl/attention/fa.py:158`：`sgl_kernel.flash_attn.flash_attn_with_kvcache`。Gate 1 目标。
- `python/minisgl/attention/trtllm.py:52-53`：TensorRT-LLM 融合调用。Gate 1 之后再考虑替代。
- `python/minisgl/attention/base.py`、`utils.py` 完全 device-agnostic（数据形状/元信息计算），portable。
- KV 缓存：
  - `python/minisgl/kvcache/base.py`、`naive_cache.py`、`radix_cache.py` 是纯 Python + torch 数据结构，`portable`。
  - `python/minisgl/kvcache/mha_pool.py` 依赖 `minisgl.kernel.store_cache`（`.cu`）——Gate 1 替换为 torch scatter 回退。
- 采样：`python/minisgl/engine/sample.py:30` 引用 `flashinfer.sampling`；在 Ascend 上极易用纯 torch 实现 top-k / top-p 替换（P0）。

**结论：注意力/KV 层的 Python 抽象干净，Gate 0 不需要提供 NPU 算子，只需保证 Registry 未注册时不会阻塞引擎启动（当前实现如为强注册需引入延迟加载或 try/except 保护）。**

## 8. 图 / 编译分析 (Graph & Compile Analysis)

- 全仓库 **无 `torch.compile`**（`rg 'torch\.compile'` 无命中），无 `torch.jit.script`；不存在 dynamo/inductor 依赖。
- CUDA Graph 集中在：
  - `python/minisgl/engine/graph.py:49-67` `_determine_cuda_graph_bs`——策略函数。
  - `python/minisgl/engine/graph.py:133` `torch.cuda.CUDAGraph()`。
  - `python/minisgl/engine/graph.py:140` `torch.cuda.graph(pool=...)`。
  - `python/minisgl/engine/engine.py:209` `destroy_cuda_graphs()`。
- Ascend 有 `aclgraph`/`torch_npu` 侧的图捕获路径，但需要 padding 与 stream capture 语义对齐，风险较高——**Gate 0 明确不做**。
- Gate 0 要求：`config.cuda_graph_max_bs = 0` 时应完全绕开图路径（当前 `python/minisgl/engine/graph.py` 已有相应分支，需要在 config 默认值层面 gate；`python/minisgl/engine/config.py` `cuda_graph_bs`/`cuda_graph_max_bs` 可通过 CLI `--cuda-graph-max-bs 0`（`python/minisgl/server/args.py:148-153`）关闭）。

**动作：Gate 0 之内不新增图相关代码；只保证 `use_cuda_graph=False` 路径 100% 走通。**

## 9. 文件级可移植性矩阵 (File-by-File Portability Matrix)

> P = portable（直接复用）；T = thin-adaptation（微调）；R = backend-replacement（后端替换）；X = major-rewrite（大改）。

**控制面 / 引擎 / 调度**

| 文件 | 判定 | 依据 |
| :-- | :-- | :-- |
| `python/minisgl/core.py` | P | 纯 dataclass |
| `python/minisgl/env.py` | T | 需新增 Ascend 相关默认值 |
| `python/minisgl/engine/config.py` | T | `cuda_graph_*` / `use_pynccl` 默认值切换 |
| `python/minisgl/engine/engine.py` | T | 全体 `torch.cuda.*` 走抽象层（第 4 节列举） |
| `python/minisgl/engine/graph.py` | R | `CUDAGraph()`/`torch.cuda.graph()` 无 Ascend 对应；Gate 0 关闭 |
| `python/minisgl/engine/sample.py` | R | `flashinfer.sampling`+nvtx；采样重写为 torch |
| `python/minisgl/scheduler/scheduler.py` | T | `Stream/Event` 抽象化 |
| `python/minisgl/scheduler/cache.py` | P | `pin_memory` 仅托管 |
| `python/minisgl/scheduler/prefill.py` | P | 同上 |
| `python/minisgl/scheduler/io.py` | P | ZMQ+`torch.distributed` CPU |
| `python/minisgl/scheduler/config.py` | P | dataclass |

**分布式面**

| 文件 | 判定 | 依据 |
| :-- | :-- | :-- |
| `python/minisgl/distributed/info.py` | P | dataclass |
| `python/minisgl/distributed/impl.py::TorchDistributedImpl` | P | 后端字符串驱动 |
| `python/minisgl/distributed/impl.py::PyNCCLDistributedImpl` | X | 依赖自研 NCCL wrapper |
| `python/minisgl/distributed/impl.py::enable_pynccl_distributed` | T | Ascend 强制走 Torch 分支 |

**算子面 / kernel/**

| 文件 | 判定 | 依据 |
| :-- | :-- | :-- |
| `python/minisgl/kernel/__init__.py` | T | 贪婪导入需 lazy 化 |
| `python/minisgl/kernel/utils.py` | T | `tvm_ffi.cpp.load/load_inline` 的 CUDA 分支需 gate |
| `python/minisgl/kernel/pynccl.py` | X | NCCL wrapper（Gate 1） |
| `python/minisgl/kernel/index.py` | R | JIT `.cu` gather；Ascend 用 torch 回退 |
| `python/minisgl/kernel/store.py` | R | JIT `.cu` KV 写入；同上 |
| `python/minisgl/kernel/tensor.py` | P | CPU AOT |
| `python/minisgl/kernel/radix.py` | P | CPU AOT |
| `python/minisgl/kernel/moe_impl.py` | R | Triton MoE；Gate 0 禁用 |
| `python/minisgl/kernel/triton/fused_moe.py` | R | 同上 |
| `python/minisgl/kernel/csrc/include/minisgl/nccl227.h` | X | NCCL 头 |
| `python/minisgl/kernel/csrc/src/pynccl.cu` | X | NCCL 实现 |

**注意力 / KV / MoE / 层**

| 文件 | 判定 | 依据 |
| :-- | :-- | :-- |
| `python/minisgl/attention/base.py` | P | ABC |
| `python/minisgl/attention/utils.py` | P | 工具 |
| `python/minisgl/attention/__init__.py` | T | Registry；Ascend 后端注册在 Gate 1 |
| `python/minisgl/attention/fi.py` | R | FlashInfer |
| `python/minisgl/attention/fa.py` | R | sgl_kernel |
| `python/minisgl/attention/trtllm.py` | R | flashinfer.trtllm.* |
| `python/minisgl/kvcache/base.py` | P | ABC |
| `python/minisgl/kvcache/naive_cache.py` | P | 纯 torch |
| `python/minisgl/kvcache/radix_cache.py` | P | 纯 torch + CPU radix |
| `python/minisgl/kvcache/mha_pool.py` | R | store_cache（`.cu`） |
| `python/minisgl/kvcache/__init__.py` | P | Registry |
| `python/minisgl/moe/fused.py` | R | sgl_kernel + Triton |
| `python/minisgl/layers/base.py` | P | ABC |
| `python/minisgl/layers/linear.py` | P | `F.linear` |
| `python/minisgl/layers/attention.py` | P | 编排 |
| `python/minisgl/layers/moe.py` | P | 编排（内部路径 Gate 0 关闭） |
| `python/minisgl/layers/rotary.py` | R | flashinfer |
| `python/minisgl/layers/norm.py` | R | flashinfer |
| `python/minisgl/layers/activation.py` | R | flashinfer |
| `python/minisgl/layers/embedding.py` | R | kernel.indexing |

**模型 / 服务 / 工具 / benchmark**

| 文件 | 判定 | 依据 |
| :-- | :-- | :-- |
| `python/minisgl/models/base.py` | P | ABC |
| `python/minisgl/models/weight.py` | P | HF 权重加载 |
| `python/minisgl/models/llama.py` | P | LlamaModel |
| `python/minisgl/models/qwen3_moe.py` | P | MoE 路径 Gate 0 禁用 |
| `python/minisgl/server/args.py` | T | 增加 Ascend 默认值 / `--disable-pynccl`/`--cuda-graph-max-bs` 默认切换 |
| `python/minisgl/server/launch.py` | P | mp spawn |
| `python/minisgl/server/http.py` | P | FastAPI |
| `python/minisgl/utils/arch.py` | T | 设备探测抽象化 |
| `python/minisgl/utils/torch_utils.py` | T | nvtx 抽象/懒加载 |
| `python/minisgl/utils/mp.py` | P | ZMQ+msgpack |
| `python/minisgl/utils/hf.py` | P | HF 下载 |
| `python/minisgl/utils/logger.py` | P | logging |
| `python/minisgl/utils/registry.py` | P | Registry |
| `python/minisgl/utils/misc.py` | P | 杂项 |
| `python/minisgl/benchmark/perf.py` | R | 使用 `CUDAGraph`，dev-only |

**统计**（按文件计数，包含 `.py` 与关键 `.cu/.h` 文件）：
- P（portable）：约 34 个文件
- T（thin-adaptation）：约 12 个文件
- R（backend-replacement）：约 13 个文件
- X（major-rewrite）：约 4 个文件（`kernel/pynccl.py` 及其 `.cu/.h`、`distributed/impl.py::PyNCCLDistributedImpl` 语义等价）
- **可直接复用比例 ≈ 53%（P）+ 20%（T 边缘复用）≈ 70–73%**。

## 10. 建议的 Ascend 后端边界 (Proposed Ascend Backend Boundary)

新增“设备抽象”和“通信后端选择”是本次迁移的**唯一横切改动**。所有新增文件默认放在：
- `python/minisgl/utils/device.py` — 提供 `class DeviceContext`：`is_available()`、`device_type()`（`"cuda"`/`"npu"`/`"cpu"`）、`set_device(idx)`、`current_stream()`、`Stream()`、`Event()`、`empty_cache()`、`synchronize()`、`mem_get_info()`、`nvtx_range(name)`；Ascend 分支懒加载 `torch_npu`。
- `python/minisgl/distributed/backend.py` — 提供 `select_device_backend()`：Ascend 返回 `"hccl"`，否则 `"nccl"`；被 `engine.py:128` 调用。
- `python/minisgl/kernel/_cuda_optional.py`（可选）— 集中管理 `try: import ... except ImportError` 的 CUDA 依赖入口，令 `kernel/__init__.py` 从贪婪导入改为延迟。

**明确排除**：本轮不新增 NPU 算子文件、`kernel/csrc/src/pyhccl.*` 等；不修改 `pyproject.toml` 的 dependencies 数组（只允许在 Gate 1+ 增加 `[project.optional-dependencies]` 中的 `[ascend]` / `[cuda]` extras）。

## 11. Gate 0 最小改动集 (Gate 0 Minimal Change Set)

**Gate 0 目标：在无 NPU 算子实现的前提下，让 `python -c "import minisgl"` 与 `minisgl-serve --model ... --tp 1 --disable-pynccl --cuda-graph-max-bs 0` 在 Ascend 主机上能拉起进程；不追求推理正确性。**

Gate 0 允许修改/新增的文件（**均属 T / 少量新文件**，无 R/X）：

1. **新增** `python/minisgl/utils/device.py` — 设备抽象层。
2. **新增** `python/minisgl/distributed/backend.py` — `select_device_backend()`。
3. **修改** `python/minisgl/utils/arch.py:12,14` — 通过 `device.py` 返回；NPU 上 `is_sm90/sm100` 均为 `False`。
4. **修改** `python/minisgl/utils/torch_utils.py:24` — `nvtx_annotate` 使用抽象层，Ascend 上退化 no-op。
5. **修改** `python/minisgl/engine/engine.py:31,35,36,38-39,128,172-174,192,204,209` — 替换直接 `torch.cuda.*` 与 `backend="nccl"` 硬编码为抽象层调用；`destroy_cuda_graphs` 在 `use_cuda_graph=False` 时被跳过。
6. **修改** `python/minisgl/engine/config.py` — Ascend 上 `cuda_graph_bs=[]`、`cuda_graph_max_bs=0`、`use_pynccl=False` 作为默认。
7. **修改** `python/minisgl/scheduler/scheduler.py:53-55,128,134` — 抽象层化。
8. **修改** `python/minisgl/distributed/impl.py:73-90` — Ascend 上 `enable_pynccl_distributed` 强制返回 `TorchDistributedImpl`。
9. **修改** `python/minisgl/kernel/__init__.py` — 将 `pynccl` / `moe_impl` / `store` / `index` 的 top-level 导入改为**延迟/守护式**（仅在 CUDA 可用时导入；否则暴露占位 `None` 或抛延迟错误）。
10. **修改** `python/minisgl/kernel/pynccl.py:30-37` — 在无 CUDA / 未开启 pynccl 时短路，不触发 `load_aot`。
11. **修改** `python/minisgl/server/args.py:124-129,148-153` — Ascend 上 `--disable-pynccl` 默认 `True`、`--cuda-graph-max-bs` 默认 `0`。

Gate 0 **不允许**触碰的项（列入 Gate 1+）：
- `python/minisgl/kernel/csrc/**` 全部 `.cu`/`.h`。
- `python/minisgl/attention/{fi,fa,trtllm}.py` 后端实现体。
- `python/minisgl/layers/{norm,rotary,activation,embedding}.py`（保留 flashinfer/kernel 引用，只要不被启动路径触达即可；如 layer 在 import 阶段直接 `import flashinfer`，则本次改为 `try/except`）。
- `python/minisgl/moe/fused.py`、`python/minisgl/kernel/moe_impl.py`、`python/minisgl/kernel/triton/fused_moe.py`（MoE 全禁用）。
- `python/minisgl/engine/graph.py`（Gate 0 走 `use_cuda_graph=False` 分支）。
- `python/minisgl/engine/sample.py` 中 `flashinfer.sampling`（暂不替换；Gate 0 允许启动失败在采样阶段，或做 lazy import + Ascend 上落回 torch 版——但那已属 Gate 1 内容）。
- `pyproject.toml`、`uv.lock`、`LICENSE`、`NOTICE`。

## 12. 风险与开放问题 (Risks and Open Questions)

1. **`kernel/__init__.py` 贪婪导入**：当前实现会在 `import minisgl` 时立即触发 `pynccl.load_aot(cuda_files=...)`（`kernel/pynccl.py:30`）。在无 CUDA 主机上会崩溃。Gate 0 必须先解决这条阻塞路径。
2. **flashinfer 顶层 import**：`python/minisgl/layers/norm.py:10`、`rotary.py:35`、`activation.py:10,16`、`engine/sample.py:30` 都在文件顶层 `import flashinfer`；在无 CUDA 主机 `flashinfer` 未安装时会 `ImportError`。Gate 0 需评估是否让这些模块的顶层导入延迟到函数体或 try/except——但这已经**逼近 Gate 0 边界**，需与用户确认是否允许。
3. **`sgl_kernel` 顶层 import**：`python/minisgl/attention/fa.py:158`、`python/minisgl/moe/fused.py:16,71` 同样风险。
4. **PyTorch NPU 后端字符串**：`torch_npu` 注册的分布式后端名（`"hccl"`）与 PyTorch 版本紧密耦合，需在 Gate 0 实机确认。
5. **`torch_npu.npu.Stream/Event`** 语义与 CUDA 完全对齐——在部分固件/驱动组合下 `Event.wait(stream)` 支持不完整；`scheduler.py` 双流重叠路径需要在 Gate 1 验证。
6. **`torch.pin_memory=True`** 在部分 NPU 环境不可用；若阻塞，需要在 Gate 1 做 fallback（当前列 P2）。
7. **`torch.cuda.mem_get_info`**：`torch_npu.npu.mem_get_info` 在旧版 CANN 上可能不存在，Gate 1 要做 try/except。
8. **License**：仓库无 `NOTICE`；建议 fork 时补充作者/上游引用（不属 Gate 0）。上游 `sgl-project` MIT，无 attribution 阻塞。
9. **`pyproject.toml` 依赖治理**：`accelerate`、`quack-kernels` 完全无源码引用，是**廉价清理项**，但按硬约束**本次不改**。
10. **未证实项**：本报告未运行任何代码，`torch_npu` API 兼容性、`hccl` 后端在当前 `torch<2.10.0` 约束下是否可用，均需在 Gate 0 实机执行阶段确认；本报告仅提供源码级导航。
