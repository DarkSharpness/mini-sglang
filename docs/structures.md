# Repository Structure

## Python package `minisgl`

![Process overview diagram](./images/overall.png)

The mini-sglang python package lives in `python/minisgl`. Its submodules and subpackages include:

- `minisgl.config.context`: Provides core dataclasses `Req` and `Batch` representing the state of requests, and class `Context` which holds the global state of the inference context.
- `minisgl.distributed`: Provides the interface to all-reduce and all-gather in tensor parallelism, and dataclass `DistributedInfo` which holds the tp information for a tp worker.
- `minisgl.layers`: Implements basic building blocks for building LLMs with tp support, including linear, layernorm, embedding, RoPE, etc. They share common base classes defined in `minisgl.layers.base`.
- `minisgl.models`: Implements LLM models, including Llama and Qwen3. Also defines utilities for loading weights from huggingface and sharding weights.
- `minisgl.attention`: Provides interface of attention Backends and implements backends of `flashattention` and `flashinfer`. They are called by `AttentionLayer` and uses metadata stored in `Context`.
- `minisgl.kvcache`: Provides interface of kvcache pool and kvcache manager, and implements `MHAKVCache`, `NaiveCacheManager` and `RadixCacheManager`.
- `minisgl.utils`: Provides a collection of utilities, including logger setup and wrappers around zmq.
- `minisgl.engine`: Implements `Engine` class, which is a tp worker on a single process. It manages the model, context, kvcache, attention backend and cuda graph replaying.
- `minisgl.message`: Defines serialization and deserialization of messages exchanged (in zmq) between api_server, tokenizer, detokenizer and scheduler.
- `minisgl.scheduler`: Implements `Scheduler` class, which runs on each tp worker process and manages the corresponding `Engine`. The rank 0 scheduler receives msgs from tokenizer, communicates with scheduler on other tp workers, and sends msgs to detokenizer.
- `minisgl.server`: Defines cli arguments and `launch_server` which starts all the subprocesses of mini-sglang. Also implements a FastAPI server in `minisgl.server.api_server` acting as a frontend, providing endpoints such as `/v1/chat/completions`.
- `minisgl.tokenizer`: Implements `tokenize_worker` function which handles tokenization and detokenization requests.
- `minisgl.llm`: Provides class `LLM` as a python interface to interact with the mini-sglang system easily.
- `minisgl.kernel_v2`: Implements custom CUDA kernels and bindings, supported by `tvm-ffi` as ffi and jit interface.
- `minisgl.benchmark`: Benchmark utilities.
