# Structure of Mini-SGLang

## Component Overview

Mini-SGLang's inference system consists of these processes:

- An API server providing endpoints such as `/v1/chat/completions` for users to interact with.
- One detokenizer worker.
- Zero or more tokenizer workers. If `num_tokenizer` is set to 0, the detokenizer worker also handles tokenization.
- For each tensor parallel (tp) rank, one scheduler worker that handles scheduling and computation on that gpu.

Check `minisgl.server.launch_server` for more details on how these processes are launched.

The API server, tokenizer, detokenizer and schedulers communicate messages through zmq. Schedulers also communicate with each other through `minisgl.distributed` interfaces (backended with NCCL or `torch.distributed`) to perform all-reduce and all-gather operations for tensor parallelism.

![Process overview diagram](./images/overall.png)

The arrows in the diagram above represent the data flow between different processes. The full lifecycle of a request is as follows:

1. User sends a request to the API server.
2. The API server forwards the request to the tokenizer worker.
3. The tokenizer worker tokenizes the input text and sends the tokenized request to the scheduler of tp rank 0.
4. The scheduler of tp rank 0 broadcasts the request to all other tp rank schedulers via zmq.
5. Each scheduler receives the request and schedules the computation on its corresponding tp worker (the `Engine`).
6. The scheduler of tp rank 0 sends the output to the detokenizer worker.
7. The detokenizer worker detokenizes the output and sends it back to the API server.
8. The API server returns the response to the user.

## The `minisgl` Python Package

The Mini-SGLang python package lives in `python/minisgl`. Its submodules and subpackages include:

- `minisgl.core`: Provides core dataclasses `Req` and `Batch` representing the state of requests, and class `Context` which holds the global state of the inference context.
- `minisgl.distributed`: Provides the interface to all-reduce and all-gather in tensor parallelism, and dataclass `DistributedInfo` which holds the tp information for a tp worker.
- `minisgl.layers`: Implements basic building blocks for building LLMs with tp support, including linear, layernorm, embedding, RoPE, etc. They share common base classes defined in `minisgl.layers.base`.
- `minisgl.models`: Implements LLM models, including Llama and Qwen3. Also defines utilities for loading weights from huggingface and sharding weights.
- `minisgl.attention`: Provides interface of attention Backends and implements backends of `flashattention` and `flashinfer`. They are called by `AttentionLayer` and uses metadata stored in `Context`.
- `minisgl.kvcache`: Provides interface of kvcache pool and kvcache manager, and implements `MHAKVCache`, `NaiveCacheManager` and `RadixCacheManager`.
- `minisgl.utils`: Provides a collection of utilities, including logger setup and wrappers around zmq.
- `minisgl.engine`: Implements `Engine` class, which is a tp worker on a single process. It manages the model, context, kvcache, attention backend and cuda graph replaying.
- `minisgl.message`: Defines serialization and deserialization of messages exchanged (in zmq) between api_server, tokenizer, detokenizer and scheduler.
- `minisgl.scheduler`: Implements `Scheduler` class, which runs on each tp worker process and manages the corresponding `Engine`. The rank 0 scheduler receives msgs from tokenizer, communicates with scheduler on other tp workers, and sends msgs to detokenizer.
- `minisgl.server`: Defines cli arguments and `launch_server` which starts all the subprocesses of Mini-SGLang. Also implements a FastAPI server in `minisgl.server.api_server` acting as a frontend, providing endpoints such as `/v1/chat/completions`.
- `minisgl.tokenizer`: Implements `tokenize_worker` function which handles tokenization and detokenization requests.
- `minisgl.llm`: Provides class `LLM` as a python interface to interact with the Mini-SGLang system easily.
- `minisgl.kernel_v2`: Implements custom CUDA kernels and bindings, supported by `tvm-ffi` as ffi and jit interface.
- `minisgl.benchmark`: Benchmark utilities.
