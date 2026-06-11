# Server

The `server/` component handles the HTTP API, process orchestration, and the IPC wiring between the frontend and backend workers.

---

## Process Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Main Process  (FastAPI / uvicorn)                                  │
│                                                                     │
│   FrontendManager                                                   │
│     ├── ZmqAsyncPushQueue  →  zmq_tokenizer_addr                    │
│     └── ZmqAsyncPullQueue  ←  zmq_frontend_addr                    │
└──────────────────────────────────────────────────────────────────────┘
              │                        ▲
        TokenizeMsg / AbortMsg    UserReply
              │                        │
              ▼                        │
┌─────────────────────────────────────────────────────────────────────┐
│  Tokenizer Processes  (N = num_tokenizer)                           │
│     tokenize_worker()                                               │
│       TokenizeMsg → input_ids → UserMsg → backend                  │
│       DetokenizeMsg ← next_token ← backend → UserReply → frontend  │
│       AbortMsg → AbortBackendMsg → backend                         │
└──────────────────────────────────────────────────────────────────────┘
              │
        UserMsg / AbortBackendMsg
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Scheduler Processes  (one per TP rank)                             │
│     Scheduler.run_forever()                                         │
│       ├── GPU 0 (primary) → sends DetokenizeMsg back               │
│       └── GPU 1..N-1      → silent TP workers                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Startup Sequence (`launch_server`)

```
1. parse_args()
2. run_api_server(config, start_backend_fn, run_shell)
        │
        ├── create FrontendManager (ZMQ sockets)
        ├── call start_backend_fn()        ← spawns subprocesses:
        │       ├── for i in range(tp_size):  mp.Process(_run_scheduler)
        │       ├── 1× detokenizer process   mp.Process(tokenize_worker)
        │       └── N× tokenizer processes   mp.Process(tokenize_worker)
        │       └── wait for ack_queue messages from all workers
        │
        └── start uvicorn  (or run interactive shell)
```

All subprocesses are spawned with `mp.set_start_method("spawn")`. They communicate via ZMQ IPC sockets (file paths configured in `ServerArgs`).

---

## HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/generate` | Raw streaming generation (SSE, token-per-line) |
| POST | `/v1/chat/completions` | OpenAI-compatible chat completions (stream or non-stream) |
| GET | `/v1/models` | List available model(s) |
| GET/POST | `/v1` | Health check |

---

## Request Lifecycle

```
HTTP POST /v1/chat/completions
    │
    ├── uid = state.new_user()         ← assign unique ID, create ack_map entry
    ├── await state.send_one(TokenizeMsg(uid, text, sampling_params))
    │
    └── if stream:
            return StreamingResponse(state.stream_chat_completions(uid))
        else:
            async for ack in state.wait_for_ack(uid):
                full_content += ack.incremental_output
            return JSON response

stream_chat_completions(uid):
    async for ack in wait_for_ack(uid):
        yield SSE chunk (OpenAI delta format)
    yield final chunk (finish_reason="stop")
    yield "data: [DONE]"
```

**Client disconnect detection**: `stream_with_cancellation` checks `request.is_disconnected()` on each chunk and sends `AbortMsg` if disconnected.

---

## FrontendManager

```python
@dataclass
class FrontendManager:
    uid_counter: int            ← monotonically increasing request ID
    ack_map: Dict[int, List[UserReply]]    ← buffered replies per uid
    event_map: Dict[int, asyncio.Event]   ← notifies when new reply arrives

listen():                       ← single asyncio task, drains recv queue
    while True:
        msg = await recv_tokenizer.get()
        for reply in _unwrap_msg(msg):
            ack_map[reply.uid].append(reply)
            event_map[reply.uid].set()

wait_for_ack(uid):              ← async generator yielding UserReplys
    while True:
        await event.wait(); event.clear()
        for ack in pending: yield ack
        if ack.finished: break
    cleanup ack_map / event_map
```

---

## Interactive Shell Mode

When `--shell` flag is set, `run_shell=True` is passed to `run_api_server`, which calls `asyncio.run(shell())` instead of starting uvicorn. The shell is a `prompt_toolkit` REPL that maintains conversation history and calls the server's internal `shell_completion()` function directly.

---

## Key Files

| File | Responsibility |
|------|---------------|
| `launch.py` | `launch_server`, subprocess spawning, ack synchronization |
| `api_server.py` | FastAPI app, `FrontendManager`, HTTP endpoints, shell |
| `args.py` | `ServerArgs` — all configuration (host, port, ZMQ addrs, TP, model path) |
