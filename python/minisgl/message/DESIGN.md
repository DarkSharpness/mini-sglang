# Message

The `message/` component defines the message types that flow between the three process tiers: frontend (HTTP server), tokenizer, and scheduler backend. All messages are serialized over ZMQ sockets.

---

## Message Hierarchy

```
                    ┌─ Frontend ──────────┐
                    │  BaseFrontendMsg    │
                    │    BatchFrontendMsg │
                    │    UserReply        │
                    └─────────────────────┘
                           ▲
                    backend → frontend

BaseTokenizerMsg
    BatchTokenizerMsg     ← batch wrapper
    TokenizeMsg           ← frontend → tokenizer (raw text)
    DetokenizeMsg         ← backend  → detokenizer (next token)
    AbortMsg              ← frontend → tokenizer (cancel request)

BaseBackendMsg
    BatchBackendMsg       ← batch wrapper
    UserMsg               ← tokenizer → backend (token IDs)
    AbortBackendMsg       ← tokenizer → backend (cancel)
    ExitMsg               ← signal backend to shut down
```

---

## Message Flow by Path

```
User HTTP request
    │  TokenizeMsg(uid, text, sampling_params)
    ▼
Tokenizer Worker
    │  UserMsg(uid, input_ids: Tensor, sampling_params)
    ▼
Scheduler Backend
    │  DetokenizeMsg(uid, next_token: int, finished: bool)
    ▼
Detokenizer Worker
    │  UserReply(uid, incremental_output: str, finished: bool)
    ▼
Frontend Manager → HTTP response stream
```

Abort path:
```
Frontend → AbortMsg(uid) → Tokenizer → AbortBackendMsg(uid) → Scheduler
```

---

## Batch Wrappers

To reduce ZMQ round-trips, multiple messages may be wrapped in a single batch envelope:

```
BatchBackendMsg(data: List[BaseBackendMsg])
BatchFrontendMsg(data: List[BaseFrontendMsg])
BatchTokenizerMsg(data: List[BaseTokenizerMsg])
```

Receivers always unwrap these before processing individual messages.

---

## Serialization

All message classes use a `serialize_type` / `deserialize_type` utility:

```python
serialize_type(msg)   → {"__type__": "UserMsg", "uid": 1, "input_ids": ..., ...}
deserialize_type(globals(), json) → UserMsg(...)
```

The `__type__` field is the class name, looked up in the module's `globals()` dict. Tensors are serialized as lists and reconstructed as `torch.int32` CPU tensors.

---

## Key Message Fields

| Message | Key Fields |
|---------|-----------|
| `TokenizeMsg` | `uid`, `text` (str or chat list), `sampling_params` |
| `UserMsg` | `uid`, `input_ids` (CPU int32 Tensor), `sampling_params` |
| `DetokenizeMsg` | `uid`, `next_token` (int), `finished` (bool) |
| `UserReply` | `uid`, `incremental_output` (str), `finished` (bool) |
| `AbortMsg` / `AbortBackendMsg` | `uid` |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `backend.py` | `BaseBackendMsg`, `UserMsg`, `BatchBackendMsg`, `ExitMsg`, `AbortBackendMsg` |
| `frontend.py` | `BaseFrontendMsg`, `UserReply`, `BatchFrontendMsg` |
| `tokenizer.py` | `BaseTokenizerMsg`, `TokenizeMsg`, `DetokenizeMsg`, `AbortMsg`, `BatchTokenizerMsg` |
| `utils.py` | `serialize_type`, `deserialize_type` |
