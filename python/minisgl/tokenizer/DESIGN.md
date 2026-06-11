# Tokenizer

The `tokenizer/` component runs in a dedicated subprocess (or multiple) bridging the HTTP frontend and the GPU scheduler backend. It handles tokenization of incoming prompts and detokenization of generated token IDs into text.

---

## Role in the System

```
Frontend (async HTTP)
    │  TokenizeMsg / AbortMsg
    ▼
tokenize_worker  (subprocess)
    ├── tokenize  → UserMsg        →  Scheduler backend
    ├── detokenize ← DetokenizeMsg ←  Scheduler backend
    └── AbortMsg  → AbortBackendMsg → Scheduler backend
    │  UserReply
    ▼
Frontend (async HTTP)
```

There are two kinds of workers using the same `tokenize_worker` function:
- **Tokenizer workers** (`N = num_tokenizer`): receive from `zmq_tokenizer_addr`, handle `TokenizeMsg`.
- **Detokenizer worker** (`1`): receives from `zmq_detokenizer_addr`, handles `DetokenizeMsg`.

Both send `UserMsg`/`AbortBackendMsg` to `zmq_backend_addr` and `UserReply` to `zmq_frontend_addr`.

---

## tokenize_worker Loop

```python
while True:
    pending = [recv_listener.get()]             ← blocking get
    while len(pending) < local_bs and not recv_listener.empty():
        pending.extend(recv_listener.get())     ← opportunistic batching

    detokenize_msg = [m for m in pending if isinstance(m, DetokenizeMsg)]
    tokenize_msg   = [m for m in pending if isinstance(m, TokenizeMsg)]
    abort_msg      = [m for m in pending if isinstance(m, AbortMsg)]

    if detokenize_msg:
        replies = DetokenizeManager.detokenize(detokenize_msg)
        send UserReply(uid, incremental_text, finished) → frontend

    if tokenize_msg:
        tensors = TokenizeManager.tokenize(tokenize_msg)
        send UserMsg(uid, input_ids, sampling_params) → backend

    if abort_msg:
        send AbortBackendMsg(uid) → backend
```

Opportunistic batching: after the first blocking receive, the worker drains any additional queued messages up to `local_bs` without blocking, processing them in one batch.

---

## TokenizeManager

```
TokenizeManager.tokenize(msgs: List[TokenizeMsg])
    │
    ├── for each msg:
    │       if msg.text is str → tokenizer.encode(text)
    │       if msg.text is list (chat messages) →
    │           tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    │
    └── returns List[Tensor[int32]]  ← CPU tensors, one per request
```

Supports both raw string prompts and OpenAI-style message lists (role/content dicts).

---

## DetokenizeManager

```
DetokenizeManager.detokenize(msgs: List[DetokenizeMsg])
    │
    ├── maintains per-uid state: List[int]  (accumulated token ids)
    ├── for each msg:
    │       append msg.next_token to uid's buffer
    │       decode(buffer) - decode(buffer[:-1])  ← incremental decode
    │       (avoids multi-byte UTF-8 boundary issues)
    │
    └── returns List[str]  ← incremental text strings

Finished requests: uid state is cleaned up when msg.finished == True
```

Incremental detokenization: re-decoding the full buffer minus the last token, then subtracting the previous string, correctly handles tokens that span byte boundaries (e.g., multi-byte UTF-8 characters).

---

## Message Types Handled

| Incoming | Direction | Action |
|----------|-----------|--------|
| `TokenizeMsg(uid, text, sampling_params)` | frontend → tokenizer | encode text → `UserMsg` |
| `DetokenizeMsg(uid, next_token, finished)` | backend → detokenizer | decode token → `UserReply` |
| `AbortMsg(uid)` | frontend → tokenizer | forward as `AbortBackendMsg` |
| `BatchTokenizerMsg` | any | unwrap and process each sub-message |

---

## Key Files

| File | Responsibility |
|------|---------------|
| `server.py` | `tokenize_worker` — main event loop, message routing |
| `tokenize.py` | `TokenizeManager` — text → token IDs |
| `detokenize.py` | `DetokenizeManager` — token ID → incremental text |
