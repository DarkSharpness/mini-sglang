from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Set, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortAckReply,
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[BaseFrontendMsg]:
    """Flatten one wire message into its atomic BaseFrontendMsg children.

    Gate 2.3f: return type widened from ``List[UserReply]`` to
    ``List[BaseFrontendMsg]`` so ``AbortAckReply`` can travel the same
    frontend-inbound channel. Downstream dispatch is ``isinstance``-based
    in ``FrontendManager.listen``.
    """
    if isinstance(msg, BatchFrontendMsg):
        return list(msg.data)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    # Gate 2.3f: end-to-end abort acknowledgement bookkeeping.
    #
    # abort_pending — uids whose AbortMsg has been sent to the tokenizer but
    # whose AbortAckReply has not yet arrived from the scheduler. While a uid
    # is in this set, ANY UserReply for that uid is dropped (see ``listen``):
    # the request has been declared cancelled; late tokens produced by the
    # scheduler between the abort msg being enqueued and the fence closing
    # must never surface to the client. Removed exactly when the ack arrives.
    #
    # abort_pending_events — one-shot asyncio.Event per pending uid, set when
    # the ack arrives. Kept separate from event_map so abort_user's waiter
    # (if any) does not race the streaming waiter over the same Event.
    # abort_user itself does NOT block on this event — the scheduler → tokenizer
    # → frontend hop plus the ack flush is not synchronous with the caller's
    # abort request; callers that need proof of completion can await
    # ``wait_for_abort_ack``. The streaming path (``stream_with_cancellation``)
    # never awaits an ack: the client is already gone, so the cleanup must
    # not block the shutdown of its task.
    abort_pending: Set[int] = field(default_factory=set)
    abort_pending_events: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                # Gate 2.3f: AbortAckReply is dispatched independently from
                # UserReply. Cleanup ordering:
                #   1. remove uid from abort_pending  → subsequent UserReplies
                #      (should any still be in flight from a slower scheduler
                #      rank) will be treated as normal state-unknown drops,
                #      not "silently swallowed under abort".
                #   2. drop ack_map + event_map      → wait_for_ack's suspended
                #      awaiter will observe the KeyError-free "no more state"
                #      condition once its Event fires; see wait_for_ack for
                #      how the abort-pending case exits the yield loop.
                #   3. fire abort_pending_events     → any explicit awaiter
                #      of the ack (e.g. tests, non-streaming abort_user
                #      variants) unblocks.
                # An AbortAckReply for a uid NOT in abort_pending is a strict
                # no-op — could arise from a duplicate ack (Scheduler emits
                # exactly one per free but a retried AbortMsg from a caller
                # could produce two). Idempotency is required by spec.
                if isinstance(msg, AbortAckReply):
                    uid = msg.uid
                    self.abort_pending.discard(uid)
                    self.ack_map.pop(uid, None)
                    ev = self.event_map.pop(uid, None)
                    if ev is not None:
                        # Wake any streaming waiter so it can observe the
                        # empty ack list and exit — see wait_for_ack.
                        ev.set()
                    pending_ev = self.abort_pending_events.pop(uid, None)
                    if pending_ev is not None:
                        pending_ev.set()
                    continue

                assert isinstance(msg, UserReply)
                if msg.uid not in self.ack_map:
                    # Unknown uid: request already finished normally or the
                    # scheduler emitted a token during the abort fence window
                    # AFTER our ack cleanup ran. Drop silently.
                    continue
                # Gate 2.3f: while a uid is abort-pending, drop any UserReply
                # that races the AbortAckReply. This keeps the streamed output
                # from being polluted by a token that was already logically
                # cancelled by the time it reached us.
                if msg.uid in self.abort_pending:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()

            # Gate 2.3f: the AbortAckReply handler in ``listen`` removes both
            # ack_map[uid] and event_map[uid] as part of the ack cleanup, then
            # fires the event so this waiter unblocks. When we return from
            # ``await event.wait()`` we may therefore find the uid gone from
            # ack_map — that is the terminal "aborted" signal for this loop.
            pending = self.ack_map.pop(uid, None)
            if pending is None:
                # ack cleanup happened; no more replies will arrive. Exit
                # without touching event_map (already removed by listen).
                return
            # The list is drained; keep the slot alive with a fresh list so
            # subsequent ticks can accumulate.
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        """Mark ``uid`` as abort-pending and send AbortMsg to the tokenizer.

        Gate 2.3f state-machine entry point.

        The caller is NOT blocked waiting for the AbortAckReply — the
        cancellation path (``stream_with_cancellation``) is triggered by a
        client disconnect, so waiting for the scheduler round-trip is both
        unnecessary and undesirable (the request task must be free to
        finalise). ack cleanup is performed by ``listen`` when the ack
        arrives; callers that need to await proof of completion should
        instead await ``wait_for_abort_ack``.

        Idempotency: a duplicate call for the same uid re-adds the (already
        present) entry to ``abort_pending`` and sends a second AbortMsg. The
        scheduler tolerates duplicate aborts (see _process_one_msg's
        non-inflight branch: unknown uid → idempotent ack); the frontend
        tolerates duplicate acks (see ``listen``).
        """
        logger.warning("Aborting request for user %s", uid)
        # Move the uid into abort-pending BEFORE the AbortMsg is enqueued so
        # any UserReply that lands between the scheduler processing the abort
        # and the ack arriving is dropped by ``listen`` (dispatched on
        # abort_pending membership).
        self.abort_pending.add(uid)
        # Create the one-shot event lazily so wait_for_abort_ack has
        # something to await. If one already exists (duplicate abort_user),
        # reuse it so both awaiters unblock on the same ack.
        if uid not in self.abort_pending_events:
            self.abort_pending_events[uid] = asyncio.Event()
        await self.send_one(AbortMsg(uid=uid))

    async def wait_for_abort_ack(self, uid: int):
        """Block until ``uid``'s AbortAckReply has been processed.

        Gate 2.3f: used by callers that need synchronous proof of
        abort completion (tests, non-streaming HTTP paths that want
        deterministic cleanup). Does NOT trigger the abort itself — call
        ``abort_user`` first. Safe to await on a uid that already received
        its ack: ``abort_pending_events`` still holds a set Event until the
        listener's cleanup removes it, and a missing entry returns
        immediately (the ack must have already fired).
        """
        ev = self.abort_pending_events.get(uid)
        if ev is None:
            return
        await ev.wait()

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
