from __future__ import annotations

import asyncio
import os
import random
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, overload

from minisgl.utils import UNSET, Unset, init_logger
from openai import AsyncOpenAI as OpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

logger = init_logger(__name__)


@dataclass(frozen=True)
class BenchmarkTrace:
    timestamp: float
    message: str  # unit (second)
    output: int  # output length in tokens
    input_: int | None = None  # input length in tokens, optional


@dataclass(frozen=True)
class BenchOneResult:
    tics: List[float]
    input_len: int
    output_len: int

    def as_json(self) -> List[float]:
        return [self.input_len, self.output_len] + self.tics

    @staticmethod
    def from_json(raw: List[float]) -> BenchOneResult:
        # check raw[0] and raw[1] are integers
        assert raw[0].is_integer() and raw[1].is_integer()
        return BenchOneResult(tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1]))


DIFF_THRESHOLD = 2


@dataclass(frozen=True)
class RawResult:
    input_len: int | None
    output_len: int
    message: str
    tics: List[float]

    def get_input_len(self, tokenizer: Any) -> int:
        if os.environ.get("SKIP_TOKENIZE_CHECK") and self.input_len is not None:
            return self.input_len
        result = len(tokenizer.encode(self.message))
        if self.input_len is not None:
            assert abs(result - self.input_len) <= DIFF_THRESHOLD, f"{result} vs {self.input_len}"
        return result


@dataclass
class Counter:
    current: int = 0
    history_max: int = 0

    def inc(self, n=1):
        self.current += n
        self.history_max = max(self.history_max, self.current)

    def dec(self, n=1):
        self.current -= n
        assert self.current >= 0


@dataclass
class Console:
    input_pbar: tqdm
    output_pbar: tqdm
    prefill_pbar: tqdm
    decode_pbar: tqdm
    disabled: bool
    inflight_counter: Counter = field(default_factory=Counter)
    queue_counter: Counter = field(default_factory=Counter)

    def update_input(self, n=1):
        self.input_pbar.update(n)
        self.input_pbar.refresh()
        self.inflight_counter.inc(n)
        self.queue_counter.inc(n)

    def update_output(self, n=1):
        self.output_pbar.update(n)
        self.output_pbar.refresh()
        self.inflight_counter.dec(n)

    def update_prefill(self, n=1):
        self.prefill_pbar.update(n)
        self.prefill_pbar.refresh()
        self.queue_counter.dec(n)

    def update_decode(self, n=1):
        self.decode_pbar.update(n)

    @contextmanager
    def inflight(self, n=1):
        self.update_input(n)
        try:
            yield
        finally:
            self.update_output(n)

    @contextmanager
    def log_stats(self):
        try:
            yield
        finally:
            self.input_pbar.close()
            self.output_pbar.close()
            self.prefill_pbar.close()
            self.decode_pbar.close()
            if not self.disabled:
                max_inflight = self.inflight_counter.history_max
                max_queue = self.queue_counter.history_max
                logger.info(
                    f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}"
                )


@dataclass(frozen=True)
class BenchmarkResult:
    raw_data: List[BenchOneResult]

    def as_json(self) -> List[List[float]]:
        return [r.as_json() for r in self.raw_data]

    @staticmethod
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])


@asynccontextmanager
async def async_guard(callback: Callable[[], Awaitable[None]] | None):
    callback_task = None
    try:
        if callback is not None:

            async def callback_runner():
                await callback()

            callback_task = asyncio.create_task(callback_runner())
        yield
    finally:
        if callback_task is not None:
            if not callback_task.done():
                logger.warning("Cancelling the callback task...")
                callback_task.cancel()
            try:
                await callback_task
            except asyncio.CancelledError:
                pass


def make_console(length: int, sum_output_len: int, use_pbar: bool = True) -> Console:
    BAR_FORMAT_0 = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    BAR_FORMAT_1 = BAR_FORMAT_0
    n_fmt_align = 5
    if len(str(sum_output_len)) > 5:
        n_fmt_align = len(str(sum_output_len))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace("{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}")
        BAR_FORMAT_1 = BAR_FORMAT_0

    if len(str(length)) < len(str(sum_output_len)):
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"
        n_fmt_align += len(str(sum_output_len)) - len(str(length))
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}")

    disabled = not use_pbar
    input_pbar = tqdm(
        total=length, desc="Requests sent", position=0, bar_format=BAR_FORMAT_0, disable=disabled
    )
    output_pbar = tqdm(
        total=length, desc="Requests done", position=1, bar_format=BAR_FORMAT_0, disable=disabled
    )
    prefill_pbar = tqdm(
        total=length, desc="Prefill token", position=2, bar_format=BAR_FORMAT_0, disable=disabled
    )
    decode_pbar = tqdm(
        total=sum_output_len,
        desc="Decode token ",
        position=3,
        bar_format=BAR_FORMAT_1,
        disable=disabled,
    )
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


def generate_message(tokenizer: Any, n: int) -> str:
    """Generate a message of approximately `n` tokens using the provided tokenizer."""
    vocab_size = tokenizer.vocab_size // 2
    msg = tokenizer.decode([random.randint(0, vocab_size) for _ in range(n - 1)])
    for _ in range(32):
        ids = tokenizer.encode(msg)
        if len(ids) == n:
            return msg
        if len(ids) < n:
            need = n - len(ids)
            ids.extend([random.randint(0, vocab_size) for _ in range(need)])
        else:
            ids = ids[:n]
        ids = ids[1:]
        msg = tokenizer.decode(ids)
    raise ValueError("Failed to generate a message of the desired length.")


async def benchmark_one(
    client: OpenAI,
    msg: str,
    out: int,
    model: str,
    console: Console | None = None,
    in_: int | None = None,
) -> RawResult:
    if console is None:
        console = make_console(1, out, use_pbar=False)
    with console.inflight(1):
        kwargs = {
            "ignore_eos": True,
            "top_k": 1,
        }
        if in_ is not None:
            kwargs["input_len"] = in_
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": msg,
                }
            ],
            max_tokens=out,
            temperature=0.0,
            extra_body=kwargs,
        )
        tics = [time.perf_counter()]
        async for _ in response:
            tics.append(time.perf_counter())
            if len(tics) == 2:
                console.update_prefill()
            elif len(tics) <= out + 1:
                console.update_decode()
        return RawResult(
            input_len=in_,
            output_len=out,
            message=msg,
            tics=tics,
        )


async def benchmark_batch(
    client: OpenAI,
    msgs: List[str],
    out: int,
    model: str,
    use_pbar: bool = True,
    callback: Callable[[], Awaitable[None]] | None = None,
    in_: int | None = None,
) -> List[RawResult]:
    console = make_console(len(msgs), (out - 1) * len(msgs), use_pbar)
    tasks = [benchmark_one(client, msg, out, model, console, in_) for msg in msgs]
    async with async_guard(callback):
        with console.log_stats():
            return await asyncio.gather(*tasks)


async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    use_pbar: bool = True,
    callback: Callable[[], Awaitable[None]] | None = None,
) -> List[RawResult]:
    console = make_console(len(msgs), sum(msg.output - 1 for msg in msgs), use_pbar)
    start = time.perf_counter()
    offset = min(msg.timestamp for msg in msgs) - 1

    async def benchmark_timed(msg: BenchmarkTrace):
        target = start + msg.timestamp - offset
        await asyncio.sleep(max(0, target - time.perf_counter()))
        return await benchmark_one(client, msg.message, msg.output, model, console, msg.input_)

    tasks = [benchmark_timed(msg) for msg in msgs]
    async with async_guard(callback):
        with console.log_stats():
            return await asyncio.gather(*tasks)


@overload
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult: ...


@overload
def process_benchmark_results(raw_data: List[RawResult]) -> None: ...


def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> BenchmarkResult | None:
    accum_times: List[float] = []
    first_times: List[float] = []
    results = [r.tics for r in raw_data]
    for tics in results:
        deltas: List[float] = []
        for i in range(len(tics) - 1):
            diff = tics[i + 1] - tics[i]
            deltas.append(diff)
        first_times.append(deltas[0])
        accum_times.extend(deltas[1:])

    e2e_times = [tics[-1] - tics[0] for tics in results]
    first_times.sort()
    accum_times.sort()
    e2e_times.sort()

    avg_ttft = sum(first_times) / len(first_times) * 1000
    p50_ttft = first_times[int(len(first_times) * 0.5)] * 1000
    p90_ttft = first_times[int(len(first_times) * 0.9)] * 1000
    p99_ttft = first_times[int(len(first_times) * 0.99)] * 1000
    max_ttft = max(first_times) * 1000

    avg_tpot = sum(accum_times) / len(accum_times) * 1000
    p50_tpot = accum_times[int(len(accum_times) * 0.5)] * 1000
    p90_tpot = accum_times[int(len(accum_times) * 0.9)] * 1000
    p99_tpot = accum_times[int(len(accum_times) * 0.99)] * 1000
    max_tpot = max(accum_times) * 1000

    avg_e2e = sum(e2e_times) / len(e2e_times)
    p50_e2e = e2e_times[int(len(e2e_times) * 0.5)]
    p90_e2e = e2e_times[int(len(e2e_times) * 0.9)]
    p99_e2e = e2e_times[int(len(e2e_times) * 0.99)]
    max_e2e = max(e2e_times)

    min_time = min(min(r) for r in results)
    max_time = max(max(r) for r in results)
    dur = max_time - min_time
    assert dur > 0, "Duration must be positive"

    tokens = sum(len(tic) for tic in results)
    batch_size = len(results)

    def fmt(x: float) -> str:
        if x >= 1000:
            return f"{int(x):>6}"
        elif x >= 10:
            return f"{x:>6.2f}"
        else:
            return f"{x:>6.4f}"

    logger.info(f"Num requests: #{batch_size}, Num tokens: #{tokens}")
    logger.info(
        f"TTFT: {fmt(avg_ttft)} ms (p50: {fmt(p50_ttft)} ms, p90: {fmt(p90_ttft)} ms, "
        f"p99: {fmt(p99_ttft)} ms, max: {fmt(max_ttft)} ms)"
    )
    logger.info(
        f"TPOT: {fmt(avg_tpot)} ms (p50: {fmt(p50_tpot)} ms, p90: {fmt(p90_tpot)} ms, "
        f"p99: {fmt(p99_tpot)} ms, max: {fmt(max_tpot)} ms)"
    )
    logger.info(
        f"E2E:  {fmt(avg_e2e) } s  (p50: {fmt(p50_e2e) }  s, p90: {fmt(p90_e2e) }  s, "
        f"p99: {fmt(p99_e2e) }  s, max: {fmt(max_e2e) }  s)"
    )
    logger.info(f"Duration: {fmt(dur)} s")
    logger.info(f"Throughput: {fmt(tokens / dur)} token/s, {fmt(batch_size / dur)} req/s")

    # normalize the time to start from zero
    results = [[r - min_time for r in tics] for tics in results]

    if isinstance(tokenizer, Unset):
        return None
    return BenchmarkResult(
        raw_data=[
            BenchOneResult(
                tics=r.tics,
                input_len=r.get_input_len(tokenizer),
                output_len=r.output_len,
            )
            for r in raw_data
        ]
    )


def read_qwen_trace(
    file_path: str,
    sample_ids: List[int],
    tokenizer: Any,
    n: int | None = None,
) -> List[BenchmarkTrace]:
    class JSONInput(BaseModel):
        chat_id: int
        parent_chat_id: int
        timestamp: float
        input_length: int
        output_length: int
        type: str  # unused
        turn: int  # unused
        hash_ids: List[int]  # unused

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    avg_input_len = sum(obj.input_length for obj in objs) / len(objs)
    avg_output_len = sum(obj.output_length for obj in objs) / len(objs)
    print(f"Average input length: {avg_input_len}, Average output length: {avg_output_len}")
    # convert to trace
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=tokenizer.decode(sample_ids[: obj.input_length]),
            input_=obj.input_length,
            output=obj.output_length,
        )
        for obj in objs
    ]


def read_mooncake_trace(
    file_path: str,
    sample_ids: List[int],
    tokenizer: Any,
    n: int | None = None,
) -> List[BenchmarkTrace]:
    class JSONInput(BaseModel):
        timestamp: int
        input_length: int
        output_length: int
        hash_ids: List[int]  # unused

    with open(file_path, "r") as f:
        lines = f.readlines()
        if n is not None:
            lines = lines[:n]
    objs = [JSONInput.model_validate_json(line) for line in lines]
    # convert to trace
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,
            message=tokenizer.decode(sample_ids[: obj.input_length]),
            input_=obj.input_length,
            output=obj.output_length,
        )
        for obj in objs
    ]


def scale_traces(
    traces: List[BenchmarkTrace],
    scale: float,
) -> List[BenchmarkTrace]:
    min_tic = min(trace.timestamp for trace in traces)
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,
                message=trace.message,
                input_=trace.input_,
                output=trace.output,
            )
            for trace in traces
        ],
        key=lambda x: x.timestamp,
    )
