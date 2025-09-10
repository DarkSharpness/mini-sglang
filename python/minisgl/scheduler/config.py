from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine.config import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    mix_decode: bool = False

    # networking config
    unique_suffix: str = field(default_factory=_get_pid_suffix)

    zmq_tokenizer_backend_line: str = "ipc:///tmp/minisgl_line_0"
    zmq_backend_tokenizer_line: str = "ipc:///tmp/minisgl_line_1"
    zmq_scheduler_broadcast_line: str = "ipc:///tmp/minisgl_line_2"

    @property
    def zmq_tokenizer_backend_addr(self) -> str:
        return self.zmq_tokenizer_backend_line + self.unique_suffix

    @property
    def zmq_backend_tokenizer_addr(self) -> str:
        return self.zmq_backend_tokenizer_line + self.unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return self.zmq_scheduler_broadcast_line + self.unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens
