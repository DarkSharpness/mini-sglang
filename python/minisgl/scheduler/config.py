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
    cache_type: str = "radix"
    offline_mode: bool = False

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    _zmq_backend_link: str = "ipc:///tmp/minisgl_line_0"
    _zmq_detokenizer_link: str = "ipc:///tmp/minisgl_line_1"
    _zmq_broadcast_link: str = "ipc:///tmp/minisgl_line_2"

    @property
    def zmq_backend_addr(self) -> str:
        return self._zmq_backend_link + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return self._zmq_detokenizer_link + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return self._zmq_broadcast_link + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
