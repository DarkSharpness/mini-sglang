from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch


@dataclass
class RequestProfiler:
    uid: int
    out_dir: str = "/tmp"
    enabled: bool = True

    def __post_init__(self) -> None:
        self._prof: torch.profiler.profile | None = None
        self._start_ts: float | None = None
        os.makedirs(self.out_dir, exist_ok=True)

    def start(self) -> None:
        if not self.enabled or self._prof is not None:
            return

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._prof = torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
        )
        self._start_ts = time.time()
        self._prof.start()

    def stop_and_export(self) -> str | None:
        if self._prof is None:
            return None

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._prof.stop()

        ts_ms = int((self._start_ts or time.time()) * 1e3)
        path = os.path.join(self.out_dir, f"minisgl-profile-uid{self.uid}-{ts_ms}.json")
        self._prof.export_chrome_trace(path)
        self._prof = None
        return path
