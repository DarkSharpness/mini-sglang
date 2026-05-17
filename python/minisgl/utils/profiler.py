from __future__ import annotations

import gzip
import os
import shutil
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
        self.out_dir = os.path.abspath(os.path.expanduser(self.out_dir))
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

    def _get_tp_rank(self) -> int:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return 0
        return int(torch.distributed.get_rank())

    def stop_and_export(self) -> str | None:
        if self._prof is None:
            return None

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._prof.stop()

        # Ensure output directory still exists at export time.
        os.makedirs(self.out_dir, exist_ok=True)

        ts_ms = int((self._start_ts or time.time()) * 1e3)
        tp_rank = self._get_tp_rank()
        pid = os.getpid()
        json_path = os.path.join(
            self.out_dir, f"minisgl-profile-uid{self.uid}-tp{tp_rank}-pid{pid}-{ts_ms}.json"
        )
        self._prof.export_chrome_trace(json_path)
        if not os.path.exists(json_path):
            retry_json_path = os.path.join(
                self.out_dir, f"minisgl-profile-uid{self.uid}-tp{tp_rank}-pid{pid}-{ts_ms}-r1.json"
            )
            self._prof.export_chrome_trace(retry_json_path)
            json_path = retry_json_path

        self._prof = None
        if not os.path.exists(json_path):
            return None

        gz_path = json_path + ".gz"
        with open(json_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(json_path)
        return gz_path if os.path.exists(gz_path) else None
