from __future__ import annotations

import pathlib
from typing import NamedTuple

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"
