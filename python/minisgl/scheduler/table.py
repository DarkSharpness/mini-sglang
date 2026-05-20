from typing import Callable

import torch


class TableManager:
    def __init__(
        self,
        max_running_reqs: int,
        page_table: torch.Tensor,
        on_slot_allocated: Callable[[int], None] | None = None,
    ) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self._on_slot_allocated = on_slot_allocated
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        slot = self._free_slots.pop()
        if self._on_slot_allocated is not None:
            self._on_slot_allocated(slot)
        return slot

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
