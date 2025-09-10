import torch


class PageTableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self.max_running_reqs = max_running_reqs
        self.free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        assert self.page_table.is_contiguous()

    @property
    def available_size(self) -> int:
        return len(self.free_slots)

    def allocate(self) -> int:
        return self.free_slots.pop()

    def free(self, slot: int) -> None:
        self.free_slots.append(slot)
