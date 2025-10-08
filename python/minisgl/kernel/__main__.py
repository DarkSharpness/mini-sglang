from minisgl.utils import init_logger

from .indexing import _load_index_module
from .pynccl import _load_pynccl_module
from .store import _load_kvcache_module
from .topk import _load_topk_module

assert __name__ == "__main__"

logger = init_logger(__name__, "kernel-compiler")

# compile these modules
for func in [
    _load_pynccl_module,
    _load_index_module,
    _load_kvcache_module,
    _load_topk_module,
]:
    logger.info(f"Compiling {func.__name__} ...")
    func()
