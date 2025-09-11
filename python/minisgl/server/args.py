from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.scheduler import SchedulerConfig
from minisgl.utils import cached_load_hf_config, init_logger


@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    server_host: str = "127.0.0.1"
    server_port: int = 1919
    share_tokenizer: bool = True
    _zmq_tokenizer_frontend_link: str = "ipc:///tmp/minisgl_line_3"
    _zmq_frontend_tokenizer_link: str = "ipc:///tmp/minisgl_line_4"

    @property
    def zmq_frontend_addr(self) -> str:
        return self._zmq_tokenizer_frontend_link + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            return self.zmq_detokenizer_addr
        result = self._zmq_frontend_tokenizer_link + self._unique_suffix
        assert result != self.zmq_detokenizer_addr
        return result

    @property
    def zmq_tokenizer_unique_addr(self) -> str:
        assert self.share_tokenizer, "tokenizer_addr is only valid when share_tokenizer is True"
        return self.zmq_detokenizer_addr

    @property
    def tokenizer_create_addr(self) -> bool:
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        return not self.share_tokenizer


def parse_args(args: List[str]) -> ServerArgs:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=256,
        help="The maximum number of running requests.",
    )

    # 0 represent infer the maximum sequence length from the model config
    parser.add_argument(
        "--max-seq-len-override",
        type=int,
        default=None,
        help="The maximum sequence length override. 0 means no override.",
    )

    parser.add_argument(
        "--memory-ratio",
        "--mem",
        type=float,
        default=0.9,
        help="The fraction of GPU memory to use for KV cache.",
    )

    parser.add_argument(
        "--dummy-weight", action="store_true", help="Use dummy weights for testing."
    )

    parser.add_argument(
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="The host address for the server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=1919,
        help="The port number for the server to listen on.",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)
    kwargs: Dict[str, Any] = {}

    # Convert dtype string to torch.dtype
    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if parsed_args.dtype != "auto":
        kwargs["dtype"] = DTYPE_MAP[parsed_args.dtype]
    else:
        dtype_or_str = cached_load_hf_config(parsed_args.model_path).torch_dtype
        if isinstance(dtype_or_str, str):
            kwargs["dtype"] = DTYPE_MAP[dtype_or_str]
        else:
            kwargs["dtype"] = dtype_or_str

    kwargs["tp_info"] = DistributedInfo(0, parsed_args.tensor_parallel_size)
    kwargs["model_path"] = parsed_args.model_path
    kwargs["max_running_req"] = parsed_args.max_running_requests
    kwargs["max_seq_len_override"] = parsed_args.max_seq_len_override
    kwargs["memory_ratio"] = parsed_args.memory_ratio
    kwargs["use_dummy_weight"] = parsed_args.dummy_weight
    kwargs["use_pynccl"] = parsed_args.use_pynccl
    kwargs["server_host"] = parsed_args.host
    kwargs["server_port"] = parsed_args.port
    result = ServerArgs(**kwargs)
    logger = init_logger(__name__)
    logger.info(f"Parsed arguments:\n{result}")
    return result
