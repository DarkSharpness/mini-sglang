from __future__ import annotations

import torch
import multiprocessing as mp
from transformers import AutoTokenizer
from unittest.mock import MagicMock, patch

from minisgl.distributed import DistributedInfo
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg
from minisgl.scheduler import Scheduler, SchedulerConfig
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, call_if_main, init_logger
from minisgl.core import Batch, Req, SamplingParams
from minisgl.engine import ForwardOutput
from minisgl.engine.sample import BatchSamplingArgs
from minisgl.message import AbortBackendMsg
from minisgl.scheduler.scheduler import ForwardInput

logger = init_logger(__name__)


def test_overlap_loop_double_free():
    """Regression test: _free_req_resources should be called exactly once when a request
    is both aborted and finished (EOS) in the same overlap_loop iteration.

    Bug: _process_one_msg (abort path) calls _free_req_resources immediately, then
    _process_last_data (finish path) calls it again because abort doesn't add req to
    finished_reqs. After fix: abort queues req in pending_aborts, and _free_pending_aborts
    skips it since it's already in finished_reqs.
    """
    # Patch Engine, CUDA streams, tokenizer, global_ctx, and logger to avoid real hardware
    mock_global_ctx = MagicMock()
    mock_global_ctx.page_size = 1
    with (
        patch("minisgl.engine.Engine") as MockEngine,
        patch("torch.cuda.Stream") as MockCudaStream,
        patch("torch.cuda.set_stream"),
        patch("torch.cuda.stream") as mock_cuda_stream_ctx,
        patch("minisgl.scheduler.scheduler.load_tokenizer") as mock_load_tokenizer,
        patch("minisgl.core.get_global_ctx", return_value=mock_global_ctx),
        patch("minisgl.scheduler.scheduler.logger") as mock_logger,
    ):
        # Set up minimal Engine mock
        mock_engine = MagicMock()
        mock_engine.device = torch.device("cpu")
        mock_engine.stream = MagicMock()
        mock_engine.tp_cpu_group = MagicMock()
        mock_engine.page_table = torch.zeros((17, 2048), dtype=torch.int32, device="cpu")
        mock_engine.num_pages = 16
        MockEngine.return_value = mock_engine

        # Make torch.cuda.Stream() return a mock
        MockCudaStream.return_value = MagicMock()
        mock_cuda_stream_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_cuda_stream_ctx.return_value.__exit__ = MagicMock(return_value=False)

        # Set up tokenizer mock with EOS token id = 2
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_load_tokenizer.return_value = mock_tokenizer

        # Create scheduler in offline_mode (skips ZMQ setup)
        config = SchedulerConfig(
            model_path="/fake",
            tp_info=DistributedInfo(0, 1),
            dtype=torch.bfloat16,
            max_running_req=16,
            page_size=1,
            offline_mode=True,
        )
        scheduler_instance = Scheduler(config)

        # Mock receive_msg to return AbortBackendMsg for uid=42
        scheduler_instance.receive_msg = MagicMock(return_value=[AbortBackendMsg(uid=42)])
        scheduler_instance.send_result = MagicMock()

        # Create a Req with uid=42 that will finish with EOS (next_token == eos_token_id)
        req = Req(
            input_ids=torch.tensor([1, 2, 3], dtype=torch.int32),
            table_idx=0,
            cached_len=2,
            output_len=2,
            uid=42,
            sampling_params=SamplingParams(max_tokens=10),
            cache_handle=MagicMock(),
        )

        # Put req in decode_manager so abort can find it
        scheduler_instance.decode_manager.running_reqs.add(req)
        scheduler_instance.eos_token_id = 2

        # Build last_data: (ForwardInput, ForwardOutput) where req finishes with EOS
        batch = Batch(reqs=[req], phase="decode")
        next_tokens_cpu = torch.tensor([2], dtype=torch.int32)  # eos_token_id = 2
        forward_output = ForwardOutput(
            next_tokens_gpu=MagicMock(),
            next_tokens_cpu=next_tokens_cpu,
            copy_done_event=MagicMock(),
        )
        sample_args = BatchSamplingArgs(temperatures=MagicMock())
        forward_input = ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=(MagicMock(), MagicMock()),
            write_tuple=(MagicMock(), MagicMock()),
        )
        last_data = (forward_input, forward_output)

        # Mock _free_req_resources to track call count
        scheduler_instance._free_req_resources = MagicMock()

        # Run overlap_loop with the EOS-finishing req in last_data
        scheduler_instance.overlap_loop(last_data)

        # Assert exactly one call (bug state: call_count == 2)
        assert (
            scheduler_instance._free_req_resources.call_count == 1
        ), f"Expected 1 call (_free_req_resources called {scheduler_instance._free_req_resources.call_count} times = double-free bug)."


@torch.inference_mode()
def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    scheduler = Scheduler(config)
    queue.put(None)
    try:
        scheduler.run_forever()
    except KeyboardInterrupt:
        logger.info_rank0("Scheduler exiting...")


@call_if_main(__name__)
def main():
    config = SchedulerConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        max_running_req=4,
        cuda_graph_bs=[2, 4, 8],
    )

    mp.set_start_method("spawn", force=True)
    q = mp.Queue()
    p = mp.Process(target=scheduler, args=(config, q))
    p.start()
    q.get()

    send_backend = ZmqPushQueue(
        config.zmq_backend_addr,
        create=False,
        encoder=BaseBackendMsg.encoder,
    )

    recv_backend = ZmqPullQueue(
        config.zmq_detokenizer_addr,
        create=False,
        decoder=BaseTokenizerMsg.decoder,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    prompt = "What's the answer to life, the universe, and everything?"
    ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
    send_backend.put(
        UserMsg(
            uid=0,
            input_ids=ids,
            sampling_params=SamplingParams(max_tokens=100),
        )
    )

    while True:
        msg = recv_backend.get()
        assert isinstance(msg, DetokenizeMsg)
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])
        if msg.finished:
            break

    logger.info(tokenizer.decode(ids.tolist()))
    send_backend.put(ExitMsg())
