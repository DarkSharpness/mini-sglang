from __future__ import annotations

import torch

from minisgl.core import SamplingParams
from minisgl.message.backend import BatchBackendMsg, ExitMsg, UserMsg


class TestBatchBackendMsg:
    def test_encoder_structure(self):
        t = torch.tensor([1], dtype=torch.int32)
        u_msg = UserMsg(uid=1, input_ids=t, sampling_params=SamplingParams())
        b_msg = BatchBackendMsg(data=[u_msg])

        encoded = b_msg.encoder()

        assert encoded["__type__"] == "BatchBackendMsg"
        assert isinstance(encoded["data"], list)
        assert len(encoded["data"]) == 1
        assert encoded["data"][0]["__type__"] == "UserMsg"
        assert encoded["data"][0]["uid"] == 1

    def test_decoder_reconstruction(self):
        serialized_inner_msg = {
            "__type__": "UserMsg",
            "uid": 99,
            "input_ids": {
                "__type__": "Tensor",
                "buffer": torch.tensor([5], dtype=torch.int32).numpy().tobytes(),
                "dtype": "torch.int32",
            },
            "sampling_params": {"__type__": "SamplingParams"},
        }
        input_data = {"__type__": "BatchBackendMsg", "data": [serialized_inner_msg]}

        decoded = BatchBackendMsg.decoder(input_data)

        assert isinstance(decoded, BatchBackendMsg)
        assert len(decoded.data) == 1
        assert isinstance(decoded.data[0], UserMsg)
        assert decoded.data[0].uid == 99


class TestExitMsg:
    def test_encoder_structure(self):
        msg = ExitMsg()
        encoded = msg.encoder()
        assert encoded["__type__"] == "ExitMsg"

    def test_decoder_reconstruction(self):
        input_data = {"__type__": "ExitMsg"}
        decoded = ExitMsg.decoder(input_data)
        assert isinstance(decoded, ExitMsg)


class TestUserMsg:
    def test_encoder_structure(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        params = SamplingParams(max_tokens=100)
        u_msg = UserMsg(uid=10, input_ids=t, sampling_params=params)

        encoded = u_msg.encoder()

        # Verify the dictionary structure
        assert isinstance(encoded, dict)
        assert encoded["__type__"] == "UserMsg"
        assert encoded["uid"] == 10

        # Verify Tensor serialization
        assert encoded["input_ids"]["__type__"] == "Tensor"
        assert encoded["input_ids"]["dtype"] == "torch.int32"
        assert encoded["input_ids"]["buffer"] == t.numpy().tobytes()

        # Verify SamplingParams serialization
        assert encoded["sampling_params"]["__type__"] == "SamplingParams"
        assert encoded["sampling_params"]["max_tokens"] == 100

    def test_decoder_reconstruction(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int32)
        serialized_tensor = {
            "__type__": "Tensor",
            "buffer": t.numpy().tobytes(),
            "dtype": "torch.int32",
        }
        serialized_params = {"__type__": "SamplingParams", "max_tokens": 50, "temperature": 0.7}
        input_data = {
            "__type__": "UserMsg",
            "uid": 42,
            "input_ids": serialized_tensor,
            "sampling_params": serialized_params,
        }

        decoded = UserMsg.decoder(input_data)

        assert isinstance(decoded, UserMsg)
        assert decoded.uid == 42
        assert torch.equal(decoded.input_ids, t)
        assert isinstance(decoded.sampling_params, SamplingParams)
        assert decoded.sampling_params.max_tokens == 50
        assert decoded.sampling_params.temperature == 0.7
