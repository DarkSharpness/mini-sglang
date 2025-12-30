from __future__ import annotations
import torch
from minisgl.core import SamplingParams
from minisgl.message.backend import BatchBackendMsg, UserMsg, ExitMsg


def test_backend_msg_roundtrip():
    t = torch.tensor([1, 2, 3], dtype=torch.int32)

    # Test UserMsg
    u_msg = UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())
    u_encoded = u_msg.encoder()
    u_decoded = UserMsg.decoder(u_encoded)

    assert isinstance(u_decoded, UserMsg)
    assert u_decoded.uid == u_msg.uid
    assert torch.equal(u_decoded.input_ids, u_msg.input_ids)
    assert u_decoded.sampling_params == u_msg.sampling_params

    # Test BatchBackendMsg
    b_msg = BatchBackendMsg(data=[u_msg])
    b_encoded = b_msg.encoder()
    b_decoded = BatchBackendMsg.decoder(b_encoded)

    assert isinstance(b_decoded, BatchBackendMsg)
    assert len(b_decoded.data) == 1
    decoded_inner_msg = b_decoded.data[0]
    assert isinstance(decoded_inner_msg, UserMsg)
    assert decoded_inner_msg.uid == u_msg.uid
    assert torch.equal(decoded_inner_msg.input_ids, u_msg.input_ids)
    assert decoded_inner_msg.sampling_params == u_msg.sampling_params

    # Test ExitMsg
    e_msg = ExitMsg()
    e_encoded = e_msg.encoder()
    e_decoded = ExitMsg.decoder(e_encoded)
    assert isinstance(e_decoded, ExitMsg)
