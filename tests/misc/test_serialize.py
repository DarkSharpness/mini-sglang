from __future__ import annotations
from dataclasses import dataclass
from typing import List

from minisgl.core import SamplingParams
import torch
from minisgl.message import BatchBackendMsg, UserMsg
from minisgl.message.utils import serialize_type, deserialize_type
from minisgl.utils import call_if_main, init_logger

logger = init_logger(__name__)


@dataclass
class A:
    x: int
    y: str
    z: List[A]
    w: torch.Tensor


@call_if_main()
def test_serialize_deserialize():

    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)
    logger.info(data)
    y = deserialize_type({"A": A}, data)
    logger.info(y)

    u = BatchBackendMsg([UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())])
    result = u.decoder(u.encoder())
    logger.info(u)
    logger.info(result)
