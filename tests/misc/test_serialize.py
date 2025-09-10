from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
from minisgl.message import BatchBackendMsg, UserMsg
from minisgl.message.utils import serialize_type, deserialize_type
from minisgl.utils import call_if_main


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
    print(data)
    y = deserialize_type({"A": A}, data)
    print(y)

    u = BatchBackendMsg([UserMsg(uid=0, output_len=10, input_ids=t)])
    result = u.decoder(u.encoder())
    print(u)
    print(result)
