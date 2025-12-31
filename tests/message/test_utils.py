from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
from minisgl.message.utils import serialize_type, deserialize_type


@dataclass
class A:
    x: int
    y: str
    z: List["A"]
    w: torch.Tensor


def test_serialize_type_structure():
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    x = A(10, "hello", [A(20, "world", [], t)], t)
    data = serialize_type(x)

    assert isinstance(data, dict)
    assert data["__type__"] == "A"
    assert data["x"] == 10
    assert data["y"] == "hello"
    assert isinstance(data["z"], list)
    assert len(data["z"]) == 1
    inner_data = data["z"][0]
    assert isinstance(inner_data, dict)
    assert inner_data["__type__"] == "A"
    assert inner_data["x"] == 20
    assert inner_data["y"] == "world"
    assert "w" in data
    assert isinstance(data["w"]["buffer"], bytes)
    assert data["w"]["__type__"] == "Tensor"
    assert data["w"]["dtype"] == "torch.int32"


def test_deserialize_type_reconstruction():
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    serialized_t = {
        "__type__": "Tensor",
        "buffer": t.numpy().tobytes(),
        "dtype": "torch.int32",
    }

    data = {
        "__type__": "A",
        "x": 10,
        "y": "hello",
        "z": [{"__type__": "A", "x": 20, "y": "world", "z": [], "w": serialized_t}],
        "w": serialized_t,
    }

    y = deserialize_type({"A": A}, data)

    assert isinstance(y, A)
    assert y.x == 10
    assert y.y == "hello"
    assert torch.equal(y.w, t)
    assert len(y.z) == 1
    assert isinstance(y.z[0], A)
    assert y.z[0].x == 20
    assert y.z[0].y == "world"
    assert len(y.z[0].z) == 0
    assert torch.equal(y.z[0].w, t)
