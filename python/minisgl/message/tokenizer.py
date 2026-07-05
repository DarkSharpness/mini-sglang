from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int


# Gate 2.3f: Scheduler → Tokenizer ack that a previously-received
# AbortBackendMsg has been fully honoured (resources released, or uid
# confirmed unknown). Deliberately carries only the uid — no token, no
# text — so the Frontend cleanup path never has to synthesise a "fake"
# stop token to represent cancellation. Idempotent: emitted at most once
# per abort in the Scheduler code path, but the Frontend handler is
# expected to tolerate duplicates in case the same uid receives a
# duplicate AbortBackendMsg from a re-tried caller.
@dataclass
class AbortAckMsg(BaseTokenizerMsg):
    uid: int
