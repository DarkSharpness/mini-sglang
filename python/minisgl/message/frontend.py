from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    uid: int
    incremental_output: str
    finished: bool


# Gate 2.3f: Tokenizer → Frontend confirmation that the Scheduler has
# finished acting on an abort. Emitted 1:1 with an incoming AbortAckMsg
# from the Scheduler. Deliberately does NOT reuse UserReply: a UserReply
# with a synthesised token would force downstream stream consumers to
# distinguish "real token" from "cancellation stub" via heuristics.
# AbortAckReply is a distinct type so the Frontend's abort-pending state
# machine can dispatch on isinstance alone.
@dataclass
class AbortAckReply(BaseFrontendMsg):
    uid: int
