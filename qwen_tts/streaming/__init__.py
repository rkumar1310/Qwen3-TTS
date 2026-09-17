"""State and scheduling primitives for true incremental Qwen3-TTS input."""

from importlib import import_module
from typing import TYPE_CHECKING

from .requests import (
    RequestAlreadyExistsError,
    RequestNotFoundError,
    StreamCondition,
    StreamConditionKind,
    StreamingRequestRegistry,
)
from .text import StableTextTokenizer, TokenizationChangedError

if TYPE_CHECKING:
    from .audio import GeneratedAudioChunk, QwenStreamingAudioDecoder
    from .engine import Qwen3TTSContinuousEngine
    from .model import GeneratedCodecFrame
    from .trace import StreamingTraceEvent

__all__ = [
    "RequestAlreadyExistsError",
    "RequestNotFoundError",
    "GeneratedAudioChunk",
    "GeneratedCodecFrame",
    "Qwen3TTSContinuousEngine",
    "QwenStreamingAudioDecoder",
    "StableTextTokenizer",
    "StreamCondition",
    "StreamConditionKind",
    "StreamingRequestRegistry",
    "StreamingTraceEvent",
    "TokenizationChangedError",
]


def __getattr__(name: str):
    if name in {"GeneratedAudioChunk", "QwenStreamingAudioDecoder"}:
        return getattr(import_module(".audio", __name__), name)
    if name == "Qwen3TTSContinuousEngine":
        return getattr(import_module(".engine", __name__), name)
    if name == "GeneratedCodecFrame":
        return getattr(import_module(".model", __name__), name)
    if name == "StreamingTraceEvent":
        return getattr(import_module(".trace", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
