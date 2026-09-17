from threading import RLock

import torch
from torch import nn

from qwen_tts.streaming.engine import (
    Qwen3TTSContinuousEngine,
    _configure_audio_decoder_dtype,
)


class _FakeQwenModel:
    @staticmethod
    def _build_assistant_text(text: str) -> str:
        return text

    @staticmethod
    def _build_instruct_text(text: str) -> str:
        return text

    @staticmethod
    def _tokenize_texts(_texts: list[str]) -> list[torch.Tensor]:
        return [torch.arange(8).reshape(1, 8)]


class _FakeAudioDecoder:
    def __init__(self) -> None:
        self.created: list[str] = []

    def create_request(self, request_id: str) -> None:
        self.created.append(request_id)


def test_audio_decoder_uses_fp32_without_changing_other_modules() -> None:
    talker = nn.Linear(2, 2).to(dtype=torch.bfloat16)
    decoder = nn.Linear(2, 2).to(dtype=torch.bfloat16)

    configured = _configure_audio_decoder_dtype(decoder, torch.float32)

    assert configured is decoder
    assert decoder.weight.dtype == torch.float32
    assert talker.weight.dtype == torch.bfloat16


def test_create_request_initializes_matching_audio_state() -> None:
    engine = Qwen3TTSContinuousEngine.__new__(Qwen3TTSContinuousEngine)
    engine.qwen_model = _FakeQwenModel()
    engine.audio_decoder = _FakeAudioDecoder()
    engine._inputs = {}
    engine._lock = RLock()

    engine.create_request("turn-1")

    assert engine.audio_decoder.created == ["turn-1"]
    assert engine.is_submitted("turn-1") is False
