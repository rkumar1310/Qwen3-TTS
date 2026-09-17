import pytest

from qwen_tts.streaming.requests import (
    StreamConditionKind,
    StreamingRequestRegistry,
)


def test_open_request_pauses_and_resumes_without_losing_identity():
    registry = StreamingRequestRegistry()
    registry.create("turn-1", [11])

    assert registry.can_decode("turn-1")
    assert registry.take_condition("turn-1").token_id == 11
    assert not registry.can_decode("turn-1")

    registry.append("turn-1", [12, 13])
    assert registry.can_decode("turn-1")
    assert registry.take_condition("turn-1").token_id == 12
    assert registry.take_condition("turn-1").token_id == 13
    assert not registry.can_decode("turn-1")


def test_closed_request_sends_end_once_then_padding_until_acoustic_eos():
    registry = StreamingRequestRegistry()
    registry.create("turn-1")
    registry.close_input("turn-1", [21])

    assert registry.take_condition("turn-1").kind == StreamConditionKind.TEXT
    assert registry.take_condition("turn-1").kind == StreamConditionKind.TEXT_END
    assert registry.take_condition("turn-1").kind == StreamConditionKind.PADDING
    assert registry.can_decode("turn-1")


def test_cancelled_request_never_becomes_ready_again():
    registry = StreamingRequestRegistry()
    registry.create("turn-1", [11])
    registry.cancel("turn-1")

    assert not registry.can_decode("turn-1")
    with pytest.raises(RuntimeError, match="cancelled"):
        registry.append("turn-1", [12])
