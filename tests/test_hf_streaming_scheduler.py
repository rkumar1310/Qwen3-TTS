from types import SimpleNamespace

from transformers.generation.continuous_batching.requests import (
    RequestState,
    RequestStatus,
)

from qwen_tts.streaming.hf_scheduler import AppendableTextFIFOScheduler
from qwen_tts.streaming.requests import StreamingRequestRegistry


class FakePagedCache:
    block_size = 4
    num_blocks = 64
    max_blocks_per_request = 16
    num_full_attention_groups = 1
    allow_block_sharing = False
    use_prefix_sharing = False
    config = SimpleNamespace(sliding_window=None)

    def __init__(self):
        self.free_blocks = self.num_blocks

    def get_num_free_blocks(self):
        return self.free_blocks

    def allocate_blocks(self, blocks, _request_id, _allocated_blocks):
        if blocks > self.free_blocks:
            return None
        self.free_blocks -= blocks
        return blocks

    def blocks_needed(self, blocks, _allocated_blocks):
        return blocks


def decoding_request(request_id: str) -> RequestState:
    state = RequestState(request_id=request_id, initial_tokens=[1])
    state.status = RequestStatus.DECODING
    state.tokens_to_process = [7]
    state.position_offset = 1
    state.allocated_blocks = 1
    return state


def test_paused_request_does_not_block_ready_peer():
    registry = StreamingRequestRegistry()
    registry.create("paused")
    registry.create("ready", [101])
    scheduler = AppendableTextFIFOScheduler(
        FakePagedCache(),
        safety_margin=0,
        max_requests_per_batch=8,
        registry=registry,
    )
    scheduler.active_requests = {
        "paused": decoding_request("paused"),
        "ready": decoding_request("ready"),
    }

    scheduled, *_ = scheduler.schedule_batch(token_budget=8, cache_budget=128)

    assert [future.state.request_id for future in scheduled] == ["ready"]


def test_appending_text_rejoins_same_active_request():
    registry = StreamingRequestRegistry()
    registry.create("turn-1")
    scheduler = AppendableTextFIFOScheduler(
        FakePagedCache(),
        safety_margin=0,
        max_requests_per_batch=8,
        registry=registry,
    )
    state = decoding_request("turn-1")
    scheduler.active_requests = {"turn-1": state}

    first_batch, *_ = scheduler.schedule_batch(token_budget=8, cache_budget=128)
    registry.append("turn-1", [101])
    second_batch, *_ = scheduler.schedule_batch(token_budget=8, cache_budget=128)

    assert first_batch == []
    assert second_batch[0].state is state
