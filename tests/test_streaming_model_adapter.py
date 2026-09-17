from types import SimpleNamespace

import torch
from torch import nn

from qwen_tts.streaming.model import (
    BatchRequestContext,
    PreparedStreamingPrompt,
    QwenStreamingTalkerAdapter,
)
from qwen_tts.streaming.requests import StreamingRequestRegistry


class FakeCodePredictor(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(16, hidden_size) for _ in range(2)])
        self.last_batch_size = 0

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        generation_steps=None,
        **_kwargs,
    ):
        batch = inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
        self.last_batch_size = batch
        step = 0 if generation_steps is None else generation_steps
        logits = torch.full((batch, 1, 16), -100.0)
        logits[:, :, 2 + step] = 100.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=None,
            generation_steps=step + 1,
        )


class FakeMainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = {}

    def forward(self, *, inputs_embeds, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(last_hidden_state=inputs_embeds + 1)


class FakeTalker(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 4
        self.config = SimpleNamespace(num_code_groups=3)
        self.codec_embeddings = nn.Embedding(16, hidden_size)
        self.text_embeddings = nn.Embedding(32, hidden_size)
        self.text_projection = nn.Identity()
        self.code_predictor = FakeCodePredictor(hidden_size)
        self.model = FakeMainModel()
        self.codec_head = nn.Linear(hidden_size, 16, bias=False)

    @property
    def device(self):
        return self.codec_embeddings.weight.device

    @property
    def dtype(self):
        return self.codec_embeddings.weight.dtype

    def get_input_embeddings(self):
        return self.codec_embeddings

    def get_text_embeddings(self):
        return self.text_embeddings


def prompt(seed: float) -> PreparedStreamingPrompt:
    return PreparedStreamingPrompt(
        embeddings=torch.full((1, 2, 4), seed, dtype=torch.float32),
        text_end_embedding=torch.full((1, 1, 4), seed + 10, dtype=torch.float32),
        text_padding_embedding=torch.full((1, 1, 4), seed + 20, dtype=torch.float32),
    )


def test_prefill_then_batches_secondary_codebooks_for_two_live_requests():
    registry = StreamingRequestRegistry()
    registry.create("a", [20])
    registry.create("b", [21])
    talker = FakeTalker()
    adapter = QwenStreamingTalkerAdapter(talker, registry)
    adapter.register_session("a", prompt(1))
    adapter.register_session("b", prompt(2))

    adapter.set_batch_context(
        [
            BatchRequestContext("a", query_length=2, past_length=0),
            BatchRequestContext("b", query_length=2, past_length=0),
        ]
    )
    adapter(input_ids=torch.zeros((1, 4), dtype=torch.int32))

    adapter.set_batch_context(
        [
            BatchRequestContext("a", query_length=1, past_length=2),
            BatchRequestContext("b", query_length=1, past_length=2),
        ]
    )
    output = adapter(input_ids=torch.tensor([[4, 5]], dtype=torch.int32))
    frames = adapter.pop_codec_frames()

    assert talker.code_predictor.last_batch_size == 2
    assert output.logits.shape == (1, 2, 16)
    assert "block_table" in talker.model.last_kwargs
    assert talker.model.last_kwargs["block_table"] is None
    assert [frame.request_id for frame in frames] == ["a", "b"]
    assert frames[0].codes.tolist() == [4, 2, 3]
    assert frames[1].codes.tolist() == [5, 2, 3]
    assert not registry.can_decode("a")
    assert not registry.can_decode("b")
