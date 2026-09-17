import time

import pytest
import torch


pytest.importorskip("librosa")


def test_real_talker_runs_through_appendable_paged_batching_on_cpu(monkeypatch):
    from transformers import ContinuousBatchingConfig, GenerationConfig
    import transformers.generation.continuous_batching.input_outputs as cb_io

    from qwen_tts.core.models.configuration_qwen3_tts import (
        Qwen3TTSTalkerCodePredictorConfig,
        Qwen3TTSTalkerConfig,
    )
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSTalkerForConditionalGeneration,
    )
    from qwen_tts.streaming.hf_manager import AppendableContinuousBatchingManager
    from qwen_tts.streaming.model import (
        PreparedStreamingPrompt,
        QwenStreamingTalkerAdapter,
    )
    from qwen_tts.streaming.requests import StreamingRequestRegistry

    # macOS reports MPS alongside CPU, which makes Transformers request pinned
    # CPU memory even though this test intentionally runs on CPU.
    monkeypatch.setattr(cb_io, "get_available_devices", lambda: ["cpu"])
    torch.set_default_device("cpu")

    predictor_config = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_code_groups=3,
        pad_token_id=0,
    )
    talker_config = Qwen3TTSTalkerConfig(
        code_predictor_config=predictor_config,
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        pad_token_id=0,
        text_hidden_size=8,
        text_vocab_size=64,
        num_code_groups=3,
        codec_eos_token_id=31,
        rope_scaling={
            "rope_type": "default",
            "mrope_section": [1, 1, 0],
            "interleaved": False,
        },
    )
    talker = Qwen3TTSTalkerForConditionalGeneration(talker_config).cpu().eval()
    registry = StreamingRequestRegistry()
    frames = []
    adapter = QwenStreamingTalkerAdapter(
        talker,
        registry,
        subtalker_dosample=False,
        frame_callback=frames.append,
    )
    manager = AppendableContinuousBatchingManager(
        adapter,
        GenerationConfig(max_new_tokens=2, do_sample=False, eos_token_id=-1),
        ContinuousBatchingConfig(
            max_requests_per_batch=2,
            max_batch_tokens=16,
            max_blocks_per_request=2,
            max_memory_percent=0.2,
            use_cuda_graph=False,
            scheduler_type="fifo",
        ),
        registry=registry,
    )
    adapter.register_session(
        "a",
        PreparedStreamingPrompt(
            embeddings=torch.randn(1, 2, 8),
            text_end_embedding=torch.randn(1, 1, 8),
            text_padding_embedding=torch.randn(1, 1, 8),
        ),
    )

    manager.start()
    try:
        manager.add_streaming_request(
            [0, 0],
            request_id="a",
            remaining_text_tokens=[4, 5],
            max_new_tokens=2,
            eos_token_id=-1,
        )
        manager.close_text_input("a")
        final = None
        deadline = time.time() + 10
        while time.time() < deadline:
            result = manager.get_result("a", timeout=0.5)
            if result is not None and result.is_finished():
                final = result
                break
        assert final is not None
        assert final.error is None
        assert len(frames) == 1
    finally:
        manager.stop(block=True, hard_stop=final is None)

