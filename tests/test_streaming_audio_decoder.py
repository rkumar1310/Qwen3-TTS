import torch

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


def tiny_decoder() -> Qwen3TTSTokenizerV2Decoder:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=16,
        codebook_dim=8,
        hidden_size=8,
        latent_dim=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=16,
        num_hidden_layers=1,
        num_quantizers=2,
        sliding_window=4,
        upsample_rates=(2, 2),
        upsampling_ratios=(2,),
        decoder_dim=16,
    )
    return Qwen3TTSTokenizerV2Decoder(config).eval()


def test_stateful_decoder_emits_exactly_one_audio_frame_per_codec_frame():
    decoder = tiny_decoder()
    cache = {}

    with torch.no_grad():
        first = decoder(torch.randint(0, 16, (1, 2, 1)), cache)
        second = decoder(torch.randint(0, 16, (1, 2, 1)), cache)

    assert first.shape == (1, 1, decoder.total_upsample)
    assert second.shape == (1, 1, decoder.total_upsample)
    assert cache["suffix_frames"] == 2
    assert "past_key_values" in cache


def test_stateful_decoder_batches_independent_requests_without_sharing_state():
    decoder = tiny_decoder()
    caches = [{}, {}]

    with torch.no_grad():
        first = decoder.batched_chunked_decode(
            torch.randint(0, 16, (2, 2, 1)), [1, 1], caches
        )
        second = decoder.batched_chunked_decode(
            torch.randint(0, 16, (2, 2, 1)), [1, 1], caches
        )

    assert [tuple(wave.shape) for wave in first] == [(1, 1, 8), (1, 1, 8)]
    assert [tuple(wave.shape) for wave in second] == [(1, 1, 8), (1, 1, 8)]
    assert [cache["suffix_frames"] for cache in caches] == [2, 2]
    assert caches[0]["suffix_quantized"].data_ptr() != caches[1]["suffix_quantized"].data_ptr()
