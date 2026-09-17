import torch

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


def _make_small_decoder() -> Qwen3TTSTokenizerV2Decoder:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    return Qwen3TTSTokenizerV2Decoder(config).eval()


def test_incremental_audio_matches_the_same_region_of_full_decode() -> None:
    torch.manual_seed(3)
    decoder = _make_small_decoder()
    decoder._incremental_chunk_frames = 25
    frames = 53
    codes = torch.randint(
        0,
        decoder.config.codebook_size,
        (1, decoder.config.num_quantizers, frames),
    )
    cache: dict = {}

    with torch.no_grad():
        first = decoder(codes[..., :4], cache=cache)
        second = decoder(codes[..., 4:12], cache=cache)
        third = decoder(codes[..., 12:28], cache=cache)
        fourth = decoder(codes[..., 28:53], cache=cache)
        incremental = torch.cat([first, second, third, fourth], dim=-1)
        full = decoder._forward_exact(codes)

    assert incremental.shape == full.shape
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-4)
