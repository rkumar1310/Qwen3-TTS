from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig


def test_talker_config_preserves_optional_pad_token_id() -> None:
    config = Qwen3TTSTalkerConfig()

    assert hasattr(config, "pad_token_id")
    assert config.pad_token_id is None


def test_talker_config_accepts_checkpoint_pad_token_id() -> None:
    config = Qwen3TTSTalkerConfig(pad_token_id=7)

    assert config.pad_token_id == 7
