from qwen_tts.streaming.text import StableTextTokenizer, TokenizationChangedError


def character_tokenizer(text: str) -> list[int]:
    return [ord(character) for character in text]


def test_holds_back_unstable_tail_until_more_text_arrives():
    tokenizer = StableTextTokenizer(character_tokenizer, hold_back_tokens=2)

    assert tokenizer.append("hello") == [ord("h"), ord("e"), ord("l")]
    assert tokenizer.append(" world") == [
        ord("l"),
        ord("o"),
        ord(" "),
        ord("w"),
        ord("o"),
        ord("r"),
    ]
    assert tokenizer.finish() == [ord("l"), ord("d")]
    assert tokenizer.finish() == []


def test_detects_rewrite_of_already_committed_tokens():
    calls = iter(([1, 2, 3, 4], [1, 9, 3, 4, 5]))
    tokenizer = StableTextTokenizer(lambda _: next(calls), hold_back_tokens=2)

    assert tokenizer.append("first") == [1, 2]
    try:
        tokenizer.append(" second")
    except TokenizationChangedError:
        pass
    else:
        raise AssertionError("expected committed-token rewrite to fail")
