"""Incremental text tokenization without changing tokens already sent to Qwen."""

from __future__ import annotations

from collections.abc import Callable, Sequence


class TokenizationChangedError(RuntimeError):
    """Raised when a later text delta rewrites a token already sent to Qwen."""


class StableTextTokenizer:
    """Turn arbitrary text deltas into an append-only token stream.

    Tokenizers can merge the end of the previous delta with the beginning of the
    next one.  The newest tokens therefore stay private until a later delta
    confirms them.  ``finish`` commits the remaining tail.
    """

    def __init__(
        self,
        tokenize: Callable[[str], Sequence[int]],
        *,
        hold_back_tokens: int = 2,
    ) -> None:
        if hold_back_tokens < 1:
            raise ValueError("hold_back_tokens must be at least 1")
        self._tokenize = tokenize
        self._hold_back_tokens = hold_back_tokens
        self._text = ""
        self._tokens: list[int] = []
        self._committed = 0
        self._finished = False

    @property
    def text(self) -> str:
        return self._text

    @property
    def committed_count(self) -> int:
        return self._committed

    @property
    def finished(self) -> bool:
        return self._finished

    def append(self, delta: str) -> list[int]:
        if self._finished:
            raise RuntimeError("cannot append text after finish")
        if not delta:
            return []
        return self._retokenize(self._text + delta, final=False)

    def finish(self) -> list[int]:
        if self._finished:
            return []
        emitted = self._retokenize(self._text, final=True)
        self._finished = True
        return emitted

    def _retokenize(self, text: str, *, final: bool) -> list[int]:
        new_tokens = [int(token) for token in self._tokenize(text)]
        committed_prefix = self._tokens[: self._committed]
        if new_tokens[: self._committed] != committed_prefix:
            raise TokenizationChangedError(
                "the tokenizer rewrote text tokens that were already sent to Qwen"
            )

        self._text = text
        self._tokens = new_tokens
        safe_length = len(new_tokens) if final else max(
            self._committed,
            len(new_tokens) - self._hold_back_tokens,
        )
        emitted = new_tokens[self._committed : safe_length]
        self._committed = safe_length
        return emitted
