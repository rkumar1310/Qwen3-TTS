"""Thread-safe lifecycle state for appendable Qwen3-TTS requests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock


class RequestAlreadyExistsError(ValueError):
    pass


class RequestNotFoundError(KeyError):
    pass


class StreamConditionKind(str, Enum):
    TEXT = "text"
    TEXT_END = "text_end"
    PADDING = "padding"


@dataclass(frozen=True)
class StreamCondition:
    kind: StreamConditionKind
    token_id: int | None = None


@dataclass
class _StreamingRequest:
    request_id: str
    pending_text_tokens: deque[int] = field(default_factory=deque)
    input_closed: bool = False
    text_end_sent: bool = False
    cancelled: bool = False


class StreamingRequestRegistry:
    """Own the text side of each live speech request.

    The Hugging Face scheduler calls :meth:`can_decode` before selecting an
    active request.  An open request pauses when its text queue is empty.  Once
    closed, it receives one text-end condition and then padding until the
    acoustic model emits its own EOS token.
    """

    def __init__(self) -> None:
        self._requests: dict[str, _StreamingRequest] = {}
        self._lock = RLock()

    def create(self, request_id: str, remaining_text_tokens: list[int] | None = None) -> None:
        with self._lock:
            if request_id in self._requests:
                raise RequestAlreadyExistsError(request_id)
            request = _StreamingRequest(request_id=request_id)
            request.pending_text_tokens.extend(remaining_text_tokens or [])
            self._requests[request_id] = request

    def append(self, request_id: str, token_ids: list[int]) -> None:
        if not token_ids:
            return
        with self._lock:
            request = self._get(request_id)
            if request.cancelled:
                raise RuntimeError(f"request {request_id!r} was cancelled")
            if request.input_closed:
                raise RuntimeError(f"request {request_id!r} input is closed")
            request.pending_text_tokens.extend(int(token) for token in token_ids)

    def close_input(self, request_id: str, final_token_ids: list[int] | None = None) -> None:
        with self._lock:
            request = self._get(request_id)
            if request.cancelled:
                return
            request.pending_text_tokens.extend(int(token) for token in (final_token_ids or []))
            request.input_closed = True

    def cancel(self, request_id: str) -> None:
        with self._lock:
            request = self._get(request_id)
            request.cancelled = True
            request.pending_text_tokens.clear()

    def remove(self, request_id: str) -> None:
        with self._lock:
            self._requests.pop(request_id, None)

    def can_decode(self, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.cancelled:
                return False
            return bool(request.pending_text_tokens) or request.input_closed

    def take_condition(self, request_id: str) -> StreamCondition:
        with self._lock:
            request = self._get(request_id)
            if request.cancelled:
                raise RuntimeError(f"request {request_id!r} was cancelled")
            if request.pending_text_tokens:
                return StreamCondition(
                    kind=StreamConditionKind.TEXT,
                    token_id=request.pending_text_tokens.popleft(),
                )
            if not request.input_closed:
                raise RuntimeError(f"request {request_id!r} is waiting for text")
            if not request.text_end_sent:
                request.text_end_sent = True
                return StreamCondition(kind=StreamConditionKind.TEXT_END)
            return StreamCondition(kind=StreamConditionKind.PADDING)

    def _get(self, request_id: str) -> _StreamingRequest:
        request = self._requests.get(request_id)
        if request is None:
            raise RequestNotFoundError(request_id)
        return request
