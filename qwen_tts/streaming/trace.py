"""Request-scoped trace events for the continuous Qwen speech pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class StreamingTraceEvent:
    request_id: str
    event: str
    at: float = field(default_factory=time.perf_counter)
    data: dict[str, Any] = field(default_factory=dict)


TraceCallback = Callable[[StreamingTraceEvent], None]

