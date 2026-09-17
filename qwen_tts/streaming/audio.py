"""Stateful codec-to-PCM decoding for live Qwen3-TTS requests."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Callable

import torch

from .model import GeneratedCodecFrame
from .trace import StreamingTraceEvent, TraceCallback


@dataclass(frozen=True)
class GeneratedAudioChunk:
    request_id: str
    pcm: torch.Tensor
    sample_rate: int
    generated_at: float
    decoded_at: float


class QwenStreamingAudioDecoder:
    """Micro-batch live codec frames while preserving per-request vocoder state."""

    def __init__(
        self,
        decoder,
        *,
        sample_rate: int,
        chunk_callback: Callable[[GeneratedAudioChunk], None],
        request_finished_callback: Callable[[str], None] | None = None,
        error_callback: Callable[[str, Exception], None] | None = None,
        max_batch_size: int = 16,
        microbatch_wait_ms: float = 1.0,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        self.decoder = decoder
        self.sample_rate = int(sample_rate)
        self.chunk_callback = chunk_callback
        self.request_finished_callback = request_finished_callback
        self.error_callback = error_callback
        self.max_batch_size = int(max_batch_size)
        self.microbatch_wait_seconds = max(0.0, microbatch_wait_ms / 1000.0)
        self.trace_callback = trace_callback
        self._queue: deque[GeneratedCodecFrame] = deque()
        self._caches: dict[str, dict] = {}
        self._closing: set[str] = set()
        self._cancelled: set[str] = set()
        self._sequences: dict[str, int] = {}
        self._condition = Condition()
        self._thread: Thread | None = None
        self._stopping = False

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = Thread(target=self._run, name="qwen-codec-decoder", daemon=True)
            self._thread.start()

    def stop(self, *, drain: bool = True) -> None:
        with self._condition:
            self._stopping = True
            if not drain:
                self._queue.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()
        with self._condition:
            self._thread = None

    def create_request(self, request_id: str) -> None:
        with self._condition:
            if request_id in self._caches:
                raise ValueError(f"audio decoder request {request_id!r} already exists")
            self._caches[request_id] = {}
            self._sequences[request_id] = 0
            self._closing.discard(request_id)
            self._cancelled.discard(request_id)

    def submit_frame(self, frame: GeneratedCodecFrame) -> None:
        with self._condition:
            if frame.request_id in self._cancelled:
                return
            if frame.request_id not in self._caches:
                raise KeyError(frame.request_id)
            self._queue.append(frame)
            self._condition.notify()

    def finish_request(self, request_id: str) -> None:
        with self._condition:
            if request_id not in self._caches:
                return
            self._closing.add(request_id)
            self._condition.notify_all()

    def cancel_request(self, request_id: str) -> None:
        with self._condition:
            self._cancelled.add(request_id)
            self._closing.discard(request_id)
            self._caches.pop(request_id, None)
            self._queue = deque(frame for frame in self._queue if frame.request_id != request_id)
            self._sequences.pop(request_id, None)
            self._trace(request_id, "audio.cancelled")
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closing or self._stopping)
                if self._stopping and not self._queue:
                    self._finish_drained_requests_locked()
                    return
                if self._queue and self.microbatch_wait_seconds:
                    self._condition.wait(timeout=self.microbatch_wait_seconds)
                frames = self._take_batch_locked()

            if not frames:
                with self._condition:
                    self._finish_drained_requests_locked()
                continue
            request_ids = [frame.request_id for frame in frames]
            try:
                batch_started_at = time.perf_counter()
                device = next(self.decoder.parameters()).device
                codes = torch.stack([frame.codes.to(device=device) for frame in frames], dim=0)
                if codes.ndim == 2:
                    codes = codes.unsqueeze(-1)
                caches = [self._caches[request_id] for request_id in request_ids]
                decoder_started_at = time.perf_counter()
                with torch.inference_mode():
                    waveforms = self.decoder.batched_chunked_decode(
                        codes,
                        [int(codes.shape[-1])] * len(frames),
                        caches,
                        max_batch_size=self.max_batch_size,
                    )
                decoder_finished_at = time.perf_counter()
                cpu_waveforms = [
                    waveform.reshape(-1).to(dtype=torch.float32).detach().cpu()
                    for waveform in waveforms
                ]
                decoded_at = time.perf_counter()
                for frame, pcm in zip(frames, cpu_waveforms, strict=True):
                    sequence = self._sequences.get(frame.request_id, 0)
                    self._trace(
                        frame.request_id,
                        "audio.decode",
                        requestIds=request_ids,
                        batchSize=len(frames),
                        decoderMs=(decoder_finished_at - decoder_started_at) * 1_000,
                        gpuToCpuMs=(decoded_at - decoder_finished_at) * 1_000,
                        totalMs=(decoded_at - batch_started_at) * 1_000,
                    )
                    self._trace(
                        frame.request_id,
                        "pcm.chunk",
                        sequence=sequence,
                        samples=int(pcm.numel()),
                        sampleRate=self.sample_rate,
                        codecToPcmMs=(decoded_at - frame.generated_at) * 1_000,
                    )
                    self._sequences[frame.request_id] = sequence + 1
                    self.chunk_callback(
                        GeneratedAudioChunk(
                            request_id=frame.request_id,
                            pcm=pcm,
                            sample_rate=self.sample_rate,
                            generated_at=frame.generated_at,
                            decoded_at=decoded_at,
                        )
                    )
            except Exception as error:
                for request_id in set(request_ids):
                    if self.error_callback is not None:
                        self.error_callback(request_id, error)
                    self.cancel_request(request_id)
            finally:
                with self._condition:
                    self._finish_drained_requests_locked()

    def _take_batch_locked(self) -> list[GeneratedCodecFrame]:
        frames: list[GeneratedCodecFrame] = []
        selected_request_ids: set[str] = set()
        deferred: deque[GeneratedCodecFrame] = deque()
        while self._queue and len(frames) < self.max_batch_size:
            frame = self._queue.popleft()
            if frame.request_id in self._cancelled or frame.request_id not in self._caches:
                continue
            if frame.request_id in selected_request_ids:
                deferred.append(frame)
                continue
            frames.append(frame)
            selected_request_ids.add(frame.request_id)
        self._queue.extendleft(reversed(deferred))
        return frames

    def _finish_drained_requests_locked(self) -> None:
        queued_request_ids = {frame.request_id for frame in self._queue}
        finished = [request_id for request_id in self._closing if request_id not in queued_request_ids]
        for request_id in finished:
            self._closing.remove(request_id)
            self._caches.pop(request_id, None)
            self._sequences.pop(request_id, None)
            self._trace(request_id, "audio.released")
        if self.request_finished_callback is not None:
            for request_id in finished:
                self.request_finished_callback(request_id)

    def _trace(self, request_id: str, event: str, **data) -> None:
        if self.trace_callback is not None:
            self.trace_callback(
                StreamingTraceEvent(
                    request_id=request_id,
                    event=event,
                    data=data,
                )
            )
