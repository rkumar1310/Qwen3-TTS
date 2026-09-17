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
        initial_chunk_frames: int = 4,
        steady_chunk_frames: int = 25,
        trace_callback: TraceCallback | None = None,
    ) -> None:
        self.decoder = decoder
        self.sample_rate = int(sample_rate)
        self.chunk_callback = chunk_callback
        self.request_finished_callback = request_finished_callback
        self.error_callback = error_callback
        self.max_batch_size = int(max_batch_size)
        self.microbatch_wait_seconds = max(0.0, microbatch_wait_ms / 1000.0)
        self.initial_chunk_frames = max(1, int(initial_chunk_frames))
        self.steady_chunk_frames = max(
            self.initial_chunk_frames,
            int(steady_chunk_frames),
        )
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
                frame_groups = self._take_batch_locked()

            if not frame_groups:
                with self._condition:
                    self._finish_drained_requests_locked()
                    if self._queue and not self._stopping:
                        self._condition.wait(timeout=0.05)
                continue
            request_ids = [frames[0].request_id for frames in frame_groups]
            try:
                batch_started_at = time.perf_counter()
                device = next(self.decoder.parameters()).device
                lengths = [len(frames) for frames in frame_groups]
                code_groups = [
                    torch.stack([frame.codes.to(device=device) for frame in frames], dim=-1)
                    for frames in frame_groups
                ]
                codes = torch.zeros(
                    len(code_groups),
                    code_groups[0].shape[0],
                    max(lengths),
                    dtype=code_groups[0].dtype,
                    device=device,
                )
                for row, request_codes in enumerate(code_groups):
                    codes[row, :, : request_codes.shape[-1]].copy_(request_codes)
                caches = [self._caches[request_id] for request_id in request_ids]
                decoder_started_at = time.perf_counter()
                with torch.inference_mode():
                    waveforms = self.decoder.batched_chunked_decode(
                        codes,
                        lengths,
                        caches,
                        max_batch_size=self.max_batch_size,
                    )
                decoder_finished_at = time.perf_counter()
                cpu_waveforms = [
                    waveform.reshape(-1).to(dtype=torch.float32).detach().cpu()
                    for waveform in waveforms
                ]
                decoded_at = time.perf_counter()
                for frames, pcm in zip(frame_groups, cpu_waveforms, strict=True):
                    first_frame = frames[0]
                    last_frame = frames[-1]
                    sequence = self._sequences.get(first_frame.request_id, 0)
                    self._trace(
                        first_frame.request_id,
                        "audio.decode",
                        requestIds=request_ids,
                        batchSize=len(frame_groups),
                        codecFrames=len(frames),
                        decoderMs=(decoder_finished_at - decoder_started_at) * 1_000,
                        gpuToCpuMs=(decoded_at - decoder_finished_at) * 1_000,
                        totalMs=(decoded_at - batch_started_at) * 1_000,
                    )
                    self._trace(
                        first_frame.request_id,
                        "pcm.chunk",
                        sequence=sequence,
                        samples=int(pcm.numel()),
                        sampleRate=self.sample_rate,
                        codecFrames=len(frames),
                        bufferMs=(last_frame.generated_at - first_frame.generated_at) * 1_000,
                        codecToPcmMs=(decoded_at - first_frame.generated_at) * 1_000,
                    )
                    self._sequences[first_frame.request_id] = sequence + 1
                    self.chunk_callback(
                        GeneratedAudioChunk(
                            request_id=first_frame.request_id,
                            pcm=pcm,
                            sample_rate=self.sample_rate,
                            generated_at=first_frame.generated_at,
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

    def _take_batch_locked(self) -> list[list[GeneratedCodecFrame]]:
        """Take several frames per request to amortize the expensive vocoder call."""
        available: dict[str, int] = {}
        request_order: list[str] = []
        filtered: deque[GeneratedCodecFrame] = deque()
        for frame in self._queue:
            if frame.request_id in self._cancelled or frame.request_id not in self._caches:
                continue
            filtered.append(frame)
            if frame.request_id not in available:
                available[frame.request_id] = 0
                request_order.append(frame.request_id)
            available[frame.request_id] += 1
        self._queue = filtered

        selected_counts: dict[str, int] = {}
        for request_id in request_order:
            if len(selected_counts) >= self.max_batch_size:
                break
            target = (
                self.initial_chunk_frames
                if self._sequences.get(request_id, 0) == 0
                else self.steady_chunk_frames
            )
            count = available[request_id]
            if count >= target:
                selected_counts[request_id] = target
            elif (request_id in self._closing or self._stopping) and count:
                selected_counts[request_id] = count

        if not selected_counts:
            return []
        groups = {request_id: [] for request_id in selected_counts}
        deferred: deque[GeneratedCodecFrame] = deque()
        while self._queue:
            frame = self._queue.popleft()
            group = groups.get(frame.request_id)
            if group is not None and len(group) < selected_counts[frame.request_id]:
                group.append(frame)
            else:
                deferred.append(frame)
        self._queue = deferred
        return [groups[request_id] for request_id in request_order if request_id in groups]

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
