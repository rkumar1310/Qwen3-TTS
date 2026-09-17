import time
from threading import Event

import torch
from torch import nn

from qwen_tts.streaming.audio import QwenStreamingAudioDecoder
from qwen_tts.streaming.model import GeneratedCodecFrame


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.batch_sizes = []
        self.frame_lengths = []

    def batched_chunked_decode(self, codes, lengths, caches, *, max_batch_size):
        self.batch_sizes.append(int(codes.shape[0]))
        self.frame_lengths.append(list(lengths))
        for cache, length in zip(caches, lengths, strict=True):
            cache["frames"] = cache.get("frames", 0) + length
        return [
            torch.full((1, 1, length * 4), float(row))
            for row, length in enumerate(lengths)
        ]


def test_audio_decoder_microbatches_requests_and_finishes_after_drain():
    decoder = FakeDecoder()
    chunks = []
    finished = []
    done = Event()

    def on_finished(request_id):
        finished.append(request_id)
        if len(finished) == 2:
            done.set()

    worker = QwenStreamingAudioDecoder(
        decoder,
        sample_rate=24_000,
        chunk_callback=chunks.append,
        request_finished_callback=on_finished,
        microbatch_wait_ms=20,
    )
    worker.create_request("a")
    worker.create_request("b")
    worker.start()
    now = time.perf_counter()
    worker.submit_frame(GeneratedCodecFrame("a", torch.tensor([1, 2]), now))
    worker.submit_frame(GeneratedCodecFrame("b", torch.tensor([3, 4]), now))
    worker.finish_request("a")
    worker.finish_request("b")

    assert done.wait(2)
    worker.stop()

    assert decoder.batch_sizes == [2]
    assert {chunk.request_id for chunk in chunks} == {"a", "b"}
    assert all(chunk.sample_rate == 24_000 for chunk in chunks)
    assert set(finished) == {"a", "b"}


def test_audio_decoder_buffers_a_small_first_chunk_then_larger_steady_chunks():
    decoder = FakeDecoder()
    chunks = []
    done = Event()
    worker = QwenStreamingAudioDecoder(
        decoder,
        sample_rate=24_000,
        chunk_callback=lambda chunk: (chunks.append(chunk), done.set() if len(chunks) == 2 else None),
        initial_chunk_frames=4,
        steady_chunk_frames=8,
    )
    worker.create_request("turn")
    worker.start()
    now = time.perf_counter()
    for sequence in range(12):
        worker.submit_frame(
            GeneratedCodecFrame(
                "turn",
                torch.tensor([sequence, sequence + 1]),
                now + sequence / 12,
                sequence,
            )
        )

    assert done.wait(2)
    worker.finish_request("turn")
    worker.stop()

    assert decoder.frame_lengths == [[4], [8]]
    assert [chunk.pcm.numel() for chunk in chunks] == [16, 32]


def test_audio_decoder_ramps_to_the_steady_chunk_without_starving_playback():
    decoder = FakeDecoder()
    chunks = []
    done = Event()

    def on_chunk(chunk):
        chunks.append(chunk)
        if len(chunks) == 4:
            done.set()

    worker = QwenStreamingAudioDecoder(
        decoder,
        sample_rate=24_000,
        chunk_callback=on_chunk,
        initial_chunk_frames=4,
        steady_chunk_frames=25,
    )
    worker.create_request("turn")
    worker.start()
    now = time.perf_counter()
    for sequence in range(53):
        worker.submit_frame(
            GeneratedCodecFrame(
                "turn",
                torch.tensor([sequence, sequence + 1]),
                now + sequence / 12,
                sequence,
            )
        )

    assert done.wait(2)
    worker.finish_request("turn")
    worker.stop()

    assert decoder.frame_lengths == [[4], [8], [16], [25]]
