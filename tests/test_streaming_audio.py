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

    def batched_chunked_decode(self, codes, lengths, caches, *, max_batch_size):
        self.batch_sizes.append(int(codes.shape[0]))
        for cache in caches:
            cache["frames"] = cache.get("frames", 0) + 1
        return [torch.full((1, 1, 4), float(row)) for row in range(codes.shape[0])]


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

