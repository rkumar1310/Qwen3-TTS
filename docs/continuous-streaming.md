# Native continuous text streaming

`Qwen3TTSContinuousEngine` keeps one acoustic request alive while text arrives.
It does not split text into independently synthesized sentences.

The runtime has three stateful layers:

1. `StableTextTokenizer` commits only the stable prefix of cumulative text.
2. Hugging Face continuous batching owns request scheduling and paged Talker KV caches; a request pauses when its text queue is empty and resumes in the same cache when more tokens arrive.
3. `QwenStreamingAudioDecoder` batches codec frames from different requests while retaining separate Code2Wav state for each request.

Only `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` is supported by this first version.

```python
import torch

from qwen_tts import Qwen3TTSModel
from qwen_tts.streaming import Qwen3TTSContinuousEngine


def on_audio(chunk):
    # chunk.pcm is mono float32 at chunk.sample_rate (24 kHz for the model above).
    send_pcm(chunk.request_id, chunk.pcm.numpy(), chunk.sample_rate)


model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
engine = Qwen3TTSContinuousEngine(model, audio_callback=on_audio)
engine.start()

engine.create_request("turn-1", speaker="Aiden", language="English")
engine.append_text("turn-1", "This arrives ")
engine.append_text("turn-1", "as live text deltas.")
engine.finish_text("turn-1")
```

Consume the corresponding generation result from `engine.manager`. When the acoustic request is terminal, call `engine.finish_audio(request_id)` so the decoder emits any queued PCM before dropping its state, then call `engine.release_request(request_id)` to release the Talker and text state. Use `engine.cancel_request(request_id)` on interruption.

The engine deliberately starts without CUDA graphs. This keeps the append/pause/resume path simple and measurable; graph capture can be added after GPU parity and concurrency tests cover the native runtime.

