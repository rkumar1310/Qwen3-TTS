"""Public text-in/codec-out continuous streaming engine for Qwen3-TTS."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch
from transformers import ContinuousBatchingConfig, GenerationConfig

from .audio import QwenStreamingAudioDecoder
from .hf_manager import AppendableContinuousBatchingManager
from .model import (
    QwenStreamingTalkerAdapter,
    prepare_custom_voice_prompt,
)
from .requests import StreamingRequestRegistry
from .text import StableTextTokenizer


@dataclass
class _PendingInput:
    tokenizer: StableTextTokenizer
    language: str
    speaker: str
    instruct_ids: torch.Tensor | None
    role_token_ids: torch.Tensor
    max_new_tokens: int | None
    submitted: bool = False


class Qwen3TTSContinuousEngine:
    """Append OpenAI text deltas to one persistent Qwen speech request."""

    def __init__(
        self,
        qwen_model,
        *,
        max_requests_per_batch: int = 16,
        max_batch_tokens: int = 2048,
        max_blocks_per_request: int = 16,
        max_memory_percent: float = 0.7,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        repetition_penalty: float = 1.05,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        frame_callback=None,
        audio_callback=None,
        audio_finished_callback=None,
        audio_error_callback=None,
        audio_microbatch_wait_ms: float = 1.0,
    ) -> None:
        if qwen_model.model.tts_model_type != "custom_voice":
            raise ValueError("continuous engine currently requires a CustomVoice checkpoint")
        self.qwen_model = qwen_model
        self.registry = StreamingRequestRegistry()
        self.audio_decoder = None
        if audio_callback is not None:
            speech_tokenizer = qwen_model.model.speech_tokenizer
            if speech_tokenizer is None or not hasattr(speech_tokenizer.model, "decoder"):
                raise ValueError("Qwen 12Hz speech tokenizer decoder is not loaded")
            self.audio_decoder = QwenStreamingAudioDecoder(
                speech_tokenizer.model.decoder,
                sample_rate=speech_tokenizer.model.get_output_sample_rate(),
                chunk_callback=audio_callback,
                request_finished_callback=audio_finished_callback,
                error_callback=audio_error_callback,
                max_batch_size=max_requests_per_batch,
                microbatch_wait_ms=audio_microbatch_wait_ms,
            )

        def handle_frame(frame):
            if frame_callback is not None:
                frame_callback(frame)
            if self.audio_decoder is not None:
                self.audio_decoder.submit_frame(frame)

        self.adapter = QwenStreamingTalkerAdapter(
            qwen_model.model.talker,
            self.registry,
            subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k,
            subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature,
            frame_callback=handle_frame,
        )
        talker_config = qwen_model.model.config.talker_config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            min_new_tokens=2,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=talker_config.codec_eos_token_id,
            suppress_tokens=[
                token
                for token in range(talker_config.vocab_size - 1024, talker_config.vocab_size)
                if token != talker_config.codec_eos_token_id
            ],
        )
        batching_config = ContinuousBatchingConfig(
            max_requests_per_batch=max_requests_per_batch,
            max_batch_tokens=max_batch_tokens,
            max_blocks_per_request=max_blocks_per_request,
            max_memory_percent=max_memory_percent,
            allow_block_sharing=False,
            use_async_batching=False,
            use_cuda_graph=False,
            scheduler_type="fifo",
        )
        self.manager = AppendableContinuousBatchingManager(
            self.adapter,
            generation_config,
            batching_config,
            registry=self.registry,
        )
        self._inputs: dict[str, _PendingInput] = {}
        self._lock = RLock()

    def start(self) -> None:
        if self.audio_decoder is not None:
            self.audio_decoder.start()
        self.manager.start()

    def stop(self, *, hard_stop: bool = False) -> None:
        self.manager.stop(block=True, hard_stop=hard_stop)
        if self.audio_decoder is not None:
            self.audio_decoder.stop(drain=not hard_stop)

    def create_request(
        self,
        request_id: str,
        *,
        language: str = "English",
        speaker: str = "Aiden",
        instruct: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        with self._lock:
            if request_id in self._inputs:
                raise ValueError(f"request {request_id!r} already exists")

            def tokenize_content(text: str) -> list[int]:
                ids = self.qwen_model._tokenize_texts(
                    [self.qwen_model._build_assistant_text(text)]
                )[0]
                if ids.shape[1] < 8:
                    raise RuntimeError("unexpected Qwen assistant prompt tokenization")
                return [int(token) for token in ids[0, 3:-5].tolist()]

            role_ids = self.qwen_model._tokenize_texts(
                [self.qwen_model._build_assistant_text("")]
            )[0][:, :3]
            instruct_ids = None
            if instruct:
                instruct_ids = self.qwen_model._tokenize_texts(
                    [self.qwen_model._build_instruct_text(instruct)]
                )[0]
            self._inputs[request_id] = _PendingInput(
                tokenizer=StableTextTokenizer(tokenize_content),
                language=language,
                speaker=speaker,
                instruct_ids=instruct_ids,
                role_token_ids=role_ids,
                max_new_tokens=max_new_tokens,
            )

    def is_submitted(self, request_id: str) -> bool:
        with self._lock:
            return self._get(request_id).submitted
            if self.audio_decoder is not None:
                self.audio_decoder.create_request(request_id)

    def append_text(self, request_id: str, delta: str) -> None:
        with self._lock:
            pending = self._get(request_id)
            token_ids = pending.tokenizer.append(delta)
            self._route_tokens(request_id, pending, token_ids)

    def finish_text(self, request_id: str) -> None:
        with self._lock:
            pending = self._get(request_id)
            final_tokens = pending.tokenizer.finish()
            self._route_tokens(request_id, pending, final_tokens)
            if not pending.submitted:
                raise ValueError("cannot synthesize an empty text stream")
            self.manager.close_text_input(request_id)

    def cancel_request(self, request_id: str) -> None:
        with self._lock:
            pending = self._inputs.get(request_id)
            if pending is not None and pending.submitted:
                self.manager.cancel_request(request_id)
                self.adapter.release_session(request_id)
            if self.audio_decoder is not None:
                self.audio_decoder.cancel_request(request_id)
            self._inputs.pop(request_id, None)

    def release_request(self, request_id: str) -> None:
        with self._lock:
            self.manager.release_streaming_request(request_id)
            self.adapter.release_session(request_id)
            self._inputs.pop(request_id, None)

    def finish_audio(self, request_id: str) -> None:
        """Release decoder state after every queued PCM frame is emitted."""
        if self.audio_decoder is not None:
            self.audio_decoder.finish_request(request_id)

    def _route_tokens(
        self,
        request_id: str,
        pending: _PendingInput,
        token_ids: list[int],
    ) -> None:
        if not token_ids:
            return
        if pending.submitted:
            self.manager.append_text_tokens(request_id, token_ids)
            return

        first_token, remaining = token_ids[0], token_ids[1:]
        prompt = prepare_custom_voice_prompt(
            self.qwen_model.model,
            role_token_ids=pending.role_token_ids,
            first_text_token_id=first_token,
            language=pending.language,
            speaker=pending.speaker,
            instruct_ids=pending.instruct_ids,
        )
        self.adapter.register_session(request_id, prompt)
        placeholder_ids = [self.qwen_model.model.config.talker_config.codec_pad_id] * prompt.length
        accepted_id = self.manager.add_streaming_request(
            placeholder_ids,
            request_id=request_id,
            remaining_text_tokens=remaining,
            max_new_tokens=pending.max_new_tokens,
            eos_token_id=self.qwen_model.model.config.talker_config.codec_eos_token_id,
        )
        if accepted_id is None:
            self.adapter.release_session(request_id)
            raise RuntimeError("continuous batching manager rejected the Qwen request")
        pending.submitted = True

    def _get(self, request_id: str) -> _PendingInput:
        pending = self._inputs.get(request_id)
        if pending is None:
            raise KeyError(request_id)
        return pending
