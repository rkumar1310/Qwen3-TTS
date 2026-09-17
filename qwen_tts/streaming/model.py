"""Qwen Talker adapter for appendable Hugging Face continuous batching."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Callable

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from .requests import (
    StreamConditionKind,
    StreamingRequestRegistry,
)


@dataclass(frozen=True)
class PreparedStreamingPrompt:
    embeddings: torch.Tensor
    text_end_embedding: torch.Tensor
    text_padding_embedding: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(frozen=True)
class BatchRequestContext:
    request_id: str
    query_length: int
    past_length: int


@dataclass(frozen=True)
class GeneratedCodecFrame:
    request_id: str
    codes: torch.Tensor
    generated_at: float


@dataclass
class _ModelSession:
    prompt: PreparedStreamingPrompt
    past_hidden: torch.Tensor | None = None


def prepare_custom_voice_prompt(
    model,
    *,
    role_token_ids: torch.Tensor,
    first_text_token_id: int,
    language: str,
    speaker: str,
    instruct_ids: torch.Tensor | None = None,
) -> PreparedStreamingPrompt:
    """Build the exact fixed prompt used by Qwen's CustomVoice generation."""
    talker = model.talker
    device = talker.device
    token_dtype = role_token_ids.dtype
    language_name = language.lower()
    speaker_name = speaker.lower()

    if speaker_name not in model.config.talker_config.spk_id:
        raise ValueError(f"Unsupported speaker: {speaker}")
    if language_name == "auto":
        language_id = None
    elif language_name in model.config.talker_config.codec_language_id:
        language_id = model.config.talker_config.codec_language_id[language_name]
    else:
        raise ValueError(f"Unsupported language: {language}")

    dialect = model.config.talker_config.spk_is_dialect.get(speaker_name, False)
    if language_name in {"chinese", "auto"} and dialect:
        language_id = model.config.talker_config.codec_language_id[dialect]

    special_ids = torch.tensor(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
        device=device,
        dtype=token_dtype,
    )
    tts_bos, tts_end, tts_pad = talker.text_projection(
        talker.get_text_embeddings()(special_ids)
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefix = [
            model.config.talker_config.codec_nothink_id,
            model.config.talker_config.codec_think_bos_id,
            model.config.talker_config.codec_think_eos_id,
        ]
    else:
        codec_prefix = [
            model.config.talker_config.codec_think_id,
            model.config.talker_config.codec_think_bos_id,
            language_id,
            model.config.talker_config.codec_think_eos_id,
        ]

    prefix_embeddings = talker.get_input_embeddings()(
        torch.tensor([codec_prefix], device=device, dtype=token_dtype)
    )
    speaker_embedding = talker.get_input_embeddings()(
        torch.tensor(
            model.config.talker_config.spk_id[speaker_name],
            device=device,
            dtype=token_dtype,
        )
    ).view(1, 1, -1)
    suffix_embeddings = talker.get_input_embeddings()(
        torch.tensor(
            [[model.config.talker_config.codec_pad_id, model.config.talker_config.codec_bos_id]],
            device=device,
            dtype=token_dtype,
        )
    )
    codec_embeddings = torch.cat(
        [prefix_embeddings, speaker_embedding, suffix_embeddings],
        dim=1,
    )

    role_embeddings = talker.text_projection(
        talker.get_text_embeddings()(role_token_ids.to(device))
    )
    aligned_codec_prefix = torch.cat(
        [
            tts_pad.expand(-1, codec_embeddings.shape[1] - 2, -1),
            tts_bos,
        ],
        dim=1,
    ) + codec_embeddings[:, :-1]
    first_text = torch.tensor([[first_text_token_id]], device=device, dtype=token_dtype)
    first_text_embedding = talker.text_projection(
        talker.get_text_embeddings()(first_text)
    ) + codec_embeddings[:, -1:]

    prompt_parts = []
    if instruct_ids is not None:
        prompt_parts.append(
            talker.text_projection(
                talker.get_text_embeddings()(instruct_ids.to(device))
            )
        )
    prompt_parts.extend([role_embeddings, aligned_codec_prefix, first_text_embedding])
    return PreparedStreamingPrompt(
        embeddings=torch.cat(prompt_parts, dim=1),
        text_end_embedding=tts_end,
        text_padding_embedding=tts_pad,
    )


class QwenStreamingTalkerAdapter(nn.Module):
    """Expose Qwen's Talker as a continuously batched causal model.

    The main Talker uses Hugging Face's paged KV cache.  Qwen's secondary
    codebook predictor is batched across every decoding request selected in the
    same scheduler step.  Request-local hidden state and text conditions remain
    attached to the request ID across pauses and resumes.
    """

    _supports_flash_attn = True

    def __init__(
        self,
        talker,
        registry: StreamingRequestRegistry,
        *,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        main_attention_implementation: str | None = None,
        frame_callback: Callable[[GeneratedCodecFrame], None] | None = None,
    ) -> None:
        super().__init__()
        self.talker = talker
        self.config = talker.config
        self.registry = registry
        self.subtalker_dosample = subtalker_dosample
        self.subtalker_top_k = subtalker_top_k
        self.subtalker_top_p = subtalker_top_p
        self.subtalker_temperature = subtalker_temperature
        self.main_attention_implementation = main_attention_implementation
        self.frame_callback = frame_callback
        self._sessions: dict[str, _ModelSession] = {}
        self._batch_context: list[BatchRequestContext] | None = None
        self._pending_frames: deque[GeneratedCodecFrame] = deque()
        self._lock = RLock()

    @property
    def device(self):
        return self.talker.device

    @property
    def dtype(self):
        return self.talker.dtype

    @property
    def tp_plan(self):
        return getattr(self.talker, "tp_plan", {})

    def set_attn_implementation(self, implementation: str) -> None:
        # Only the main acoustic Talker uses the paged cache owned by the
        # continuous-batching manager. The secondary codebook predictor runs
        # a normal short batched generation and must keep ordinary attention.
        self.talker.model.set_attn_implementation(
            {
                "": self.main_attention_implementation or implementation,
                "code_predictor_config": self.talker.code_predictor.config._attn_implementation,
            }
        )

    def _get_logits_processor(self, generation_config):
        return self.talker._get_logits_processor(generation_config)

    def register_session(self, request_id: str, prompt: PreparedStreamingPrompt) -> None:
        with self._lock:
            if request_id in self._sessions:
                raise ValueError(f"model session {request_id!r} already exists")
            self._sessions[request_id] = _ModelSession(prompt=prompt)

    def release_session(self, request_id: str) -> None:
        with self._lock:
            self._sessions.pop(request_id, None)

    def set_batch_context(self, contexts: list[BatchRequestContext]) -> None:
        self._batch_context = contexts

    def clear_batch_context(self) -> None:
        self._batch_context = None

    def flush_frame_callbacks(self) -> None:
        if self.frame_callback is None:
            return
        while self._pending_frames:
            self.frame_callback(self._pending_frames.popleft())

    def pop_codec_frames(self) -> list[GeneratedCodecFrame]:
        frames = list(self._pending_frames)
        self._pending_frames.clear()
        return frames

    def _supports_logits_to_keep(self) -> bool:
        return False

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        cu_seq_lens_q=None,
        max_seqlen_q=None,
        logits_indices=None,
        logits_processor_args=None,
        cu_seq_lens_k=None,
        max_seqlen_k=None,
        read_index=None,
        write_index=None,
        cache=None,
        block_table=None,
        use_cache=False,
        **kwargs,
    ):
        del logits_indices, logits_processor_args, kwargs
        contexts = self._batch_context
        if not contexts:
            raise RuntimeError("Qwen batch context was not set before forward")

        offsets = []
        offsets_by_request: dict[str, int] = {}
        cursor = 0
        for context in contexts:
            offsets.append(cursor)
            offsets_by_request[context.request_id] = cursor
            cursor += context.query_length
        if cursor != input_ids.shape[1]:
            raise RuntimeError("Qwen batch context does not match packed input length")

        decode_contexts = [context for context in contexts if context.past_length > 0]
        decode_embeddings: dict[str, torch.Tensor] = {}
        if decode_contexts:
            main_tokens = torch.cat(
                [
                    input_ids[
                        :,
                        offsets_by_request[context.request_id] : offsets_by_request[context.request_id] + 1,
                    ]
                    for context in decode_contexts
                ],
                dim=0,
            ).long()
            sessions = [self._sessions[context.request_id] for context in decode_contexts]
            if any(session.past_hidden is None for session in sessions):
                raise RuntimeError("Qwen decode started before its prompt was prefetched")
            past_hidden = torch.cat([session.past_hidden for session in sessions], dim=0)
            first_codebook_hidden = self.talker.get_input_embeddings()(main_tokens)
            predictor = self.talker.code_predictor.generate(
                inputs_embeds=torch.cat([past_hidden, first_codebook_hidden], dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=self.subtalker_dosample,
                top_k=self.subtalker_top_k,
                top_p=self.subtalker_top_p,
                temperature=self.subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_codes = torch.cat([main_tokens, predictor.sequences], dim=-1)
            codec_embeddings = [first_codebook_hidden]
            codec_embeddings.extend(
                self.talker.code_predictor.get_input_embeddings()[index](
                    predictor.sequences[:, index : index + 1]
                )
                for index in range(self.config.num_code_groups - 1)
            )
            summed_codec_embeddings = torch.cat(codec_embeddings, dim=1).sum(
                dim=1,
                keepdim=True,
            )

            for index, (context, session) in enumerate(zip(decode_contexts, sessions)):
                condition = self.registry.take_condition(context.request_id)
                if condition.kind == StreamConditionKind.TEXT:
                    token = torch.tensor(
                        [[condition.token_id]],
                        device=self.device,
                        dtype=torch.long,
                    )
                    text_embedding = self.talker.text_projection(
                        self.talker.get_text_embeddings()(token)
                    )
                elif condition.kind == StreamConditionKind.TEXT_END:
                    text_embedding = session.prompt.text_end_embedding
                else:
                    text_embedding = session.prompt.text_padding_embedding
                decode_embeddings[context.request_id] = (
                    summed_codec_embeddings[index : index + 1] + text_embedding
                )
                self._pending_frames.append(
                    GeneratedCodecFrame(
                        request_id=context.request_id,
                        codes=codec_codes[index].detach(),
                        generated_at=time.perf_counter(),
                    )
                )

        packed_embeddings = []
        for context in contexts:
            session = self._sessions.get(context.request_id)
            if session is None:
                raise RuntimeError(f"unknown Qwen model session {context.request_id!r}")
            if context.past_length == 0:
                embeddings = session.prompt.embeddings
                if embeddings.shape[1] != context.query_length:
                    raise RuntimeError("Qwen prompt length does not match scheduler prefill length")
            else:
                if context.query_length != 1:
                    raise RuntimeError("Qwen decode requests must contain exactly one codec token")
                embeddings = decode_embeddings[context.request_id]
            packed_embeddings.append(embeddings)

        model_kwargs = {
            "cu_seq_lens_q": cu_seq_lens_q,
            "max_seqlen_q": max_seqlen_q,
            "cu_seq_lens_k": cu_seq_lens_k,
            "max_seqlen_k": max_seqlen_k,
            "read_index": read_index,
            "write_index": write_index,
            "cache": cache,
            "block_table": block_table,
        }
        outputs = self.talker.model(
            inputs_embeds=torch.cat(packed_embeddings, dim=1),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=use_cache,
            **model_kwargs,
        )
        hidden_states = outputs.last_hidden_state
        for context, offset in zip(contexts, offsets):
            self._sessions[context.request_id].past_hidden = hidden_states[
                :,
                offset + context.query_length - 1 : offset + context.query_length,
            ]

        return CausalLMOutputWithPast(
            logits=self.talker.codec_head(hidden_states),
            past_key_values=None,
        )
