"""Native batched scheduler for appendable Qwen3-TTS requests.

The scheduler deliberately stays on Qwen's supported Transformers runtime.  It
executes the same prefill and recurrent decode operations as ``generate()``,
but keeps each request's cache alive so text can pause and resume.  Compatible
requests are stacked into real GPU batches for both the main Talker and the
secondary codebook predictor.
"""

from __future__ import annotations

import copy
import math
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from threading import Condition, Thread
from typing import Callable

import torch

from .cuda_graphs import PredictorCudaGraph, TalkerCudaGraph
from .model import GeneratedCodecFrame, PreparedStreamingPrompt
from .requests import StreamConditionKind, StreamingRequestRegistry
from .trace import StreamingTraceEvent, TraceCallback


class NativeRequestStatus(str, Enum):
    PREFILLING = "prefilling"
    DECODING = "decoding"
    WAITING_FOR_TEXT = "waiting_for_text"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


_TERMINAL_STATUSES = {
    NativeRequestStatus.COMPLETED,
    NativeRequestStatus.CANCELLED,
    NativeRequestStatus.FAILED,
}


@dataclass(frozen=True)
class NativeGenerationResult:
    request_id: str
    status: NativeRequestStatus
    generated_tokens: tuple[int, ...]
    error: str | None = None

    def is_finished(self) -> bool:
        return self.status in _TERMINAL_STATUSES


@dataclass
class NativeBackgroundThreadStatus:
    fatal_error: Exception | None = None


@dataclass(frozen=True)
class NativeSamplingConfig:
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0
    temperature: float = 0.9
    repetition_penalty: float = 1.05


@dataclass
class _NativeRequest:
    request_id: str
    prompt: PreparedStreamingPrompt
    max_new_tokens: int
    sampling: NativeSamplingConfig
    subtalker_sampling: NativeSamplingConfig
    generator: torch.Generator
    status: NativeRequestStatus = NativeRequestStatus.PREFILLING
    past_key_values: object | None = None
    cache_length: int = 0
    past_hidden: torch.Tensor | None = None
    next_token: int | None = None
    generated_tokens: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.perf_counter)
    last_scheduled_order: int = 0


def _cache_length(cache: object) -> int:
    return int(cache.get_seq_length())


def _request_cache_length(request: _NativeRequest) -> int:
    """Return the logical length even while KV state lives in a static graph."""
    return request.cache_length or _cache_length(request.past_key_values)


def _batch_caches(caches: list[object]) -> object:
    """Stack same-length DynamicCache objects along their batch dimension."""
    if not caches:
        raise ValueError("cannot batch an empty cache list")
    if len(caches) == 1:
        return caches[0]
    batched = copy.deepcopy(caches[0])
    if hasattr(batched, "layers"):
        for output_layer, input_layers in zip(
            batched.layers,
            zip(*(cache.layers for cache in caches), strict=True),
            strict=True,
        ):
            keys = [layer.keys for layer in input_layers]
            values = [layer.values for layer in input_layers]
            if any(key is None for key in keys) or any(value is None for value in values):
                if not all(key is None for key in keys) or not all(value is None for value in values):
                    raise ValueError("cannot batch partially initialized caches")
                output_layer.keys = None
                output_layer.values = None
            else:
                output_layer.keys = torch.cat(keys, dim=0)
                output_layer.values = torch.cat(values, dim=0)
        return batched

    if hasattr(batched, "key_cache") and hasattr(batched, "value_cache"):
        batched.key_cache = [
            torch.cat([cache.key_cache[index] for cache in caches], dim=0)
            for index in range(len(batched.key_cache))
        ]
        batched.value_cache = [
            torch.cat([cache.value_cache[index] for cache in caches], dim=0)
            for index in range(len(batched.value_cache))
        ]
        return batched
    raise TypeError(f"unsupported cache type: {type(batched)!r}")


def _slice_cache(cache: object, row: int) -> object:
    """Detach one request row from a batched DynamicCache."""
    if row == 0:
        if hasattr(cache, "layers"):
            populated = next(
                (layer.keys for layer in cache.layers if layer.keys is not None),
                None,
            )
            if populated is None or populated.shape[0] == 1:
                return cache
        elif hasattr(cache, "key_cache"):
            populated = next((tensor for tensor in cache.key_cache if tensor is not None), None)
            if populated is None or populated.shape[0] == 1:
                return cache
    result = copy.deepcopy(cache)
    if hasattr(result, "layers"):
        for layer in result.layers:
            if layer.keys is not None:
                layer.keys = layer.keys[row : row + 1].clone()
            if layer.values is not None:
                layer.values = layer.values[row : row + 1].clone()
        return result
    if hasattr(result, "key_cache") and hasattr(result, "value_cache"):
        result.key_cache = [tensor[row : row + 1].clone() for tensor in result.key_cache]
        result.value_cache = [tensor[row : row + 1].clone() for tensor in result.value_cache]
        return result
    raise TypeError(f"unsupported cache type: {type(result)!r}")


class NativeQwenTalkerExecutor:
    """Correct Qwen recurrent inference with request-local sampling state."""

    def __init__(
        self,
        talker,
        *,
        subtalker_sampling: NativeSamplingConfig | None = None,
        enable_cuda_graphs: bool = True,
        cuda_graph_max_sequence_length: int = 2_048,
    ) -> None:
        self.talker = talker
        self.config = talker.config
        self.eos_token_id = int(self.config.codec_eos_token_id)
        self.suppressed_tokens = torch.tensor(
            [
                token
                for token in range(self.config.vocab_size - 1024, self.config.vocab_size)
                if token != self.eos_token_id
            ],
            device=talker.device,
            dtype=torch.long,
        )
        self.last_prefill_metrics: dict[str, float] = {}
        self.last_decode_metrics: dict[str, float] = {}
        self.predictor_graph: PredictorCudaGraph | None = None
        self.talker_graph: TalkerCudaGraph | None = None
        self.graph_owner: _NativeRequest | None = None
        self.cuda_graph_error: str | None = None
        graph_sampling = subtalker_sampling or NativeSamplingConfig()
        if enable_cuda_graphs and talker.device.type == "cuda":
            try:
                dtype = next(talker.parameters()).dtype
                self.predictor_graph = PredictorCudaGraph(
                    talker.code_predictor,
                    talker_hidden_size=int(talker.config.hidden_size),
                    dtype=dtype,
                    do_sample=graph_sampling.do_sample,
                    top_k=graph_sampling.top_k,
                    top_p=graph_sampling.top_p,
                    temperature=graph_sampling.temperature,
                )
                self.talker_graph = TalkerCudaGraph(
                    talker.model,
                    dtype=dtype,
                    max_sequence_length=cuda_graph_max_sequence_length,
                )
                self.predictor_graph.capture()
                self.talker_graph.capture()
            except Exception as error:
                self.predictor_graph = None
                self.talker_graph = None
                self.cuda_graph_error = str(error)

    def prefill(self, requests: list[_NativeRequest]) -> None:
        if not requests:
            return
        self.deactivate_cuda_graph()
        prompt_length = requests[0].prompt.length
        if any(request.prompt.length != prompt_length for request in requests):
            raise ValueError("prefill batch contains different prompt lengths")
        started_at = time.perf_counter()
        batch_size = len(requests)
        embeddings = torch.cat([request.prompt.embeddings for request in requests], dim=0)
        model_started_at = time.perf_counter()
        outputs = self.talker(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(
                (batch_size, prompt_length),
                dtype=torch.long,
                device=self.talker.device,
            ),
            cache_position=torch.arange(prompt_length, device=self.talker.device),
            use_cache=True,
            output_hidden_states=True,
            trailing_text_hidden=torch.cat(
                [request.prompt.text_padding_embedding for request in requests], dim=0
            ),
            tts_pad_embed=torch.cat(
                [request.prompt.text_padding_embedding for request in requests], dim=0
            ),
            subtalker_dosample=False,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
        )
        logits = outputs.logits
        cache_started_at = time.perf_counter()
        for row, request in enumerate(requests):
            request.past_key_values = _slice_cache(outputs.past_key_values, row)
            request.cache_length = _cache_length(request.past_key_values)
            request.past_hidden = outputs.past_hidden[row : row + 1]
            token = self._choose_token(
                logits[row : row + 1],
                request,
                request.sampling,
                history=request.generated_tokens,
                enforce_minimum=True,
            )
            request.next_token = token
            request.generated_tokens.append(token)
        finished_at = time.perf_counter()
        self.last_prefill_metrics = {
            "totalMs": (finished_at - started_at) * 1_000,
            "talkerMs": (cache_started_at - model_started_at) * 1_000,
            "cacheMs": (finished_at - cache_started_at) * 1_000,
        }

    def decode(
        self,
        requests: list[_NativeRequest],
        conditions: list[torch.Tensor],
    ) -> list[GeneratedCodecFrame]:
        if not requests:
            return []
        if len(requests) != len(conditions):
            raise ValueError("requests and text conditions must have equal length")
        if (
            len(requests) == 1
            and self.predictor_graph is not None
            and self.talker_graph is not None
            and requests[0].subtalker_sampling.do_sample
            and _request_cache_length(requests[0]) < self.talker_graph.max_sequence_length
        ):
            return self._decode_with_cuda_graph(requests[0], conditions[0])

        self.deactivate_cuda_graph()
        past_length = _request_cache_length(requests[0])
        if any(_request_cache_length(request) != past_length for request in requests):
            raise ValueError("decode batch contains different cache lengths")
        if any(request.next_token is None or request.past_hidden is None for request in requests):
            raise RuntimeError("decode request was not prefetched")

        main_tokens = torch.tensor(
            [[request.next_token] for request in requests],
            dtype=torch.long,
            device=self.talker.device,
        )
        if all(not request.subtalker_sampling.do_sample for request in requests):
            return self._decode_with_official_talker(
                requests,
                conditions,
                main_tokens,
                past_length,
            )

        started_at = time.perf_counter()
        first_codebook_hidden = self.talker.get_input_embeddings()(main_tokens)
        predictor_started_at = time.perf_counter()
        predictor_sequences = self._predict_secondary_codebooks(
            requests,
            torch.cat(
                [
                    torch.cat([request.past_hidden for request in requests], dim=0),
                    first_codebook_hidden,
                ],
                dim=1,
            ),
        )
        predictor_finished_at = time.perf_counter()
        codec_codes = torch.cat([main_tokens, predictor_sequences], dim=1)
        codec_embeddings = [first_codebook_hidden]
        codec_embeddings.extend(
            self.talker.code_predictor.get_input_embeddings()[index](
                predictor_sequences[:, index : index + 1]
            )
            for index in range(self.config.num_code_groups - 1)
        )
        acoustic_embedding = torch.cat(codec_embeddings, dim=1).sum(dim=1, keepdim=True)
        inputs_embeds = acoustic_embedding + torch.cat(conditions, dim=0)
        cache_started_at = time.perf_counter()
        batched_cache = _batch_caches([request.past_key_values for request in requests])
        cache_finished_at = time.perf_counter()
        batch_size = len(requests)
        position_ids = torch.full(
            (3, batch_size, 1),
            past_length,
            dtype=torch.long,
            device=self.talker.device,
        )
        talker_started_at = time.perf_counter()
        outputs = self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(
                (batch_size, past_length + 1),
                dtype=torch.long,
                device=self.talker.device,
            ),
            position_ids=position_ids,
            past_key_values=batched_cache,
            cache_position=torch.tensor([past_length], device=self.talker.device),
            use_cache=True,
        )
        logits = self.talker.codec_head(outputs.last_hidden_state)
        generated_at = time.perf_counter()
        frames: list[GeneratedCodecFrame] = []
        for row, request in enumerate(requests):
            request.past_key_values = _slice_cache(outputs.past_key_values, row)
            request.cache_length = past_length + 1
            request.past_hidden = outputs.last_hidden_state[row : row + 1, -1:, :]
            frames.append(
                GeneratedCodecFrame(
                    request_id=request.request_id,
                    codes=codec_codes[row].detach(),
                    generated_at=generated_at,
                    sequence_index=len(request.generated_tokens) - 1,
                )
            )
            token = self._choose_token(
                logits[row : row + 1],
                request,
                request.sampling,
                history=request.generated_tokens,
                enforce_minimum=True,
            )
            request.next_token = token
            request.generated_tokens.append(token)
        finished_at = time.perf_counter()
        self.last_decode_metrics = {
            "totalMs": (finished_at - started_at) * 1_000,
            "predictorMs": (predictor_finished_at - predictor_started_at) * 1_000,
            "cacheMs": (cache_finished_at - cache_started_at) * 1_000,
            "talkerMs": (finished_at - talker_started_at) * 1_000,
        }
        return frames

    def _decode_with_cuda_graph(
        self,
        request: _NativeRequest,
        condition: torch.Tensor,
    ) -> list[GeneratedCodecFrame]:
        """Run the exact Qwen predictor and Talker modules from fixed GPU buffers."""
        if self.predictor_graph is None or self.talker_graph is None:
            raise RuntimeError("CUDA graph executor is unavailable")
        if request.next_token is None or request.past_hidden is None:
            raise RuntimeError("decode request was not prefetched")
        past_length = _request_cache_length(request)
        if past_length >= self.talker_graph.max_sequence_length:
            self.deactivate_cuda_graph()
            raise RuntimeError("request is too long for the Talker CUDA graph")
        if self.graph_owner is not request:
            self.deactivate_cuda_graph()
            loaded_length = self.talker_graph.load_dynamic_cache(request.past_key_values)
            if loaded_length != past_length:
                raise RuntimeError(
                    f"Talker cache length mismatch: {loaded_length} != {past_length}"
                )
            self.graph_owner = request

        started_at = time.perf_counter()
        main_tokens = torch.tensor(
            [[request.next_token]],
            dtype=torch.long,
            device=self.talker.device,
        )
        first_codebook_hidden = self.talker.get_input_embeddings()(main_tokens)
        predictor_started_at = time.perf_counter()
        predictor_sequences = self.predictor_graph.run(
            torch.cat([request.past_hidden, first_codebook_hidden], dim=1)
        )
        predictor_finished_at = time.perf_counter()
        codec_codes = torch.cat([main_tokens, predictor_sequences], dim=1)
        codec_embeddings = [first_codebook_hidden]
        codec_embeddings.extend(
            self.talker.code_predictor.get_input_embeddings()[index](
                predictor_sequences[:, index : index + 1]
            )
            for index in range(self.config.num_code_groups - 1)
        )
        inputs_embeds = torch.cat(codec_embeddings, dim=1).sum(dim=1, keepdim=True)
        inputs_embeds = inputs_embeds + condition
        talker_started_at = time.perf_counter()
        hidden = self.talker_graph.run(inputs_embeds, position=past_length)
        logits = self.talker.codec_head(hidden)
        generated_at = time.perf_counter()
        frame = GeneratedCodecFrame(
            request_id=request.request_id,
            codes=codec_codes[0].detach(),
            generated_at=generated_at,
            sequence_index=len(request.generated_tokens) - 1,
        )
        request.past_hidden = hidden.clone()
        request.cache_length = past_length + 1
        token = self._choose_token(
            logits,
            request,
            request.sampling,
            history=request.generated_tokens,
            enforce_minimum=True,
        )
        request.next_token = token
        request.generated_tokens.append(token)
        finished_at = time.perf_counter()
        self.last_decode_metrics = {
            "totalMs": (finished_at - started_at) * 1_000,
            "predictorMs": (predictor_finished_at - predictor_started_at) * 1_000,
            "cacheMs": 0.0,
            "talkerMs": (finished_at - talker_started_at) * 1_000,
            "cudaGraph": True,
        }
        return [frame]

    def deactivate_cuda_graph(self) -> None:
        """Copy the graph-owned KV state back before using eager batching."""
        if self.graph_owner is None or self.talker_graph is None:
            return
        self.talker_graph.save_dynamic_cache(
            self.graph_owner.past_key_values,
            self.graph_owner.cache_length,
        )
        self.graph_owner = None

    @torch.inference_mode()
    def release_request(self, request: _NativeRequest) -> None:
        if self.graph_owner is request:
            self.graph_owner = None
            if self.talker_graph is not None:
                self.talker_graph.static_cache.reset()

    def _decode_with_official_talker(
        self,
        requests: list[_NativeRequest],
        conditions: list[torch.Tensor],
        main_tokens: torch.Tensor,
        past_length: int,
    ) -> list[GeneratedCodecFrame]:
        """Run Qwen's supported recurrent step as one real GPU batch.

        Keeping this path inside the upstream Talker wrapper is important: it
        remains the byte-for-byte correctness oracle for deterministic
        generation while the scheduler owns request admission and lifecycle.
        """
        started_at = time.perf_counter()
        batch_size = len(requests)
        cache_started_at = time.perf_counter()
        batched_cache = _batch_caches(
            [request.past_key_values for request in requests]
        )
        cache_finished_at = time.perf_counter()
        talker_started_at = time.perf_counter()
        outputs = self.talker(
            input_ids=main_tokens,
            attention_mask=torch.ones(
                (batch_size, past_length + 1),
                dtype=torch.long,
                device=self.talker.device,
            ),
            past_key_values=batched_cache,
            cache_position=torch.tensor([past_length], device=self.talker.device),
            past_hidden=torch.cat(
                [request.past_hidden for request in requests], dim=0
            ),
            generation_step=0,
            trailing_text_hidden=torch.cat(conditions, dim=0),
            tts_pad_embed=torch.cat(
                [request.prompt.text_padding_embedding for request in requests], dim=0
            ),
            use_cache=True,
            output_hidden_states=True,
            subtalker_dosample=False,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
        )
        codec_codes = outputs.hidden_states[1]
        generated_at = time.perf_counter()
        frames: list[GeneratedCodecFrame] = []
        for row, request in enumerate(requests):
            request.past_key_values = _slice_cache(outputs.past_key_values, row)
            request.cache_length = past_length + 1
            request.past_hidden = outputs.past_hidden[row : row + 1]
            frames.append(
                GeneratedCodecFrame(
                    request_id=request.request_id,
                    codes=codec_codes[row].detach(),
                    generated_at=generated_at,
                    sequence_index=len(request.generated_tokens) - 1,
                )
            )
            token = self._choose_token(
                outputs.logits[row : row + 1],
                request,
                request.sampling,
                history=request.generated_tokens,
                enforce_minimum=True,
            )
            request.next_token = token
            request.generated_tokens.append(token)
        finished_at = time.perf_counter()
        self.last_decode_metrics = {
            "totalMs": (finished_at - started_at) * 1_000,
            "predictorMs": 0.0,
            "cacheMs": (cache_finished_at - cache_started_at) * 1_000,
            "talkerMs": (finished_at - talker_started_at) * 1_000,
        }
        return frames

    def _predict_secondary_codebooks(
        self,
        requests: list[_NativeRequest],
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        predictor = self.talker.code_predictor
        # Keep the deterministic path byte-identical to Qwen's supported
        # generation loop. It is still one real predictor batch across all
        # selected requests. The explicit sampler below exists for stochastic
        # request-local generators, which Hugging Face generate() cannot accept.
        if all(not request.subtalker_sampling.do_sample for request in requests):
            result = predictor.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            return result.sequences
        output = predictor(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones(
                inputs_embeds.shape[:2],
                dtype=torch.long,
                device=inputs_embeds.device,
            ),
            cache_position=torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device),
            use_cache=True,
        )
        tokens: list[torch.Tensor] = []
        total = self.config.num_code_groups - 1
        for step in range(total):
            selected = []
            for row, request in enumerate(requests):
                selected.append(
                    self._choose_token(
                        output.logits[row : row + 1, -1:, :],
                        request,
                        request.subtalker_sampling,
                        history=[],
                        enforce_minimum=False,
                    )
                )
            token = torch.tensor(
                selected,
                dtype=torch.long,
                device=inputs_embeds.device,
            ).unsqueeze(1)
            tokens.append(token)
            if step + 1 == total:
                break
            position = inputs_embeds.shape[1] + step
            output = predictor(
                input_ids=token,
                attention_mask=torch.ones(
                    (len(requests), position + 1),
                    dtype=torch.long,
                    device=inputs_embeds.device,
                ),
                past_key_values=output.past_key_values,
                cache_position=torch.tensor([position], device=inputs_embeds.device),
                generation_steps=output.generation_steps,
                use_cache=True,
            )
        return torch.cat(tokens, dim=1)

    def _choose_token(
        self,
        logits: torch.Tensor,
        request: _NativeRequest,
        sampling: NativeSamplingConfig,
        *,
        history: list[int],
        enforce_minimum: bool,
    ) -> int:
        scores = logits[:, -1, :].float().clone()
        if enforce_minimum:
            scores[:, self.suppressed_tokens] = -torch.inf
            if history and not math.isclose(sampling.repetition_penalty, 1.0):
                previous = torch.tensor([history], device=scores.device, dtype=torch.long)
                previous_scores = torch.gather(scores, 1, previous)
                previous_scores = torch.where(
                    previous_scores < 0,
                    previous_scores * sampling.repetition_penalty,
                    previous_scores / sampling.repetition_penalty,
                )
                scores.scatter_(1, previous, previous_scores)
            if len(history) < 2:
                scores[:, self.eos_token_id] = -torch.inf
        if not sampling.do_sample:
            return int(torch.argmax(scores, dim=-1).item())

        temperature = max(float(sampling.temperature), 1e-5)
        scores = scores / temperature
        if sampling.top_k > 0 and sampling.top_k < scores.shape[-1]:
            threshold = torch.topk(scores, sampling.top_k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < threshold, -torch.inf)
        if 0.0 < sampling.top_p < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
            remove = cumulative > sampling.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
            filtered = torch.full_like(scores, -torch.inf)
            scores = filtered.scatter(1, sorted_indices, sorted_scores)
        probabilities = torch.softmax(scores, dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=request.generator).item())


class NativeContinuousBatchingManager:
    """Background request scheduler with appendable text and true GPU batches."""

    def __init__(
        self,
        talker,
        registry: StreamingRequestRegistry,
        *,
        max_requests_per_batch: int,
        sampling: NativeSamplingConfig,
        subtalker_sampling: NativeSamplingConfig,
        frame_callback: Callable[[GeneratedCodecFrame], None] | None = None,
        trace_callback: TraceCallback | None = None,
        enable_cuda_graphs: bool = True,
        cuda_graph_max_sequence_length: int = 2_048,
    ) -> None:
        self.registry = registry
        self.executor = NativeQwenTalkerExecutor(
            talker,
            subtalker_sampling=subtalker_sampling,
            enable_cuda_graphs=enable_cuda_graphs,
            cuda_graph_max_sequence_length=cuda_graph_max_sequence_length,
        )
        self.max_requests_per_batch = max(1, int(max_requests_per_batch))
        self.default_sampling = sampling
        self.default_subtalker_sampling = subtalker_sampling
        self.frame_callback = frame_callback
        self.trace_callback = trace_callback
        self.background_thread_status = NativeBackgroundThreadStatus()
        self._requests: dict[str, _NativeRequest] = {}
        self._results: dict[str, deque[NativeGenerationResult]] = defaultdict(deque)
        self._condition = Condition()
        self._thread: Thread | None = None
        self._stopping = False
        self._schedule_order = 0

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._thread = Thread(target=self._run, name="qwen-native-scheduler", daemon=True)
            self._thread.start()

    def stop(self, *, block: bool = True, hard_stop: bool = False) -> None:
        del hard_stop
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread
        if block and thread is not None:
            thread.join()
        with self._condition:
            self._thread = None

    def add_request(
        self,
        request_id: str,
        prompt: PreparedStreamingPrompt,
        *,
        max_new_tokens: int,
        seed: int | None = None,
    ) -> str:
        generator = torch.Generator(device=self.executor.talker.device)
        generator.manual_seed(seed if seed is not None else secrets.randbits(63))
        with self._condition:
            if request_id in self._requests:
                raise ValueError(f"request {request_id!r} already exists")
            self._requests[request_id] = _NativeRequest(
                request_id=request_id,
                prompt=prompt,
                max_new_tokens=int(max_new_tokens),
                sampling=self.default_sampling,
                subtalker_sampling=self.default_subtalker_sampling,
                generator=generator,
            )
            self._trace(
                request_id,
                "scheduler.admitted",
                maxNewTokens=int(max_new_tokens),
                promptTokens=prompt.length,
            )
            self._trace(
                request_id,
                "request.state",
                state=NativeRequestStatus.PREFILLING.value,
            )
            self._condition.notify_all()
        return request_id

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def cancel_request(self, request_id: str) -> None:
        with self._condition:
            request = self._requests.get(request_id)
            if request is None or request.status in _TERMINAL_STATUSES:
                return
            self.executor.release_request(request)
            self._set_status(request, NativeRequestStatus.CANCELLED)
            self._publish_locked(request)
            self._condition.notify_all()

    def release_request(self, request_id: str) -> None:
        with self._condition:
            request = self._requests.pop(request_id, None)
            if request is not None:
                self.executor.release_request(request)
            self._results.pop(request_id, None)

    def get_result(self, request_id: str, timeout: float | None = None) -> NativeGenerationResult | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._results[request_id]:
                if request_id not in self._requests:
                    return None
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(timeout=remaining)
            return self._results[request_id].popleft()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(self._has_work_or_stop)
                    if self._stopping:
                        return
                    prefill_batch = self._select_prefill_locked()
                    decode_batch, conditions = ([], [])
                    if not prefill_batch:
                        decode_batch, conditions = self._select_decode_locked()
                if prefill_batch:
                    self._run_prefill(prefill_batch)
                elif decode_batch:
                    self._run_decode(decode_batch, conditions)
        except Exception as error:
            with self._condition:
                self.background_thread_status.fatal_error = error
                for request in self._requests.values():
                    if request.status not in _TERMINAL_STATUSES:
                        self._set_status(
                            request,
                            NativeRequestStatus.FAILED,
                            error=str(error),
                        )
                        self._publish_locked(request, error=str(error))
                self._condition.notify_all()

    def _has_work_or_stop(self) -> bool:
        if self._stopping:
            return True
        for request in self._requests.values():
            if request.status == NativeRequestStatus.PREFILLING:
                return True
            if request.status in {NativeRequestStatus.DECODING, NativeRequestStatus.WAITING_FOR_TEXT}:
                if self.registry.can_decode(request.request_id):
                    return True
        return False

    def _select_prefill_locked(self) -> list[_NativeRequest]:
        candidates = [
            request
            for request in self._requests.values()
            if request.status == NativeRequestStatus.PREFILLING
        ]
        if not candidates:
            return []
        length = candidates[0].prompt.length
        return [
            request for request in candidates if request.prompt.length == length
        ][: self.max_requests_per_batch]

    def _select_decode_locked(self) -> tuple[list[_NativeRequest], list[torch.Tensor]]:
        candidates = [
            request
            for request in self._requests.values()
            if request.status in {NativeRequestStatus.DECODING, NativeRequestStatus.WAITING_FOR_TEXT}
            and self.registry.can_decode(request.request_id)
            and request.past_key_values is not None
        ]
        if not candidates:
            return [], []
        anchor = min(
            candidates,
            key=lambda request: (request.last_scheduled_order, request.created_at),
        )
        length = _request_cache_length(anchor)
        selected = sorted(
            (
                request
                for request in candidates
                if _request_cache_length(request) == length
            ),
            key=lambda request: (request.last_scheduled_order, request.created_at),
        )[: self.max_requests_per_batch]
        self._schedule_order += 1
        for request in selected:
            request.last_scheduled_order = self._schedule_order
        conditions = []
        for request in selected:
            condition = self.registry.take_condition(request.request_id)
            if condition.kind == StreamConditionKind.TEXT:
                token = torch.tensor(
                    [[condition.token_id]],
                    dtype=torch.long,
                    device=self.executor.talker.device,
                )
                conditions.append(
                    self.executor.talker.text_projection(
                        self.executor.talker.get_text_embeddings()(token)
                    )
                )
            elif condition.kind == StreamConditionKind.TEXT_END:
                conditions.append(request.prompt.text_end_embedding)
            else:
                conditions.append(request.prompt.text_padding_embedding)
            self._trace(
                request.request_id,
                "text.condition",
                kind=condition.kind.value,
                tokenId=condition.token_id,
            )
            self._set_status(request, NativeRequestStatus.DECODING)
        return selected, conditions

    def _run_prefill(self, requests: list[_NativeRequest]) -> None:
        try:
            with torch.inference_mode():
                self.executor.prefill(requests)
            request_ids = [request.request_id for request in requests]
            for request in requests:
                self._trace(
                    request.request_id,
                    "model.prefill",
                    requestIds=request_ids,
                    batchSize=len(requests),
                    **self.executor.last_prefill_metrics,
                )
            with self._condition:
                for request in requests:
                    if request.status == NativeRequestStatus.CANCELLED:
                        continue
                    if request.next_token == self.executor.eos_token_id:
                        self._set_status(request, NativeRequestStatus.COMPLETED)
                        self._publish_locked(request)
                    elif self.registry.can_decode(request.request_id):
                        self._set_status(request, NativeRequestStatus.DECODING)
                    else:
                        self._set_status(request, NativeRequestStatus.WAITING_FOR_TEXT)
                self._condition.notify_all()
        except Exception as error:
            self._fail_requests(requests, error)

    def _run_decode(
        self,
        requests: list[_NativeRequest],
        conditions: list[torch.Tensor],
    ) -> None:
        try:
            with torch.inference_mode():
                frames = self.executor.decode(requests, conditions)
            request_ids = [request.request_id for request in requests]
            for request in requests:
                self._trace(
                    request.request_id,
                    "model.step",
                    requestIds=request_ids,
                    batchSize=len(requests),
                    **self.executor.last_decode_metrics,
                )
            for frame in frames:
                self._trace(
                    frame.request_id,
                    "codec.frame",
                    sequence=frame.sequence_index,
                    codeCount=int(frame.codes.numel()),
                )
            if self.frame_callback is not None:
                for frame in frames:
                    self.frame_callback(frame)
            with self._condition:
                for request in requests:
                    if request.status == NativeRequestStatus.CANCELLED:
                        continue
                    reached_eos = request.next_token == self.executor.eos_token_id
                    reached_limit = len(request.generated_tokens) >= request.max_new_tokens
                    if reached_eos or reached_limit:
                        self.executor.release_request(request)
                        self._set_status(request, NativeRequestStatus.COMPLETED)
                        self._publish_locked(request)
                    elif self.registry.can_decode(request.request_id):
                        self._set_status(request, NativeRequestStatus.DECODING)
                    else:
                        self._set_status(request, NativeRequestStatus.WAITING_FOR_TEXT)
                self._condition.notify_all()
        except Exception as error:
            self._fail_requests(requests, error)

    def _fail_requests(self, requests: list[_NativeRequest], error: Exception) -> None:
        with self._condition:
            for request in requests:
                if request.status not in _TERMINAL_STATUSES:
                    self.executor.release_request(request)
                    self._set_status(request, NativeRequestStatus.FAILED, error=str(error))
                    self._publish_locked(request, error=str(error))
            self._condition.notify_all()

    def _publish_locked(self, request: _NativeRequest, *, error: str | None = None) -> None:
        self._results[request.request_id].append(
            NativeGenerationResult(
                request_id=request.request_id,
                status=request.status,
                generated_tokens=tuple(request.generated_tokens),
                error=error,
            )
        )
        self._condition.notify_all()

    def _set_status(
        self,
        request: _NativeRequest,
        status: NativeRequestStatus,
        **data,
    ) -> None:
        if request.status == status:
            return
        previous = request.status
        request.status = status
        self._trace(
            request.request_id,
            "request.state",
            previous=previous.value,
            state=status.value,
            **data,
        )

    def _trace(self, request_id: str, event: str, **data) -> None:
        if self.trace_callback is not None:
            self.trace_callback(
                StreamingTraceEvent(
                    request_id=request_id,
                    event=event,
                    data=data,
                )
            )
