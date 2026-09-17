"""CUDA-graph helpers for the latency-sensitive Qwen3-TTS decode loop.

The graph path deliberately executes Qwen's own Transformer modules.  It only
replaces dynamic cache allocation and Python dispatch with fixed buffers, so it
does not use the predictor re-prefill approximation that can alter speech.
"""

from __future__ import annotations

import torch
from transformers import StaticCache


def _sample_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    top_k: int,
    top_p: float,
    temperature: float,
) -> torch.Tensor:
    """Sample a batch without bringing token ids back to the CPU."""
    scores = logits.float()
    if not do_sample:
        return torch.argmax(scores, dim=-1)
    scores = scores / max(float(temperature), 1e-5)
    if 0 < top_k < scores.shape[-1]:
        threshold = torch.topk(scores, int(top_k), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        remove = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1) > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(
            1,
            sorted_indices,
            sorted_scores,
        )
    return torch.multinomial(torch.softmax(scores, dim=-1), 1).squeeze(-1)


class PredictorCudaGraph:
    """Capture Qwen's complete 15-codebook predictor as one GPU graph."""

    def __init__(
        self,
        predictor,
        *,
        talker_hidden_size: int,
        dtype: torch.dtype,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> None:
        self.predictor = predictor
        self.model = predictor.model
        self.config = predictor.config
        self.device = predictor.device
        self.dtype = dtype
        self.num_codebooks = int(self.config.num_code_groups) - 1
        self.max_sequence_length = 2 + self.num_codebooks
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.static_cache = StaticCache(
            config=self.model.config,
            max_cache_len=self.max_sequence_length,
        )
        self.input_buffer = torch.zeros(
            1,
            2,
            talker_hidden_size,
            dtype=dtype,
            device=self.device,
        )
        self.output_tokens = torch.zeros(
            1,
            self.num_codebooks,
            dtype=torch.long,
            device=self.device,
        )
        self.prefill_position = torch.arange(2, device=self.device)
        self.decode_positions = [
            torch.tensor([2 + index], device=self.device)
            for index in range(self.num_codebooks - 1)
        ]
        self.prefill_mask = None
        self.decode_masks: list[dict[str, torch.Tensor]] = []
        self.graph: torch.cuda.CUDAGraph | None = None

    def _initialize_cache(self) -> None:
        config = self.model.config
        heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dummy = torch.zeros(
            1,
            heads,
            1,
            head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy, dummy)

    def _attention_mask(
        self,
        input_embeddings: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del input_embeddings
        key_positions = torch.arange(self.max_sequence_length, device=self.device)
        allowed = key_positions.unsqueeze(0) <= cache_position.unsqueeze(1)
        minimum = torch.finfo(self.dtype).min
        full = torch.where(
            allowed,
            torch.zeros((), dtype=self.dtype, device=self.device),
            torch.full((), minimum, dtype=self.dtype, device=self.device),
        ).unsqueeze(0).unsqueeze(0)
        masks = {"full_attention": full}
        if "sliding_attention" in getattr(self.model.config, "layer_types", []):
            window = int(self.model.config.sliding_window)
            in_window = key_positions.unsqueeze(0) > cache_position.unsqueeze(1) - window
            masks["sliding_attention"] = torch.where(
                allowed & in_window,
                torch.zeros((), dtype=self.dtype, device=self.device),
                torch.full((), minimum, dtype=self.dtype, device=self.device),
            ).unsqueeze(0).unsqueeze(0)
        return masks

    def _build_masks(self) -> None:
        self.prefill_mask = self._attention_mask(self.input_buffer, self.prefill_position)
        one_token = self.input_buffer[:, :1]
        self.decode_masks = [
            self._attention_mask(one_token, position) for position in self.decode_positions
        ]

    def _forward(self) -> None:
        hidden = self.predictor.small_to_mtp_projection(self.input_buffer)
        output = self.model(
            inputs_embeds=hidden,
            attention_mask=self.prefill_mask,
            past_key_values=self.static_cache,
            cache_position=self.prefill_position,
            use_cache=True,
        )
        hidden = output.last_hidden_state
        logits = self.predictor.lm_head[0](hidden[:, -1, :])
        token = _sample_logits(
            logits,
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        self.output_tokens[:, 0].copy_(token)

        embeddings = self.predictor.get_input_embeddings()
        for codebook in range(1, self.num_codebooks):
            hidden = embeddings[codebook - 1](token.unsqueeze(1))
            hidden = self.predictor.small_to_mtp_projection(hidden)
            output = self.model(
                inputs_embeds=hidden,
                attention_mask=self.decode_masks[codebook - 1],
                past_key_values=self.static_cache,
                cache_position=self.decode_positions[codebook - 1],
                use_cache=True,
            )
            logits = self.predictor.lm_head[codebook](output.last_hidden_state[:, -1, :])
            token = _sample_logits(
                logits,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            self.output_tokens[:, codebook].copy_(token)

    @torch.inference_mode()
    def capture(self, *, warmups: int = 3) -> None:
        self._initialize_cache()
        self._build_masks()
        for _ in range(warmups):
            self.static_cache.reset()
            self._forward()
        torch.cuda.synchronize()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            self.static_cache.reset()
            self._forward()
            torch.cuda.synchronize()
            self.static_cache.reset()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._forward()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

    @torch.inference_mode()
    def run(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("predictor CUDA graph has not been captured")
        self.input_buffer.copy_(input_embeddings)
        self.static_cache.reset()
        self.graph.replay()
        return self.output_tokens


class TalkerCudaGraph:
    """Persistent single-request Talker cache with graphed one-token decode."""

    def __init__(
        self,
        model,
        *,
        dtype: torch.dtype,
        max_sequence_length: int,
    ) -> None:
        self.model = model
        self.config = model.config
        self.device = model.device
        self.dtype = dtype
        self.max_sequence_length = int(max_sequence_length)
        self.static_cache = StaticCache(
            config=self.config,
            max_cache_len=self.max_sequence_length,
        )
        self.input_buffer = torch.zeros(
            1,
            1,
            self.config.hidden_size,
            dtype=dtype,
            device=self.device,
        )
        self.output_buffer = torch.zeros_like(self.input_buffer)
        self.cache_position = torch.zeros(1, dtype=torch.long, device=self.device)
        self.position_ids = torch.zeros(3, 1, 1, dtype=torch.long, device=self.device)
        self.attention_mask = torch.zeros(
            1,
            1,
            1,
            self.max_sequence_length,
            dtype=dtype,
            device=self.device,
        )
        key_positions = torch.arange(self.max_sequence_length, device=self.device)
        self.mask_table = torch.empty(
            self.max_sequence_length,
            self.max_sequence_length,
            dtype=dtype,
            device=self.device,
        )
        minimum = torch.finfo(dtype).min
        positions = torch.arange(self.max_sequence_length, device=self.device).unsqueeze(1)
        allowed = key_positions.unsqueeze(0) <= positions
        sliding_window = getattr(self.config, "sliding_window", None)
        if sliding_window is not None:
            allowed &= key_positions.unsqueeze(0) > positions - int(sliding_window)
        self.mask_table.copy_(
            torch.where(
                allowed,
                torch.zeros((), dtype=dtype, device=self.device),
                torch.full((), minimum, dtype=dtype, device=self.device),
            )
        )
        self.graph: torch.cuda.CUDAGraph | None = None

    def _initialize_cache(self) -> None:
        heads = getattr(
            self.config,
            "num_key_value_heads",
            self.config.num_attention_heads,
        )
        head_dim = getattr(
            self.config,
            "head_dim",
            self.config.hidden_size // self.config.num_attention_heads,
        )
        dummy = torch.zeros(
            1,
            heads,
            1,
            head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        for layer in self.static_cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy, dummy)

    def _forward(self) -> None:
        output = self.model(
            inputs_embeds=self.input_buffer,
            attention_mask=self.attention_mask,
            past_key_values=self.static_cache,
            cache_position=self.cache_position,
            position_ids=self.position_ids,
            use_cache=True,
        )
        self.output_buffer.copy_(output.last_hidden_state)

    def _set_position(self, position: int) -> None:
        if position >= self.max_sequence_length:
            raise RuntimeError(
                f"Talker sequence reached CUDA graph capacity {self.max_sequence_length}"
            )
        self.cache_position.fill_(position)
        self.position_ids.fill_(position)
        self.attention_mask[0, 0, 0].copy_(self.mask_table[position])

    @torch.inference_mode()
    def capture(self, *, warmups: int = 3) -> None:
        self._initialize_cache()
        self._set_position(min(100, self.max_sequence_length - 1))
        for _ in range(warmups):
            self._forward()
        torch.cuda.synchronize()
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            self._forward()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._forward()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        self.static_cache.reset()

    @torch.inference_mode()
    def load_dynamic_cache(self, dynamic_cache: object) -> int:
        self.static_cache.reset()
        sequence_length = int(dynamic_cache.get_seq_length())
        if sequence_length >= self.max_sequence_length:
            raise RuntimeError("prefill is too large for the Talker CUDA graph")
        for index, (source, target) in enumerate(
            zip(dynamic_cache.layers, self.static_cache.layers, strict=True)
        ):
            del index
            if source.keys is None or source.values is None:
                continue
            target.update(source.keys, source.values)
        return sequence_length

    @torch.inference_mode()
    def save_dynamic_cache(self, dynamic_cache: object, sequence_length: int) -> None:
        for source, target in zip(
            self.static_cache.layers,
            dynamic_cache.layers,
            strict=True,
        ):
            if not source.is_initialized:
                continue
            if getattr(source, "is_sliding", False):
                target.keys = source.keys.clone()
                target.values = source.values.clone()
                if hasattr(target, "cumulative_length"):
                    target.cumulative_length = int(sequence_length)
            else:
                target.keys = source.keys[:, :, :sequence_length, :].clone()
                target.values = source.values[:, :, :sequence_length, :].clone()
            target.is_initialized = True

    @torch.inference_mode()
    def run(self, input_embeddings: torch.Tensor, *, position: int) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("Talker CUDA graph has not been captured")
        self.input_buffer.copy_(input_embeddings)
        self._set_position(position)
        self.graph.replay()
        return self.output_buffer
