"""Continuous batching manager with appendable text-input requests."""

from __future__ import annotations

from typing import Any

from transformers.generation.continuous_batching.continuous_api import (
    ContinuousBatchingManager,
)

from .hf_scheduler import AppendableTextFIFOScheduler
from .model import BatchRequestContext, QwenStreamingTalkerAdapter
from .requests import StreamingRequestRegistry


class AppendableContinuousBatchingManager(ContinuousBatchingManager):
    """Keep Hugging Face paged batching while allowing live text appends."""

    def __init__(self, *args, registry: StreamingRequestRegistry | None = None, **kwargs) -> None:
        self.streaming_registry = registry or StreamingRequestRegistry()
        super().__init__(*args, **kwargs)

    def _create_batch_processor(self):
        processor = super()._create_batch_processor()
        if isinstance(processor.scheduler, AppendableTextFIFOScheduler):
            return processor

        scheduler = AppendableTextFIFOScheduler(
            cache=processor.cache,
            safety_margin=self.continuous_batching_config.safety_margin,
            max_requests_per_batch=self.continuous_batching_config.max_requests_per_batch,
            registry=self.streaming_registry,
        )
        processor.scheduler = scheduler
        processor.offloading_manager.scheduler = scheduler
        return processor

    def _generation_step(self) -> None:
        if self.batch_processor is None:
            raise RuntimeError("batch processor was not initialized")
        if not isinstance(self.model, QwenStreamingTalkerAdapter):
            raise TypeError("AppendableContinuousBatchingManager requires QwenStreamingTalkerAdapter")
        contexts = [
            BatchRequestContext(
                request_id=future.state.request_id,
                query_length=future.query_length,
                past_length=future.state.position_offset - future.query_length,
            )
            for future in self.batch_processor.inputs_and_outputs.requests_in_batch
        ]
        self.model.set_batch_context(contexts)
        try:
            super()._generation_step()
            self.model.flush_frame_callbacks()
        finally:
            self.model.clear_batch_context()

    def add_streaming_request(
        self,
        input_ids: list[int],
        *,
        request_id: str,
        remaining_text_tokens: list[int] | None = None,
        max_new_tokens: int | None = None,
        record_timestamps: bool = True,
        eos_token_id: int | list[int] | None = None,
        **logit_processor_kwargs: Any,
    ) -> str | None:
        """Submit the fixed Qwen prompt and register its remaining live text."""
        self.streaming_registry.create(request_id, remaining_text_tokens)
        accepted_id = super().add_request(
            input_ids,
            request_id=request_id,
            max_new_tokens=max_new_tokens,
            streaming=True,
            record_timestamps=record_timestamps,
            eos_token_id=eos_token_id,
            **logit_processor_kwargs,
        )
        if accepted_id is None:
            self.streaming_registry.remove(request_id)
        return accepted_id

    def append_text_tokens(self, request_id: str, token_ids: list[int]) -> None:
        self.streaming_registry.append(request_id, token_ids)
        self._has_new_requests.set()

    def close_text_input(self, request_id: str, final_token_ids: list[int] | None = None) -> None:
        self.streaming_registry.close_input(request_id, final_token_ids)
        self._has_new_requests.set()

    def cancel_request(self, request_id: str) -> None:
        self.streaming_registry.cancel(request_id)
        super().cancel_request(request_id)

    def release_streaming_request(self, request_id: str) -> None:
        """Discard text-side state after the acoustic request reaches a terminal state."""
        self.streaming_registry.remove(request_id)
