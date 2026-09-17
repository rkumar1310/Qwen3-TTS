"""Hugging Face continuous-batching extensions for appendable TTS text."""

from __future__ import annotations

from transformers.generation.continuous_batching.requests import RequestStatus
from transformers.generation.continuous_batching.scheduler import FIFOScheduler

from .requests import StreamingRequestRegistry


class AppendableTextFIFOScheduler(FIFOScheduler):
    """Pause decode requests with no text while scheduling every ready peer."""

    def __init__(self, *args, registry: StreamingRequestRegistry, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.registry = registry

    def schedule_batch(self, token_budget: int, cache_budget: int):
        priority_states = []
        second_priority_states = []

        for state in self.active_requests.values():
            if state.status == RequestStatus.DECODING:
                if self.registry.can_decode(state.request_id):
                    priority_states.append(state)
            elif state.status == RequestStatus.PREFILLING:
                second_priority_states.append(state)

        if not self.block_new_requests:
            second_priority_states.extend(self._get_waiting_candidates())

        request_ids_to_remove_from_waiting: set[str] = set()
        scheduled, allocation_failed, decode_fast_path, num_q_tokens, max_kv_read = (
            self._process_candidates(
                priority_states + second_priority_states,
                token_budget,
                cache_budget,
                request_ids_to_remove_from_waiting,
            )
        )
        self._cleanup_waiting_queue(request_ids_to_remove_from_waiting)
        if not scheduled and allocation_failed:
            return None, decode_fast_path, 0, 0
        return scheduled, decode_fast_path, num_q_tokens, max_kv_read

    def has_pending_requests(self) -> bool:
        if self.waiting_requests:
            return True
        return any(
            state.status == RequestStatus.PREFILLING
            or (
                state.status == RequestStatus.DECODING
                and self.registry.can_decode(state.request_id)
            )
            for state in self.active_requests.values()
        )
