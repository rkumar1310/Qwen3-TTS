# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""qwen_tts: Qwen-TTS package."""

from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .inference.qwen3_tts_model import Qwen3TTSModel, VoiceClonePromptItem
    from .inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

__all__ = ["Qwen3TTSModel", "Qwen3TTSTokenizer", "VoiceClonePromptItem"]


def __getattr__(name: str):
    """Load the heavyweight audio inference dependencies only when requested."""
    if name in {"Qwen3TTSModel", "VoiceClonePromptItem"}:
        module = import_module(".inference.qwen3_tts_model", __name__)
        return getattr(module, name)
    if name == "Qwen3TTSTokenizer":
        module = import_module(".inference.qwen3_tts_tokenizer", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
