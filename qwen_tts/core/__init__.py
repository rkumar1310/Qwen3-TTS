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
from importlib import import_module
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    from .tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config
    from .tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model

__all__ = [
    "Qwen3TTSTokenizerV1Config",
    "Qwen3TTSTokenizerV1Model",
    "Qwen3TTSTokenizerV2Config",
    "Qwen3TTSTokenizerV2Model",
]


def __getattr__(name: str):
    modules = {
        "Qwen3TTSTokenizerV1Config": ".tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1",
        "Qwen3TTSTokenizerV1Model": ".tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1",
        "Qwen3TTSTokenizerV2Config": ".tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
        "Qwen3TTSTokenizerV2Model": ".tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
    }
    module_name = modules.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name, __name__), name)
