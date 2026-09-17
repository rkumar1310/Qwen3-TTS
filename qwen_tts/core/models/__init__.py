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
__all__ = [
    "Qwen3TTSConfig",
    "Qwen3TTSForConditionalGeneration",
    "Qwen3TTSProcessor",
]


def __getattr__(name):
    if name == "Qwen3TTSConfig":
        from .configuration_qwen3_tts import Qwen3TTSConfig

        return Qwen3TTSConfig
    if name == "Qwen3TTSForConditionalGeneration":
        from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration

        return Qwen3TTSForConditionalGeneration
    if name == "Qwen3TTSProcessor":
        from .processing_qwen3_tts import Qwen3TTSProcessor

        return Qwen3TTSProcessor
    raise AttributeError(name)
