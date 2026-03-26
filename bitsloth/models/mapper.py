# Copyright 2023-present Daniel Han-Chen & the Bitsloth team. All rights reserved.
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
    "INT_TO_FLOAT_MAPPER",
    "FLOAT_TO_INT_MAPPER",
    "MAP_TO_BITSLOTH_16bit",
    "FLOAT_TO_FP8_BLOCK_MAPPER",
    "FLOAT_TO_FP8_ROW_MAPPER",
]

__INT_TO_FLOAT_MAPPER = \
{
    "bitsloth/mistral-7b-bnb-4bit" : (
        "bitsloth/mistral-7b",
        "mistralai/Mistral-7B-v0.1",
    ),
    "bitsloth/llama-2-7b-bnb-4bit" : (
        "bitsloth/llama-2-7b",
        "meta-llama/Llama-2-7b-hf",
    ),
    "bitsloth/llama-2-13b-bnb-4bit" : (
        "bitsloth/llama-2-13b",
        "meta-llama/Llama-2-13b-hf",
    ),
    "bitsloth/codellama-34b-bnb-4bit" : (
        "codellama/CodeLlama-34b-hf",
    ),
    "bitsloth/zephyr-sft-bnb-4bit" : (
        "bitsloth/zephyr-sft",
        "HuggingFaceH4/mistral-7b-sft-beta",
    ),
    "bitsloth/tinyllama-bnb-4bit" : (
        "bitsloth/tinyllama",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    ),
    "bitsloth/tinyllama-chat-bnb-4bit" : (
        "bitsloth/tinyllama-chat",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    "bitsloth/mistral-7b-instruct-v0.1-bnb-4bit" : (
        "bitsloth/mistral-7b-instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ),
    "bitsloth/mistral-7b-instruct-v0.2-bnb-4bit" : (
        "bitsloth/mistral-7b-instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ),
    "bitsloth/llama-2-7b-chat-bnb-4bit" : (
        "bitsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "bitsloth/llama-2-7b-chat-bnb-4bit" : (
        "bitsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "bitsloth/Mixtral-8x7B-v0.1-bitsloth-bnb-4bit" : (
        "bitsloth/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x7B-v0.1",
        "bitsloth/Mixtral-8x7B-v0.1-bnb-4bit",
    ),
    "bitsloth/Mixtral-8x7B-Instruct-v0.1-bitsloth-bnb-4bit" : (
        "bitsloth/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "bitsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
    ),
    "bitsloth/codellama-7b-bnb-4bit" : (
        "bitsloth/codellama-7b",
        "codellama/CodeLlama-7b-hf",
    ),
    "bitsloth/codellama-13b-bnb-4bit" : (
        "codellama/CodeLlama-13b-hf",
    ),
    "bitsloth/yi-6b-bnb-4bit" : (
        "bitsloth/yi-6b",
        "01-ai/Yi-6B",
    ),
    "bitsloth/solar-10.7b-bnb-4bit" : (
        "upstage/SOLAR-10.7B-v1.0",
    ),
    "bitsloth/gemma-7b-bnb-4bit" : (
        "bitsloth/gemma-7b",
        "google/gemma-7b",
    ),
    "bitsloth/gemma-2b-bnb-4bit" : (
        "bitsloth/gemma-2b",
        "google/gemma-2b",
    ),
    "bitsloth/gemma-7b-it-bnb-4bit" : (
        "bitsloth/gemma-7b-it",
        "google/gemma-7b-it",
    ),
    "bitsloth/gemma-2b-bnb-4bit" : (
        "bitsloth/gemma-2b-it",
        "google/gemma-2b-it",
    ),
    "bitsloth/mistral-7b-v0.2-bnb-4bit" : (
        "bitsloth/mistral-7b-v0.2",
        "alpindale/Mistral-7B-v0.2-hf",
    ),
    "bitsloth/gemma-1.1-2b-it-bnb-4bit" : (
        "bitsloth/gemma-1.1-2b-it",
        "google/gemma-1.1-2b-it",
    ),
    "bitsloth/gemma-1.1-7b-it-bnb-4bit" : (
        "bitsloth/gemma-1.1-7b-it",
        "google/gemma-1.1-7b-it",
    ),
    "bitsloth/Starling-LM-7B-beta" : (
        "bitsloth/Starling-LM-7B-beta",
        "Nexusflow/Starling-LM-7B-beta",
    ),
    "bitsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit" : (
        "bitsloth/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
    ),
    "bitsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit" : (
        "bitsloth/OpenHermes-2.5-Mistral-7B",
        "teknium/OpenHermes-2.5-Mistral-7B",
    ),
    "bitsloth/codegemma-2b-bnb-4bit" : (
        "bitsloth/codegemma-2b",
        "google/codegemma-2b",
    ),
    "bitsloth/codegemma-7b-bnb-4bit" : (
        "bitsloth/codegemma-7b",
        "google/codegemma-7b",
    ),
    "bitsloth/codegemma-7b-it-bnb-4bit" : (
        "bitsloth/codegemma-7b-it",
        "google/codegemma-7b-it",
    ),
    "bitsloth/llama-3-8b-bnb-4bit" : (
        "bitsloth/llama-3-8b",
        "meta-llama/Meta-Llama-3-8B",
    ),
    "bitsloth/llama-3-8b-Instruct-bnb-4bit" : (
        "bitsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ),
    "bitsloth/llama-3-70b-bnb-4bit" : (
        "meta-llama/Meta-Llama-3-70B",
    ),
    "bitsloth/llama-3-70b-Instruct-bnb-4bit" : (
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "bitsloth/Phi-3-mini-4k-instruct-bnb-4bit" : (
        "bitsloth/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
    ),
    "bitsloth/mistral-7b-v0.3-bnb-4bit" : (
        "bitsloth/mistral-7b-v0.3",
        "mistralai/Mistral-7B-v0.3",
    ),
    "bitsloth/mistral-7b-instruct-v0.3-bnb-4bit" : (
        "bitsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "bitsloth/Phi-3-medium-4k-instruct-bnb-4bit" : (
        "bitsloth/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
    ),
    "bitsloth/Qwen2-0.5B-bnb-4bit" : (
        "bitsloth/Qwen2-0.5B",
        "Qwen/Qwen2-0.5B",
    ),
    "bitsloth/Qwen2-0.5B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
    ),
    "bitsloth/Qwen2-1.5B-bnb-4bit" : (
        "bitsloth/Qwen2-1.5B",
        "Qwen/Qwen2-1.5B",
    ),
    "bitsloth/Qwen2-1.5B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
    ),
    "bitsloth/Qwen2-7B-bnb-4bit" : (
        "bitsloth/Qwen2-7B",
        "Qwen/Qwen2-7B",
    ),
    "bitsloth/Qwen2-7B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ),
    "bitsloth/Qwen2-70B-bnb-4bit" : (
        "Qwen/Qwen2-70B",
    ),
    "bitsloth/Qwen2-70B-Instruct-bnb-4bit" : (
        "Qwen/Qwen2-70B-Instruct",
    ),
    "mistralai/Codestral-22B-v0.1" : (
        "mistral-community/Codestral-22B-v0.1",
    ),
    "bitsloth/gemma-2-9b-bnb-4bit" : (
        "bitsloth/gemma-2-9b",
        "google/gemma-2-9b",
    ),
    "bitsloth/gemma-2-27b-bnb-4bit" : (
        "bitsloth/gemma-2-27b",
        "google/gemma-2-27b",
    ),
    "bitsloth/gemma-2-9b-it-bnb-4bit" : (
        "bitsloth/gemma-2-9b-it",
        "google/gemma-2-9b-it",
    ),
    "bitsloth/gemma-2-27b-it-bnb-4bit" : (
        "bitsloth/gemma-2-27b-it",
        "google/gemma-2-27b-it",
    ),
    "bitsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit" : ( # Old Phi pre July
        "bitsloth/Phi-3-mini-4k-instruct-v0",
    ),
    "bitsloth/Mistral-Nemo-Instruct-2407-bnb-4bit" : ( # New 12b Mistral models
        "bitsloth/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ),
    "bitsloth/Mistral-Nemo-Base-2407-bnb-4bit" : ( # New 12b Mistral models
        "bitsloth/Mistral-Nemo-Base-2407",
        "mistralai/Mistral-Nemo-Base-2407",
    ),
    "bitsloth/Meta-Llama-3.1-8B-bitsloth-bnb-4bit" : (
        "bitsloth/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B",
        "bitsloth/Meta-Llama-3.1-8B-bnb-4bit",
    ),
    "bitsloth/Meta-Llama-3.1-8B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "RedHatAI/Llama-3.1-8B-Instruct-FP8",
            "bitsloth/Llama-3.1-8B-Instruct-FP8-Block",
            "bitsloth/Llama-3.1-8B-Instruct-FP8-Dynamic",
        ),
        "16" : (
            "bitsloth/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "bitsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Llama-3.1-8B-bitsloth-bnb-4bit" : (
        "bitsloth/Llama-3.1-8B",
        "meta-llama/Llama-3.1-8B",
        "bitsloth/Llama-3.1-8B-bnb-4bit",
    ),
    "bitsloth/Llama-3.1-8B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "RedHatAI/Llama-3.1-8B-Instruct-FP8",
            "bitsloth/Llama-3.1-8B-Instruct-FP8-Block",
            "bitsloth/Llama-3.1-8B-Instruct-FP8-Dynamic",
        ),
        "16" : (
            "bitsloth/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "bitsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Meta-Llama-3.1-70B-bnb-4bit" : (
        "bitsloth/Meta-Llama-3.1-70B",
        "meta-llama/Meta-Llama-3.1-70B",
    ),
    "bitsloth/Meta-Llama-3.1-405B-bnb-4bit" : (
        "meta-llama/Meta-Llama-3.1-405B",
    ),
    "bitsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit" : (
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
    ),
    "bitsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" : (
        "bitsloth/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    "bitsloth/Mistral-Large-Instruct-2407-bnb-4bit" : (
        "mistralai/Mistral-Large-Instruct-2407",
    ),
    "bitsloth/gemma-2-2b-bnb-4bit" : (
        "bitsloth/gemma-2-2b",
        "google/gemma-2-2b",
    ),
    "bitsloth/gemma-2-2b-it-bnb-4bit" : (
        "bitsloth/gemma-2-2b-it",
        "google/gemma-2-2b-it",
    ),
    "bitsloth/Phi-3.5-mini-instruct-bnb-4bit" : (
        "bitsloth/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ),
    "bitsloth/c4ai-command-r-08-2024-bnb-4bit" : (
        "CohereForAI/c4ai-command-r-08-2024",
    ),
    "bitsloth/c4ai-command-r-plus-08-2024-bnb-4bit" : (
        "CohereForAI/c4ai-command-r-plus-08-2024",
    ),
    "bitsloth/Llama-3.1-Storm-8B-bnb-4bit" : (
        "bitsloth/Llama-3.1-Storm-8B",
        "akjindal53244/Llama-3.1-Storm-8B",
    ),
    "bitsloth/Hermes-3-Llama-3.1-8B-bnb-4bit" : (
        "bitsloth/Hermes-3-Llama-3.1-8B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
    ),
    "bitsloth/Hermes-3-Llama-3.1-70B-bnb-4bit" : (
        "bitsloth/Hermes-3-Llama-3.1-70B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
    ),
    "bitsloth/Hermes-3-Llama-3.1-405B-bnb-4bit" : (
        "NousResearch/Hermes-3-Llama-3.1-405B",
    ),
    "bitsloth/SmolLM-135M-bnb-4bit" : (
        "bitsloth/SmolLM-135M",
        "HuggingFaceTB/SmolLM-135M",
    ),
    "bitsloth/SmolLM-360M-bnb-4bit" : (
        "bitsloth/SmolLM-360M",
        "HuggingFaceTB/SmolLM-360M",
    ),
    "bitsloth/SmolLM-1.7B-bnb-4bit" : (
        "bitsloth/SmolLM-1.7B",
        "HuggingFaceTB/SmolLM-1.7B",
    ),
    "bitsloth/SmolLM-135M-Instruct-bnb-4bit" : (
        "bitsloth/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-135M-Instruct",
    ),
    "bitsloth/SmolLM-360M-Instruct-bnb-4bit" : (
        "bitsloth/SmolLM-360M-Instruct",
        "HuggingFaceTB/SmolLM-360M-Instruct",
    ),
    "bitsloth/SmolLM-1.7B-Instruct-bnb-4bit" : (
        "bitsloth/SmolLM-1.7B-Instruct",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
    ),
    "bitsloth/Mistral-Small-Instruct-2409-bnb-4bit" : (
        "bitsloth/Mistral-Small-Instruct-2409",
        "mistralai/Mistral-Small-Instruct-2409",
    ),
    "bitsloth/Qwen2.5-0.5B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "bitsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-1.5B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "bitsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-3B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "bitsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-7B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "bitsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-14B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "bitsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-32B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
    ),
    "bitsloth/Qwen2.5-72B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ),
    "bitsloth/Qwen2.5-0.5B-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B",
        "bitsloth/Qwen2.5-0.5B-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-1.5B-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B",
        "bitsloth/Qwen2.5-1.5B-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-3B-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-3B",
        "Qwen/Qwen2.5-3B",
        "bitsloth/Qwen2.5-3B-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-7B-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B",
        "bitsloth/Qwen2.5-7B-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-14B-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-14B",
        "Qwen/Qwen2.5-14B",
        "bitsloth/Qwen2.5-14B-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-32B-bnb-4bit" : (
        "bitsloth/Qwen2.5-32B",
        "Qwen/Qwen2.5-32B",
    ),
    "bitsloth/Qwen2.5-72B-bnb-4bit" : (
        "bitsloth/Qwen2.5-72B",
        "Qwen/Qwen2.5-72B",
    ),
    "bitsloth/Qwen2.5-Math-1.5B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Math-1.5B",
        "Qwen/Qwen2.5-Math-1.5B",
    ),
    "bitsloth/Qwen2.5-Math-7B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Math-7B",
        "Qwen/Qwen2.5-Math-7B",
    ),
    "bitsloth/Qwen2.5-Math-72B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Math-72B",
        "Qwen/Qwen2.5-Math-72B",
    ),
    "bitsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
    ),
    "bitsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    ),
    "bitsloth/Qwen2.5-Math-72B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Math-72B-Instruct",
        "Qwen/Qwen2.5-Math-72B-Instruct",
    ),
    "bitsloth/Qwen2.5-Coder-0.5B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-0.5B",
        "Qwen/Qwen2.5-Coder-0.5B",
    ),
    "bitsloth/Qwen2.5-Coder-1.5B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-1.5B",
    ),
    "bitsloth/Qwen2.5-Coder-3B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-3B",
        "Qwen/Qwen2.5-Coder-3B",
    ),
    "bitsloth/Qwen2.5-Coder-7B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-7B",
        "Qwen/Qwen2.5-Coder-7B",
    ),
    "bitsloth/Qwen2.5-Coder-14B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-14B",
        "Qwen/Qwen2.5-Coder-14B",
    ),
    "bitsloth/Qwen2.5-Coder-32B-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-32B",
        "Qwen/Qwen2.5-Coder-32B",
    ),
    "bitsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    ),
    "bitsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    ),
    "bitsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
    ),
    "bitsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ),
    "bitsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ),
    "bitsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    "bitsloth/Llama-3.2-1B-bitsloth-bnb-4bit" : (
        "bitsloth/Llama-3.2-1B",
        "meta-llama/Llama-3.2-1B",
        "bitsloth/Llama-3.2-1B-bnb-4bit",
    ),
    "bitsloth/Llama-3.2-3B-bitsloth-bnb-4bit" : (
        "bitsloth/Llama-3.2-3B",
        "meta-llama/Llama-3.2-3B",
        "bitsloth/Llama-3.2-3B-bnb-4bit",
    ),
    "bitsloth/Llama-3.2-1B-Instruct-bitsloth-bnb-4bit" : {
        "8": (
            "RedHatAI/Llama-3.2-1B-Instruct-FP8",
            "bitsloth/Llama-3.2-1B-Instruct-FP8-Block",
            "bitsloth/Llama-3.2-1B-Instruct-FP8-Dynamic",
        ),
        "16" : (
            "bitsloth/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "bitsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Llama-3.2-3B-Instruct-bitsloth-bnb-4bit" : {
        "8": (
            "RedHatAI/Llama-3.2-3B-Instruct-FP8",
            "bitsloth/Llama-3.2-3B-Instruct-FP8-Block",
            "bitsloth/Llama-3.2-3B-Instruct-FP8-Dynamic",
        ),
        "16" : (
            "bitsloth/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "bitsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit" : (
        "bitsloth/Llama-3.1-Nemotron-70B-Instruct",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    ),
    "bitsloth/Qwen2-VL-2B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2-VL-2B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
        "bitsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2-VL-7B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "bitsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2-VL-72B-Instruct-bnb-4bit" : (
        "bitsloth/Qwen2-VL-72B-Instruct",
        "Qwen/Qwen2-VL-72B-Instruct",
    ),
    "bitsloth/Qwen2-VL-2B-bnb-4bit" : (
        "bitsloth/Qwen2-VL-2B",
        "Qwen/Qwen2-VL-2B",
    ),
    "bitsloth/Qwen2-VL-7B-bnb-4bit" : (
        "bitsloth/Qwen2-VL-7B",
        "Qwen/Qwen2-VL-7B",
    ),
    "bitsloth/Qwen2-VL-72B-bnb-4bit" : (
        "bitsloth/Qwen2-VL-72B",
        "Qwen/Qwen2-VL-72B",
    ),
    "bitsloth/Llama-3.2-11B-Vision-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "bitsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    ),
    "bitsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit" : (
        "bitsloth/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
    ),
    "bitsloth/Llama-3.2-11B-Vision-bitsloth-bnb-4bit" : (
        "bitsloth/Llama-3.2-11B-Vision",
        "meta-llama/Llama-3.2-11B-Vision",
        "bitsloth/Llama-3.2-11B-Vision-bnb-4bit",
    ),
    "bitsloth/Llama-3.2-90B-Vision-bnb-4bit" : (
        "bitsloth/Llama-3.2-90B-Vision",
        "meta-llama/Llama-3.2-90B-Vision",
    ),
    "bitsloth/Pixtral-12B-2409-bitsloth-bnb-4bit" : (
        "bitsloth/Pixtral-12B-2409",
        "mistralai/Pixtral-12B-2409",
        "bitsloth/Pixtral-12B-2409-bnb-4bit",
    ),
    "bitsloth/Pixtral-12B-2409-Base-bnb-4bit" : (
        "bitsloth/Pixtral-12B-Base-2409",
        "mistralai/Pixtral-12B-Base-2409",
    ),
    "bitsloth/llava-1.5-7b-hf-bnb-4bit" : (
        "bitsloth/llava-1.5-7b-hf",
        "llava-hf/llava-1.5-7b-hf",
    ),
    "bitsloth/llava-v1.6-mistral-7b-hf-bnb-4bit" : (
        "bitsloth/llava-v1.6-mistral-7b-hf",
        "llava-hf/llava-v1.6-mistral-7b-hf",
    ),
    "bitsloth/Llama-3.1-Tulu-3-8B-bnb-4bit" : (
        "bitsloth/Llama-3.1-Tulu-3-8B",
        "allenai/Llama-3.1-Tulu-3-8B",
    ),
    "bitsloth/Llama-3.1-Tulu-3-70B-bnb-4bit" : (
        "bitsloth/Llama-3.1-Tulu-3-70B",
        "allenai/Llama-3.1-Tulu-3-70B",
    ),
    "bitsloth/QwQ-32B-Preview-bnb-4bit" : (
        "bitsloth/QwQ-32B-Preview",
        "Qwen/QwQ-32B-Preview",
    ),
    "bitsloth/Llama-3.3-70B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "RedHatAI/Llama-3.3-70B-Instruct-FP8",
            "bitsloth/Llama-3.3-70B-Instruct-FP8-Block",
            "bitsloth/Llama-3.3-70B-Instruct-FP8-Dynamic",
        ),
        "16" : (
            "bitsloth/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "bitsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/phi-4-bitsloth-bnb-4bit" : (
        "bitsloth/phi-4",
        "microsoft/phi-4",
        "bitsloth/phi-4-bnb-4bit",
    ),
    "bitsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    ),
    "bitsloth/DeepSeek-R1-Distill-Qwen-14B-bitsloth-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "bitsloth/DeepSeek-R1-Distill-Qwen-14B-bnb-4bit",
    ),
    "bitsloth/DeepSeek-R1-Distill-Qwen-7B-bitsloth-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "bitsloth/DeepSeek-R1-Distill-Qwen-7B-bnb-4bit",
    ),
    "bitsloth/DeepSeek-R1-Distill-Qwen-1.5B-bitsloth-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "bitsloth/DeepSeek-R1-Distill-Qwen-1.5B-bnb-4bit",
    ),
    "bitsloth/DeepSeek-R1-Distill-Llama-8B-bitsloth-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "bitsloth/DeepSeek-R1-Distill-Llama-8B-bnb-4bit",
    ),
    "bitsloth/DeepSeek-R1-Distill-Llama-70B-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    ),
    "bitsloth/Mistral-Small-24B-Base-2501-bitsloth-bnb-4bit" : (
        "bitsloth/Mistral-Small-24B-Base-2501",
        "mistralai/Mistral-Small-24B-Base-2501",
        "bitsloth/Mistral-Small-24B-Base-2501-bnb-4bit",
    ),
    "bitsloth/Mistral-Small-24B-Instruct-2501-bitsloth-bnb-4bit" : (
        "bitsloth/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "bitsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-VL-3B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "bitsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-VL-7B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "bitsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-VL-32B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "bitsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
    ),
    "bitsloth/Qwen2.5-VL-72B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "bitsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
    ),
    "bitsloth/DeepScaleR-1.5B-Preview-bitsloth-bnb-4bit" : (
        "bitsloth/DeepHermes-3-Llama-3-8B-Preview",
        "agentica-org/DeepScaleR-1.5B-Preview",
        "bitsloth/DeepScaleR-1.5B-Preview-bnb-4bit",
    ),
    "bitsloth/OpenThinker-7B-bitsloth-bnb-4bit" : (
        "bitsloth/OpenThinker-7B",
        "open-thoughts/OpenThinker-7B",
        "bitsloth/OpenThinker-7B-bnb-4bit",
    ),
    "bitsloth/granite-3.2-2b-instruct-bitsloth-bnb-4bit" : (
        "bitsloth/granite-3.2-2b-instruct",
        "ibm-granite/granite-3.2-2b-instruct",
        "bitsloth/granite-3.2-2b-instruct-bnb-4bit",
    ),
    "bitsloth/granite-3.2-8b-instruct-bitsloth-bnb-4bit" : (
        "bitsloth/granite-3.2-8b-instruct",
        "ibm-granite/granite-3.2-8b-instruct",
        "bitsloth/granite-3.2-8b-instruct-bnb-4bit",
    ),
    "bitsloth/QwQ-32B-bitsloth-bnb-4bit" : (
        "bitsloth/QwQ-32B",
        "Qwen/QwQ-32B",
        "bitsloth/QwQ-32B-bnb-4bit",
    ),
    "bitsloth/gemma-3-1b-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-1b-it",
        "google/gemma-3-1b-it",
        "bitsloth/gemma-3-1b-it-bnb-4bit",
    ),
    "bitsloth/gemma-3-4b-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "bitsloth/gemma-3-4b-it-bnb-4bit",
    ),
    "bitsloth/gemma-3-12b-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-12b-it",
        "google/gemma-3-12b-it",
        "bitsloth/gemma-3-12b-it-bnb-4bit",
    ),
    "bitsloth/gemma-3-27b-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-27b-it",
        "google/gemma-3-27b-it",
        "bitsloth/gemma-3-27b-it-bnb-4bit",
    ),
    "bitsloth/gemma-3-1b-pt-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-1b-pt",
        "google/gemma-3-1b-pt",
        "bitsloth/gemma-3-1b-pt-bnb-4bit",
    ),
    "bitsloth/gemma-3-4b-pt-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-4b-pt",
        "google/gemma-3-4b-pt",
        "bitsloth/gemma-3-4b-pt-bnb-4bit",
    ),
    "bitsloth/gemma-3-12b-pt-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-12b-pt",
        "google/gemma-3-12b-pt",
        "bitsloth/gemma-3-12b-pt-bnb-4bit",
    ),
    "bitsloth/gemma-3-27b-pt-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-27b-pt",
        "google/gemma-3-27b-pt",
        "bitsloth/gemma-3-27b-pt-bnb-4bit",
    ),
    "bitsloth/reka-flash-3-bitsloth-bnb-4bit" : (
        "bitsloth/reka-flash-3",
        "RekaAI/reka-flash-3",
        "bitsloth/reka-flash-3-bnb-4bit",
    ),
    "bitsloth/c4ai-command-a-03-2025-bitsloth-bnb-4bit" : (
        "bitsloth/c4ai-command-a-03-2025",
        "CohereForAI/c4ai-command-a-03-2025",
        "bitsloth/c4ai-command-a-03-2025-bnb-4bit",
    ),
    "bitsloth/aya-vision-32b-bitsloth-bnb-4bit" : (
        "bitsloth/aya-vision-32b",
        "CohereForAI/aya-vision-32b",
        "bitsloth/aya-vision-32b-bnb-4bit",
    ),
    "bitsloth/aya-vision-8b-bitsloth-bnb-4bit" : (
        "bitsloth/aya-vision-8b",
        "CohereForAI/aya-vision-8b",
        "bitsloth/aya-vision-8b-bnb-4bit",
    ),
    "bitsloth/granite-vision-3.2-2b-bitsloth-bnb-4bit" : (
        "bitsloth/granite-vision-3.2-2b",
        "ibm-granite/granite-vision-3.2-2b",
        "bitsloth/granite-vision-3.2-2b-bnb-4bit",
    ),
    "bitsloth/OLMo-2-0325-32B-Instruct-bitsloth-bnb-4bit" : (
        "bitsloth/OLMo-2-0325-32B-Instruct",
        "allenai/OLMo-2-0325-32B-Instruct",
        "bitsloth/OLMo-2-0325-32B-Instruct-bnb-4bit",
    ),
    "bitsloth/Mistral-Small-3.1-24B-Instruct-2503-bitsloth-bnb-4bit" : (
        "bitsloth/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "bitsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
    ),
    "bitsloth/Mistral-Small-3.1-24B-Base-2503-bitsloth-bnb-4bit" : (
        "bitsloth/Mistral-Small-3.1-24B-Base-2503",
        "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "bitsloth/Mistral-Small-3.1-24B-Base-2503-bnb-4bit",
    ),
    "bitsloth/Qwen3-0.6B-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-0.6B-FP8",
            "bitsloth/Qwen3-0.6B-FP8",
            "bitsloth/Qwen3-0.6B-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-0.6B",
            "Qwen/Qwen3-0.6B",
            "bitsloth/Qwen3-0.6B-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-1.7B-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-1.7B-FP8",
            "bitsloth/Qwen3-1.7B-FP8",
            "bitsloth/Qwen3-1.7B-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-1.7B",
            "Qwen/Qwen3-1.7B",
            "bitsloth/Qwen3-1.7B-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-4B-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-4B-FP8",
            "bitsloth/Qwen3-4B-FP8",
            "bitsloth/Qwen3-4B-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-4B",
            "Qwen/Qwen3-4B",
            "bitsloth/Qwen3-4B-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-8B-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-8B-FP8",
            "bitsloth/Qwen3-8B-FP8",
            "bitsloth/Qwen3-8B-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-8B",
            "Qwen/Qwen3-8B",
            "bitsloth/Qwen3-8B-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-14B-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-14B-FP8",
            "bitsloth/Qwen3-14B-FP8",
            "bitsloth/Qwen3-14B-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-14B",
            "Qwen/Qwen3-14B",
            "bitsloth/Qwen3-14B-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-32B-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-32B-FP8",
            "bitsloth/Qwen3-32B-FP8",
            "bitsloth/Qwen3-32B-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-32B",
            "Qwen/Qwen3-32B",
            "bitsloth/Qwen3-32B-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-30B-A3B-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B",
        "bitsloth/Qwen3-30B-A3B-bnb-4bit",
    ),
    "bitsloth/Qwen3-0.6B-Base-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen3-0.6B-Base",
        "Qwen/Qwen3-0.6B-Base",
        "bitsloth/Qwen3-0.6B-Base-bnb-4bit",
    ),
    "bitsloth/Qwen3-1.7B-Base-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen3-1.7B-Base",
        "Qwen/Qwen3-1.7B-Base",
        "bitsloth/Qwen3-1.7B-Base-bnb-4bit",
    ),
    "bitsloth/Qwen3-4B-Base-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen3-4B-Base",
        "Qwen/Qwen3-4B-Base",
        "bitsloth/Qwen3-4B-Base-bnb-4bit",
    ),
    "bitsloth/Qwen3-8B-Base-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen3-8B-Base",
        "Qwen/Qwen3-8B-Base",
        "bitsloth/Qwen3-8B-Base-bnb-4bit",
    ),
    "bitsloth/Qwen3-14B-Base-bitsloth-bnb-4bit" : (
        "bitsloth/Qwen3-14B-Base",
        "Qwen/Qwen3-14B-Base",
        "bitsloth/Qwen3-14B-Base-bnb-4bit",
    ),
    "bitsloth/Qwen3-30B-A3B-Base-bnb-4bit" : (
        "bitsloth/Qwen3-30B-A3B-Base",
        "Qwen/Qwen3-30B-A3B-Base",
    ),
    "bitsloth/phi-4-reasoning-bitsloth-bnb-4bit" : (
        "bitsloth/phi-4-reasoning",
        "microsoft/Phi-4-reasoning",
        "bitsloth/phi-4-reasoning-bnb-4bit",
    ),
    "bitsloth/phi-4-reasoning-plus-bitsloth-bnb-4bit" : (
        "bitsloth/phi-4-reasoning-plus",
        "microsoft/Phi-4-reasoning-plus",
        "bitsloth/phi-4-reasoning-plus-bnb-4bit",
    ),
    "bitsloth/phi-4-mini-reasoning-bitsloth-bnb-4bit" : (
        "bitsloth/phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-reasoning",
        "bitsloth/phi-4-mini-reasoning-bnb-4bit",
    ),
    "bitsloth/Phi-4-mini-instruct-bitsloth-bnb-4bit" : (
        "bitsloth/Phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "bitsloth/Phi-4-mini-instruct-bnb-4bit",
    ),
    "bitsloth/orpheus-3b-0.1-pretrained-bitsloth-bnb-4bit" : (
        "bitsloth/orpheus-3b-0.1-pretrained",
        "canopylabs/orpheus-3b-0.1-pretrained",
        "bitsloth/orpheus-3b-0.1-pretrained-bnb-4bit",
    ),
    "bitsloth/orpheus-3b-0.1-ft-bitsloth-bnb-4bit" : (
        "bitsloth/orpheus-3b-0.1-ft",
        "canopylabs/orpheus-3b-0.1-ft",
        "bitsloth/orpheus-3b-0.1-ft-bnb-4bit",
    ),
    "bitsloth/csm-1b" : (
        "bitsloth/csm-1b",
        "sesame/csm-1b",
    ),
    "bitsloth/whisper-large-v3" : (
        "bitsloth/whisper-large-v3",
        "openai/whisper-large-v3",
    ),
    "bitsloth/whisper-large-v3-turbo" : (
        "bitsloth/whisper-large-v3-turbo",
        "openai/whisper-large-v3-turbo",
    ),
    "bitsloth/whisper-small" : (
        "bitsloth/whisper-small",
        "openai/whisper-small",
    ),
    "bitsloth/CrisperWhisper" : (
        "bitsloth/CrisperWhisper",
        "nyrahealth/CrisperWhisper",
    ),
    "bitsloth/Llasa-1B" : (
        "bitsloth/Llasa-1B",
        "HKUSTAudio/Llasa-1B",
    ),
    "bitsloth/Spark-TTS-0.5B" : (
        "bitsloth/Spark-TTS-0.5B",
        "SparkAudio/Spark-TTS-0.5B",
    ),
    "bitsloth/Llama-OuteTTS-1.0-1B" : (
        "bitsloth/Llama-OuteTTS-1.0-1B",
        "OuteAI/Llama-OuteTTS-1.0-1B",
    ),
    "bitsloth/medgemma-4b-it-bitsloth-bnb-4bit" : (
        "bitsloth/medgemma-4b-it",
        "google/medgemma-4b-it",
        "bitsloth/medgemma-4b-it-bnb-4bit",
    ),
    "bitsloth/medgemma-27b-text-it-bitsloth-bnb-4bit" : (
        "bitsloth/medgemma-27b-text-it",
        "google/medgemma-27b-text-it",
        "bitsloth/medgemma-27b-text-it-bnb-4bit",
    ),
    "bitsloth/Devstral-Small-2505-bitsloth-bnb-4bit" : (
        "bitsloth/Devstral-Small-2505",
        "mistralai/Devstral-Small-2505",
        "bitsloth/Devstral-Small-2505-bnb-4bit",
    ),
    "bitsloth/DeepSeek-R1-0528-Qwen3-8B-bitsloth-bnb-4bit" : (
        "bitsloth/DeepSeek-R1-0528-Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "bitsloth/DeepSeek-R1-0528-Qwen3-8B-bnb-4bit",
    ),
    "bitsloth/Magistral-Small-2506-bitsloth-bnb-4bit" : (
        "bitsloth/Magistral-Small-2506",
        "mistralai/Magistral-Small-2506",
        "bitsloth/Magistral-Small-2506-bnb-4bit",
    ),
    "bitsloth/Mistral-Small-3.2-24B-Instruct-2506-bitsloth-bnb-4bit" : {
        "8" : (
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "bitsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8",
            "bitsloth/Mistral-Small-3.2-24B-Instruct-2506-FP8",
        ),
        "16" : (
            "bitsloth/Mistral-Small-3.2-24B-Instruct-2506",
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            "bitsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
        ),
    },
    "bitsloth/gemma-3n-E4B-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3n-E4B-it",
        "google/gemma-3n-E4B-it",
        "bitsloth/gemma-3n-E4B-it-bitsloth-bnb-4bit",
    ),
    "bitsloth/gemma-3n-E2B-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3n-E2B-it",
        "google/gemma-3n-E2B-it",
        "bitsloth/gemma-3n-E2B-it-bitsloth-bnb-4bit",
    ),
    "bitsloth/gemma-3n-E4B-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3n-E4B",
        "google/gemma-3n-E4B",
        "bitsloth/gemma-3n-E4B-bitsloth-bnb-4bit",
    ),
    "bitsloth/gemma-3n-E2B-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3n-E2B",
        "google/gemma-3n-E2B",
        "bitsloth/gemma-3n-E2B-bitsloth-bnb-4bit",
    ),
    "bitsloth/Devstral-Small-2507-bitsloth-bnb-4bit" : (
        "bitsloth/Devstral-Small-2507",
        "mistralai/Devstral-Small-2507",
        "bitsloth/Devstral-Small-2507-bnb-4bit",
    ),
    "bitsloth/Qwen3-30B-A3B-Thinking-2507" : (
        "bitsloth/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
    ),
    "bitsloth/Qwen3-30B-A3B-Instruct-2507" : (
        "bitsloth/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
    ),
    "bitsloth/Qwen3-Coder-30B-A3B-Instruct" : (
        "bitsloth/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    ),
    "bitsloth/gpt-oss-20b-bitsloth-bnb-4bit" : (
        "bitsloth/gpt-oss-20b",
        "openai/gpt-oss-20b",
        "bitsloth/gpt-oss-20b-bitsloth-bnb-4bit",
    ),
    "bitsloth/gpt-oss-120b-bitsloth-bnb-4bit" : (
        "bitsloth/gpt-oss-120b",
        "openai/gpt-oss-120b",
        "bitsloth/gpt-oss-120b-bitsloth-bnb-4bit",
    ),
    "bitsloth/Qwen3-4B-Instruct-2507-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-4B-Instruct-2507-FP8",
            "bitsloth/Qwen3-4B-Instruct-2507-FP8",
            "bitsloth/Qwen3-4B-Instruct-2507-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-4B-Instruct-2507",
            "bitsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-4B-Thinking-2507-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-4B-Thinking-2507-FP8",
            "bitsloth/Qwen3-4B-Thinking-2507-FP8",
            "bitsloth/Qwen3-4B-Thinking-2507-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-4B-Thinking-2507",
            "Qwen/Qwen3-4B-Thinking-2507",
            "bitsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        ),
    },
    "bitsloth/gemma-3-270m-it-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-270m-it",
        "google/gemma-3-270m-it",
        "bitsloth/gemma-3-270m-it-bnb-4bit",
    ),
    "bitsloth/gemma-3-270m-bitsloth-bnb-4bit" : (
        "bitsloth/gemma-3-270m",
        "google/gemma-3-270m",
        "bitsloth/gemma-3-270m-bnb-4bit",
    ),
    "bitsloth/Magistral-Small-2507-bitsloth-bnb-4bit" : (
        "bitsloth/Magistral-Small-2507",
        "mistralai/Magistral-Small-2507",
        "bitsloth/Magistral-Small-2507-bnb-4bit",
    ),
    "bitsloth/Magistral-Small-2509-bitsloth-bnb-4bit" : {
        "8" : (
            "mistralai/Magistral-Small-2509",
            "bitsloth/Magistral-Small-2509-FP8-Dynamic",
            "bitsloth/Magistral-Small-2509-FP8-Dynamic",
        ),
        "16" : (
            "bitsloth/Magistral-Small-2509",
            "mistralai/Magistral-Small-2509",
            "bitsloth/Magistral-Small-2509-bnb-4bit",
        ),
    },
    "bitsloth/Apertus-70B-Instruct-2509-bitsloth-bnb-4bit" : (
        "bitsloth/Apertus-70B-Instruct-2509",
        "swiss-ai/Apertus-70B-2509",
        "bitsloth/Apertus-70B-Instruct-2509-bitsloth-bnb-4bit",
    ),
    "bitsloth/Apertus-8B-Instruct-2509-bitsloth-bnb-4bit" : (
        "bitsloth/Apertus-8B-Instruct-2509",
        "swiss-ai/Apertus-8B-2509",
        "bitsloth/Apertus-8B-Instruct-2509-bitsloth-bnb-4bit",
    ),
    "bitsloth/granite-4.0-micro-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-micro",
        "ibm-granite/granite-4.0-micro",
        "bitsloth/granite-4.0-micro-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-micro-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-h-micro",
        "ibm-granite/granite-4.0-h-micro",
        "bitsloth/granite-4.0-h-micro-bnb-4bit",
    ),
    "bitsloth/granite-4.0-micro-base-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-micro-base",
        "ibm-granite/granite-4.0-micro-base",
        "bitsloth/granite-4.0-micro-base-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-micro-base-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-h-micro-base",
        "ibm-granite/granite-4.0-h-micro-base",
        "bitsloth/granite-4.0-h-micro-base-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-tiny" : (
        "bitsloth/granite-4.0-h-tiny",
        "ibm-granite/granite-4.0-h-tiny",
    ),
    "bitsloth/granite-4.0-h-small" : (
        "bitsloth/granite-4.0-h-small",
        "ibm-granite/granite-4.0-h-small",
    ),
    "bitsloth/granite-4.0-h-tiny-base" : (
        "bitsloth/granite-4.0-h-tiny-base",
        "ibm-granite/granite-4.0-h-tiny-base",
    ),
    "bitsloth/granite-4.0-h-small-base" : (
        "bitsloth/granite-4.0-h-small-base",
        "ibm-granite/granite-4.0-h-small-base",
    ),
    "bitsloth/Qwen3-VL-4B-Thinking-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-4B-Thinking-FP8",
            "bitsloth/Qwen3-VL-4B-Thinking-FP8",
            "bitsloth/Qwen3-VL-4B-Thinking-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-4B-Thinking",
            "Qwen/Qwen3-VL-4B-Thinking",
            "bitsloth/Qwen3-VL-4B-Thinking-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-8B-Thinking-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-8B-Thinking-FP8",
            "bitsloth/Qwen3-VL-8B-Thinking-FP8",
            "bitsloth/Qwen3-VL-8B-Thinking-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
            "bitsloth/Qwen3-VL-8B-Thinking-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-4B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-4B-Instruct-FP8",
            "bitsloth/Qwen3-VL-4B-Instruct-FP8",
            "bitsloth/Qwen3-VL-4B-Instruct-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "bitsloth/Qwen3-VL-4B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-8B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-8B-Instruct-FP8",
            "bitsloth/Qwen3-VL-8B-Instruct-FP8",
            "bitsloth/Qwen3-VL-8B-Instruct-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "bitsloth/Qwen3-VL-8B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-2B-Thinking-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-2B-Thinking-FP8",
            "bitsloth/Qwen3-VL-2B-Thinking-FP8",
            "bitsloth/Qwen3-VL-2B-Thinking-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-2B-Thinking",
            "Qwen/Qwen3-VL-2B-Thinking",
            "bitsloth/Qwen3-VL-2B-Thinking-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-32B-Thinking-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-32B-Thinking-FP8",
            "bitsloth/Qwen3-VL-32B-Thinking-FP8",
            "bitsloth/Qwen3-VL-32B-Thinking-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-32B-Thinking",
            "Qwen/Qwen3-VL-32B-Thinking",
            "bitsloth/Qwen3-VL-32B-Thinking-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-2B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-2B-Instruct-FP8",
            "bitsloth/Qwen3-VL-2B-Instruct-FP8",
            "bitsloth/Qwen3-VL-2B-Instruct-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-2B-Instruct",
            "bitsloth/Qwen3-VL-2B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/Qwen3-VL-32B-Instruct-bitsloth-bnb-4bit" : {
        "8" : (
            "Qwen/Qwen3-VL-32B-Instruct-FP8",
            "bitsloth/Qwen3-VL-32B-Instruct-FP8",
            "bitsloth/Qwen3-VL-32B-Instruct-FP8",
        ),
        "16" : (
            "bitsloth/Qwen3-VL-32B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct",
            "bitsloth/Qwen3-VL-32B-Instruct-bnb-4bit",
        ),
    },
    "bitsloth/granite-4.0-350m-base-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-350m-base",
        "ibm-granite/granite-4.0-350m-base",
        "bitsloth/granite-4.0-350m-base-bnb-4bit",
    ),
    "bitsloth/granite-4.0-350m-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-350m",
        "ibm-granite/granite-4.0-350m",
        "bitsloth/granite-4.0-350m-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-350m-base-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-h-350m-base",
        "ibm-granite/granite-4.0-h-350m-base",
        "bitsloth/granite-4.0-h-350m-base-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-350m-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-h-350m",
        "ibm-granite/granite-4.0-h-350m",
        "bitsloth/granite-4.0-h-350m-bnb-4bit",
    ),
    "bitsloth/granite-4.0-1b-base-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-1b-base",
        "ibm-granite/granite-4.0-1b-base",
        "bitsloth/granite-4.0-1b-base-bnb-4bit",
    ),
    "bitsloth/granite-4.0-1b-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-1b",
        "ibm-granite/granite-4.0-1b",
        "bitsloth/granite-4.0-1b-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-1b-base-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-h-1b-base",
        "ibm-granite/granite-4.0-h-1b-base",
        "bitsloth/granite-4.0-h-1b-base-bnb-4bit",
    ),
    "bitsloth/granite-4.0-h-1b-bitsloth-bnb-4bit" : (
        "bitsloth/granite-4.0-h-1b",
        "ibm-granite/granite-4.0-h-1b",
        "bitsloth/granite-4.0-h-1b-bnb-4bit",
    ),
    "bitsloth/gpt-oss-safeguard-20b" : (
        "bitsloth/gpt-oss-safeguard-20b",
        "openai/gpt-oss-safeguard-20b",
    ),
    "bitsloth/gpt-oss-safeguard-120b" : (
        "bitsloth/gpt-oss-safeguard-120b",
        "openai/gpt-oss-safeguard-120b",
    ),
    "bitsloth/functiongemma-270m-it-bitsloth-bnb-4bit" : (
        "bitsloth/functiongemma-270m-it",
        "google/functiongemma-270m-it",
        "bitsloth/functiongemma-270m-it-bitsloth-bnb-4bit",
    ),
    # Ministral 3 models
    "bitsloth/Ministral-3-3B-Instruct-2512-bitsloth-bnb-4bit" : {
        "8" : (
            "mistralai/Ministral-3-3B-Instruct-2512",
            "bitsloth/Ministral-3-3B-Instruct-2512-FP8",
            "bitsloth/Ministral-3-3B-Instruct-2512-FP8",
        ),
        "16" : (
            "bitsloth/Ministral-3-3B-Instruct-2512",
            "mistralai/Ministral-3-3B-Instruct-2512",
            "bitsloth/Ministral-3-3B-Instruct-2512-bnb-4bit",
        ),
    },
    "bitsloth/Ministral-3-3B-Base-2512-bitsloth-bnb-4bit" : (
        "bitsloth/Ministral-3-3B-Base-2512",
        "mistralai/Ministral-3-3B-Base-2512",
        "bitsloth/Ministral-3-3B-Base-2512-bnb-4bit",
    ),
    "bitsloth/Ministral-3-3B-Reasoning-2512-bitsloth-bnb-4bit" : (
        "bitsloth/Ministral-3-3B-Reasoning-2512",
        "mistralai/Ministral-3-3B-Reasoning-2512",
        "bitsloth/Ministral-3-3B-Reasoning-2512-bnb-4bit",
    ),
    "bitsloth/Ministral-3-8B-Instruct-2512-bitsloth-bnb-4bit" : {
        "8" : (
            "mistralai/Ministral-3-8B-Instruct-2512",
            "bitsloth/Ministral-3-8B-Instruct-2512-FP8",
            "bitsloth/Ministral-3-8B-Instruct-2512-FP8",
        ),
        "16" : (
            "bitsloth/Ministral-3-8B-Instruct-2512",
            "mistralai/Ministral-3-8B-Instruct-2512",
            "bitsloth/Ministral-3-8B-Instruct-2512-bnb-4bit",
        ),
    },
    "bitsloth/Ministral-3-8B-Base-2512-bitsloth-bnb-4bit" : (
        "bitsloth/Ministral-3-8B-Base-2512",
        "mistralai/Ministral-3-8B-Base-2512",
        "bitsloth/Ministral-3-8B-Base-2512-bnb-4bit",
    ),
    "bitsloth/Ministral-3-8B-Reasoning-2512-bitsloth-bnb-4bit" : (
        "bitsloth/Ministral-3-8B-Reasoning-2512",
        "mistralai/Ministral-3-8B-Reasoning-2512",
        "bitsloth/Ministral-3-8B-Reasoning-2512-bnb-4bit",
    ),
    "bitsloth/Ministral-3-14B-Instruct-2512-bitsloth-bnb-4bit" : {
        "8" : (
            "mistralai/Ministral-3-14B-Instruct-2512",
            "bitsloth/Ministral-3-14B-Instruct-2512-FP8",
            "bitsloth/Ministral-3-14B-Instruct-2512-FP8",
        ),
        "16" : (
            "bitsloth/Ministral-3-14B-Instruct-2512",
            "mistralai/Ministral-3-14B-Instruct-2512",
            "bitsloth/Ministral-3-14B-Instruct-2512-bnb-4bit",
        ),
    },
    "bitsloth/Ministral-3-14B-Base-2512-bitsloth-bnb-4bit" : (
        "bitsloth/Ministral-3-14B-Base-2512",
        "mistralai/Ministral-3-14B-Base-2512",
        "bitsloth/Ministral-3-14B-Base-2512-bnb-4bit",
    ),
    "bitsloth/Ministral-3-14B-Reasoning-2512-bitsloth-bnb-4bit" : (
        "bitsloth/Ministral-3-14B-Reasoning-2512",
        "mistralai/Ministral-3-14B-Reasoning-2512",
        "bitsloth/Ministral-3-14B-Reasoning-2512-bnb-4bit",
    ),
    "bitsloth/Kimi-K2-Instruct-BF16" : (
        "bitsloth/Kimi-K2-Instruct",
    ),
}

INT_TO_FLOAT_MAPPER  = {}
FLOAT_TO_INT_MAPPER  = {}
MAP_TO_BITSLOTH_16bit = {}
FLOAT_TO_FP8_BLOCK_MAPPER = {}
FLOAT_TO_FP8_ROW_MAPPER   = {}


def _add_with_lower(mapper, key, value):
    if key is None:
        return
    mapper[key] = value
    mapper[key.lower()] = value


def _add_lower_only(mapper, key, value):
    if key is None:
        return
    mapper[key.lower()] = value

for key, values in __INT_TO_FLOAT_MAPPER.items():
    block, row = None, None
    if type(values) is dict:
        assert "16" in values
        float16_values = values["16"]
        # Float8 and other quantized types
        if "8" in values:
            float8_values = values["8"]
            assert len(float8_values) == 3
            official, block, row = float8_values
            _add_lower_only(FLOAT_TO_FP8_BLOCK_MAPPER, key, block)
            _add_lower_only(FLOAT_TO_FP8_ROW_MAPPER, key, row)
            _add_lower_only(FLOAT_TO_FP8_BLOCK_MAPPER, official + "-dynamic", block)
            _add_lower_only(FLOAT_TO_FP8_ROW_MAPPER, official, row)
            _add_lower_only(FLOAT_TO_FP8_ROW_MAPPER, official + "-dynamic", row)
            for k in float8_values + float16_values:
                _add_lower_only(FLOAT_TO_FP8_BLOCK_MAPPER, k, block)
                _add_lower_only(FLOAT_TO_FP8_ROW_MAPPER, k, row)

            if float8_values[1] is not None and float8_values[1].startswith("bitsloth"):
                for value in float8_values:
                    if value is not None:
                        _add_with_lower(MAP_TO_BITSLOTH_16bit, value, float8_values[1])

            for value in float8_values:
                if value is not None:
                    FLOAT_TO_INT_MAPPER[value] = key
                    FLOAT_TO_INT_MAPPER[value.lower()] = key.lower()
        values = float16_values
    INT_TO_FLOAT_MAPPER[key] = values[0]

    for value in values:
        FLOAT_TO_INT_MAPPER[value] = key

    # Map to Bitsloth version for 16bit versions
    if len(values) == 2:
        if values[0].startswith("bitsloth"):
            _add_with_lower(MAP_TO_BITSLOTH_16bit, values[1], values[0])
            _add_with_lower(MAP_TO_BITSLOTH_16bit, block, values[0])
            _add_with_lower(MAP_TO_BITSLOTH_16bit, row, values[0])
    elif len(values) == 3:
        # Dynamic Bitsloth quantization
        if values[0].startswith("bitsloth"):
            _add_with_lower(MAP_TO_BITSLOTH_16bit, values[1], values[0])
            _add_with_lower(MAP_TO_BITSLOTH_16bit, values[2], values[0])
            _add_with_lower(MAP_TO_BITSLOTH_16bit, block, values[0])
            _add_with_lower(MAP_TO_BITSLOTH_16bit, row, values[0])
        pass

    # Get lowercased
    lowered_key = key.lower()
    INT_TO_FLOAT_MAPPER[lowered_key] = values[0].lower()

    for value in values:
        FLOAT_TO_INT_MAPPER[value.lower()] = lowered_key
