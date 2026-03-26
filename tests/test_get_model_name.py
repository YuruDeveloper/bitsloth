import unittest
from unittest.mock import patch
from bitsloth.models.loader_utils import get_model_name
from bitsloth.models import loader_utils
from bitsloth.models.mapper import FLOAT_TO_INT_MAPPER, MAP_TO_BITSLOTH_16bit


def _no_remote_mapper():
    return {}, {}, {}


class TestGetModelName(unittest.TestCase):
    def _assert_mapping(self, model_name, load_in_4bit, expected, should_change):
        mapped = get_model_name(model_name, load_in_4bit = load_in_4bit)
        self.assertEqual(mapped.lower(), expected.lower())
        if should_change:
            self.assertNotEqual(mapped.lower(), model_name.lower())
        else:
            self.assertEqual(mapped.lower(), model_name.lower())

    @patch.object(loader_utils, "_get_new_mapper", _no_remote_mapper)
    def test_resolution_matrix(self):
        cases = [
            # Core mappings
            ("meta-llama/Llama-2-7b-hf", True, bitsloth/llama-2-7b-bnb-4bit", True),
            ("meta-llama/Llama-2-7b-hf", False, bitsloth/llama-2-7b", True),
            (
                "mistralai/Ministral-8B-Instruct-2410",
                True,
                "mistralai/Ministral-8B-Instruct-2410",
                False,
            ),
            (
                "meta-llama/Llama-3.2-1B-Instruct",
                False,
                bitsloth/Llama-3.2-1B-Instruct",
                True,
            ),
            (
                "meta-llama/Llama-2-7b-chat-hf",
                True,
                bitsloth/llama-2-7b-chat-bnb-4bit",
                True,
            ),
            (
                "meta-llama/Llama-3.3-70B-Instruct",
                True,
                bitsloth/llama-3.3-70b-instruct-bitsloth-bnb-4bit",
                True,
            ),
            ("Qwen/Qwen3-8B", True, bitsloth/Qwen3-8B-bitsloth-bnb-4bit", True),
            ("Qwen/Qwen3-8B", False, bitsloth/Qwen3-8B", True),
            ("Qwen/Qwen3-8B-FP8", False, bitsloth/Qwen3-8B-FP8", True),
            ("Qwen/Qwen3-8B-FP8", True, bitsloth/Qwen3-8B-bitsloth-bnb-4bit", True),
            (
                "mistralai/Ministral-3-3B-Instruct-2512",
                True,
                bitsloth/Ministral-3-3B-Instruct-2512-bitsloth-bnb-4bit",
                True,
            ),
            (
                "mistralai/Ministral-3-3B-Instruct-2512",
                False,
                bitsloth/Ministral-3-3B-Instruct-2512",
                True,
            ),
            (bitsloth/Kimi-K2-Instruct", True, bitsloth/Kimi-K2-Instruct-BF16", True),
            (bitsloth/Kimi-K2-Instruct", False, bitsloth/Kimi-K2-Instruct", False),
            # Fallback-to-original behavior
            "nonexistent-user/nonexistent-model-123",
            "google/gemma-3-random-prototype-123",
            "imdatta0/nanoqwen-fp8",
            "imdatta0/nanoqwen-bf16",
            # Backward compatibility for legacy 4bit names
            (bitsloth/llama-2-7b-bnb-4bit", True, bitsloth/llama-2-7b-bnb-4bit", False),
            (bitsloth/llama-2-7b-bnb-4bit", False, bitsloth/llama-2-7b", True),
            ("google/gemma-2-9b", True, bitsloth/gemma-2-9b-bnb-4bit", True),
            # GPT-OSS behavior
            ("openai/gpt-oss-20b", False, bitsloth/gpt-oss-20b", True),
            ("openai/gpt-oss-20b", True, bitsloth/gpt-oss-20b-bitsloth-bnb-4bit", True),
            (bitsloth/gpt-oss-20b", True, bitsloth/gpt-oss-20b-bitsloth-bnb-4bit", True),
            (bitsloth/gpt-oss-20b-bf16", True, bitsloth/gpt-oss-20b-bf16", False),
            (
                bitsloth/gpt-oss-20b-bitsloth-bnb-4bit",
                False,
                bitsloth/gpt-oss-20b",
                True,
            ),
            (
                bitsloth/gpt-oss-20b-bnb-4bit",
                True,
                bitsloth/gpt-oss-20b-bnb-4bit",
                False,
            ),
        ]
        for case in cases:
            if isinstance(case, str):
                model_name = case
                with self.subTest(model_name = model_name, load_in_4bit = True):
                    self._assert_mapping(model_name, True, model_name, False)
            else:
                model_name, load_in_4bit, expected, should_change = case
                with self.subTest(model_name = model_name, load_in_4bit = load_in_4bit):
                    self._assert_mapping(
                        model_name, load_in_4bit, expected, should_change
                    )

    def test_static_mapper_contract(self):
        contracts = [
            ("qwen/qwen3-8b", bitsloth/qwen3-8b-bitsloth-bnb-4bit"),
            ("qwen/qwen3-8b-fp8", bitsloth/qwen3-8b-bitsloth-bnb-4bit"),
            (
                "mistralai/ministral-3-3b-instruct-2512",
                bitsloth/ministral-3-3b-instruct-2512-bitsloth-bnb-4bit",
            ),
            (bitsloth/kimi-k2-instruct", bitsloth/kimi-k2-instruct-bf16"),
        ]
        for src, expected in contracts:
            with self.subTest(src = src):
                self.assertEqual(FLOAT_TO_INT_MAPPER[src], expected)
        self.assertEqual(
            MAP_TO_BITSLOTH_16bit["qwen/qwen3-8b-fp8"], bitsloth/Qwen3-8B-FP8"
        )


if __name__ == "__main__":
    unittest.main()
