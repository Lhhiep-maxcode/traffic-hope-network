"""Tensor/cache regressions. Run with torch and transformers installed.

Uses a tiny randomly initialized Qwen3; no model download or GPU is required.
The frozen v1 implementation is the behavioral reference.
"""

import importlib.util
import itertools
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

try:
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.cache_utils import DynamicCache
except ImportError:
    torch = None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return "User:" + messages[0]["content"] + "\nAssistant:"

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False,
                 add_special_tokens=False):
        ids = [ord(char) % 61 + 1 for char in text]
        result = {"input_ids": torch.tensor([ids]) if return_tensors else ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(map(str, ids))


@unittest.skipIf(torch is None, "requires torch and transformers with Qwen3")
class CachePerformanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[1]
        cls.fast = load_module("self_coding_fast", directory / "self_coding.py")
        cls.old = load_module("self_coding_old", directory / "v1_dont_edit/self-coding.py")
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=4096,
            attention_dropout=0.0, eos_token_id=None,
        )
        config._attn_implementation = "eager"
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.tokenizer = CharacterTokenizer()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def generator(self, module=None, **kwargs):
        return (module or self.fast).TokenByTokenGenerator(
            self.model, self.tokenizer, "Question " * 8, **kwargs
        )

    def assert_cache_matches(self, branch):
        reference = self.generator(self.old)
        reference.full_prompt = branch.full_prompt
        reference.start(branch.generated_ids)
        torch.testing.assert_close(branch.next_logits, reference.next_logits, atol=1e-6, rtol=1e-5)
        self.assertEqual(branch.cache_length, reference.cache_length)
        self.assertEqual(branch.past_key_values.get_seq_length(), branch.cache_length)
        for actual, expected in zip(branch.past_key_values.layers, reference.past_key_values.layers):
            torch.testing.assert_close(actual.keys, expected.keys, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual.values, expected.values, atol=1e-6, rtol=1e-5)
        self.assertEqual(branch.attention_mask.shape, (1, branch.cache_length))

    def test_rejected_speculation_and_backtrack_need_no_model_forward(self):
        branch = self.generator()
        branch.fix_backtrack = 3
        branch.start()
        for token in [4, 5, 6, 7, 8]:
            branch._accept(token, branch._forward_token(token))
        branch._forward_token(33, output_attentions=True)
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            branch._restore_prefix(branch.generated_ids)
            self.assertEqual(forward.call_count, 0)
            branch._restore_prefix(branch.generated_ids[:-3])
            self.assertEqual(forward.call_count, 0)
        self.assert_cache_matches(branch)

    def test_clean_branch_only_evaluates_missing_or_changed_suffix(self):
        branch = self.generator()
        branch.start([4, 5, 6])
        for prefix, expected_suffix in [([4, 5, 6, 7, 8], [7, 8]), ([4, 5, 9], [9])]:
            with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
                branch._restore_prefix(prefix)
                self.assertEqual(forward.call_count, 1)
                self.assertEqual(forward.call_args.kwargs["input_ids"].tolist(), [expected_suffix])
            self.assert_cache_matches(branch)

    def test_expired_logits_recompute_one_token_and_empty_prefix_is_safe(self):
        branch = self.generator()
        branch.start([4, 5, 6])
        with patch.object(self.model, "forward", wraps=self.model.forward) as forward:
            branch._restore_prefix([4, 5])
            self.assertEqual(forward.call_args.kwargs["input_ids"].tolist(), [[5]])
        self.assert_cache_matches(branch)
        branch._restore_prefix([])
        self.assert_cache_matches(branch)

    def test_legacy_cache_crop_preserves_prefix(self):
        branch = self.generator()
        branch.start([4, 5, 6])
        branch.past_key_values = branch.past_key_values.to_legacy_cache()
        original = branch.past_key_values
        self.assertTrue(branch._crop_cache(branch.cache_length - 2))
        for actual, expected in zip(branch.past_key_values, original):
            for tensor, old_tensor in zip(actual, expected):
                torch.testing.assert_close(tensor, old_tensor[..., :-2, :], rtol=0, atol=0)

    def test_truncated_legacy_cache_cannot_be_mistaken_for_full_prefix(self):
        branch = self.generator()
        branch.start([4, 5, 6])
        branch.past_key_values = tuple(
            (layer.keys[..., -4:, :], layer.values[..., -4:, :])
            for layer in branch.past_key_values.layers
        )
        self.assertFalse(branch._crop_cache(2))

    def test_unknown_cache_training_and_disabled_reuse_fall_back(self):
        for mode in ["unknown", "training", "disabled", "prompt_changed", "dynamic_rope"]:
            with self.subTest(mode=mode):
                branch = self.generator(reuse_kv_cache=mode != "disabled")
                branch.start([4, 5, 6])
                if mode == "unknown":
                    branch.past_key_values = object()
                if mode == "prompt_changed":
                    branch.full_prompt += "changed"
                try:
                    self.model.train(mode == "training")
                    with (
                        patch.object(self.model.config, "rope_scaling",
                                     {"rope_type": "dynamic"} if mode == "dynamic_rope" else None),
                        patch.object(branch, "start", wraps=branch.start) as start,
                    ):
                        branch._restore_prefix([4, 5])
                        start.assert_called_once_with([4, 5])
                finally:
                    self.model.eval()
                self.assert_cache_matches(branch)

    def test_sliding_cache_is_not_cropped(self):
        branch = self.generator()
        config = Qwen3Config(num_hidden_layers=2, use_sliding_window=True,
                             sliding_window=4, max_window_layers=0)
        branch.past_key_values = DynamicCache(config=config)
        self.assertFalse(branch._crop_cache(1))

    def test_prompt_tokenization_and_logits_storage_are_bounded(self):
        branch = self.generator()
        branch.fix_backtrack = 2
        with patch.object(self.fast, "_input_ids", wraps=self.fast._input_ids) as tokenize:
            branch.start()
            branch.start([4])
            self.assertEqual(tokenize.call_count, 1)
        self.assertEqual(branch.next_logits.untyped_storage().nbytes(), branch.next_logits.numel() * branch.next_logits.element_size())
        for token in range(5, 15):
            branch._accept(token, branch._forward_token(token))
        self.assertEqual(len(branch._logits_history), 3)

    def test_attention_score_is_exactly_equal_to_original(self):
        generator = self.fast.CustomGenerator.__new__(self.fast.CustomGenerator)
        generator.heads = [
            {"layer": 1, "head": 2, "score": 0.3},
            {"layer": 0, "head": 0, "score": 0.7},
            {"layer": 1, "head": 1, "score": -0.2},
        ]
        generator.privileged_context_token_indices = [1, 3, 5, 8]
        for dtype, aggregation in itertools.product(
            [torch.float32, torch.float16, torch.bfloat16], ["weighted", "mean"]
        ):
            generator.aggregation = aggregation
            attentions = [torch.rand(1, 4, 1, 12).to(dtype) for _ in range(2)]
            self.assertEqual(generator._attention_score(attentions),
                             self.old.CustomGenerator._attention_score(generator, attentions))

    def assert_result_matches(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_result_matches(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_result_matches(left, right)
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, places=6)
        else:
            self.assertEqual(actual, expected)

    def test_full_generation_matches_original_for_all_repair_modes(self):
        with TemporaryDirectory() as temp:
            detector_path = Path(temp) / "detector.json"
            detector_path.write_text(json.dumps({"test": {
                "heads": [{"layer": 0, "head": 1}], "threshold": 0.5, "window_size": 5,
            }}))
            for method, clean, loop, window in itertools.product(
                ["attention_score", "js_divergence"], [False, True], [False, True], [False, True]
            ):
                with self.subTest(method=method, clean=clean, loop=loop, window=window):
                    results = []
                    for module in [self.old, self.fast]:
                        generator = module.CustomGenerator(
                            self.model, self.tokenizer, detector_path, "test",
                            prompt="Question " * 8, privileged_context="secret",
                            max_new_tokens=24, seed=2026, top_k=12, top_p=0.9,
                            fix_comparison_method=method,
                            debug_clean_backtrack=clean, debug_fix_infinite_loop=loop,
                            debug_wait_safe_window=window, max_repair_steps=16,
                        )
                        real_attention = generator._attention_score
                        real_js = generator.jensen_shannon_score

                        def attention(attentions, branch=generator, score_fn=real_attention):
                            score = score_fn(attentions)
                            leaking = (len(branch.generated_ids) in [6, 15]
                                       if not branch.repairing else branch.repair_steps < 3)
                            return float(leaking) + score * 0.01

                        def js(left, right, branch=generator, score_fn=real_js):
                            return float(branch.repair_steps < 3) + score_fn(left, right) * 0.01

                        generator._attention_score = attention
                        generator.jensen_shannon_score = js
                        results.append(generator.generate(show_progress=False))
                    self.assertTrue(results[0]["repair_events"])
                    self.assert_result_matches(results[1], results[0])


if __name__ == "__main__":
    unittest.main()
