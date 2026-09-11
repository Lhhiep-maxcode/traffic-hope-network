"""Small real Qwen3 tensor/cache regressions; no model download required."""

import copy
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

try:
    from .attention import enable_selective_attention
    from .batching import BatchScheduler, forward_batch
    from .self_coding import CustomGenerator
except ImportError:
    from attention import enable_selective_attention
    from batching import BatchScheduler, forward_batch
    from self_coding import CustomGenerator


class Tokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return 'User:' + messages[0]['content'] + '\nAssistant:'

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        ids = [ord(char) % 61 + 1 for char in text]
        result = {'input_ids': torch.tensor([ids]) if return_tensors else ids}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, **kwargs):
        return ','.join(map(str, ids))


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(17)
        config = Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=1024, eos_token_id=None,
            attention_dropout=0.0,
        )
        config._attn_implementation = 'eager'
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.tokenizer = Tokenizer()
        cls.detector = {'heads': [{'layer': 1, 'head': 2}], 'threshold': 2.0, 'window_size': 3}

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def generator(self, sample=None, model=None, repair=False, **kwargs):
        sample = sample or {'sample_index': 0, 'question': 'Question'}
        generator = CustomGenerator(
            model or self.model, self.tokenizer, None, 'test',
            detector_config={'test': self.detector},
            prompt=sample['question'], privileged_context=' Answer 0.5',
            max_new_tokens=8, seed=100 + sample['sample_index'],
            decode_tokens=False, debug_clean_backtrack=True,
            debug_wait_safe_window=True, **kwargs,
        )
        if repair:
            # Deliberately stagger repair between samples independently of logits.
            generator._attention_score = lambda _: (
                1.0 if len(generator.generated_ids) == 2 + sample['sample_index'] % 2
                and not generator.repair_events else 0.0
            )
            generator.threshold = 0.5
        return generator

    def test_batch_forward_matches_scalar_logits_attention_and_cache(self):
        generators = [self.generator({'sample_index': i, 'question': 'Q' * (i * 5 + 1)}) for i in range(3)]
        for generator in generators:
            generator.begin()
        original_lengths = [g.cache_length for g in generators]
        outputs = forward_batch(generators, [2, 3, 4])
        for generator, token, actual, original_length in zip(generators, [2, 3, 4], outputs, original_lengths):
            self.assertEqual(generator.past_key_values.get_seq_length(), original_length)
            expected = generator._forward_token(token, output_attentions=True)
            torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
            for actual_layer, expected_layer in zip(actual.past_key_values.layers, expected.past_key_values.layers):
                torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=1e-6, rtol=1e-5)
            for a, b in zip(actual.attentions, expected.attentions):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
            generator._accept(token, actual)
            generator._restore_prefix([])
            self.assertEqual(generator.past_key_values.get_seq_length(), original_length)

    def test_batch_and_scalar_outputs_match_with_and_without_repair(self):
        samples = [{'sample_index': i, 'question': 'Q' * (i + 1)} for i in range(5)]
        for repair in (False, True):
            expected = {s['sample_index']: self.generator(s, repair=repair).generate(False, False) for s in samples}
            for batch_size in (1, 2, 4):
                scheduler = BatchScheduler(lambda s: self.generator(s, repair=repair), batch_size)
                rows = list(scheduler.run(iter(samples)))
                self.assertEqual(len(rows), len(samples))
                for row in rows:
                    for key, value in expected[row['sample_index']].items():
                        self.assertEqual(row[key], value)
            if repair:
                self.assertTrue(all(output['repair_events'] for output in expected.values()))

    def test_rng_is_independent_of_sample_order_and_global_rng(self):
        state = torch.random.get_rng_state().clone()
        sample = {'sample_index': 4, 'question': 'Q'}
        expected = self.generator(sample).generate(False, False)
        self.generator().generate(False, False)
        actual = self.generator(sample).generate(False, False)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_early_eos_refills_slots_without_losing_samples(self):
        def factory(sample):
            generator = self.generator(sample)
            generator.tokenizer = copy.copy(self.tokenizer)
            generator.tokenizer.eos_token_id = 2
            generator._sample_token = lambda _: torch.tensor([
                [2 if len(generator.generated_ids) == sample['sample_index'] % 3 else 3]
            ])
            return generator
        samples = [{'sample_index': i, 'question': 'Q' * (i + 1)} for i in range(7)]
        rows = list(BatchScheduler(factory, 3).run(samples))
        self.assertEqual(sorted(row['sample_index'] for row in rows), list(range(7)))
        for row in rows:
            self.assertEqual(row['token_ids'], [3] * (row['sample_index'] % 3) + [2])

    def test_selective_ragged_batch_matches_scalar(self):
        model = copy.deepcopy(self.model)
        enable_selective_attention(model, self.detector)
        generators = [self.generator({'sample_index': i, 'question': 'Q' * (i * 7 + 1)}, model=model) for i in range(3)]
        for generator in generators:
            generator.begin()
        outputs = forward_batch(generators, [2, 3, 4])
        for generator, token, actual in zip(generators, [2, 3, 4], outputs):
            expected = generator._forward_token(token, True)
            torch.testing.assert_close(actual.logits, expected.logits, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual.attentions[1], expected.attentions[1], atol=1e-6, rtol=1e-5)
            for a, b in zip(actual.past_key_values.layers, expected.past_key_values.layers):
                torch.testing.assert_close(a.keys, b.keys, atol=1e-6, rtol=1e-5)

    def test_selective_attention_matches_eager_detector_weights(self):
        model = copy.deepcopy(self.model)
        enable_selective_attention(model, self.detector)
        eager = self.generator()
        selective = self.generator(model=model)
        eager.begin()
        selective.begin()
        torch.testing.assert_close(eager.next_logits, selective.next_logits, atol=1e-6, rtol=1e-5)
        for token in (2, 3, 4):
            expected = eager._forward_token(token, True)
            actual = selective._forward_token(token, True)
            torch.testing.assert_close(expected.logits, actual.logits, atol=1e-6, rtol=1e-5)
            self.assertEqual(len(actual.attentions), 3)
            self.assertEqual(actual.attentions[0].numel(), 0)
            torch.testing.assert_close(expected.attentions[1], actual.attentions[1], atol=1e-6, rtol=1e-5)
            eager._accept(token, expected)
            selective._accept(token, actual)
        rows = list(BatchScheduler(lambda s: self.generator(s, model=model, repair=True), 2).run([
            {'sample_index': 0, 'question': 'Q'}, {'sample_index': 1, 'question': 'Longer question'},
        ]))
        self.assertEqual(len(rows), 2)


if __name__ == '__main__':
    unittest.main()
