"""CPU tensor regressions with real Mamba1/Mamba2 recurrent implementations."""

import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from batching import BatchScheduler, forward_batch
from cache import merge_caches
from detector import load_detector, resolve_heads
from runtime import make_runtime
from self_coding import CustomGenerator, TokenByTokenGenerator
from torch import nn
from transformers import (
    FalconMambaConfig,
    FalconMambaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
    MambaConfig,
    MambaForCausalLM,
)
from transformers.models.mamba2.modeling_mamba2 import Mamba2Mixer


class Tokenizer:
    eos_token_id = None
    chat_template = 'test'

    def apply_chat_template(self, messages, **kwargs):
        return 'U:' + messages[0]['content'] + '\nA:'

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        ids = [ord(c) % 61 + 1 for c in text]
        result = {'input_ids': torch.tensor([ids]) if return_tensors else ids}
        if return_offsets_mapping:
            result['offset_mapping'] = [(i, i + 1) for i in range(len(text))]
        return result

    def decode(self, ids, **kwargs):
        return ','.join(map(str, ids))


def mamba2_config(**kwargs):
    return Mamba2Config(vocab_size=64, hidden_size=16, expand=2, num_hidden_layers=2,
                        num_heads=4, head_dim=8, state_size=4, n_groups=1,
                        chunk_size=4, conv_kernel=4, eos_token_id=None, **kwargs)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads, self.num_key_value_heads, self.head_dim = 4, 2, 4
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False)
        self.v_proj = nn.Linear(16, 8, bias=False)
        self.o_proj = nn.Linear(16, 16, bias=False)


class TinyNemotron(nn.Module):
    """Nemotron-H block layout with actual native Mamba2 mixers, no downloads."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type='nemotron_h')
        self.generation_config = SimpleNamespace(eos_token_id=None)
        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Embedding(64, 16)
        self.backbone.norm_f = nn.RMSNorm(16, eps=1e-5)
        self.backbone.layers = nn.ModuleList()
        for i, kind in enumerate(('mamba', 'attention', 'mlp', 'mamba', 'attention', 'moe')):
            layer = nn.Module()
            layer.block_type = kind
            layer.residual_in_fp32 = True
            layer.norm = nn.RMSNorm(16, eps=1e-5)
            if kind == 'mamba':
                layer.mixer = Mamba2Mixer(mamba2_config(), layer_idx=i)
            elif kind == 'attention':
                layer.mixer = TinyAttention()
            else:
                layer.mixer = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16))
            self.backbone.layers.append(layer)
        self.lm_head = nn.Linear(16, 64, bias=False)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        cls.models = [
            make_runtime(TinyNemotron().eval()),
            make_runtime(MambaForCausalLM(MambaConfig(
                vocab_size=64, hidden_size=16, num_hidden_layers=2,
                state_size=4, conv_kernel=4, eos_token_id=None,
            )).eval()),
            make_runtime(Mamba2ForCausalLM(mamba2_config()).eval()),
            make_runtime(FalconMambaForCausalLM(FalconMambaConfig(
                vocab_size=64, hidden_size=16, num_hidden_layers=2,
                state_size=4, conv_kernel=4, eos_token_id=None,
            )).eval()),
        ]

    def generator(self, model=None, sample=None, cache_mode='cached', **kwargs):
        sample = sample or {'sample_index': 0, 'question': 'Q'}
        config = {'heads': [{'layer': 1, 'head': 1}],
                  'threshold': 1.0, 'window_size': 3}
        return CustomGenerator(
            model or self.models[0], Tokenizer(), None, 'test',
            detector_config={'test': config},
            cache_mode=cache_mode, prompt=sample['question'], privileged_context=' answer 0.5',
            max_new_tokens=6, seed=100 + sample['sample_index'], decode_tokens=False,
            fix_comparison_method='js_divergence', debug_clean_backtrack=True,
            debug_wait_safe_window=True, **kwargs,
        )

    def token_generator(self, model, cache_mode='cached'):
        return TokenByTokenGenerator(model, Tokenizer(), 'Q', privileged_context=' answer 0.5',
                                     cache_mode=cache_mode, do_sample=False)

    def assert_cache_close(self, actual, expected):
        for name in ('conv_states', 'ssm_states', 'key_cache', 'value_cache'):
            if hasattr(actual, name):
                for a, b in zip(getattr(actual, name), getattr(expected, name)):
                    torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-4)

    @torch.inference_mode()
    def test_decode_and_rejected_token_rollback_match_full_prefill(self):
        for model in self.models:
            with self.subTest(kind=model.config.model_type):
                cached = self.token_generator(model)
                reference = self.token_generator(model, cache_mode='recompute')
                cached.start()
                reference.start()
                for token in (2, 3, 4):
                    a, b = cached._forward_token(token), reference._forward_token(token)
                    torch.testing.assert_close(a.logits, b.logits, atol=3e-5, rtol=3e-4)
                    cached._accept(token, a)
                    reference._accept(token, b)
                cached._forward_token(55)  # Mutates conv/SSM state, but never accepted.
                with self.assertRaisesRegex(RuntimeError, 'speculative'):
                    cached._forward_token(3)
                cached._restore_prefix([2])
                fresh = self.token_generator(model)
                fresh.start([2])
                torch.testing.assert_close(cached.next_logits, fresh.next_logits)
                self.assert_cache_close(cached.past_key_values, fresh.past_key_values)
                cached._restore_prefix([2, 8, 9])  # Append-only replay must be one token at a time.
                fresh.start([2, 8, 9])
                torch.testing.assert_close(cached.next_logits, fresh.next_logits, atol=3e-5, rtol=3e-4)
                self.assert_cache_close(cached.past_key_values, fresh.past_key_values)

    @torch.inference_mode()
    def test_batched_states_are_independent_and_match_scalar(self):
        for model in self.models:
            a, b = self.token_generator(model), self.token_generator(model)
            a.start()
            b.start()
            original = copy.deepcopy(a.past_key_values)
            results = forward_batch([a, b], [2, 3], output_attentions=False)
            self.assert_cache_close(a.past_key_values, original)
            for generator, token, result in zip((a, b), (2, 3), results):
                expected = generator._forward_token(token)
                torch.testing.assert_close(result.logits, expected.logits, atol=3e-5, rtol=3e-4)
                self.assert_cache_close(result.past_key_values, expected.past_key_values)
            states = results[0].past_key_values.ssm_states
            other = results[1].past_key_values.ssm_states
            old = other[0].clone()
            states[0].zero_()
            torch.testing.assert_close(other[0], old)

    @torch.inference_mode()
    def test_attention_block_indices_and_cached_weights(self):
        runtime = self.models[0]
        self.assertEqual(runtime.attention_layers, {1: 4, 4: 4})
        detector = {'heads': [{'layer': 0, 'head': 0}, {'layer': 1, 'head': 1}]}
        self.assertEqual([h['layer'] for h in resolve_heads(detector, runtime.attention_layers)], [1, 4])
        self.assertEqual([h['layer'] for h in detector['heads']], [0, 1])
        with self.assertRaisesRegex(ValueError, 'attention ordinals'):
            resolve_heads({'heads': [{'layer': 2, 'head': 0}]}, runtime.attention_layers)
        a = self.generator()
        b = self.generator(cache_mode='recompute')
        self.assertEqual(a.heads[0]['layer'], 4)
        a.begin()
        b.begin()
        for token in (2, 3):
            out_a, out_b = a._forward_token(token, True), b._forward_token(token, True)
            self.assertEqual(set(out_a.attentions), {1, 4})
            for index in out_a.attentions:
                torch.testing.assert_close(out_a.attentions[index], out_b.attentions[index], atol=2e-5, rtol=2e-4)
                self.assertEqual(out_a.attentions[index].shape[-1], a.cache_length + 1)
            a._accept(token, out_a)
            b._accept(token, out_b)
        eager = make_runtime(copy.deepcopy(runtime.wrapped), 'eager')
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        sdpa_result, eager_result = runtime(ids, output_attentions=True), eager(ids, output_attentions=True)
        torch.testing.assert_close(sdpa_result.logits, eager_result.logits)
        for i in sdpa_result.attentions:
            torch.testing.assert_close(sdpa_result.attentions[i], eager_result.attentions[i])

    def test_supplied_detector_scores_the_mapped_heads_with_original_weights(self):
        blocks = [12, 17, 24, 32]
        model = SimpleNamespace(attention_layers=dict.fromkeys(blocks, 40))
        for task in ('logic', 'math', 'multihop-reasoning', 'science'):
            key = 'Nemo3-Nano-4B-BF16-' + task
            args = SimpleNamespace(model_key=key, detector_config=Path(__file__).with_name('detector_config.json'))
            _, payload = load_detector(args)
            g = CustomGenerator(model, Tokenizer(), None, key, detector_config=payload,
                                prompt='Q', privileged_context=' answer 0.5')
            length = len(g.full_prompt)
            weights = {}
            for ordinal, block in enumerate(blocks):
                # Distinct probabilities for every attention layer and head.
                per_head = (torch.arange(40, dtype=torch.float32) + 40 * ordinal + 1) / 10000
                weights[block] = per_head[None, :, None, None].expand(1, 40, 1, length)
            detector = payload[key]
            count = len(g.privileged_context_token_indices)
            expected = sum(
                ((h['layer'] * 40 + h['head'] + 1) / 10000) * count * h['score']
                for h in detector['heads']
            ) / sum(h['score'] for h in detector['heads'])
            self.assertAlmostEqual(g._attention_score(weights), expected, places=6)
            self.assertEqual(g.threshold, detector['threshold'])
            self.assertEqual(g.window_size, detector['window_size'])

    def test_scheduler_agreement_with_ragged_lengths_and_forced_repair(self):
        samples = [{'sample_index': i, 'question': 'Q' * (i % 2 + 1)} for i in range(5)]
        for method in ('attention_score', 'js_divergence'):
            def factory(sample, cache_mode='cached', method=method):
                g = self.generator(sample=sample, cache_mode=cache_mode)
                g.fix_comparison_method = method
                # Force one repair at different positions to exercise isolated rollback.
                g.check = lambda _: len(g.generated_ids) == 2 + sample['sample_index'] % 2 and not g.repair_events
                return g
            expected = {s['sample_index']: factory(s, 'recompute').generate(False, False) for s in samples}
            for size in (1, 2, 4):
                scheduler = BatchScheduler(factory, size)
                rows = list(scheduler.run(samples))
                self.assertEqual(len(rows), len(samples))
                for row in rows:
                    reference = expected[row['sample_index']]
                    self.assertEqual(row['token_ids'], reference['token_ids'])
                    self.assertTrue(row['repair_events'])
                    self.assertEqual(len(row['repair_events']), len(reference['repair_events']))
                    for a, b in zip(row['repair_events'], reference['repair_events']):
                        self.assertEqual(a['start_index'], b['start_index'])
                        self.assertEqual(a['replacement_token_ids'], b['replacement_token_ids'])

    def test_jsd_repair_compares_same_accepted_prefix(self):
        g = self.generator(do_sample=False)
        g.begin()
        g.step()
        g.step()
        candidate = g.next_logits.argmax(-1).item()
        g._forward_token(candidate)
        g._start_repair(candidate, 1.0)
        g.debug_clean_backtrack = False
        clean = TokenByTokenGenerator(g.model, g.tokenizer, g.prompt, do_sample=False)
        clean.start(g.generated_ids)
        expected = g.jensen_shannon_score(g.next_logits, clean.next_logits)
        g.step()
        self.assertAlmostEqual(g.repair_events[-1]['steps'][-1]['comparison_score'], expected, places=5)
        self.assertGreater(expected, 0)

    def test_attention_repair_discards_speculative_recurrent_state(self):
        def generator(cache_mode):
            g = self.generator(cache_mode=cache_mode, do_sample=False)
            g.fix_comparison_method = 'attention_score'
            # One triggered backtrack; repair comparisons subsequently pass.
            g._attention_score = lambda _: (
                2.0 if len(g.generated_ids) == 2 and not g.repair_events else 0.0
            )
            # Make clean replacement differ from the privileged candidate.
            g.clean_generator._sample_token = lambda _: torch.tensor([[7]])
            return g
        cached = generator('cached').generate(False, False)
        expected = generator('recompute').generate(False, False)
        self.assertEqual(cached, expected)
        self.assertTrue(cached['repair_events'])

    def test_repair_budget_counts_backtrack_and_requires_consecutive_safe_checks(self):
        for method in ('attention_score', 'js_divergence'):
            for converges in (False, True):
                with self.subTest(method=method, converges=converges):
                    g = CustomGenerator(
                        self.models[0], Tokenizer(), None, 'test', sample_index=41,
                        detector_config={'test': {
                            'heads': [{'layer': 1, 'head': 1}],
                            'threshold': 1.0, 'window_size': 7,
                        }},
                        prompt='Q', privileged_context=' answer 0.5',
                        max_new_tokens=20, max_repair_steps=8, do_sample=False,
                        fix_comparison_method=method, fix_js_divergence_threshold=0.1,
                        debug_clean_backtrack=True, debug_wait_safe_window=True,
                    )
                    g._sample_token = lambda _: torch.tensor([[7]])
                    g.clean_generator._sample_token = lambda _: torch.tensor([[7]])
                    g._attention_score = lambda _: 0.0
                    g.begin()
                    for _ in range(3):
                        g.step()
                    g._start_repair(7, 1.2)
                    unsafe = 1.2 if method == 'attention_score' else 0.2
                    scores = iter(
                        [unsafe, unsafe, 0.05, 0.05, 0.05] if converges
                        else [0.05, 0.05, unsafe, 0.05, 0.05]
                    )
                    if method == 'attention_score':
                        g._attention_score = lambda _, scores=scores: next(scores)
                    else:
                        g.jensen_shannon_score = lambda *_, scores=scores: next(scores)
                    for _ in range(7):
                        g.step()
                    if converges:
                        g.step()  # A safe streak completed on the final allowed step succeeds.
                        self.assertFalse(g.repairing)
                        self.assertEqual(len(g.repair_events[-1]['steps']), 8)
                        continue
                    with self.assertRaisesRegex(RuntimeError, 'Repair did not converge') as error:
                        g.step()
                    details = json.loads(str(error.exception).split('Details: ', 1)[1])
                    self.assertEqual(details['sample_index'], 41)
                    self.assertEqual(details['model_key'], 'test')
                    self.assertEqual(details['comparison_method'], method)
                    self.assertEqual(details['comparison_threshold'],
                                     1.0 if method == 'attention_score' else 0.1)
                    self.assertEqual(details['repair_steps'], details['max_repair_steps'])
                    self.assertEqual(details['safe_steps'], 2)
                    self.assertEqual(details['best_safe_steps'], 2)
                    self.assertEqual(details['required_safe_steps'], 3)
                    self.assertEqual(details['clean_backtrack_steps'], 3)
                    self.assertEqual(details['detected_index'], 3)
                    self.assertEqual(details['start_index'], 0)
                    self.assertEqual([s['passed_check'] for s in details['recent_checks']],
                                     [None, None, None, True, True, False, True, True])

    def test_ssm_precision_and_generation_eos_list(self):
        model = make_runtime(copy.deepcopy(self.models[0].wrapped).to(torch.float64), ssm_dtype=torch.float32)
        g = self.generator(model)
        g.begin()
        self.assertEqual(g.past_key_values.conv_states[0].dtype, torch.float64)
        self.assertEqual(g.past_key_values.ssm_states[0].dtype, torch.float32)
        model.generation_config.eos_token_id = [6, 7]
        g._sample_token = lambda _: torch.tensor([[7]])
        self.assertEqual(g.generate(False, False)['token_ids'], [7])

    def test_rng_order_eos_baseline_and_zero_tokens(self):
        state = torch.random.get_rng_state().clone()
        g = self.generator()
        g.tokenizer.eos_token_id = 2
        g._sample_token = lambda _: torch.tensor([[2]])
        output = g.generate(include_unfixed=True, show_progress=False)
        self.assertEqual(output['token_ids'], [2])
        self.assertIsNotNone(output['unfixed'])
        expected = self.generator().generate(False, False)
        self.generator(sample={'sample_index': 9, 'question': 'other'}).generate(False, False)
        self.assertEqual(self.generator().generate(False, False), expected)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        zero = self.generator()
        zero.max_new_tokens = 0
        self.assertEqual(next(iter(BatchScheduler(lambda _: zero).run([{}])))['token_ids'], [])

    def test_reject_mixed_lengths_and_pure_mamba_attention(self):
        a, b = self.generator(), self.generator(sample={'sample_index': 1, 'question': 'longer'})
        a.begin()
        b.begin()
        with self.assertRaisesRegex(ValueError, 'equal lengths'):
            forward_batch([a, b], [2, 3])
        with self.assertRaisesRegex(ValueError, 'equal prefix'):
            merge_caches([a.past_key_values, b.past_key_values])
        with self.assertRaisesRegex(ValueError, 'Pure Mamba'):
            self.models[1](torch.tensor([[2, 3]]), output_attentions=True)


if __name__ == '__main__':
    unittest.main()
