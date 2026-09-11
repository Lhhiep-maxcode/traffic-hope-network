"""Run with python3 -m unittest discover -s method/generate/v2_simple_love.

Uses deterministic model stubs; no model downloads or torch installation needed.
"""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


DIRECTORY = Path(__file__).resolve().parent


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Progress:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
        self.n = 0

    def __iter__(self):
        return iter(self.iterable)

    def update(self, amount):
        self.n += amount

    def close(self):
        pass


class Tokenizer:
    eos_token_id = None

    def __init__(self):
        self.decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def __call__(self, text, **kwargs):
        return {'offset_mapping': [(i, i + 1) for i in range(len(text))]}

    def decode(self, ids, skip_special_tokens=True):
        self.decode_calls.append((list(ids), skip_special_tokens))
        return ','.join(map(str, ids))


torch_stub = types.ModuleType('torch')
torch_stub.no_grad = lambda: lambda function: function
torch_stub.bfloat16 = object()
torch_stub.manual_seed = lambda seed: None
tqdm_stub = types.ModuleType('tqdm')
tqdm_stub.tqdm = Progress
tqdm_auto_stub = types.ModuleType('tqdm.auto')
tqdm_auto_stub.tqdm = Progress
transformers_stub = types.ModuleType('transformers')
transformers_stub.AutoTokenizer = Mock()
transformers_stub.AutoModelForCausalLM = Mock()
with patch.dict(sys.modules, {
    'torch': torch_stub, 'tqdm': tqdm_stub, 'tqdm.auto': tqdm_auto_stub,
    'transformers': transformers_stub,
}):
    coding = load_module('v2_coding_tests', 'self_coding.py')
    with patch.dict(sys.modules, {'self_coding': coding}):
        bulk = load_module('v2_bulk_tests', 'generate.py')


CONFIG = {'test': {'heads': [{'layer': 0, 'head': 0}], 'threshold': 0.5,
                   'window_size': 3}}


def fake_start(branch, generated_ids=None):
    branch.generated_ids = list(generated_ids or [])
    branch.next_logits = object()
    branch.finished = False
    return 0


def fake_accept(branch, token_id, outputs):
    branch.generated_ids.append(token_id)


class OptimizationTests(unittest.TestCase):
    def custom(self, tokenizer=None, **kwargs):
        return coding.CustomGenerator(
            model=object(), tokenizer=tokenizer or Tokenizer(), prompt='Question ',
            privileged_context='Answer 0.5', detector_config_path='unused.json',
            model_key='test', detector_config=CONFIG, max_new_tokens=5,
            debug_clean_backtrack=True, debug_wait_safe_window=True, **kwargs,
        )

    def test_decode_flag_preserves_fixed_baseline_and_repair_results(self):
        results = []
        for enabled in (True, False):
            tokenizer = Tokenizer()
            generator = self.custom(tokenizer, decode_tokens=enabled)
            self.assertEqual(generator.clean_generator.decode_tokens, enabled)

            # Trigger one repair after two accepted tokens, then safe checks.
            def score(branch, attentions):
                return 1.0 if len(branch.generated_ids) == 2 and not branch.repair_events else 0.0

            with (
                patch.object(coding.TokenByTokenGenerator, 'start', fake_start),
                patch.object(coding.TokenByTokenGenerator, '_restore_prefix', fake_start),
                patch.object(coding.TokenByTokenGenerator, '_accept', fake_accept),
                patch.object(coding.TokenByTokenGenerator, '_sample_token',
                             return_value=types.SimpleNamespace(item=lambda: 2)),
                patch.object(coding.TokenByTokenGenerator, '_forward_token',
                             return_value=types.SimpleNamespace(attentions=object())),
                patch.object(coding.CustomGenerator, '_attention_score', score),
            ):
                result = generator.generate(include_unfixed=True, show_progress=False)
            self.assertTrue(result['repair_events'])
            self.assertTrue(result['repair_events'][0]['clean_backtrack_token_ids'])
            self.assertEqual(result['text'], '2,2,2,2,2')
            self.assertEqual(result['unfixed']['text'], '2,2,2,2,2')
            per_token_calls = [call for call in tokenizer.decode_calls if not call[1]]
            self.assertEqual(bool(per_token_calls), enabled)
            self.assertEqual(sum(call[1] for call in tokenizer.decode_calls), 2)
            results.append(result)
        self.assertEqual(results[0], results[1])

    def test_step_text_defaults_to_enabled_and_can_be_disabled(self):
        branch = coding.TokenByTokenGenerator(object(), Tokenizer(), 'Question')
        fake_start(branch)
        with (
            patch.object(branch, '_sample_token', return_value=types.SimpleNamespace(item=lambda: 2)),
            patch.object(branch, '_forward_token', return_value=object()),
            patch.object(coding.TokenByTokenGenerator, '_accept', fake_accept),
        ):
            self.assertEqual(branch.step(), (2, '2', False))
            branch.decode_tokens = False
            self.assertEqual(branch.step(), (2, None, False))

    def test_preloaded_config_avoids_file_reads_and_path_mode_still_works(self):
        with patch.object(coding.Path, 'read_text', side_effect=AssertionError('Unexpected read')):
            self.assertEqual(self.custom().detector, CONFIG['test'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detector.json'
            path.write_text(json.dumps(CONFIG))
            generator = coding.CustomGenerator(
                object(), Tokenizer(), path, 'test', prompt='Question ',
                privileged_context='Answer 0.5',
            )
            self.assertEqual(generator.detector, CONFIG['test'])

    def test_jsonl_is_lazy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.jsonl'
            path.write_text('{"question": "First", "ground_truth": 0}\nINVALID\n')
            records = bulk.load_jsonl(path)
            self.assertIs(iter(records), records)
            self.assertEqual(next(records)['question'], 'First')
            with self.assertRaises(json.JSONDecodeError):
                next(records)

    def test_bulk_writes_each_record_before_reading_next_and_loads_config_once(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output.jsonl'
            config = Path(directory) / 'detector.json'
            config.write_text(json.dumps(CONFIG))

            def samples(path):
                yield {'question': 'First', 'ground_truth': '0.5'}
                # This observes real file contents, so it also checks flushing.
                self.assertEqual(json.loads(output.read_text())['question'], 'First')
                yield {'question': 'Second', 'ground_truth': '42.'}

            instances = []

            def make_generator(**kwargs):
                self.assertNotIn('include_unfixed', kwargs)
                self.assertNotIn('show_progress', kwargs)
                instance = Mock()
                instance.generate.return_value = {'text': 'Solution', 'token_ids': [2]}
                instances.append((kwargs, instance))
                return instance

            with (
                patch.object(bulk, 'OUTPUT_PATH', output),
                patch.object(bulk, 'DETECTOR_CONFIG_PATH', config),
                patch.object(bulk, 'load_jsonl', samples),
                patch.object(bulk, 'CustomGenerator', side_effect=make_generator),
                patch('builtins.open', wraps=open) as opened,
            ):
                bulk.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row['ground_truth'] for row in rows], ['0.5', '42.'])
            self.assertTrue(all('privileged_context' in row and row['text'] == 'Solution' for row in rows))
            self.assertEqual(sum(call.args[0] == config for call in opened.call_args_list), 1)
            self.assertEqual(sum(call.args[0] == output for call in opened.call_args_list), 1)
            self.assertIs(instances[0][0]['detector_config'], instances[1][0]['detector_config'])
            for kwargs, instance in instances:
                self.assertFalse(kwargs['decode_tokens'])
                self.assertTrue(kwargs['debug_fix_infinite_loop'])
                instance.generate.assert_called_once_with(include_unfixed=False, show_progress=False)


if __name__ == '__main__':
    unittest.main()
