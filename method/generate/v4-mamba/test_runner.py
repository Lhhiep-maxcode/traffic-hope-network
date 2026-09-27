import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from detector import load_detector, resolve_heads
from safetensors.torch import save_file
from test_runtime import TinyNemotron
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from generate import (
    file_hash,
    format_record,
    iter_samples,
    parser,
    preflight,
    prepare_output,
    run_workers,
)


def write_tiny_checkpoint(model_path):
    """Local remote-code fixture: exercise actual AutoModel loading in spawn workers."""
    model_path.mkdir()
    (model_path / 'tiny_model.py').write_text('''from transformers import PretrainedConfig, PreTrainedModel
from test_runtime import TinyNemotron

class TinyConfig(PretrainedConfig):
    model_type = "nemotron_h"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_override_pattern = "M*-M*-"
        self.num_attention_heads = 4

class TinyModel(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config):
        super().__init__(config)
        fixture = TinyNemotron()
        self.backbone = fixture.backbone
        self.lm_head = fixture.lm_head
''')
    (model_path / 'config.json').write_text(json.dumps({
        'model_type': 'nemotron_h', 'torch_dtype': 'float32',
        'auto_map': {'AutoConfig': 'tiny_model.TinyConfig', 'AutoModelForCausalLM': 'tiny_model.TinyModel'},
    }))
    save_file(TinyNemotron().state_dict(), model_path / 'model.safetensors', metadata={'format': 'pt'})
    backend = Tokenizer(WordLevel({'[UNK]': 0, 'Q': 1, '0': 2}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]').save_pretrained(model_path)


class RunnerTests(unittest.TestCase):
    def test_decimal_prompts_and_resume_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'input.jsonl', root / 'output.jsonl'
            source.write_text('{"question":"Q", "ground_truth":"0.5"}\n' * 3)
            args = parser().parse_args(['--input', str(source), '--output', str(output),
                                       '--model-key', 'Nemo3-Nano-4B-BF16-math'])
            self.assertEqual(prepare_output(args), set())
            meta = json.loads(Path(str(output) + '.meta.json').read_text())
            self.assertEqual(meta['version'], 4)
            self.assertEqual(meta['detector_sha256'], file_hash(args.detector_config))
            self.assertEqual(meta['detector_layer_indexing'], 'attention_order')
            output.write_text('{"sample_index":1}\n{"sample_index":')
            args.resume = True
            self.assertEqual(prepare_output(args), {1})
            self.assertEqual(output.read_text(), '{"sample_index":1}\n')
            self.assertEqual([s['sample_index'] for s in iter_samples(source, completed={1})], [0, 2])
            args.model_key = 'Nemo3-Nano-4B-BF16-logic'
            with self.assertRaisesRegex(ValueError, 'metadata'):
                prepare_output(args)
        record = format_record({'question': 'Q?', 'ground_truth': '0.5'}, 2)
        self.assertTrue(record['prompt_w_answer'].endswith('answer is 0.5'))

    def test_detector_validation_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'detector.json'
            path.write_text(json.dumps({'test': {'heads': [{'layer': 0, 'head': 0}],
                                                'threshold': 0.2, 'window_size': 4}}))
            args = parser().parse_args(['--input', 'unused', '--output', 'unused', '--model-key', 'test',
                                       '--detector-config', str(path)])
            with patch('transformers.AutoConfig.from_pretrained') as loader:
                with self.assertRaisesRegex(ValueError, 'odd integer'):
                    preflight(args)
                loader.assert_not_called()

    def test_supplied_nemotron_configs_use_attention_order(self):
        # NVIDIA Nano 4B config: four attention blocks among 42 decoder blocks.
        model_config = SimpleNamespace(model_type='nemotron_h', num_attention_heads=40,
                                       hybrid_override_pattern='M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-')
        blocks = [12, 17, 24, 32]
        for task in ('logic', 'math', 'multihop-reasoning', 'science'):
            key = 'Nemo3-Nano-4B-BF16-' + task
            args = parser().parse_args(['--input', 'unused', '--output', 'unused', '--model-key', key])
            self.assertEqual(args.detector_mode, 'attention')
            self.assertEqual(args.fix_comparison_method, 'attention_score')
            selected, payload = load_detector(args)
            self.assertEqual(selected, key)
            mapped = resolve_heads(payload[key], dict.fromkeys(blocks, 40))
            for original, resolved in zip(payload[key]['heads'], mapped):
                self.assertEqual(resolved, {**original, 'layer': blocks[original['layer']]})
            with patch('transformers.AutoConfig.from_pretrained', return_value=model_config), \
                    patch('generate.importlib.import_module'):
                self.assertIs(preflight(args), model_config)

    def test_hybrid_model_two_spawned_workers_and_refill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / 'model'
            write_tiny_checkpoint(model_path)
            detector = root / 'detector.json'
            detector.write_text(json.dumps({'test': {'heads': [{'layer': 1, 'head': 2, 'score': 0.7}],
                                                    'threshold': 1.0, 'window_size': 3}}))
            args = parser().parse_args([
                '--input', str(root / 'unused'), '--output', str(root / 'output'),
                '--model', str(model_path), '--dtype', 'float32', '--batch-size', '3',
                '--max-new-tokens', '4', '--cpu-threads', '1',
                '--model-key', 'test', '--detector-config', str(detector),
            ])
            samples = [format_record({'question': 'Q ' * (i % 2 + 1), 'ground_truth': 0}, i) for i in range(7)]
            output = io.StringIO()
            stats = run_workers(args, ['cpu', 'cpu'], iter(samples), output)
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(sorted(r['sample_index'] for r in rows), list(range(7)))
            self.assertEqual(stats['samples'], 7)
            self.assertEqual(len(stats['workers']), 2)
            self.assertTrue(all(len(r['token_ids']) == 4 for r in rows))
            self.assertTrue(all(r['generation_stats']['detector_mode'] == 'attention' for r in rows))


if __name__ == '__main__':
    unittest.main()
