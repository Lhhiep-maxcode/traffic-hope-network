import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

try:
    from .generate import format_record, iter_samples, parser, prepare_output, run_workers
except ImportError:
    from generate import format_record, iter_samples, parser, prepare_output, run_workers


class RunnerTests(unittest.TestCase):
    def test_prompts_preserve_decimal_and_trailing_period(self):
        for answer in ('0.5', '42.', 0):
            record = format_record({'question': 'Question?', 'ground_truth': answer, 'id': 'source'}, 3)
            self.assertEqual(record['ground_truth'], answer)
            self.assertEqual(record['privileged_context'], '\n\nGiven the ground truth answer is ' + str(answer))
            self.assertEqual(record['prompt_w_answer'], record['prompt_wo_answer'] + record['privileged_context'])
            self.assertEqual(record['id'], 'source')

    def test_resume_rejects_changed_input_and_recovers_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, output = root / 'input.jsonl', root / 'detector.json', root / 'output.jsonl'
            source.write_text('{"question":"Q", "ground_truth":0}\n' * 3)
            config.write_text('{}')
            args = parser().parse_args(['--input', str(source), '--output', str(output), '--detector-config', str(config)])
            self.assertEqual(prepare_output(args), set())
            output.write_text('{"sample_index":1}\n{"sample_index":')
            args.resume = True
            self.assertEqual(prepare_output(args), {1})
            self.assertEqual(output.read_text(), '{"sample_index":1}\n')
            self.assertEqual([s['sample_index'] for s in iter_samples(source, completed={1})], [0, 2])
            source.write_text('{"question":"Changed", "ground_truth":0}\n')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                prepare_output(args)

    def test_two_spawned_workers_with_real_local_tiny_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / 'model'
            torch.manual_seed(1)
            config = Qwen3Config(
                vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                head_dim=8, max_position_embeddings=1024, eos_token_id=None,
            )
            Qwen3ForCausalLM(config).save_pretrained(model_path)
            backend = Tokenizer(WordLevel({'[UNK]': 0, 'Q': 1, '0': 2}, unk_token='[UNK]'))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]')
            tokenizer.chat_template = "{{ messages[0]['content'] }}"
            tokenizer.save_pretrained(model_path)
            detector_path = root / 'detector.json'
            detector_path.write_text(json.dumps({'test': {
                'heads': [{'layer': 1, 'head': 0}], 'threshold': 2, 'window_size': 3,
            }}))
            args = parser().parse_args([
                '--input', str(root / 'unused'), '--output', str(root / 'unused-output'),
                '--model', str(model_path), '--model-key', 'test',
                '--detector-config', str(detector_path), '--dtype', 'float32',
                '--batch-size', '2', '--max-new-tokens', '4', '--cpu-threads', '1',
            ])
            samples = [format_record({'question': 'Q ' * (i + 1), 'ground_truth': 0}, i) for i in range(7)]
            output = io.StringIO()
            stats = run_workers(args, ['cpu', 'cpu'], iter(samples), output)
            rows = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(sorted(row['sample_index'] for row in rows), list(range(7)))
            self.assertEqual(stats['samples'], 7)
            self.assertEqual(len(stats['workers']), 2)
            self.assertTrue(all(len(row['token_ids']) == 4 for row in rows))
            self.assertTrue(all(row['unfixed'] is None for row in rows))


if __name__ == '__main__':
    unittest.main()
