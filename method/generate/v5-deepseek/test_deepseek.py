"""DeepSeek template, EOS, and checkpoint/detector compatibility regressions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen3Config

from generate import DEFAULT_CONFIG, load_runtime, parser
from self_coding import CustomGenerator
from validation import validate_detector, validate_model_config


FIXTURES = Path(__file__).resolve().parent / 'fixtures'


def deepseek_tokenizer():
    config = json.loads((FIXTURES / 'tokenizer_config.json').read_text())
    bos, eos = config['bos_token']['content'], config['eos_token']['content']
    backend = Tokenizer(WordLevel({'[UNK]': 0, bos: 1, eos: 2, 'Q': 3, 'answer': 4, '0.5': 5},
                                 unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single=bos + ' $A', special_tokens=[(bos, 1)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', bos_token=bos,
                                  eos_token=eos, pad_token=eos, chat_template=config['chat_template'])


class DeepSeekTests(unittest.TestCase):
    def test_supplied_four_detectors_fit_original_14b_config(self):
        config = Qwen2Config.from_json_file(FIXTURES / 'config.json')
        validate_model_config(config)
        payload = json.loads(DEFAULT_CONFIG.read_text())
        prefix = 'DeepSeek-R1-Distill-Qwen-14B-'
        keys = [key for key in payload if key.startswith(prefix)]
        self.assertEqual({key.removeprefix(prefix) for key in keys},
                         {'math', 'science', 'logic', 'multihop-reasoning'})
        for key in keys:
            with self.subTest(key=key):
                validate_detector(payload[key], config)

    def test_wrong_architecture_sliding_layers_and_dynamic_rope_are_rejected(self):
        cases = [Qwen3Config(), Qwen2Config(use_sliding_window=True)]
        hybrid = Qwen2Config(num_hidden_layers=2)
        hybrid.layer_types = ['full_attention', 'sliding_attention']
        cases.append(hybrid)
        for rope_type in ('dynamic', 'longrope'):
            config = Qwen2Config()
            config.rope_scaling = {'rope_type': rope_type, 'factor': 2.0}
            cases.append(config)
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_model_config(config)

    def test_unknown_key_and_wrong_checkpoint_size_fail_before_download(self):
        for key, model in [
            ('missing', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B'),
            ('DeepSeek-R1-Distill-Qwen-14B-math', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'),
        ]:
            args = parser().parse_args(['--input', 'unused', '--model', model, '--model-key', key,
                                        '--output', 'unused-output'])
            with patch('transformers.AutoConfig.from_pretrained') as config_loader:
                with self.assertRaises(ValueError):
                    load_runtime(args, 'cpu')
                config_loader.assert_not_called()

    def test_invalid_detector_indices_fail_before_weight_loading_on_both_backends(self):
        for backend in ('selective', 'eager'):
            for head in ({'layer': 2, 'head': 0}, {'layer': 0, 'head': 2}):
                with self.subTest(backend=backend, head=head), tempfile.TemporaryDirectory() as directory:
                    detector = Path(directory) / 'detector.json'
                    detector.write_text(json.dumps({'test': {'heads': [head], 'threshold': 0.2,
                                                             'window_size': 3}}))
                    args = parser().parse_args(['--input', 'unused', '--output', 'unused-output',
                                                '--model', directory, '--model-key', 'test',
                                                '--detector-config', str(detector), '--attention-backend', backend])
                    config = Qwen2Config(num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
                    with patch('transformers.AutoConfig.from_pretrained', return_value=config), \
                         patch('transformers.AutoModelForCausalLM.from_pretrained') as weights:
                        with self.assertRaisesRegex(ValueError, 'indices'):
                            load_runtime(args, 'cpu')
                        weights.assert_not_called()

    def test_native_template_bos_privileged_span_and_eos(self):
        tokenizer = deepseek_tokenizer()
        model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, eos_token_id=2,
        )).eval()
        model.config._attn_implementation = 'eager'
        generator = CustomGenerator(model, tokenizer, None, 'test', prompt='Q',
                                    privileged_context=' answer 0.5', max_new_tokens=4,
                                    detector_config={'test': {'heads': [{'layer': 0, 'head': 0}],
                                                              'threshold': 2, 'window_size': 3}})
        expected = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Q answer 0.5'}],
                                                  tokenize=False, add_generation_prompt=True)
        self.assertEqual(generator.full_prompt, expected)
        self.assertEqual(expected.count(tokenizer.bos_token), 1)
        self.assertEqual(expected.count('<think>'), 1)
        self.assertIn('<｜Assistant｜><think>', expected)
        self.assertNotIn('answer 0.5', generator.clean_generator.full_prompt)
        self.assertEqual(generator.clean_generator.full_prompt.count('<think>'), 1)
        generator.begin()
        ids = generator._prompt_token_ids()
        self.assertEqual(ids.tolist().count(tokenizer.bos_token_id), 1)
        self.assertTrue(all(ids[index] != tokenizer.bos_token_id
                            for index in generator.privileged_context_token_indices))
        generator._sample_token = lambda _: torch.tensor([[tokenizer.eos_token_id]])
        _, _, finished = generator.step()
        self.assertTrue(finished)
        self.assertEqual(generator.generated_ids, [tokenizer.eos_token_id])
        generator._restore_prefix([])
        self.assertFalse(generator.finished)


if __name__ == '__main__':
    unittest.main()
