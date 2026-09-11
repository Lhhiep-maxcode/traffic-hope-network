import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch


using_torch_stub = False
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    # These tests exercise the repair state machine, not tensor operations.
    # A tiny import stub keeps the regression test runnable without installing
    # the project's heavyweight model dependency.
    torch_stub = types.ModuleType("torch")

    def no_grad():
        return lambda function: function

    torch_stub.no_grad = no_grad
    sys.modules["torch"] = torch_stub
    using_torch_stub = True

using_tqdm_stub = False
try:
    from tqdm.auto import tqdm as _tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_auto_stub = types.ModuleType("tqdm.auto")
    tqdm_auto_stub.tqdm = lambda *args, **kwargs: None
    tqdm_stub.auto = tqdm_auto_stub
    sys.modules["tqdm"] = tqdm_stub
    sys.modules["tqdm.auto"] = tqdm_auto_stub
    using_tqdm_stub = True


MODULE_PATH = Path(__file__).resolve().parents[1] / "self_coding.py"
SPEC = importlib.util.spec_from_file_location("self_coding", MODULE_PATH)
self_coding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(self_coding)
if using_torch_stub:
    sys.modules.pop("torch", None)
if using_tqdm_stub:
    sys.modules.pop("tqdm.auto", None)
    sys.modules.pop("tqdm", None)


class FakeSample:
    def __init__(self, token_id=2):
        self.token_id = token_id

    def item(self):
        return self.token_id


class FakeTokenizer:
    eos_token_id = None

    def apply_chat_template(self, messages, **kwargs):
        return f"user:{messages[0]['content']}|thinking:{kwargs['enable_thinking']}"

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(str(token_id) for token_id in token_ids)


class FakeCleanGenerator:
    next_logits = object()

    def start(self, generated_ids=None):
        self.generated_ids = list(generated_ids or [])

    def _sample_token(self, logits):
        del logits
        return FakeSample()


class FakeProgressBar:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.n = 0

    def update(self, amount):
        self.n += amount

    def refresh(self):
        pass

    def close(self):
        pass


def fake_start(generator, generated_ids=None):
    generator.generated_ids = list(generated_ids or [])
    generator.next_logits = object()
    generator.finished = False
    return 0


class CustomGeneratorLoopTest(unittest.TestCase):
    def make_generator(self, **overrides):
        generator = self_coding.CustomGenerator.__new__(
            self_coding.CustomGenerator
        )
        attributes = {
            "tokenizer": FakeTokenizer(),
            "clean_generator": FakeCleanGenerator(),
            "max_new_tokens": 8,
            "fix_backtrack": 2,
            "fix_comparison_method": "attention_score",
            "fix_js_divergence_threshold": 0.1,
            "debug_clean_backtrack": False,
            "debug_fix_infinite_loop": True,
            "debug_wait_safe_window": False,
            "max_repair_steps": 10,
            "max_repair_cycles_per_span": 1_000,
            "max_attempts": 1_000,
            "threshold": 0.5,
            "seed": None,
            "generated_ids": [],
            "finished": False,
            "repairing": False,
            "repair_steps": 0,
            "repair_safe_steps": 0,
            "repair_required_safe_steps": 1,
            "repair_cycles_by_start": {},
            "repair_events": [],
            "attention_score": None,
            "attempts": 0,
        }
        attributes.update(overrides)
        for name, value in attributes.items():
            setattr(generator, name, value)

        generator._sample_token = MethodType(
            lambda self, logits: FakeSample(),
            generator,
        )
        generator._forward_token = MethodType(
            lambda self, token_id, output_attentions=False: SimpleNamespace(
                attentions=object()
            ),
            generator,
        )

        def accept_on_both_branches(self, token_id, privileged_outputs=None):
            del privileged_outputs
            self.generated_ids.append(int(token_id))

        generator._accept_on_both_branches = MethodType(
            accept_on_both_branches,
            generator,
        )
        return generator

    def run_generator(self, generator):
        with (
            patch.object(self_coding.TokenByTokenGenerator, "start", fake_start),
            patch.object(self_coding, "tqdm", FakeProgressBar),
        ):
            return generator.generate(include_unfixed=False, show_progress=False)

    def test_repair_cannot_exit_before_replacing_backtracked_span(self):
        generator = self.make_generator()
        generator._attention_score = MethodType(
            lambda self, _: 1.0 if not self.repairing else 0.0,
            generator,
        )

        result = self.run_generator(generator)

        self.assertEqual(len(result["token_ids"]), 8)
        self.assertLess(generator.attempts, 100)
        self.assertEqual(generator.repair_events[1]["required_safe_steps"], 2)
        self.assertEqual(
            len(generator.repair_events[1]["replacement_token_ids"]),
            2,
        )

    def test_non_converging_repair_is_bounded(self):
        for enabled in (True, False):
            with self.subTest(debug_fix_infinite_loop=enabled):
                generator = self.make_generator(
                    max_repair_steps=4,
                    debug_fix_infinite_loop=enabled,
                )
                generator._attention_score = MethodType(
                    lambda self, _: 1.0, generator
                )

                with self.assertRaisesRegex(RuntimeError, "max_repair_steps"):
                    self.run_generator(generator)

                self.assertEqual(generator.attempts, 4)

    def test_disabled_loop_fix_restores_oscillation_but_keeps_attempt_limit(self):
        for method in ("attention_score", "js_divergence"):
            with self.subTest(method=method):
                generator = self.make_generator(
                    debug_fix_infinite_loop=False,
                    fix_comparison_method=method,
                    max_repair_cycles_per_span=1,
                    max_attempts=6,
                )
                generator._attention_score = MethodType(
                    lambda self, _: 1.0 if not self.repairing else 0.0,
                    generator,
                )
                generator.jensen_shannon_score = Mock(return_value=0.0)

                with self.assertRaisesRegex(RuntimeError, "too many repair attempts"):
                    self.run_generator(generator)

                self.assertEqual(len(generator.generated_ids), 1)
                self.assertEqual(len(generator.repair_events), 6)
                self.assertGreater(generator.repair_events[-1]["cycle_count"], 1)
                for event in generator.repair_events:
                    self.assertFalse(event["debug_fix_infinite_loop"])
                    self.assertFalse(event["backtrack_suppressed"])
                    self.assertEqual(event["required_safe_steps"], 1)
                    self.assertEqual(len(event["replacement_token_ids"]), 1)

    def test_loop_fix_defaults_on_and_controls_minimum_repair_budget(self):
        config = '{"test": {"heads": [], "threshold": 0.5, "window_size": 5}}'
        cases = [
            ({}, True, False),
            ({"debug_fix_infinite_loop": False}, False, False),
            ({"max_repair_steps": 1}, True, True),
            ({"debug_fix_infinite_loop": False, "max_repair_steps": 1}, False, False),
            (
                {
                    "debug_fix_infinite_loop": False,
                    "debug_clean_backtrack": True,
                    "max_repair_steps": 1,
                },
                False,
                True,
            ),
        ]
        for options, expected_flag, raises in cases:
            with (
                self.subTest(options=options),
                patch.object(self_coding.Path, "read_text", return_value=config),
                patch.object(self_coding, "_substring_token_indices", return_value=[1]),
            ):
                arguments = {
                    "model": object(),
                    "tokenizer": FakeTokenizer(),
                    "detector_config_path": "unused.json",
                    "model_key": "test",
                    "prompt": "question",
                    "privileged_context": " secret answer",
                    **options,
                }
                if raises:
                    with self.assertRaisesRegex(ValueError, "fix_backtrack"):
                        self_coding.CustomGenerator(**arguments)
                else:
                    generator = self_coding.CustomGenerator(**arguments)
                    self.assertEqual(generator.debug_fix_infinite_loop, expected_flag)
                    self.assertFalse(generator.debug_wait_safe_window)

    def test_repeated_span_eventually_disables_backtracking(self):
        generator = self.make_generator(max_repair_cycles_per_span=2)
        generator._attention_score = MethodType(
            lambda self, _: 1.0 if not self.repairing else 0.0,
            generator,
        )

        self.run_generator(generator)

        suppressed_event = generator.repair_events[2]
        self.assertTrue(suppressed_event["backtrack_suppressed"])
        self.assertEqual(
            suppressed_event["start_index"],
            suppressed_event["detected_index"],
        )

    def test_fixed_generation_starts_from_requested_seed(self):
        generator = self.make_generator(seed=123)
        generator._attention_score = MethodType(
            lambda self, _: 1.0 if not self.repairing else 0.0,
            generator,
        )

        with patch.object(
            self_coding.torch,
            "manual_seed",
            create=True,
        ) as manual_seed:
            self.run_generator(generator)

        manual_seed.assert_called_once_with(123)

    def make_debug_generator(self, **overrides):
        attributes = {
            "debug_clean_backtrack": True,
            "generated_ids": [1] * 82 + [308, 13],
            "max_new_tokens": 128,
        }
        attributes.update(overrides)
        generator = self.make_generator(**attributes)
        generator._sample_token = Mock(return_value=FakeSample(9))
        generator._attention_score = Mock(return_value=0.0)
        generator.jensen_shannon_score = Mock(return_value=0.0)
        generator._forward_token = Mock(return_value=SimpleNamespace(attentions=object()))

        # Use real branch synchronization, mocking only model transitions.
        del generator._accept_on_both_branches

        def accept(token_id, outputs):
            del outputs
            generator.generated_ids.append(token_id)
            generator.finished = token_id == generator.tokenizer.eos_token_id

        generator._accept = accept
        clean = generator.clean_generator
        clean._sample_token = Mock(return_value=FakeSample(7))
        clean._forward_token = Mock(return_value=object())
        clean._accept = lambda token_id, _: clean.generated_ids.append(token_id)
        return generator

    def test_debug_flag_is_opt_in_and_branches_use_distinct_prompts(self):
        config = '{"test": {"heads": [], "threshold": 0.5, "window_size": 5}}'
        for options in ({}, {"debug_clean_backtrack": True}):
            with (
                self.subTest(options=options),
                patch.object(self_coding.Path, "read_text", return_value=config),
                patch.object(self_coding, "_substring_token_indices", return_value=[1]),
            ):
                generator = self_coding.CustomGenerator(
                    model=object(),
                    tokenizer=FakeTokenizer(),
                    detector_config_path="unused.json",
                    model_key="test",
                    prompt="question",
                    privileged_context=" secret answer",
                    enable_thinking=False,
                    **options,
                )
                self.assertEqual(generator.debug_clean_backtrack, bool(options))
                self.assertEqual(
                    generator.full_prompt, "user:question secret answer|thinking:False"
                )
                self.assertEqual(
                    generator.clean_generator.full_prompt, "user:question|thinking:False"
                )

    def test_debug_backtrack_skips_comparison_until_detected_position(self):
        for method in ("attention_score", "js_divergence"):
            for score in (0.0, 1.0):
                with (
                    self.subTest(method=method, score=score),
                    patch.object(self_coding.TokenByTokenGenerator, "start", fake_start),
                ):
                    generator = self.make_debug_generator(fix_comparison_method=method)
                    generator._attention_score.return_value = score
                    generator.jensen_shannon_score.return_value = score
                    clean = generator.clean_generator
                    generator._start_repair(576, 1.0)
                    for _ in range(2):
                        generator._repair_step()
                        self.assertEqual(generator.generated_ids, clean.generated_ids)

                    self.assertEqual(generator.generated_ids[-2:], [7, 7])
                    generator._sample_token.assert_not_called()
                    generator._attention_score.assert_not_called()
                    generator.jensen_shannon_score.assert_not_called()
                    self.assertEqual(clean._sample_token.call_count, 2)
                    self.assertTrue(generator.repairing)
                    self.assertEqual(generator.repair_safe_steps, 0)
                    self.assertIsNone(generator.attention_score)
                    for call in generator._forward_token.call_args_list:
                        self.assertFalse(call.kwargs.get("output_attentions", False))

                    selected_id, _, _ = generator._repair_step()
                    self.assertEqual(selected_id, 7 if score else 9)
                    generator._sample_token.assert_called_once()
                    comparator = (
                        generator._attention_score
                        if method == "attention_score"
                        else generator.jensen_shannon_score
                    )
                    comparator.assert_called_once()
                    self.assertEqual(generator.generated_ids, clean.generated_ids)
                    event = generator.repair_events[-1]
                    self.assertEqual(event["clean_backtrack_token_ids"], [7, 7])
                    self.assertEqual(event["replacement_token_ids"], [7, 7, selected_id])

    def test_disabled_debug_flag_still_checks_backtracked_tokens(self):
        generator = self.make_debug_generator(debug_clean_backtrack=False)
        with patch.object(self_coding.TokenByTokenGenerator, "start", fake_start):
            generator._start_repair(576, 1.0)
            generator._repair_step()
            generator._repair_step()

        self.assertEqual(generator.generated_ids[-2:], [9, 9])
        generator.clean_generator._sample_token.assert_not_called()
        self.assertEqual(generator._attention_score.call_count, 2)
        self.assertEqual(generator.repair_events[-1]["clean_backtrack_token_ids"], [])

    def test_clean_backtrack_remains_active_with_loop_fix_disabled(self):
        generator = self.make_debug_generator(debug_fix_infinite_loop=False)
        with patch.object(self_coding.TokenByTokenGenerator, "start", fake_start):
            generator._start_repair(576, 1.0)
            generator._repair_step()
            generator._repair_step()
            self.assertTrue(generator.repairing)
            generator._repair_step()

        self.assertFalse(generator.repairing)
        self.assertEqual(generator.generated_ids[-3:], [7, 7, 9])
        self.assertEqual(generator.generated_ids, generator.clean_generator.generated_ids)
        generator._attention_score.assert_called_once()
        event = generator.repair_events[-1]
        self.assertTrue(event["debug_clean_backtrack"])
        self.assertFalse(event["debug_fix_infinite_loop"])
        self.assertEqual(event["required_safe_steps"], 1)

    def test_debug_backtrack_handles_short_prefix_and_suppressed_backtrack(self):
        cases = [
            ({"generated_ids": []}, 0),
            ({"generated_ids": [308]}, 1),
            ({"repair_cycles_by_start": {82: 2}, "max_repair_cycles_per_span": 2}, 0),
        ]
        for options, forced_steps in cases:
            with (
                self.subTest(options=options),
                patch.object(self_coding.TokenByTokenGenerator, "start", fake_start),
            ):
                generator = self.make_debug_generator(**options)
                generator._start_repair(576, 1.0)
                for _ in range(forced_steps):
                    generator._repair_step()
                generator._sample_token.assert_not_called()
                generator._repair_step()
                generator._sample_token.assert_called_once()
                generator._attention_score.assert_called_once()
                self.assertEqual(
                    generator.repair_events[-1]["clean_backtrack_token_ids"],
                    [7] * forced_steps,
                )

    def test_debug_backtrack_respects_eos_and_repair_budget(self):
        with patch.object(self_coding.TokenByTokenGenerator, "start", fake_start):
            generator = self.make_debug_generator()
            generator.tokenizer.eos_token_id = 7
            generator._start_repair(576, 1.0)
            token_id, _, finished = generator._repair_step()
            self.assertEqual(token_id, 7)
            self.assertTrue(finished)
            self.assertFalse(generator.repairing)
            generator._attention_score.assert_not_called()

            generator = self.make_debug_generator(max_repair_steps=3)
            generator._attention_score.return_value = 1.0
            generator._start_repair(576, 1.0)
            generator._repair_step()
            generator._repair_step()
            with self.assertRaisesRegex(RuntimeError, "max_repair_steps"):
                generator._repair_step()

    def test_debug_backtrack_respects_output_limit(self):
        generator = self.make_debug_generator(max_new_tokens=83)
        with patch.object(self_coding.TokenByTokenGenerator, "start", fake_start):
            generator._start_repair(576, 1.0)
            generator.step()
            self.assertEqual(generator.step(), (None, None, True))

        self.assertEqual(len(generator.generated_ids), 83)
        generator._attention_score.assert_not_called()
        self.assertEqual(generator.repair_events[-1]["clean_backtrack_token_ids"], [7])

    def test_safe_window_holds_clean_until_consecutive_checks_pass(self):
        for method in ("attention_score", "js_divergence"):
            for loop_fix in (False, True):
                with (
                    self.subTest(method=method, loop_fix=loop_fix),
                    patch.object(self_coding.TokenByTokenGenerator, "start", fake_start),
                ):
                    generator = self.make_debug_generator(
                        debug_wait_safe_window=True,
                        debug_fix_infinite_loop=loop_fix,
                        fix_comparison_method=method,
                    )
                    comparator = (
                        generator._attention_score
                        if method == "attention_score"
                        else generator.jensen_shannon_score
                    )
                    comparator.side_effect = [0.0, 1.0, 0.0, 0.0]
                    generator._start_repair(576, 1.0)
                    expected_streaks = [0, 0, 1, 0, 1, 2]
                    for index, streak in enumerate(expected_streaks):
                        token_id, _, _ = generator.step()
                        self.assertEqual(token_id, 7)
                        self.assertEqual(
                            generator.generated_ids, generator.clean_generator.generated_ids
                        )
                        self.assertEqual(generator.repairing, index < 5)
                        self.assertEqual(
                            generator.repair_events[-1]["steps"][-1]["safe_steps"], streak
                        )

                    self.assertEqual(comparator.call_count, 4)
                    event = generator.repair_events[-1]
                    self.assertTrue(event["debug_wait_safe_window"])
                    self.assertEqual(event["required_safe_steps"], 2)
                    self.assertEqual(event["replacement_token_ids"], [7] * 6)
                    self.assertEqual(
                        [step["token_index"] for step in event["steps"]],
                        list(range(82, 88)),
                    )
                    self.assertEqual(
                        [step["passed_check"] for step in event["steps"]],
                        [None, None, True, False, True, True],
                    )
                    self.assertTrue(
                        all(step["selected_branch"] == "clean" for step in event["steps"])
                    )

                    # Only the next ordinary generation step uses privileged.
                    comparator.side_effect = None
                    comparator.return_value = 0.0
                    token_id, _, _ = generator.step()
                    self.assertEqual(token_id, 9)
                    self.assertEqual(len(generator.repair_events), 1)
                    self.assertEqual(len(event["replacement_token_ids"]), 6)

    def test_safe_window_keeps_the_optional_anti_loop_span_guard(self):
        for loop_fix, expected_steps in ((False, 2), (True, 3)):
            with (
                self.subTest(loop_fix=loop_fix),
                patch.object(self_coding.TokenByTokenGenerator, "start", fake_start),
            ):
                generator = self.make_debug_generator(
                    debug_wait_safe_window=True,
                    debug_clean_backtrack=False,
                    debug_fix_infinite_loop=loop_fix,
                )
                generator._start_repair(576, 1.0)
                for index in range(expected_steps):
                    self.assertEqual(generator.step()[0], 7)
                    self.assertEqual(generator.repairing, index < expected_steps - 1)

                event = generator.repair_events[-1]
                self.assertEqual(event["required_safe_steps"], 2)
                self.assertEqual(len(event["replacement_token_ids"]), expected_steps)
                self.assertEqual(event["clean_backtrack_token_ids"], [])

    def test_safe_window_size_and_minimum_repair_budget(self):
        cases = [
            # window, clean backtrack, anti-loop guard, minimum budget, safe checks
            (1, False, False, 1, 1),
            (3, True, False, 2, 1),
            (5, True, True, 4, 2),
            (5, False, False, 2, 2),
            (5, False, True, 3, 2),
            (9, True, True, 8, 4),
        ]
        for window, clean_backtrack, loop_fix, budget, safe_checks in cases:
            config = (
                '{"test": {"heads": [], "threshold": 0.5, "window_size": '
                + str(window)
                + '}}'
            )
            with (
                self.subTest(window=window, clean_backtrack=clean_backtrack, loop_fix=loop_fix),
                patch.object(self_coding.Path, "read_text", return_value=config),
                patch.object(self_coding, "_substring_token_indices", return_value=[1]),
                patch.object(self_coding.TokenByTokenGenerator, "start", fake_start),
            ):
                arguments = {
                    "model": object(),
                    "tokenizer": FakeTokenizer(),
                    "detector_config_path": "unused.json",
                    "model_key": "test",
                    "prompt": "question",
                    "privileged_context": " secret answer",
                    "debug_wait_safe_window": True,
                    "debug_clean_backtrack": clean_backtrack,
                    "debug_fix_infinite_loop": loop_fix,
                }
                generator = self_coding.CustomGenerator(max_repair_steps=budget, **arguments)
                generator._start_repair(9, 1.0)
                self.assertEqual(generator.repair_required_safe_steps, safe_checks)
                with self.assertRaisesRegex(ValueError, "max_repair_steps"):
                    self_coding.CustomGenerator(max_repair_steps=budget - 1, **arguments)

    def test_safe_window_discards_safe_privileged_speculation_from_cache(self):
        generator = self.make_debug_generator(
            debug_wait_safe_window=True,
            debug_clean_backtrack=False,
            generated_ids=[],
        )

        def rebuild_cache(branch, generated_ids=None):
            fake_start(branch, generated_ids)
            branch.cache_tokens = list(branch.generated_ids)

        def forward_token(token_id, output_attentions=False):
            del output_attentions
            generator.cache_tokens.append(token_id)
            return SimpleNamespace(attentions=object())

        generator._forward_token.side_effect = forward_token
        with patch.object(self_coding.TokenByTokenGenerator, "start", rebuild_cache):
            generator._start_repair(576, 1.0)
            for _ in range(2):
                self.assertEqual(generator.step()[0], 7)
                self.assertEqual(generator.cache_tokens, generator.generated_ids)
            self.assertFalse(generator.repairing)
            self.assertEqual(generator.step()[0], 9)
            self.assertEqual(generator.cache_tokens, [7, 7, 9])


if __name__ == "__main__":
    unittest.main()
