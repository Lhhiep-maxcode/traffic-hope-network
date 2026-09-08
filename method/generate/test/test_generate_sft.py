"""Resume/export/process regressions; no model download or torch required."""

import json
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from method.generate import generate_sft as bulk
from method.generate.modeling import load_model


def fake_initialize(config, device):
    pass


def fake_generate(job, config, runtime):
    if job[0] == "crash":
        os._exit(9)
    return success(job[0])


def success(text="trace", finish_reason="eos"):
    return {
        "status": "success",
        "assistant_text": text,
        "finish_reason": finish_reason,
        "attempts": 1,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class BulkTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.output = self.root / "output.jsonl"
        self.sft = self.root / "output.sft.jsonl"
        self.source.write_text(
            json.dumps({"question": "Limit?", "ground_truth": 0}) + "\n"
        )

    def args(self, *extra):
        return bulk.parse_args(
            [
                "--input",
                str(self.source),
                "--output",
                str(self.output),
                "--model",
                "Qwen/Qwen3-4B",
                *extra,
            ]
        )

    def cache(self, config=None, args=None):
        cache = bulk.Cache(self.root / "cache.sqlite", config or {"seed": 42})
        self.addCleanup(cache.db.close)
        cache.ingest(args or self.args())
        return cache

    def test_prompts_preserve_math_and_numeric_zero(self):
        question = r"Evaluate: \lim_{x \to \infty} \sqrt{x}"
        prepared = bulk.prepare_sample({"question": question, "ground_truth": 0})
        self.assertEqual(
            prepared["prompt_wo_answer"],
            question + "\n\nExplain your solution step by step.",
        )
        self.assertEqual(
            prepared["privileged_context"], "Given the ground truth answer is 0"
        )
        self.assertEqual(
            prepared["prompt_w_answer"],
            prepared["prompt_wo_answer"] + " Given the ground truth answer is 0",
        )
        for invalid in [None, "", [], {}, True, float("nan")]:
            with self.subTest(answer=invalid), self.assertRaises(ValueError):
                bulk.prepare_sample({"question": question, "ground_truth": invalid})

    def test_resume_deduplicates_jobs_but_preserves_input_order_and_metadata(self):
        rows = [
            {"question": "A?", "ground_truth": 0, "id": "first"},
            {"question": "B?", "ground_truth": "2", "id": "second"},
            {"question": "A?", "ground_truth": 0, "id": "duplicate"},
        ]
        self.source.write_text("\n".join(map(json.dumps, rows)))
        cache = self.cache()
        count, jobs = cache.pending()
        jobs = list(jobs)
        self.assertEqual(count, 2)
        cache.save(jobs[0][0], success("A trace"))
        cache.db.close()
        # Simulate restart, reorder input, and retain the same cache.
        self.source.write_text("\n".join(map(json.dumps, reversed(rows))))
        resumed = self.cache()
        count, pending = resumed.pending()
        self.assertEqual(count, 1)
        remaining = next(pending)
        self.assertEqual(remaining[1]["question"], "B?")
        resumed.save(remaining[0], success("B trace"))
        counts = resumed.export(self.output, self.sft)
        self.assertEqual(
            counts, {"success": 3, "error": 0, "pending": 0, "skipped": 0, "sft": 3}
        )
        details, sft = read_jsonl(self.output), read_jsonl(self.sft)
        self.assertEqual(
            [r["input_sample"]["id"] for r in details], ["duplicate", "second", "first"]
        )
        self.assertEqual(
            [r["messages"][1]["content"] for r in sft],
            ["A trace", "B trace", "A trace"],
        )
        self.assertNotIn("Given the ground truth", sft[0]["messages"][0]["content"])

    def test_config_change_and_multiple_traces_have_independent_cache_keys(self):
        first = self.cache(args=self.args("--num-traces", "2"))
        count, jobs = first.pending()
        jobs = list(jobs)
        self.assertEqual(count, 2)
        self.assertNotEqual(jobs[0][0], jobs[1][0])
        first.save(jobs[0][0], success())
        first.db.close()
        changed = self.cache({"seed": 43})
        count, _ = changed.pending()
        self.assertEqual(count, 1)

    def test_errors_retry_and_truncated_traces_can_be_excluded(self):
        cache = self.cache()
        _, jobs = cache.pending()
        key, _ = next(jobs)
        cache.save(key, {"status": "error", "attempts": 2, "error": {"message": "OOM"}})
        self.assertEqual(cache.export(self.output, self.sft)["error"], 1)
        self.assertEqual(read_jsonl(self.sft), [])
        cache.db.close()
        resumed = self.cache()
        count, jobs = resumed.pending()
        self.assertEqual(count, 1)
        next(jobs)
        resumed.save(key, success(finish_reason="length"))
        counts = resumed.export(self.output, self.sft, require_eos=True)
        self.assertEqual(counts["success"], 1)
        self.assertEqual(counts["sft"], 0)
        self.assertEqual(read_jsonl(self.output)[0]["attempts"], 3)
        self.assertEqual(resumed.export(self.output, self.sft)["sft"], 1)

    def test_atomic_export_retains_previous_file_on_exception(self):
        self.output.write_text("original\n")
        with self.assertRaises(RuntimeError), bulk.atomic_jsonl(self.output) as output:
            output.write("partial")
            raise RuntimeError("interrupted")
        self.assertEqual(self.output.read_text(), "original\n")
        self.assertEqual(list(self.root.glob(".output.jsonl.*")), [])

    def test_real_spawn_workers_and_crash_leave_committed_results_resumable(self):
        saved = {}
        jobs = iter([(str(i), {}) for i in range(8)])
        bulk.run_jobs(
            jobs,
            8,
            {},
            ["cpu", "cpu"],
            saved.__setitem__,
            initializer=fake_initialize,
            generate=fake_generate,
        )
        self.assertEqual(set(saved), {str(i) for i in range(8)})
        cache = self.cache()
        with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
            bulk.run_jobs(
                iter([("saved", {}), ("crash", {})]),
                2,
                {},
                ["cpu"],
                cache.save,
                initializer=fake_initialize,
                generate=fake_generate,
            )
        cache.db.close()
        resumed = self.cache()
        row = resumed.db.execute(
            "SELECT status FROM results WHERE sample_id='saved'"
        ).fetchone()
        self.assertEqual(row, ("success",))

    def test_prepare_and_export_only_work_without_torch(self):
        common = ["--input", str(self.source), "--output", str(self.output)]
        self.assertEqual(bulk.main([*common, "--prepare-only"]), 0)
        self.assertEqual(read_jsonl(self.output)[0]["ground_truth"], "0")
        self.assertEqual(
            bulk.main([*common, "--model", "Qwen/Qwen3-4B", "--export-only"]), 1
        )
        self.assertEqual(read_jsonl(self.output)[0]["status"], "pending")
        self.assertEqual(read_jsonl(self.sft), [])

    def test_interrupt_after_commit_keeps_only_unfinished_jobs_pending(self):
        cache = self.cache(args=self.args("--num-traces", "3"))
        count, jobs = cache.pending()

        def save_then_interrupt(key, result):
            cache.save(key, result)
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            bulk.run_jobs(
                jobs,
                count,
                {},
                ["cpu"],
                save_then_interrupt,
                initializer=fake_initialize,
                generate=fake_generate,
            )
        cache.db.close()
        resumed = self.cache(args=self.args("--num-traces", "3"))
        count, _ = resumed.pending()
        self.assertEqual(count, 2)
        counts = resumed.export(self.output, self.sft)
        self.assertEqual(counts["success"], 1)
        self.assertEqual(counts["pending"], 2)

    def test_fingerprint_changes_with_generation_but_not_device_or_retry_count(self):
        baseline = bulk.make_config(self.args())
        self.assertEqual(
            baseline,
            bulk.make_config(self.args("--devices", "cpu", "--max-retries", "5")),
        )
        self.assertNotEqual(
            baseline, bulk.make_config(self.args("--temperature", "0.2"))
        )

    def test_cache_lock_rejects_second_writer(self):
        with (
            bulk.cache_lock(self.root / "locked.sqlite"),
            self.assertRaisesRegex(RuntimeError, "Another process"),
            bulk.cache_lock(self.root / "locked.sqlite"),
        ):
            self.fail("Second writer acquired the cache")

    def test_adapter_exact_prompts_seed_and_thinking_markers(self):
        captured = []

        class Tokenizer:
            eos_token_id = 99

            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"] + "\nAssistant:<think>\n"

            def decode(self, ids, skip_special_tokens=False):
                self.asserted_ids = ids
                return "steps</think>answer"

        class Generator:
            def __init__(self, **kwargs):
                captured.append(self)
                self.kwargs = kwargs

            def generate(self, include_unfixed, show_progress):
                if include_unfixed or show_progress:
                    raise AssertionError("Unneeded baseline/progress enabled")
                return {
                    "text": "steps answer",
                    "token_ids": [1, 2, 99],
                    "repair_events": [],
                }

        tokenizer = Tokenizer()
        pipeline = SimpleNamespace(
            CustomGenerator=Generator,
            _substring_token_indices=lambda tokenizer, text, context: [
                text.index(context)
            ],
        )
        torch = SimpleNamespace(
            inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: False)
        )
        config = {
            **bulk.make_config(self.args()),
            "detector_config_path": "unused",
            "max_retries": 0,
        }
        prepared = bulk.prepare_sample({"question": "Test?", "ground_truth": 0})
        runtime = (pipeline, object(), tokenizer, torch, "cpu")
        result = bulk.generate_one(("a" * 64, prepared), config, runtime)
        repeated = bulk.generate_one(("a" * 64, prepared), config, runtime)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["assistant_text"], "<think>\nsteps</think>answer")
        self.assertEqual(tokenizer.asserted_ids, [1, 2])
        self.assertEqual(result["seed"], repeated["seed"])
        self.assertEqual(captured[0].kwargs["prompt"], prepared["prompt_wo_answer"])
        self.assertEqual(
            captured[0].kwargs["privileged_context"], prepared["privileged_context"]
        )
        self.assertEqual(
            captured[0].full_prompt,
            prepared["prompt_w_answer"] + "\nAssistant:<think>\n",
        )
        config["max_retries"] = 1
        with (
            patch.object(bulk.time, "sleep"),
            patch.object(
                Generator,
                "generate",
                side_effect=[
                    RuntimeError("transient"),
                    {
                        "text": "steps answer",
                        "token_ids": [1, 2, 99],
                        "repair_events": [],
                    },
                ],
            ),
        ):
            retried = bulk.generate_one(("a" * 64, prepared), config, runtime)
        self.assertEqual(retried["status"], "success")
        self.assertEqual(retried["attempts"], 2)
        self.assertEqual(retried["seed"], result["seed"])

    def test_notebook_api_ignores_kernel_arguments_runs_inline_and_resumes(self):
        config = bulk.SFTConfig(
            input=self.source,
            output=self.output,
            model="Qwen/Qwen3-4B",
            devices="cpu",
            num_traces=2,
        )
        with (
            patch.object(sys, "argv", ["ipykernel_launcher", "-f", "kernel.json"]),
            patch.object(
                bulk, "parse_args", side_effect=AssertionError("API parsed argv")
            ),
            patch.object(
                bulk.mp,
                "get_context",
                side_effect=AssertionError("Single device spawned"),
            ),
            patch.object(
                bulk, "initialize_worker", side_effect=fake_initialize
            ) as initialize,
            patch.object(bulk, "generate_one", side_effect=fake_generate) as generate,
        ):
            summary = bulk.generate_dataset(config)
            self.assertEqual(summary.counts["sft"], 2)
            self.assertEqual(summary.sft_output, self.sft.resolve())
            self.assertEqual(summary.exit_code, 0)
            initialize.assert_called_once()
            self.assertEqual(generate.call_count, 2)
            bulk.generate_dataset(config)
            self.assertEqual(generate.call_count, 2)
            initialize.assert_called_once()

    def test_notebook_interrupt_exports_committed_rows_before_propagating(self):
        config = bulk.SFTConfig(
            input=self.source,
            output=self.output,
            model="Qwen/Qwen3-4B",
            devices="cpu",
            num_traces=2,
        )
        with (
            patch.object(bulk, "initialize_worker", side_effect=fake_initialize),
            patch.object(
                bulk, "generate_one", side_effect=[success("saved"), KeyboardInterrupt]
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            bulk.generate_dataset(config)
        self.assertEqual(len(read_jsonl(self.sft)), 1)
        self.assertEqual(
            [r["status"] for r in read_jsonl(self.output)], ["success", "pending"]
        )
        with (
            patch.object(bulk, "initialize_worker", side_effect=fake_initialize),
            patch.object(bulk, "generate_one", side_effect=fake_generate) as generate,
        ):
            summary = bulk.generate_dataset(config)
            self.assertEqual(summary.counts["sft"], 2)
            generate.assert_called_once()

    def test_imported_api_uses_real_spawn_for_multiple_devices(self):
        config = bulk.SFTConfig(
            input=self.source,
            output=self.output,
            model="Qwen/Qwen3-4B",
            devices=("cuda:0", "cuda:1"),
            num_traces=4,
        )
        with (
            patch.object(bulk, "initialize_worker", fake_initialize),
            patch.object(bulk, "generate_one", fake_generate),
        ):
            summary = bulk.generate_dataset(config)
        self.assertEqual(summary.counts["sft"], 4)

    def test_api_validates_config_with_python_exceptions(self):
        for options in [
            {"devices": ()},
            {"devices": "cuda:0,cuda:0"},
            {"dtype": "invalid"},
            {"fix_comparison_method": "invalid"},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                bulk.SFTConfig(
                    input=self.source,
                    output=self.output,
                    model="Qwen/Qwen3-4B",
                    **options,
                )
        self.assertEqual(
            self.args(),
            bulk.SFTConfig(
                input=self.source, output=self.output, model="Qwen/Qwen3-4B"
            ),
        )

    def test_model_loader_passes_local_paths_and_eager_attention(self):
        for version, dtype_key in [("4.55.0", "torch_dtype"), ("5.0.0", "dtype")]:
            tokenizer = SimpleNamespace(pad_token_id=None, eos_token="eos")
            model = Mock()
            model.eval.return_value = model
            model.generation_config.pad_token_id = None
            hf = SimpleNamespace(
                __version__=version,
                AutoTokenizer=SimpleNamespace(
                    from_pretrained=Mock(return_value=tokenizer)
                ),
                AutoModelForCausalLM=SimpleNamespace(
                    from_pretrained=Mock(return_value=model)
                ),
            )
            with (
                self.subTest(version=version),
                patch.dict(
                    sys.modules,
                    {
                        "torch": SimpleNamespace(bfloat16="test-dtype"),
                        "transformers": hf,
                    },
                ),
            ):
                actual = load_model(
                    self.root / "model",
                    tokenizer_path=self.root / "tokenizer",
                    device="cuda:1",
                    dtype="bfloat16",
                    local_files_only=True,
                )
            self.assertEqual(actual, (model, tokenizer))
            self.assertEqual(tokenizer.pad_token, "eos")
            hf.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                str(self.root / "model"),
                device_map={"": "cuda:1"},
                attn_implementation="eager",
                revision="main",
                trust_remote_code=False,
                local_files_only=True,
                **{dtype_key: "test-dtype"},
            )

    def test_empty_answer_at_index_99592_does_not_abort_ingestion(self):
        valid = json.dumps({"question": "Limit?", "ground_truth": "0"}) + "\n"
        invalid = json.dumps({"question": "Empty answer?", "ground_truth": ""}) + "\n"
        last = json.dumps({"question": "After invalid?", "ground_truth": "1"}) + "\n"
        self.source.write_text(valid * 99592 + invalid + last)
        cache = self.cache()
        count, jobs = cache.pending()
        self.assertEqual(count, 2)  # Repeated valid rows still share one generation.
        self.assertEqual(
            [job[1]["question"] for job in jobs], ["Limit?", "After invalid?"]
        )
        row = cache.db.execute(
            "SELECT source_index, validation_error FROM inputs WHERE validation_error IS NOT NULL"
        ).fetchone()
        self.assertEqual(row[0], 99592)
        self.assertEqual(json.loads(row[1])["message"], "The answer must not be empty.")
        self.assertEqual(
            cache.db.execute("SELECT COUNT(*) FROM inputs").fetchone()[0], 99594
        )

    def test_skipped_sample_is_reported_excluded_from_sft_and_revalidated_on_resume(
        self,
    ):
        rows = [
            {"question": "A?", "ground_truth": 0},
            {"question": "B?", "ground_truth": "", "id": "invalid"},
            {"question": "C?", "ground_truth": "2"},
        ]
        self.source.write_text("\n".join(map(json.dumps, rows)))
        config = bulk.SFTConfig(
            input=self.source,
            output=self.output,
            model="Qwen/Qwen3-4B",
            devices="cpu",
            num_traces=2,
        )
        with (
            patch.object(
                bulk, "initialize_worker", side_effect=fake_initialize
            ) as initialize,
            patch.object(bulk, "generate_one", side_effect=fake_generate) as generate,
        ):
            summary = bulk.generate_dataset(config)
            self.assertEqual(
                summary.counts,
                {"success": 4, "error": 0, "pending": 0, "skipped": 2, "sft": 4},
            )
            self.assertEqual(summary.exit_code, 0)
            self.assertEqual(generate.call_count, 4)
            details = read_jsonl(self.output)
            self.assertEqual(
                [row["source_index"] for row in details], [0, 0, 1, 1, 2, 2]
            )
            self.assertEqual(details[2]["status"], "skipped")
            self.assertEqual(details[2]["input_sample"], rows[1])
            self.assertNotIn("messages", details[2])
            self.assertEqual(len(read_jsonl(self.sft)), 4)
            bulk.generate_dataset(config)
            self.assertEqual(generate.call_count, 4)
            initialize.assert_called_once()
            rows[1]["ground_truth"] = "3"
            self.source.write_text("\n".join(map(json.dumps, rows)))
            repaired = bulk.generate_dataset(config)
            self.assertEqual(generate.call_count, 6)
            self.assertEqual(repaired.counts["skipped"], 0)
            self.assertEqual(repaired.counts["success"], 6)

    def test_all_invalid_inputs_skip_model_loading_and_return_nonzero(self):
        rows = [
            {"question": "A?", "ground_truth": ""},
            {"question": "B?", "ground_truth": " \n\t"},
            {"question": "C?", "ground_truth": None},
            {"question": "D?"},
        ]
        self.source.write_text("\n".join(map(json.dumps, rows)))
        config = bulk.SFTConfig(
            input=self.source, output=self.output, model="Qwen/Qwen3-4B"
        )
        with patch.object(
            bulk, "initialize_worker", side_effect=AssertionError("No valid input")
        ):
            summary = bulk.generate_dataset(config)
        self.assertEqual(summary.counts["skipped"], 4)
        self.assertEqual(summary.exit_code, 1)
        self.assertEqual(read_jsonl(self.sft), [])

    def test_prepare_only_and_limit_use_same_skip_policy(self):
        self.source.write_text(
            "\n".join(
                map(
                    json.dumps,
                    [
                        {"problem": "A?", "answer": ""},
                        {"problem": "B?", "answer": 0},
                        {"problem": "C?", "answer": "1"},
                    ],
                )
            )
        )
        config = bulk.SFTConfig(
            input=self.source,
            output=self.output,
            question_field="problem",
            answer_field="answer",
            prepare_only=True,
            limit=2,
        )
        summary = bulk.generate_dataset(config)
        self.assertEqual(summary.counts, {"prepared": 1, "skipped": 1})
        self.assertEqual(summary.exit_code, 0)
        self.assertEqual(len(read_jsonl(self.output)), 2)
        self.assertEqual(read_jsonl(self.output)[1]["ground_truth"], "0")

    def test_strict_input_keeps_fail_fast_behavior_and_existing_output(self):
        self.source.write_text(json.dumps({"question": "A?", "ground_truth": ""}))
        self.output.write_text("previous output\n")
        config = bulk.SFTConfig(
            input=self.source, output=self.output, prepare_only=True, strict_input=True
        )
        with self.assertRaisesRegex(
            ValueError, "Sample at index 0: The answer must not be empty"
        ):
            bulk.generate_dataset(config)
        self.assertEqual(self.output.read_text(), "previous output\n")
        config.prepare_only = False
        config.model = "Qwen/Qwen3-4B"
        with self.assertRaisesRegex(
            ValueError, "Sample at index 0: The answer must not be empty"
        ):
            bulk.generate_dataset(config)
        self.assertTrue(self.args("--strict-input").strict_input)

    def test_malformed_json_still_fails_instead_of_silently_skipping(self):
        for malformed in ['{"question":', '{"question": "A?", "ground_truth": NaN}']:
            self.source.write_text(malformed)
            config = bulk.SFTConfig(
                input=self.source, output=self.output, prepare_only=True
            )
            with (
                self.subTest(malformed=malformed),
                self.assertRaisesRegex(ValueError, "Invalid JSON"),
            ):
                bulk.generate_dataset(config)

    def test_live_jsonl_exists_during_ingestion_and_is_readable_before_next_sample(
        self,
    ):
        self.source.write_text(
            "\n".join(
                map(
                    json.dumps,
                    [
                        {"question": "A?", "ground_truth": 0},
                        {"question": "B?", "ground_truth": 1},
                        {"question": "A?", "ground_truth": 0},
                    ],
                )
            )
        )
        config = bulk.SFTConfig(
            input=self.source, output=self.output, model="Qwen/Qwen3-4B", devices="cpu"
        )
        original_ingest = bulk.Cache.ingest
        snapshots = []

        def ingest(cache, args):
            self.assertTrue(self.output.exists())
            self.assertTrue(self.sft.exists())
            return original_ingest(cache, args)

        def initialize(config, device):
            self.assertEqual(read_jsonl(self.output), [])
            self.assertEqual(read_jsonl(self.sft), [])

        def generate(job, config, runtime):
            snapshots.append(read_jsonl(self.output))
            if job[1]["question"] == "B?":
                self.assertEqual([row["source_index"] for row in snapshots[-1]], [0, 2])
                self.assertEqual(len(read_jsonl(self.sft)), 2)
                self.assertEqual(self.output.read_bytes()[-1:], b"\n")
            return success(job[1]["question"])

        with (
            patch.object(bulk.Cache, "ingest", ingest),
            patch.object(bulk, "initialize_worker", initialize),
            patch.object(bulk, "generate_one", generate),
        ):
            summary = bulk.generate_dataset(config)
        self.assertEqual([len(snapshot) for snapshot in snapshots], [0, 2])
        self.assertEqual(summary.counts["success"], 3)
        # The final snapshot restores dataset order after appending completion order.
        self.assertEqual(
            [row["source_index"] for row in read_jsonl(self.output)], [0, 1, 2]
        )

    def test_live_append_filters_truncated_and_failed_traces_before_final_export(self):
        cache = self.cache(args=self.args("--num-traces", "3"))
        _, jobs = cache.pending()
        jobs = list(jobs)
        cache.export(self.output, self.sft, require_eos=True, ready_only=True)
        with cache.append_results(self.output, self.sft, require_eos=True) as save:
            save(jobs[0][0], success("complete"))
            self.assertEqual(len(read_jsonl(self.output)), 1)
            self.assertEqual(len(read_jsonl(self.sft)), 1)
            save(jobs[1][0], success("truncated", finish_reason="length"))
            self.assertEqual(len(read_jsonl(self.output)), 2)
            self.assertEqual(len(read_jsonl(self.sft)), 1)
            save(
                jobs[2][0],
                {"status": "error", "attempts": 1, "error": {"message": "failed"}},
            )
            self.assertEqual(
                [r["status"] for r in read_jsonl(self.output)],
                ["success", "success", "error"],
            )
            self.assertEqual(len(read_jsonl(self.sft)), 1)

    def test_resume_repairs_partial_files_and_a_committed_result_missing_from_jsonl(
        self,
    ):
        config = bulk.SFTConfig(
            input=self.source,
            output=self.output,
            model="Qwen/Qwen3-4B",
            devices="cpu",
            num_traces=2,
        )
        cache = bulk.Cache(config.cache, bulk.make_config(config))
        try:
            cache.ingest(config)
            _, jobs = cache.pending()
            first, _ = next(jobs)
            cache.export(self.output, self.sft, ready_only=True)
            # Simulate dying after the DB commit but before a complete JSONL write.
            with (
                cache.append_results(self.output, self.sft) as save,
                patch.object(
                    cache, "write_record", side_effect=OSError("write interrupted")
                ),
                self.assertRaises(OSError),
            ):
                save(first, success("committed"))
        finally:
            cache.db.close()
        self.output.write_text('{"partial":')
        self.sft.write_text('{"messages":')

        def initialize(config, device):
            self.assertEqual(
                [r["assistant_text"] for r in read_jsonl(self.output)], ["committed"]
            )
            self.assertEqual(len(read_jsonl(self.sft)), 1)

        with (
            patch.object(bulk, "initialize_worker", initialize),
            patch.object(bulk, "generate_one", side_effect=fake_generate) as generate,
        ):
            summary = bulk.generate_dataset(config)
            generate.assert_called_once()
        self.assertEqual(summary.counts["success"], 2)
        self.assertEqual(len({r["sample_id"] for r in read_jsonl(self.output)}), 2)
        self.assertEqual(len(read_jsonl(self.sft)), 2)

    def test_multiple_workers_publish_each_result_before_progress_returns(self):
        cache = self.cache(args=self.args("--num-traces", "4"))
        count, jobs = cache.pending()
        cache.export(self.output, self.sft, ready_only=True)
        observed = []
        with cache.append_results(self.output, self.sft) as append:

            def save(key, result):
                append(key, result)
                observed.append(key)
                self.assertEqual(
                    [r["sample_id"] for r in read_jsonl(self.output)], observed
                )
                self.assertEqual(len(read_jsonl(self.sft)), len(observed))

            bulk.run_jobs(
                jobs,
                count,
                {},
                ["cpu", "cpu"],
                save,
                initializer=fake_initialize,
                generate=fake_generate,
            )
        self.assertEqual(len(observed), 4)


if __name__ == "__main__":
    unittest.main()
