"""Bulk self-coding generation with isolated model workers and durable resume.

See README_sft.md for commands and the output schema. Only the worker imports
torch/transformers; prompt preparation, cache export and tests use the stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import signal
import sqlite3
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from multiprocessing.connection import wait
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
INSTRUCTION = "Explain your solution step by step."
RUNTIME = None


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def prepare_sample(sample, question_field="question", answer_field="ground_truth"):
    if not isinstance(sample, dict):
        raise ValueError("Each sample must be a JSON object.")  # noqa: TRY004 -- input validation
    question = sample.get(question_field)
    answer = sample.get(answer_field)
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{question_field!r} must be a nonempty string.")
    if answer is None or isinstance(answer, (dict, list, bool)):
        raise ValueError(f"{answer_field!r} must be a string or number, including 0.")
    if isinstance(answer, float) and not math.isfinite(answer):
        raise ValueError("The answer must be finite.")
    answer = str(answer)
    if not answer.strip():
        raise ValueError("The answer must not be empty.")
    context = f"Given the ground truth answer is {answer}"
    clean = f"{question}\n\n{INSTRUCTION}"
    return {
        "question": question,
        "ground_truth": answer,
        "prompt_wo_answer": clean,
        "prompt_w_answer": f"{clean} {context}",
        "privileged_context": context,
    }


def read_samples(path):
    with path.open(encoding="utf-8") as source:
        if path.suffix.lower() == ".json":
            rows = json.load(source)
            if not isinstance(rows, list):
                raise ValueError("A .json input must contain an array of samples.")
            yield from enumerate(rows)
        else:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield line_number - 1, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc


@contextmanager
def atomic_jsonl(path):
    """An interrupted export never leaves a half-written public JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def cache_lock(path):
    # flock is released by the OS even after SIGKILL; keep the inode in place.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(path) + ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using cache {path}") from exc
        yield


class Cache:
    def __init__(self, path, config):
        self.run_id = digest(config)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, config TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS results (
                run_id TEXT NOT NULL, sample_id TEXT NOT NULL,
                status TEXT NOT NULL, payload TEXT NOT NULL, attempts INTEGER NOT NULL,
                PRIMARY KEY (run_id, sample_id)
            );
            CREATE TEMP TABLE inputs (
                position INTEGER PRIMARY KEY, source_index INTEGER NOT NULL,
                trace_index INTEGER NOT NULL, sample_id TEXT NOT NULL,
                sample TEXT NOT NULL, prepared TEXT NOT NULL
            );
            CREATE INDEX temp.inputs_sample ON inputs(sample_id);
        """)
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO runs VALUES (?, ?)",
                            (self.run_id, dumps(config)))

    def ingest(self, args):
        count = 0
        with self.db:
            for index, sample in read_samples(args.input):
                if args.limit is not None and count >= args.limit:
                    break
                try:
                    prepared = prepare_sample(sample, args.question_field, args.answer_field)
                except ValueError as exc:
                    raise ValueError(f"Sample at index {index}: {exc}") from exc
                for trace_index in range(args.num_traces):
                    sample_id = digest([prepared, trace_index])
                    self.db.execute("INSERT INTO inputs VALUES (?, ?, ?, ?, ?, ?)",
                                    (count * args.num_traces + trace_index, index,
                                     trace_index, sample_id, dumps(sample), dumps(prepared)))
                count += 1
        return count * args.num_traces

    def pending(self):
        # Materialize before updating results; duplicate questions share one job.
        self.db.execute("""
            CREATE TEMP TABLE pending AS
            SELECT i.sample_id, i.prepared, MIN(i.position) AS position
            FROM inputs i LEFT JOIN results r
                ON r.sample_id=i.sample_id AND r.run_id=?
            WHERE r.status IS NULL OR r.status != 'success'
            GROUP BY i.sample_id
        """, (self.run_id,))
        count = self.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        rows = self.db.execute("SELECT sample_id, prepared FROM pending ORDER BY position")
        return count, ((key, json.loads(prepared)) for key, prepared in rows)

    def save(self, sample_id, result):
        with self.db:
            self.db.execute("""
                INSERT INTO results VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sample_id) DO UPDATE SET
                    status=excluded.status, payload=excluded.payload,
                    attempts=results.attempts + excluded.attempts
            """, (self.run_id, sample_id, result["status"], dumps(result), result["attempts"]))

    def export(self, output_path, sft_path, require_eos=False):
        counts = {"success": 0, "error": 0, "pending": 0, "sft": 0}
        rows = self.db.execute("""
            SELECT i.source_index, i.trace_index, i.sample_id, i.sample, i.prepared,
                   r.payload, r.attempts
            FROM inputs i LEFT JOIN results r
                ON r.sample_id=i.sample_id AND r.run_id=? ORDER BY i.position
        """, (self.run_id,))
        with atomic_jsonl(output_path) as output, atomic_jsonl(sft_path) as sft:
            for index, trace_index, key, sample, prepared, payload, attempts in rows:
                result = json.loads(payload) if payload else {"status": "pending"}
                record = {
                    **json.loads(prepared), **result,
                    "sample_id": key, "run_id": self.run_id,
                    "source_index": index, "trace_index": trace_index,
                    "input_sample": json.loads(sample), "attempts": attempts or 0,
                }
                counts[record["status"]] += 1
                if record["status"] == "success":
                    record["messages"] = [
                        {"role": "user", "content": record["prompt_wo_answer"]},
                        {"role": "assistant", "content": record["assistant_text"]},
                    ]
                    if not require_eos or record["finish_reason"] == "eos":
                        sft.write(dumps({"messages": record["messages"]}) + "\n")
                        counts["sft"] += 1
                output.write(dumps(record) + "\n")
        return counts


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def initialize_worker(config, device):
    global RUNTIME
    import torch

    torch.set_num_threads(config["torch_threads"])
    if device.startswith("cuda:"):
        torch.cuda.set_device(torch.device(device))
    pipeline = load_module("bulk_self_coding", DIRECTORY / "self-coding.py")
    loader = load_module("bulk_model_loader", DIRECTORY / "generate_oop.py")
    model, tokenizer = loader.host_model(
        config["model"], tokenizer_name_or_path=config["tokenizer"],
        device_map="auto" if device == "auto" else {"": device},
        dtype=config["dtype"], attn_implementation="eager",
        revision=config["revision"], trust_remote_code=config["trust_remote_code"],
        local_files_only=config["local_files_only"],
    )
    RUNTIME = (pipeline, model, tokenizer, torch, device)


def generate_one(job, config):
    sample_id, prepared = job
    pipeline, model, tokenizer, torch, device = RUNTIME
    seed = (config["seed"] + int(sample_id[:16], 16)) % (2**63 - 1)
    started = time.monotonic()
    for attempt in range(1, config["max_retries"] + 2):
        generator = None
        try:
            with torch.inference_mode():
                generator = pipeline.CustomGenerator(
                    model=model, tokenizer=tokenizer,
                    detector_config_path=config["detector_config_path"],
                    model_key=config["model_key"],
                    prompt=prepared["prompt_wo_answer"],
                    privileged_context=prepared["privileged_context"],
                    seed=seed, **config["generation"],
                )
                # The existing generator concatenates strings with no separator.
                # Explicitly template the exact requested privileged prompt, and
                # remap its context tokens. Its clean branch is already exact.
                generator.full_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prepared["prompt_w_answer"]}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=config["generation"]["enable_thinking"],
                )
                generator.privileged_context_token_indices = pipeline._substring_token_indices(
                    tokenizer, generator.full_prompt, prepared["privileged_context"],
                )
                result = generator.generate(include_unfixed=False, show_progress=False)
                ids = list(result["token_ids"])
                ended = bool(ids and ids[-1] == tokenizer.eos_token_id)
                content_ids = ids[:-1] if ended else ids
                # Preserve <think>/</think> (skip_special_tokens=True may erase
                # them). EOS is a sequence terminator, not assistant content.
                assistant = tokenizer.decode(content_ids, skip_special_tokens=False)
                if generator.full_prompt.rstrip().endswith("<think>"):
                    assistant = "<think>\n" + assistant
                if not assistant.strip() or not content_ids:
                    raise ValueError("The pipeline generated an empty assistant response.")
                return {
                    "status": "success", "assistant_text": assistant,
                    "text": result["text"], "token_ids": ids,
                    "repair_events": result["repair_events"],
                    "finish_reason": "eos" if ended else "length",
                    "num_generated_tokens": len(ids), "seed": seed,
                    "device": device, "attempts": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
        except Exception as exc:  # noqa: BLE001 -- persist per-sample model failures
            error = {"type": type(exc).__name__, "message": str(exc),
                     "traceback": traceback.format_exc()}
        finally:
            # No per-sample empty_cache on success: keep the CUDA allocator warm.
            del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if attempt <= config["max_retries"]:
            time.sleep(min(2 ** (attempt - 1), 10))
    return {"status": "error", "error": error, "seed": seed,
            "attempts": attempt, "elapsed_seconds": round(time.monotonic() - started, 3)}


def worker_loop(connection, config, device, initializer, generate):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        initializer(config, device)
        connection.send({"ready": True})
        while (job := connection.recv()) is not None:
            connection.send(generate(job, config))
    except BaseException:  # noqa: BLE001 -- report worker initialization/death to parent
        try:
            connection.send({"fatal": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def run_jobs(jobs, count, config, devices, save,
             initializer=initialize_worker, generate=generate_one):
    """One in-flight job per model; bounded memory and prompt result commits."""
    if not count:
        return
    context = mp.get_context("spawn")
    workers = {}
    completed = 0
    last_report = time.monotonic()
    try:
        for device in devices[:count]:
            parent, child = context.Pipe()
            process = context.Process(target=worker_loop,
                                      args=(child, config, device, initializer, generate))
            process.start()
            child.close()
            workers[parent] = [process, None]
        active = set(workers)
        while active:
            for connection in wait(active, timeout=1):
                process, job = workers[connection]
                try:
                    result = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f"Worker {process.pid} exited unexpectedly; rerun to resume.") from exc
                if "fatal" in result:
                    raise RuntimeError(f"Worker failed:\n{result['fatal']}")
                if job is not None:
                    save(job[0], result)  # FULL SQLite transaction before next job.
                    completed += 1
                    if result["status"] == "error":
                        print(f"Sample {job[0][:12]} failed: {result['error']['message']}",
                              file=sys.stderr, flush=True)
                next_job = next(jobs, None)
                workers[connection][1] = next_job
                connection.send(next_job)
                if next_job is None:
                    active.remove(connection)
            if time.monotonic() - last_report >= 10 or completed == count:
                print(f"Completed {completed}/{count} unique jobs", file=sys.stderr, flush=True)
                last_report = time.monotonic()
    finally:
        for connection, (process, _) in workers.items():
            if completed < count and process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            connection.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL or a .json array")
    parser.add_argument("--output", type=Path, required=True, help="Detailed JSONL")
    parser.add_argument("--sft-output", type=Path, help="Default: OUTPUT_STEM.sft.jsonl")
    parser.add_argument("--cache", type=Path, help="Default: OUTPUT with .sqlite suffix")
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-tag", default="", help="Change when local model weights change")
    parser.add_argument("--model-key", help="Key in detector_config.json; defaults to model basename")
    parser.add_argument("--detector-config", type=Path, default=
                        DIRECTORY.parent / "calibrate_leakage_detector/output/detector_config.json")
    parser.add_argument("--devices", default="auto",
                        help="auto (one sharded model), cuda:0,cuda:1 (replicas), cpu, or mps")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="ground_truth")
    parser.add_argument("--num-traces", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per failed sample per invocation")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-kv-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fix-comparison-method", choices=["attention_score", "js_divergence"], default="attention_score")
    parser.add_argument("--fix-js-divergence-threshold", type=float, default=0.1)
    parser.add_argument("--max-repair-steps", type=int, default=128)
    parser.add_argument("--max-repair-cycles-per-span", type=int, default=2)
    parser.add_argument("--debug-clean-backtrack", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-fix-infinite-loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-wait-safe-window", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-eos", action="store_true", help="Exclude truncated traces from SFT export")
    parser.add_argument("--prepare-only", action="store_true", help="Write prompts without loading a model")
    parser.add_argument("--export-only", action="store_true", help="Export current input/config from cache without loading a model")
    args = parser.parse_args(argv)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.sft_output = (args.sft_output or args.output.with_name(args.output.stem + ".sft.jsonl")).resolve()
    args.cache = (args.cache or args.output.with_suffix(".sqlite")).resolve()
    args.detector_config = args.detector_config.resolve()
    paths = [args.input, args.output, args.sft_output, args.cache, args.detector_config,
             Path(str(args.cache) + ".lock"), Path(str(args.cache) + "-wal"),
             Path(str(args.cache) + "-shm")]
    if len(paths) != len(set(paths)):
        parser.error("Input, output, SFT, cache, detector and cache sidecar paths must be distinct.")
    if args.prepare_only and args.export_only:
        parser.error("--prepare-only and --export-only are mutually exclusive.")
    if not args.prepare_only and not args.model:
        parser.error("--model is required for generation/export (it identifies the cache).")
    if args.num_traces < 1 or args.torch_threads < 1 or args.max_new_tokens < 1:
        parser.error("num-traces, torch-threads and max-new-tokens must be positive.")
    if args.max_retries < 0 or args.top_k < 0 or (args.limit is not None and args.limit < 1):
        parser.error("max-retries/top-k must be non-negative and limit must be positive.")
    if not 0 < args.top_p <= 1 or not math.isfinite(args.temperature) or (args.do_sample and args.temperature <= 0):
        parser.error("top-p must be in (0, 1]; sampling temperature must be finite and positive.")
    args.devices = [device.strip() for device in args.devices.split(",")]
    if len(set(args.devices)) != len(args.devices) or not all(
        device in {"auto", "cpu", "mps"} or
        (device.startswith("cuda:") and device[5:].isdigit()) for device in args.devices
    ) or (len(args.devices) > 1 and any(not d.startswith("cuda:") for d in args.devices)):
        parser.error("Use one of auto/cpu/mps, or distinct CUDA devices: cuda:0,cuda:1")
    return args


def make_config(args):
    detector = json.loads(args.detector_config.read_text(encoding="utf-8"))
    key = args.model_key or args.model.rstrip("/").split("/")[-1]
    selected = detector[key]
    if not selected.get("heads"):
        raise ValueError(f"Detector {key!r} must have nonempty heads.")
    backtrack = max((int(selected["window_size"]) - 1) // 2, 0)
    minimum = 1
    if args.debug_wait_safe_window:
        minimum = max(1, backtrack) + (backtrack if args.debug_clean_backtrack else 0)
    if args.debug_fix_infinite_loop or (args.debug_clean_backtrack and not args.debug_wait_safe_window):
        minimum = max(minimum, backtrack + 1)
    if args.max_repair_steps < minimum or args.max_repair_cycles_per_span < 1:
        raise ValueError(f"max-repair-steps must be >= {minimum}; max-repair-cycles-per-span >= 1.")
    if not math.isfinite(args.fix_js_divergence_threshold) or args.fix_js_divergence_threshold < 0:
        raise ValueError("The JSD threshold must be finite and non-negative.")
    generation_names = (
        "max_new_tokens", "temperature", "top_k", "top_p", "do_sample",
        "enable_thinking", "reuse_kv_cache", "fix_comparison_method",
        "fix_js_divergence_threshold", "max_repair_steps", "max_repair_cycles_per_span",
        "debug_clean_backtrack", "debug_fix_infinite_loop", "debug_wait_safe_window",
    )
    return {
        "model": args.model, "tokenizer": args.tokenizer, "revision": args.revision,
        "dtype": args.dtype, "model_key": key, "detector": selected,
        "cache_tag": args.cache_tag, "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "torch_threads": args.torch_threads,
        "generation": {name: getattr(args, name) for name in generation_names},
        "code_hashes": {name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
                        for name in ("generate_sft.py", "self-coding.py", "generate_oop.py")},
    }


def main(argv=None):
    args = parse_args(argv)
    if args.prepare_only:
        with atomic_jsonl(args.output) as output:
            for count, (index, sample) in enumerate(read_samples(args.input)):
                if args.limit is not None and count >= args.limit:
                    break
                prepared = prepare_sample(sample, args.question_field, args.answer_field)
                output.write(dumps({**prepared, "source_index": index,
                                    "messages": [{"role": "user", "content": prepared["prompt_wo_answer"]}]}) + "\n")
        print(f"Prepared prompts: {args.output}")
        return 0
    config = make_config(args)
    with cache_lock(args.cache):
        cache = Cache(args.cache, config)
        try:
            total = cache.ingest(args)
            pending_count, jobs = cache.pending()
            print(f"Input traces: {total}; unique pending jobs: {pending_count}; cache: {args.cache}",
                  file=sys.stderr, flush=True)
            interrupted = False
            try:
                if not args.export_only:
                    # Snapshot the detector used for the fingerprint so a file
                    # edit during a long run cannot silently change generation.
                    with tempfile.TemporaryDirectory(prefix="self-coding-detector-") as directory:
                        detector_path = Path(directory) / "detector.json"
                        detector_path.write_text(dumps({config["model_key"]: config["detector"]}), encoding="utf-8")
                        worker_config = {**config, "detector_config_path": str(detector_path),
                                         "max_retries": args.max_retries,
                                         "local_files_only": args.local_files_only}
                        run_jobs(jobs, pending_count, worker_config, args.devices, cache.save)
            except KeyboardInterrupt:
                interrupted = True
                print("Interrupted. Committed samples are safe; rerun the same command to resume.", file=sys.stderr)
            finally:
                counts = cache.export(args.output, args.sft_output, args.require_eos)
                print(f"{dumps(counts)}\nDetails: {args.output}\nSFT: {args.sft_output}", file=sys.stderr)
            return 130 if interrupted else (1 if counts["error"] or counts["pending"] else 0)
        finally:
            cache.db.close()


if __name__ == "__main__":
    def interrupt_on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)