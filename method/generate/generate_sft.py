"""Importable bulk self-coding generation with durable resume.

Use SFTConfig and generate_dataset from Python or notebooks. The CLI calls the
same API. Only generation loads torch/transformers; preparation/export are light.
"""

from __future__ import annotations

import argparse
import hashlib
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
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
INSTRUCTION = "Explain your solution step by step."


@dataclass
class SFTConfig:
    """Configuration shared by notebook imports and the optional CLI."""

    input: str | Path
    output: str | Path
    model: str | Path | None = None
    tokenizer: str | Path | None = None
    model_key: str | None = None
    detector_config: str | Path = (
        DIRECTORY.parent / "calibrate_leakage_detector/output/detector_config.json"
    )
    devices: str | tuple[str, ...] = "auto"
    dtype: str = "auto"
    revision: str = "main"
    local_files_only: bool = False
    trust_remote_code: bool = False
    torch_threads: int = 1
    sft_output: str | Path | None = None
    cache: str | Path | None = None
    cache_tag: str = ""
    question_field: str = "question"
    answer_field: str = "ground_truth"
    num_traces: int = 1
    limit: int | None = None
    seed: int = 42
    max_retries: int = 1
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True
    enable_thinking: bool = True
    reuse_kv_cache: bool = True
    fix_comparison_method: str = "attention_score"
    fix_js_divergence_threshold: float = 0.1
    max_repair_steps: int = 128
    max_repair_cycles_per_span: int = 2
    debug_clean_backtrack: bool = False
    debug_fix_infinite_loop: bool = True
    debug_wait_safe_window: bool = False
    require_eos: bool = False
    prepare_only: bool = False
    export_only: bool = False
    strict_input: bool = False

    def __post_init__(self):
        normalize_config(self)


@dataclass(frozen=True)
class GenerationSummary:
    output: Path
    sft_output: Path | None
    cache: Path | None
    counts: dict[str, int]

    @property
    def exit_code(self) -> int:
        no_valid_inputs = self.counts.get("skipped") and not (
            self.counts.get("success") or self.counts.get("prepared")
        )
        return int(
            bool(
                self.counts.get("error")
                or self.counts.get("pending")
                or no_valid_inputs
            )
        )


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


def reject_json_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def read_samples(path):
    with path.open(encoding="utf-8") as source:
        if path.suffix.lower() == ".json":
            rows = json.load(source, parse_constant=reject_json_constant)
            if not isinstance(rows, list):
                raise ValueError("A .json input must contain an array of samples.")
            yield from enumerate(rows)
        else:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield (
                        line_number - 1,
                        json.loads(line, parse_constant=reject_json_constant),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc


def read_prepared_samples(args):
    """Validate each input; preserve invalid rows for reporting, never for SFT."""
    for count, (index, sample) in enumerate(read_samples(args.input)):
        if args.limit is not None and count >= args.limit:
            break
        try:
            prepared = prepare_sample(sample, args.question_field, args.answer_field)
        except ValueError as exc:
            if args.strict_input:
                raise ValueError(f"Sample at index {index}: {exc}") from exc
            error = {"type": "InvalidSample", "message": str(exc)}
            print(
                f"Skipping sample at index {index}: {exc}", file=sys.stderr, flush=True
            )
            yield index, sample, {}, error
        else:
            yield index, sample, prepared, None


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
                sample TEXT NOT NULL, prepared TEXT NOT NULL, validation_error TEXT
            );
            CREATE INDEX temp.inputs_sample ON inputs(sample_id);
        """)
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?)", (self.run_id, dumps(config))
            )

    def ingest(self, args):
        count = 0
        with self.db:
            for index, sample, prepared, error in read_prepared_samples(args):
                for trace_index in range(args.num_traces):
                    sample_id = (
                        digest(["invalid", sample, trace_index])
                        if error
                        else digest([prepared, trace_index])
                    )
                    self.db.execute(
                        "INSERT INTO inputs VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            count * args.num_traces + trace_index,
                            index,
                            trace_index,
                            sample_id,
                            dumps(sample),
                            dumps(prepared),
                            dumps(error) if error else None,
                        ),
                    )
                count += 1
        return count * args.num_traces

    def pending(self):
        # Materialize before updating results; duplicate questions share one job.
        self.db.execute(
            """
            CREATE TEMP TABLE pending AS
            SELECT i.sample_id, i.prepared, MIN(i.position) AS position
            FROM inputs i LEFT JOIN results r
                ON r.sample_id=i.sample_id AND r.run_id=?
            WHERE i.validation_error IS NULL AND (r.status IS NULL OR r.status != 'success')
            GROUP BY i.sample_id
        """,
            (self.run_id,),
        )
        count = self.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        rows = self.db.execute(
            "SELECT sample_id, prepared FROM pending ORDER BY position"
        )
        return count, ((key, json.loads(prepared)) for key, prepared in rows)

    def save(self, sample_id, result):
        with self.db:
            self.db.execute(
                """
                INSERT INTO results VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sample_id) DO UPDATE SET
                    status=excluded.status, payload=excluded.payload,
                    attempts=results.attempts + excluded.attempts
            """,
                (
                    self.run_id,
                    sample_id,
                    result["status"],
                    dumps(result),
                    result["attempts"],
                ),
            )

    def records(self, sample_id=None):
        """Return current input occurrences, optionally just one completed job."""
        query = """
            SELECT i.source_index, i.trace_index, i.sample_id, i.sample, i.prepared,
                   r.payload, r.attempts, i.validation_error
            FROM inputs i LEFT JOIN results r
                ON r.sample_id=i.sample_id AND r.run_id=?
        """
        parameters = [self.run_id]
        if sample_id is not None:
            query += " WHERE i.sample_id=?"
            parameters.append(sample_id)
        rows = self.db.execute(query + " ORDER BY i.position", parameters)
        for (
            index,
            trace_index,
            key,
            sample,
            prepared,
            payload,
            attempts,
            validation_error,
        ) in rows:
            result = json.loads(payload) if payload else {"status": "pending"}
            if validation_error:
                result = {"status": "skipped", "error": json.loads(validation_error)}
            record = {
                **json.loads(prepared),
                **result,
                "sample_id": key,
                "run_id": self.run_id,
                "source_index": index,
                "trace_index": trace_index,
                "input_sample": json.loads(sample),
                "attempts": attempts or 0,
            }
            if record["status"] == "success":
                record["messages"] = [
                    {"role": "user", "content": record["prompt_wo_answer"]},
                    {"role": "assistant", "content": record["assistant_text"]},
                ]
            yield record

    @staticmethod
    def write_record(output, sft, record, require_eos):
        output.write(dumps(record) + "\n")
        eligible = record["status"] == "success" and (
            not require_eos or record["finish_reason"] == "eos"
        )
        if eligible:
            sft.write(dumps({"messages": record["messages"]}) + "\n")
        return eligible

    def export(self, output_path, sft_path, require_eos=False, *, ready_only=False):
        counts = {"success": 0, "error": 0, "pending": 0, "skipped": 0, "sft": 0}
        with atomic_jsonl(output_path) as output, atomic_jsonl(sft_path) as sft:
            for record in self.records():
                # Failed jobs will be retried, so don't duplicate their old rows
                # in the live snapshot. Final/export-only snapshots include all.
                if ready_only and record["status"] not in {"success", "skipped"}:
                    continue
                counts[record["status"]] += 1
                counts["sft"] += self.write_record(output, sft, record, require_eos)
        return counts

    @contextmanager
    def append_results(self, output_path, sft_path, require_eos=False):
        """One parent writer; commit to SQLite before publishing complete lines."""
        with (
            output_path.open("a", encoding="utf-8") as output,
            sft_path.open("a", encoding="utf-8") as sft,
        ):

            def save_and_append(sample_id, result):
                self.save(sample_id, result)
                for record in self.records(sample_id):
                    self.write_record(output, sft, record, require_eos)
                # Readers see each completed job before the next job is issued.
                # On a crash between commit and append, resume rebuilds from DB.
                for handle in (output, sft):
                    handle.flush()
                    os.fsync(handle.fileno())

            yield save_and_append


def initialize_worker(config, device):
    import torch

    if __package__:
        from . import self_coding as pipeline
        from .modeling import load_model
    else:  # Support executing the file directly as well as python -m.
        import self_coding as pipeline
        from modeling import load_model

    torch.set_num_threads(config["torch_threads"])
    if device.startswith("cuda:"):
        torch.cuda.set_device(torch.device(device))
    model, tokenizer = load_model(
        config["model"],
        tokenizer_path=config["tokenizer"],
        device=device,
        dtype=config["dtype"],
        revision=config["revision"],
        trust_remote_code=config["trust_remote_code"],
        local_files_only=config["local_files_only"],
    )
    return pipeline, model, tokenizer, torch, device


def generate_one(job, config, runtime):
    sample_id, prepared = job
    pipeline, model, tokenizer, torch, device = runtime
    seed = (config["seed"] + int(sample_id[:16], 16)) % (2**63 - 1)
    started = time.monotonic()
    for attempt in range(1, config["max_retries"] + 2):
        generator = None
        try:
            with torch.inference_mode():
                generator = pipeline.CustomGenerator(
                    model=model,
                    tokenizer=tokenizer,
                    detector_config_path=config["detector_config_path"],
                    model_key=config["model_key"],
                    prompt=prepared["prompt_wo_answer"],
                    privileged_context=prepared["privileged_context"],
                    seed=seed,
                    **config["generation"],
                )
                # The existing generator concatenates strings with no separator.
                # Explicitly template the exact requested privileged prompt, and
                # remap its context tokens. Its clean branch is already exact.
                generator.full_prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prepared["prompt_w_answer"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=config["generation"]["enable_thinking"],
                )
                generator.privileged_context_token_indices = (
                    pipeline._substring_token_indices(
                        tokenizer,
                        generator.full_prompt,
                        prepared["privileged_context"],
                    )
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
                    raise ValueError(
                        "The pipeline generated an empty assistant response."
                    )
                return {
                    "status": "success",
                    "assistant_text": assistant,
                    "text": result["text"],
                    "token_ids": ids,
                    "repair_events": result["repair_events"],
                    "finish_reason": "eos" if ended else "length",
                    "num_generated_tokens": len(ids),
                    "seed": seed,
                    "device": device,
                    "attempts": attempt,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }
        except Exception as exc:  # noqa: BLE001 -- persist per-sample model failures
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            # No per-sample empty_cache on success: keep the CUDA allocator warm.
            del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if attempt <= config["max_retries"]:
            time.sleep(min(2 ** (attempt - 1), 10))
    return {
        "status": "error",
        "error": error,
        "seed": seed,
        "attempts": attempt,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def worker_loop(connection, config, device, initializer, generate):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        runtime = initializer(config, device)
        connection.send({"ready": True})
        while (job := connection.recv()) is not None:
            connection.send(generate(job, config, runtime))
    except BaseException:  # noqa: BLE001 -- report worker initialization/death to parent
        try:
            connection.send({"fatal": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def run_jobs(
    jobs, count, config, devices, save, initializer=None, generate=None, *, inline=False
):
    """Show one progress bar for pending traces, updated after each cache commit."""
    if not count:
        return
    from tqdm.auto import tqdm

    counts = {"success": 0, "error": 0}
    with tqdm(
        total=count, desc="Generating traces", unit="trace", dynamic_ncols=True
    ) as progress:

        def save_and_update(sample_id, result):
            save(sample_id, result)
            counts[result["status"]] += 1
            progress.set_postfix(counts, refresh=False)
            progress.update(1)
            if result["status"] == "error":
                progress.write(
                    f"Sample {sample_id[:12]} failed: {result['error']['message']}",
                    file=sys.stderr,
                )

        _run_jobs(
            jobs,
            count,
            config,
            devices,
            save_and_update,
            initializer,
            generate,
            inline=inline,
        )


def _run_jobs(
    jobs, count, config, devices, save, initializer=None, generate=None, *, inline=False
):
    """One in-flight job per model; bounded memory and prompt result commits."""
    if not count:
        return
    initializer = initializer or initialize_worker
    generate = generate or generate_one
    if inline:
        if len(devices) != 1:
            raise ValueError("Inline generation requires exactly one device.")
        runtime = initializer(config, devices[0])
        try:
            for job in jobs:
                result = generate(job, config, runtime)
                save(job[0], result)
        except KeyboardInterrupt as exc:
            # IPython retains exception frames. Drop interrupted model frames so
            # rerunning the cell does not retain a second copy of the weights.
            raise exc.with_traceback(None)
        finally:
            del runtime
        return
    context = mp.get_context("spawn")
    workers = {}
    completed = 0
    try:
        for device in devices[:count]:
            parent, child = context.Pipe()
            process = context.Process(
                target=worker_loop, args=(child, config, device, initializer, generate)
            )
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
                    raise RuntimeError(
                        f"Worker {process.pid} exited unexpectedly; rerun to resume."
                    ) from exc
                if "fatal" in result:
                    raise RuntimeError(f"Worker failed:\n{result['fatal']}")
                if job is not None:
                    save(job[0], result)  # FULL SQLite transaction before next job.
                    completed += 1
                next_job = next(jobs, None)
                workers[connection][1] = next_job
                connection.send(next_job)
                if next_job is None:
                    active.remove(connection)
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
    parser.add_argument(
        "--input", type=Path, required=True, help="JSONL or a .json array"
    )
    parser.add_argument("--output", type=Path, required=True, help="Detailed JSONL")
    parser.add_argument(
        "--sft-output", type=Path, help="Default: OUTPUT_STEM.sft.jsonl"
    )
    parser.add_argument(
        "--cache", type=Path, help="Default: OUTPUT with .sqlite suffix"
    )
    parser.add_argument("--model")
    parser.add_argument("--tokenizer")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--cache-tag", default="", help="Change when local model weights change"
    )
    parser.add_argument(
        "--model-key", help="Key in detector_config.json; defaults to model basename"
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=DIRECTORY.parent
        / "calibrate_leakage_detector/output/detector_config.json",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help="auto (one sharded model), cuda:0,cuda:1 (replicas), cpu, or mps",
    )
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="ground_truth")
    parser.add_argument("--num-traces", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Retries per failed sample per invocation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--reuse-kv-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fix-comparison-method",
        choices=["attention_score", "js_divergence"],
        default="attention_score",
    )
    parser.add_argument("--fix-js-divergence-threshold", type=float, default=0.1)
    parser.add_argument("--max-repair-steps", type=int, default=128)
    parser.add_argument("--max-repair-cycles-per-span", type=int, default=2)
    parser.add_argument(
        "--debug-clean-backtrack", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--debug-fix-infinite-loop", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--debug-wait-safe-window", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--require-eos",
        action="store_true",
        help="Exclude truncated traces from SFT export",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write prompts without loading a model",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export current input/config from cache without loading a model",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="Stop at an invalid sample instead of skipping it and recording the reason",
    )
    try:
        return SFTConfig(**vars(parser.parse_args(argv)))
    except ValueError as exc:
        parser.error(str(exc))


def normalize_config(args: SFTConfig) -> None:
    args.input = Path(args.input).expanduser().resolve()
    args.output = Path(args.output).expanduser().resolve()
    args.sft_output = (
        Path(args.sft_output or args.output.with_name(args.output.stem + ".sft.jsonl"))
        .expanduser()
        .resolve()
    )
    args.cache = (
        Path(args.cache or args.output.with_suffix(".sqlite")).expanduser().resolve()
    )
    args.detector_config = Path(args.detector_config).expanduser().resolve()
    for name in ("model", "tokenizer"):
        value = getattr(args, name)
        if value is not None:
            candidate = Path(value).expanduser()
            setattr(
                args,
                name,
                str(candidate.resolve()) if candidate.exists() else str(value),
            )
    paths = [
        args.input,
        args.output,
        args.sft_output,
        args.cache,
        args.detector_config,
        Path(str(args.cache) + ".lock"),
        Path(str(args.cache) + "-wal"),
        Path(str(args.cache) + "-shm"),
    ]
    if len(paths) != len(set(paths)):
        raise ValueError(
            "Input, output, SFT, cache, detector and cache sidecar paths must be distinct."
        )
    if args.prepare_only and args.export_only:
        raise ValueError("prepare_only and export_only are mutually exclusive.")
    if not args.prepare_only and not args.model:
        raise ValueError(
            "model is required for generation/export (it identifies the cache)."
        )
    if args.num_traces < 1 or args.torch_threads < 1 or args.max_new_tokens < 1:
        raise ValueError(
            "num_traces, torch_threads and max_new_tokens must be positive."
        )
    if (
        args.max_retries < 0
        or args.top_k < 0
        or (args.limit is not None and args.limit < 1)
    ):
        raise ValueError(
            "max_retries/top_k must be non-negative and limit must be positive."
        )
    if (
        not 0 < args.top_p <= 1
        or not math.isfinite(args.temperature)
        or (args.do_sample and args.temperature <= 0)
    ):
        raise ValueError(
            "top_p must be in (0, 1]; sampling temperature must be finite and positive."
        )
    if args.dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be auto, float32, float16 or bfloat16.")
    if args.fix_comparison_method not in {"attention_score", "js_divergence"}:
        raise ValueError(
            "fix_comparison_method must be attention_score or js_divergence."
        )
    devices = args.devices.split(",") if isinstance(args.devices, str) else args.devices
    args.devices = tuple(device.strip() for device in devices)
    if (
        not args.devices
        or len(set(args.devices)) != len(args.devices)
        or not all(
            device in {"auto", "cpu", "mps"}
            or (device.startswith("cuda:") and device[5:].isdigit())
            for device in args.devices
        )
        or (
            len(args.devices) > 1
            and any(not d.startswith("cuda:") for d in args.devices)
        )
    ):
        raise ValueError(
            "Use one of auto/cpu/mps, or distinct CUDA devices: cuda:0,cuda:1"
        )


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
    if args.debug_fix_infinite_loop or (
        args.debug_clean_backtrack and not args.debug_wait_safe_window
    ):
        minimum = max(minimum, backtrack + 1)
    if args.max_repair_steps < minimum or args.max_repair_cycles_per_span < 1:
        raise ValueError(
            f"max-repair-steps must be >= {minimum}; max-repair-cycles-per-span >= 1."
        )
    if (
        not math.isfinite(args.fix_js_divergence_threshold)
        or args.fix_js_divergence_threshold < 0
    ):
        raise ValueError("The JSD threshold must be finite and non-negative.")
    generation_names = (
        "max_new_tokens",
        "temperature",
        "top_k",
        "top_p",
        "do_sample",
        "enable_thinking",
        "reuse_kv_cache",
        "fix_comparison_method",
        "fix_js_divergence_threshold",
        "max_repair_steps",
        "max_repair_cycles_per_span",
        "debug_clean_backtrack",
        "debug_fix_infinite_loop",
        "debug_wait_safe_window",
    )
    return {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "revision": args.revision,
        "dtype": args.dtype,
        "model_key": key,
        "detector": selected,
        "cache_tag": args.cache_tag,
        "seed": args.seed,
        "trust_remote_code": args.trust_remote_code,
        "torch_threads": args.torch_threads,
        "generation": {name: getattr(args, name) for name in generation_names},
        "code_hashes": {
            name: hashlib.sha256((DIRECTORY / name).read_bytes()).hexdigest()
            for name in ("generate_sft.py", "self_coding.py", "modeling.py")
        },
    }


def generate_dataset(config: SFTConfig) -> GenerationSummary:
    """Generate or export SFT data from Python, including notebook cells.

    A single device runs in the calling process. Multiple GPUs use imported
    worker functions with spawn. KeyboardInterrupt propagates after exporting
    committed results; rerunning with the same config resumes unfinished work.
    """
    args = config
    if args.prepare_only:
        prepared_count = 0
        skipped_count = 0
        with atomic_jsonl(args.output) as output:
            for index, sample, prepared, error in read_prepared_samples(args):
                if error:
                    output.write(
                        dumps(
                            {
                                "source_index": index,
                                "input_sample": sample,
                                "status": "skipped",
                                "error": error,
                            }
                        )
                        + "\n"
                    )
                    skipped_count += 1
                    continue
                output.write(
                    dumps(
                        {
                            **prepared,
                            "source_index": index,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": prepared["prompt_wo_answer"],
                                }
                            ],
                        }
                    )
                    + "\n"
                )
                prepared_count += 1
        print(f"Prepared prompts: {args.output}")
        return GenerationSummary(
            args.output,
            None,
            None,
            {"prepared": prepared_count, "skipped": skipped_count},
        )
    config = make_config(args)
    with cache_lock(args.cache):
        # Make output paths visible even while the input is being scanned.
        # Existing files are rebuilt from the cache after successful validation.
        for path in (args.output, args.sft_output):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        cache = Cache(args.cache, config)
        try:
            total = cache.ingest(args)
            pending_count, jobs = cache.pending()
            print(
                f"Input traces: {total}; unique pending jobs: {pending_count}; cache: {args.cache}",
                file=sys.stderr,
                flush=True,
            )
            try:
                if not args.export_only:
                    # Reconcile missing/partial JSONL lines with committed cache
                    # entries before append, also avoiding duplicates on resume.
                    cache.export(
                        args.output, args.sft_output, args.require_eos, ready_only=True
                    )
                    # Snapshot the detector used for the fingerprint so a file
                    # edit during a long run cannot silently change generation.
                    with tempfile.TemporaryDirectory(
                        prefix="self-coding-detector-"
                    ) as directory:
                        detector_path = Path(directory) / "detector.json"
                        detector_path.write_text(
                            dumps({config["model_key"]: config["detector"]}),
                            encoding="utf-8",
                        )
                        worker_config = {
                            **config,
                            "detector_config_path": str(detector_path),
                            "max_retries": args.max_retries,
                            "local_files_only": args.local_files_only,
                        }
                        with cache.append_results(
                            args.output, args.sft_output, args.require_eos
                        ) as save:
                            run_jobs(
                                jobs,
                                pending_count,
                                worker_config,
                                args.devices,
                                save,
                                inline=len(args.devices) == 1,
                            )
            except KeyboardInterrupt:
                print(
                    "Interrupted. Committed samples are safe; call generate_dataset again to resume.",
                    file=sys.stderr,
                )
                raise
            finally:
                counts = cache.export(args.output, args.sft_output, args.require_eos)
                print(
                    f"{dumps(counts)}\nDetails: {args.output}\nSFT: {args.sft_output}",
                    file=sys.stderr,
                )
            return GenerationSummary(args.output, args.sft_output, args.cache, counts)
        finally:
            cache.db.close()


def main(argv=None):
    try:
        return generate_dataset(parse_args(argv)).exit_code
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":

    def interrupt_on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
