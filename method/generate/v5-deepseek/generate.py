"""Streaming DeepSeek-R1-Distill-Qwen generation with leakage repair."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback


DEFAULT_CONFIG = Path(__file__).resolve().with_name('detector_config.json')


def parser(output_required=True):
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input', type=Path, required=True)
    result.add_argument('--output', type=Path, required=output_required)
    result.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-14B')
    result.add_argument('--revision', default=None)
    result.add_argument('--model-key', required=True,
                        help='Exact detector key, e.g. DeepSeek-R1-Distill-Qwen-14B-math.')
    result.add_argument('--detector-config', type=Path, default=DEFAULT_CONFIG)
    result.add_argument('--devices', default='auto', help='auto, cpu, or CUDA indices such as 0,1,2,3')
    result.add_argument('--batch-size', type=int, default=8, help='Concurrent samples per GPU; benchmark 1,4,8,16.')
    result.add_argument('--attention-backend', choices=['selective', 'eager'], default='selective')
    result.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    result.add_argument('--max-new-tokens', type=int, default=2048)
    result.add_argument('--temperature', type=float, default=0.1)
    result.add_argument('--top-p', type=float, default=0.9)
    result.add_argument('--top-k', type=int, default=0)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--decode-tokens', action=argparse.BooleanOptionalAction, default=False)
    result.add_argument('--include-unfixed', action=argparse.BooleanOptionalAction, default=False)
    result.add_argument('--fix-comparison-method', choices=['attention_score', 'js_divergence'], default='attention_score')
    result.add_argument('--fix-js-divergence-threshold', type=float, default=0.2)
    result.add_argument('--max-repair-steps', type=int, default=32)
    result.add_argument('--debug-clean-backtrack', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--debug-fix-infinite-loop', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--debug-wait-safe-window', action=argparse.BooleanOptionalAction, default=True)
    result.add_argument('--cpu-threads', type=int, default=2, help='Torch CPU threads per GPU worker.')
    result.add_argument('--limit', type=int, default=None, help='Only process the first N input records.')
    result.add_argument('--resume', action='store_true', help='Skip completed sample indices after verifying run metadata.')
    return result


def format_record(sample, sample_index):
    question = sample['question']
    answer = sample['ground_truth']
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f'Sample {sample_index}: question must be a non-empty string.')
    if answer is None or not str(answer).strip():
        raise ValueError(f'Sample {sample_index}: ground_truth must not be empty.')
    prompt = f'{question}\n\nExplain your solution step by step.'
    privileged = f'\n\nGiven the ground truth answer is {answer}'
    return {**sample, 'sample_index': sample_index, 'prompt_wo_answer': prompt,
            'privileged_context': privileged, 'prompt_w_answer': prompt + privileged}


def iter_samples(path, limit=None, completed=()):
    with open(path, encoding='utf-8') as source:
        for index, line in enumerate(source):
            if limit is not None and index >= limit:
                break
            if index not in completed:
                yield format_record(json.loads(line), index)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_output(args):
    """Prevent accidental duplicate appends or mixing different generation runs."""
    excluded = {'input', 'output', 'detector_config', 'devices', 'batch_size',
                'cpu_threads', 'limit', 'resume'}
    metadata = {'version': '5-deepseek', 'input_sha256': file_hash(args.input),
                'detector_sha256': file_hash(args.detector_config),
                'generation': {key: value for key, value in vars(args).items() if key not in excluded}}
    meta_path = Path(str(args.output) + '.meta.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not meta_path.exists() or json.loads(meta_path.read_text()) != metadata:
            raise ValueError('Resume metadata differs or is missing. Use a new output path.')
    elif args.output.exists() or meta_path.exists():
        raise FileExistsError('Output already exists. Use --resume or a new output path.')
    else:
        # Exclusive creation makes conflicting launchers fail instead of overwrite.
        with meta_path.open('x', encoding='utf-8') as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
    completed = set()
    if args.output.exists():
        with args.output.open('rb+') as file:
            while True:
                start = file.tell()
                line = file.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    # Only a trailing incomplete write is recoverable automatically.
                    file.truncate(start)
                    break
                record = json.loads(line)
                index = record['sample_index']
                if type(index) is not int or index < 0 or index in completed:
                    raise ValueError('Invalid/duplicate sample_index in existing output.')
                completed.add(index)
    return completed


def resolve_devices(value):
    import torch
    if value == 'cpu':
        return ['cpu']
    if value == 'auto':
        return [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']
    ids = [int(item) for item in value.split(',')]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= torch.cuda.device_count() for i in ids):
        raise ValueError('CUDA indices must be unique and visible to this process.')
    return [f'cuda:{i}' for i in ids]


def load_runtime(args, device):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    try:
        from .attention import enable_selective_attention
        from .self_coding import CustomGenerator
        from .validation import load_detector_config, validate_model_config, validate_detector
    except ImportError:
        from attention import enable_selective_attention
        from self_coding import CustomGenerator
        from validation import load_detector_config, validate_model_config, validate_detector

    if transformers.__version__ != '4.57.6':
        raise RuntimeError('v5-deepseek cache/attention integration requires transformers==4.57.6 (see requirements.txt).')
    torch.set_num_threads(args.cpu_threads)
    if device.startswith('cuda'):
        torch.cuda.set_device(device)
    detector_config = load_detector_config(args.detector_config, args.model_key, args.model)
    detector = detector_config[args.model_key]
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    validate_model_config(config)
    validate_detector(detector, config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError('DeepSeek generation requires a tokenizer with a chat template and EOS token.')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, config=config, dtype=getattr(torch, args.dtype),
        device_map={'': device}, attn_implementation='eager',
    ).eval()
    if args.attention_backend == 'selective':
        enable_selective_attention(model, detector)

    def make_generator(sample):
        return CustomGenerator(
            model=model, tokenizer=tokenizer, detector_config_path=None,
            detector_config=detector_config, model_key=args.model_key,
            max_new_tokens=args.max_new_tokens,
            fix_comparison_method=args.fix_comparison_method,
            fix_js_divergence_threshold=args.fix_js_divergence_threshold,
            max_repair_steps=args.max_repair_steps,
            prompt=sample['prompt_wo_answer'], privileged_context=sample['privileged_context'],
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            do_sample=args.do_sample, seed=(args.seed + sample['sample_index']) % (2**63),
            decode_tokens=args.decode_tokens,
            debug_clean_backtrack=args.debug_clean_backtrack,
            debug_fix_infinite_loop=args.debug_fix_infinite_loop,
            debug_wait_safe_window=args.debug_wait_safe_window,
        )
    return model, make_generator


def worker(rank, device, args, tasks, responses):
    try:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        import torch
        try:
            from .batching import BatchScheduler
        except ImportError:
            from batching import BatchScheduler
        model, factory = load_runtime(args, device)
        responses.put(('ready', rank, {'device': device}))
        scheduler = BatchScheduler(factory, args.batch_size, args.include_unfixed)
        for record in scheduler.run(iter(tasks.get, None)):
            responses.put(('result', rank, record))
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        responses.put(('finished', rank, scheduler.stats))
    except BaseException:
        responses.put(('error', rank, traceback.format_exc()))
        raise


def run_workers(args, devices, samples, output_file):
    context = mp.get_context('spawn')
    capacity = max(2, len(devices) * args.batch_size * 2)
    tasks, responses = context.Queue(capacity), context.Queue(capacity)
    processes = [context.Process(target=worker, args=(rank, device, args, tasks, responses))
                 for rank, device in enumerate(devices)]
    source = iter(samples)
    pending = None
    exhausted = False
    sentinels = 0
    finished = set()
    count = tokens = 0
    started = time.perf_counter()
    last_log = started
    all_stats = {}
    try:
        for process in processes:
            process.start()
        while len(finished) < len(processes):
            # Bound both the work queue and results queue. Never block feeding
            # work while workers might be blocked trying to return results.
            while sentinels < len(processes):
                if pending is None and not exhausted:
                    try:
                        pending = next(source)
                    except StopIteration:
                        exhausted = True
                try:
                    tasks.put_nowait(pending)
                except queue.Full:
                    break
                if exhausted:
                    sentinels += 1
                pending = None
            try:
                kind, rank, payload = responses.get(timeout=0.2)
            except queue.Empty:
                for rank, process in enumerate(processes):
                    if process.exitcode is not None and rank not in finished:
                        raise RuntimeError(f'Worker {rank} exited unexpectedly ({process.exitcode}). Resume after fixing the error.')
                continue
            if kind == 'result':
                output_file.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
                output_file.flush()
                count += 1
                tokens += len(payload['token_ids'])
            elif kind == 'ready':
                print(f"Worker {rank} ready on {payload['device']}", file=sys.stderr, flush=True)
            elif kind == 'finished':
                finished.add(rank)
                all_stats[str(rank)] = payload
            elif kind == 'error':
                raise RuntimeError(f'Worker {rank} failed:\n{payload}')
            now = time.perf_counter()
            if now - last_log >= 10:
                print(f'{count} samples | {tokens / (now - started):.1f} accepted tokens/s (including startup)', file=sys.stderr, flush=True)
                last_log = now
        return {'samples': count, 'accepted_tokens': tokens,
                'seconds_including_startup': time.perf_counter() - started, 'workers': all_stats}
    finally:
        for process in processes:
            if process.pid is not None:
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        for channel in (tasks, responses):
            channel.cancel_join_thread()
            channel.close()


def main():
    args = parser().parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1 or args.max_new_tokens < 0 or (args.limit is not None and args.limit < 0):
        raise ValueError('Invalid batch size, CPU thread count, or token/sample limit.')
    if args.input.resolve() == args.output.resolve():
        raise ValueError('Input and output must be different files.')
    devices = resolve_devices(args.devices)
    # The launcher is the only output writer, including across separate runs.
    import fcntl
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.output) + '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another launcher is already writing this output.') from error
        execute(args, devices)


def execute(args, devices):
    completed = prepare_output(args)
    print(f'{len(devices)} workers | batch {args.batch_size}/worker | {len(completed)} samples already saved', file=sys.stderr)
    samples = iter_samples(args.input, args.limit, completed)
    try:
        first = next(samples)
    except StopIteration:
        print('No pending samples.')
        return
    import itertools
    with args.output.open('a', encoding='utf-8') as output:
        stats = run_workers(args, devices, itertools.chain([first], samples), output)
    Path(str(args.output) + '.stats.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
