"""Measure batch throughput on the actual GPU; load weights only once."""

import gc
import json
import time
from pathlib import Path

import torch

try:
    from .batching import BatchScheduler
    from .generate import iter_samples, load_runtime, parser, preflight, resolve_devices
except ImportError:
    from batching import BatchScheduler

    from generate import iter_samples, load_runtime, parser, preflight, resolve_devices


def main():
    options = parser(output_required=False)
    options.description = __doc__
    options.set_defaults(limit=32, max_new_tokens=256, devices='0')
    options.add_argument('--batch-sizes', default='1,2,4,8,16')
    options.add_argument('--repeats', type=int, default=2)
    options.add_argument('--report', type=Path, default=Path('benchmark-v4-mamba.json'))
    args = options.parse_args()
    preflight(args)
    devices = resolve_devices(args.devices)
    if len(devices) != 1:
        options.error('Benchmark one GPU at a time with --devices 0 (or cpu).')
    sizes = sorted({int(value) for value in args.batch_sizes.split(',')} | {1})
    if sizes[0] < 1 or args.repeats < 1 or args.limit is None or args.limit < 1:
        options.error('Batch sizes, repeats, and limit must be positive.')
    samples = list(iter_samples(args.input, args.limit))
    if not samples:
        options.error('Input contains no samples.')
    device = devices[0]
    _model, factory = load_runtime(args, device)

    def synchronize():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)

    # Warm up kernels before measurement, without downloading another model.
    original_limit = args.max_new_tokens
    args.max_new_tokens = min(32, original_limit)
    list(BatchScheduler(factory, 1).run(samples[:1]))
    args.max_new_tokens = original_limit
    synchronize()
    measurements = []
    reference = None
    for size in sizes:
        for repeat in range(args.repeats):
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            scheduler = BatchScheduler(factory, size, args.include_unfixed)
            synchronize()
            started = time.perf_counter()
            try:
                rows = list(scheduler.run(iter(samples)))
                synchronize()
                elapsed = time.perf_counter() - started
                tokens = sum(len(row['token_ids']) for row in rows)
                by_id = {row['sample_index']: row['token_ids'] for row in rows}
                if reference is None:
                    reference = by_id
                measurement = {
                    'batch_size': size, 'repeat': repeat, 'seconds': elapsed,
                    'samples_per_second': len(rows) / elapsed,
                    'accepted_tokens_per_second': tokens / elapsed,
                    'accepted_tokens': tokens,
                    'repair_events': sum(len(row['repair_events']) for row in rows),
                    'samples_matching_batch1': sum(reference[i] == ids for i, ids in by_id.items()),
                    'samples': len(rows), 'scheduler': scheduler.stats,
                }
                if device.startswith('cuda'):
                    measurement['peak_allocated_gib'] = torch.cuda.max_memory_allocated(device) / 2**30
                del rows
            except torch.OutOfMemoryError:
                measurement = {'batch_size': size, 'repeat': repeat, 'error': 'out_of_memory'}
            measurements.append(measurement)
            print(json.dumps(measurement), flush=True)
            if 'error' in measurement:
                break
    valid_sizes = [size for size in sizes if len([
        row for row in measurements if row['batch_size'] == size and 'error' not in row
    ]) == args.repeats]
    best = max(valid_sizes, key=lambda size: sum(
        row['samples_per_second'] for row in measurements if row['batch_size'] == size
    )) if valid_sizes else None
    report = {
        'model': args.model, 'device': device,
        'device_name': torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU',
        'attention_backend': args.attention_backend, 'dtype': args.dtype,
        'detector_mode': args.detector_mode, 'cache_mode': args.cache_mode,
        'max_new_tokens': args.max_new_tokens, 'samples': len(samples),
        'best_measured_batch_size': best, 'measurements': measurements,
        'note': 'Best for this benchmark only. Compare output agreement and test representative long/repair-heavy samples.',
    }
    args.report.write_text(json.dumps(report, indent=2))
    print(f'Best measured batch size: {best}; report: {args.report}')


if __name__ == '__main__':
    main()
