"""Compare cached decode, equal-length batching and rollback with full-prefill logits."""

import json

import torch
from batching import forward_batch

from generate import iter_samples, load_runtime, parser, preflight, resolve_devices


@torch.inference_mode()
def main():
    options = parser(output_required=False)
    options.description = __doc__
    options.set_defaults(devices='0', do_sample=False)
    options.add_argument('--check-steps', type=int, default=4)
    options.add_argument('--atol', type=float, default=0.05)
    options.add_argument('--rtol', type=float, default=0.05)
    args = options.parse_args()
    if args.check_steps < 1 or args.atol < 0 or args.rtol < 0:
        options.error('Check steps must be positive and tolerances nonnegative.')
    preflight(args)
    devices = resolve_devices(args.devices)
    if len(devices) != 1:
        options.error('Choose one device, e.g. --devices 0.')
    sample = next(iter_samples(args.input, limit=1), None)
    if sample is None:
        options.error('Input is empty.')
    _model, factory = load_runtime(args, devices[0])
    args.cache_mode = 'cached'
    cached = factory(sample)
    args.cache_mode = 'recompute'
    reference = factory(sample)
    args.cache_mode = 'cached'
    cached.begin()
    reference.begin()
    errors = []
    def compare(actual, expected):
        errors.append(float((actual - expected).abs().max()))
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise RuntimeError('Non-finite model logits.')
        torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
    compare(cached.next_logits, reference.next_logits)
    for _ in range(args.check_steps):
        token = cached.next_logits.argmax(-1).item()
        a = cached._forward_token(token, output_attentions=True)
        b = reference._forward_token(token, output_attentions=True)
        compare(a.logits, b.logits)
        for layer in a.attentions:
            compare(a.attentions[layer], b.attentions[layer])
        cached._accept(token, a)
        reference._accept(token, b)
    prefix = cached.generated_ids[:max(0, len(cached.generated_ids) - 2)]
    cached._forward_token(0)  # Reject a candidate that has mutated recurrent state.
    cached._restore_prefix(prefix)
    reference.start(prefix)
    compare(cached.next_logits, reference.next_logits)
    other = factory(sample)
    other.start(prefix)
    tokens = [cached.next_logits.argmax(-1).item(), other.next_logits.argmax(-1).item()]
    outputs = forward_batch([cached, other], tokens)
    for g, token, output in zip((cached, other), tokens, outputs):
        compare(output.logits, g._forward_token(token, output_attentions=True).logits)
    print(json.dumps({'status': 'passed', 'model': args.model, 'device': devices[0],
                      'dtype': args.dtype, 'checks': len(errors), 'max_abs_error': max(errors),
                      'atol': args.atol, 'rtol': args.rtol,
                      'note': 'Numerical smoke test only; does not validate detector calibration.'}, indent=2))


if __name__ == '__main__':
    main()
