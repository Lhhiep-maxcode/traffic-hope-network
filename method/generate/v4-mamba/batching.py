"""Continuous slots with equal-length decode batches; never pad recurrent state."""

from collections import defaultdict
from types import SimpleNamespace

import torch

try:
    from .cache import merge_caches, split_cache
except ImportError:
    from cache import merge_caches, split_cache


@torch.inference_mode()
def forward_batch(generators, token_ids, output_attentions=True):
    if not generators or len(generators) != len(token_ids):
        raise ValueError('Expected one token per generator.')
    first = generators[0]
    if len(generators) == 1:
        return [first._forward_token(token_ids[0], output_attentions)]
    for g in generators:
        if (g.model is not first.model or g.cache_length != first.cache_length
                or g.cache_mode != 'cached' or g._cache_dirty):
            raise ValueError('Batched decode requires equal lengths and clean, compatible cached states.')
    merged = merge_caches([g.past_key_values for g in generators])
    outputs = first.model(
        input_ids=torch.tensor(token_ids, device=first.next_logits.device)[:, None],
        past_key_values=merged, cache_length=first.cache_length,
        use_cache=True, output_attentions=output_attentions,
    )
    states = split_cache(outputs.past_key_values, len(generators))
    return [SimpleNamespace(
        logits=outputs.logits[row:row + 1].clone(), past_key_values=state,
        attentions={layer: value[row:row + 1].clone() for layer, value in outputs.attentions.items()}
        if outputs.attentions is not None else None,
    ) for row, state in enumerate(states)]


class BatchScheduler:
    def __init__(self, make_generator, batch_size=4, include_unfixed=False):
        if batch_size < 1:
            raise ValueError('batch_size must be positive.')
        self.make_generator = make_generator
        self.batch_size = batch_size
        self.include_unfixed = include_unfixed
        self.stats = {'decode_forwards': 0, 'batched_forwards': 0,
                      'batched_samples': 0, 'scalar_forwards': 0, 'repair_steps': 0}

    @torch.inference_mode()
    def tick(self, active):
        groups = defaultdict(list)
        for _, generator in active:
            if generator.finished or len(generator.generated_ids) >= generator.max_new_tokens:
                continue
            if generator.repairing:
                generator.step()
                self.stats['repair_steps'] += 1
            else:
                generator.attempts += 1
                if generator.attempts > generator.max_attempts:
                    raise RuntimeError('Generation stopped after too many repair attempts.')
                key = (id(generator.model), generator.cache_length)
                if generator.cache_mode == 'recompute':
                    key += (id(generator),)
                groups[key].append(generator)
        for group in groups.values():
            tokens = torch.cat([g._sample_token(g.next_logits).reshape(-1) for g in group]).tolist()
            outputs = forward_batch(group, tokens)
            self.stats['decode_forwards'] += 1
            if len(group) > 1:
                self.stats['batched_forwards'] += 1
                self.stats['batched_samples'] += len(group)
            else:
                self.stats['scalar_forwards'] += 1
            for generator, token, output in zip(group, tokens, outputs):
                generator.consume_candidate(token, output)

    def run(self, samples):
        pending, active, exhausted = iter(samples), [], False
        while active or not exhausted:
            while len(active) < self.batch_size and not exhausted:
                try:
                    sample = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                generator = self.make_generator(sample)
                with torch.inference_mode():
                    generator.begin()
                active.append((sample, generator))
            if not active:
                break
            self.tick(active)
            remaining = []
            for sample, generator in active:
                if generator.finished or len(generator.generated_ids) >= generator.max_new_tokens:
                    with torch.inference_mode():
                        output = generator.finish(include_unfixed=self.include_unfixed)
                    yield {**sample, **output, 'generation_stats': {
                        'attempts': generator.attempts, 'repair_events': len(generator.repair_events),
                        'detector_mode': 'attention', 'cache_mode': generator.cache_mode,
                    }}
                else:
                    remaining.append((sample, generator))
            active = remaining
