"""Batch ordinary decode steps while keeping each sample's repair state private.

Full-attention Qwen2 + DynamicCache only. Prefill and repair remain independent;
batching never synchronizes rollback positions between unrelated samples.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer


@torch.inference_mode()
def forward_batch(generators, token_ids):
    """Forward one token per sample, stripping temporary padding from each cache."""
    if len(generators) != len(token_ids) or not generators:
        raise ValueError("Expected one token per non-empty generator batch.")
    if len(generators) == 1:
        return [generators[0]._forward_token(token_ids[0], output_attentions=True)]
    model = generators[0].model
    lengths = [g.cache_length for g in generators]
    max_length = max(lengths)
    caches = [g.past_key_values for g in generators]
    for generator, cache in zip(generators, caches):
        if generator.model is not model:
            raise ValueError("A batch must share one model instance.")
        if type(cache) is not DynamicCache or any(type(layer) is not DynamicLayer for layer in cache.layers):
            raise ValueError("Batch decoding requires full-attention DynamicCache layers.")
        if cache.get_seq_length() != generator.cache_length:
            raise ValueError("Cannot batch a cache containing an uncommitted candidate.")

    # Existing caches stay untouched. This also makes a failed batch forward
    # recoverable without silently accepting partially mutated KV state.
    layers = []
    for layer_index in range(len(caches[0].layers)):
        tensors = []
        for attribute in ('keys', 'values'):
            tensors.append(torch.cat([
                F.pad(getattr(cache.layers[layer_index], attribute), (0, 0, 0, max_length - length))
                for cache, length in zip(caches, lengths)
            ], dim=0))
        layers.append(tuple(tensors))
    batch_cache = DynamicCache(layers)
    device = generators[0].next_logits.device
    positions = torch.tensor(lengths, device=device, dtype=torch.long)
    mask = torch.arange(max_length + 1, device=device)[None, :] < positions[:, None]
    mask[:, -1] = True
    attention_mask = mask.long()
    if getattr(model, '_v5_selective_attention', False):
        # No future keys for one query. Equal-length caches need no mask at all;
        # unequal lengths only need a compact padding mask, not a causal matrix.
        additive = None
        if min(lengths) != max_length:
            additive = torch.zeros(mask.shape, device=device, dtype=generators[0].next_logits.dtype)
            additive.masked_fill_(~mask, torch.finfo(additive.dtype).min)
            additive = additive[:, None, None, :]
        attention_mask = {'full_attention': additive}
    outputs = model(
        input_ids=torch.tensor(token_ids, device=device, dtype=torch.long)[:, None],
        attention_mask=attention_mask,
        position_ids=positions[:, None],
        cache_position=torch.tensor([max_length], device=device),
        past_key_values=batch_cache,
        use_cache=True,
        output_attentions=True,
        return_dict=True,
        **generators[0]._last_logits_kwargs(),
    )

    # Use compact owned tensors: a sample must not retain the whole batch's
    # storage, nor keep padding that would corrupt later rollback/positions.
    result = []
    for row, length in enumerate(lengths):
        def compact(tensor):
            return torch.cat((tensor[row:row + 1, :, :length, :],
                              tensor[row:row + 1, :, max_length:max_length + 1, :]), dim=-2)

        cache = DynamicCache([
            (compact(layer.keys), compact(layer.values))
            for layer in outputs.past_key_values.layers
        ])
        attentions = tuple(
            None if attention is None else torch.cat((
                attention[row:row + 1, :, :, :length],
                attention[row:row + 1, :, :, max_length:max_length + 1],
            ), dim=-1)
            for attention in outputs.attentions
        )
        result.append(SimpleNamespace(
            logits=outputs.logits[row:row + 1].clone(),
            past_key_values=cache, attentions=attentions,
        ))
    return result


class BatchScheduler:
    """Continuously refill completed slots; yield results as samples finish."""

    def __init__(self, make_generator, batch_size=8, include_unfixed=False):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.make_generator = make_generator
        self.batch_size = batch_size
        self.include_unfixed = include_unfixed
        self.stats = {'batched_forwards': 0, 'batched_samples': 0, 'repair_steps': 0}

    @torch.inference_mode()
    def tick(self, active):
        ordinary = []
        for _, generator in active:
            if generator.finished or len(generator.generated_ids) >= generator.max_new_tokens:
                continue
            if generator.repairing:
                generator.step()
                self.stats['repair_steps'] += 1
            else:
                generator.attempts += 1
                if generator.attempts > generator.max_attempts:
                    raise RuntimeError("Generation stopped after too many repair attempts.")
                ordinary.append(generator)
        if ordinary:
            # Keep each sample's RNG independent, but synchronize with the host
            # once for the whole batch instead of once per sample.
            tokens = torch.cat([g._sample_token(g.next_logits).reshape(-1) for g in ordinary]).tolist()
            outputs = forward_batch(ordinary, tokens)
            self.stats['batched_forwards'] += 1
            self.stats['batched_samples'] += len(ordinary)
            for generator, token, output in zip(ordinary, tokens, outputs):
                generator.consume_candidate(token, output)

    def run(self, samples):
        pending = iter(samples)
        active = []
        exhausted = False
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
                del generator
            if not active:
                break
            self.tick(active)
            remaining = []
            for sample, generator in active:
                if generator.finished or len(generator.generated_ids) >= generator.max_new_tokens:
                    with torch.inference_mode():
                        output = generator.finish(include_unfixed=self.include_unfixed)
                    yield {**sample, **output, 'generation_stats': {
                        'attempts': generator.attempts,
                        'repair_events': len(generator.repair_events),
                    }}
                else:
                    remaining.append((sample, generator))
            active = remaining
            del generator
