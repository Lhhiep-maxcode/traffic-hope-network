"""Owned recurrent state; merge only equally long, unpadded prefixes."""

import copy

import torch


class StateList(list):
    # NVIDIA's reference CPU path also reads states.device on this list.
    @property
    def device(self):
        return self[0].device


class NemotronCache:
    """Mamba2 convolution/SSM state and attention KV indexed by decoder block.

    There is deliberately no crop(): recurrent state cannot be truncated to
    undo a token. Rollbacks rebuild from the accepted token prefix.
    """

    def __init__(self, layers, batch_size, device, dtype, ssm_dtype):
        self.seen_tokens = 0
        self.has_previous_state = False
        self.key_cache = []
        self.value_cache = []
        self.conv_states = StateList()
        self.ssm_states = StateList()
        self.conv_kernel_size = None
        for layer in layers:
            empty = lambda: torch.empty(batch_size, 0, device=device, dtype=dtype)
            self.key_cache.append(empty())
            self.value_cache.append(empty())
            if layer.block_type == 'mamba':
                mixer = layer.mixer
                kernel = mixer.conv1d.kernel_size[0]
                if self.conv_kernel_size not in (None, kernel):
                    raise ValueError('Mixed Mamba convolution kernel sizes are unsupported.')
                self.conv_kernel_size = kernel
                self.conv_states.append(torch.zeros(
                    batch_size, mixer.conv1d.in_channels, kernel, device=device, dtype=dtype,
                ))
                self.ssm_states.append(torch.zeros(
                    batch_size, mixer.num_heads, mixer.head_dim, mixer.ssm_state_size,
                    device=device, dtype=ssm_dtype,
                ))
            else:
                self.conv_states.append(empty())
                self.ssm_states.append(empty())

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx].numel():
            key = torch.cat((self.key_cache[layer_idx], key), dim=2)
            value = torch.cat((self.value_cache[layer_idx], value), dim=2)
        self.key_cache[layer_idx], self.value_cache[layer_idx] = key, value
        return key, value

    def update_conv_state(self, layer_idx, new_conv_state, cache_init=False):
        old = self.conv_states[layer_idx]
        if cache_init:
            old.copy_(new_conv_state.to(old))
        else:
            old.copy_(old.roll(-1, dims=-1))
            old[:, :, -1].copy_(new_conv_state[:, 0, :].to(old))
        return old

    def update_ssm_state(self, layer_idx, new_ssm_state):
        old = self.ssm_states[layer_idx]
        old.copy_(new_ssm_state.to(old))
        return old

    def get_seq_length(self, layer_idx=0):
        return self.seen_tokens


def _fields(cache):
    if isinstance(cache, NemotronCache):
        return ('key_cache', 'value_cache', 'conv_states', 'ssm_states')
    # Pin these contracts to transformers 4.57.6, not arbitrary cache objects.
    if (type(cache).__module__.startswith('transformers.models.')
            and type(cache).__name__ in ('MambaCache', 'Mamba2Cache', 'FalconMambaCache')):
        return ('conv_states', 'ssm_states')
    raise TypeError(f'Unsupported recurrent cache: {type(cache).__name__}')


def merge_caches(caches):
    if not caches or any(type(c) is not type(caches[0]) for c in caches):
        raise ValueError('A batch must contain the same cache type.')
    if isinstance(caches[0], NemotronCache) and len({c.seen_tokens for c in caches}) != 1:
        raise ValueError('Hybrid caches must have equal prefix lengths; padding is not safe for SSM state.')
    merged = copy.copy(caches[0])
    for name in _fields(merged):
        values = [getattr(c, name) for c in caches]
        if isinstance(values[0], torch.Tensor):
            # Native Mamba2 stores [layer, batch, ...].
            value = torch.cat(values, dim=1)
        else:
            value = type(values[0])([
                torch.cat([v[i] for v in values], dim=0) for i in range(len(values[0]))
            ])
        setattr(merged, name, value)
    if hasattr(merged, 'max_batch_size'):
        merged.max_batch_size = sum(c.max_batch_size for c in caches)
    return merged


def split_cache(cache, batch_size):
    results = []
    for row in range(batch_size):
        result = copy.copy(cache)
        for name in _fields(cache):
            value = getattr(cache, name)
            if isinstance(value, torch.Tensor):
                part = value[:, row:row + 1].clone()
            else:
                part = type(value)([v[row:row + 1].clone() for v in value])
            setattr(result, name, part)
        if hasattr(result, 'max_batch_size'):
            result.max_batch_size = 1
        results.append(result)
    return results
