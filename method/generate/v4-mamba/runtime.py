"""Inference adapters for NVIDIA's remote Nemotron-H and native Mamba models.

NVIDIA remote revisions do not consistently forward cache/masks to attention,
or expose attention weights. The H adapter drives the existing pretrained
blocks explicitly, preserving their norm, residual, Mamba and MLP/MoE modules.
No weights, global classes, or files in the Hugging Face cache are modified.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .cache import NemotronCache
except ImportError:
    from cache import NemotronCache


class Runtime(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.wrapped = model
        self.config = model.config
        self.generation_config = getattr(model, 'generation_config', None)

    @property
    def device(self):
        return next(self.wrapped.parameters()).device

    @property
    def dtype(self):
        return next(self.wrapped.parameters()).dtype


class NemotronRuntime(Runtime):
    def __init__(self, model, attention_backend='sdpa', ssm_dtype=torch.float32):
        super().__init__(model)
        self.attention_backend = attention_backend
        self.ssm_dtype = ssm_dtype
        backbone = getattr(model, 'backbone', None)
        if backbone is None:
            backbone = getattr(model, 'model', None)
        if backbone is None or not all(hasattr(backbone, a) for a in ('layers', 'embeddings', 'norm_f')):
            raise ValueError('Unsupported Nemotron-H backbone. Expected layers, embeddings and norm_f.')
        # Avoid registering another alias of the same modules in state_dict.
        object.__setattr__(self, 'backbone', backbone)
        self.attention_layers = {}
        for index, layer in enumerate(backbone.layers):
            if not all(hasattr(layer, attr) for attr in ('block_type', 'norm', 'mixer', 'residual_in_fp32')):
                raise ValueError(f'Unsupported Nemotron-H block contract at layer {index}.')
            if layer.block_type not in ('mamba', 'attention', 'mlp', 'moe'):
                raise ValueError(f'Unsupported block type: {layer.block_type}')
            mixer = layer.mixer
            if layer.block_type == 'mamba' and not all(
                hasattr(mixer, a) for a in ('conv1d', 'num_heads', 'head_dim', 'ssm_state_size')
            ):
                raise ValueError(f'Layer {index} is not a supported Nemotron Mamba2 mixer.')
            if layer.block_type == 'attention':
                required = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'num_heads',
                            'num_key_value_heads', 'head_dim')
                if not all(hasattr(mixer, a) for a in required):
                    raise ValueError(f'Unsupported Nemotron attention projections at layer {index}.')
                if any(getattr(mixer, a, None) is not None for a in ('rotary_emb', 'q_norm', 'k_norm')):
                    raise ValueError('This adapter supports the no-RoPE Nemotron-H attention contract only.')
                if mixer.num_heads % mixer.num_key_value_heads:
                    raise ValueError('Attention heads must be divisible by KV heads.')
                self.attention_layers[index] = mixer.num_heads

    def _attention(self, mixer, hidden, cache, layer_idx, collect):
        batch, length, _ = hidden.shape
        def project(module, heads):
            return module(hidden).reshape(batch, length, heads, mixer.head_dim).transpose(1, 2)
        query = project(mixer.q_proj, mixer.num_heads)
        key = project(mixer.k_proj, mixer.num_key_value_heads)
        value = project(mixer.v_proj, mixer.num_key_value_heads)
        if cache is not None:
            key, value = cache.update(key, value, layer_idx)
        repeat = mixer.num_heads // mixer.num_key_value_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        offset = key.shape[-2] - length
        mask = None
        if length > 1 and (self.attention_backend == 'eager' or offset > 0):
            mask = torch.arange(key.shape[-2], device=hidden.device)[None, :] <= (
                torch.arange(length, device=hidden.device)[:, None] + offset
            )
        scale = mixer.head_dim ** -0.5
        if self.attention_backend == 'eager':
            scores = query @ key.transpose(-1, -2) * scale
            if mask is not None:
                scores.masked_fill_(~mask, float('-inf'))
            probabilities = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
            output = probabilities @ value
            weights = probabilities[:, :, -1:, :].clone() if collect else None
        else:
            # Prefill uses SDPA's causal fast path, avoiding a quadratic mask.
            causal = length > 1 and offset == 0
            output = F.scaled_dot_product_attention(
                query, key, value, attn_mask=None if causal else mask, is_causal=causal,
            )
            # Only the last query is used by the leakage detector.
            weights = None
            if collect:
                scores = query[:, :, -1:, :] @ key.transpose(-1, -2) * scale
                weights = scores.softmax(-1, dtype=torch.float32).to(query.dtype)
        output = output.transpose(1, 2).reshape(batch, length, mixer.num_heads * mixer.head_dim)
        return mixer.o_proj(output), weights

    def forward(self, input_ids, past_key_values=None, use_cache=True,
                output_attentions=False, cache_length=0, **kwargs):
        if self.training:
            raise RuntimeError('v4 adapters are inference-only. Call eval().')
        hidden = self.backbone.embeddings(input_ids)
        cache = past_key_values
        if not use_cache:
            cache = None
        elif cache is None:
            cache = NemotronCache(self.backbone.layers, input_ids.shape[0],
                                  hidden.device, hidden.dtype, self.ssm_dtype)
        start = cache.seen_tokens if cache is not None else 0
        if start != cache_length:
            raise ValueError('Cache length does not match the accepted token prefix.')
        if start and input_ids.shape[1] != 1:
            raise ValueError('Cached Mamba decode must consume exactly one token per sequence.')
        positions = torch.arange(start, start + input_ids.shape[1], device=hidden.device)
        attentions = {}
        for index, layer in enumerate(self.backbone.layers):
            residual = hidden.float() if layer.residual_in_fp32 else hidden
            normalized = layer.norm(hidden.to(layer.norm.weight.dtype))
            if layer.block_type == 'mamba':
                hidden = layer.mixer(normalized, cache_params=cache, cache_position=positions)
            elif layer.block_type == 'attention':
                hidden, weights = self._attention(layer.mixer, normalized, cache, index, output_attentions)
                if weights is not None:
                    attentions[index] = weights
            else:
                hidden = layer.mixer(normalized)
            hidden = residual + hidden
        hidden = self.backbone.norm_f(hidden)
        logits = self.wrapped.lm_head(hidden[:, -1:, :].to(self.wrapped.lm_head.weight.dtype)).float()
        if cache is not None:
            cache.seen_tokens = start + input_ids.shape[1]
            cache.has_previous_state = True
        return SimpleNamespace(logits=logits, past_key_values=cache,
                               attentions=attentions if output_attentions else None)


class MambaRuntime(Runtime):
    """Native transformers Mamba/Mamba2 cache_params API, normalized for v4."""

    def __init__(self, model):
        super().__init__(model)
        self.attention_layers = {}

    def forward(self, input_ids, past_key_values=None, use_cache=True,
                output_attentions=False, cache_length=0, **kwargs):
        if output_attentions:
            raise ValueError('Pure Mamba has no attention weights for the calibrated detector.')
        position = None
        if past_key_values is not None:
            if input_ids.shape[1] != 1:
                raise ValueError('Cached Mamba decode must consume one token.')
            # Mamba1's convolution update indexes the last slot after prefill.
            position = torch.tensor([max(cache_length, self.config.conv_kernel)], device=input_ids.device)
        outputs = self.wrapped(
            input_ids=input_ids, cache_params=past_key_values, cache_position=position,
            use_cache=use_cache, return_dict=True,
        )
        return SimpleNamespace(logits=outputs.logits[:, -1:, :],
                               past_key_values=outputs.cache_params, attentions=None)


def make_runtime(model, attention_backend='sdpa', ssm_dtype=torch.float32):
    kind = model.config.model_type
    if kind == 'nemotron_h':
        result = NemotronRuntime(model, attention_backend, ssm_dtype)
    elif kind in ('mamba', 'mamba2', 'falcon_mamba'):
        result = MambaRuntime(model)
    else:
        raise ValueError(f'Unsupported model_type={kind!r}. v4 supports nemotron_h, mamba, mamba2, falcon_mamba.')
    return result.eval()
