"""Use SDPA except at detector layers when attention weights are requested."""

import copy

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

try:
    from .validation import validate_model_config, validate_detector
except ImportError:
    from validation import validate_model_config, validate_detector


def selective_attention(module, query, key, value, attention_mask, **kwargs):
    requested = kwargs.pop('output_attentions', False)
    needs_weights = requested and module._v5_detector_layer
    q_len, k_len = query.shape[-2], key.shape[-2]
    if attention_mask is None and q_len > 1 and (needs_weights or q_len != k_len):
        # Eager needs an explicit causal mask. SDPA also needs one for a cached
        # multi-token suffix: its implicit triangle aligns at the top left,
        # whereas these queries follow k_len - q_len existing prefix tokens.
        allowed = torch.arange(k_len, device=query.device)[None, :] <= (
            torch.arange(q_len, device=query.device)[:, None] + k_len - q_len
        )
        attention_mask = torch.zeros((q_len, k_len), device=query.device, dtype=query.dtype)
        attention_mask.masked_fill_(~allowed, torch.finfo(query.dtype).min)
        attention_mask = attention_mask[None, None, :, :]
    if needs_weights:
        return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
    output, _ = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    # Transformers' output collector drops None values. A zero-head tensor
    # keeps layer indices stable without allocating an attention matrix.
    placeholder = query.new_empty((query.shape[0], 0, query.shape[-2], key.shape[-2])) if requested else None
    return output, placeholder


def enable_selective_attention(model, detector):
    """Qwen2 only; preserves detector weights but backend rounding can differ."""
    validate_model_config(model.config)
    validate_detector(detector, model.config)
    detector_layers = {int(head['layer']) for head in detector['heads']}
    if not detector_layers or not detector_layers <= set(range(len(model.model.layers))):
        raise ValueError("Detector layer indices are invalid for this model.")
    AttentionInterface.register('v5_deepseek_selective', selective_attention)
    model._v5_selective_attention = True
    # Keep eager's mask and output collection contract at the model boundary;
    # only the layer implementations switch kernels.
    model.config._attn_implementation = 'eager'
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.config = copy.copy(model.config)
        layer.self_attn.config._attn_implementation = 'v5_deepseek_selective'
        layer.self_attn._v5_detector_layer = index in detector_layers
