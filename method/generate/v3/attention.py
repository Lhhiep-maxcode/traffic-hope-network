"""Use SDPA except at detector layers when attention weights are requested."""

import copy

import torch
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward


def selective_attention(module, query, key, value, attention_mask, **kwargs):
    requested = kwargs.pop('output_attentions', False)
    needs_weights = requested and module._v3_detector_layer
    if needs_weights:
        if attention_mask is None and query.shape[-2] > 1:
            # SDPA mask creation may omit a pure causal mask. Eager still needs it.
            q_len, k_len = query.shape[-2], key.shape[-2]
            allowed = torch.arange(k_len, device=query.device)[None, :] <= (
                torch.arange(q_len, device=query.device)[:, None] + k_len - q_len
            )
            attention_mask = torch.zeros((q_len, k_len), device=query.device, dtype=query.dtype)
            attention_mask.masked_fill_(~allowed, torch.finfo(query.dtype).min)
        return eager_attention_forward(module, query, key, value, attention_mask, **kwargs)
    output, _ = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    # Transformers' output collector drops None values. A zero-head tensor
    # keeps layer indices stable without allocating an attention matrix.
    placeholder = query.new_empty((query.shape[0], 0, query.shape[-2], key.shape[-2])) if requested else None
    return output, placeholder


def enable_selective_attention(model, detector):
    """Qwen3 only; preserves detector weights but backend rounding can differ."""
    if model.config.model_type != 'qwen3' or getattr(model.config, 'use_sliding_window', False):
        raise ValueError("Selective attention supports full-attention Qwen3 only.")
    detector_layers = {int(head['layer']) for head in detector['heads']}
    if not detector_layers or not detector_layers <= set(range(len(model.model.layers))):
        raise ValueError("Detector layer indices are invalid for this model.")
    AttentionInterface.register('v3_selective', selective_attention)
    model._v3_selective_attention = True
    # Keep eager's mask and output collection contract at the model boundary;
    # only the layer implementations switch kernels.
    model.config._attn_implementation = 'eager'
    for index, layer in enumerate(model.model.layers):
        layer.self_attn.config = copy.copy(model.config)
        layer.self_attn.config._attn_implementation = 'v3_selective'
        layer.self_attn._v3_detector_layer = index in detector_layers
