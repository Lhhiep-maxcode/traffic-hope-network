"""Validate the checkpoint and calibrated detector before loading weights."""

import json
import math
from pathlib import Path


def load_detector_config(path, model_key, model_name):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if model_key not in payload:
        available = ', '.join(sorted(payload))
        raise ValueError(f'Detector key {model_key!r} not found in {path}. Available: {available}')
    # Local checkpoint directory names need not match a Hugging Face model ID.
    # For named DeepSeek checkpoints, prevent using a detector for another size.
    name = str(model_name).rstrip('/').rsplit('/', 1)[-1]
    if name.startswith('DeepSeek-R1-Distill-Qwen-') and not (
        model_key == name or model_key.startswith(name + '-')
    ):
        raise ValueError(f'Detector {model_key!r} does not match checkpoint {name!r}.')
    return payload


def validate_model_config(config):
    if config.model_type != 'qwen2':
        raise ValueError('v5-deepseek requires full-attention Qwen2 (DeepSeek-R1-Distill-Qwen).')
    layer_types = getattr(config, 'layer_types', None)
    if getattr(config, 'use_sliding_window', False) or (
        layer_types is not None and any(kind != 'full_attention' for kind in layer_types)
    ):
        raise ValueError('v5-deepseek does not support sliding-window or hybrid attention caches.')
    rope = getattr(config, 'rope_scaling', None)
    if rope and rope.get('rope_type', rope.get('type')) in {'dynamic', 'longrope'}:
        raise ValueError('Batching dynamic/longrope positional scaling is not supported.')


def validate_detector(detector, config):
    heads = detector.get('heads')
    if not isinstance(heads, list) or not heads:
        raise ValueError('Detector must contain at least one attention head.')
    for head in heads:
        if not isinstance(head, dict) or any(
            type(head.get(key)) is not int or not 0 <= head[key] < limit
            for key, limit in [('layer', config.num_hidden_layers), ('head', config.num_attention_heads)]
        ):
            raise ValueError(f'Detector layer/head indices are invalid for this model: {head!r}.')
        score = float(head.get('score', 1.0))
        if not math.isfinite(score) or score < 0:
            raise ValueError('Detector head scores must be finite and nonnegative.')
    if detector.get('aggregation', 'weighted') not in {'weighted', 'mean'}:
        raise ValueError('Detector aggregation must be weighted or mean.')
    window = detector.get('window_size')
    if type(window) is not int or window < 1:
        raise ValueError('Detector window_size must be a positive integer.')
    threshold = float(detector['threshold'])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Detector threshold must be finite and nonnegative.')
