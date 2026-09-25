"""Load the calibrated attention detector and map attention ordinals to blocks."""

import json
import math


def load_detector(args):
    key = args.model_key
    payload = json.loads(args.detector_config.read_text(encoding='utf-8'))
    detector = payload[key]
    threshold = detector.get('threshold')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Detector threshold must be a calibrated, finite, nonnegative number.')
    window = detector.get('window_size')
    if type(window) is not int or window < 1 or window % 2 != 1:
        raise ValueError('Detector window_size must be a positive odd integer.')
    if detector.get('aggregation', 'weighted') not in ('mean', 'weighted'):
        raise ValueError('Detector aggregation must be mean or weighted.')
    heads = detector['heads']
    if not heads:
        raise ValueError('An attention detector needs at least one head.')
    for head in heads:
        if any(type(head[k]) is not int or head[k] < 0 for k in ('layer', 'head')):
            raise ValueError('Detector layer/head must be nonnegative integers.')
        weight = head.get('score', 1.0)
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Detector head weights must be finite and nonnegative.')
    return key, payload


def resolve_heads(detector, attention_layers):
    """The calibration indexes the attention tuple, excluding Mamba/MLP blocks.

    Runtime tensors and caches use absolute decoder-block indices. Return a
    mapped copy so the supplied JSON and its calibration values stay intact.
    """
    blocks = sorted(attention_layers)
    if not blocks:
        raise ValueError('The calibrated attention detector requires a hybrid model with attention layers.')
    resolved = []
    for head in detector['heads']:
        layer, index = head['layer'], head['head']
        if not 0 <= layer < len(blocks) or not 0 <= index < attention_layers[blocks[layer]]:
            raise ValueError(
                f'Invalid detector head ({layer}, {index}); attention ordinals map to blocks '
                f'{blocks}, with head counts {[attention_layers[b] for b in blocks]}.'
            )
        resolved.append({**head, 'layer': blocks[layer]})
    return resolved
