from __future__ import annotations

import inspect
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

LayerHead = tuple[int, int] | dict[str, Any]


def _resolve_torch_dtype(dtype: str | torch.dtype | None):
    if dtype is None or dtype == "auto":
        return "auto"
    if not isinstance(dtype, str):
        return dtype

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_by_name:
        choices = ", ".join(["auto", *dtype_by_name])
        raise ValueError(f"Unsupported dtype {dtype!r}. Choose one of: {choices}.")
    return dtype_by_name[dtype]


def host_model(
    model_name_or_path: str | Path,
    *,
    tokenizer_name_or_path: str | Path | None = None,
    device_map: str | dict[str, Any] | int | None = "auto",
    dtype: str | torch.dtype | None = "auto",
    attn_implementation: str = "eager",
    trust_remote_code: bool = False,
    revision: str = "main",
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    model_kwargs: dict[str, Any] | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
):
    """Load and keep a causal language model in the current process.

    ``model_name_or_path`` can be a Hugging Face model id or a local model
    directory. The default attention implementation is ``eager`` because this
    generation pipeline requires ``output_attentions=True``.

    Returns:
        ``(model, tokenizer)`` ready to pass to :func:`generate_main`.
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
        from transformers import (
            __version__ as transformers_version,
        )
    except ImportError as exc:
        raise RuntimeError(
            "host_model requires Transformers. Install it with "
            "`pip install transformers accelerate`."
        ) from exc

    model_source = str(model_name_or_path)
    tokenizer_source = str(tokenizer_name_or_path or model_name_or_path)
    shared_kwargs = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
        "token": token,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "local_files_only": local_files_only,
    }
    shared_kwargs = {
        key: value for key, value in shared_kwargs.items() if value is not None
    }

    final_tokenizer_kwargs = {**shared_kwargs, **(tokenizer_kwargs or {})}
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        **final_tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    final_model_kwargs = {**shared_kwargs, **(model_kwargs or {})}
    final_model_kwargs.setdefault("attn_implementation", attn_implementation)
    if device_map is not None:
        final_model_kwargs.setdefault("device_map", device_map)

    # Transformers 5 renamed the public loading argument from torch_dtype to
    # dtype. Select it by major version to keep this draft usable with both.
    try:
        transformers_major = int(transformers_version.split(".", 1)[0])
    except (TypeError, ValueError):
        transformers_major = 4
    dtype_argument = "dtype" if transformers_major >= 5 else "torch_dtype"
    if "dtype" not in final_model_kwargs and "torch_dtype" not in final_model_kwargs:
        final_model_kwargs[dtype_argument] = _resolve_torch_dtype(dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        **final_model_kwargs,
    )
    model.eval()
    if (
        hasattr(model, "generation_config")
        and model.generation_config.pad_token_id is None
        and tokenizer.pad_token_id is not None
    ):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def _model_device(model):
    """Return the device of the model, whether it's a single device or a module on multiple devices."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _input_ids(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"][0].detach().cpu()


def _eos_token_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(token_id) for token_id in eos}


def _select_detector(payload: Any, model_key: str | None) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"heads": payload}
    if not isinstance(payload, dict):
        raise TypeError("Detector data must be a JSON object or list.")
    if "heads" in payload:
        return payload
    if model_key is not None:
        if model_key not in payload:
            raise KeyError(f"No detector config for {model_key!r}.")
        detector = payload[model_key]
    elif len(payload) == 1:
        detector = next(iter(payload.values()))
    else:
        keys = ", ".join(sorted(map(str, payload)))
        raise ValueError(
            "Detector file contains more than one model; pass model_key. "
            f"Available keys: {keys}"
        )
    if not isinstance(detector, dict) or "heads" not in detector:
        raise ValueError("The selected detector config does not contain 'heads'.")
    return detector


def _load_detector(
    data_path: str | Path, model_key: str | None = None
) -> dict[str, Any]:
    path = Path(data_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Detector data does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Detector data is not valid JSON: {path}") from exc
    return _select_detector(payload, model_key)


def _normalise_layer_heads(
    layer_head_list: Sequence[LayerHead],
) -> list[tuple[int, int, float]]:
    normalised: list[tuple[int, int, float]] = []
    for item in layer_head_list:
        if isinstance(item, dict):
            if "layer" not in item or "head" not in item:
                raise ValueError("Each head object must contain 'layer' and 'head'.")
            layer = int(item["layer"])
            head = int(item["head"])
            weight = max(float(item.get("score", 1.0)), 0.0)
        else:
            if len(item) != 2:
                raise ValueError(
                    "Each layer/head item must contain exactly two integers."
                )
            layer, head = map(int, item)
            weight = 1.0
        if layer < 0 or head < 0:
            raise ValueError("Layer and head indices must be non-negative.")
        normalised.append((layer, head, weight))

    if not normalised:
        raise ValueError("layer_head_list cannot be empty.")
    if not any(weight > 0 for _, _, weight in normalised):
        normalised = [(layer, head, 1.0) for layer, head, _ in normalised]
    return normalised


def _forward_accepts_argument(model, name: str) -> bool:
    """Return whether the model's forward method declares ``name``."""
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _forward_sequence(
    model,
    input_ids: torch.Tensor,
    *,
    past_key_values=None,
    past_length: int = 0,
    use_cache: bool,
    output_attentions: bool,
):
    """Forward a full sequence or only uncached tokens.

    With ``past_key_values``, ``input_ids`` must contain only tokens that are not
    in the cache. The attention mask still covers cached and new tokens.
    """
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("input_ids must be a non-empty one-dimensional tensor.")
    if past_key_values is None and past_length != 0:
        raise ValueError("past_length must be zero when no cache is supplied.")

    device = _model_device(model)
    batched_ids = input_ids.unsqueeze(0).to(device)
    total_length = past_length + input_ids.numel()
    attention_mask = torch.ones(
        (1, total_length),
        dtype=torch.long,
        device=device,
    )
    kwargs = {
        "input_ids": batched_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "output_attentions": output_attentions,
        "return_dict": True,
    }
    if _forward_accepts_argument(model, "cache_position"):
        kwargs["cache_position"] = torch.arange(
            past_length,
            total_length,
            dtype=torch.long,
            device=device,
        )
    if _forward_accepts_argument(model, "logits_to_keep"):
        kwargs["logits_to_keep"] = 1

    with torch.inference_mode():
        outputs = model(**kwargs)

    if outputs.logits is None:
        raise RuntimeError("The model did not return logits.")
    if use_cache and outputs.past_key_values is None:
        raise RuntimeError("The model did not return past_key_values.")
    if output_attentions and outputs.attentions is None:
        raise RuntimeError(
            "The model did not return attentions. Load it with "
            "attn_implementation='eager'."
        )

    # Clone the last row so a view does not keep the full prefill logits storage
    # alive on models that return logits for every input position.
    next_logits = outputs.logits[0, -1].detach().clone()
    cache = outputs.past_key_values if use_cache else None
    attentions = outputs.attentions if output_attentions else None
    del outputs
    return next_logits, cache, attentions


def _sample_next_token_id(
    model,
    logits: torch.Tensor,
    generation_kwargs: dict[str, Any] | None,
) -> int:
    """Select one token from logits using common Transformers settings."""
    settings = dict(generation_kwargs or {})
    supported = {"do_sample", "temperature", "top_k", "top_p", "generator"}
    ignored = {
        "cache_implementation",
        "eos_token_id",
        "max_new_tokens",
        "pad_token_id",
        "use_cache",
    }
    unsupported = sorted(set(settings) - supported - ignored)
    if unsupported:
        raise ValueError(
            "The optimized loop does not yet support generation arguments: "
            + ", ".join(unsupported)
        )

    config = getattr(model, "generation_config", None)

    def setting(name: str, default):
        if name in settings:
            return settings[name]
        return getattr(config, name, default) if config is not None else default

    do_sample = bool(setting("do_sample", False))
    scores = logits.detach().float()
    if scores.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector.")
    if not do_sample:
        return int(torch.argmax(scores).item())

    temperature = float(setting("temperature", 1.0))
    top_k = int(setting("top_k", 0) or 0)
    top_p = float(setting("top_p", 1.0))
    if temperature <= 0:
        raise ValueError("temperature must be positive when do_sample=True.")
    if top_k < 0:
        raise ValueError("top_k must be non-negative.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")

    scores = scores / temperature
    if top_k > 0:
        top_k = min(top_k, scores.numel())
        cutoff = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)

    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probabilities = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative_probabilities > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=settings.get("generator"),
        )
        return int(sorted_indices[sampled_index].item())

    probabilities = torch.softmax(scores, dim=-1)
    return int(
        torch.multinomial(
            probabilities,
            num_samples=1,
            generator=settings.get("generator"),
        ).item()
    )


def _copy_generation_kwargs_for_baseline(
    generation_kwargs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Copy sampling settings and snapshot an explicit torch.Generator."""
    if generation_kwargs is None:
        return None
    copied = dict(generation_kwargs)
    generator = copied.get("generator")
    if generator is None:
        return copied
    try:
        baseline_generator = torch.Generator(device=generator.device)
        baseline_generator.set_state(generator.get_state())
    except (AttributeError, RuntimeError, TypeError):
        # Custom generators cannot always be cloned; retain their normal
        # sequential sampling behavior rather than rejecting valid settings.
        return copied
    copied["generator"] = baseline_generator
    return copied


def _aggregate_current_to_privileged_attention(
    attentions,
    layer_head_list: Sequence[LayerHead],
    privileged_token_indices: Sequence[int],
    aggregation: str = "weighted",
) -> torch.Tensor:
    """Aggregate current-query attention only over privileged key tokens."""
    if attentions is None:
        raise ValueError("attentions are required.")
    if aggregation not in {"weighted", "mean"}:
        raise ValueError("aggregation must be either 'weighted' or 'mean'.")

    heads = _normalise_layer_heads(layer_head_list)
    if aggregation == "mean":
        heads = [(layer, head, 1.0) for layer, head, _ in heads]

    aggregate = torch.zeros(len(privileged_token_indices), dtype=torch.float32)
    total_weight = 0.0
    for layer, head, weight in heads:
        if layer >= len(attentions):
            raise IndexError(
                f"Layer {layer} is out of range for {len(attentions)} layers."
            )
        layer_attention = attentions[layer]
        if head >= layer_attention.shape[1]:
            raise IndexError(
                f"Head {head} is out of range for layer {layer}, which has "
                f"{layer_attention.shape[1]} heads."
            )
        key_indices = torch.tensor(
            privileged_token_indices,
            dtype=torch.long,
            device=layer_attention.device,
        )
        selected = layer_attention[0, head, -1].index_select(0, key_indices)
        aggregate += selected.detach().float().cpu() * weight
        total_weight += weight

    return aggregate / total_weight


def _normalise_indices(indices: Sequence[int], size: int, name: str) -> list[int]:
    normalised = []
    for index in indices:
        index = int(index)
        index = size + index if index < 0 else index
        if index < 0 or index >= size:
            raise IndexError(f"{name} index {index} is outside 0..{size - 1}.")
        normalised.append(index)
    if not normalised:
        raise ValueError(f"{name} indices cannot be empty.")
    return normalised


def get_token_score_from_attention_matrix(
    attention_matrix,
    token_index,
    key_token_indices: Sequence[int] | None = None,
) -> float:
    """
    Get the attention score for one query token.

    When ``key_token_indices`` is supplied, the score is the total attention
    paid by ``token_index`` to those key tokens (the privileged-context score
    used by ``generate_main``). Without it, the largest value in the row is
    returned as a generic fallback; summing the full row would be approximately
    one and would not be useful for detection.
    """
    matrix = torch.as_tensor(attention_matrix)
    if matrix.ndim == 1:
        if int(token_index) not in {0, -1}:
            raise IndexError("A one-dimensional attention row only has token index 0.")
        row = matrix
    elif matrix.ndim == 2:
        query_index = _normalise_indices([token_index], matrix.shape[0], "Token")[0]
        row = matrix[query_index]
    else:
        raise ValueError(
            "attention_matrix must be a one-dimensional row or a "
            "two-dimensional query-by-key tensor."
        )
    if key_token_indices is None:
        return float(row.max())
    keys = _normalise_indices(key_token_indices, row.shape[0], "Key token")
    return float(row[keys].sum())


def get_jensen_shannon_divergence(
    privileged_logits: torch.Tensor,
    clean_logits: torch.Tensor,
) -> float:
    """Return Jensen-Shannon divergence between two next-token distributions.

    The inputs are vocabulary logits produced at the same generation position.
    Natural logarithms are used, so the result is in ``[0, ln(2)]``. Computing
    JSD between only the two sampled token IDs would reduce the signal to zero
    for equal IDs and ``ln(2)`` for unequal IDs, so the full distributions are
    compared instead.
    """
    privileged_logits = torch.as_tensor(privileged_logits).detach().float()
    clean_logits = torch.as_tensor(clean_logits).detach().float()
    if privileged_logits.ndim != 1 or clean_logits.ndim != 1:
        raise ValueError("Jensen-Shannon inputs must be one-dimensional logits.")
    if privileged_logits.shape != clean_logits.shape:
        raise ValueError(
            "Jensen-Shannon inputs must use the same tokenizer vocabulary."
        )

    privileged_log_prob = torch.log_softmax(privileged_logits, dim=-1)
    clean_log_prob = torch.log_softmax(clean_logits, dim=-1)
    mixture_log_prob = torch.logaddexp(privileged_log_prob, clean_log_prob) - math.log(
        2.0
    )
    divergence = 0.5 * (
        (privileged_log_prob.exp() * (privileged_log_prob - mixture_log_prob)).sum()
        + (clean_log_prob.exp() * (clean_log_prob - mixture_log_prob)).sum()
    )
    # Floating-point roundoff can produce a tiny negative value near zero.
    return max(float(divergence.item()), 0.0)


def _prefill_privileged_state(
    model,
    privileged_prompt_ids: torch.Tensor,
    generated_ids: Sequence[int],
):
    generated = torch.tensor(generated_ids, dtype=torch.long)
    full_prefix = torch.cat([privileged_prompt_ids, generated])
    logits, cache, _ = _forward_sequence(
        model,
        full_prefix,
        use_cache=True,
        output_attentions=False,
    )
    return logits, cache, full_prefix.numel()


def _generate_without_repair(
    model,
    tokenizer,
    privileged_prompt_ids: torch.Tensor,
    *,
    max_length: int,
    generation_kwargs: dict[str, Any] | None,
    show_progress: bool,
) -> list[int]:
    """Generate the privileged-context baseline without detection or repair."""
    if max_length == 0:
        return []

    generated_ids: list[int] = []
    eos_token_ids = _eos_token_ids(tokenizer)
    logits, cache, cache_length = _prefill_privileged_state(
        model,
        privileged_prompt_ids,
        generated_ids,
    )
    progress_bar = tqdm(
        total=max_length,
        desc="Generating unfixed",
        unit="token",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    while len(generated_ids) < max_length:
        token_id = _sample_next_token_id(model, logits, generation_kwargs)
        generated_ids.append(token_id)
        progress_bar.update(1)
        if token_id in eos_token_ids:
            break
        logits, cache, _ = _forward_sequence(
            model,
            torch.tensor([token_id], dtype=torch.long),
            past_key_values=cache,
            past_length=cache_length,
            use_cache=True,
            output_attentions=False,
        )
        cache_length += 1

    progress_bar.close()
    return generated_ids


def _advance_and_score_privileged_token(
    model,
    token_id: int,
    cache,
    cache_length: int,
    layer_head_list: Sequence[LayerHead],
    privileged_token_indices: Sequence[int],
    aggregation: str,
):
    """Advance one token and score its attention to privileged keys."""
    next_logits, next_cache, attentions = _forward_sequence(
        model,
        torch.tensor([token_id], dtype=torch.long),
        past_key_values=cache,
        past_length=cache_length,
        use_cache=True,
        output_attentions=True,
    )
    attention_row = _aggregate_current_to_privileged_attention(
        attentions,
        layer_head_list,
        privileged_token_indices,
        aggregation,
    )
    score = get_token_score_from_attention_matrix(
        attention_row,
        token_index=0,
        key_token_indices=range(attention_row.numel()),
    )
    del attention_row, attentions
    return next_logits, next_cache, score


def _substring_token_indices(tokenizer, text: str, substring: str) -> list[int]:
    start_char = text.find(substring)
    if start_char < 0:
        raise ValueError("privileged_context was not found in prompt_with_answer.")
    end_char = start_char + len(substring)

    try:
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded["offset_mapping"]
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start_char and start < end_char
        ]
    except (KeyError, NotImplementedError, TypeError, ValueError):
        # Slow tokenizers may not expose offsets. This is a best-effort fallback
        # and assumes the substring boundaries are also token boundaries.
        prefix_ids = _input_ids(tokenizer, text[:start_char])
        through_context_ids = _input_ids(tokenizer, text[:end_char])
        indices = list(range(prefix_ids.numel(), through_context_ids.numel()))

    if not indices:
        raise ValueError("privileged_context did not map to any tokens.")
    return indices


def _token_span_report(
    tokenizer,
    token_ids: Sequence[int],
    start_index: int,
) -> dict[str, Any]:
    token_ids = [int(token_id) for token_id in token_ids]
    return {
        "range": [start_index, start_index + len(token_ids)],
        "text": tokenizer.decode(token_ids, skip_special_tokens=False),
        "token_ids": token_ids,
    }


def _start_repair_event(
    tokenizer,
    *,
    event_id: int,
    start_index: int,
    detected_index: int,
    detector_score: float,
    before_token_ids: Sequence[int],
    comparison_method: str,
    comparison_threshold: float,
) -> dict[str, Any]:
    """Create one report for the complete contiguous repair span."""
    return {
        "event_id": event_id,
        "status": "repairing",
        "start_index": start_index,
        "detected_index": detected_index,
        "detector_score": round(float(detector_score), 6),
        "before": _token_span_report(
            tokenizer,
            before_token_ids,
            start_index,
        ),
        "after": _token_span_report(tokenizer, [], start_index),
        "comparison": {
            "method": comparison_method,
            "threshold": round(float(comparison_threshold), 6),
            "initial_value": None,
            "final_value": None,
            "repair_steps": 0,
        },
    }


def _append_repair_result(
    event: dict[str, Any],
    tokenizer,
    *,
    selected_token_id: int,
    comparison_value: float,
) -> None:
    """Append one selected token while keeping the event report span-based."""
    after_ids = [*event["after"]["token_ids"], int(selected_token_id)]
    event["after"] = _token_span_report(
        tokenizer,
        after_ids,
        event["start_index"],
    )
    comparison = event["comparison"]
    value = round(float(comparison_value), 6)
    if comparison["initial_value"] is None:
        comparison["initial_value"] = value
    comparison["final_value"] = value
    comparison["repair_steps"] += 1


def generate_main(
    model,
    tokenizer,
    prompt_with_answer: str,
    prompt_without_answer: str,
    *,
    privileged_context: str,
    detector_config_path: str | Path | None = None,
    model_key: str | None = None,
    layer_head_list: Sequence[LayerHead] | None = None,
    threshold: float | None = None,
    window_size: int | None = None,
    max_length: int = 50,
    generation_kwargs: dict[str, Any] | None = None,
    fix_comparison_method: str = "attention_score",
    fix_score_difference_threshold: float = 0.1,
    fix_js_divergence_threshold: float = 0.1,
    max_repair_steps: int = 128,
    repair_safe_streak: int | None = None,
    max_repair_cycles_per_span: int = 2,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Generate with privileged context and replace flagged tokens online.

    ``prompt_with_answer`` and ``prompt_without_answer`` must already be
    rendered into model-ready text. Selected heads, threshold, and window size
    can be read from a detector config; explicit arguments override config
    values. Detection backtracks by half the detector window. During repair,
    privileged and clean prompts propose a token each. The privileged candidate
    must pass the leakage detector before repair can end. ``js_divergence`` adds
    a second convergence requirement between the two next-token distributions.
    Accepted repair tokens are required to remain safe for a short streak, and
    repeated detection at the same span eventually disables backtracking there
    so generation cannot oscillate over the same token range indefinitely.

    ``fix_score_difference_threshold`` is retained for call compatibility but
    is no longer used: attention repair now compares the privileged candidate's
    detector score directly with the calibrated detector threshold.

    Returns:
        A dictionary containing ``unfixed`` and ``fixed`` generation versions,
        a per-token debug ``trace``, and one span-based report per repair event.
        Top-level ``text`` and ``token_ids`` remain aliases of ``fixed`` for
        backward compatibility.
    """
    if not prompt_with_answer or not prompt_without_answer:
        raise ValueError(
            "Both prompt_with_answer and prompt_without_answer are required."
        )
    if not privileged_context:
        raise ValueError("privileged_context is required.")
    if max_length < 0:
        raise ValueError("max_length must be non-negative.")
    comparison_aliases = {
        "attention": "attention_score",
        "attention_score": "attention_score",
        "jsd": "js_divergence",
        "jensen_shannon": "js_divergence",
        "jensen_shannon_divergence": "js_divergence",
        "js_divergence": "js_divergence",
    }
    try:
        fix_comparison_method = comparison_aliases[fix_comparison_method]
    except KeyError:
        raise ValueError(
            "fix_comparison_method must be 'attention_score' or 'js_divergence'."
        ) from None
    if fix_score_difference_threshold < 0:
        raise ValueError("fix_score_difference_threshold must be non-negative.")
    if fix_js_divergence_threshold < 0:
        raise ValueError("fix_js_divergence_threshold must be non-negative.")
    if max_repair_steps < 1:
        raise ValueError("max_repair_steps must be at least 1.")
    if repair_safe_streak is not None and repair_safe_streak < 1:
        raise ValueError("repair_safe_streak must be at least 1 when provided.")
    if max_repair_cycles_per_span < 1:
        raise ValueError("max_repair_cycles_per_span must be at least 1.")

    detector: dict[str, Any] = {}
    if detector_config_path is not None:
        detector = _load_detector(detector_config_path, model_key)
    if layer_head_list is None:
        layer_head_list = detector.get("heads")
    if threshold is None:
        threshold = detector.get("threshold")
    if window_size is None:
        window_size = detector.get("window_size")
    aggregation = detector.get("aggregation", "weighted")
    if layer_head_list is None or threshold is None or window_size is None:
        raise ValueError(
            "Provide layer_head_list, threshold, and window_size directly or "
            "through detector_config_path."
        )

    # Validate before generation starts, so malformed detector data fails early.
    _normalise_layer_heads(layer_head_list)
    if int(window_size) < 1:
        raise ValueError("window_size must be at least 1.")
    threshold = float(threshold)
    repair_comparison_threshold = float(
        fix_js_divergence_threshold
        if fix_comparison_method == "js_divergence"
        else threshold
    )

    privileged_prompt_ids = _input_ids(tokenizer, prompt_with_answer)
    clean_prompt_ids = _input_ids(tokenizer, prompt_without_answer)
    privileged_token_indices = _substring_token_indices(
        tokenizer,
        prompt_with_answer,
        privileged_context,
    )
    generated_ids: list[int] = []
    trace: list[dict[str, Any]] = []
    repair_events: list[dict[str, Any]] = []
    eos_token_ids = _eos_token_ids(tokenizer)
    window_size = int(window_size)
    fix_backtrack = max((window_size - 1) // 2, 0)
    required_safe_streak = (
        max(fix_backtrack + 1, 1)
        if repair_safe_streak is None
        else int(repair_safe_streak)
    )

    if max_length == 0:
        empty_fixed = {"text": "", "token_ids": []}
        empty_unfixed = {"text": "", "token_ids": []}
        return {
            "text": "",
            "token_ids": [],
            "fixed": empty_fixed,
            "unfixed": empty_unfixed,
            "trace": [],
            "repair_events": [],
        }

    # If the caller supplied an explicit generator, snapshot its initial state
    # so fixed and unfixed sampling begin from the same random state.
    unfixed_generation_kwargs = _copy_generation_kwargs_for_baseline(generation_kwargs)

    privileged_logits, privileged_cache, privileged_cache_length = (
        _prefill_privileged_state(
            model,
            privileged_prompt_ids,
            generated_ids,
        )
    )
    repairing = False
    active_repair_event: dict[str, Any] | None = None
    repair_step = 0
    repair_safe_count = 0
    repair_cycles_by_start: dict[int, int] = {}
    attempts = 0
    candidate_cache = None
    max_attempts = max(
        1_000,
        max_length * (max_repair_steps + fix_backtrack + 2),
    )
    fixed_progress_bar = tqdm(
        total=max_length,
        desc="Generating fixed",
        unit="token",
        dynamic_ncols=True,
        disable=not show_progress,
    )

    while len(generated_ids) < max_length:
        attempts += 1
        if attempts > max_attempts:
            fixed_progress_bar.close()
            raise RuntimeError(
                "Generation did not make progress after repeated detector "
                "backtracking and repair attempts."
            )

        if not repairing:
            candidate_id = _sample_next_token_id(
                model,
                privileged_logits,
                generation_kwargs,
            )
            (
                candidate_logits,
                candidate_cache,
                candidate_score,
            ) = _advance_and_score_privileged_token(
                model,
                candidate_id,
                privileged_cache,
                privileged_cache_length,
                layer_head_list,
                privileged_token_indices,
                aggregation,
            )

            if candidate_score < threshold:
                privileged_logits = candidate_logits
                privileged_cache = candidate_cache
                privileged_cache_length += 1
                generated_ids.append(candidate_id)
                fixed_progress_bar.update(1)
                trace.append(
                    {
                        "step": len(generated_ids) - 1,
                        "mode": "privileged",
                        "score": candidate_score,
                        "selected_score": candidate_score,
                        "needs_fix": False,
                        "candidate_token_id": candidate_id,
                        "candidate_token": tokenizer.decode(
                            [candidate_id],
                            skip_special_tokens=False,
                        ),
                        "selected_token_id": candidate_id,
                        "selected_token": tokenizer.decode(
                            [candidate_id],
                            skip_special_tokens=False,
                        ),
                    }
                )
                if candidate_id in eos_token_ids:
                    break
                continue

            detected_token_index = len(generated_ids)
            nominal_fix_start = max(0, detected_token_index - fix_backtrack)
            repair_cycle_count = repair_cycles_by_start.get(nominal_fix_start, 0) + 1
            repair_cycles_by_start[nominal_fix_start] = repair_cycle_count
            backtrack_suppressed = repair_cycle_count > max_repair_cycles_per_span
            fix_start_index = (
                detected_token_index if backtrack_suppressed else nominal_fix_start
            )
            removed_ids = generated_ids[fix_start_index:]
            active_repair_event = _start_repair_event(
                tokenizer,
                event_id=len(repair_events),
                start_index=fix_start_index,
                detected_index=detected_token_index,
                detector_score=candidate_score,
                before_token_ids=[*removed_ids, candidate_id],
                comparison_method=fix_comparison_method,
                comparison_threshold=repair_comparison_threshold,
            )
            active_repair_event.update(
                {
                    "nominal_start_index": nominal_fix_start,
                    "cycle_count": repair_cycle_count,
                    "backtrack_suppressed": backtrack_suppressed,
                    "required_safe_streak": required_safe_streak,
                }
            )
            repair_events.append(active_repair_event)

            del generated_ids[fix_start_index:]
            del trace[fix_start_index:]
            fixed_progress_bar.n = len(generated_ids)
            fixed_progress_bar.refresh()
            privileged_cache = None
            candidate_cache = None
            privileged_logits = None
            del candidate_logits
            (
                privileged_logits,
                privileged_cache,
                privileged_cache_length,
            ) = _prefill_privileged_state(
                model,
                privileged_prompt_ids,
                generated_ids,
            )
            repairing = True
            repair_step = 0
            repair_safe_count = 0
            continue

        token_1 = _sample_next_token_id(
            model,
            privileged_logits,
            generation_kwargs,
        )
        accepted_prefix = torch.tensor(generated_ids, dtype=torch.long)
        clean_input = torch.cat([clean_prompt_ids, accepted_prefix])
        clean_logits, _, _ = _forward_sequence(
            model,
            clean_input,
            use_cache=False,
            output_attentions=False,
        )
        token_2 = _sample_next_token_id(
            model,
            clean_logits,
            generation_kwargs,
        )

        token_1_score = None
        token_2_score = None
        selected_score = None
        score_difference = None
        js_divergence = None

        if fix_comparison_method == "js_divergence":
            js_divergence = get_jensen_shannon_divergence(
                privileged_logits,
                clean_logits,
            )
            comparison_value = js_divergence
            comparison_threshold = repair_comparison_threshold
            comparison_converged = comparison_value <= comparison_threshold
        else:
            comparison_value = None
            comparison_threshold = threshold
            comparison_converged = True
        del clean_logits

        # In both repair modes, the privileged candidate must pass the actual
        # leakage detector. This speculative forward can mutate DynamicCache.
        token_1_logits, token_1_cache, token_1_score = (
            _advance_and_score_privileged_token(
                model,
                token_1,
                privileged_cache,
                privileged_cache_length,
                layer_head_list,
                privileged_token_indices,
                aggregation,
            )
        )
        privileged_candidate_safe = token_1_score < threshold
        if fix_comparison_method == "attention_score":
            comparison_value = token_1_score
            comparison_converged = privileged_candidate_safe

        if privileged_candidate_safe and comparison_converged:
            # Keep the safe privileged candidate and continue checking until a
            # long enough safe streak has moved beyond the backtracked region.
            repair_safe_count += 1
            selected_id = token_1
            selected_branch = "privileged"
            selected_score = token_1_score
            privileged_logits = token_1_logits
            privileged_cache = token_1_cache
            repair_continues = repair_safe_count < required_safe_streak
        else:
            # Reject the privileged candidate, rebuild the accepted common
            # prefix, and teacher-force the clean candidate into the privileged
            # branch. The next comparison therefore uses the same output prefix
            # for the clean and privileged prompts.
            repair_safe_count = 0
            privileged_cache = None
            privileged_logits = None
            del token_1_logits, token_1_cache
            (
                privileged_logits,
                privileged_cache,
                privileged_cache_length,
            ) = _prefill_privileged_state(
                model,
                privileged_prompt_ids,
                generated_ids,
            )
            privileged_logits, privileged_cache, _ = _forward_sequence(
                model,
                torch.tensor([token_2], dtype=torch.long),
                past_key_values=privileged_cache,
                past_length=privileged_cache_length,
                use_cache=True,
                output_attentions=False,
            )
            selected_id = token_2
            selected_branch = "clean"
            repair_continues = True

        generated_ids.append(selected_id)
        fixed_progress_bar.update(1)
        privileged_cache_length += 1
        repair_step += 1
        if active_repair_event is None:
            raise RuntimeError("Repair state is missing its span report.")
        _append_repair_result(
            active_repair_event,
            tokenizer,
            selected_token_id=selected_id,
            comparison_value=comparison_value,
        )
        if not repair_continues:
            active_repair_event["status"] = "converged"
        elif selected_id in eos_token_ids:
            active_repair_event["status"] = "eos_during_repair"
        repair_event_id = active_repair_event["event_id"]
        trace.append(
            {
                "step": len(generated_ids) - 1,
                "mode": "repair",
                "repair_event_id": repair_event_id,
                "repair_step": repair_step,
                "repair_comparison_method": fix_comparison_method,
                "comparison_value": comparison_value,
                "comparison_threshold": comparison_threshold,
                "detector_score": token_1_score,
                "detector_threshold": threshold,
                "privileged_candidate_safe": privileged_candidate_safe,
                "repair_safe_count": repair_safe_count,
                "required_safe_streak": required_safe_streak,
                "score": token_1_score,
                "selected_score": selected_score,
                "selected_branch": selected_branch,
                "needs_fix": True,
                "token_1_id": token_1,
                "token_1": tokenizer.decode(
                    [token_1],
                    skip_special_tokens=False,
                ),
                "token_1_score": token_1_score,
                "token_2_id": token_2,
                "token_2": tokenizer.decode(
                    [token_2],
                    skip_special_tokens=False,
                ),
                "token_2_score": token_2_score,
                "score_difference": score_difference,
                "jensen_shannon_divergence": js_divergence,
                "repair_continues": repair_continues,
                "candidate_token_id": token_1,
                "candidate_token": tokenizer.decode(
                    [token_1],
                    skip_special_tokens=False,
                ),
                "selected_token_id": selected_id,
                "selected_token": tokenizer.decode(
                    [selected_id],
                    skip_special_tokens=False,
                ),
            }
        )

        if selected_id in eos_token_ids:
            break
        if repair_continues and repair_step >= max_repair_steps:
            active_repair_event["status"] = "max_repair_steps_exceeded"
            fixed_progress_bar.close()
            raise RuntimeError(
                f"Repair did not converge within max_repair_steps={max_repair_steps}."
            )
        if not repair_continues:
            repairing = False
            active_repair_event = None
            repair_step = 0
            repair_safe_count = 0

    if active_repair_event is not None and active_repair_event["status"] == "repairing":
        active_repair_event["status"] = "max_length_reached"
    fixed_progress_bar.close()
    fixed_token_ids = list(generated_ids)
    fixed_text = tokenizer.decode(fixed_token_ids, skip_special_tokens=True)

    # Do not keep the fixed branch's KV cache alive while generating the
    # baseline; otherwise both full caches coexist at peak memory.
    privileged_cache = None
    candidate_cache = None
    privileged_logits = None
    unfixed_token_ids = _generate_without_repair(
        model,
        tokenizer,
        privileged_prompt_ids,
        max_length=max_length,
        generation_kwargs=unfixed_generation_kwargs,
        show_progress=show_progress,
    )
    unfixed_text = tokenizer.decode(
        unfixed_token_ids,
        skip_special_tokens=True,
    )

    return {
        "text": fixed_text,
        "token_ids": fixed_token_ids,
        "fixed": {
            "text": fixed_text,
            "token_ids": list(fixed_token_ids),
        },
        "unfixed": {
            "text": unfixed_text,
            "token_ids": list(unfixed_token_ids),
        },
        "trace": trace,
        "repair_events": repair_events,
    }
