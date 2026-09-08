from __future__ import annotations

import inspect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch
from tqdm.auto import tqdm

LayerHead = tuple[int, int] | dict[str, Any]


@dataclass
class TokenTransition:
    """The model state obtained after teacher-forcing one token."""

    token_id: int
    source_cache_length: int
    next_logits: torch.Tensor
    past_key_values: Any
    attentions: Any = None


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
    """Load a causal language model suitable for attention-based generation."""
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

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        **{**shared_kwargs, **(tokenizer_kwargs or {})},
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    final_model_kwargs = {**shared_kwargs, **(model_kwargs or {})}
    final_model_kwargs.setdefault("attn_implementation", attn_implementation)
    if device_map is not None:
        final_model_kwargs.setdefault("device_map", device_map)

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
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _input_ids(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("The tokenizer must return one batch of input_ids.")
    return input_ids[0].detach().cpu().to(dtype=torch.long)


def _eos_token_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return {int(token_id) for token_id in eos}


def _forward_accepts_argument(model, name: str) -> bool:
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
    """Forward a complete prefix or only the tokens missing from a KV cache."""
    if input_ids.ndim != 1 or input_ids.numel() == 0:
        raise ValueError("input_ids must be a non-empty one-dimensional tensor.")
    if past_key_values is None and past_length != 0:
        raise ValueError("past_length must be zero when no cache is supplied.")

    device = _model_device(model)
    batched_ids = input_ids.unsqueeze(0).to(device)
    total_length = past_length + input_ids.numel()
    kwargs = {
        "input_ids": batched_ids,
        "attention_mask": torch.ones(
            (1, total_length),
            dtype=torch.long,
            device=device,
        ),
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
    """Sample one token with the common Transformers generation settings."""
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
            "The token-by-token loop does not support generation arguments: "
            + ", ".join(unsupported)
        )

    config = getattr(model, "generation_config", None)

    def setting(name: str, default):
        if name in settings:
            return settings[name]
        return getattr(config, name, default) if config is not None else default

    scores = torch.as_tensor(logits).detach().float()
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector.")
    if not bool(setting("do_sample", False)):
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
        cutoff = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)

    if top_p < 1:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probabilities = torch.softmax(
            sorted_scores,
            dim=-1,
        ).cumsum(dim=-1)
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


def _copy_generation_kwargs(
    generation_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Copy sampling settings and snapshot an explicit torch.Generator."""
    copied = dict(generation_kwargs or {})
    generator = copied.get("generator")
    if generator is None:
        return copied
    try:
        cloned = torch.Generator(device=generator.device)
        cloned.set_state(generator.get_state())
    except (AttributeError, RuntimeError, TypeError):
        return copied
    copied["generator"] = cloned
    return copied


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
    data_path: str | Path,
    model_key: str | None = None,
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
        prefix_ids = _input_ids(tokenizer, text[:start_char])
        through_context_ids = _input_ids(tokenizer, text[:end_char])
        indices = list(range(prefix_ids.numel(), through_context_ids.numel()))

    if not indices:
        raise ValueError("privileged_context did not map to any tokens.")
    return indices


def get_jensen_shannon_divergence(
    privileged_logits: torch.Tensor,
    clean_logits: torch.Tensor,
) -> float:
    """Compare two vocabulary distributions using natural-log JSD."""
    privileged_logits = torch.as_tensor(privileged_logits).detach().float()
    clean_logits = torch.as_tensor(clean_logits).detach().float()
    if privileged_logits.ndim == 2 and privileged_logits.shape[0] == 1:
        privileged_logits = privileged_logits[0]
    if clean_logits.ndim == 2 and clean_logits.shape[0] == 1:
        clean_logits = clean_logits[0]
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
    return max(float(divergence.item()), 0.0)


def _token_span_report(
    tokenizer,
    token_ids: Sequence[int],
    start_index: int,
) -> dict[str, Any]:
    ids = [int(token_id) for token_id in token_ids]
    return {
        "range": [start_index, start_index + len(ids)],
        "text": tokenizer.decode(ids, skip_special_tokens=False),
        "token_ids": ids,
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
    return {
        "event_id": event_id,
        "status": "repairing",
        "start_index": start_index,
        "detected_index": detected_index,
        "detector_score": round(float(detector_score), 6),
        "before": _token_span_report(tokenizer, before_token_ids, start_index),
        "after": _token_span_report(tokenizer, [], start_index),
        "branch_generations": {
            "privileged": _token_span_report(tokenizer, [], start_index),
            "clean": _token_span_report(tokenizer, [], start_index),
        },
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


def _append_branch_generation(
    event: dict[str, Any],
    tokenizer,
    *,
    branch: str,
    token_id: int,
) -> None:
    """Record a branch proposal without confusing it with the accepted prefix."""
    if branch not in {"privileged", "clean"}:
        raise ValueError("branch must be either 'privileged' or 'clean'.")
    report = event["branch_generations"][branch]
    token_ids = [*report["token_ids"], int(token_id)]
    event["branch_generations"][branch] = _token_span_report(
        tokenizer,
        token_ids,
        event["start_index"],
    )


def _repair_branch_generation_reports(
    repair_events: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for event in repair_events:
        branch_generations = event["branch_generations"]
        reports.append(
            {
                "event_id": event["event_id"],
                "start_index": event["start_index"],
                "privileged": {
                    **branch_generations["privileged"],
                    "token_ids": list(branch_generations["privileged"]["token_ids"]),
                },
                "clean": {
                    **branch_generations["clean"],
                    "token_ids": list(branch_generations["clean"]["token_ids"]),
                },
            }
        )
    return reports


class TokenByTokenGenerator:
    """Own the prompt, sampling policy, and KV-cache of one generation branch.

    The class separates proposing, evaluating, and accepting a token. That is
    important for ``CustomGenerator`` because it must evaluate two candidates
    from the same prefix before choosing one.
    """

    def __init__(
        self,
        model,
        tokenizer,
        prompt: str,
        priviledge_context: str | None = None,
        *,
        privileged_context: str | None = None,
        temperature: float | None = 0.7,
        top_k: int | None = 0,
        top_p: float | None = 1.0,
        do_sample: bool | None = True,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        if not prompt:
            raise ValueError("prompt is required.")
        if (
            privileged_context is not None
            and priviledge_context is not None
            and privileged_context != priviledge_context
        ):
            raise ValueError("privileged_context and priviledge_context disagree.")
        privileged_context = privileged_context or priviledge_context

        self.model = model
        self.tokenizer = tokenizer
        self.base_prompt = prompt
        self.privileged_context = privileged_context
        self.priviledge_context = privileged_context
        self.prompt = prompt + privileged_context if privileged_context else prompt

        settings = dict(generation_kwargs or {})
        explicit_settings = {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
        }
        for name, value in explicit_settings.items():
            if value is not None:
                settings.setdefault(name, value)
        self.generation_kwargs = settings

        self.input_ids: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.past_key_values = None
        self.next_logits: torch.Tensor | None = None
        self.cache_length = 0

        self.generated_ids: list[int] = []
        self.finished = False
        self._started = False
        self._eos_token_ids = _eos_token_ids(tokenizer)

    def _sync_attention_mask(self) -> None:
        if not self._started:
            self.attention_mask = None
            return
        self.attention_mask = torch.ones(
            (1, self.cache_length),
            dtype=torch.long,
            device=_model_device(self.model),
        )

    def start(self, generated_ids: Sequence[int] | None = None) -> int:
        """Prefill the prompt plus an optional accepted generation prefix."""
        prompt_ids = _input_ids(self.tokenizer, self.prompt)
        if prompt_ids.numel() == 0:
            raise ValueError("prompt must map to at least one token.")
        prefix_ids = [int(token_id) for token_id in (generated_ids or [])]
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long)
        full_prefix = torch.cat([prompt_ids, prefix_tensor])

        # Drop a previous branch state before allocating a replacement cache.
        self.input_ids = None
        self.attention_mask = None
        self.past_key_values = None
        self.next_logits = None
        self.cache_length = 0
        self._started = False
        next_logits, cache, _ = _forward_sequence(
            self.model,
            full_prefix,
            use_cache=True,
            output_attentions=False,
        )
        device = _model_device(self.model)
        self.input_ids = prompt_ids.unsqueeze(0).to(device)
        self.next_logits = next_logits
        self.past_key_values = cache
        self.cache_length = int(full_prefix.numel())
        self.generated_ids = prefix_ids
        self.finished = bool(prefix_ids and prefix_ids[-1] in self._eos_token_ids)
        self._started = True
        self._sync_attention_mask()
        return int(prompt_ids.numel())

    def restore(self, generated_ids: Sequence[int]) -> int:
        """Rebuild the branch cache at an accepted generation prefix."""
        return self.start(generated_ids=generated_ids)

    def clear_cache(self) -> None:
        """Release model state while retaining generated token IDs."""
        self.input_ids = None
        self.attention_mask = None
        self.past_key_values = None
        self.next_logits = None
        self.cache_length = 0
        self._started = False

    def _ensure_started(self) -> None:
        if not self._started or self.next_logits is None:
            raise RuntimeError("Call start() before generating tokens.")

    def sample_token_id(self, logits: torch.Tensor | None = None) -> int:
        self._ensure_started()
        scores = self.next_logits if logits is None else logits
        if scores is None:
            raise RuntimeError("No logits are available for sampling.")
        return _sample_next_token_id(
            self.model,
            scores,
            self.generation_kwargs,
        )

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Backward-compatible tensor-returning sampling helper."""
        token_id = _sample_next_token_id(
            self.model,
            logits,
            self.generation_kwargs,
        )
        return torch.tensor(
            [[token_id]],
            dtype=torch.long,
            device=torch.as_tensor(logits).device,
        )

    def evaluate_token(
        self,
        token_id: int,
        *,
        output_attentions: bool = False,
    ) -> TokenTransition:
        """Evaluate a token without adding it to the accepted-token list.

        Some Transformers cache implementations mutate a cache in place. If a
        caller evaluates an alternate candidate afterwards, it must call
        ``restore`` first. ``CustomGenerator`` handles that automatically.
        """
        self._ensure_started()
        token_id = int(token_id)
        next_logits, cache, attentions = _forward_sequence(
            self.model,
            torch.tensor([token_id], dtype=torch.long),
            past_key_values=self.past_key_values,
            past_length=self.cache_length,
            use_cache=True,
            output_attentions=output_attentions,
        )
        return TokenTransition(
            token_id=token_id,
            source_cache_length=self.cache_length,
            next_logits=next_logits,
            past_key_values=cache,
            attentions=attentions,
        )

    def accept_transition(self, transition: TokenTransition) -> None:
        if transition.source_cache_length != self.cache_length:
            raise RuntimeError("Cannot accept a transition from a stale prefix.")
        self.next_logits = transition.next_logits
        self.past_key_values = transition.past_key_values
        self.cache_length += 1
        self.generated_ids.append(int(transition.token_id))
        self.finished = transition.token_id in self._eos_token_ids
        self._sync_attention_mask()

    def accept_token(self, token_id: int) -> TokenTransition:
        transition = self.evaluate_token(token_id)
        self.accept_transition(transition)
        return transition

    def step(self) -> tuple[int | None, str | None, bool]:
        """Generate and accept exactly one token."""
        self._ensure_started()
        if self.finished:
            return None, None, True

        token_id = self.sample_token_id()
        transition = self.evaluate_token(token_id)
        self.accept_transition(transition)
        token_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )
        return token_id, token_text, self.finished

    def generate(
        self,
        max_new_tokens: int,
        *,
        reset: bool = True,
        show_progress: bool = False,
        progress_description: str = "Generating",
    ) -> dict[str, Any]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if reset:
            self.start()
        else:
            self._ensure_started()

        progress_bar = tqdm(
            total=max_new_tokens,
            initial=min(len(self.generated_ids), max_new_tokens),
            desc=progress_description,
            unit="token",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        try:
            while len(self.generated_ids) < max_new_tokens and not self.finished:
                self.step()
                progress_bar.update(1)
        finally:
            progress_bar.close()

        ids = list(self.generated_ids[:max_new_tokens])
        return {
            "text": self.tokenizer.decode(ids, skip_special_tokens=True),
            "token_ids": ids,
        }

    def get_generated_text(self, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=skip_special_tokens,
        )


class CustomGenerator(TokenByTokenGenerator):
    """Generate with a privileged prompt and repair detected leakage online.

    ``prompt_with_answer``/``prompt_without_answer`` is the preferred API. For
    compatibility with the original outline, callers may instead pass
    ``prompt`` and ``privileged_context``; the privileged prompt is then built
    as ``prompt + privileged_context``. The misspelled
    ``priviledge_context`` argument is retained as a compatibility alias.
    """

    _COMPARISON_ALIASES: ClassVar[dict[str, str]] = {
        "attention": "attention_score",
        "attention_score": "attention_score",
        "jsd": "js_divergence",
        "jensen_shannon": "js_divergence",
        "jensen_shannon_divergence": "js_divergence",
        "js_divergence": "js_divergence",
    }

    def __init__(
        self,
        model,
        tokenizer,
        detector_config_path: str | Path | None = None,
        model_key: str | None = None,
        max_new_tokens: int = 512,
        *,
        prompt: str | None = None,
        privileged_context: str | None = None,
        priviledge_context: str | None = None,
        prompt_with_answer: str | None = None,
        prompt_without_answer: str | None = None,
        layer_head_list: Sequence[LayerHead] | None = None,
        threshold: float | None = None,
        window_size: int | None = None,
        aggregation: str | None = None,
        fix_comparison_method: str = "attention_score",
        fix_score_difference_threshold: float = 0.1,
        fix_js_divergence_threshold: float = 0.1,
        max_repair_steps: int = 128,
        temperature: float | None = 0.7,
        top_k: int | None = 0,
        top_p: float | None = 1.0,
        do_sample: bool | None = True,
        generation_kwargs: dict[str, Any] | None = None,
    ):
        if (
            privileged_context is not None
            and priviledge_context is not None
            and privileged_context != priviledge_context
        ):
            raise ValueError("privileged_context and priviledge_context disagree.")
        privileged_context = privileged_context or priviledge_context
        if not privileged_context:
            raise ValueError("privileged_context is required.")

        if prompt_with_answer is None:
            if not prompt:
                raise ValueError(
                    "Provide prompt_with_answer or prompt plus privileged_context."
                )
            prompt_with_answer = prompt + privileged_context
        if prompt_without_answer is None:
            if prompt is None:
                raise ValueError(
                    "Provide prompt_without_answer when prompt_with_answer is used."
                )
            prompt_without_answer = prompt
        if not prompt_with_answer or not prompt_without_answer:
            raise ValueError(
                "Both prompt_with_answer and prompt_without_answer are required."
            )
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if max_repair_steps < 1:
            raise ValueError("max_repair_steps must be at least 1.")
        if fix_score_difference_threshold < 0:
            raise ValueError("fix_score_difference_threshold must be non-negative.")
        if fix_js_divergence_threshold < 0:
            raise ValueError("fix_js_divergence_threshold must be non-negative.")

        try:
            fix_comparison_method = self._COMPARISON_ALIASES[fix_comparison_method]
        except KeyError:
            raise ValueError(
                "fix_comparison_method must be 'attention_score' or 'js_divergence'."
            ) from None

        detector: dict[str, Any] = {}
        if detector_config_path is not None:
            detector = _load_detector(detector_config_path, model_key)
        if layer_head_list is None:
            layer_head_list = detector.get("heads")
        if threshold is None:
            threshold = detector.get("threshold")
        if window_size is None:
            window_size = detector.get("window_size")
        if aggregation is None:
            aggregation = detector.get("aggregation", "weighted")
        if layer_head_list is None or threshold is None or window_size is None:
            raise ValueError(
                "Provide layer_head_list, threshold, and window_size directly "
                "or through detector_config_path."
            )
        if aggregation not in {"weighted", "mean"}:
            raise ValueError("aggregation must be either 'weighted' or 'mean'.")
        if int(window_size) < 1:
            raise ValueError("window_size must be at least 1.")

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_with_answer,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            generation_kwargs=generation_kwargs,
        )

        self.prompt_with_answer = prompt_with_answer
        self.prompt_without_answer = prompt_without_answer
        self.privileged_context = privileged_context
        # Compatibility with the spelling used in the outline.
        self.priviledge_context = privileged_context
        self.detector = detector
        self.layer_heads = _normalise_layer_heads(layer_head_list)
        self.threshold = float(threshold)
        self.window_size = int(window_size)
        self.aggregation = aggregation
        self.privileged_token_indices = _substring_token_indices(
            tokenizer,
            prompt_with_answer,
            privileged_context,
        )

        self.max_new_tokens = int(max_new_tokens)
        self.max_length = self.max_new_tokens
        self.max_repair_steps = int(max_repair_steps)
        self.fix_comparison_method = fix_comparison_method
        self.fix_score_difference_threshold = float(fix_score_difference_threshold)
        self.fix_js_divergence_threshold = float(fix_js_divergence_threshold)
        self.repair_comparison_threshold = (
            self.fix_js_divergence_threshold
            if fix_comparison_method == "js_divergence"
            else self.threshold
        )
        self.fix_backtrack = max((self.window_size - 1) // 2, 0)

        # The clean branch is kept only while repairing. It shares the same
        # sampling generator intentionally, preserving sequential RNG behavior.
        self.clean_generator = TokenByTokenGenerator(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_without_answer,
            temperature=None,
            top_k=None,
            top_p=None,
            do_sample=None,
            generation_kwargs=self.generation_kwargs,
        )

        self.trace: list[dict[str, Any]] = []
        self.repair_events: list[dict[str, Any]] = []
        self.repairing = False
        self.active_repair_event: dict[str, Any] | None = None
        self.repair_step = 0
        self.attempts = 0
        self.max_attempts = max(
            1_000,
            self.max_new_tokens * (self.max_repair_steps + self.fix_backtrack + 2),
        )

        self.current_index = 0
        self.fixed_index = 0
        self.attention_score: float | None = None
        self.last_detector_score: float | None = None
        self.last_head_scores: list[float] = []
        self._last_privileged_logits: torch.Tensor | None = None
        self._last_clean_logits: torch.Tensor | None = None
        self._baseline_generation_kwargs: dict[str, Any] = {}
        self.fixed_token: dict[str, Any] = {}
        self._update_fixed_token()

    def _update_fixed_token(self) -> None:
        self.fixed_token = {
            "next_logits": self.next_logits,
            "kv_cache": self.past_key_values,
            "index": len(self.generated_ids),
            "prompt": self.prompt_without_answer,
            "privileged_context": self.privileged_context,
        }

    def start(self, generated_ids: Sequence[int] | None = None) -> int:
        if generated_ids:
            raise ValueError(
                "CustomGenerator.start() always starts a new run; use restore() "
                "only for low-level branch manipulation."
            )
        prompt_length = TokenByTokenGenerator.start(self)
        self.clean_generator.clear_cache()
        self.trace = []
        self.repair_events = []
        self.repairing = False
        self.active_repair_event = None
        self.repair_step = 0
        self.attempts = 0
        self.current_index = 0
        self.fixed_index = 0
        self.attention_score = None
        self.last_detector_score = None
        self.last_head_scores = []
        self._last_privileged_logits = None
        self._last_clean_logits = None
        self._baseline_generation_kwargs = _copy_generation_kwargs(
            self.generation_kwargs
        )
        self._update_fixed_token()
        return prompt_length

    def restore(self, generated_ids: Sequence[int]) -> int:
        """Rebuild only the privileged branch at an accepted prefix."""
        self.fixed_token = {}
        prompt_length = TokenByTokenGenerator.start(
            self,
            generated_ids=generated_ids,
        )
        self.current_index = len(self.generated_ids)
        self._update_fixed_token()
        return prompt_length

    def get_scores(self, attentions=None) -> list[float]:
        """Return current-to-privileged attention for every selected head."""
        if attentions is None:
            if not self.last_head_scores:
                raise ValueError("No attention scores are available yet.")
            return list(self.last_head_scores)

        scores: list[float] = []
        for layer, head, _ in self.layer_heads:
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
                self.privileged_token_indices,
                dtype=torch.long,
                device=layer_attention.device,
            )
            row = layer_attention[0, head, -1]
            if int(key_indices.max()) >= row.numel():
                raise IndexError(
                    "A privileged-context token index is outside the attention row."
                )
            score = row.index_select(0, key_indices).sum()
            scores.append(float(score.detach().float().cpu()))
        self.last_head_scores = scores
        return list(scores)

    def _aggregate_scores(self, scores: Sequence[float]) -> float:
        if len(scores) != len(self.layer_heads):
            raise ValueError("One attention score is required for each selected head.")
        if self.aggregation == "mean":
            return float(sum(scores) / len(scores))
        weights = [weight for _, _, weight in self.layer_heads]
        total_weight = sum(weights)
        return float(
            sum(score * weight for score, weight in zip(scores, weights)) / total_weight
        )

    def _score_attentions(self, attentions) -> float:
        score = self._aggregate_scores(self.get_scores(attentions))
        self.attention_score = score
        return score

    def check(self, score: float | None = None) -> bool:
        """Return whether a detector score crosses the calibrated threshold."""
        if score is None:
            score = self.last_detector_score
        if score is None:
            raise ValueError("No detector score is available yet.")
        return float(score) >= self.threshold

    def jensen_shannon_score(
        self,
        privileged_logits: torch.Tensor | None = None,
        clean_logits: torch.Tensor | None = None,
    ) -> float:
        privileged_logits = (
            self._last_privileged_logits
            if privileged_logits is None
            else privileged_logits
        )
        clean_logits = self._last_clean_logits if clean_logits is None else clean_logits
        if privileged_logits is None or clean_logits is None:
            raise ValueError("Both privileged and clean logits are required.")
        return get_jensen_shannon_divergence(
            privileged_logits,
            clean_logits,
        )

    def _record_privileged_token(self, token_id: int, score: float) -> None:
        token_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )
        self.trace.append(
            {
                "step": len(self.generated_ids) - 1,
                "mode": "privileged",
                "score": score,
                "selected_score": score,
                "needs_fix": False,
                "candidate_token_id": token_id,
                "candidate_token": token_text,
                "selected_token_id": token_id,
                "selected_token": token_text,
            }
        )

    def _begin_repair(self, candidate_id: int, detector_score: float) -> None:
        detected_index = len(self.generated_ids)
        self.fixed_index = max(0, detected_index - self.fix_backtrack)
        removed_ids = self.generated_ids[self.fixed_index :]
        event = _start_repair_event(
            self.tokenizer,
            event_id=len(self.repair_events),
            start_index=self.fixed_index,
            detected_index=detected_index,
            detector_score=detector_score,
            before_token_ids=[*removed_ids, candidate_id],
            comparison_method=self.fix_comparison_method,
            comparison_threshold=self.repair_comparison_threshold,
        )
        self.repair_events.append(event)
        self.active_repair_event = event

        del self.generated_ids[self.fixed_index :]
        del self.trace[self.fixed_index :]

        # The candidate forward may have mutated the original DynamicCache.
        # Rebuild both branches at the exact common prefix before comparison.
        self.fixed_token = {}
        TokenByTokenGenerator.start(self, generated_ids=self.generated_ids)
        self.clean_generator.start(generated_ids=self.generated_ids)
        self.repairing = True
        self.repair_step = 0
        self.current_index = len(self.generated_ids)
        self._update_fixed_token()

    def _restore_privileged_prefix(self) -> None:
        self.fixed_token = {}
        TokenByTokenGenerator.start(self, generated_ids=self.generated_ids)

    def _advance_clean_branch(self, selected_id: int) -> None:
        transition = self.clean_generator.evaluate_token(selected_id)
        self.clean_generator.accept_transition(transition)

    def step_regenerate(self) -> tuple[int | None, str | None, bool]:
        """Accept one token while keeping clean and privileged prefixes aligned.

        In attention mode, the detector directly decides whether the current
        privileged candidate is safe. A leaking candidate is discarded and a
        clean candidate is teacher-forced into both branches. A safe privileged
        candidate is accepted and ends the repair event. JSD mode follows the
        same prefix-synchronization rule: high divergence selects the clean
        proposal, while convergence selects the privileged proposal.
        """
        if not self.repairing or self.active_repair_event is None:
            raise RuntimeError("No repair event is active.")

        event = self.active_repair_event
        token_1 = self.sample_token_id()
        _append_branch_generation(
            event,
            self.tokenizer,
            branch="privileged",
            token_id=token_1,
        )
        self._last_privileged_logits = self.next_logits
        self._last_clean_logits = None

        token_2: int | None = None
        token_1_score: float | None = None
        token_2_score: float | None = None
        selected_score: float | None = None
        score_difference: float | None = None
        js_divergence: float | None = None

        if self.fix_comparison_method == "js_divergence":
            token_2 = self.clean_generator.sample_token_id()
            _append_branch_generation(
                event,
                self.tokenizer,
                branch="clean",
                token_id=token_2,
            )
            self._last_clean_logits = self.clean_generator.next_logits
            js_divergence = self.jensen_shannon_score()
            comparison_value = js_divergence
            repair_continues = comparison_value > self.repair_comparison_threshold
            selected_id = token_2 if repair_continues else token_1
            selected_transition = self.evaluate_token(selected_id)
            self.accept_transition(selected_transition)
        else:
            token_1_transition = self.evaluate_token(
                token_1,
                output_attentions=True,
            )
            token_1_score = self._score_attentions(token_1_transition.attentions)
            token_1_transition.attentions = None
            self.last_detector_score = token_1_score
            comparison_value = token_1_score
            repair_continues = self.check(token_1_score)

            if repair_continues:
                # The privileged candidate is still leaking. Its speculative
                # forward may have mutated the DynamicCache, so rebuild the
                # accepted prefix before teacher-forcing the clean candidate.
                del token_1_transition
                token_2 = self.clean_generator.sample_token_id()
                _append_branch_generation(
                    event,
                    self.tokenizer,
                    branch="clean",
                    token_id=token_2,
                )
                self._last_clean_logits = self.clean_generator.next_logits
                self._restore_privileged_prefix()
                selected_id = token_2
                selected_transition = self.evaluate_token(selected_id)
                self.accept_transition(selected_transition)
            else:
                # The detector considers the privileged candidate safe. Keep
                # its already-computed transition and leave repair mode.
                selected_id = token_1
                selected_score = token_1_score
                self.accept_transition(token_1_transition)

        self._advance_clean_branch(selected_id)
        self.repair_step += 1
        self.current_index = len(self.generated_ids)
        _append_repair_result(
            event,
            self.tokenizer,
            selected_token_id=selected_id,
            comparison_value=comparison_value,
        )

        if not repair_continues:
            event["status"] = "converged"
        elif selected_id in self._eos_token_ids:
            event["status"] = "eos_during_repair"

        token_1_text = self.tokenizer.decode(
            [token_1],
            skip_special_tokens=False,
        )
        token_2_text = (
            self.tokenizer.decode(
                [token_2],
                skip_special_tokens=False,
            )
            if token_2 is not None
            else None
        )
        selected_text = self.tokenizer.decode(
            [selected_id],
            skip_special_tokens=False,
        )
        self.trace.append(
            {
                "step": len(self.generated_ids) - 1,
                "mode": "repair",
                "repair_event_id": event["event_id"],
                "repair_step": self.repair_step,
                "repair_comparison_method": self.fix_comparison_method,
                "comparison_value": comparison_value,
                "comparison_threshold": self.repair_comparison_threshold,
                "detector_score": token_1_score,
                "detector_threshold": self.threshold,
                "score": token_1_score,
                "selected_score": selected_score,
                "selected_branch": "clean" if repair_continues else "privileged",
                "needs_fix": True,
                "token_1_id": token_1,
                "token_1": token_1_text,
                "token_1_score": token_1_score,
                "token_2_id": token_2,
                "token_2": token_2_text,
                "token_2_score": token_2_score,
                "score_difference": score_difference,
                "jensen_shannon_divergence": js_divergence,
                "repair_continues": repair_continues,
                "candidate_token_id": token_1,
                "candidate_token": token_1_text,
                "selected_token_id": selected_id,
                "selected_token": selected_text,
            }
        )

        if repair_continues and self.repair_step >= self.max_repair_steps:
            event["status"] = "max_repair_steps_exceeded"
            self._update_fixed_token()
            raise RuntimeError(
                "Repair did not converge within max_repair_steps="
                f"{self.max_repair_steps}."
            )
        if not repair_continues or self.finished:
            self.repairing = False
            self.active_repair_event = None
            self.repair_step = 0
            self.clean_generator.clear_cache()

        self._update_fixed_token()
        return selected_id, selected_text, self.finished

    def step(self) -> tuple[int | None, str | None, bool]:
        """Run detection or repair until exactly one token is accepted.

        A detection can backtrack already accepted tokens before this method
        returns. Consumers that stream text should therefore render
        ``generated_ids`` after each call instead of blindly appending the
        returned token text.
        """
        self._ensure_started()
        if self.finished or len(self.generated_ids) >= self.max_new_tokens:
            self.finished = True
            return None, None, True

        self.attempts += 1
        if self.attempts > self.max_attempts:
            raise RuntimeError(
                "Generation did not make progress after repeated detector "
                "backtracking and repair attempts."
            )

        if self.repairing:
            return self.step_regenerate()

        candidate_id = self.sample_token_id()
        transition = self.evaluate_token(
            candidate_id,
            output_attentions=True,
        )
        candidate_score = self._score_attentions(transition.attentions)
        transition.attentions = None
        self.last_detector_score = candidate_score

        if not self.check(candidate_score):
            self.accept_transition(transition)
            self.current_index = len(self.generated_ids)
            self._record_privileged_token(candidate_id, candidate_score)
            self._update_fixed_token()
            token_text = self.tokenizer.decode(
                [candidate_id],
                skip_special_tokens=False,
            )
            return candidate_id, token_text, self.finished

        del transition
        self._begin_repair(candidate_id, candidate_score)
        return self.step_regenerate()

    def _generate_unfixed(self, show_progress: bool) -> dict[str, Any]:
        baseline = TokenByTokenGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.prompt_with_answer,
            temperature=None,
            top_k=None,
            top_p=None,
            do_sample=None,
            generation_kwargs=self._baseline_generation_kwargs,
        )
        return baseline.generate(
            self.max_new_tokens,
            show_progress=show_progress,
            progress_description="Generating unfixed",
        )

    def generate(
        self,
        max_new_tokens: int | None = None,
        *,
        reset: bool = True,
        show_progress: bool = True,
        include_unfixed: bool = True,
    ) -> dict[str, Any]:
        """Run the complete detector/repair pipeline and return both versions."""
        if max_new_tokens is not None:
            if max_new_tokens < 0:
                raise ValueError("max_new_tokens must be non-negative.")
            self.max_new_tokens = int(max_new_tokens)
            self.max_length = self.max_new_tokens
            self.max_attempts = max(
                1_000,
                self.max_new_tokens * (self.max_repair_steps + self.fix_backtrack + 2),
            )
        if reset:
            self.start()
        else:
            self._ensure_started()

        progress_bar = tqdm(
            total=self.max_new_tokens,
            initial=min(len(self.generated_ids), self.max_new_tokens),
            desc="Generating fixed",
            unit="token",
            dynamic_ncols=True,
            disable=not show_progress,
        )
        try:
            while len(self.generated_ids) < self.max_new_tokens and not self.finished:
                self.step()
                # Backtracking can decrease the number of accepted tokens.
                progress_bar.n = min(
                    len(self.generated_ids),
                    self.max_new_tokens,
                )
                progress_bar.refresh()
        finally:
            progress_bar.close()

        if (
            self.active_repair_event is not None
            and self.active_repair_event["status"] == "repairing"
        ):
            self.active_repair_event["status"] = "max_length_reached"
        if len(self.generated_ids) >= self.max_new_tokens:
            self.finished = True

        fixed_ids = list(self.generated_ids)
        fixed = {
            "text": self.tokenizer.decode(fixed_ids, skip_special_tokens=True),
            "token_ids": fixed_ids,
        }

        if include_unfixed:
            # Avoid holding fixed, clean, and baseline KV caches simultaneously.
            self.clear_cache()
            self.clean_generator.clear_cache()
            self._update_fixed_token()
            unfixed = self._generate_unfixed(show_progress)
        else:
            unfixed = None

        return {
            "text": fixed["text"],
            "token_ids": list(fixed_ids),
            "fixed": fixed,
            "unfixed": unfixed,
            "trace": list(self.trace),
            "repair_events": list(self.repair_events),
            "repair_branch_generations": _repair_branch_generation_reports(
                self.repair_events
            ),
        }


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
    show_progress: bool = True,
) -> dict[str, Any]:
    """Backward-compatible functional entry point backed by the OOP pipeline."""
    generator = CustomGenerator(
        model=model,
        tokenizer=tokenizer,
        detector_config_path=detector_config_path,
        model_key=model_key,
        max_new_tokens=max_length,
        prompt_with_answer=prompt_with_answer,
        prompt_without_answer=prompt_without_answer,
        privileged_context=privileged_context,
        layer_head_list=layer_head_list,
        threshold=threshold,
        window_size=window_size,
        fix_comparison_method=fix_comparison_method,
        fix_score_difference_threshold=fix_score_difference_threshold,
        fix_js_divergence_threshold=fix_js_divergence_threshold,
        max_repair_steps=max_repair_steps,
        temperature=None,
        top_k=None,
        top_p=None,
        do_sample=None,
        generation_kwargs=generation_kwargs,
    )
    return generator.generate(show_progress=show_progress)
