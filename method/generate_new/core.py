from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
from tqdm.auto import tqdm


def model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def encode_ids(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"][0].detach().cpu().long()


def render_user_prompt(tokenizer, text: str, enable_thinking: bool = True) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def substring_token_indices(tokenizer, text: str, substring: str) -> list[int]:
    start_char = text.find(substring)
    if start_char < 0:
        raise ValueError("privileged_context was not found in the rendered prompt.")
    end_char = start_char + len(substring)

    try:
        offsets = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )["offset_mapping"]
        indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start_char and start < end_char
        ]
    except (KeyError, NotImplementedError, TypeError, ValueError):
        prefix_len = encode_ids(tokenizer, text[:start_char]).numel()
        context_end = encode_ids(tokenizer, text[:end_char]).numel()
        indices = list(range(prefix_len, context_end))

    if not indices:
        raise ValueError("privileged_context did not map to any tokens.")
    return indices


def sample_from_logits(
    logits: torch.Tensor,
    *,
    rng: torch.Generator,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    if not do_sample:
        return int(torch.argmax(logits, dim=-1).item())
    if temperature <= 0:
        raise ValueError("temperature must be positive when do_sample=True.")

    scores = logits.float() / temperature
    if top_k > 0:
        k = min(top_k, scores.shape[-1])
        threshold = torch.topk(scores, k=k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(1, sorted_indices, sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=rng).item())


@dataclass
class DecodeSettings:
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    enable_thinking: bool = True
    seed: int | None = None


@dataclass
class RepairSettings:
    pass


@dataclass
class AttentionLeakageDetector:
    heads: list[dict]
    threshold: float
    window_size: int
    aggregation: str = "weighted"
    weights: list[float] = field(default_factory=list)
    context_token_indices: list[int] = field(default_factory=list)
    _indices_by_device: dict = field(default_factory=dict, init=False)

    @classmethod
    def from_file(cls, path: str | Path, model_key: str):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if model_key in payload:
            config = payload[model_key]
        elif "heads" in payload:
            config = payload
        else:
            raise KeyError(f"Detector config does not contain model key {model_key!r}.")
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: dict):
        heads = list(config["heads"])
        if not heads:
            raise ValueError("Detector config must contain at least one head.")
        weights = [max(float(head.get("score", 1.0)), 0.0) for head in heads]
        return cls(
            heads=heads,
            threshold=float(config["threshold"]),
            window_size=int(config["window_size"]),
            aggregation=config.get("aggregation", "weighted"),
            weights=weights,
        )

    def with_context(self, context_token_indices: list[int]):
        return AttentionLeakageDetector(
            heads=self.heads,
            threshold=self.threshold,
            window_size=self.window_size,
            aggregation=self.aggregation,
            weights=self.weights,
            context_token_indices=list(context_token_indices),
        )

    def score(self, attentions) -> float:
        if attentions is None:
            raise RuntimeError(
                "The model did not return attentions. Load it with eager attention."
            )
        if not self.context_token_indices:
            raise RuntimeError("The detector does not know privileged-context tokens.")

        scores = [0.0] * len(self.heads)
        scores_by_device = {}
        for position, head in enumerate(self.heads):
            layer_index = int(head["layer"])
            head_index = int(head["head"])
            layer_attention = attentions[layer_index]
            if layer_attention is None or layer_attention.shape[1] <= head_index:
                raise RuntimeError(
                    f"Missing attention weights for layer {layer_index}, head {head_index}."
                )

            attention_row = layer_attention[0, head_index, -1]
            device = attention_row.device
            if device not in self._indices_by_device:
                self._indices_by_device[device] = torch.tensor(
                    self.context_token_indices,
                    dtype=torch.long,
                    device=device,
                )
            indices = self._indices_by_device[device]
            scores_by_device.setdefault(device, []).append(
                (position, attention_row.index_select(0, indices).sum())
            )

        for entries in scores_by_device.values():
            values = torch.stack([value for _, value in entries]).tolist()
            for (position, _), value in zip(entries, values):
                scores[position] = float(value)

        weight_sum = sum(self.weights)
        if self.aggregation == "mean" or weight_sum <= 0:
            return sum(scores) / len(scores)
        return sum(score * weight for score, weight in zip(scores, self.weights)) / weight_sum


class DecodeBranch:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        user_prompt: str,
        settings: DecodeSettings,
        rng: torch.Generator | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.user_prompt = user_prompt
        self.settings = settings
        self.rendered_prompt = render_user_prompt(
            tokenizer,
            user_prompt,
            enable_thinking=settings.enable_thinking,
        )
        self.rng = rng
        self.generated_ids: list[int] = []
        self.past_key_values = None
        self.next_logits = None
        self.cache_length = 0
        self.finished = False
        self._prompt_ids = None
        self._mask_buffer = None
        self._forward_kwargs = None

    def prompt_ids(self):
        if self._prompt_ids is None:
            self._prompt_ids = encode_ids(self.tokenizer, self.rendered_prompt)
        return self._prompt_ids

    def mask(self, length: int):
        device = torch.device(model_device(self.model))
        if (
            self._mask_buffer is None
            or self._mask_buffer.device != device
            or self._mask_buffer.shape[1] < length
        ):
            capacity = max(length, 2 * self._mask_buffer.shape[1] if self._mask_buffer is not None else 0)
            self._mask_buffer = torch.ones((1, capacity), dtype=torch.long, device=device)
        return self._mask_buffer[:, :length]

    def last_logit_kwargs(self):
        if self._forward_kwargs is not None:
            return self._forward_kwargs
        try:
            parameters = inspect.signature(self.model.forward).parameters
        except (AttributeError, TypeError, ValueError):
            parameters = {}
        self._forward_kwargs = {}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in parameters:
                self._forward_kwargs = {name: 1}
                break
        return self._forward_kwargs

    def crop_cache(self, length: int) -> bool:
        cache = self.past_key_values
        cache_type = type(cache)
        if (
            cache_type.__module__ == "transformers.cache_utils"
            and cache_type.__name__ == "DynamicCache"
        ):
            layers = getattr(cache, "layers", None)
            if layers is not None:
                if not layers or any(
                    type(layer).__module__ != "transformers.cache_utils"
                    or type(layer).__name__ != "DynamicLayer"
                    or layer.get_seq_length() < length
                    for layer in layers
                ):
                    return False
            elif not getattr(cache, "key_cache", None) or any(
                key.ndim != 4 or key.shape[-2] < length for key in cache.key_cache
            ):
                return False
            if getattr(cache, "offloading", False):
                return False
            cache.crop(length)
            return True

        if isinstance(cache, (tuple, list)) and cache and all(
            isinstance(layer, (tuple, list))
            and len(layer) == 2
            and all(
                isinstance(value, torch.Tensor)
                and value.ndim == 4
                and value.shape[-2] >= length
                for value in layer
            )
            for layer in cache
        ):
            self.past_key_values = tuple(
                tuple(value[..., :length, :] for value in layer) for layer in cache
            )
            return True
        return False

    def make_rng(self):
        device = torch.device(model_device(self.model))
        rng = torch.Generator(device=device)
        if self.settings.seed is None:
            rng.seed()
        else:
            rng.manual_seed(int(self.settings.seed))
        return rng

    @torch.no_grad()
    def start(self, prefix_ids: list[int] | None = None):
        self.generated_ids = [int(token_id) for token_id in (prefix_ids or [])]
        prompt_ids = self.prompt_ids()
        suffix_ids = torch.tensor(self.generated_ids, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, suffix_ids]).unsqueeze(0)
        input_ids = input_ids.to(model_device(self.model))

        self.past_key_values = None
        self.next_logits = None
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=self.mask(input_ids.shape[1]),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            **self.last_logit_kwargs(),
        )
        self.past_key_values = outputs.past_key_values
        self.next_logits = outputs.logits[:, -1, :].clone()
        self.cache_length = input_ids.shape[1]
        self.finished = bool(
            self.generated_ids
            and self.tokenizer.eos_token_id is not None
            and self.generated_ids[-1] == self.tokenizer.eos_token_id
        )

    @torch.no_grad()
    def restore_prefix(self, prefix_ids: list[int], *, require_logits: bool = True):
        prefix_ids = [int(token_id) for token_id in prefix_ids]
        if self.past_key_values is None or self._prompt_ids is None:
            self.start(prefix_ids)
            return

        common = 0
        for old, new in zip(self.generated_ids, prefix_ids):
            if old != new:
                break
            common += 1

        prompt_length = self.prompt_ids().numel()
        cropped = common < len(self.generated_ids)
        suffix = prefix_ids[common:]
        if require_logits and cropped and not suffix:
            self.start(prefix_ids)
            return

        if cropped:
            if not self.crop_cache(prompt_length + common):
                self.start(prefix_ids)
                return
            self.generated_ids = self.generated_ids[:common]
            self.cache_length = prompt_length + common
            self.next_logits = None

        if suffix:
            input_ids = torch.tensor(
                [suffix],
                dtype=torch.long,
                device=model_device(self.model),
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=self.mask(prompt_length + len(prefix_ids)),
                past_key_values=self.past_key_values,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
                **self.last_logit_kwargs(),
            )
            self.past_key_values = outputs.past_key_values
            self.next_logits = outputs.logits[:, -1, :].clone()

        self.generated_ids = prefix_ids
        self.cache_length = prompt_length + len(prefix_ids)
        self.finished = bool(
            self.generated_ids
            and self.tokenizer.eos_token_id is not None
            and self.generated_ids[-1] == self.tokenizer.eos_token_id
        )

    def sample(self) -> int:
        if self.next_logits is None:
            raise RuntimeError("Call start() before sampling.")
        if self.rng is None:
            self.rng = self.make_rng()
        return sample_from_logits(
            self.next_logits,
            rng=self.rng,
            do_sample=self.settings.do_sample,
            temperature=self.settings.temperature,
            top_k=self.settings.top_k,
            top_p=self.settings.top_p,
        )

    @torch.no_grad()
    def forward_token(self, token_id: int, *, output_attentions: bool = False):
        token = torch.tensor(
            [[int(token_id)]],
            dtype=torch.long,
            device=model_device(self.model),
        )
        return self.model(
            input_ids=token,
            attention_mask=self.mask(self.cache_length + 1),
            past_key_values=self.past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
            **self.last_logit_kwargs(),
        )

    def accept(self, token_id: int, outputs):
        token_id = int(token_id)
        self.past_key_values = outputs.past_key_values
        self.next_logits = outputs.logits[:, -1, :]
        self.cache_length += 1
        self.generated_ids.append(token_id)
        self.finished = (
            self.tokenizer.eos_token_id is not None
            and token_id == self.tokenizer.eos_token_id
        )

    def step(self):
        token_id = self.sample()
        outputs = self.forward_token(token_id)
        self.accept(token_id, outputs)
        return token_id

    def generate(self, show_progress: bool = False):
        self.rng = self.make_rng()
        self.start()
        progress = tqdm(
            total=self.settings.max_new_tokens,
            desc="Generating",
            unit="token",
            disable=not show_progress,
        )
        try:
            while (
                not self.finished
                and len(self.generated_ids) < self.settings.max_new_tokens
            ):
                self.step()
                progress.update(1)
        finally:
            progress.close()
        return {
            "text": self.text(),
            "token_ids": list(self.generated_ids),
        }

    def text(self, skip_special_tokens: bool = True):
        return self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=skip_special_tokens,
        )


class RepairState(Enum):
    SAFE = 1
    TEMP_SAFE = 2


class LeakageSafeGenerator:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        clean_prompt: str = "",
        privileged_context: str = "",
        privileged_prompt: str | None = None,
        detector: AttentionLeakageDetector,
        decode: DecodeSettings | None = None,
        repair: RepairSettings | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.clean_prompt = clean_prompt
        self.privileged_context = privileged_context
        self.privileged_prompt = privileged_prompt or clean_prompt + privileged_context
        self.decode = decode or DecodeSettings()
        self.repair = repair or RepairSettings()
        self.detector = detector
        self.backtrack_tokens = max((self.detector.window_size - 1) // 2, 0)
        self.repair_events: list[dict] = []
        self.last_detector_score = None
        self.generated_tokens_state: list[RepairState] = []

    def begin(self):
        self.privileged = DecodeBranch(
            model=self.model,
            tokenizer=self.tokenizer,
            user_prompt=self.privileged_prompt,
            settings=self.decode,
        )
        context_indices = substring_token_indices(
            self.tokenizer,
            self.privileged.rendered_prompt,
            self.privileged_context,
        )
        self.detector = self.detector.with_context(context_indices)
        self.clean = DecodeBranch(
            model=self.model,
            tokenizer=self.tokenizer,
            user_prompt=self.clean_prompt,
            settings=self.decode,
        )
        rng = self.privileged.make_rng()
        self.privileged.rng = rng
        self.clean.rng = rng
        self.privileged.start()
        self.clean.past_key_values = None
        self.clean.next_logits = None
        self.repair_events = []
        self.last_detector_score = None
        self.generated_tokens_state = []

    def output_ids(self):
        return self.privileged.generated_ids

    def finished(self):
        return (
            self.privileged.finished
            or len(self.output_ids()) >= self.decode.max_new_tokens
        )

    def detected_leakage(self, score: float) -> bool:
        return score >= self.detector.threshold

    def start_repair(self, candidate_id: int, detector_score: float):
        detected_index = len(self.output_ids())
        first_sus_index = next((
            i for i, state in enumerate(self.generated_tokens_state) 
            if state == RepairState.TEMP_SAFE
        ), detected_index)
        if detected_index - first_sus_index > self.backtrack_tokens:
            raise RuntimeError(
                "Started repair index is out of backtrack window"
            )
        start_index = first_sus_index
        removed_ids = self.output_ids()[start_index:]

        prefix = self.output_ids()[:start_index]
        self.generated_tokens_state = self.generated_tokens_state[:start_index]
        self.privileged.restore_prefix(prefix, require_logits=False)
        self.clean.restore_prefix(prefix, require_logits=True)
        self.repair_events.append(
            {
                "detected_index": detected_index,
                "start_index": start_index,
                "detector_score": detector_score,
                "removed_token_ids": [*removed_ids, int(candidate_id)],
                "removed_token_text": self.tokenizer.decode(removed_ids + [int(candidate_id)], skip_special_tokens=True),
                "replacement_token_ids": [],
                "steps": [],
            }
        )

    def accept_on_both_branches(self, token_id: int, privileged_outputs=None):
        if privileged_outputs is None:
            privileged_outputs = self.privileged.forward_token(token_id)
        self.privileged.accept(token_id, privileged_outputs)

        clean_outputs = self.clean.forward_token(token_id)
        self.clean.accept(token_id, clean_outputs)

    def repair_step(self):
        event = self.repair_events[-1]
        privileged_outputs = None

        selected_id = self.clean.sample()
        self.generated_tokens_state.append(RepairState.SAFE)
        self.last_detector_score = None
        
        self.accept_on_both_branches(selected_id, privileged_outputs)
        event["replacement_token_ids"].append(int(selected_id))
        event["replacement_token_text"] = self.tokenizer.decode(
            event["replacement_token_ids"],
            skip_special_tokens=True,
        )
        event["steps"].append(
            {
                "token_index": len(self.output_ids()) - 1,
                "selected_token_id": int(selected_id),
                "selected_branch": "clean",
            }
        )
        return int(selected_id)

    def step(self):
        if self.finished():
            self.privileged.finished = True
            return None

        candidate_id = self.privileged.sample()
        outputs = self.privileged.forward_token(candidate_id, output_attentions=True)
        score = self.detector.score(outputs.attentions)
        self.last_detector_score = score
        if not self.detected_leakage(score):
            self.privileged.accept(candidate_id, outputs)
            self.generated_tokens_state.append(RepairState.TEMP_SAFE)
            safe_index = len(self.generated_tokens_state) - self.backtrack_tokens - 1
            if safe_index >= 0:
                self.generated_tokens_state[safe_index] = RepairState.SAFE
            return int(candidate_id)

        self.start_repair(candidate_id, score)
        return self.repair_step()

    def generate(
        self, 
        clean_prompt: str, 
        privileged_prompt: str, 
        privileged_context: str, 
        include_unfixed: bool = False, 
        show_progress: bool = True
    ):
        self.clean_prompt = clean_prompt
        self.privileged_prompt = privileged_prompt
        self.privileged_context = privileged_context
        self.begin()
        progress = tqdm(
            total=self.decode.max_new_tokens,
            desc="Generating fixed",
            unit="token",
            disable=not show_progress,
        )
        try:
            while not self.finished():
                previous = progress.n
                self.step()
                progress.n = min(len(self.output_ids()), self.decode.max_new_tokens)
                if progress.n != previous:
                    progress.refresh()
        finally:
            progress.close()

        fixed_ids = list(self.output_ids())
        fixed = {
            "text": self.tokenizer.decode(fixed_ids, skip_special_tokens=True),
            "token_ids": fixed_ids,
        }
        unfixed = None
        if include_unfixed:
            branch = DecodeBranch(
                model=self.model,
                tokenizer=self.tokenizer,
                user_prompt=self.privileged_prompt,
                settings=self.decode,
            )
            unfixed = branch.generate(show_progress=show_progress)

        return {
            "text": fixed["text"],
            "token_ids": fixed["token_ids"],
            "fixed": fixed,
            "unfixed": unfixed,
            "repair_events": list(self.repair_events),
        }
