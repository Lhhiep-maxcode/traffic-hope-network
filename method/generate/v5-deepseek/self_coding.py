import inspect
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm


def _model_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _input_ids(tokenizer, text):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"][0].detach().cpu().long()


def _substring_token_indices(tokenizer, text, substring):
    start_char = text.find(substring)
    if start_char < 0:
        raise ValueError("privileged_context was not found in the full prompt.")
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
        prefix_ids = _input_ids(tokenizer, text[:start_char])
        context_ids = _input_ids(tokenizer, text[:end_char])
        indices = list(range(prefix_ids.numel(), context_ids.numel()))

    if not indices:
        raise ValueError("privileged_context did not map to any tokens.")
    return indices


class TokenByTokenGenerator:
    """Token generation with reusable prompt tensors and bounded KV rollback.

    ``reuse_kv_cache=False`` restores full-prefill rollback for comparison.
    Cache reuse preserves sampling/repair rules, but floating-point results of
    cached decoding and full prefill need not be bit-identical on every backend.

    ``decode_tokens=False`` skips per-token decoding: ``step()`` returns None
    for token text. Final output text is always decoded by ``generate()``.
    """

    def __init__(
        self,
        model,
        tokenizer,
        prompt,
        privileged_context=None,
        temperature=0.7,
        top_k=0,
        top_p=1.0,
        do_sample=True,
        seed=None,
        reuse_kv_cache=True,
        decode_tokens=True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.privileged_context = privileged_context
        self.reuse_kv_cache = bool(reuse_kv_cache)
        self.decode_tokens = bool(decode_tokens)
        raw_full_prompt = prompt + (privileged_context or "")
        self.full_prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": raw_full_prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample
        self.seed = None if seed is None else int(seed)
        self._rng = None

        self.attention_mask = None
        self.past_key_values = None
        self.next_logits = None
        self.cache_length = 0
        self.generated_ids = []
        self.finished = False
        self._prompt_ids = None
        self._tokenized_prompt = None
        self._mask_buffer = None
        self._logits_history = {}
        self._prefill_kwargs = None

    def _prompt_token_ids(self):
        if self._tokenized_prompt != self.full_prompt:
            self._prompt_ids = _input_ids(self.tokenizer, self.full_prompt)
            self._tokenized_prompt = self.full_prompt
        return self._prompt_ids

    def _mask(self, length):
        device = torch.device(_model_device(self.model))
        buffer = self._mask_buffer
        if buffer is None or buffer.device != device or buffer.shape[1] < length:
            capacity = max(length, 2 * buffer.shape[1] if buffer is not None else 0)
            self._mask_buffer = torch.ones((1, capacity), dtype=torch.long, device=device)
        return self._mask_buffer[:, :length]

    def _last_logits_kwargs(self):
        # Only opt in when the model explicitly declares the argument. Custom
        # models accepting **kwargs do not necessarily implement this feature.
        if self._prefill_kwargs is None:
            try:
                parameters = inspect.signature(self.model.forward).parameters
            except (AttributeError, TypeError, ValueError):
                parameters = {}
            self._prefill_kwargs = {}
            for name in ("logits_to_keep", "num_logits_to_keep"):
                if name in parameters:
                    self._prefill_kwargs = {name: 1}
                    break
        return self._prefill_kwargs

    def _remember_logits(self):
        # Keep only the small backtracking window, never whole cache snapshots.
        position = len(self.generated_ids)
        self._logits_history[position] = self.next_logits
        oldest = position - getattr(self, "fix_backtrack", 0)
        for index in list(self._logits_history):
            if index < oldest or index > position:
                del self._logits_history[index]

    def _crop_cache(self, length):
        """Crop full-attention caches; leave unfamiliar cache layouts alone."""
        cache = self.past_key_values
        cache_type = type(cache)
        if (
            cache_type.__module__ == "transformers.cache_utils"
            and cache_type.__name__ == "DynamicCache"
        ):
            layers = getattr(cache, "layers", None)
            if layers is not None:
                # Sliding/linear/quantized layers may have discarded old states.
                if not layers or any(
                    type(layer).__module__ != "transformers.cache_utils"
                    or type(layer).__name__ != "DynamicLayer"
                    or layer.get_seq_length() < length
                    or layer.get_seq_length() not in (self.cache_length, self.cache_length + 1)
                    for layer in layers
                ):
                    return False
            elif not getattr(cache, "key_cache", None) or any(
                key.ndim != 4
                or key.shape[-2] < length
                or key.shape[-2] not in (self.cache_length, self.cache_length + 1)
                for key in cache.key_cache
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
                and value.shape[-2] in (self.cache_length, self.cache_length + 1)
                for value in layer
            )
            for layer in cache
        ):
            self.past_key_values = tuple(
                tuple(value[..., :length, :] for value in layer) for layer in cache
            )
            return True
        return False

    @torch.no_grad()
    def _restore_prefix(self, generated_ids):
        """Reuse the common prefix and evaluate only its missing suffix.

        Unsupported caches and training-mode models retain the full-prefill
        path. Sampling and repair decisions are independent of this fast path.
        """
        generated_ids = [int(token_id) for token_id in generated_ids]
        if (
            not getattr(self, "reuse_kv_cache", False)
            or getattr(self.model, "training", False)
            or self.past_key_values is None
            or self._tokenized_prompt != self.full_prompt
        ):
            return self.start(generated_ids)
        rope_scaling = getattr(getattr(self.model, "config", None), "rope_scaling", None)
        if rope_scaling and rope_scaling.get("rope_type", rope_scaling.get("type")) in {
            "dynamic", "longrope",
        }:
            # These schemes can change earlier keys when sequence length changes.
            return self.start(generated_ids)

        common = 0
        for old, new in zip(self.generated_ids, generated_ids):
            if old != new:
                break
            common += 1
        prompt_length = self._prompt_ids.numel()
        logits = self._logits_history.get(common)
        if common == len(generated_ids) and logits is None:
            if common == 0:
                return self.start(generated_ids)
            common -= 1  # Recompute just the final token if its logits expired.
        if not self._crop_cache(prompt_length + common):
            return self.start(generated_ids)

        suffix = generated_ids[common:]
        if suffix:
            input_ids = torch.tensor(
                [suffix], dtype=torch.long, device=_model_device(self.model)
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=self._mask(prompt_length + len(generated_ids)),
                past_key_values=self.past_key_values,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
                **self._last_logits_kwargs(),
            )
            self.past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].clone()
        self.next_logits = logits
        self.cache_length = prompt_length + len(generated_ids)
        self.attention_mask = self._mask(self.cache_length)
        self.generated_ids = generated_ids
        self.finished = bool(
            generated_ids
            and self.tokenizer.eos_token_id is not None
            and generated_ids[-1] == self.tokenizer.eos_token_id
        )
        # Entries after the common prefix may belong to a discarded branch.
        self._logits_history = {
            index: value for index, value in self._logits_history.items()
            if index <= common
        }
        self._remember_logits()
        return prompt_length

    @torch.no_grad()
    def start(self, generated_ids=None):
        """Prefill prompt and rebuild the KV cache at an accepted prefix."""
        generated_ids = [int(token_id) for token_id in (generated_ids or [])]
        prompt_ids = self._prompt_token_ids()
        prefix_ids = torch.tensor(generated_ids, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, prefix_ids]).unsqueeze(0)
        input_ids = input_ids.to(_model_device(self.model))
        attention_mask = self._mask(input_ids.shape[1])
        # Release obsolete cache/logits before allocating a full replacement.
        self.past_key_values = None
        self.next_logits = None
        self._logits_history = {}

        outputs = self.model(
            input_ids=input_ids,
            # An unpadded full prefill can use SDPA's causal fast path without
            # materializing a quadratic mask. Eager mode keeps its usual mask.
            attention_mask=({'full_attention': None}
                            if getattr(self.model, '_v5_selective_attention', False)
                            else attention_mask),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
            **self._last_logits_kwargs(),
        )

        self.attention_mask = attention_mask
        self.past_key_values = outputs.past_key_values
        self.next_logits = outputs.logits[:, -1, :].clone()
        self.cache_length = input_ids.shape[1]
        self.generated_ids = generated_ids
        self.finished = bool(
            generated_ids
            and self.tokenizer.eos_token_id is not None
            and generated_ids[-1] == self.tokenizer.eos_token_id
        )
        self._remember_logits()
        return prompt_ids.numel()

    def _reset_rng(self):
        device = torch.device(_model_device(self.model))
        self._rng = torch.Generator(device=device)
        if self.seed is None:
            self._rng.seed()
        else:
            self._rng.manual_seed(self.seed)

    def _sample_token(self, logits):
        if not self.do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)

        if self.temperature <= 0:
            raise ValueError("temperature must be positive when sampling.")
        logits = logits.float() / self.temperature

        if self.top_k > 0:
            k = min(self.top_k, logits.shape[-1])
            threshold = torch.topk(logits, k=k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            remove_mask = cumulative_probs > self.top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)

        if self._rng is None:
            self._reset_rng()
        return torch.multinomial(
            torch.softmax(logits, dim=-1), num_samples=1, generator=self._rng
        )

    @torch.no_grad()
    def _forward_token(self, token_id, output_attentions=False):
        token = torch.tensor(
            [[int(token_id)]],
            dtype=torch.long,
            device=_model_device(self.model),
        )
        attention_mask = self._mask(self.cache_length + 1)
        return self.model(
            input_ids=token,
            # Single-token decode has no future keys or padding in this cache.
            attention_mask=({'full_attention': None}
                            if getattr(self.model, '_v5_selective_attention', False)
                            else attention_mask),
            past_key_values=self.past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
            return_dict=True,
        )

    def _accept(self, token_id, outputs):
        token_id = int(token_id)
        self.past_key_values = outputs.past_key_values
        self.next_logits = outputs.logits[:, -1, :]
        self.cache_length += 1
        self.attention_mask = self._mask(self.cache_length)
        self.generated_ids.append(token_id)
        self._remember_logits()
        self.finished = (
            self.tokenizer.eos_token_id is not None
            and token_id == self.tokenizer.eos_token_id
        )

    def _decode_token(self, token_id):
        if not self.decode_tokens:
            return None
        return self.tokenizer.decode([token_id], skip_special_tokens=False)

    def step(self):
        """Generate and accept exactly one token."""
        if self.next_logits is None:
            raise RuntimeError("Call start() before step().")
        if self.finished:
            return None, None, True

        token_id = self._sample_token(self.next_logits).item()
        outputs = self._forward_token(token_id)
        self._accept(token_id, outputs)
        token_text = self._decode_token(token_id)
        return token_id, token_text, self.finished

    def generate(self, max_new_tokens, show_progress=True, description="Generating"):
        self._reset_rng()
        self.start()
        progress_bar = tqdm(
            total=max_new_tokens,
            desc=description,
            unit="token",
            disable=not show_progress,
        )
        try:
            while not self.finished and len(self.generated_ids) < max_new_tokens:
                self.step()
                progress_bar.update(1)
        finally:
            progress_bar.close()

        token_ids = list(self.generated_ids)
        return {
            "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
            "token_ids": token_ids,
        }

    def get_generated_text(self, skip_special_tokens=True):
        return self.tokenizer.decode(
            self.generated_ids,
            skip_special_tokens=skip_special_tokens,
        )


class CustomGenerator(TokenByTokenGenerator):
    """Repair leakage, optionally regenerating the backtracked prefix cleanly.

    ``debug_fix_infinite_loop=True`` (the default) prevents repair from ending
    before the backtracked span is replaced and suppresses repeated backtracking
    at the same span. With the other debug flags off, setting it to False restores
    the original behavior: a single safe candidate ends repair.
    The original ``max_repair_steps`` and ``max_attempts`` limits remain active.

    With ``debug_clean_backtrack=True``, positions from ``start_index`` up to
    but excluding ``detected_index`` are sampled only from the clean branch.
    Normal attention/JSD comparison resumes at ``detected_index``. Forced
    clean steps count toward ``max_repair_steps``, but not the safe streak.
    This option is independent of ``debug_fix_infinite_loop``.

    With ``debug_wait_safe_window=True``, all repair tokens come from the clean
    branch until ``max(1, (window_size - 1) // 2)`` consecutive privileged
    candidate checks pass (attention or JSD). A failed check resets the streak.
    The token completing the streak still comes from clean; the next step resumes
    normal privileged generation. The anti-loop span guard still applies when
    enabled, and forced clean backtrack steps never count as passing checks.

    ``detector_config`` accepts the full, model-keyed configuration already
    loaded into memory. When supplied, ``detector_config_path`` is not read.
    """

    def __init__(
        self,
        model,
        tokenizer,
        detector_config_path,
        model_key,
        max_new_tokens=512,
        fix_comparison_method="attention_score",
        fix_js_divergence_threshold=0.1,
        max_repair_steps=128,
        max_repair_cycles_per_span=2,
        seed=None,
        debug_clean_backtrack=False,
        debug_fix_infinite_loop=True,
        debug_wait_safe_window=False,
        detector_config=None,
        **kwargs,
    ):
        super().__init__(model=model, tokenizer=tokenizer, seed=seed, **kwargs)

        payload = detector_config
        if payload is None:
            path = Path(detector_config_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.detector = payload[model_key]
        self.heads = self.detector["heads"]
        self.threshold = float(self.detector["threshold"])
        self.window_size = int(self.detector["window_size"])
        self.aggregation = self.detector.get("aggregation", "weighted")
        self._head_weights = [max(float(head.get("score", 1.0)), 0.0) for head in self.heads]
        self._weight_sum = sum(self._head_weights)

        if not self.privileged_context:
            raise ValueError("privileged_context is required.")
        if fix_comparison_method not in {"attention_score", "js_divergence"}:
            raise ValueError(
                "fix_comparison_method must be 'attention_score' or 'js_divergence'."
            )
        if max_repair_steps < 1:
            raise ValueError("max_repair_steps must be at least 1.")
        if max_repair_cycles_per_span < 1:
            raise ValueError("max_repair_cycles_per_span must be at least 1.")

        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        self.fix_backtrack = max((self.window_size - 1) // 2, 0)
        self.fix_comparison_method = fix_comparison_method
        self.fix_js_divergence_threshold = float(fix_js_divergence_threshold)
        self.max_repair_steps = int(max_repair_steps)
        self.max_repair_cycles_per_span = int(max_repair_cycles_per_span)
        self.debug_clean_backtrack = bool(debug_clean_backtrack)
        self.debug_fix_infinite_loop = bool(debug_fix_infinite_loop)
        self.debug_wait_safe_window = bool(debug_wait_safe_window)
        if self.debug_wait_safe_window:
            min_repair_steps = max(1, self.fix_backtrack)
            if self.debug_clean_backtrack:
                min_repair_steps += self.fix_backtrack
            if self.debug_fix_infinite_loop:
                min_repair_steps = max(min_repair_steps, self.fix_backtrack + 1)
            if self.max_repair_steps < min_repair_steps:
                raise ValueError(
                    "max_repair_steps must be at least "
                    f"{min_repair_steps} with debug_wait_safe_window enabled "
                    "to cover backtracking and the required safe checks."
                )
        elif (
            (self.debug_fix_infinite_loop or self.debug_clean_backtrack)
            and self.max_repair_steps < self.fix_backtrack + 1
        ):
            raise ValueError(
                "max_repair_steps must be at least fix_backtrack + 1 "
                "so repair can move past the backtracked span."
            )
        self.privileged_context_token_indices = _substring_token_indices(
            tokenizer,
            self.full_prompt,
            self.privileged_context,
        )

        self.clean_generator = TokenByTokenGenerator(
            model=model,
            tokenizer=tokenizer,
            prompt=self.prompt,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            reuse_kv_cache=self.reuse_kv_cache,
            decode_tokens=self.decode_tokens,
        )
        self.clean_generator.fix_backtrack = self.fix_backtrack
        self.repairing = False
        self.repair_steps = 0
        self.repair_safe_steps = 0
        self.repair_required_safe_steps = 1
        self.repair_cycles_by_start = {}
        self.attention_score = None
        self.repair_events = []
        self.attempts = 0
        self.max_attempts = max(
            1_000,
            self.max_new_tokens * (self.max_repair_steps + self.fix_backtrack + 2),
        )

    def _attention_score(self, attentions):
        if attentions is None:
            raise RuntimeError(
                "The model did not return attentions. Load it with "
                "attn_implementation='eager'."
            )
        scores = [None] * len(self.heads)
        weights = self._head_weights
        token_indices = tuple(self.privileged_context_token_indices)
        if getattr(self, "_attention_indices_key", None) != token_indices:
            self._attention_indices_key = token_indices
            self._attention_indices_by_device = {}
        scores_by_device = {}

        for position, record in enumerate(self.heads):
            attention = attentions[int(record["layer"])][0, int(record["head"]), -1]
            device = attention.device
            if device not in self._attention_indices_by_device:
                self._attention_indices_by_device[device] = torch.tensor(
                    token_indices, dtype=torch.long, device=device
                )
            indices = self._attention_indices_by_device[device]
            scores_by_device.setdefault(device, []).append(
                (position, attention.index_select(0, indices).sum())
            )

        # One host transfer per device instead of a blocking .item() per head.
        # Keep each original sum's dtype and Python aggregation order intact.
        for entries in scores_by_device.values():
            values = torch.stack([value for _, value in entries]).tolist()
            for (position, _), value in zip(entries, values):
                scores[position] = float(value)

        if self.aggregation == "mean" or not self._weight_sum:
            return sum(scores) / len(scores)
        return sum(score * weight for score, weight in zip(scores, weights)) / self._weight_sum

    def check(self, attention_score):
        return attention_score >= self.threshold

    @staticmethod
    def jensen_shannon_score(privileged_logits, clean_logits):
        log_p = torch.log_softmax(privileged_logits.float(), dim=-1)
        log_q = torch.log_softmax(clean_logits.float(), dim=-1)
        log_m = torch.logaddexp(log_p, log_q) - torch.log(
            torch.tensor(2.0, device=log_p.device)
        )
        js = 0.5 * (
            (log_p.exp() * (log_p - log_m)).sum()
            + (log_q.exp() * (log_q - log_m)).sum()
        )
        return max(float(js.item()), 0.0)

    def _start_repair(self, candidate_id, detector_score):
        detected_index = len(self.generated_ids)
        nominal_start_index = max(0, detected_index - self.fix_backtrack)
        cycle_count = (
            self.repair_cycles_by_start.get(nominal_start_index, 0) + 1
        )
        self.repair_cycles_by_start[nominal_start_index] = cycle_count

        # Repeatedly backtracking to the same prefix can otherwise make the
        # outer generation loop oscillate forever. After a small number of
        # retries, keep the accepted prefix and repair only the new candidate
        # when the anti-loop fix is enabled.
        backtrack_suppressed = (
            self.debug_fix_infinite_loop
            and cycle_count > self.max_repair_cycles_per_span
        )
        start_index = (
            detected_index if backtrack_suppressed else nominal_start_index
        )
        removed_ids = self.generated_ids[start_index:]

        # Both branches resume at exactly the same accepted output prefix.
        self._restore_prefix(self.generated_ids[:start_index])
        TokenByTokenGenerator._restore_prefix(self.clean_generator, self.generated_ids)
        self.repairing = True
        self.repair_steps = 0
        self.repair_safe_steps = 0
        if self.debug_wait_safe_window:
            self.repair_required_safe_steps = max(1, self.fix_backtrack)
        else:
            self.repair_required_safe_steps = (
                detected_index - start_index + 1 if self.debug_fix_infinite_loop else 1
            )
        self.repair_events.append(
            {
                "detected_index": detected_index,
                "start_index": start_index,
                "nominal_start_index": nominal_start_index,
                "cycle_count": cycle_count,
                "backtrack_suppressed": backtrack_suppressed,
                "debug_fix_infinite_loop": self.debug_fix_infinite_loop,
                "debug_clean_backtrack": self.debug_clean_backtrack,
                "debug_wait_safe_window": self.debug_wait_safe_window,
                "required_safe_steps": self.repair_required_safe_steps,
                "detector_score": detector_score,
                "removed_token_ids": [*removed_ids, int(candidate_id)],
                "replacement_token_ids": [],
                "clean_backtrack_token_ids": [],
                "steps": [],
            }
        )

    def _accept_on_both_branches(self, token_id, privileged_outputs=None):
        if privileged_outputs is None:
            privileged_outputs = self._forward_token(token_id)
        self._accept(token_id, privileged_outputs)

        clean_outputs = self.clean_generator._forward_token(token_id)
        self.clean_generator._accept(token_id, clean_outputs)

    def _repair_step(self):
        event = self.repair_events[-1]
        force_clean_backtrack = (
            self.debug_clean_backtrack
            and len(self.generated_ids) < event["detected_index"]
        )
        privileged_outputs = None
        comparison_score = None

        if force_clean_backtrack:
            # Skip privileged sampling and comparison until the original
            # detection position. Both caches still consume the clean token.
            selected_id = self.clean_generator._sample_token(
                self.clean_generator.next_logits
            ).item()
            candidate_needs_repair = None
            self.attention_score = None
        else:
            privileged_id = self._sample_token(self.next_logits).item()

            if self.fix_comparison_method == "js_divergence":
                clean_id = self.clean_generator._sample_token(
                    self.clean_generator.next_logits
                ).item()
                comparison_score = self.jensen_shannon_score(
                    self.next_logits,
                    self.clean_generator.next_logits,
                )
                candidate_needs_repair = (
                    comparison_score > self.fix_js_divergence_threshold
                )
                selected_id = (
                    clean_id
                    if candidate_needs_repair or self.debug_wait_safe_window
                    else privileged_id
                )
            else:
                privileged_outputs = self._forward_token(
                    privileged_id,
                    output_attentions=True,
                )
                comparison_score = self._attention_score(privileged_outputs.attentions)
                self.attention_score = comparison_score
                candidate_needs_repair = self.check(comparison_score)
                selected_id = privileged_id

                if candidate_needs_repair or self.debug_wait_safe_window:
                    selected_id = self.clean_generator._sample_token(
                        self.clean_generator.next_logits
                    ).item()
                    if (
                        selected_id != privileged_id
                        or not getattr(self, "reuse_kv_cache", False)
                        or getattr(self.model, "training", False)
                    ):
                        # Discard the speculative token without redoing prefill.
                        privileged_outputs = None
                        self._restore_prefix(self.generated_ids)

        self._accept_on_both_branches(selected_id, privileged_outputs)
        event["replacement_token_ids"].append(selected_id)
        if force_clean_backtrack:
            event["clean_backtrack_token_ids"].append(selected_id)
        self.repair_steps += 1

        if force_clean_backtrack or candidate_needs_repair:
            self.repair_safe_steps = 0
        else:
            self.repair_safe_steps += 1
        repair_continues = (
            force_clean_backtrack
            or candidate_needs_repair
            or self.repair_safe_steps < self.repair_required_safe_steps
            # A shorter safe window must not undo the anti-loop progress guard.
            or (
                self.debug_fix_infinite_loop
                and len(self.generated_ids) <= event["detected_index"]
            )
        )
        event["steps"].append(
            {
                "token_index": len(self.generated_ids) - 1,
                "selected_token_id": selected_id,
                "selected_branch": (
                    "clean"
                    if force_clean_backtrack
                    or candidate_needs_repair
                    or self.debug_wait_safe_window
                    else "privileged"
                ),
                "comparison_score": comparison_score,
                "passed_check": (
                    None if force_clean_backtrack else not candidate_needs_repair
                ),
                "safe_steps": self.repair_safe_steps,
            }
        )

        if not repair_continues or self.finished:
            self.repairing = False
            self.repair_steps = 0
            self.repair_safe_steps = 0
        elif self.repair_steps >= self.max_repair_steps:
            raise RuntimeError("Repair did not converge within max_repair_steps.")

        token_text = self._decode_token(selected_id)
        return selected_id, token_text, self.finished

    def step(self):
        if self.next_logits is None:
            raise RuntimeError("Call start() before step().")
        if self.finished or len(self.generated_ids) >= self.max_new_tokens:
            self.finished = True
            return None, None, True
        self.attempts += 1
        if self.attempts > self.max_attempts:
            raise RuntimeError("Generation stopped after too many repair attempts.")
        if self.repairing:
            return self._repair_step()

        candidate_id = self._sample_token(self.next_logits).item()
        outputs = self._forward_token(candidate_id, output_attentions=True)
        return self.consume_candidate(candidate_id, outputs)

    def consume_candidate(self, candidate_id, outputs):
        """Apply the unchanged detector/repair rules to a scalar or batched forward."""
        self.attention_score = self._attention_score(outputs.attentions)

        if not self.check(self.attention_score):
            self._accept(candidate_id, outputs)
            token_text = self._decode_token(candidate_id)
            return candidate_id, token_text, self.finished

        del outputs
        self._start_repair(candidate_id, self.attention_score)
        return self._repair_step()

    def begin(self):
        """Initialize one independent sample for scalar or scheduled generation."""
        self.repairing = False
        self.repair_steps = 0
        self.repair_safe_steps = 0
        self.repair_required_safe_steps = 1
        self.repair_cycles_by_start = {}
        self.attention_score = None
        self.repair_events = []
        self.attempts = 0
        self._reset_rng()
        # Repair uses the same per-sample stream; other samples cannot shift it.
        self.clean_generator._rng = self._rng
        self.start()
        # A new run may follow a model-weight update; do not reuse clean state
        # from a previous run. It will be initialized at the first repair.
        self.clean_generator.past_key_values = None
        self.clean_generator._logits_history = {}

    def generate(self, include_unfixed=True, show_progress=True):
        self.begin()
        progress_bar = tqdm(
            total=self.max_new_tokens,
            desc="Generating fixed",
            unit="token",
            disable=not show_progress,
        )
        try:
            while not self.finished and len(self.generated_ids) < self.max_new_tokens:
                self.step()
                # Backtracking can reduce the number of accepted tokens.
                progress_bar.update(len(self.generated_ids) - progress_bar.n)
        finally:
            progress_bar.close()

        return self.finish(include_unfixed=include_unfixed, show_progress=show_progress)

    def finish(self, include_unfixed=False, show_progress=False):
        """Decode a completed sample and optionally run its baseline."""
        fixed_ids = list(self.generated_ids)
        fixed = {
            "text": self.tokenizer.decode(fixed_ids, skip_special_tokens=True),
            "token_ids": fixed_ids,
        }

        unfixed = None
        if include_unfixed:
            baseline = TokenByTokenGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=self.prompt,
                privileged_context=self.privileged_context,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
                seed=self.seed,
                reuse_kv_cache=self.reuse_kv_cache,
                decode_tokens=self.decode_tokens,
            )
            unfixed = baseline.generate(
                self.max_new_tokens,
                show_progress=show_progress,
                description="Generating unfixed",
            )

        return {
            "text": fixed["text"],
            "token_ids": fixed["token_ids"],
            "fixed": fixed,
            "unfixed": unfixed,
            "repair_events": list(self.repair_events),
        }
