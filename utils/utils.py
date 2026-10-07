import json

import torch
from pathlib import Path
import matplotlib.pyplot as plt

def load_model_and_tokenizer(model, attn_implementation="eager", device_map="auto", dtype="float16", trust_remote_code=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)

    # Load the model with specified attention implementation and device map
    model = AutoModelForCausalLM.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation=attn_implementation
    )

    return model, tokenizer

def build_prompt(tokenizer, system_prompt=None, user_prompt=None, encode=True, add_generation_prompt=True, enable_thinking=True):
    system_prompt = system_prompt or "You are a helpful assistant."
    user_prompt = user_prompt or "Hi"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    return tokenizer.apply_chat_template(messages, tokenize=encode, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)

def find_leveraging_context(
    tokenizer,
    user_prompt,
    leveraging_context,
    system_prompt=None,
):
    rendered_prompt = build_prompt(
        tokenizer,
        system_prompt,
        user_prompt,
        encode=False,
    )
    start_char = rendered_prompt.index(leveraging_context)
    end_char = start_char + len(leveraging_context)
    encoded = tokenizer(rendered_prompt, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]

    start_token = next(i for i, (start, end) in enumerate(offsets) if end > start_char)
    end_token = next(
        (i for i, (start, end) in enumerate(offsets) if start >= end_char),
        len(offsets),
    )
    return start_token, end_token

def generate_and_concate(model, tokenizer, prompt: str, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else inputs.input_ids.shape[1]
    with torch.inference_mode():
        concated_ids = model.generate(**inputs, **kwargs)
    generated_ids = concated_ids[0][prompt_length:]  # Exclude the prompt tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return concated_ids[0], generated_text

def get_attention_weights(
    model,
    sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    device = model.device
    sequence_ids = sequence_ids.unsqueeze(0).to(device) if sequence_ids.dim() == 1 else sequence_ids.to(device)
    attention_mask = torch.ones_like(sequence_ids, device=device)

    with torch.inference_mode():
        outputs = model(
            input_ids=sequence_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    if outputs.attentions is None:
        raise RuntimeError(
            "The model did not return attentions. Try --attn-implementation eager "
            "and make sure this architecture supports output_attentions=True."
        )

    attentions = tuple(attn.detach().float().cpu() for attn in outputs.attentions)
    return attentions, outputs

def _token_text(tokenizer, sequence_ids, idx, max_chars=10):
    text = tokenizer.decode([int(sequence_ids[idx])], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "_")
    text = text.replace("$", r"\$")
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return f"{idx}:{text}"


def _token_labels(tokenizer, sequence_ids, indices, max_chars=10):
    return [_token_text(tokenizer, sequence_ids, idx, max_chars) for idx in indices]

def _token_ticks_and_labels(tokenizer, sequence_ids, indices, max_chars=10):
    tick_positions = []
    tick_labels = []

    if indices[0] > 0:
        tick_positions.append(-1)
        tick_labels.append("...")

    tick_positions.extend(range(len(indices)))
    tick_labels.extend(_token_labels(tokenizer, sequence_ids, indices, max_chars))

    if indices[-1] < len(sequence_ids) - 1:
        tick_positions.append(len(indices))
        tick_labels.append("...")

    return tick_positions, tick_labels


def _find_all(text, needle):
    start = 0
    while needle:
        idx = text.find(needle, start)
        if idx < 0:
            break
        yield idx
        start = idx + len(needle)


def _overlapping_token_positions(offsets, start_char, end_char):
    return [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start_char and start < end_char
    ]


def _leakage_output_indices(
    tokenizer,
    leakage_spans=None,
    prompt_len=None,
    full_text=None,
    response_text=None,
):
    if leakage_spans is None:
        return set()
    if isinstance(leakage_spans, str):
        leakage_spans = [leakage_spans]

    if full_text is not None:
        if prompt_len is None:
            raise ValueError("prompt_len is required when full_text is used.")
        encoded = tokenizer(full_text, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        output_indices = set()
        for span in leakage_spans:
            for start_char in _find_all(full_text, span):
                for token_idx in _overlapping_token_positions(offsets, start_char, start_char + len(span)):
                    if token_idx >= prompt_len:
                        output_indices.add(token_idx - prompt_len)
        return output_indices

    if response_text is not None:
        encoded = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoded["offset_mapping"]
        output_indices = set()
        for span in leakage_spans:
            for start_char in _find_all(response_text, span):
                output_indices.update(
                    _overlapping_token_positions(offsets, start_char, start_char + len(span))
                )
        return output_indices

    return set()


def _compact_token_ticks_and_labels(
    tokenizer,
    sequence_ids,
    label_indices,
    leakage_positions,
    context_tokens=8,
    edge_tokens=2,
    max_chars=10,
    leakage_max_chars=40,
):
    total = len(label_indices)
    leakage_positions = {pos for pos in leakage_positions if 0 <= pos < total}

    if leakage_positions:
        selected = set(range(min(edge_tokens, total)))
        selected.update(range(max(0, total - edge_tokens), total))
        for pos in leakage_positions:
            selected.update(range(max(0, pos - context_tokens), min(total, pos + context_tokens + 1)))
    else:
        max_labels = 80
        step = max(1, total // max_labels)
        selected = set(range(0, total, step))
        selected.update(range(max(0, total - edge_tokens), total))

    tick_positions, tick_labels, tick_is_leakage = [], [], []
    previous = -1
    for pos in sorted(selected):
        if pos > previous + 1:
            tick_positions.append((previous + 1 + pos - 1) / 2)
            tick_labels.append("...")
            tick_is_leakage.append(False)

        max_len = leakage_max_chars if pos in leakage_positions else max_chars
        tick_positions.append(pos)
        tick_labels.append(_token_text(tokenizer, sequence_ids, label_indices[pos], max_len))
        tick_is_leakage.append(pos in leakage_positions)
        previous = pos

    if previous < total - 1:
        tick_positions.append((previous + 1 + total - 1) / 2)
        tick_labels.append("...")
        tick_is_leakage.append(False)

    return tick_positions, tick_labels, tick_is_leakage


def _style_leakage_ticklabels(labels, tick_is_leakage, leakage_label_color):
    for label, is_leakage in zip(labels, tick_is_leakage):
        label.set_clip_on(False)
        if is_leakage:
            label.set_color(leakage_label_color)
            label.set_fontweight("bold")


def _annotate_vertical_token_labels(
    ax,
    tick_positions,
    tick_labels,
    tick_is_leakage,
    label_fontsize,
    leakage_label_color,
    rows=4,
):
    from matplotlib.transforms import blended_transform_factory

    transform = blended_transform_factory(ax.transData, ax.transAxes)
    rows = max(1, rows)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=1)

    for i, (pos, label, is_leakage) in enumerate(zip(tick_positions, tick_labels, tick_is_leakage)):
        row = i % rows
        y = -0.08 - row * 0.075
        x_shift = (row - (rows - 1) / 2) * 0.18
        color = leakage_label_color if is_leakage else ("0.45" if label == "..." else "black")
        bbox = (
            dict(facecolor="white", edgecolor=leakage_label_color, linewidth=0.7, alpha=0.85, pad=1.5)
            if is_leakage
            else None
        )
        ax.text(
            pos + x_shift,
            y,
            label,
            transform=transform,
            ha="center",
            va="top",
            rotation=0,
            fontsize=label_fontsize,
            color=color,
            fontweight="bold" if is_leakage else "normal",
            bbox=bbox,
            clip_on=False,
        )
        if is_leakage:
            ax.plot(
                [pos, pos + x_shift],
                [-0.01, y + 0.015],
                transform=transform,
                color=leakage_label_color,
                linewidth=0.6,
                alpha=0.65,
                clip_on=False,
            )

def _set_token_axis_labels(ax, tokenizer, sequence_ids, x_indices, y_indices, label_fontsize):
    x_ticks, x_labels = _token_ticks_and_labels(tokenizer, sequence_ids, x_indices)
    y_ticks, y_labels = _token_ticks_and_labels(tokenizer, sequence_ids, y_indices)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_labels, rotation=90, fontsize=label_fontsize)
    ax.set_yticklabels(y_labels, fontsize=label_fontsize)
    ax.tick_params(length=0, pad=1)

    ax.set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
    ax.set_ylim(max(y_ticks) + 0.5, min(y_ticks) - 0.5)

def _normalize_indices(indices, total, name):
    if indices is None:
        return list(range(total))

    normalized = []
    for idx in indices:
        idx = total + idx if idx < 0 else idx
        if idx < 0 or idx >= total:
            raise ValueError(f"{name} index {idx} is out of range 0..{total - 1}")
        normalized.append(idx)
    if not normalized:
        raise ValueError(f"{name} selection cannot be empty.")
    return normalized

def plot_selective_attention_map(
    attn_weights,
    layers: tuple,
    heads: tuple,
    input_range: tuple=(None, None),
    output_range: tuple=(None, None),
    prompt_len=None,
    tokenizer=None,
    sequence_ids=None,
    out_path=None,
    cmap="Blues",
    dpi=180,
    figsize=(18, 14),
    label_fontsize=6,
    quantile=0.995,
    tight=True,
):

    num_layers = len(attn_weights)
    num_heads = attn_weights[0].shape[1]
    seq_len = attn_weights[0].shape[-1]
    selected_layers = _normalize_indices(layers, num_layers, "Layer")
    selected_heads = _normalize_indices(heads, num_heads, "Head")

    start_x, end_x = input_range
    start_y, end_y = output_range

    if start_x is None:
        start_x = 0
    if end_x is None:
        end_x = seq_len if prompt_len is None else prompt_len
    if start_y is None:
        start_y = 0 if prompt_len is None else prompt_len
    if end_y is None:
        end_y = seq_len

    x_indices = list(range(start_x, end_x))
    y_indices = list(range(start_y, end_y))

    show_token_labels = tokenizer is not None and sequence_ids is not None
    if show_token_labels:
        sequence_ids = sequence_ids.detach().cpu()
        if sequence_ids.dim() > 1:
            sequence_ids = sequence_ids[0]

    fig, axes = plt.subplots(
        len(selected_layers),
        len(selected_heads),
        figsize=figsize,
        squeeze=False,
    )

    for row, layer_idx in enumerate(selected_layers):
        layer_attn = attn_weights[layer_idx]
        for col, head_idx in enumerate(selected_heads):
            ax = axes[row, col]
            attn_map = layer_attn[0, head_idx][y_indices][:, x_indices]
            vmax = float(torch.quantile(attn_map, quantile))
            if vmax <= 0:
                vmax = float(attn_map.max()) or 1.0

            ax.imshow(attn_map.numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)

            if show_token_labels:
                _set_token_axis_labels(ax, tokenizer, sequence_ids, x_indices, y_indices, label_fontsize)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if row == 0:
                ax.set_title(f"H{head_idx}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=8)

    title = "Selective output-to-prompt attention maps" if prompt_len is not None else "Selective token-to-token attention maps"
    fig.suptitle(title, y=1.0)
    if tight:
        fig.tight_layout(pad=0.3)
    else:
        fig.subplots_adjust(left=0.02, right=0.995, bottom=0.02, top=0.99, wspace=0.08, hspace=0.08)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs["bbox_inches"] = "tight"
        fig.savefig(out_path, **save_kwargs)

    return fig, axes

def plot_selective_attention_bar(
    one_dimensional_attn_weights,
    layers: tuple,
    heads: tuple,
    output_range: tuple=(None, None),
    prompt_len=None,
    tokenizer=None,
    sequence_ids=None,
    leakage_spans=None,
    full_text=None,
    response_text=None,
    out_path=None,
    color="tab:blue",
    leakage_color="tab:red",
    leakage_label_color="crimson",
    dpi=180,
    figsize=(18, 14),
    label_fontsize=6,
    compact_labels=False,
    label_context_tokens=8,
    edge_label_tokens=2,
    leakage_label_max_chars=40,
    orientation="vertical",
    tight=True,
):
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'.")

    num_layers = len(one_dimensional_attn_weights)
    num_heads = one_dimensional_attn_weights[0].shape[1]
    output_len = one_dimensional_attn_weights[0].shape[2]
    selected_layers = _normalize_indices(layers, num_layers, "Layer")
    selected_heads = _normalize_indices(heads, num_heads, "Head")

    start_y, end_y = output_range
    if start_y is None:
        start_y = 0
    if end_y is None:
        end_y = output_len
    if start_y < 0 or end_y > output_len or start_y >= end_y:
        raise ValueError(f"output_range must satisfy 0 <= start < end <= {output_len}.")

    y_indices = list(range(start_y, end_y))
    if prompt_len is None:
        label_indices = y_indices
    else:
        label_indices = [prompt_len + idx for idx in y_indices if prompt_len + idx < len(sequence_ids)]

    show_token_labels = tokenizer is not None and sequence_ids is not None
    if show_token_labels:
        sequence_ids = sequence_ids.detach().cpu()
        if sequence_ids.dim() > 1:
            sequence_ids = sequence_ids[0]
        leakage_output_indices = _leakage_output_indices(
            tokenizer,
            leakage_spans=leakage_spans,
            prompt_len=prompt_len,
            full_text=full_text,
            response_text=response_text,
        )
        leakage_positions = {
            idx - start_y
            for idx in leakage_output_indices
            if start_y <= idx < end_y
        }
        if compact_labels:
            x_ticks, x_labels, tick_is_leakage = _compact_token_ticks_and_labels(
                tokenizer,
                sequence_ids,
                label_indices,
                leakage_positions,
                context_tokens=label_context_tokens,
                edge_tokens=edge_label_tokens,
                leakage_max_chars=leakage_label_max_chars,
            )
        else:
            x_ticks, x_labels = _token_ticks_and_labels(tokenizer, sequence_ids, label_indices)
            tick_is_leakage = [False] * len(x_labels)
    else:
        leakage_positions = set()

    fig, axes = plt.subplots(
        len(selected_layers),
        len(selected_heads),
        figsize=figsize,
        squeeze=False,
    )

    for row, layer_idx in enumerate(selected_layers):
        layer_attn = one_dimensional_attn_weights[layer_idx]
        for col, head_idx in enumerate(selected_heads):
            ax = axes[row, col]
            values = layer_attn[0, head_idx, start_y:end_y]
            if values.dim() > 1 and values.shape[-1] == 1:
                values = values.squeeze(-1)
            elif values.dim() > 1:
                values = values.sum(dim=-1)

            positions = range(len(y_indices))
            values = values.detach().cpu().numpy()
            bar_colors = [
                leakage_color if pos in leakage_positions else color
                for pos in range(len(y_indices))
            ]

            if orientation == "vertical":
                ax.bar(positions, values, color=bar_colors)
                if show_token_labels:
                    if compact_labels:
                        _annotate_vertical_token_labels(
                            ax,
                            x_ticks,
                            x_labels,
                            tick_is_leakage,
                            label_fontsize,
                            leakage_label_color,
                        )
                    else:
                        ax.set_xticks(x_ticks)
                        ax.set_xticklabels(
                            x_labels,
                            rotation=45,
                            fontsize=label_fontsize,
                            ha="right",
                            va="top",
                            rotation_mode="anchor",
                        )
                        _style_leakage_ticklabels(ax.get_xticklabels(), tick_is_leakage, leakage_label_color)
                        ax.tick_params(axis="x", length=0, pad=2)
                    ax.set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
                else:
                    ax.set_xticks([])
            else:
                ax.barh(positions, values, color=bar_colors)
                if show_token_labels:
                    ax.set_yticks(x_ticks)
                    ax.set_yticklabels(x_labels, fontsize=label_fontsize)
                    _style_leakage_ticklabels(ax.get_yticklabels(), tick_is_leakage, leakage_label_color)
                    ax.tick_params(axis="y", length=0, pad=1)
                    ax.set_ylim(max(x_ticks) + 0.5, min(x_ticks) - 0.5)
                else:
                    ax.set_yticks([])

            if row == 0:
                ax.set_title(f"H{head_idx}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=8)

    fig.suptitle("Selective attention mass to leveraging-context cluster", y=1.0)
    if tight:
        fig.tight_layout(pad=0.3)
        if orientation == "vertical" and show_token_labels and compact_labels:
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.28))
    else:
        fig.subplots_adjust(left=0.02, right=0.995, bottom=0.02, top=0.99, wspace=0.15, hspace=0.15)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs["bbox_inches"] = "tight"
        fig.savefig(out_path, **save_kwargs)

    return fig, axes

def plot_aggregated_attention_map(
    attn_weights,
    input_range: tuple=(None, None),
    output_range: tuple=(None, None),
    prompt_len=None,
    tokenizer=None,
    sequence_ids=None,
    out_path=None,
    cmap="Blues",
    dpi=180,
    figsize=(18, 14),
    label_fontsize=6,
    quantile=0.995,
    tight=True,
):

    # Average over all layers and all heads.
    attn = torch.stack([layer_attn[0] for layer_attn in attn_weights])
    attn_map = attn.mean(dim=(0, 1))

    seq_len = attn_map.shape[-1]

    start_x, end_x = input_range
    start_y, end_y = output_range

    if start_x is None:
        start_x = 0
    if end_x is None:
        end_x = seq_len if prompt_len is None else prompt_len
    if start_y is None:
        start_y = 0 if prompt_len is None else prompt_len
    if end_y is None:
        end_y = seq_len

    x_indices = list(range(start_x, end_x))
    y_indices = list(range(start_y, end_y))
    attn_map = attn_map[start_y:end_y, start_x:end_x]

    vmax = float(torch.quantile(attn_map, quantile))
    if vmax <= 0:
        vmax = float(attn_map.max()) or 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn_map.numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="mean attention score")

    if tokenizer is not None and sequence_ids is not None:
        sequence_ids = sequence_ids.detach().cpu()
        if sequence_ids.dim() > 1:
            sequence_ids = sequence_ids[0]
        _set_token_axis_labels(ax, tokenizer, sequence_ids, x_indices, y_indices, label_fontsize)

    if prompt_len is None:
        ax.set_xlabel("Key token")
        ax.set_ylabel("Query token")
        ax.set_title("Aggregated token-to-token attention over all layers and heads")
    else:
        ax.set_xlabel("Prompt/input token")
        ax.set_ylabel("Generated/output token")
        ax.set_title("Aggregated output-to-prompt attention over all layers and heads")

    if tight:
        fig.tight_layout()

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)

    return attn_map, fig, ax

def torch_dtype(name: str):
    return "auto" if name == "auto" else getattr(torch, name)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict], path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite or not path.exists() else "a"
    with path.open(mode, encoding="utf-8") as file:
        file.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
