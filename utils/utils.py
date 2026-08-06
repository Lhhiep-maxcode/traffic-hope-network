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

def _token_labels(tokenizer, sequence_ids, indices, max_chars=10):
    labels = []
    for idx in indices:
        text = tokenizer.decode([int(sequence_ids[idx])], skip_special_tokens=False)
        text = text.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "_")
        text = text.replace("$", r"\$")
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        labels.append(f"{idx}:{text}")
    return labels

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
    out_path=None,
    color="tab:blue",
    dpi=180,
    figsize=(18, 14),
    label_fontsize=6,
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
        x_ticks, x_labels = _token_ticks_and_labels(tokenizer, sequence_ids, label_indices)

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

            if orientation == "vertical":
                ax.bar(positions, values, color=color)
                if show_token_labels:
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_labels, rotation=90, fontsize=label_fontsize)
                    ax.tick_params(axis="x", length=0, pad=1)
                    ax.set_xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
                else:
                    ax.set_xticks([])
            else:
                ax.barh(positions, values, color=color)
                if show_token_labels:
                    ax.set_yticks(x_ticks)
                    ax.set_yticklabels(x_labels, fontsize=label_fontsize)
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
