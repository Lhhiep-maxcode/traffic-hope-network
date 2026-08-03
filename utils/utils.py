import torch

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
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        labels.append(f"{idx}:{text}")
    return labels

def plot_all_layer_head_attention(
    attn_weights,
    prompt_len=None,
    tokenizer=None,
    sequence_ids=None,
    out_path=None,
    cmap="Blues",
    dpi=160,
    panel_size=3.2,
    label_fontsize=3,
    quantile=0.995,
    tight=False,
):
    from pathlib import Path

    import matplotlib.pyplot as plt

    num_layers = len(attn_weights)
    num_heads = attn_weights[0].shape[1]
    seq_len = attn_weights[0].shape[-1]
    if prompt_len is None:
        x_indices = list(range(seq_len))
        y_indices = list(range(seq_len))
    else:
        x_indices = list(range(prompt_len))
        y_indices = list(range(prompt_len, seq_len))

    show_token_labels = tokenizer is not None and sequence_ids is not None
    if show_token_labels:
        sequence_ids = sequence_ids.detach().cpu()
        x_labels = _token_labels(tokenizer, sequence_ids, x_indices)
        y_labels = _token_labels(tokenizer, sequence_ids, y_indices)

    fig, axes = plt.subplots(
        num_layers,
        num_heads,
        figsize=(num_heads * panel_size, num_layers * panel_size),
        squeeze=False,
    )

    for layer_idx, layer_attn in enumerate(attn_weights):
        print(f"Plotting layer {layer_idx + 1}/{num_layers}...")
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            attn_map = layer_attn[0, head_idx]
            if prompt_len is not None:
                attn_map = attn_map[prompt_len:, :prompt_len]
            vmax = float(torch.quantile(attn_map, quantile))
            if vmax <= 0:
                vmax = float(attn_map.max()) or 1.0

            ax.imshow(attn_map.numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)

            if show_token_labels:
                ax.set_xticks(range(len(x_indices)))
                ax.set_yticks(range(len(y_indices)))
                ax.set_xticklabels(x_labels, rotation=90, fontsize=label_fontsize)
                ax.set_yticklabels(y_labels, fontsize=label_fontsize)
                ax.tick_params(length=0, pad=1)
            else:
                ax.set_xticks([])
                ax.set_yticks([])

            if layer_idx == 0:
                ax.set_title(f"H{head_idx}", fontsize=8)
            if head_idx == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=8)

    title = "Output-to-prompt attention maps" if prompt_len is not None else "Token-to-token attention maps"
    fig.suptitle(f"{title} by layer and head", y=1.0)
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

def plot_aggregated_attention(
    attn_weights,
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
    from pathlib import Path

    import matplotlib.pyplot as plt

    # Average over all layers and all heads.
    attn = torch.stack([layer_attn[0] for layer_attn in attn_weights])
    attn_map = attn.mean(dim=(0, 1))

    seq_len = attn_map.shape[-1]
    if prompt_len is None:
        x_indices = list(range(seq_len))
        y_indices = list(range(seq_len))
    else:
        x_indices = list(range(prompt_len))
        y_indices = list(range(prompt_len, seq_len))
        attn_map = attn_map[prompt_len:, :prompt_len]

    vmax = float(torch.quantile(attn_map, quantile))
    if vmax <= 0:
        vmax = float(attn_map.max()) or 1.0

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attn_map.numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="mean attention score")

    if tokenizer is not None and sequence_ids is not None:
        sequence_ids = sequence_ids.detach().cpu()
        ax.set_xticks(range(len(x_indices)))
        ax.set_yticks(range(len(y_indices)))
        ax.set_xticklabels(_token_labels(tokenizer, sequence_ids, x_indices), rotation=90, fontsize=label_fontsize)
        ax.set_yticklabels(_token_labels(tokenizer, sequence_ids, y_indices), fontsize=label_fontsize)
        ax.tick_params(length=0, pad=1)

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
