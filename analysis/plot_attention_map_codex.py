"""Plot token attention maps for decoder-only Hugging Face language models.

This script is meant for local/open-weight causal LMs whose forward pass can
return attention tensors. Hosted API models usually do not expose those tensors.

Example:
    python analysis/plot_attention_map.py ^
      --model-id Qwen/Qwen3-4B ^
      --prompt "Explain why traffic lights are timed." ^
      --chat ^
      --max-new-tokens 80 ^
      --layer -1 ^
      --head mean ^
      --map output_to_input ^
      --out analysis/qwen_attention.png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text and plot attention maps for a local causal LM."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model id or local model path, e.g. Qwen/Qwen3-4B.",
    )
    parser.add_argument("--prompt", required=True, help="Input prompt or user message.")
    parser.add_argument(
        "--completion",
        default=None,
        help=(
            "Optional known output text. If omitted, the script generates an output "
            "with model.generate()."
        ),
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Apply the tokenizer chat template to the prompt before generation.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system message used only with --chat.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help=(
            "Passed to from_pretrained. Use 'auto' for accelerate sharding or "
            "'none' to move the whole model to one device."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help=(
            "Attention backend. 'eager' is the safest choice when you need "
            "output_attentions=True."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable this only for model repos you trust.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer index to plot. Negative values count from the end.",
    )
    parser.add_argument(
        "--head",
        default="mean",
        help="Attention head index, or 'mean' to average all heads.",
    )
    parser.add_argument(
        "--map",
        choices=["output_to_input", "full"],
        default="output_to_input",
        help=(
            "output_to_input plots generated tokens as rows and prompt tokens as "
            "columns. full plots all selected query/key tokens."
        ),
    )
    parser.add_argument(
        "--renormalize-slice",
        action="store_true",
        help=(
            "Renormalize each displayed row over the displayed columns. Useful for "
            "pattern reading, but no longer shows raw attention mass."
        ),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--max-full-tokens", type=int, default=192)
    parser.add_argument("--max-x-labels", type=int, default=80)
    parser.add_argument("--max-y-labels", type=int, default=80)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--out",
        default="analysis/attention_map.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV path. Defaults to the PNG path with .csv suffix.",
    )
    parser.add_argument(
        "--metadata-out",
        default=None,
        help="Optional JSON metadata path. Defaults to the PNG path with .json suffix.",
    )
    return parser.parse_args()


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: torch. Install dependencies with "
            "'pip install -r analysis/requirements-attention.txt'."
        ) from exc
    return torch


def torch_dtype_from_name(name: str):
    if name == "auto":
        return "auto"
    torch = require_torch()
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_prompt_text(tokenizer, args: argparse.Namespace) -> str:
    if not args.chat:
        return args.prompt

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        raise ValueError(
            "--chat was set, but this tokenizer does not define a chat template."
        )

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {name: value.to(device) for name, value in batch.items()}


def model_input_device(model) -> torch.device:
    torch = require_torch()
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def load_model_and_tokenizer(args: argparse.Namespace):
    torch = require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: transformers. Install dependencies with "
            "'pip install -r analysis/requirements-attention.txt'."
        ) from exc

    if args.device_map == "auto" and importlib.util.find_spec("accelerate") is None:
        raise RuntimeError(
            "device_map='auto' requires accelerate. Install it with "
            "'pip install accelerate', or pass --device-map none."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = {
        "torch_dtype": torch_dtype_from_name(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()

    if args.device_map == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def generate_or_attach_completion(
    tokenizer,
    model,
    prompt_text: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int, str]:
    torch = require_torch()
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]

    if args.completion is not None:
        prompt_ids = prompt_inputs["input_ids"][0]
        completion_ids = tokenizer(
            args.completion,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        sequence_ids = torch.cat([prompt_ids, completion_ids], dim=0)
        return sequence_ids, prompt_len, args.completion

    device = model_input_device(model)
    prompt_inputs = move_batch_to_device(prompt_inputs, device)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "return_dict_in_generate": False,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.inference_mode():
        sequences = model.generate(**prompt_inputs, **generate_kwargs)

    sequence_ids = sequences[0].detach().cpu()
    completion_ids = sequence_ids[prompt_len:]
    completion_text = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return sequence_ids, prompt_len, completion_text


def run_attention_forward(
    tokenizer,
    model,
    sequence_ids: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    torch = require_torch()
    device = model_input_device(model)
    input_ids = sequence_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
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
    return sequence_ids.cpu(), attentions


def normalize_layer_index(index: int, num_layers: int) -> int:
    if index < 0:
        index = num_layers + index
    if index < 0 or index >= num_layers:
        raise IndexError(f"Layer index out of range: {index} for {num_layers} layers.")
    return index


def select_head(layer_attention: torch.Tensor, head_arg: str) -> torch.Tensor:
    # layer_attention shape: batch, heads, query_tokens, key_tokens
    heads = layer_attention.shape[1]
    if head_arg == "mean":
        return layer_attention[0].mean(dim=0)

    head = int(head_arg)
    if head < 0:
        head = heads + head
    if head < 0 or head >= heads:
        raise IndexError(f"Head index out of range: {head} for {heads} heads.")
    return layer_attention[0, head]


def limit_indices(indices: list[int], max_count: int, keep_end: bool) -> list[int]:
    if max_count <= 0 or len(indices) <= max_count:
        return indices
    if keep_end:
        return indices[-max_count:]
    return indices[:max_count]


def build_slices(
    seq_len: int,
    prompt_len: int,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], str, str]:
    output_indices = list(range(prompt_len, seq_len))
    if args.map == "output_to_input":
        if not output_indices:
            raise ValueError(
                "No output tokens are available. Increase --max-new-tokens or pass "
                "--completion."
            )
        x_indices = limit_indices(
            list(range(prompt_len)),
            args.max_prompt_tokens,
            keep_end=True,
        )
        y_indices = limit_indices(
            output_indices,
            args.max_output_tokens,
            keep_end=False,
        )
        return x_indices, y_indices, "Prompt tokens", "Output tokens"

    selected = limit_indices(
        list(range(seq_len)),
        args.max_full_tokens,
        keep_end=True,
    )
    return selected, selected, "Key tokens", "Query tokens"


def token_label(tokenizer, token_id: int, position: int, max_chars: int = 24) -> str:
    text = tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if text == "":
        text = str(tokenizer.convert_ids_to_tokens([int(token_id)])[0])
    text = text.encode("unicode_escape").decode("ascii")
    text = text.replace(" ", "_")
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return f"{position}:{text}"


def labels_for_indices(tokenizer, sequence_ids: torch.Tensor, indices: Iterable[int]):
    return [token_label(tokenizer, int(sequence_ids[idx]), idx) for idx in indices]


def maybe_renormalize(matrix: torch.Tensor) -> torch.Tensor:
    torch = require_torch()
    denom = matrix.sum(dim=1, keepdim=True)
    return torch.where(denom > 0, matrix / denom.clamp_min(1e-12), matrix)


def write_csv(path: Path, matrix, x_labels: list[str], y_labels: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query/key", *x_labels])
        for label, row in zip(y_labels, matrix.tolist()):
            writer.writerow([label, *[f"{value:.8g}" for value in row]])


def tick_positions(num_labels: int, max_labels: int) -> list[int]:
    if num_labels <= max_labels:
        return list(range(num_labels))
    step = math.ceil(num_labels / max_labels)
    return list(range(0, num_labels, step))


def plot_matrix(
    matrix,
    x_labels: list[str],
    y_labels: list[str],
    args: argparse.Namespace,
    layer_index: int,
    out_path: Path,
    x_axis_label: str,
    y_axis_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = min(26, max(8, 2.8 + 0.22 * len(x_labels)))
    fig_height = min(22, max(6, 2.4 + 0.24 * len(y_labels)))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=args.cmap)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Attention probability")

    x_ticks = tick_positions(len(x_labels), args.max_x_labels)
    y_ticks = tick_positions(len(y_labels), args.max_y_labels)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([x_labels[idx] for idx in x_ticks], rotation=90, fontsize=7)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([y_labels[idx] for idx in y_ticks], fontsize=7)

    head_label = "mean heads" if args.head == "mean" else f"head {args.head}"
    renorm_label = ", row-renormalized" if args.renormalize_slice else ""
    ax.set_title(f"{args.map} attention, layer {layer_index}, {head_label}{renorm_label}")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    prompt_text: str,
    completion_text: str,
    prompt_len: int,
    seq_len: int,
    layer_index: int,
    num_layers: int,
    num_heads: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": args.model_id,
        "chat": args.chat,
        "prompt": args.prompt,
        "rendered_prompt": prompt_text,
        "completion": completion_text,
        "prompt_token_count": prompt_len,
        "sequence_token_count": seq_len,
        "layer_index": layer_index,
        "num_layers": num_layers,
        "head": args.head,
        "num_heads": num_heads,
        "map": args.map,
        "renormalize_slice": args.renormalize_slice,
        "attn_implementation": args.attn_implementation,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    csv_path = Path(args.csv_out) if args.csv_out else out_path.with_suffix(".csv")
    metadata_path = (
        Path(args.metadata_out) if args.metadata_out else out_path.with_suffix(".json")
    )

    tokenizer, model = load_model_and_tokenizer(args)
    prompt_text = build_prompt_text(tokenizer, args)
    sequence_ids, prompt_len, completion_text = generate_or_attach_completion(
        tokenizer,
        model,
        prompt_text,
        args,
    )
    sequence_ids, attentions = run_attention_forward(tokenizer, model, sequence_ids)

    layer_index = normalize_layer_index(args.layer, len(attentions))
    layer_attention = attentions[layer_index]
    selected_attention = select_head(layer_attention, args.head)
    num_heads = layer_attention.shape[1]

    x_indices, y_indices, x_axis_label, y_axis_label = build_slices(
        selected_attention.shape[0],
        prompt_len,
        args,
    )
    matrix = selected_attention[y_indices][:, x_indices]
    if args.renormalize_slice:
        matrix = maybe_renormalize(matrix)

    x_labels = labels_for_indices(tokenizer, sequence_ids, x_indices)
    y_labels = labels_for_indices(tokenizer, sequence_ids, y_indices)
    matrix_numpy = matrix.numpy()

    plot_matrix(
        matrix_numpy,
        x_labels,
        y_labels,
        args,
        layer_index,
        out_path,
        x_axis_label,
        y_axis_label,
    )
    write_csv(csv_path, matrix_numpy, x_labels, y_labels)
    write_metadata(
        metadata_path,
        args,
        prompt_text,
        completion_text,
        prompt_len,
        int(sequence_ids.shape[0]),
        layer_index,
        len(attentions),
        int(num_heads),
    )

    print(f"Generated/completion text:\n{completion_text}\n")
    print(f"Saved heatmap: {out_path}")
    print(f"Saved matrix CSV: {csv_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
