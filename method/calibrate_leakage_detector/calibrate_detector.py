from __future__ import annotations

from pathlib import Path
import sys
from importlib import reload

project_root = Path.cwd()
if not (project_root / "utils").exists():
    project_root = project_root.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import utils.utils as utils
reload(utils)

import argparse
import gc
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from utils.utils import load_model_and_tokenizer, build_prompt, get_attention_weights



def parse_args():
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["calibrate", "evaluate", "total"], default="calibrate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibrate-path", type=Path, default=base / "generated-responses-with-leakage-spans.jsonl")
    parser.add_argument("--test-path", type=Path, default=base / "test-generated-responses-with-leakage.jsonl")
    parser.add_argument("--result-path", type=Path, default=base / "calibrate_result.jsonl")
    parser.add_argument("--score-cache-path", type=Path, default=base / "calibrate_head_scores.pt")
    parser.add_argument("--eval-path", type=Path, default=base / "calibrate_eval_result.jsonl")
    parser.add_argument("--plot-dir", type=Path, default=base / "lowest_score_plots")
    parser.add_argument("--plot-lowest-n", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)     # how many attn heads to keep/choose
    parser.add_argument("--max-samples", type=int, default=None)    # how many samples to use
    parser.add_argument("--max-seq-len", type=int, default=2048)    # skip samples where seq-len > max-seq-len
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_name_key(model: str) -> str:
    name = re.split(r"[\\/]", model.rstrip("\\/"))[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def torch_dtype(name: str):
    if name == "auto":
        return "auto"
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: list[dict], path: Path, overwrite: bool):
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def privileged_context(prompt: str) -> str | None:
    start = prompt.find("Given the ground truth answer is ")
    if start < 0:
        return None
    return prompt[start:]


def overlapping_tokens(offsets, start_char: int, end_char: int) -> list[int]:
    return [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start_char and start < end_char
    ]


def find_all(text: str, needle: str):
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        yield idx
        start = idx + max(len(needle), 1)


def build_sample(row: dict, tokenizer, args) -> dict | None:
    if not row.get("leakage_spans"):
        return None
    
    prompt_text = build_prompt(
        tokenizer, system_prompt="", user_prompt=row['prompt'], encode=False, 
        add_generation_prompt=True, enable_thinking=not args.disable_thinking
    )
    full_text = prompt_text + row["response"]
    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    offsets = encoded["offset_mapping"]     # [[start_char, end_char), [start_char, end_char), ...]
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    output_len = len(input_ids) - prompt_len

    if output_len <= 0 or len(input_ids) > args.max_seq_len:
        return None

    ideal = torch.zeros(output_len)
    for span in row.get("leakage_spans", []):
        for char_idx in find_all(row["response"], span):
            start = len(prompt_text) + char_idx
            end = start + len(span)
            for token_idx in overlapping_tokens(offsets, start, end):
                if token_idx < prompt_len:
                    print(f"Warning: leakage span overlaps with prompt in sample {row['prompt'][:50]}...")
                else:
                    ideal[token_idx - prompt_len] = 1.0

    if ideal.sum() == 0:
        raise RuntimeError(f"No leakage tokens found in sample {row['prompt'][:50]} though filtered...")

    context = privileged_context(row["prompt"])
    if context is None:
        return None
    context_start = full_text.find(context)
    if context_start < 0:
        return None
    key_tokens = overlapping_tokens(offsets, context_start, context_start + len(context))
    if not key_tokens:
        return None

    return {"input_ids": input_ids, "prompt_len": prompt_len, "key_tokens": key_tokens, "ideal": ideal}


def model_device(model):
    return model.device if hasattr(model, "device") else next(model.parameters()).device

def attention_to_context(attentions, prompt_len: int, key_tokens: list[int]) -> torch.Tensor:
    key_tokens = torch.tensor(key_tokens)
    vectors = []
    for layer_attn in attentions:
        values = layer_attn[0, :, prompt_len:, :].index_select(-1, key_tokens).sum(dim=-1)
        vectors.append(values)
    return torch.stack(vectors)  # [layers, heads, output_tokens]


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm()).clamp_min(1e-12)


def score_all_heads_and_layers(attentions, sample: dict) -> torch.Tensor:
    vectors = attention_to_context(attentions, sample["prompt_len"], sample["key_tokens"])
    return cosine(vectors, sample["ideal"])


def score_selected_heads_and_layers(attentions, sample: dict, heads: list[dict]) -> tuple[float, torch.Tensor]:
    vectors = attention_to_context(attentions, sample["prompt_len"], sample["key_tokens"])
    selected = torch.stack([vectors[item["layer"], item["head"]] for item in heads]).mean(dim=0)
    return float(cosine(selected, sample["ideal"])), selected


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def top_heads(scores: torch.Tensor, top_k: int) -> list[dict]:
    num_heads = scores.shape[1]
    values, indices = torch.topk(scores.flatten(), min(top_k, scores.numel()))
    return [
        {"layer": int(idx // num_heads), "head": int(idx % num_heads), "score": float(score)}
        for score, idx in zip(values, indices)
    ]


def save_selected_heads(scores: torch.Tensor, args):
    ranked = top_heads(scores, args.top_k)
    heads = [{"layer": item["layer"], "head": item["head"]} for item in ranked]
    write_jsonl([{model_name_key(args.model): heads}], args.result_path, args.overwrite)
    for item in ranked:
        print(f"L{item['layer']} H{item['head']} cosine={item['score']:.4f}")


def save_score_cache(scores: torch.Tensor, used: int, args):
    if args.score_cache_path.exists() and not args.overwrite:
        raise FileExistsError(f"{args.score_cache_path} exists. Use --overwrite to replace it.")
    args.score_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model_name_key(args.model), "scores": scores.cpu(), "used": used},
        args.score_cache_path,
    )


def calibrate_from_cache(args):
    cache = torch.load(args.score_cache_path, map_location="cpu")
    if cache.get("model") != model_name_key(args.model):
        raise ValueError(f"Cache is for {cache.get('model')}, but --model is {model_name_key(args.model)}.")
    save_selected_heads(cache["scores"], args)
    print(f"Loaded scores from {args.score_cache_path}. Used {cache.get('used')} cached samples.")


def calibrate(model, tokenizer, args):
    rows = [row for row in read_jsonl(args.calibrate_path) if row.get("leakage_spans")]
    rows = rows[: args.max_samples] if args.max_samples else rows

    total_scores = None
    used = 0
    for row in tqdm(rows, desc="Calibrating"):
        sample = build_sample(row, tokenizer, args)
        if sample is None:
            continue
        attentions, _ = get_attention_weights(model, sample["input_ids"])
        scores = score_all_heads_and_layers(attentions, sample)
        total_scores = scores if total_scores is None else total_scores + scores
        used += 1
        del attentions
        clear_memory()

    if used == 0:
        raise RuntimeError("No usable calibration samples.")

    scores = total_scores / used
    save_score_cache(scores, used, args)
    save_selected_heads(scores, args)
    print(f"Used {used} samples.")
    print(f"Saved score cache to {args.score_cache_path}.")


def read_heads(path: Path, model: str) -> list[dict]:
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return row.get(model_name_key(model), next(iter(row.values())))


def plot_sample(real: torch.Tensor, ideal: torch.Tensor, score: float, out_path: Path):
    real = real / real.max().clamp_min(1e-12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(real.numpy(), label="attention to privileged context")
    ax.plot(ideal.numpy(), "--", drawstyle="steps-post", label="ideal leakage mask")
    ax.set_title(f"cosine={score:.4f}")
    ax.set_xlabel("output token index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate(model, tokenizer, args):
    heads = read_heads(args.result_path, args.model)
    rows = [row for row in read_jsonl(args.test_path) if row.get("leakage_spans")]
    rows = rows[: args.max_samples] if args.max_samples else rows

    results = []
    plots = []
    for sample_id, row in tqdm(list(enumerate(rows, 1)), desc="Evaluating"):
        sample = build_sample(row, tokenizer, args)
        if sample is None:
            continue
        attentions, _ = get_attention_weights(model, sample["input_ids"])
        score, real = score_selected_heads_and_layers(attentions, sample, heads)
        results.append({"sample_id": sample_id, "score": score})
        if args.plot_lowest_n:
            plots.append((score, sample_id, real, sample["ideal"]))
        del attentions
        clear_memory()

    write_jsonl(results, args.eval_path, args.overwrite)
    for rank, (score, sample_id, real, ideal) in enumerate(sorted(plots)[: args.plot_lowest_n], 1):
        plot_sample(real, ideal, score, args.plot_dir / f"lowest_{rank:02d}_sample_{sample_id:04d}.png")

    if results:
        mean_score = sum(row["score"] for row in results) / len(results)
        print(f"Mean cosine: {mean_score:.4f}")


def main():
    args = parse_args()
    if args.phase == "calibrate" and args.from_cache:
        calibrate_from_cache(args)
        return

    model, tokenizer = load_model_and_tokenizer(
        model=args.model, 
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    if args.phase == "total":
        if args.from_cache:
            calibrate_from_cache(args)
        else:
            calibrate(model, tokenizer, args)
        evaluate(model, tokenizer, args)
    elif args.phase == "calibrate":
        calibrate(model, tokenizer, args)
    else:
        evaluate(model, tokenizer, args)


if __name__ == "__main__":
    main()
