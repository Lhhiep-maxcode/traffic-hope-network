from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch
import warnings
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import build_prompt, load_model_and_tokenizer, read_json, read_jsonl, write_json, write_jsonl, torch_dtype

EPS = 1e-12
PRIVILEGED_MARKER = "Given the ground truth answer is "
SCORE_METHOD = "leakage_cosine_or_non_leak_top5_penalty"


def parse_args():
    base = Path(__file__).parent
    data = base / "data"
    output = base / "output"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key")
    parser.add_argument("--calibrate-path", type=Path, default=data / "generated-responses-with-leakage-spans.jsonl")
    parser.add_argument("--val-calibrate-path", type=Path, default=data / "val-generated-responses-with-leakage-spans.jsonl")
    parser.add_argument("--detector-config-path", type=Path, default=output / "detector_config.json")
    parser.add_argument("--score-cache-path", type=Path, default=output / "calibrate_head_scores.pt")
    parser.add_argument("--all-experiments-path", type=Path, default=output / "all_experiments.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-k-values", default=None)
    parser.add_argument("--window-sizes", default="1,3,5,7")
    parser.add_argument("--threshold-steps", type=int, default=80, help=argparse.SUPPRESS)
    parser.add_argument("--aggregation", choices=["weighted", "mean"], default="weighted")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_name_key(args) -> str:
    if args.model_key:
        return args.model_key
    name = re.split(r"[\\/]", args.model.rstrip("\\/"))[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def read_detector(args) -> dict:
    if args.model_key:
        key = args.model_key
    else:
        key = model_name_key(args)
    config = read_json(args.detector_config_path)
    if key not in config:
        raise KeyError(f"No detector config for {key} in {args.detector_config_path}.")
    return config[key]


def int_grid(text: str | None, default: int | None = None, max_value: int | None = None) -> list[int]:
    values = [default] if text is None else [int(item) for item in text.split(",") if item.strip()]
    values = sorted({value for value in values if value and value > 0})
    if max_value is not None:
        values = sorted({min(value, max_value) for value in values})
    if not values:
        raise ValueError("Integer grid cannot be empty.")
    return values


def find_all(text: str, needle: str):
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        yield idx
        start = idx + max(len(needle), 1)


def overlapping_tokens(offsets, start_char: int, end_char: int) -> list[int]:
    return [i for i, (start, end) in enumerate(offsets) if end > start_char and start < end_char]


def privileged_context(row: dict) -> str | None:
    if "privileged_context" in row:
        return row["privileged_context"]
    start = row['prompt'].find(PRIVILEGED_MARKER)
    return None if start < 0 else row['prompt'][start:]


def build_inference_sample(row: dict, tokenizer, args) -> dict | None:
    prompt = row["prompt"]
    response = row["response"]
    prompt_text = build_prompt(
        tokenizer,
        system_prompt="",
        user_prompt=prompt,
        encode=False,
        add_generation_prompt=True,
        enable_thinking=not args.disable_thinking,
    )
    if prompt_text.rstrip().endswith("<think>") and response.lstrip().startswith("<think>"):
        response = response.lstrip()[len("<think>"):].lstrip()

    full_text = prompt_text + response
    encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
    offsets = encoded["offset_mapping"]
    prompt_len = next((i for i, (_, end) in enumerate(offsets) if end > len(prompt_text)), len(input_ids))

    context = privileged_context(row)
    if context is None or len(input_ids) <= prompt_len or len(input_ids) > args.max_seq_len:
        return None

    context_start = full_text.find(context)
    key_tokens = overlapping_tokens(offsets, context_start, context_start + len(context))
    if context_start < 0 or not key_tokens:
        return None

    return {
        "input_ids": input_ids,
        "prompt_len": prompt_len,
        "key_tokens": key_tokens,
        "offsets": offsets,
        "prompt_text": prompt_text,
    }


def build_sample(row: dict, tokenizer, args) -> dict | None:
    # if not row.get("leakage_spans"):
    #     return None
    sample = build_inference_sample(row, tokenizer, args)
    if sample is None:
        return None

    ideal = torch.zeros(len(sample["input_ids"]) - sample["prompt_len"])
    for span in row["leakage_spans"]:
        for char_idx in find_all(row["response"], span):
            start = len(sample["prompt_text"]) + char_idx
            for token_idx in overlapping_tokens(sample["offsets"], start, start + len(span)):
                if token_idx >= sample["prompt_len"]:
                    ideal[token_idx - sample["prompt_len"]] = 1.0

    # if ideal.sum() == 0:
    #     raise RuntimeError(f"No leakage tokens found in sample: {row['prompt'][:80]}")
    return {**sample, "ideal": ideal}


def model_device(model):
    return model.device if hasattr(model, "device") else next(model.parameters()).device


def pad_token_id(tokenizer) -> int:
    return tokenizer.pad_token_id or tokenizer.eos_token_id or 0


def batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def get_batch_attention_weights(model, tokenizer, samples: list[dict]):
    max_len = max(len(sample["input_ids"]) for sample in samples)
    input_ids = torch.full((len(samples), max_len), pad_token_id(tokenizer), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)

    for i, sample in enumerate(samples):
        seq_len = len(sample["input_ids"])
        input_ids[i, :seq_len] = sample["input_ids"]
        attention_mask[i, :seq_len] = 1
    
    with torch.inference_mode():
        output = model(
            input_ids=input_ids.to(model_device(model)),
            attention_mask=attention_mask.to(model_device(model)),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    if output.attentions is None:
        raise RuntimeError("No attentions returned. Use --attn-implementation eager.")
    return tuple(attn.detach().float().cpu() for attn in output.attentions)


def take_sample_attentions(batch_attentions, batch_idx: int, seq_len: int):
    return tuple(attn[batch_idx : batch_idx + 1, :, :seq_len, :seq_len] for attn in batch_attentions)


def attention_to_context(attentions, prompt_len: int, key_tokens: list[int]) -> torch.Tensor:
    key_tokens = torch.tensor(key_tokens)
    return torch.stack([
        layer_attn[0, :, prompt_len:, :].index_select(-1, key_tokens).sum(dim=-1)
        for layer_attn in attentions
    ])  # shape = (num_layers, num_heads, seq_len - prompt_len)


def positive_median_normalize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.median(dim=-1, keepdim=True).values).clamp_min(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm()).clamp_min(EPS)


def topk_mean(values: torch.Tensor, k: int = 5) -> torch.Tensor:
    if values.shape[-1] == 0:
        return torch.zeros(values.shape[:-1], dtype=values.dtype, device=values.device)
    k = min(k, values.shape[-1])
    return values.topk(k, dim=-1).values.mean(dim=-1)


def leakage_shape_score(real: torch.Tensor, ideal: torch.Tensor) -> torch.Tensor:
    if bool(ideal.bool().any()):
        return cosine(real, ideal)
    return -topk_mean(real)


def head_scores(attentions, sample: dict) -> torch.Tensor:
    raw = attention_to_context(attentions, sample["prompt_len"], sample["key_tokens"])
    return leakage_shape_score(raw, sample["ideal"])  # shape = (num_layers, num_heads)


def top_heads(scores: torch.Tensor, top_k: int) -> list[dict]:
    num_heads = scores.shape[1]
    values, indices = torch.topk(scores.flatten(), min(top_k, scores.numel()))
    return [
        {"layer": int(idx // num_heads), "head": int(idx % num_heads), "score": float(value)}
        for value, idx in zip(values, indices)
    ]


def aggregate_normalized_vectors(vectors: torch.Tensor, heads: list[dict], aggregation: str = "weighted") -> torch.Tensor:
    selected = torch.stack([vectors[h["layer"], h["head"]] for h in heads])
    weights = torch.tensor([max(float(h.get("score", 1.0)), 0.0) for h in heads], dtype=selected.dtype)
    if aggregation == "weighted" and weights.sum() > 0:
        return (selected * weights[:, None]).sum(dim=0) / weights.sum()
    return selected.mean(dim=0)


def aggregate_selected_heads(attentions, sample: dict, heads: list[dict], aggregation: str = "weighted") -> torch.Tensor:
    raw = attention_to_context(attentions, sample["prompt_len"], sample["key_tokens"])
    return aggregate_normalized_vectors(raw, heads, aggregation)


def rolling_max(values: torch.Tensor, window_size: int) -> torch.Tensor:
    if window_size <= 1:
        return values
    left = window_size // 2
    right = window_size - 1 - left
    return F.max_pool1d(
        F.pad(values.view(1, 1, -1), (left, right), value=0),
        kernel_size=window_size,
        stride=1,
    ).view(-1)


def token_spans(mask: torch.Tensor) -> list[dict]:
    spans, start = [], None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            spans.append({"start_token": start, "end_token": i})
            start = None
    return spans



def shape_metrics(records: list[dict], args) -> list[dict]:
    rows = []
    for window_size in int_grid(args.window_sizes):
        scores, leakage_scores, non_leak_penalties = [], [], []
        for record in records:
            real = rolling_max(record["real"], window_size)
            if bool(record["ideal"].bool().any()):
                score = float(cosine(real, record["ideal"]))
                leakage_scores.append(score)
            else:
                penalty = float(topk_mean(real))
                score = -penalty
                non_leak_penalties.append(penalty)
            scores.append(score)
        score_tensor = torch.tensor(scores)
        rows.append({
            "window_size": window_size,
            "mean_score": float(score_tensor.mean()),
            "median_score": float(score_tensor.median()),
            "min_score": float(score_tensor.min()),
            "leakage_samples": len(leakage_scores),
            "non_leakage_samples": len(non_leak_penalties),
            "mean_leakage_cosine": (
                float(torch.tensor(leakage_scores).mean()) if leakage_scores else None
            ),
            "mean_non_leak_top5": (
                float(torch.tensor(non_leak_penalties).mean()) if non_leak_penalties else None
            ),
            "samples": len(records),
        })
    return rows


def choose_config(rows: list[dict], args) -> dict:
    return max(
        rows,
        key=lambda r: (
            r["mean_score"],
            r["median_score"],
            -r["window_size"],
            -r.get("top_k", 0),
        ),
    )


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def build_samples(path: Path, tokenizer, args):
    rows = [row for row in read_jsonl(path)]
    if args.max_samples:
        rows = rows[: args.max_samples]
    samples = [sample for row in rows if (sample := build_sample(row, tokenizer, args)) is not None]
    if not samples:
        raise RuntimeError(f"No usable samples in {path}.")
    return samples


def calibrate_head_scores(model, tokenizer, samples: list[dict], args) -> torch.Tensor:
    total, used = None, 0
    for batch in tqdm(list(batches(samples, args.batch_size)), desc="Calibrating heads"):
        batch_attn = get_batch_attention_weights(model, tokenizer, batch)
        for batch_idx, sample in enumerate(batch):
            sample_attn = take_sample_attentions(batch_attn, batch_idx, len(sample["input_ids"]))
            scores = head_scores(sample_attn, sample)
            total = scores if total is None else total + scores
            used += 1
        del batch_attn
        clear_memory()

    scores = total / used
    torch.save({
        "model": model_name_key(args),
        "score_method": SCORE_METHOD,
        "scores": scores.cpu(),
        "used": used,
    }, args.score_cache_path)
    return scores


def load_head_scores(args) -> torch.Tensor:
    cache = torch.load(args.score_cache_path, map_location="cpu")
    if cache.get("model") != model_name_key(args):
        warnings.warn(
            f"Cache is for {cache.get('model')}, but --model is {model_name_key(args)}.",
            UserWarning,
        )
    if cache.get("score_method") != SCORE_METHOD:
        warnings.warn(
            "Head-score cache was created with an older/different scoring method. "
            "Recompute without --from-cache to include non-leakage top-5 penalties.",
            UserWarning,
        )
    print(f"Loaded head scores from {args.score_cache_path}.")
    return cache["scores"]


def detector_records_by_k(model, tokenizer, samples: list[dict], scores: torch.Tensor, args):
    top_ks = int_grid(args.top_k_values, args.top_k, scores.numel())    # [1, 2, 4, 8, 16, ...]
    heads_by_k = {k: top_heads(scores, k) for k in top_ks}  # {k: [{"layer": 0, "head": 0, "score": 1.0}, ...], ...}
    records_by_k = {k: [] for k in top_ks}

    for batch in tqdm(list(batches(samples, args.batch_size)), desc="Calibrating k/window"):
        batch_attn = get_batch_attention_weights(model, tokenizer, batch)
        for batch_idx, sample in enumerate(batch):
            sample_attn = take_sample_attentions(batch_attn, batch_idx, len(sample["input_ids"]))
            # compute real attention vectors to privileged context for each head
            raw = attention_to_context(sample_attn, sample["prompt_len"], sample["key_tokens"])
            vectors = raw
            for k, heads in heads_by_k.items():
                real = aggregate_normalized_vectors(vectors, heads, args.aggregation)
                records_by_k[k].append({"real": real.cpu(), "ideal": sample["ideal"].cpu()})
        del batch_attn
        clear_memory()

    return heads_by_k, records_by_k, top_ks


def save_detector_config(heads: list[dict], best: dict, args):
    key = model_name_key(args)
    config = read_json(args.detector_config_path)

    config[key] = {
        "top_k": best["top_k"],
        "top_k_values": best["top_k_values"],
        "heads": heads,
        "aggregation": args.aggregation,
        "window_size": best["window_size"],
        "threshold": None,
        "val_calibration_mean_score": best["mean_score"],
        "val_calibration_median_score": best["median_score"],
        "val_calibration_min_score": best["min_score"],
        "test_precision": None,
        "test_recall": None,
        "test_full_recall": None,
        "test_f1": None,
        "test_token_precision": None,
        "test_avg_pred_spans": None,
    }
    write_json(config, args.detector_config_path)
    print(
        f"top_k={best['top_k']} window={best['window_size']} "
        f"mean_score={best['mean_score']:.4f} "
        f"median_score={best['median_score']:.4f}"
    )


def calibrate(model, tokenizer, args):
    key = model_name_key(args)
    if key in read_json(args.detector_config_path) and not args.overwrite:
        print(f"Detector for {key} already exists in {args.detector_config_path}. Skipping calibration.")
        return

    samples = build_samples(args.calibrate_path, tokenizer, args)
    val_samples = build_samples(args.val_calibrate_path, tokenizer, args)

    # compute the similarity score for each head
    # scores shape = (num_layers, num_heads)
    scores = load_head_scores(args) if args.from_cache else calibrate_head_scores(model, tokenizer, samples, args)

    # compute real attention vectors for each data sample by each top-k config
    heads_by_k, records_by_k, top_ks = detector_records_by_k(model, tokenizer, val_samples, scores, args)
    # heads_by_k = {k: [{"layer": 0, "head": 0, "score": 1.0}, ...], ...}
    # records_by_k = {k: [{"real": tensor, "ideal": tensor}, ...], ...}
    # top_ks = [1, 2, 4, 8, 16] or user-specified

    # compute leakage cosine or non-leakage top-5-mean penalty for all top-k/window configs
    all_metrics = []
    for k in top_ks:
        rows = [{**row, "model": key, "top_k": k} for row in shape_metrics(records_by_k[k], args)]
        all_metrics.extend(rows)

    best = choose_config(all_metrics, args)
    best["top_k_values"] = top_ks
    write_jsonl(all_metrics, args.all_experiments_path, args.overwrite)
    save_detector_config(heads_by_k[best["top_k"]], best, args)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    for path in (args.score_cache_path, args.all_experiments_path, args.detector_config_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        model=args.model,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    calibrate(model, tokenizer, args)


if __name__ == "__main__":
    main()
