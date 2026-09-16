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
    parser.add_argument("--threshold-steps", type=int, default=80)
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


def head_scores(attentions, sample: dict) -> torch.Tensor:
    raw = attention_to_context(attentions, sample["prompt_len"], sample["key_tokens"])
    # gold = sample["ideal"].bool()
    shape = cosine(raw, sample["ideal"])    # shape = (num_layers, num_heads)
    # pos = raw[..., gold].mean(dim=-1)
    # neg = raw[..., ~gold].mean(dim=-1) if (~gold).any() else torch.zeros_like(pos)
    return shape


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


def metric_row(real: torch.Tensor, ideal: torch.Tensor, threshold: float) -> tuple[dict, torch.Tensor]:
    pred, gold = real >= threshold, ideal.bool()
    gold_spans = token_spans(gold)
    pred_spans = token_spans(pred)
    gold_starts = [span["start_token"] for span in gold_spans]

    detected_gold_spans = sum(int(bool(pred[start])) for start in gold_starts)
    correct_pred_spans = sum(
        int(any(span["start_token"] <= start < span["end_token"] for start in gold_starts))
        for span in pred_spans
    )
    missed_gold_spans = len(gold_spans) - detected_gold_spans
    false_pred_spans = len(pred_spans) - correct_pred_spans
    correct_pred_tokens = int((pred & gold).sum())
    predicted_tokens = int(pred.sum())

    return {
        "detected_gold_spans": detected_gold_spans,
        "missed_gold_spans": missed_gold_spans,
        "total_gold_spans": len(gold_spans),
        "correct_pred_spans": correct_pred_spans,
        "false_pred_spans": false_pred_spans,
        "total_pred_spans": len(pred_spans),
        "correct_pred_tokens": correct_pred_tokens,
        "predicted_tokens": predicted_tokens,
        "precision": correct_pred_spans / max(len(pred_spans), 1),
        "recall": detected_gold_spans / max(len(gold_spans), 1),
        "full_recall": float(detected_gold_spans == len(gold_spans)),
    }, pred


def pooled_metrics(records: list[dict], window_size: int, threshold: float) -> dict:
    total = {
        "detected_gold_spans": 0,
        "missed_gold_spans": 0,
        "correct_pred_spans": 0,
        "false_pred_spans": 0,
        "correct_pred_tokens": 0,
        "predicted_tokens": 0,
        "full_hits": 0,
    }
    for record in records:
        stats, _ = metric_row(rolling_max(record["real"], window_size), record["ideal"], threshold)
        total["detected_gold_spans"] += stats["detected_gold_spans"]
        total["missed_gold_spans"] += stats["missed_gold_spans"]
        total["correct_pred_spans"] += stats["correct_pred_spans"]
        total["false_pred_spans"] += stats["false_pred_spans"]
        total["correct_pred_tokens"] += stats["correct_pred_tokens"]
        total["predicted_tokens"] += stats["predicted_tokens"]
        total["full_hits"] += int(stats["full_recall"] == 1.0)
    return {
        "window_size": window_size,
        "threshold": float(threshold),
        "precision": total["correct_pred_spans"] / max(
            total["correct_pred_spans"] + total["false_pred_spans"], 1
        ),
        "recall": total["detected_gold_spans"] / max(
            total["detected_gold_spans"] + total["missed_gold_spans"], 1
        ),
        "token_precision": total["correct_pred_tokens"] / max(total["predicted_tokens"], 1),
        "total_pred_spans": total["correct_pred_spans"] + total["false_pred_spans"],
        "avg_pred_spans_per_sample": (total["correct_pred_spans"] + total["false_pred_spans"]) / max(len(records), 1),
        "full_recall": total["full_hits"] / max(len(records), 1),
        **total,
        "samples": len(records),
    }


def threshold_metrics(records: list[dict], args) -> list[dict]:
    rows = []
    for window_size in int_grid(args.window_sizes):
        rolled = [(rolling_max(r["real"], window_size), r["ideal"]) for r in records]
        max_score = max(float(real.max()) for real, _ in rolled)
        thresholds = [0.0] if max_score <= 0 else torch.linspace(0, max_score, args.threshold_steps).tolist()
        # thresholds.append(min(float(real[ideal.bool()].min()) for real, ideal in rolled))
        rows.extend(pooled_metrics(records, window_size, threshold) for threshold in sorted(set(thresholds)))
    return rows


def harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / max(a + b, EPS)


def best_config_criteria(row: dict) -> tuple:
    aug_precision = harmonic_mean(row["precision"], row["token_precision"])
    score = harmonic_mean(aug_precision, row["recall"])
    return (
        score,
        -row.get("top_k", 0),
        -row["window_size"],
        row["threshold"],
    )


def choose_config(rows: list[dict], args) -> dict:
    feasible = [r for r in rows if r["recall"] >= 0.8 and r['threshold'] > 0]
    if feasible:
        return max(feasible, key=best_config_criteria)
    return max(
        rows,
        key=lambda r: (
            r["recall"],
            best_config_criteria(r)[0],
            -r.get("top_k", 0),
            -r["window_size"],
            r["threshold"],
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
    torch.save({"model": model_name_key(args), "scores": scores.cpu(), "used": used}, args.score_cache_path)
    return scores


def load_head_scores(args) -> torch.Tensor:
    cache = torch.load(args.score_cache_path, map_location="cpu")
    if cache.get("model") != model_name_key(args):
        warnings.warn(
            f"Cache is for {cache.get('model')}, but --model is {model_name_key(args)}.",
            UserWarning,
        )
    print(f"Loaded head scores from {args.score_cache_path}.")
    return cache["scores"]


def detector_records_by_k(model, tokenizer, samples: list[dict], scores: torch.Tensor, args):
    top_ks = int_grid(args.top_k_values, args.top_k, scores.numel())    # [1, 2, 4, 8, 16, ...]
    heads_by_k = {k: top_heads(scores, k) for k in top_ks}  # {k: [{"layer": 0, "head": 0, "score": 1.0}, ...], ...}
    records_by_k = {k: [] for k in top_ks}

    for batch in tqdm(list(batches(samples, args.batch_size)), desc="Calibrating k/window/threshold"):
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
        "threshold": best["threshold"],
        "val_calibration_precision": best["precision"],
        "val_calibration_recall": best["recall"],
        "val_calibration_full_recall": best["full_recall"],
        "val_selection_token_precision": best["token_precision"],
        "val_avg_pred_spans_per_sample": best["avg_pred_spans_per_sample"],
        "test_precision": None,
        "test_recall": None,
        "test_full_recall": None,
    }
    write_json(config, args.detector_config_path)
    print(
        f"top_k={best['top_k']} threshold={best['threshold']:.6g} window={best['window_size']} "
        f"span_precision={best['precision']:.4f} token_precision={best['token_precision']:.4f} "
        f"avg_pred_spans={best['avg_pred_spans_per_sample']:.2f} "
        f"recall={best['recall']:.4f}"
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

    # compute metrics for all posible config
    all_metrics = []
    for k in top_ks:
        rows = [{**row, "model": key, "top_k": k} for row in threshold_metrics(records_by_k[k], args)]
        all_metrics.extend(rows)

    best = choose_config(all_metrics, args)
    best["top_k_values"] = top_ks
    write_jsonl(all_metrics, args.all_experiments_path, args.overwrite)
    save_detector_config(heads_by_k[best["top_k"]], best, args)


def main():
    args = parse_args()
    if args.batch_size < 1 or args.threshold_steps < 1:
        raise ValueError("--batch-size and --threshold-steps must be positive.")
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
