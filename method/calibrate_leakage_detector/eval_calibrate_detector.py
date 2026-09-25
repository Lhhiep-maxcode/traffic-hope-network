from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from calibrate_detector import (
    aggregate_selected_heads,
    batches,
    build_samples,
    clear_memory,
    cosine,
    get_batch_attention_weights,
    model_name_key,
    read_detector,
    rolling_max,
    take_sample_attentions,
    token_spans,
)
from utils.utils import load_model_and_tokenizer, read_json, torch_dtype, write_json

EPS = 1e-12


def parse_args():
    base = Path(__file__).parent
    data = base / "data"
    output = base / "output"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key")
    parser.add_argument("--test-path", type=Path, default=data / "test-generated-responses-with-leakage-spans.jsonl")
    parser.add_argument("--detector-config-path", type=Path, default=output / "detector_config.json")
    parser.add_argument("--plot-dir", type=Path, default=output / "lowest_score_plots")
    parser.add_argument("--plot-lowest-n", type=int, default=0)
    parser.add_argument("--aggregation", choices=["weighted", "mean"], default="weighted")
    parser.add_argument("--threshold-tuning", action="store_true")
    parser.add_argument("--threshold-steps", type=int, default=200)
    parser.add_argument("--min-span-recall", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def span_f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / max(precision + recall, EPS)


def save_test_metrics(args, metrics: dict):
    if args.model_key:
        key = args.model_key
    else:
        key = model_name_key(args)
    config = read_json(args.detector_config_path)
    if key not in config:
        raise KeyError(f"No detector config for {key} in {args.detector_config_path}.")
    config[key].update({
        "threshold": metrics["threshold"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_full_recall": metrics["full_recall"],
        "test_f1": metrics["f1"],
        "test_token_precision": metrics["token_precision"],
        "test_avg_pred_spans": metrics["avg_pred_spans"],
        "threshold_selection_min_recall": args.min_span_recall,
    })
    write_json(config, args.detector_config_path)

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


def pooled_threshold_metrics(records: list[dict], threshold: float) -> dict:
    total = {
        "detected_gold_spans": 0,
        "missed_gold_spans": 0,
        "correct_pred_spans": 0,
        "false_pred_spans": 0,
        "full_hits": 0,
        "correct_pred_tokens": 0,
        "predicted_tokens": 0,
    }
    for record in records:
        stats, _ = metric_row(record["real"], record["ideal"], threshold)
        total["detected_gold_spans"] += stats["detected_gold_spans"]
        total["missed_gold_spans"] += stats["missed_gold_spans"]
        total["correct_pred_spans"] += stats["correct_pred_spans"]
        total["false_pred_spans"] += stats["false_pred_spans"]
        total["full_hits"] += int(stats["full_recall"] == 1.0)
        total["correct_pred_tokens"] += stats["correct_pred_tokens"]
        total["predicted_tokens"] += stats["predicted_tokens"]

    precision = total["correct_pred_spans"] / max(
        total["correct_pred_spans"] + total["false_pred_spans"], 1
    )
    recall = total["detected_gold_spans"] / max(
        total["detected_gold_spans"] + total["missed_gold_spans"], 1
    )
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": span_f1(precision, recall),
        "full_recall": total["full_hits"] / max(len(records), 1),
        "avg_pred_spans": (total["correct_pred_spans"] + total["false_pred_spans"]) / max(len(records), 1),
        "token_precision": total["correct_pred_tokens"] / max(total["predicted_tokens"], 1),
        **total,
    }

def harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / max(a + b, EPS)


def best_config_criteria(row: dict) -> tuple:
    aug_precision = harmonic_mean(row["precision"], row["token_precision"])
    score = harmonic_mean(aug_precision, row["recall"])
    return (
        score,
        row['precision'],
        row['threshold'],
    )

def choose_threshold(records: list[dict], args) -> dict:
    max_score = max(float(record["real"].max()) for record in records)
    thresholds = [0.0] if max_score <= 0 else torch.linspace(0, max_score, args.threshold_steps).tolist()
    rows = [pooled_threshold_metrics(records, threshold) for threshold in sorted(set(thresholds))]
    feasible = [row for row in rows if row["recall"] >= args.min_span_recall]
    if feasible:
        return max(feasible, key=best_config_criteria)
    return max(rows, key=lambda row: (row["recall"], best_config_criteria(row)[0], row["precision"], row["threshold"]))


def plot_sample(
    real: torch.Tensor,
    ideal: torch.Tensor,
    pred: torch.Tensor,
    score: float,
    stats: dict,
    out_path: Path,
    threshold: float,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(range(len(real)), real.numpy(), where="mid", label="detector score")
    ax.axhline(threshold, color="tab:red", linestyle=":", label="threshold")

    for i, span in enumerate(token_spans(ideal.bool())):
        ax.axvspan(
            span["start_token"] - 0.5,
            span["end_token"] - 0.5,
            color="tab:orange",
            alpha=0.18,
            label="gold leakage span" if i == 0 else None,
        )
    for i, span in enumerate(token_spans(pred.bool())):
        ax.axvspan(
            span["start_token"] - 0.5,
            span["end_token"] - 0.5,
            color="tab:blue",
            alpha=0.08,
            label="predicted span" if i == 0 else None,
        )

    ax.set_title(
        f"cosine={score:.4f} "
        f"span_precision={stats['precision']:.4f} "
        f"span_recall={stats['recall']:.4f} full_recall={stats['full_recall']:.0f}"
    )
    ax.set_xlabel("output token index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def evaluate(model, tokenizer, args):
    detector = read_detector(args)
    samples = [(i, s) for i, s in enumerate(build_samples(args.test_path, tokenizer, args), 1)]
    heads = detector["heads"]
    window_size = int(detector["window_size"])
    aggregation = detector.get("aggregation", args.aggregation)

    records = []
    for batch in tqdm(list(batches(samples, args.batch_size)), desc="Evaluating"):
        batch_attn = get_batch_attention_weights(model, tokenizer, [sample for _, sample in batch])
        for batch_idx, (sample_id, sample) in enumerate(batch):
            sample_attn = take_sample_attentions(batch_attn, batch_idx, len(sample["input_ids"]))
            real = rolling_max(aggregate_selected_heads(sample_attn, sample, heads, aggregation), window_size)
            score = float(cosine(real, sample["ideal"]))
            records.append({
                "sample_id": sample_id,
                "real": real.cpu(),
                "ideal": sample["ideal"].cpu(),
                "score": score,
            })
        del batch_attn
        clear_memory()

    if args.threshold_tuning:
        best = choose_threshold(records, args)
        threshold = best["threshold"]
    else:
        if detector.get("threshold") is None:
            raise ValueError("No saved threshold found. Run with --threshold-tuning first.")
        threshold = float(detector["threshold"])
        best = pooled_threshold_metrics(records, threshold)

    for rank, record in enumerate(sorted(records, key=lambda row: row["score"])[: args.plot_lowest_n], 1):
        stats, pred = metric_row(record["real"], record["ideal"], threshold)
        plot_sample(
            record["real"],
            record["ideal"],
            pred,
            record["score"],
            stats,
            args.plot_dir / f"lowest_{rank:02d}_sample_{record['sample_id']:04d}.png",
            threshold,
        )

    save_test_metrics(args, best)
    print(
        f"threshold={threshold:.6g} f1={best['f1']:.4f} "
        f"precision={best['precision']:.4f} recall={best['recall']:.4f} "
        f"full_recall={best['full_recall']:.4f} "
        f"avg_pred_spans={best['avg_pred_spans']:.4f} "
        f"token_precision={best['token_precision']:.4f}"
    )


def main():
    args = parse_args()
    if args.batch_size < 1 or args.threshold_steps < 1:
        raise ValueError("--batch-size and --threshold-steps must be positive.")
    if not 0 <= args.min_span_recall <= 1:
        raise ValueError("--min-span-recall must be between 0 and 1.")

    model, tokenizer = load_model_and_tokenizer(
        model=args.model,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    evaluate(model, tokenizer, args)


if __name__ == "__main__":
    main()
