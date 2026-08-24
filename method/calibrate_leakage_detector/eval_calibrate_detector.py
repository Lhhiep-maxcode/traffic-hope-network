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
    metric_row,
    model_name_key,
    read_detector,
    rolling_max,
    take_sample_attentions,
    token_spans,
)
from utils.utils import load_model_and_tokenizer, read_json, torch_dtype, write_json


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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def save_test_metrics(args, precision: float, recall: float, full_recall: float, token_precision: float, avg_pred_spans: float):
    if args.model_key:
        key = args.model_key
    else:
        key = model_name_key(args)
    config = read_json(args.detector_config_path)
    if key not in config:
        raise KeyError(f"No detector config for {key} in {args.detector_config_path}.")
    config[key].update({
        "test_precision": precision,
        "test_recall": recall,
        "test_full_recall": full_recall,
        "test_token_precision": token_precision,
        "test_avg_pred_spans": avg_pred_spans,
    })
    write_json(config, args.detector_config_path)


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
    ax.plot(real.numpy(), label="detector score")
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
    heads, threshold = detector["heads"], float(detector["threshold"])
    window_size = int(detector["window_size"])
    aggregation = detector.get("aggregation", args.aggregation)

    plots, total = [], {
        "detected_gold_spans": 0,
        "missed_gold_spans": 0,
        "correct_pred_spans": 0,
        "false_pred_spans": 0,
        "full_hits": 0,
        "correct_pred_tokens": 0,
        "predicted_tokens": 0,
    }
    for batch in tqdm(list(batches(samples, args.batch_size)), desc="Evaluating"):
        batch_attn = get_batch_attention_weights(model, tokenizer, [sample for _, sample in batch])
        for batch_idx, (sample_id, sample) in enumerate(batch):
            sample_attn = take_sample_attentions(batch_attn, batch_idx, len(sample["input_ids"]))
            real = rolling_max(aggregate_selected_heads(sample_attn, sample, heads, aggregation), window_size)
            score = float(cosine(real, sample["ideal"]))
            stats, pred = metric_row(real, sample["ideal"], threshold)
            total["detected_gold_spans"] += stats["detected_gold_spans"]
            total["missed_gold_spans"] += stats["missed_gold_spans"]
            total["correct_pred_spans"] += stats["correct_pred_spans"]
            total["false_pred_spans"] += stats["false_pred_spans"]
            total["full_hits"] += int(stats["full_recall"] == 1.0)
            total["correct_pred_tokens"] += stats["correct_pred_tokens"]
            total["predicted_tokens"] += stats["predicted_tokens"]
            if args.plot_lowest_n:
                plots.append((score, sample_id, real, sample["ideal"], pred, stats))
        del batch_attn
        clear_memory()

    for rank, (score, sample_id, real, ideal, pred, stats) in enumerate(sorted(plots)[: args.plot_lowest_n], 1):
        plot_sample(real, ideal, pred, score, stats, args.plot_dir / f"lowest_{rank:02d}_sample_{sample_id:04d}.png", threshold)

    precision = total["correct_pred_spans"] / max(total["correct_pred_spans"] + total["false_pred_spans"], 1)
    recall = total["detected_gold_spans"] / max(total["detected_gold_spans"] + total["missed_gold_spans"], 1)
    full_recall = total["full_hits"] / max(len(samples), 1)
    avg_pred_spans = (total["correct_pred_spans"] + total["false_pred_spans"]) / max(len(samples), 1)
    token_precision = total['correct_pred_tokens'] / total['predicted_tokens'] if total['predicted_tokens'] > 0 else 0.0
    save_test_metrics(args, precision, recall, full_recall, token_precision, avg_pred_spans)
    print(f"precision={precision:.4f} recall={recall:.4f} full_recall={full_recall:.4f} avg_pred_spans={avg_pred_spans:.4f} token_precision={token_precision:.4f}")


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

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
