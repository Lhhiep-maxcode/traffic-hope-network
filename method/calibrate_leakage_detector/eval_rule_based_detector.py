from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from calibrate_detector import build_sample, metric_row, overlapping_tokens, token_spans
except ImportError:
    from method.calibrate_leakage_detector.calibrate_detector import (
        build_sample,
        metric_row,
        overlapping_tokens,
        token_spans,
    )
from utils.utils import read_jsonl, write_json


VALUE_RE = re.compile(
    r"(\\boxed\{[^{}]*\}|\$[^$\n]{1,120}\$|\*\*[^*\n]{1,80}\*\*|[-+]?\d[\d,./:%-]*)"
)
VALUE_PATTERN = r"(?:\\boxed\{[^{}]{1,80}\}|\$[^$\n]{1,120}\$|\*\*[^*\n]{1,80}\*\*|[-+]?\d[\d,./:%-]*)"


def parse_args():
    base = Path(__file__).parent
    data = base / "data"
    output = base / "output"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Tokenizer id/path used for token-level metrics.")
    parser.add_argument("--calibrate-path", type=Path, default=data / "generated-responses-with-leakage-spans.jsonl")
    parser.add_argument("--input-path", type=Path, default=data / "test-generated-responses-with-leakage-spans.jsonl")
    parser.add_argument("--output-path", type=Path, default=output / "rule_based_metrics.json")
    parser.add_argument("--rules-output-path", type=Path, default=output / "rule_based_rules.json")
    parser.add_argument("--predictions-path", type=Path, default=None)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--max-rules", type=int, default=None)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def enough_text(text: str, args) -> bool:
    return len(text) >= args.min_chars and len(re.findall(r"\w+", text)) >= args.min_words


def escape_spaces(text: str) -> str:
    return re.escape(text).replace(r"\ ", r"\s+")


def exact_regex(text: str) -> str:
    return rf"(?<!\w){escape_spaces(text)}(?!\w)"


def generalized_regex(text: str) -> str | None:
    pieces, last = [], 0
    for match in VALUE_RE.finditer(text):
        pieces.append(escape_spaces(text[last : match.start()]))
        pieces.append(VALUE_PATTERN)
        last = match.end()
    if not pieces:
        return None
    pieces.append(escape_spaces(text[last:]))
    return rf"(?<!\w){''.join(pieces)}(?!\w)"


def add_rule(rules: dict[str, dict], pattern: str, kind: str, source: str):
    rule = rules.setdefault(pattern, {"pattern": pattern, "kind": kind, "support": 0, "examples": []})
    rule["support"] += 1
    if len(rule["examples"]) < 5 and source not in rule["examples"]:
        rule["examples"].append(source)


def learn_rules(args) -> list[dict]:
    rules = {}
    for row in read_jsonl(args.calibrate_path):
        for span in row.get("leakage_spans", []):
            text = normalize_text(span)
            if not enough_text(text, args):
                continue
            add_rule(rules, exact_regex(text), "exact", text)
            if not args.exact_only:
                pattern = generalized_regex(text)
                if pattern and pattern != exact_regex(text):
                    add_rule(rules, pattern, "generalized", text)

    learned = [rule for rule in rules.values() if rule["support"] >= args.min_support]
    learned.sort(key=lambda r: (r["support"], len(r["pattern"])), reverse=True)
    return learned[: args.max_rules] if args.max_rules else learned


def merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged = []
    for start, end in sorted(spans):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def rule_char_spans(response: str, rules: list[dict]) -> list[tuple[int, int]]:
    spans = []
    for rule in rules:
        pattern = rule["compiled"]
        for match in pattern.finditer(response):
            spans.append(trim_span(response, match.start(), match.end()))
    return merge_spans(spans)


def rule_token_mask(row: dict, sample: dict, rules: list[dict]) -> tuple[torch.Tensor, list[str]]:
    pred = torch.zeros_like(sample["ideal"], dtype=torch.bool)
    char_spans = rule_char_spans(row["response"], rules)

    for start, end in char_spans:
        full_start = len(sample["prompt_text"]) + start
        for token_idx in overlapping_tokens(sample["offsets"], full_start, full_start + end - start):
            if token_idx >= sample["prompt_len"]:
                pred[token_idx - sample["prompt_len"]] = True

    pred_text_spans = [row["response"][start:end] for start, end in char_spans]
    return pred, pred_text_spans


def add_stats(total: dict, stats: dict):
    for key in (
        "detected_gold_spans",
        "missed_gold_spans",
        "correct_pred_spans",
        "false_pred_spans",
        "correct_pred_tokens",
        "predicted_tokens",
    ):
        total[key] += stats[key]
    total["full_hits"] += int(stats["full_recall"] == 1.0)


def evaluate(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    rules = learn_rules(args)
    if not rules:
        raise RuntimeError(f"No usable rules learned from {args.calibrate_path}.")
    for rule in rules:
        rule["compiled"] = re.compile(rule["pattern"], re.IGNORECASE)

    rows = [row for row in read_jsonl(args.input_path) if row.get("leakage_spans")]
    if args.max_samples:
        rows = rows[: args.max_samples]

    total = {
        "detected_gold_spans": 0,
        "missed_gold_spans": 0,
        "correct_pred_spans": 0,
        "false_pred_spans": 0,
        "correct_pred_tokens": 0,
        "predicted_tokens": 0,
        "full_hits": 0,
        "samples": 0,
    }
    prediction_rows = []

    for sample_id, row in tqdm(list(enumerate(rows, 1)), desc="Rule baseline"):
        sample = build_sample(row, tokenizer, args)
        if sample is None:
            continue

        pred, pred_text_spans = rule_token_mask(row, sample, rules)
        stats, _ = metric_row(pred.float(), sample["ideal"], threshold=0.5)
        add_stats(total, stats)
        total["samples"] += 1

        if args.predictions_path:
            prediction_rows.append({
                "sample_id": sample_id,
                "predicted_leakage_spans": pred_text_spans,
                "predicted_token_spans": token_spans(pred),
                "metrics": stats,
            })

    metrics = {
        "method": "calibration_rule_based",
        "calibrate_path": str(args.calibrate_path),
        "input_path": str(args.input_path),
        "num_rules": len(rules),
        "samples": total["samples"],
        "span_precision": total["correct_pred_spans"] / max(total["correct_pred_spans"] + total["false_pred_spans"], 1),
        "span_recall": total["detected_gold_spans"] / max(total["detected_gold_spans"] + total["missed_gold_spans"], 1),
        "full_recall": total["full_hits"] / max(total["samples"], 1),
        "token_precision": total["correct_pred_tokens"] / max(total["predicted_tokens"], 1),
        "avg_pred_spans": (total["correct_pred_spans"] + total["false_pred_spans"]) / max(total["samples"], 1),
        **total,
    }
    write_json(metrics, args.output_path)
    write_json(
        {
            "method": "calibration_rule_based",
            "calibrate_path": str(args.calibrate_path),
            "min_support": args.min_support,
            "min_words": args.min_words,
            "min_chars": args.min_chars,
            "exact_only": args.exact_only,
            "rules": [{k: v for k, v in rule.items() if k != "compiled"} for rule in rules],
        },
        args.rules_output_path,
    )

    if args.predictions_path:
        args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_path.open("w", encoding="utf-8") as file:
            for row in prediction_rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("| Span precision | Span recall | Full recall | Token precision | Avg pred spans |")
    print("| :------------: | :---------: | :---------: | :-------------: | :------------: |")
    print(
        f"| {metrics['span_precision']:.4f} | {metrics['span_recall']:.4f} | "
        f"{metrics['full_recall']:.4f} | {metrics['token_precision']:.4f} | "
        f"{metrics['avg_pred_spans']:.4f} |"
    )
    print(f"learned_rules={len(rules)}")
    print(f"wrote {args.output_path}")
    print(f"wrote {args.rules_output_path}")


def main():
    evaluate(parse_args())


if __name__ == "__main__":
    main()
