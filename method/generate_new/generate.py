"""Simple CLI for leakage-aware generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import load_model_and_tokenizer, read_jsonl
from method.generate_new import (
    AttentionLeakageDetector,
    DecodeSettings,
    LeakageSafeGenerator,
    RepairSettings,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--model-key", default=None)
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=ROOT / "method/calibrate_leakage_detector/output/detector_config.json",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="ground_truth")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--comparison-method",
        choices=["attention_score", "js_divergence"],
        default="attention_score",
    )
    parser.add_argument("--js-threshold", type=float, default=0.1)
    parser.add_argument("--max-repair-steps", type=int, default=1000000)
    parser.add_argument("--max-repair-cycles-per-span", type=int, default=2)
    parser.add_argument(
        "--clean-backtrack",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--prevent-infinite-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--wait-safe-window",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--include-unfixed", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_prompt(row: dict, args) -> dict:
    if "prompt_wo_answer" in row and "privileged_context" in row:
        clean_prompt = str(row["prompt_wo_answer"])
        context = str(row["privileged_context"])
        privileged_prompt = row.get("prompt_w_answer") or clean_prompt + context
    elif "prompt" in row and "privileged_context" in row:
        privileged_prompt = str(row["prompt"])
        context = str(row["privileged_context"])
        clean_prompt = privileged_prompt.replace(context, "", 1)
    else:
        question = row.get(args.question_field)
        answer = row.get(args.answer_field)
        if not isinstance(question, str) or answer is None:
            raise ValueError("Each row needs prompt fields or question/answer fields.")
        clean_prompt = question.strip() + "\n\nExplain your solution step by step."
        context = f" Given the ground truth answer is {answer}"
        privileged_prompt = clean_prompt + context

    return {
        "clean_prompt": clean_prompt,
        "privileged_context": context,
        "privileged_prompt": privileged_prompt,
    }


def write_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Pass --overwrite.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output.exists():
        args.output.unlink()

    model_key = args.model_key or Path(str(args.model).rstrip("/\\")).name
    detector = AttentionLeakageDetector.from_file(args.detector_config, model_key)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
    )

    rows = tqdm(read_jsonl(args.input), total=args.limit, desc="Generating")
    decode = DecodeSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )
    repair = RepairSettings(
        comparison_method=args.comparison_method,
        js_threshold=args.js_threshold,
        max_steps=args.max_repair_steps,
    )
    generator = LeakageSafeGenerator(
        model=model,
        tokenizer=tokenizer,
        detector=detector,
        decode=decode,
        repair=repair,
    )
    for index, row in enumerate(rows):
        if args.limit is not None and index >= args.limit:
            break
        prompt = prepare_prompt(row, args)
        result = generator.generate(
            clean_prompt=prompt["clean_prompt"],
            privileged_prompt=prompt["privileged_prompt"],
            privileged_context=prompt["privileged_context"],
            include_unfixed=args.include_unfixed,
            show_progress=False,
        )
        write_jsonl(
            args.output,
            {
                **row,
                "sample_index": index,
                "prompt_wo_answer": prompt["clean_prompt"],
                "prompt_w_answer": prompt["privileged_prompt"],
                "privileged_context": prompt["privileged_context"],
                **result,
            },
        )


if __name__ == "__main__":
    main()
