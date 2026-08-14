from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
calibrate_dir = project_root / "method" / "calibrate_leakage_detector"
calibrate_output_dir = calibrate_dir / "output"
for path in (project_root, calibrate_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import calibrate_detector as cd
from utils.utils import build_prompt, load_model_and_tokenizer

PRIVILEGED_MARKER = "Given the ground truth answer is "
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=calibrate_output_dir / "repaired_responses.jsonl")
    parser.add_argument("--detector-config-path", type=Path, default=calibrate_output_dir / "detector_config.json")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--repair-extra-tokens", type=int, default=4)
    parser.add_argument("--js-threshold", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def remove_privileged_context(prompt: str) -> str:
    start = prompt.find(PRIVILEGED_MARKER)
    return prompt if start < 0 else prompt[:start].strip()


def read_detector_config(path: Path, model: str) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    key = cd.model_name_key(model)
    if key not in config:
        raise KeyError(f"No detector config for {key} in {path}.")
    return config[key]


def output_token_spans_to_chars(sample: dict, token_spans: list[dict]) -> list[dict]:
    spans = []
    prompt_chars = len(sample["prompt_text"])
    for span in token_spans:
        first = sample["prompt_len"] + span["start_token"]
        last = sample["prompt_len"] + span["end_token"] - 1
        start_char = max(0, sample["offsets"][first][0] - prompt_chars)
        end_char = max(0, sample["offsets"][last][1] - prompt_chars)
        if end_char > start_char:
            spans.append({**span, "start_char": start_char, "end_char": end_char})
    return spans


def detect_token_spans(model, tokenizer, row: dict, detector: dict, args) -> tuple[list[dict], torch.Tensor | None]:
    sample_args = SimpleNamespace(disable_thinking=args.disable_thinking, max_seq_len=args.max_seq_len)
    sample = cd.build_inference_sample(row, tokenizer, sample_args)
    if sample is None:
        return [], None

    batch_attn = cd.get_batch_attention_weights(model, tokenizer, [sample])
    attentions = cd.take_sample_attentions(batch_attn, 0, len(sample["input_ids"]))
    scores = cd.aggregate_selected_heads(attentions, sample, detector["heads"], detector.get("aggregation", "weighted"))
    scores = cd.rolling_max(scores, int(detector.get("window_size", 1)))
    mask = scores >= float(detector["threshold"])
    spans = output_token_spans_to_chars(sample, cd.token_spans(mask))
    del batch_attn, attentions
    cd.clear_memory()
    return spans, scores


def prompt_with_prefix(tokenizer, prompt: str, prefix: str, disable_thinking: bool) -> str:
    return build_prompt(
        tokenizer,
        system_prompt="",
        user_prompt=prompt,
        encode=False,
        add_generation_prompt=True,
        enable_thinking=not disable_thinking,
    ) + prefix


def generate_replacement(model, tokenizer, text: str, max_new_tokens: int, args) -> str:
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(cd.model_device(model))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": cd.pad_token_id(tokenizer),
    }
    if args.temperature > 0:
        kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)
    new_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def next_logits(model, tokenizer, text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(cd.model_device(model))
    with torch.inference_mode():
        return model(**inputs, use_cache=False).logits[0, -1].float().cpu()


def js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.softmax(logits_a, dim=-1).clamp_min(EPS)
    q = F.softmax(logits_b, dim=-1).clamp_min(EPS)
    m = 0.5 * (p + q)
    js = 0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()
    return float(js)


def repair_row(model, tokenizer, row: dict, detector: dict, args) -> dict:
    spans, _ = detect_token_spans(model, tokenizer, row, detector, args)
    clean_prompt = remove_privileged_context(row["prompt"])
    repaired = row["response"]
    repairs = []
    delta = 0

    for span in spans:
        start_char = span["start_char"] + delta
        end_char = span["end_char"] + delta
        original = repaired[start_char:end_char]
        prefix = repaired[:start_char]
        max_new_tokens = max(1, span["end_token"] - span["start_token"] + args.repair_extra_tokens)
        clean_text = prompt_with_prefix(tokenizer, clean_prompt, prefix, args.disable_thinking)
        replacement = generate_replacement(model, tokenizer, clean_text, max_new_tokens, args)

        priv_next = prompt_with_prefix(tokenizer, row["prompt"], prefix + replacement, args.disable_thinking)
        clean_next = prompt_with_prefix(tokenizer, clean_prompt, prefix + replacement, args.disable_thinking)
        js = js_divergence(next_logits(model, tokenizer, priv_next), next_logits(model, tokenizer, clean_next))
        accepted = js <= args.js_threshold

        if accepted:
            repaired = repaired[:start_char] + replacement + repaired[end_char:]
            delta += len(replacement) - (end_char - start_char)

        repairs.append({
            **span,
            "original": original,
            "replacement": replacement,
            "js": js,
            "accepted": accepted,
        })
        cd.clear_memory()

    return {**row, "detected_token_spans": spans, "repairs": repairs, "repaired_response": repaired}


def main():
    args = parse_args()
    detector = read_detector_config(args.detector_config_path, args.model)
    model, tokenizer = load_model_and_tokenizer(
        model=args.model,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        dtype=torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    rows = [repair_row(model, tokenizer, row, detector, args) for row in tqdm(read_jsonl(args.input_path), desc="Repairing")]
    write_jsonl(rows, args.output_path, args.overwrite)


if __name__ == "__main__":
    main()
