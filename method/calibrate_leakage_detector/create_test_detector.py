"""Create natural counterfactual trajectories for iterative leakage detection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.generate.v3.self_coding import TokenByTokenGenerator
from method.calibrate_leakage_detector.prompt import JUDGE_SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", default=None)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Set this for reasoning models; temperature is then omitted.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backtrack-tokens", type=int, default=2)
    parser.add_argument("--min-repair-tokens", type=int, default=8)
    parser.add_argument("--max-repair-tokens", type=int, default=64)
    parser.add_argument("--safe-steps", type=int, default=3)
    parser.add_argument("--js-threshold", type=float, default=0.1)
    parser.add_argument("--max-repair-stages", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def model_key(model: str) -> str:
    return Path(model.rstrip("/\\")).name


def build_prompt(row: dict) -> tuple[str, str, str, str | None, str]:
    """Return clean prompt, full prompt, context, response, and model label."""
    if "prompt" in row and "privileged_context" in row:
        full_prompt = str(row["prompt"])
        context = str(row["privileged_context"])
        if not context or context not in full_prompt:
            raise ValueError("privileged_context must occur in prompt.")
        clean_prompt = full_prompt.replace(context, "", 1)
        return (
            clean_prompt,
            full_prompt,
            context,
            row.get("response"),
            str(row.get("generation_model", "")),
        )

    question = row.get("question")
    answer = row.get("ground_truth", row.get("answer"))
    if not isinstance(question, str) or answer is None:
        raise ValueError(
            "Each input row needs prompt/privileged_context or question/ground_truth."
        )
    answer = str(answer)
    context = f"\nGiven the ground truth answer is $\\boxed{{{answer}}}."
    clean_prompt = f"{question} "
    full_prompt = clean_prompt + context
    return clean_prompt, full_prompt, context, row.get("response"), ""


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    kwargs = {
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
    }
    if args.dtype != "auto":
        kwargs["dtype"] = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **kwargs,
    ).eval()
    return model, tokenizer


def make_generator(model, tokenizer, prompt, context, args, seed):
    return TokenByTokenGenerator(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        privileged_context=context,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=args.do_sample,
        seed=seed,
        enable_thinking=args.enable_thinking,
        reuse_kv_cache=True,
    )


def js_divergence(left: torch.Tensor, right: torch.Tensor) -> float:
    p = F.softmax(left.float(), dim=-1)
    q = F.softmax(right.float(), dim=-1)
    m = (p + q) / 2
    value = 0.5 * (
        F.kl_div(m.log(), p, reduction="batchmean")
        + F.kl_div(m.log(), q, reduction="batchmean")
    )
    return float(value.item())


def first_span_start(tokenizer, response: str, span: str) -> int:
    char_start = response.find(span)
    if char_start < 0:
        raise ValueError("The judged leakage span is not present in the response.")
    encoded = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
        if end > char_start:
            return token_index
    raise ValueError("Could not map the judged leakage span to a token.")


def response_ids(tokenizer, response: str) -> list[int]:
    return tokenizer(response, add_special_tokens=False)["input_ids"]


def repair_response(
    model,
    tokenizer,
    clean_prompt: str,
    privileged_context: str,
    response: str,
    first_span: str,
    args,
    seed: int,
) -> str:
    ids = response_ids(tokenizer, response)
    start = first_span_start(tokenizer, response, first_span)
    prefix_end = max(0, start - args.backtrack_tokens)
    prefix_ids = ids[:prefix_end]

    privileged = make_generator(
        model, tokenizer, clean_prompt, privileged_context, args, seed
    )
    clean = make_generator(model, tokenizer, clean_prompt, None, args, seed)
    privileged.start(prefix_ids)
    clean.start(prefix_ids)
    privileged._reset_rng()
    clean._reset_rng()

    safe_steps = 0
    generated_repair_tokens = 0
    while (
        not privileged.finished
        and not clean.finished
        and generated_repair_tokens < args.max_repair_tokens
    ):
        js = js_divergence(privileged.next_logits, clean.next_logits)
        token_id = int(clean._sample_token(clean.next_logits).item())

        clean_outputs = clean._forward_token(token_id)
        clean._accept(token_id, clean_outputs)
        privileged_outputs = privileged._forward_token(token_id)
        privileged._accept(token_id, privileged_outputs)
        generated_repair_tokens += 1

        safe_steps = safe_steps + 1 if js <= args.js_threshold else 0
        if (
            generated_repair_tokens >= args.min_repair_tokens
            and safe_steps >= args.safe_steps
        ):
            break

    while not privileged.finished and len(privileged.generated_ids) < args.max_new_tokens:
        privileged.step()

    return privileged.get_generated_text(skip_special_tokens=True)


def parse_spans(text: str, response: str) -> list[str]:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Judge did not return a JSON array: {text[:300]!r}")
    values = json.loads(text[start : end + 1])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Judge output must be a JSON array of strings.")

    spans = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        if value not in response:
            raise ValueError(f"Judge returned a span not copied from the response: {value!r}")
        if value not in spans:
            spans.append(value)
    return sorted(spans, key=response.find)


def openai_client(args):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI package with: pip install openai") from exc
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --api-key.")
    return AsyncOpenAI(base_url=args.base_url, api_key=api_key)


async def judge(client, args, full_prompt: str, context: str, response: str) -> list[str]:
    user_prompt = f"""
PROMPT WITH PRIVILEGED CONTEXT:
{full_prompt}

PRIVILEGED CONTEXT:
{context}

MODEL RESPONSE:
{response}
"""
    last_error = None
    for _ in range(args.judge_retries + 1):
        try:
            if last_error is not None:
                user_prompt = f"Previous judge attempt failed with error: {last_error}\n\nYour previous response was: \n{result.choices[0].message.content or ''}\n"
            request = {
                "model": args.judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": args.judge_max_tokens,
            }
            if args.judge_reasoning_effort:
                request["reasoning_effort"] = args.judge_reasoning_effort
            else:
                request["temperature"] = 0.0

            result = await client.chat.completions.create(**request)
            return parse_spans(result.choices[0].message.content or "", response)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Could not parse judge output after retries: {last_error}")


async def run(args):
    if args.max_repair_stages < 0 or args.max_repair_tokens < 1:
        raise ValueError("Repair limits must be positive.")
    if args.min_repair_tokens < 1 or args.safe_steps < 1:
        raise ValueError("--min-repair-tokens and --safe-steps must be positive.")
    if args.min_repair_tokens > args.max_repair_tokens:
        raise ValueError("--min-repair-tokens cannot exceed --max-repair-tokens.")
    if args.backtrack_tokens < 0 or args.judge_max_tokens < 1:
        raise ValueError("Token limits must be non-negative/positive as appropriate.")

    rows = read_jsonl(args.input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    model, tokenizer = load_model(args)
    client = openai_client(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_path.exists():
        args.output_path.unlink()

    try:
        with args.output_path.open("a", encoding="utf-8") as output:
            for sample_index, row in enumerate(tqdm(rows, desc="Creating test trajectories")):
                clean_prompt, full_prompt, context, response, source_model = build_prompt(row)
                generation_model = source_model or args.model_key or model_key(args.model)

                if not response:
                    generator = make_generator(
                        model,
                        tokenizer,
                        clean_prompt,
                        context,
                        args,
                        args.seed + sample_index,
                    )
                    response = generator.generate(
                        args.max_new_tokens,
                        show_progress=False,
                    )["text"]

                for stage in range(args.max_repair_stages + 1):
                    spans = await judge(client, args, full_prompt, context, response)
                    first_span = spans[0] if spans else None
                    first_span_end_index = response.find(first_span) + len(first_span) if first_span else -1
                    output_row = {
                        "generation_model": generation_model,
                        "prompt": full_prompt,
                        "privileged_context": context,
                        "response": response[:first_span_end_index],
                        "leakage_spans": [spans[0]] if spans else [],
                    }
                    output.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    output.flush()

                    if not spans or stage == args.max_repair_stages:
                        break

                    repaired = repair_response(
                        model,
                        tokenizer,
                        clean_prompt,
                        context,
                        response,
                        spans[0],
                        args,
                        args.seed + sample_index * 1000 + stage + 1,
                    )
                    if repaired == response:
                        break
                    response = repaired
    finally:
        await client.close()


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
