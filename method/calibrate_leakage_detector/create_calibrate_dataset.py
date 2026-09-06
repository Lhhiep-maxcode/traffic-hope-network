from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from prompt import SYSTEM_PROMPT, USER_PROMPT
from tqdm import tqdm
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import read_jsonl, write_jsonl



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["generate", "judge", "all"])
    parser.add_argument("--generation-model", nargs="+", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--dataset", default="Hiepppp/reasoning")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--problem-field", default="question")
    parser.add_argument("--solution-field", default="solution")
    parser.add_argument("--splitter", default=None)
    parser.add_argument("--answer-field", default="ground_truth")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--generation-base-url", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--generation-api-key", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--judge-max-new-tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    name = re.split(r"[\\/]", model_name.rstrip("\\/"))[-1]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return name[:1].lower() + name[1:]


def model_tag(model_names: list[str]) -> str:
    return "-".join(model_slug(name) for name in model_names)


def default_output_path(args) -> Path:
    if args.phase == "judge":
        return args.input_path.with_name(f"{args.input_path.stem}-calibrate-dataset.jsonl")
    suffix = "responses" if args.phase == "generate" else "calibrate-dataset"
    return Path(__file__).with_name("data") / f"{model_tag(args.generation_model)}-{suffix}.jsonl"


def braced_blocks(text: str, command: str) -> list[str]:
    blocks = []
    for match in re.finditer(re.escape(command), text):
        pos = match.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            continue

        depth, chars = 0, []
        for char in text[pos + 1 :]:
            if char == "{" and (not chars or chars[-1] != "\\"):
                depth += 1
            elif char == "}" and (not chars or chars[-1] != "\\"):
                if depth == 0:
                    blocks.append("".join(chars).strip())
                    break
                depth -= 1
            chars.append(char)
    return blocks


def parse_answer(example: dict, solution_field: str, answer_field: str, splitter: str) -> str | None:
    answer = example.get(answer_field)
    if answer is not None:
        return str(answer).strip()
    if splitter and splitter in example.get(solution_field, ""):
        return str(example[solution_field]).split(splitter)[-1].strip()
    boxes = braced_blocks(str(example.get(solution_field, "")), r"\boxed")
    return boxes[-1] if boxes else None


def load_seed_examples(args) -> list[dict]:
    from datasets import load_dataset

    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    dataset = load_dataset(*dataset_args, split=args.split)
    dataset = dataset.shuffle(seed=42)

    seeds = []
    for example in dataset:
        answer = parse_answer(example, args.solution_field, args.answer_field, args.splitter)
        problem = example.get(args.problem_field)
        if problem and answer:
            seeds.append({"problem": str(problem), "answer": answer})
        if args.max_examples and len(seeds) >= args.max_examples:
            break
    return seeds


def build_prompt(problem: str, answer: str) -> dict:
    return {
        'prompt': f"{problem} \nGiven the ground truth answer is $\\boxed{{{answer}}}$.",
        'privileged_context': f"\nGiven the ground truth answer is $\\boxed{{{answer}}}$."
    }


def leakage_spans(judge_output: str, response: str) -> list[str]:
    spans = []
    for span in braced_blocks(judge_output, r"\text"):
        span = span.strip().strip('"').strip("'")
        if span in response and span not in spans:
            spans.append(span)
    return spans


def leakage_label(judge_output: str) -> str:
    for box in braced_blocks(judge_output, r"\boxed"):
        match = re.fullmatch(r"\s*(FAILED|PASS)\s*", box.upper())
        if match:
            return match.group(1)
    return "FAILED"



def openai_client(base_url: str, api_key: str):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI Python package first: pip install openai") from exc
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def thinking_extra_body(enable_thinking: bool | None):
    if enable_thinking is False:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


async def close_client(client):
    result = client.close()
    if asyncio.iscoroutine(result):
        await result


async def complete(
    client,
    model: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    extra_body = thinking_extra_body(enable_thinking)
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = await client.chat.completions.create(**kwargs)
    message = response.choices[0].message

    reasoning = (
        getattr(message, "reasoning", None)
        or getattr(message, "reasoning_content", None)
        or ""
    )
    content = message.content or ""

    return (reasoning + content).strip()


async def limited_map(items: list, fn, max_concurrency: int, desc: str) -> list:
    semaphore = asyncio.Semaphore(max_concurrency)
    results = [None] * len(items)

    async def run(i, item):
        async with semaphore:
            results[i] = await fn(item)

    tasks = [asyncio.create_task(run(i, item)) for i, item in enumerate(items)]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        await task
    return results


async def generate_rows(args) -> list[dict]:
    seeds = load_seed_examples(args)
    print(f"Loaded {len(seeds)} seed examples from {args.dataset}/{args.split}")

    base_url = args.generation_base_url or args.base_url
    api_key = args.generation_api_key or args.api_key
    client = openai_client(base_url, api_key)
    rows = []

    try:
        for model_name in args.generation_model:
            prompts = [build_prompt(seed["problem"], seed["answer"]) for seed in seeds]

            async def generate_one(prompt: dict):
                response = await complete(
                    client,
                    model_name,
                    prompt["prompt"],
                    args.generation_max_new_tokens,
                    args.generation_temperature,
                    enable_thinking=None if not args.disable_thinking else False,
                )
                return {
                    "generation_model": model_name, 
                    "prompt": prompt["prompt"], 
                    "privileged_context": prompt["privileged_context"], 
                    "response": response
                }

            rows.extend(
                await limited_map(
                    prompts,
                    generate_one,
                    args.max_concurrency,
                    desc=f"Generating {model_slug(model_name)}",
                )
            )
    finally:
        await close_client(client)
    return rows


async def judge_rows(args, rows: list[dict], output_path: Path):
    base_url = args.judge_base_url or args.base_url
    api_key = args.judge_api_key or args.api_key
    client = openai_client(base_url, api_key)

    async def judge_one(row: dict):
        judge_prompt = USER_PROMPT.format(prompt=row["prompt"], response=row["response"])
        judge_output = await complete(
            client,
            args.judge_model,
            judge_prompt,
            args.judge_max_new_tokens,
            temperature=0.0,
            system_prompt=SYSTEM_PROMPT,
            enable_thinking=False,
        )
        label = leakage_label(judge_output)
        return {
            "generation_model": row["generation_model"],
            "prompt": row["prompt"],
            "response": row["response"],
            "privileged_context": row["privileged_context"],
            "leakage": label == "FAILED",
        }

    try:
        judged_rows = await limited_map(rows, judge_one, args.max_concurrency, desc="Detecting leakage")
    finally:
        await close_client(client)
    write_jsonl(judged_rows, output_path)


async def async_main():
    args = parse_args()
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")
    if args.phase in {"generate", "all"} and not args.generation_model:
        raise ValueError("--generation-model is required for --phase generate/all")
    if args.phase == "judge" and args.input_path is None:
        raise ValueError("--input-path is required for --phase judge")
    if args.phase == "judge" and not args.judge_model:
        raise ValueError("--judge-model is required for --phase judge")
    if args.phase == "all":
        args.judge_model = args.judge_model or args.generation_model[0]

    output_path = args.output_path or default_output_path(args)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")

    if args.phase == "generate":
        write_jsonl(await generate_rows(args), output_path, args.overwrite)
    elif args.phase == "judge":
        await judge_rows(args, read_jsonl(args.input_path), output_path)
    else:
        await judge_rows(args, await generate_rows(args), output_path)

    print(f"Saved {output_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
