from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from tqdm import tqdm
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from method.calibrate_leakage_detector.prompt import JUDGE_SYSTEM_PROMPT, USER_PROMPT

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
    parser.add_argument("--generation-output-path", type=Path, default=None)
    parser.add_argument("--judge-output-path", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--generation-base-url", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--generation-api-key", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--generation-max-new-tokens", type=int, default=1024)
    parser.add_argument("--generation-temperature", type=float, default=0.7)
    parser.add_argument("--judge-max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Set this for reasoning judge models; temperature is then omitted.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def model_slug(model_name: str) -> str:
    name = re.split(r"[\\/]", model_name.rstrip("\\/"))[-1]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")
    return name[:1].lower() + name[1:]


def model_tag(model_names: list[str]) -> str:
    return "-".join(model_slug(name) for name in model_names)


def generation_output_path(args) -> Path:
    if args.generation_output_path:
        return args.generation_output_path
    if args.phase == "generate" and args.output_path:
        return args.output_path
    return Path(__file__).with_name("data") / f"{model_tag(args.generation_model)}-generated-responses.jsonl"


def judge_output_path(args) -> Path:
    if args.judge_output_path:
        return args.judge_output_path
    if args.phase in {"judge", "all"} and args.output_path:
        return args.output_path
    if args.phase == "judge":
        return args.input_path.with_name(f"{args.input_path.stem}-judged-leakage.jsonl")
    return Path(__file__).with_name("data") / f"{model_tag(args.generation_model)}-judged-leakage.jsonl"


def same_path(left: Path, right: Path) -> bool:
    return left.resolve() == right.resolve()


def prepare_output(path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    path.touch(exist_ok=True)


def read_jsonl_loose(path: Path, repair: bool = False) -> list[dict]:
    if not path or not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    valid_lines = []
    needs_repair = bool(text and not text.endswith(("\n", "\r")))
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
            valid_lines.append(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSONL line {line_no} in {path}")
            needs_repair = True
    if repair and needs_repair:
        path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
    return rows


def generation_key(row: dict) -> tuple:
    return row.get("generation_model"), row.get("prompt")


def judge_key(row: dict) -> tuple:
    return row.get("generation_model"), row.get("prompt"), row.get("response")


def existing_keys(path: Path, key_fn) -> set[tuple]:
    return {key_fn(row) for row in read_jsonl_loose(path, repair=True)}


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


def parse_spans(judge_output: str, response: str) -> list[str]:
    start, end = judge_output.find("["), judge_output.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Judge did not return a JSON array: {judge_output[:300]!r}")
    values = json.loads(judge_output[start : end + 1])
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



def openai_client(base_url: str, api_key: str):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI Python package first: pip install openai") from exc
    if base_url.rstrip("/") == "https://api.openai.com/v1" and api_key == "EMPTY":
        api_key = os.getenv("OPENAI_API_KEY") or api_key
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


async def stream_jsonl(items: list, fn, max_concurrency: int, desc: str, output_path: Path):
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run(item):
        async with semaphore:
            return await fn(item)

    tasks = [asyncio.create_task(run(item)) for item in items]
    with output_path.open("a", encoding="utf-8") as file:
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            row = await task
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()


async def generate_rows(args, output_path: Path) -> list[dict]:
    seeds = load_seed_examples(args)
    print(f"Loaded {len(seeds)} seed examples from {args.dataset}/{args.split}")
    done = existing_keys(output_path, generation_key)

    base_url = args.generation_base_url or args.base_url
    api_key = args.generation_api_key or args.api_key
    client = openai_client(base_url, api_key)
    items = []
    for model_name in args.generation_model:
        for seed in seeds:
            item = {"generation_model": model_name, **build_prompt(seed["problem"], seed["answer"])}
            if generation_key(item) not in done:
                items.append(item)
    print(f"Generation resume: {len(done)} existing rows, {len(items)} remaining rows")

    try:
        async def generate_one(item: dict):
            response = await complete(
                client,
                item["generation_model"],
                item["prompt"],
                args.generation_max_new_tokens,
                args.generation_temperature,
                enable_thinking=None if not args.disable_thinking else False,
            )
            return {
                "generation_model": item["generation_model"],
                "prompt": item["prompt"],
                "privileged_context": item["privileged_context"],
                "response": response,
            }

        if items:
            await stream_jsonl(items, generate_one, args.max_concurrency, "Generating", output_path)
    finally:
        await close_client(client)
    return read_jsonl_loose(output_path)


async def judge_rows(args, rows: list[dict], output_path: Path):
    done = {
        judge_key(row)
        for row in read_jsonl_loose(output_path, repair=True)
        if "leakage_spans" in row
    }
    rows = [row for row in rows if judge_key(row) not in done]
    print(f"Judge resume: {len(done)} existing rows, {len(rows)} remaining rows")

    base_url = args.judge_base_url or args.base_url
    api_key = args.judge_api_key or args.api_key
    client = openai_client(base_url, api_key)

    async def judge_one(row: dict):
        judge_prompt = USER_PROMPT.format(
            full_prompt=row["prompt"],
            context=row["privileged_context"],
            response=row["response"],
        )
        last_error = None
        for _ in range(args.judge_retries + 1):
            user_prompt = judge_prompt
            if last_error is not None:
                user_prompt = (
                    f"{judge_prompt}\n\n"
                    f"Previous judge attempt failed with error: {last_error}\n"
                    "Return only a valid JSON array of exact copied leakage spans."
                )
            request = {
                "model": args.judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": args.judge_max_new_tokens,
            }
            if args.judge_reasoning_effort:
                request["reasoning_effort"] = args.judge_reasoning_effort
            else:
                request["temperature"] = 0.0

            result = await client.chat.completions.create(**request)
            judge_output = result.choices[0].message.content or ""
            try:
                return {**row, "leakage_spans": parse_spans(judge_output, row["response"])}
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ValueError(f"Could not parse judge output after retries: {last_error}")

    try:
        if rows:
            await stream_jsonl(rows, judge_one, args.max_concurrency, "Detecting leakage", output_path)
    finally:
        await close_client(client)


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

    if args.phase == "generate":
        output_path = generation_output_path(args)
        prepare_output(output_path, args.overwrite)
        await generate_rows(args, output_path)
        print(f"Saved {output_path}")
    elif args.phase == "judge":
        output_path = judge_output_path(args)
        if same_path(args.input_path, output_path):
            raise ValueError("Judge output path must be different from --input-path.")
        prepare_output(output_path, args.overwrite)
        await judge_rows(args, read_jsonl_loose(args.input_path), output_path)
        print(f"Saved {output_path}")
    else:
        gen_path = generation_output_path(args)
        judged_path = judge_output_path(args)
        if same_path(gen_path, judged_path):
            raise ValueError("Generation and judge output paths must be different.")
        prepare_output(gen_path, args.overwrite)
        prepare_output(judged_path, args.overwrite)
        rows = await generate_rows(args, gen_path)
        await judge_rows(args, rows, judged_path)
        print(f"Saved generated rows to {gen_path}")
        print(f"Saved judged rows to {judged_path}")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
