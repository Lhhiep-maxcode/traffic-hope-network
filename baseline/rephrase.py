"""Rephrase responses with the same model through a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rephrase_prompt import REPHRASE_PROMPT, REPHRASE_SYSTEM_PROMPT
except ImportError:
    from baseline.rephrase_prompt import (
        REPHRASE_PROMPT,
        REPHRASE_SYSTEM_PROMPT,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model name served by vLLM.")
    parser.add_argument("--dataset", default="Hiepppp/reasoning")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--problem-field", default="question")
    parser.add_argument("--solution-field", default="solution")
    parser.add_argument("--answer-field", default="ground_truth")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10000000000)
    parser.add_argument("--splitter", default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--rewrite-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def braced_blocks(text: str, command: str = r"\boxed") -> list[str]:
    blocks, pos = [], 0
    while True:
        start = text.find(command, pos)
        if start < 0:
            return blocks
        pos = start + len(command)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            continue
        depth, chars = 0, []
        for i, char in enumerate(text[pos + 1 :], pos + 1):
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    blocks.append("".join(chars).strip())
                    pos = i + 1
                    break
                depth -= 1
            chars.append(char)


def parse_answer(example: dict, solution_field: str, answer_field: str, splitter: str | None):
    answer = example.get(answer_field)
    if answer is not None:
        return str(answer).strip()
    solution = str(example.get(solution_field, ""))
    if splitter and splitter in solution:
        return solution.split(splitter)[-1].strip()
    boxes = braced_blocks(solution)
    return boxes[-1] if boxes else None


def load_seed_examples(args) -> list[dict]:
    from datasets import load_dataset

    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    dataset = load_dataset(*dataset_args, split=args.split).shuffle(seed=42)
    dataset = dataset.select(range(args.start_index, args.end_index))

    rows = []
    for example in dataset:
        problem = example.get(args.problem_field)
        answer = parse_answer(
            example,
            args.solution_field,
            args.answer_field,
            args.splitter,
        )
        if problem and answer:
            rows.append({"question": str(problem), "answer": str(answer), "domain": example.get("domain")})
        if args.max_samples and len(rows) >= args.max_samples:
            break
    return rows


def processed_questions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    questions = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            content = row["messages"][0]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        if isinstance(content, str):
            questions.add(content.strip())
    return questions


def ground_truth_from_context(context: str):
    boxes = braced_blocks(context)
    if boxes:
        return boxes[-1]
    marker = "Given the ground truth answer is"
    start = context.find(marker)
    if start < 0:
        return None
    return context[start + len(marker) :].strip(" .\n\t")


def get_prompts(row: dict) -> tuple[str, str, str, str | None]:
    question = row.get("question")
    answer = row.get("ground_truth", row.get("answer"))
    if not isinstance(question, str) or answer is None:
        raise ValueError(
            "Each row needs prompt/privileged_context or question/ground_truth."
        )
    context = f"\nGiven the ground truth answer is $\\boxed{{{answer}}}$."
    clean_prompt = f"{question}"
    return clean_prompt, clean_prompt + context, context, str(answer)


def response_text(message: dict) -> str:
    reasoning = message.get("reasoning_content")
    content = message.get("content") or ""
    if reasoning:
        return f"<think>\n{reasoning.strip()}\n</think>\n{content}".strip()
    return content.strip()


def split_reasoning(text: str) -> tuple[str, str | None]:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if not match:
        return text.strip(), None
    reasoning = match.group(1).strip()
    content = (text[: match.start()] + text[match.end() :]).strip()
    return content, reasoning or None


def thinking_extra_body(enable_thinking: bool | None):
    if enable_thinking is False:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def message_field(message, name: str):
    value = getattr(message, name, None)
    if value is not None:
        return value
    return (getattr(message, "model_extra", None) or {}).get(name)


async def complete(
    client,
    args,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool | None,
    sampling: bool,
) -> dict:
    async with semaphore:
        request = {
            "model": args.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if sampling:
            request.update(
                {
                    "temperature": temperature,
                    "top_p": args.top_p,
                }
            )
            if args.top_k > 0:
                request["top_k"] = args.top_k
        else:
            request["temperature"] = 0.0

        extra_body = thinking_extra_body(enable_thinking)
        if extra_body:
            request["extra_body"] = extra_body

        result = await client.chat.completions.create(**request)
        message = result.choices[0].message
        content = message.content or ""
        reasoning = message_field(message, "reasoning") or message_field(
            message, "reasoning_content"
        )

        if not reasoning and "</think>" in content:
            reasoning = content.split("</think>", 1)[0].replace("<think>", "").strip()
            content = content.split("</think>", 1)[1].strip()
        return {
            "content": content,
            "reasoning_content": reasoning.strip() if reasoning else None,
        }


def parse_rewritten(text: str) -> str:
    marker = "<REWRITTEN_TEXT>"
    end_marker = "</REWRITTEN_TEXT>"
    start, end = text.find(marker), text.rfind(end_marker)
    if start < 0 or end < start:
        raise ValueError("Model did not return the required rewrite tags.")
    response = text[start + len(marker) : end].strip()
    if not response:
        raise ValueError("Model returned an empty rewritten response.")
    return response


async def rephrase(client, args, clean_prompt, context, response) -> str:
    messages = [
        {
            "role": "system",
            "content": REPHRASE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": REPHRASE_PROMPT.format(
                prompt=clean_prompt,
                context=context,
                response=response,
            ),
        },
    ]
    output = await complete(
        client,
        args,
        messages,
        args.rewrite_max_tokens,
        temperature=0.0,
        enable_thinking=False,
        sampling=False,
    )
    return parse_rewritten(response_text(output))


async def process_row(client, args, row):
    clean_prompt, full_prompt, privileged_context, ground_truth = get_prompts(row)
    generated = await complete(
        client,
        args,
        [{"role": "user", "content": full_prompt}],
        args.max_new_tokens,
        args.temperature,
        enable_thinking=True if not args.disable_thinking else False,
        sampling=True,
    )
    content = generated['content']
    reasoning = generated['reasoning_content']
    try:
        if content:
            content = await rephrase(client, args, clean_prompt, privileged_context, content)
        if reasoning:
            reasoning = await rephrase(client, args, clean_prompt, privileged_context, reasoning)
        return {
            "messages": [
                {
                    "role": "user",
                    "content": clean_prompt,
                },
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
            ],
            "privileged_context": privileged_context,
            "ground_truth": ground_truth,
            "domain": row.get("domain"),
        }
    except Exception as e:
        return None


async def run(args):
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1.")
    chunk_size = args.chunk_size or args.max_concurrency
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1.")

    rows = load_seed_examples(args)
    print(f"Loaded {len(rows)} seed examples from {args.dataset}/{args.split}")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_path.exists():
        args.output_path.unlink()
    processed_item = processed_questions(args.output_path)
    rows = [row for row in rows if row["question"].strip() not in processed_item]
    print(f"Loaded {len(rows)} unprocessed examples from {args.dataset}/{args.split}. Continuing to rephrase...")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI package with: pip install openai") from exc

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1500)
    global semaphore
    semaphore = asyncio.Semaphore(args.max_concurrency)

    try:
        with args.output_path.open("a", encoding="utf-8") as output:
            with tqdm(total=len(rows), desc="Rephrasing") as progress:
                for start in range(0, len(rows), chunk_size):
                    chunk = rows[start : start + chunk_size]
                    tasks = [
                        asyncio.create_task(process_row(client, args, row))
                        for row in chunk
                    ]
                    for task in asyncio.as_completed(tasks):
                        result = await task
                        if result is not None:
                            output.write(json.dumps(result, ensure_ascii=False) + "\n")
                            output.flush()
                        progress.update(1)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
