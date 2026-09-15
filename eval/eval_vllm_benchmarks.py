from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

from tqdm.auto import tqdm


BASE = Path(__file__).parent

BENCHMARKS = {
    "math500": BASE / "data" / "math500.jsonl",
    "aime": BASE / "data" / "aime.jsonl",
}

SYSTEM_PROMPT = (
    "You are a careful math problem solver. Solve the problem step by step, "
    "then put only the final answer inside \\boxed{}."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vLLM-hosted models on boxed-answer JSONL benchmarks.")
    parser.add_argument("--model", required=True, help="Model name served by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--output-dir", type=Path, default=BASE / "output" / "benchmark_eval")
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def read_jsonl_loose(path: Path, repair: bool = False) -> list[dict]:
    if not path.exists():
        return []
    rows, valid_lines, broken = [], [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
            valid_lines.append(line)
        except json.JSONDecodeError:
            broken = True
    if repair and broken:
        path.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""), encoding="utf-8")
    return rows


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def model_key(model: str) -> str:
    return Path(model.rstrip("/\\")).name


def braced_blocks(text: str, command: str = r"\boxed") -> list[str]:
    blocks = []
    pos = 0
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


def boxed_answer(text) -> str:
    text = str(text)
    boxes = braced_blocks(text)
    return boxes[-1] if boxes else text


def normalize_answer(text) -> str:
    text = boxed_answer(text).strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    return re.sub(r"\s+", "", text)


def existing_prompts(path: Path) -> set[str]:
    return {row.get("prompt", "") for row in read_jsonl_loose(path, repair=True)}


def extra_body(args) -> dict | None:
    if not args.disable_thinking:
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}


def openai_client(args):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI package with: pip install openai") from exc
    return AsyncOpenAI(base_url=args.base_url, api_key=args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY")


async def complete(client, args, prompt: str) -> str:
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    body = extra_body(args)
    if body:
        request["extra_body"] = body
    response = await client.chat.completions.create(**request)
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
    return (reasoning + (message.content or "")).strip()


async def evaluate_benchmark(client, args, name: str, path: Path) -> dict:
    rows = read_jsonl(path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    out_path = args.output_dir / f"{model_key(args.model)}_{name}.jsonl"
    if args.overwrite and out_path.exists():
        out_path.unlink()
    done = existing_prompts(out_path)
    pending = [(i, row) for i, row in enumerate(rows) if row["prompt"] not in done]

    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def run_one(item):
        index, row = item
        async with semaphore:
            output = await complete(client, args, row["prompt"])
        pred = boxed_answer(output)
        gold = boxed_answer(row["ground_truth"])
        result = {
            "index": index,
            "benchmark": name,
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
            "output": output,
            "prediction": pred,
            "correct": normalize_answer(pred) == normalize_answer(gold),
        }
        append_jsonl(out_path, result)
        return result

    tasks = [asyncio.create_task(run_one(item)) for item in pending]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=name):
        await task

    results = read_jsonl_loose(out_path, repair=True)
    correct = sum(bool(row.get("correct")) for row in results)
    total = len(results)
    return {"benchmark": name, "path": str(path), "total": total, "correct": correct, "accuracy": correct / max(total, 1)}


async def main_async():
    args = parse_args()
    client = openai_client(args)
    summaries = []
    try:
        for name in args.benchmarks:
            if name not in BENCHMARKS:
                raise KeyError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}")
            summaries.append(await evaluate_benchmark(client, args, name, BENCHMARKS[name]))
    finally:
        await client.close()

    summary_path = args.output_dir / f"{model_key(args.model)}_summary.json"
    write_json(summary_path, {"model": args.model, "benchmarks": summaries})
    for row in summaries:
        print(f"{row['benchmark']}: accuracy={row['accuracy']:.4f} ({row['correct']}/{row['total']})")
    print(f"Saved summary to {summary_path}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
