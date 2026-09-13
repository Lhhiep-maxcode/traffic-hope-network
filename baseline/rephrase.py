"""Rephrase responses with the same model through a vLLM OpenAI server."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from tqdm.auto import tqdm


JUDGE_PROMPT = r"""
You are a strict leakage-span judge.

The prompt contains a privileged ground-truth answer. Find every smallest
contiguous substring in the model response that reveals awareness or use of
that privileged answer. Leakage includes referring to the given/reference
answer, checking a result against it, reasoning from it, or mentioning that it
should be hidden. Normal derivation and stating a correct answer are not
leakage by themselves.

Return only a valid JSON array of exact strings copied from the response.
Return [] if there is no leakage. Do not return explanations or Markdown.
"""


REPHRASE_PROMPT = r"""
Rewrite MODEL RESPONSE to remove the listed leakage spans.

PROBLEM WITHOUT PRIVILEGED CONTEXT:
{prompt}

Requirements:
- Preserve the mathematical/logical meaning, answer, and useful reasoning.
- Keep the response fluent and natural.
- Change as little surrounding text as possible.
- Do not mention the ground-truth, given, reference, or provided answer.
- Do not mention leakage, this rewrite instruction, or privileged context.
- If the response contains <think>...</think>, preserve those tags.
- Return only the rewritten response inside the required XML tags.

LEAKAGE SPANS:
{spans}

MODEL RESPONSE:
{response}

Output exactly:
<REWRITTEN_RESPONSE>
your rewritten response
</REWRITTEN_RESPONSE>
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model name served by vLLM.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--rewrite-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_prompts(row: dict) -> tuple[str, str, str, str | None, str]:
    if "prompt" in row and "privileged_context" in row:
        full_prompt = str(row["prompt"])
        context = str(row["privileged_context"])
        if not context or context not in full_prompt:
            raise ValueError("privileged_context must occur in prompt.")
        return (
            full_prompt.replace(context, "", 1),
            full_prompt,
            context,
            row.get("response"),
            str(row.get("generation_model", "")),
        )

    question = row.get("question")
    answer = row.get("ground_truth", row.get("answer"))
    if not isinstance(question, str) or answer is None:
        raise ValueError(
            "Each row needs prompt/privileged_context or question/ground_truth."
        )
    context = f"\nGiven the ground truth answer is $\\boxed{{{answer}}}."
    clean_prompt = f"{question} "
    return clean_prompt, clean_prompt + context, context, row.get("response"), ""


def thinking_extra_body(enable_thinking: bool | None):
    if enable_thinking is False:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


async def complete(
    client,
    args,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    enable_thinking: bool | None,
    sampling: bool,
) -> str:
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
    reasoning = getattr(message, "reasoning", None) or getattr(
        message, "reasoning_content", None
    ) or ""
    content = message.content or ""
    return (reasoning + content).strip()


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
            raise ValueError(f"Judge returned text not copied from response: {value!r}")
        if value not in spans:
            spans.append(value)
    return sorted(spans, key=response.find)


def parse_rewritten(text: str) -> str:
    marker = "<REWRITTEN_RESPONSE>"
    end_marker = "</REWRITTEN_RESPONSE>"
    start, end = text.find(marker), text.rfind(end_marker)
    if start < 0 or end < start:
        raise ValueError("Model did not return the required rewrite tags.")
    response = text[start + len(marker) : end].strip()
    if not response:
        raise ValueError("Model returned an empty rewritten response.")
    return response


async def judge(client, args, full_prompt: str, response: str) -> list[str]:
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": f"PROMPT WITH PRIVILEGED CONTEXT:\n{full_prompt}\n\nMODEL RESPONSE:\n{response}",
        },
    ]
    last_error = None
    for _ in range(args.judge_retries + 1):
        try:
            output = await complete(
                client,
                args,
                messages,
                args.judge_max_tokens,
                temperature=0.0,
                enable_thinking=False,
                sampling=False,
            )
            return parse_spans(output, response)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Could not parse judge output after retries: {last_error}")


async def rephrase(client, args, clean_prompt, response, spans) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful answer editor. Follow the user's exact output format.",
        },
        {
            "role": "user",
            "content": REPHRASE_PROMPT.format(
                prompt=clean_prompt,
                spans=json.dumps(spans, ensure_ascii=False),
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
    return parse_rewritten(output)


def output_row(row, full_prompt, context, response, generation_model, spans):
    result = {
        **row,
        "generation_model": row.get("generation_model", generation_model),
        "prompt": full_prompt,
        "privileged_context": context,
        "response": response,
        "detected_leakage_spans": spans,
    }
    if "leakage_spans" not in row:
        result["leakage_spans"] = spans
    return result


async def process_row(client, args, row):
    clean_prompt, full_prompt, context, response, source_model = get_prompts(row)
    generation_model = source_model or args.model
    if not response:
        response = await complete(
            client,
            args,
            [{"role": "user", "content": full_prompt}],
            args.max_new_tokens,
            args.temperature,
            enable_thinking=None if not args.disable_thinking else False,
            sampling=True,
        )

    spans = await judge(client, args, full_prompt, response)
    for _ in range(args.max_rounds):
        if not spans:
            break
        rewritten = await rephrase(client, args, clean_prompt, response, spans)
        if rewritten == response:
            break
        response = rewritten
        spans = await judge(client, args, full_prompt, response)

    return output_row(row, full_prompt, context, response, generation_model, spans)


async def run(args):
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1.")
    if args.max_rounds < 0:
        raise ValueError("--max-rounds must be non-negative.")

    rows = read_jsonl(args.input_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_path.exists():
        args.output_path.unlink()

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI package with: pip install openai") from exc

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def limited(row):
        async with semaphore:
            return await process_row(client, args, row)

    tasks = [asyncio.create_task(limited(row)) for row in rows]
    try:
        with args.output_path.open("a", encoding="utf-8") as output:
            for task in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Rephrasing",
            ):
                result = await task
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
