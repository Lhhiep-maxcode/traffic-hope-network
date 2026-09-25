from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from tqdm.auto import tqdm


AUG_SUP_SYSTEM_PROMPT = """Your role as an assistant involves providing precise and accurate solutions before providing detailed explanations with your full work showing your systematic thinking process leading to each solution. Your explanations should show how you engaged in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Solution and Explanation. In the Solution section, present your well-thought solution that accurately answers the question. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|>. In the Explanation section, comprehensively detail your reasoning process using the specified format: <|begin_of_explanation|> {explanation with steps separated with '\n\n'} <|end_of_explanation|> Each step should show detailed considerations leading to your solutions such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. **DO NOT** explicitly output any information of Solution section in the Explanation section. **DO NOT** explicitly output any information of Solution section in the Explanation section. **DO NOT** explicitly output any information of Solution section in the Explanation section. **PROHIBITION**: When outputting your Explanation, you are strictly forbidden from displaying ANY discernible signs that you have peeked at the Solution. **DO NOT** explicitly output any information of Solution section in the Explanation section. Should ANY form of Solution leakage occur, you will be severely punished by the Almighty Ruler."""

SOLUTION_RE = re.compile(
    r"<\|begin_of_solution\|>\s*(.*?)\s*<\|end_of_solution\|>",
    re.IGNORECASE | re.DOTALL,
)
EXPLANATION_RE = re.compile(
    r"<\|begin_of_explanation\|>\s*(.*?)\s*<\|end_of_explanation\|>",
    re.IGNORECASE | re.DOTALL,
)

AUG_SUP_RETRY_SUFFIX = """Your previous response could not be parsed. Follow the required format exactly: output one complete `<|begin_of_solution|>...<|end_of_solution|>` block followed by one complete `<|begin_of_explanation|>...<|end_of_explanation|>` block, with no other blocks or commentary."""
FINAL_ANSWER_PREFILL = "__FINAL_ANSWER_PREFILL_8F9A3C__"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model name served by vLLM.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Local tokenizer path or Hugging Face tokenizer name.",
    )
    parser.add_argument("--dataset", default="Hiepppp/reasoning")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--problem-field", default="question")
    parser.add_argument("--solution-field", default="solution")
    parser.add_argument("--answer-field", default="ground_truth")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=10_000_000_000)
    parser.add_argument("--splitter", default=None)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--parse-retries",
        type=int,
        default=3,
        help="Additional generation attempts after an AUG-SUP parse failure.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model-native thinking (disabled in the paper's setup).",
    )
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
        for index, char in enumerate(text[pos + 1 :], pos + 1):
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    blocks.append("".join(chars).strip())
                    pos = index + 1
                    break
                depth -= 1
            chars.append(char)


def parse_answer(
    example: dict, solution_field: str, answer_field: str, splitter: str | None
):
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
    dataset = dataset.select(range(args.start_index, min(args.end_index, len(dataset))))

    rows = []
    errors = 0
    for example in dataset:
        question = example.get(args.problem_field)
        answer = parse_answer(
            example, args.solution_field, args.answer_field, args.splitter
        )
        if isinstance(question, str) and question.strip() and answer:
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "domain": example.get("domain"),
                }
            )
        else:
            errors += 1
        if args.max_samples and len(rows) >= args.max_samples:
            break
    print(f"Loaded {len(rows)} examples with {errors} errors.")
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


def get_prompts(row: dict) -> tuple[str, str, str, str]:
    question = row.get("question")
    answer = row.get("ground_truth", row.get("answer"))
    if not isinstance(question, str) or not question.strip() or answer is None:
        raise ValueError("Each row needs question and ground_truth/answer fields.")
    answer = str(answer).strip()
    if not answer:
        raise ValueError("The ground-truth answer must not be empty.")

    clean_prompt = question.strip()
    privileged_context = f"\nGiven the ground truth answer is $\\boxed{{{answer}}}$."
    return clean_prompt, clean_prompt + privileged_context, privileged_context, answer


def parse_aug_sup_output(text: str) -> tuple[str, str]:
    solution_match = SOLUTION_RE.search(text)
    explanation_match = EXPLANATION_RE.search(text)
    if not solution_match or not explanation_match:
        raise ValueError("Model output is missing a complete solution or explanation block.")
    if solution_match.end() > explanation_match.start():
        raise ValueError("The solution block must appear before the explanation block.")

    solution = solution_match.group(1).strip()
    explanation = explanation_match.group(1).strip()
    if not solution or not explanation:
        raise ValueError("Model returned an empty solution or explanation block.")
    return solution, explanation


def message_field(message, name: str):
    value = getattr(message, name, None)
    if value is not None:
        return value
    return (getattr(message, "model_extra", None) or {}).get(name)


async def complete(
    client,
    args,
    semaphore,
    messages: list[dict],
    *,
    temperature: float | None = None,
    enable_thinking: bool | None = None,
) -> str:
    if temperature is None:
        temperature = args.temperature
    if enable_thinking is None:
        enable_thinking = args.enable_thinking
    request = {
        "model": args.model,
        "messages": messages,
        "max_tokens": args.max_new_tokens,
        "temperature": temperature,
        "top_p": args.top_p,
    }
    extra_body = {}
    if args.top_k > 0:
        extra_body["top_k"] = args.top_k
    if not enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        request["extra_body"] = extra_body

    async with semaphore:
        result = await client.chat.completions.create(**request)
    message = result.choices[0].message
    content = (message.content or "").strip()
    reasoning = message_field(message, "reasoning") or message_field(
        message, "reasoning_content"
    )
    if content:
        return content
    if reasoning:
        return reasoning.strip()
    raise ValueError("Model returned no text.")


async def generate_reasoning(client, args, semaphore, full_prompt):
    last_error = None
    messages = [
        {"role": "system", "content": AUG_SUP_SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]
    for attempt in range(args.parse_retries + 1):
        raw_output = await complete(client, args, semaphore, messages)
        try:
            _, explanation = parse_aug_sup_output(raw_output)
        except ValueError as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": f"{AUG_SUP_RETRY_SUFFIX}\nParser error: {exc}",
                },
            ]
            continue
        return raw_output, explanation, attempt + 1

    attempts = args.parse_retries + 1
    raise ValueError(
        f"Failed to parse AUG-SUP output after {attempts} attempts: {last_error}"
    )


def reasoning_prefill_content(reasoning: str) -> str:
    reasoning = reasoning.strip()
    match = re.fullmatch(r"<think>\s*(.*?)\s*</think>", reasoning, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
    return f"<think>\n{reasoning}\n</think>\n"


def build_final_answer_prompt(tokenizer, question, reasoning):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return f"{prompt}{reasoning_prefill_content(reasoning)}"


async def generate_final_answer(
    client, args, semaphore, tokenizer, question, reasoning
):
    prompt = build_final_answer_prompt(tokenizer, question, reasoning)
    request = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "top_p": args.top_p,
    }
    if args.top_k > 0:
        request["extra_body"] = {"top_k": args.top_k}

    async with semaphore:
        result = await client.completions.create(**request)
    answer = (result.choices[0].text or "").strip()
    if not answer:
        raise ValueError("Model returned no final answer.")
    return answer


async def process_row(client, args, semaphore, tokenizer, row):
    clean_prompt, full_prompt, _, _ = get_prompts(row)
    _, explanation, _ = await generate_reasoning(
        client, args, semaphore, full_prompt
    )
    final_answer = await generate_final_answer(
        client, args, semaphore, tokenizer, clean_prompt, explanation
    )
    return {
        "messages": [
            {"role": "user", "content": clean_prompt},
            {
                "role": "assistant",
                "content": final_answer,
                "reasoning_content": explanation,
            },
        ]
    }


async def run(args):
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1.")
    chunk_size = args.chunk_size or args.max_concurrency
    if chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1.")
    if args.parse_retries < 0:
        raise ValueError("--parse-retries must be >= 0.")

    rows = load_seed_examples(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_path.exists():
        args.output_path.unlink()
    done = processed_questions(args.output_path)
    rows = [row for row in rows if row["question"].strip() not in done]
    print(f"Generating AUG-SUP traces for {len(rows)} unprocessed examples.")

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError("Install the OpenAI package with: pip install openai") from exc

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install Transformers with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1500)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    failures = 0
    try:
        with args.output_path.open("a", encoding="utf-8") as output:
            with tqdm(total=len(rows), desc="Generating AUG-SUP") as progress:
                for start in range(0, len(rows), chunk_size):
                    chunk = rows[start : start + chunk_size]
                    tasks = [
                        asyncio.create_task(
                            process_row(client, args, semaphore, tokenizer, row)
                        )
                        for row in chunk
                    ]
                    for task in asyncio.as_completed(tasks):
                        try:
                            result = await task
                        except Exception as exc:  # Keep large dataset runs resumable.
                            failures += 1
                            tqdm.write(f"Skipped one example: {exc}")
                        else:
                            output.write(json.dumps(result, ensure_ascii=False) + "\n")
                            output.flush()
                        progress.update(1)
    finally:
        await client.close()

    if failures:
        print(f"Completed with {failures} failed examples; rerun to retry them.")


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
