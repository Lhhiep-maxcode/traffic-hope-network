from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from tqdm.auto import tqdm


SSR_SYSTEM_PROMPT = """You are an expert AI assistant reconstructing a structured reasoning trace for a final assistant turn in a dialogue.

Output exactly two blocks, in this order: `<skeleton>` and `<reason>`. The already-provided assistant turn is context; return the reasoning trace only.
After closing `</skeleton>`, the very next nonblank line must be `<reason>`. Never write `</reason>` before writing `<reason>`.

The trace should be moderately developed: more informative than a terse outline, less expansive than a long chain of thought. Add depth where the task or the importance of a judgment calls for it, while keeping the trace derivational rather than a surface restatement. Use deeper reflection dynamically for objectively hard steps and for subjectively central steps where the response value depends on a key conclusion, audience need, method choice, uncertainty boundary, implementation tradeoff, or paper-style claim.

### Tags
Use these tags only:
- [PLAN]: understand the request, constraints, and answer shape.
- [RETR]: bring in needed facts, context, or remembered information.
- [INFR]: infer, calculate, transform, or connect task details.
- [EVAL]: check a risky assumption, calculation, source, or consistency point.
- [SUMM]: organize the constraints, checks, and response plan.
- [BRCH]: compare plausible approaches before selecting one.
- [RFLX]: pause on an important judgment and name what keeps it disciplined.
- [BTRK]: revise a concrete earlier path after a check fails.

### Skeleton Rules
1. Put one numbered line per step inside `<skeleton>`.
2. After the number, write one real tag from the tag list, then `[HIGH]` or `[LOW]` with no space between them, then a short task-specific action sentence.
3. Use 6-8 steps for simple tasks, 8-10 for ordinary tasks, and 10-12 for genuinely multi-part, technical, mathematical, coding, policy, safety, scientific, or long-form tasks.
4. Keep each line short and grounded in the user's request; do not use placeholders, examples, or template words.
5. Do not put explanatory second sentences in skeleton lines; put all extra criteria, caveats, and checks in the matching reason paragraph.
6. Mark at least half the lines `[HIGH]`.
7. Add diagnostic steps only when they are earned by real difficulty: ambiguity, competing approaches, missing information, verification risk, or a failed path.
8. Add importance steps when a judgment deserves extra care because it affects a main claim, recommendation, method choice, numerical interpretation, safety or risk tradeoff, paper-style conclusion, or final prioritization.
9. Also add information-density steps when the value of the response depends on interpreting user intent, weighing audience impact, selecting a framing for a conclusion, or separating a robust claim from a tempting but overconfident one.
10. For nontrivial tasks, include at least one diagnostic step, importance step, or information-density step, and usually no more than three total.
11. Do not repeat the same tag or sentence frame three times in a row.
12. In skeleton lines, name the reasoning operation, check, selection criterion, or response shape rather than reusing the assistant turn's surface wording.

### Reason Rules
1. Write one paragraph for each skeleton line, in the same order, with exactly one blank line between paragraphs.
2. Each paragraph should be compact but informative. Use one sentence for plain setup or wrap-up; use two sentences for most `[HIGH]`, comparison, check, reflection, technical, numerical, safety, or recommendation steps.
3. Reconstruct the reasoning process as a derivational trace. Use task-level categories, ordinary terms, and any task-relevant terms needed for faithful reasoning; make the connection through criteria, checks, and transitions rather than surface restatement.
4. Build toward the response stance progressively. Earlier paragraphs should develop criteria, constraints, alternatives, and checks; the final paragraphs may integrate the conclusion direction when it follows from the trace.
5. For `[EVAL]`, `[BRCH]`, `[RFLX]`, and `[BTRK]`, name the actual uncertainty, tradeoff, evidence standard, or correction. Avoid generic claims that something is simply correct.
6. For importance-driven or information-density-driven depth, explain why the point deserves extra care and what evidence, constraint, consequence, audience need, or uncertainty boundary keeps it honest.
7. When adding depth, discuss the evidence standard, boundary condition, stakeholder need, format constraint, uncertainty, or failure mode before narrowing the response direction.
8. When the assistant turn contains a strong claim, concrete recommendation, named method, or exact wording, ground it by articulating the underlying criteria and transition so the trace reads as derivation rather than quotation.
9. For ordinary or harder tasks, several paragraphs should contain a second sentence that records a constraint, caveat, comparison, verification target, or reflective check. Do not add such a sentence when it would merely pad an obvious step.
10. Keep reasoning task-specific, but avoid broad background, extra examples, and meta-commentary about this protocol.
11. Do not number paragraphs, copy skeleton tags, or output anything outside the two required blocks.
12. The reasoning paragraphs must be between `<reason>` and `</reason>`, not before the opening tag or after the closing tag.
13. Always write the final closing tag `</reason>` after the last reasoning paragraph.
14. Close `</skeleton>` immediately after the final skeleton line. Then write `<reason>`, the reasoning paragraphs, and finally `</reason>`.
"""

SKELETON_RE = re.compile(
    r"<skeleton>\s*(.*?)\s*</skeleton>", re.IGNORECASE | re.DOTALL
)
REASON_RE = re.compile(r"<reason>\s*(.*?)\s*</reason>", re.IGNORECASE | re.DOTALL)

SSR_RETRY_SUFFIX = """Your previous response could not be parsed. Follow the required format exactly: output one complete `<skeleton>...</skeleton>` block followed by one complete `<reason>...</reason>` block, with no other blocks or commentary."""
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
        help="Additional generation attempts after an SSR parse failure.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model-native thinking (disabled in the paper's prompting setup).",
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
    conversation = f"User: {clean_prompt}\n\nAssistant: {answer}"
    privileged_context = f"\nGiven the ground truth answer is $\\boxed{{{answer}}}$."
    return clean_prompt, conversation, privileged_context, answer


def parse_ssr_output(text: str) -> tuple[str, str, str]:
    skeleton_match = SKELETON_RE.search(text)
    reason_match = REASON_RE.search(text)
    if not skeleton_match or not reason_match:
        raise ValueError("Model output is missing a complete skeleton or reason block.")
    if skeleton_match.end() > reason_match.start():
        raise ValueError("The skeleton block must appear before the reason block.")

    skeleton = skeleton_match.group(1).strip()
    reason = reason_match.group(1).strip()
    if not skeleton or not reason:
        raise ValueError("Model returned an empty skeleton or reason block.")
    formatted = (
        f"<skeleton>\n{skeleton}\n</skeleton>\n"
        f"<reason>\n{reason}\n</reason>"
    )
    return formatted, skeleton, reason


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


async def generate_reasoning(client, args, semaphore, conversation):
    last_error = None
    messages = [
        {"role": "system", "content": SSR_SYSTEM_PROMPT},
        {"role": "user", "content": conversation},
    ]
    for attempt in range(args.parse_retries + 1):
        raw_output = await complete(client, args, semaphore, messages)
        try:
            reasoning_content, skeleton, reason = parse_ssr_output(raw_output)
        except ValueError as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": f"{SSR_RETRY_SUFFIX}\nParser error: {exc}",
                },
            ]
            continue
        return raw_output, reasoning_content, skeleton, reason, attempt + 1

    attempts = args.parse_retries + 1
    raise ValueError(
        f"Failed to parse SSR output after {attempts} attempts: {last_error}"
    )


def build_final_answer_prompt(tokenizer, question, reasoning):
    messages = [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": FINAL_ANSWER_PREFILL,
        },
    ]
    prompt_with_prefill = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message="content",
        enable_thinking=False,
    )
    if reasoning not in prompt_with_prefill:
        raise ValueError(
            "The tokenizer chat template did not render assistant reasoning_content."
        )
    prefill_start = prompt_with_prefill.rfind(FINAL_ANSWER_PREFILL)
    if prefill_start < 0:
        raise ValueError(
            "The tokenizer chat template did not preserve the final-answer prefill."
        )
    return prompt_with_prefill[:prefill_start]


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
    clean_prompt, conversation, _, _ = get_prompts(row)
    _, _, _, reasoning_content, _ = await generate_reasoning(
        client, args, semaphore, conversation
    )
    final_answer = await generate_final_answer(
        client, args, semaphore, tokenizer, clean_prompt, reasoning_content
    )
    return {
        "messages": [
            {"role": "user", "content": clean_prompt},
            {
                "role": "assistant",
                "content": final_answer,
                "reasoning_content": reasoning_content,
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
    print(f"Generating SSR traces for {len(rows)} unprocessed examples.")

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
            with tqdm(total=len(rows), desc="Generating SSR") as progress:
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
