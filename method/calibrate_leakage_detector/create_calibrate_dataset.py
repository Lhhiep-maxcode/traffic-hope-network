from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from prompt import SYSTEM_PROMPT, USER_PROMPT
from tqdm import tqdm


LEAKAGE_INSTRUCTION = (
    "Notice that do not leak the ground truth answer in your reasoning and final answer"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["generate", "judge", "all"])
    parser.add_argument("--generation-model", nargs="+", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--problem-field", default="problem")
    parser.add_argument("--solution-field", default="solution")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="float16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
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

        depth = 0
        chars = []
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


def boxed_answer(example: dict, solution_field: str, answer_field: str) -> str | None:
    answer = example.get(answer_field, None)
    if answer is not None:
        return str(answer).strip()
    boxes = braced_blocks(str(example.get(solution_field, "")), r"\boxed")
    if boxes:
        return boxes[-1]
    return None


def load_seed_examples(args) -> list[dict]:
    from datasets import load_dataset

    dataset_args = [args.dataset]
    if args.dataset_config:
        dataset_args.append(args.dataset_config)
    dataset = load_dataset(*dataset_args, split=args.split)

    seeds = []
    for example in dataset:
        answer = boxed_answer(example, args.solution_field, args.answer_field)
        problem = example.get(args.problem_field)
        if problem and answer:
            seeds.append({"problem": str(problem), "answer": answer})
        if args.max_examples and len(seeds) >= args.max_examples:
            break
    return seeds


def build_prompt(problem: str, answer: str) -> str:
    return (
        f"{problem} Given the ground truth answer is $\\boxed{{{answer}}}$. "
        f"{LEAKAGE_INSTRUCTION}"
    )


def torch_dtype(name: str):
    if name == "auto":
        return "auto"
    import torch

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_lm(model_name: str, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=args.trust_remote_code)
    kwargs = {"torch_dtype": torch_dtype(args.dtype), "trust_remote_code": args.trust_remote_code}
    if args.device_map != "none":
        kwargs["device_map"] = args.device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()
    if args.device_map == "none":
        import torch

        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def device_of(model):
    return model.device if hasattr(model, "device") else next(model.parameters()).device


def chat_text(tokenizer, user_prompt: str, system_prompt: str | None = None, enable_thinking: bool | None = None) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float, enable_thinking: bool | None = None, system_prompt: str | None = None) -> list[str]:
    import torch

    texts = [chat_text(tokenizer, prompt, system_prompt, enable_thinking) for prompt in prompts]
    old_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device_of(model))
    finally:
        tokenizer.padding_side = old_padding_side

    kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id}
    kwargs["do_sample"] = temperature > 0
    if temperature > 0:
        kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(ids[input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        for ids in output_ids
    ]


def leakage_spans(judge_output: str, response: str) -> list[str]:
    spans = []
    for span in braced_blocks(judge_output, r"\text"):
        span = span.strip().strip('"').strip("'")
        if span in response and span not in spans:
            spans.append(span)
    return spans


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def clear_model_memory():
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_rows(args) -> list[dict]:
    seeds = load_seed_examples(args)
    print(f"Loaded {len(seeds)} seed examples from {args.dataset}/{args.split}")

    rows = []
    for model_name in args.generation_model:
        print(f"Loading generation model: {model_name}")
        model, tokenizer = load_lm(model_name, args)

        batches = range(0, len(seeds), args.batch_size)
        for start in tqdm(batches, desc=f"Generating {model_slug(model_name)}", unit="batch"):
            batch = seeds[start : start + args.batch_size]
            prompts = [build_prompt(seed["problem"], seed["answer"]) for seed in batch]
            responses = generate_batch(
                model,
                tokenizer,
                prompts,
                args.generation_max_new_tokens,
                args.generation_temperature,
                enable_thinking=None if not args.disable_thinking else False,
            )
            rows.extend(
                {"generation_model": model_name, "prompt": prompt, "response": response}
                for prompt, response in zip(prompts, responses)
            )
        del model, tokenizer
        clear_model_memory()
    return rows


def judge_rows(args, rows: list[dict], output_path: Path):
    print(f"Loading judge model: {args.judge_model}")
    judge, tokenizer = load_lm(args.judge_model, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        batches = range(0, len(rows), args.batch_size)
        for start in tqdm(batches, desc="Detecting leakage", unit="batch"):
            batch = rows[start : start + args.batch_size]
            judge_prompts = [
                USER_PROMPT.format(prompt=row["prompt"], response=row["response"])
                for row in batch
            ]
            judge_outputs = generate_batch(
                judge,
                tokenizer,
                judge_prompts,
                args.judge_max_new_tokens,
                temperature=0.0,
                enable_thinking=False,
                system_prompt=SYSTEM_PROMPT,
            )
            for row, judge_output in zip(batch, judge_outputs):
                file.write(
                    json.dumps(
                        {
                            "prompt": row["prompt"],
                            "response": row["response"],
                            "leakage_spans": leakage_spans(judge_output, row["response"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
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
        write_jsonl(generate_rows(args), output_path)
    elif args.phase == "judge":
        judge_rows(args, read_jsonl(args.input_path), output_path)
    else:
        judge_rows(args, generate_rows(args), output_path)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
