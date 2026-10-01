from copy import deepcopy
import json
from pathlib import Path
import os
import re

from datasets import load_dataset
from openai import OpenAI
from numpy import mean, exp, median, percentile
from transformers import AutoTokenizer


# Fraction of token-level privileged-context effects retained by PCS.
PCS_TOP_RATIO = 0.05

# IMPORTANT: change this string if the generation methods expose the
# privileged answer using a different prompt format.
PRIVILEGED_CONTEXT_TEMPLATE = "\n\nGiven the ground truth answer is $\\boxed{{{answer}}}$."


def replace_boxed(text):
    def replacer(match):
        content = match.group(0)
        answer = parse_boxed_answer(content)
        if not answer:
            return content
        return str(answer)
    return re.sub(r'\\boxed\{([^{}]*)\}', replacer, text)


def parse_boxed_answer(text):
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def get_ground_truth(example):
    answer = example.get(
        "ground_truth",
        parse_boxed_answer(example["messages"][-1]["content"])
    )
    if not answer:
        return None

    answer = str(answer)
    if "another answer" in answer.lower():
        return None

    parsed_answer = parse_boxed_answer(answer)
    if parsed_answer:
        answer = parsed_answer
    return answer


def get_reasoning_content(example):
    if example["messages"][-1].get("reasoning_content", None):
        return example["messages"][-1]["reasoning_content"]

    reasoning_content = example["messages"][-1]["content"].split("</think>")[0]
    if reasoning_content.startswith("<think>/n"):
        reasoning_content = reasoning_content.split("<think>/n")[-1]
    elif reasoning_content.startswith("<think>"):
        reasoning_content = reasoning_content.split("<think>")[-1]
    return reasoning_content


def get_question_messages(example, correctness_system=False):
    messages = deepcopy(example["messages"])
    messages = messages[:-1]

    if correctness_system:
        system_prompt = (
            r"You are a precise problem solver. Solve step by step, then give "
            r"the final answer on its own line as \boxed{answer}, with only "
            r"the result inside (no words/units), using exactly one \boxed{}."
        )
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            messages[0] = {"role": "system", "content": system_prompt}

    return messages


def add_privileged_context(messages, answer):
    """Append privileged answer to the last user message."""
    messages = deepcopy(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "user":
            messages[i]["content"] += PRIVILEGED_CONTEXT_TEMPLATE.format(
                answer=answer
            )
            return messages
    raise ValueError("No user message found for privileged-context insertion.")


def get_reasoning_scoring_prompt(messages):
    """Prompt that ends exactly where rationale content starts."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if prompt.rfind("<think>") == -1:
        prompt += "<think>\n"
    return prompt


def get_answer_scoring_prompt(messages, reasoning_content=None):
    """Prompt that ends immediately before the gold-answer content."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if reasoning_content is None:
        # Question-only baseline for GALG.
        if prompt.rfind("<think>") != -1:
            prompt += "</think>\nFinal answer: "
        else:
            prompt += "Final answer: "
    else:
        # Question + rationale condition for GALG.
        if prompt.rfind("<think>") != -1:
            prompt += reasoning_content + "\n</think>\nFinal answer: "
        else:
            prompt += "<think>\n" + reasoning_content + "\n</think>\nFinal answer: "

    return replace_boxed(prompt)


def get_completion_logps(client, prompt, completion):
    """
    Return log-probabilities only for `completion`.

    max_tokens=1 is kept for OpenAI/vLLM compatibility with echo=True.
    The final returned log-probability corresponds to that extra generated
    token and is therefore removed.
    """
    n_prompt_tokens = len(tokenizer.encode(prompt))

    response = client.completions.create(
        model=MODEL,
        prompt=prompt + completion,
        max_tokens=1,
        logprobs=1,
        echo=True
    )

    logps = response.choices[0].logprobs.token_logprobs[n_prompt_tokens:]

    # Remove the one newly generated token requested by max_tokens=1.
    if len(logps) > 0:
        logps = logps[:-1]

    return [x for x in logps if x is not None]


def f_fluency(example):
    """Answer-Blind Rationale Perplexity (AB-PPL)."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    messages = get_question_messages(example)
    reasoning_content = get_reasoning_content(example)
    prompt = get_reasoning_scoring_prompt(messages)

    logps = get_completion_logps(client, prompt, reasoning_content)
    if len(logps) == 0:
        return {"rendered_chat_fluency": "", "ab_ppl": -1.0}

    logp = mean(logps)
    ppl = exp(-logp)

    return {
        "rendered_chat_fluency": prompt + reasoning_content,
        "ab_ppl": float(ppl)
    }


def f_pcs(example):
    """Privileged-Context Sensitivity (PCS)."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    answer = get_ground_truth(example)
    if answer is None:
        return {
            "rendered_chat_pcs_blind": "",
            "rendered_chat_pcs_privileged": "",
            "pcs": -1.0
        }

    reasoning_content = get_reasoning_content(example)

    blind_messages = get_question_messages(example)
    privileged_messages = add_privileged_context(blind_messages, answer)

    blind_prompt = get_reasoning_scoring_prompt(blind_messages)
    privileged_prompt = get_reasoning_scoring_prompt(privileged_messages)

    blind_logps = get_completion_logps(
        client, blind_prompt, reasoning_content
    )
    privileged_logps = get_completion_logps(
        client, privileged_prompt, reasoning_content
    )

    if (
        len(blind_logps) == 0
        or len(privileged_logps) == 0
        or len(blind_logps) != len(privileged_logps)
    ):
        return {
            "rendered_chat_pcs_blind": blind_prompt + reasoning_content,
            "rendered_chat_pcs_privileged": privileged_prompt + reasoning_content,
            "pcs": -1.0
        }

    # Δ_t^priv = log p(r_t | q,c,r_<t) - log p(r_t | q,r_<t)
    deltas = [
        priv_lp - blind_lp
        for priv_lp, blind_lp in zip(privileged_logps, blind_logps)
    ]

    # Only positive privileged-context effects count toward leakage sensitivity.
    positive_deltas = [max(0.0, x) for x in deltas]

    k = max(1, int(len(positive_deltas) * PCS_TOP_RATIO + 0.999999))
    topk = sorted(positive_deltas, reverse=True)[:k]
    pcs = mean(topk)

    return {
        "rendered_chat_pcs_blind": blind_prompt + reasoning_content,
        "rendered_chat_pcs_privileged": privileged_prompt + reasoning_content,
        "pcs": float(pcs)
    }


def f_correctness_gain(example):
    """Gold-Answer Log-Likelihood Gain (GALG)."""
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    answer = get_ground_truth(example)
    if answer is None:
        return {
            "rendered_chat_galg_base": "",
            "rendered_chat_galg_reason": "",
            "galg": -999.0
        }

    reasoning_content = get_reasoning_content(example)
    messages = get_question_messages(example, correctness_system=True)

    # Score the answer content only; do not include \boxed{...} formatting.
    base_prompt = get_answer_scoring_prompt(messages, reasoning_content=None)
    reason_prompt = get_answer_scoring_prompt(
        messages, reasoning_content=reasoning_content
    )

    base_logps = get_completion_logps(client, base_prompt, answer)
    reason_logps = get_completion_logps(client, reason_prompt, answer)

    if (
        len(base_logps) == 0
        or len(reason_logps) == 0
        or len(base_logps) != len(reason_logps)
    ):
        return {
            "rendered_chat_galg_base": base_prompt + answer,
            "rendered_chat_galg_reason": reason_prompt + answer,
            "galg": -999.0
        }

    # Token-normalized gold-answer log-likelihood gain:
    # GALG = mean log p(a* | q,r) - mean log p(a* | q)
    galg = mean(reason_logps) - mean(base_logps)

    return {
        "rendered_chat_galg_base": base_prompt + answer,
        "rendered_chat_galg_reason": reason_prompt + answer,
        "galg": float(galg)
    }


def summarize_robust(values):
    values = [x for x in values if x >= 0.0]
    q25 = percentile(values, 25)
    q50 = median(values)
    q75 = percentile(values, 75)
    return {
        "mean": float(mean(values)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "n": len(values),
    }


def main():
    print(f"base_url={BASE_URL}")
    print(f"MODEL={MODEL}")

    ds = load_dataset("json", data_files=DATA_FILES, split="train[:1000]")

    # 1) Naturalness: Answer-Blind Rationale Perplexity
    ds = ds.map(f_fluency, num_proc=NUM_PROC)
    ab_ppl_summary = summarize_robust(ds["ab_ppl"])
    print("AB-PPL median [Q25, Q75]:",
          ab_ppl_summary["median"],
          [ab_ppl_summary["q25"], ab_ppl_summary["q75"]])

    # 2) Leakage safety proxy: Privileged-Context Sensitivity
    ds = ds.map(f_pcs, num_proc=NUM_PROC)
    pcs_summary = summarize_robust(ds["pcs"])
    print("PCS median [Q25, Q75]:",
          pcs_summary["median"],
          [pcs_summary["q25"], pcs_summary["q75"]])

    # 3) Whole-chain contribution to the gold answer: GALG
    ds = ds.map(f_correctness_gain, num_proc=NUM_PROC)
    valid_galg = [x for x in ds["galg"] if x != -999.0]
    galg_summary = summarize_robust(valid_galg)
    print("GALG mean:", galg_summary["mean"])
    print("GALG median:", galg_summary["median"])

    if SAVE_PATH:
        results = {
            "ab_ppl": ab_ppl_summary,
            "pcs": pcs_summary,
            "galg": galg_summary,

            # Per-example values are saved for later statistical analysis.
            "per_example": [
                {
                    "ab_ppl": float(ab_ppl),
                    "pcs": float(pcs),
                    "galg": float(galg),
                }
                for ab_ppl, pcs, galg in zip(
                    ds["ab_ppl"], ds["pcs"], ds["galg"]
                )
            ],

            "rendered_chat_fluency": [
                {
                    "text": text,
                    "ab_ppl": float(ab_ppl),
                }
                for text, ab_ppl in zip(
                    ds["rendered_chat_fluency"],
                    ds["ab_ppl"]
                )
            ],
            "rendered_chat_pcs": [
                {
                    "blind": blind,
                    "privileged": privileged,
                    "pcs": float(pcs),
                }
                for blind, privileged, pcs in zip(
                    ds["rendered_chat_pcs_blind"],
                    ds["rendered_chat_pcs_privileged"],
                    ds["pcs"]
                )
            ],
            "rendered_chat_galg": [
                {
                    "base": base,
                    "reason": reason,
                    "galg": float(galg),
                }
                for base, reason, galg in zip(
                    ds["rendered_chat_galg_base"],
                    ds["rendered_chat_galg_reason"],
                    ds["galg"]
                )
            ],
        }

        save_dir = Path(SAVE_PATH).parent.absolute()
        os.makedirs(save_dir, exist_ok=True)

        with open(SAVE_PATH, "w") as f:
            f.write(json.dumps(results))

    print(f"Outputs saved at {SAVE_PATH}")


if __name__ == "__main__":
    print(">>> Evaluating")

    ROOT_DIR = Path(
        "/workspace/storage-shared/nlp/huypq51/projects/"
        "traffic-hope-network-train/data_new/all_methods"
    )

    # METHODS = ["ours"]
    # MODELS = [
    #     "Nemo3-Nano-4B",
    #     "Qwen3-4B",
    #     "Qwen3-8B",
    #     "Qwen3-14B",
    #     "DeepSeek-R1-Distill-Qwen-14B"
    # ]
    # PORTS = ["8000", "8001", "8002", "8003", "8004"]
    # DOMAINS = ["logic"]

    METHODS = ["ours_clean"]
    MODELS = ["Qwen3-4B"]
    PORTS = ["8001"]
    DOMAINS = ["logic", "Math", "multihop-reasoning", "science"]

    MODEL = None
    NUM_PROC = 10
    API_KEY = "dummy"

    for method in METHODS:
        for i, model in enumerate(MODELS):
            for domain in DOMAINS:
                try:
                    SAVE_PATH = (
                        "/workspace/storage-shared/nlp/hieplh8/research/"
                        "traffic-hope-network-main/eval/outputs/new_metrics/"
                        f"{model}_{method}_{domain}.json"
                    )
                    TOKENIZER_PATH = f"/workspace/storage-shared/models/{model}"
                    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
                    BASE_URL = f"http://localhost:{PORTS[i]}/v1"
                    MODEL = model

                    print(
                        f">>> Evaluating METHOD={method}, "
                        f"MODEL={model}, DOMAIN={domain}..."
                    )

                    DATA_DIR = ROOT_DIR / method / model / domain
                    DATA_FILES = str(
                        list(DATA_DIR.glob("ready2train_filtered*.jsonl"))[0]
                    )
                    main()

                except Exception as e:
                    print("=" * 20)
                    print("Error:", e)
                    print("Skipping...")
                    print("=" * 20)
