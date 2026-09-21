from copy import deepcopy

from datasets import load_dataset
from openai import OpenAI
from numpy import mean, exp
from transformers import AutoTokenizer


BASE_URL = "http://localhost:8000/v1"
API_KEY = "empty"
MODEL_NAME = ...
DATA_FILES = ...
NUM_PROC = 64
SAVE_DIR = ...
TOKENIZER_PATH = ...


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def compute_logprob(example):
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    
    messages = deepcopy(example["messages"])
    messages[-1]["content"] = ""
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    n_prompt_tokens = len(tokenizer.encode(prompt))
    completion = example["ground_truth"]
    
    response = client.completions.create(
        model=MODEL_NAME,
        prompt=prompt + completion,
        max_tokens=1,
        logprobs=1,
        echo=True
    )
    
    logps = response.choices[0].logprobs.token_logprobs[n_prompt_tokens:]
    logp = mean(logps)
    ppl = exp(-logp)
    
    return {"ground_truth_ppl": ppl}


def main():
    print(f"base_url={BASE_URL}")
    print(f"model_name={MODEL_NAME}")
    
    ds = load_dataset("json", data_files=DATA_FILES, split="train")
    ds_ppl = ds.map(compute_logprob, num_proc=NUM_PROC)
    
    score = mean(ds_ppl["ground_truth_ppl"])
    ds_ppl.to_json(SAVE_DIR, force_ascii=False, lines=True)
    
    print(f"Outputs saved at {SAVE_DIR}")
    print(f"Ground truth PPL = {score:4f}")

