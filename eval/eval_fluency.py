from datasets import load_dataset
from openai import OpenAI
from numpy import mean, exp


BASE_URL = ...
API_KEY = ...
MODEL_NAME = ...
DATA_FILES = ...
NUM_PROC = 64
SAVE_DIR = ...


def compute_logprob(example):
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    response = client.completions.create(
        model=MODEL_NAME,
        prompt=example["text"],
        max_tokens=1,
        logprobs=1,
        echo=True
    )
    
    logps = response.choices[0].logprobs.token_logprobs[1:]
    logp = mean(logps)
    ppl = exp(-logp)
    
    return {"ppl": ppl}


def main():
    print(f"base_url={BASE_URL}")
    print(f"model_name={MODEL_NAME}")
    
    ds = load_dataset("json", data_files=DATA_FILES, split="train")
    ds_ppl = ds.map(compute_logprob, num_proc=NUM_PROC)
    
    score = mean(ds_ppl["ppl"])
    ds_ppl.to_json(SAVE_DIR, force_ascii=False, lines=True)
    
    print(f"Outputs saved at {SAVE_DIR}")
    print(f"score={score:4f}")

