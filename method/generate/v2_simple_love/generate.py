from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from self_coding import CustomGenerator
import json
from tqdm import tqdm

MODEL_PATH = "Qwen/Qwen3-4B"
DATA_PATH = "/Users/rcyuh/Downloads/train_DeepMath-103K.jsonl"
OUTPUT_PATH = "/Users/rcyuh/Downloads/train_DeepMath-103K_formatted.jsonl"
INSTRUCT_PROMPT = "Explain your solution step by step."
LEAKAGE_PROMPT = "Given the ground truth answer is"
DETECTOR_CONFIG_PATH = "/kaggle/input/datasets/huyphung7/detector-config/detector_config.json"
MODEL_KEY = "Qwen3-4B"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.1
TOP_P = 0.9
DO_SAMPLE = True
DECODE_TOKENS = False  # True returns token text at each step; final text is always decoded.

def load_jsonl(file_path):
    """Read one sample at a time without retaining the dataset in memory."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line.strip())

def append_jsonl(output, output_file):
    json.dump(output, output_file, ensure_ascii=False)
    output_file.write('\n')
    output_file.flush()

def format_record(sample):
    question = sample['question']
    ground_truth = sample['ground_truth']
    prompt_wo_answer = f"{question}. {INSTRUCT_PROMPT}."
    privileged_context = f"{LEAKAGE_PROMPT} {ground_truth}"
    prompt_w_answer = f"{prompt_wo_answer} {privileged_context}."

    new_record = {
        "question": question,
        "ground_truth": ground_truth,
        "prompt_wo_answer": prompt_wo_answer,
        "prompt_w_answer": prompt_w_answer,
        "privileged_context": privileged_context
    }

    return new_record

def main():
    with open(DETECTOR_CONFIG_PATH, 'r', encoding='utf-8') as config_file:
        detector_config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    ).eval()

    with open(OUTPUT_PATH, 'a', encoding='utf-8') as output_file:
        for raw_sample in tqdm(load_jsonl(DATA_PATH), desc="Processing samples", unit="sample"):
            sample = format_record(raw_sample)
            generator = CustomGenerator(
                model=model,
                tokenizer=tokenizer,
                detector_config_path=DETECTOR_CONFIG_PATH,
                detector_config=detector_config,
                model_key=MODEL_KEY,
                max_new_tokens=MAX_NEW_TOKENS,
                fix_comparison_method="attention_score", # js_divergence, attention_score
                fix_js_divergence_threshold=0.2,
                max_repair_steps=32,
                prompt=sample['prompt_wo_answer'],
                privileged_context=sample['privileged_context'],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                seed=42,
                enable_thinking=True,
                decode_tokens=DECODE_TOKENS,
                debug_clean_backtrack=True,
                debug_fix_infinite_loop=True,
                debug_wait_safe_window=True,
            )

            generated_output = generator.generate(include_unfixed=False, show_progress=False)
            append_jsonl({**sample, **generated_output}, output_file)
            # Release this sample's KV caches before creating the next generator.
            del generator


if __name__ == "__main__":
    main()
