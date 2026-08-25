# Leakage Attention Detector

This script calibrates attention heads, calibrates a span-start threshold/window, and evaluates the final detector.

Input rows should contain:

```json
{"prompt": "...", "response": "...", "leakage_spans": ["exact leaked text"]}
```

Input JSONL files live in:

```text
method/calibrate_leakage_detector/data/
```

Generated calibration artifacts live in:

```text
method/calibrate_leakage_detector/output/
```

## Create Calibration Dataset

The dataset builder samples problems from Hugging Face, extracts the boxed answer, asks one or more generation models through a vLLM OpenAI-compatible server, then asks a judge model to mark leaked text spans.

Start a vLLM OpenAI-compatible server separately, for example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --served-model-name Qwen/Qwen3-4B \
  --port 8000
```

The script uses the OpenAI Python client:

```bash
pip install openai
```

All-in-one generation plus judging:

```bash
python method/calibrate_leakage_detector/create_calibrate_dataset.py \
  --phase all \
  --generation-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B \
  --base-url http://localhost:8000/v1 \
  --max-examples 100 \
  --max-concurrency 8 \
  --generation-max-new-tokens 2048 \
  --generation-temperature 0.7 \
  --judge-max-new-tokens 1024 \
  --output-path method/calibrate_leakage_detector/data/qwen3-4B-calibrate-dataset.jsonl \
  --overwrite
```

This command writes:

```text
method/calibrate_leakage_detector/data/qwen3-4B-calibrate-dataset.jsonl
```

To separate generation and leakage judging, first generate responses:

```bash
python method/calibrate_leakage_detector/create_calibrate_dataset.py \
  --phase generate \
  --generation-model Qwen/Qwen3-4B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --base-url http://localhost:8000/v1 \
  --max-examples 100 \
  --max-concurrency 8 \
  --generation-max-new-tokens 2048 \
  --generation-temperature 0.7 \
  --output-path method/calibrate_leakage_detector/data/generated-responses.jsonl \
  --overwrite
```

Then judge the generated responses:

```bash
python method/calibrate_leakage_detector/create_calibrate_dataset.py \
  --phase judge \
  --judge-model Qwen/Qwen3-4B \
  --base-url http://localhost:8000/v1 \
  --input-path method/calibrate_leakage_detector/data/generated-responses.jsonl \
  --max-concurrency 8 \
  --judge-max-new-tokens 1024 \
  --output-path method/calibrate_leakage_detector/data/generated-responses-with-leakage-spans.jsonl \
  --overwrite
```

If the generation and judge models are served by different vLLM servers, use `--generation-base-url` and `--judge-base-url`.

The model names passed to `--generation-model` and `--judge-model` must match the model names served by vLLM. For Qwen thinking models, add `--disable-thinking` if you also want to disable thinking during response generation; leakage judging disables thinking by default.

## Calibrate

Calibration learns which attention heads are useful, searches over `top_k`, `window_size`, and `threshold`, then saves the chosen detector config.

Minimal command:

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size 2 \
  --window-sizes 1,3,5,7 \
  --threshold-steps 80 \
  --span-penalty-alpha 0.01 \
  --overwrite
```

Full command showing every option. Omit `--max-samples` for the full dataset, and omit `--disable-thinking` if you want Qwen thinking enabled in the rendered prompt.

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --calibrate-path method/calibrate_leakage_detector/data/generated-responses-with-leakage-spans.jsonl \
  --val-calibrate-path method/calibrate_leakage_detector/data/val-generated-responses-with-leakage-spans.jsonl \
  --detector-config-path method/calibrate_leakage_detector/output/detector_config.json \
  --score-cache-path method/calibrate_leakage_detector/output/calibrate_head_scores.pt \
  --all-experiments-path method/calibrate_leakage_detector/output/all_experiments.jsonl \
  --top-k 8 \
  --top-k-values 1,2,4,8,16,32 \
  --window-sizes 1,3,5,7 \
  --threshold-steps 80 \
  --span-penalty-alpha 0.01 \
  --aggregation weighted \
  --batch-size 1 \
  --max-samples 100 \
  --max-seq-len 2048 \
  --attn-implementation eager \
  --device-map auto \
  --dtype float16 \
  --disable-thinking \
  --trust-remote-code \
  --overwrite
```

This writes:

```text
method/calibrate_leakage_detector/output/calibrate_head_scores.pt
method/calibrate_leakage_detector/output/all_experiments.jsonl
method/calibrate_leakage_detector/output/detector_config.json
```

Head scores use:

```text
cosine(positive_median_normalized_attention, ideal_mask) * positive_contrast
```

The final detector aggregates selected heads by normalizing each head first, then using the calibrated head scores as weights.

When `--top-k-values` is provided, the script calibrates each candidate `k`, then chooses the detector with span-start recall/full-recall equal to `1.0`, highest `token_precision - alpha * avg_pred_spans_per_sample`, highest token precision, highest span precision, smallest `k`, smallest window, and highest threshold. `output/all_experiments.jsonl` contains every tested `(model, top_k, window_size, threshold)` row.

Calibration parameters:

- `--model`: Required Hugging Face model id or local model path. The saved config key is the final model-name part, for example `Qwen3-4B`.
- `--calibrate-path`: JSONL used to score/rank individual heads. Default: `data/generated-responses-with-leakage-spans.jsonl`.
- `--val-calibrate-path`: JSONL used to select `top_k`, `window_size`, and `threshold`. Default: `data/val-generated-responses-with-leakage-spans.jsonl`.
- `--detector-config-path`: JSON file where the final detector config is saved. Default: `output/detector_config.json`.
- `--score-cache-path`: Torch cache for calibrated head scores. Default: `output/calibrate_head_scores.pt`.
- `--all-experiments-path`: JSONL containing every tried `(model, top_k, window_size, threshold)` metric row. Default: `output/all_experiments.jsonl`.
- `--top-k`: Single fallback `k` when `--top-k-values` is not provided. Default: `8`.
- `--top-k-values`: Comma-separated `k` candidates to search, for example `1,2,4,8,16,32`. If provided, this is preferred over `--top-k`.
- `--window-sizes`: Comma-separated rolling-max window sizes. Default: `1,3,5,7`.
- `--threshold-steps`: Number of uniformly spaced thresholds searched between `0` and the max rolled detector score for each window size. Default: `80`.
- `--span-penalty-alpha`: Penalty strength for many predicted spans during config selection. The selection score is `token_precision - alpha * avg_pred_spans_per_sample`. Default: `0.01`.
- `--aggregation`: How selected heads are aggregated. `weighted` uses calibrated head scores as weights; `mean` averages selected heads equally. Default: `weighted`.
- `--batch-size`: Number of samples per forward pass. Increase for speed if GPU memory allows. Default: `1`.
- `--max-samples`: Optional cap on number of rows read from each calibration file. Useful for debugging. Default: no cap.
- `--max-seq-len`: Skip samples whose tokenized prompt plus response is longer than this. Default: `2048`.
- `--attn-implementation`: Attention backend passed to Transformers. Use `eager` when `output_attentions=True` is needed. Default: `eager`.
- `--device-map`: Device placement passed to `from_pretrained`, commonly `auto`. Default: `auto`.
- `--dtype`: Model dtype: `auto`, `float32`, `float16`, or `bfloat16`. Default: `float16`.
- `--disable-thinking`: Disable Qwen thinking in the chat template when building the prompt text.
- `--trust-remote-code`: Pass `trust_remote_code=True` when loading model/tokenizer.
- `--from-cache`: Reuse `--score-cache-path` instead of recomputing head scores. This still recalibrates `top_k`, threshold, and window.
- `--overwrite`: Allow replacing an existing detector config for this model and overwrite `all_experiments.jsonl`. Without this, calibration skips the model if it already exists in `detector_config.json`.

## Evaluate

Evaluation loads a saved detector config, applies it to the test JSONL, updates test metrics in `detector_config.json`, and optionally plots the lowest-scoring examples.

Minimal command:

```bash
python method/calibrate_leakage_detector/eval_calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --batch-size 2 \
  --plot-lowest-n 5
```

Full command showing every option. Omit `--max-samples` for the full test set, and omit `--disable-thinking` if you want Qwen thinking enabled in the rendered prompt.

```bash
python method/calibrate_leakage_detector/eval_calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --test-path method/calibrate_leakage_detector/data/test-generated-responses-with-leakage-spans.jsonl \
  --detector-config-path method/calibrate_leakage_detector/output/detector_config.json \
  --plot-dir method/calibrate_leakage_detector/output/lowest_score_plots \
  --plot-lowest-n 5 \
  --aggregation weighted \
  --batch-size 1 \
  --max-samples 100 \
  --max-seq-len 2048 \
  --attn-implementation eager \
  --device-map auto \
  --dtype float16 \
  --disable-thinking \
  --trust-remote-code
```

This reads `output/detector_config.json`, updates that model's aggregate test metrics inside the same file, and optionally plots the lowest-scoring samples.

Evaluation parameters:

- `--model`: Required model id or local model path. Must map to a key that already exists in `detector_config.json`.
- `--test-path`: JSONL test set with `prompt`, `response`, and `leakage_spans`. Default: `data/test-generated-responses-with-leakage-spans.jsonl`.
- `--detector-config-path`: JSON file produced by calibration. Evaluation reads the selected heads, aggregation, window size, and threshold from this file. Default: `output/detector_config.json`.
- `--plot-dir`: Directory for lowest-score diagnostic plots. Default: `output/lowest_score_plots`.
- `--plot-lowest-n`: Save plots for the `n` lowest cosine-score samples. Use `0` to disable plotting. Default: `0`.
- `--aggregation`: Fallback aggregation if the loaded detector config does not contain an `aggregation` field. Normally the saved config value is used. Choices: `weighted`, `mean`. Default: `weighted`.
- `--batch-size`: Number of samples per forward pass. Increase if memory allows. Default: `1`.
- `--max-samples`: Optional cap on number of test rows. Useful for debugging. Default: no cap.
- `--max-seq-len`: Skip samples longer than this many tokens after prompt and response are concatenated. Default: `2048`.
- `--attn-implementation`: Attention backend. Use `eager` for reliable attention outputs. Default: `eager`.
- `--device-map`: Device placement passed to `from_pretrained`. Default: `auto`.
- `--dtype`: Model dtype: `auto`, `float32`, `float16`, or `bfloat16`. Default: `float16`.
- `--disable-thinking`: Disable Qwen thinking in the chat template when rebuilding prompt text.
- `--trust-remote-code`: Pass `trust_remote_code=True` when loading model/tokenizer.

Evaluation updates these fields inside the detector config for the chosen model:

```json
{
  "test_precision": 0.0,
  "test_recall": 0.0,
  "test_full_recall": 0.0
}
```

## Rule-Based Baseline

Use this to compare the attention detector against a calibration-derived rule detector. The baseline learns text patterns from `leakage_spans` in the model's calibration data, then applies those learned patterns to the test responses. It does not load the model; it only loads the tokenizer so predicted text spans and gold leakage spans can be mapped to tokens consistently.

```bash
python method/calibrate_leakage_detector/eval_rule_based_detector.py \
  --model Qwen/Qwen3-4B \
  --calibrate-path method/calibrate_leakage_detector/data/Qwen3-4B/generated-responses_Qwen3-4B_with-leakage-spans.jsonl \
  --input-path method/calibrate_leakage_detector/data/Qwen3-4B/test_generated-responses_Qwen3-4B_with-leakage-spans.jsonl \
  --output-path method/calibrate_leakage_detector/output/Qwen3-4B/rule_based_metrics.json \
  --rules-output-path method/calibrate_leakage_detector/output/Qwen3-4B/rule_based_rules.json \
  --predictions-path method/calibrate_leakage_detector/output/Qwen3-4B/rule_based_predictions.jsonl \
  --trust-remote-code
```

The script prints and saves these five main metrics:

| Span precision | Span recall | Full recall | Token precision | Avg pred spans |
| :------------: | :---------: | :---------: | :-------------: | :------------: |

Rule-baseline parameters:

- `--model`: Required tokenizer id or local tokenizer path. Use the same model tokenizer as the attention detector.
- `--calibrate-path`: JSONL calibration file used to learn rule patterns from annotated `leakage_spans`.
- `--input-path`: JSONL test file with `prompt`, `response`, and `leakage_spans`.
- `--output-path`: JSON file where aggregate rule-baseline metrics are saved. Default: `output/rule_based_metrics.json`.
- `--rules-output-path`: JSON file where the learned exact/generalized rules are saved. Default: `output/rule_based_rules.json`.
- `--predictions-path`: Optional JSONL path for per-sample predicted leakage text spans, predicted token spans, and sample metrics.
- `--min-support`: Keep learned rules that appear in at least this many calibration leakage spans. Default: `1`.
- `--min-words`: Ignore very short leakage spans with fewer words than this. Default: `2`.
- `--min-chars`: Ignore very short leakage spans with fewer characters than this. Default: `6`.
- `--max-rules`: Optional cap on the number of learned rules, sorted by support and rule length.
- `--exact-only`: Use only exact phrase rules. Without this flag, the baseline also learns generalized rules where numbers, LaTeX math, `\boxed{...}`, and bold answer values are replaced by value placeholders.
- `--max-samples`: Optional cap for quick debugging.
- `--max-seq-len`: Skip samples longer than this many tokens after prompt and response are concatenated. Default: `2048`.
- `--disable-thinking`: Disable Qwen thinking in the chat template when rebuilding prompt text.
- `--trust-remote-code`: Pass `trust_remote_code=True` when loading the tokenizer.

## Calibrate Then Evaluate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size 2 \
  --overwrite

python method/calibrate_leakage_detector/eval_calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --batch-size 2 \
  --plot-lowest-n 5
```

## Reuse Head Cache

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --model Qwen/Qwen3-4B \
  --from-cache \
  --top-k-values 1,2,4,8,16,32 \
  --overwrite
```

This reuses `output/calibrate_head_scores.pt` for head ranking, then recalibrates `k`, threshold, and window.

## Repair

```bash
python method/leakage_repair/repair_with_detector.py \
  --model Qwen/Qwen3-4B \
  --input-path method/calibrate_leakage_detector/data/test-generated-responses-with-leakage-spans.jsonl \
  --detector-config-path method/calibrate_leakage_detector/output/detector_config.json \
  --output-path method/calibrate_leakage_detector/output/repaired_responses.jsonl \
  --overwrite
```

The repair script detects token spans, regenerates those spans from the prompt with the privileged context removed, and accepts each replacement only when the privileged-vs-clean next-token JS divergence is below `--js-threshold`.
