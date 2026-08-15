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

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase calibrate \
  --model Qwen/Qwen3-4B \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size 2 \
  --window-sizes 1,3,5,7 \
  --threshold-steps 80 \
  --span-penalty-alpha 0.01 \
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

## Evaluate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase evaluate \
  --model Qwen/Qwen3-4B \
  --batch-size 2 \
  --plot-lowest-n 5 \
  --overwrite
```

This reads `output/detector_config.json`, updates that model's aggregate test metrics inside the same file, and optionally plots the lowest-scoring samples.

## Calibrate And Evaluate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase total \
  --model Qwen/Qwen3-4B \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size 2 \
  --plot-lowest-n 5 \
  --overwrite
```

## Reuse Head Cache

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase calibrate \
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
