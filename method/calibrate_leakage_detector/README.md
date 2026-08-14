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

The dataset builder samples problems from Hugging Face, extracts the boxed answer, asks one or more generation models to answer with the privileged context present, then asks a judge model to mark leaked text spans.

All-in-one generation plus judging:

```bash
python method/calibrate_leakage_detector/create_calibrate_dataset.py \
  --phase all \
  --generation-model Qwen/Qwen3-4B \
  --judge-model Qwen/Qwen3-4B \
  --max-examples 100 \
  --batch-size 8 \
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
  --max-examples 100 \
  --batch-size 8 \
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
  --input-path method/calibrate_leakage_detector/data/generated-responses.jsonl \
  --batch-size 8 \
  --judge-max-new-tokens 1024 \
  --output-path method/calibrate_leakage_detector/data/generated-responses-with-leakage-spans.jsonl \
  --overwrite
```

For Qwen thinking models, add `--disable-thinking` if you want to disable thinking during generation and judging.

## Calibrate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase calibrate \
  --model Qwen/Qwen3-4B \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size 2 \
  --window-sizes 1,3,5,7 \
  --threshold-steps 80 \
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

When `--top-k-values` is provided, the script calibrates each candidate `k`, then chooses the detector with span-start recall/full-recall equal to `1.0`, highest token precision, highest span precision, smallest `k`, smallest window, and highest threshold. `output/all_experiments.jsonl` contains every tested `(model, top_k, window_size, threshold)` row.

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
