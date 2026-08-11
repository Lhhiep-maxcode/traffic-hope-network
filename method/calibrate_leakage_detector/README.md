# Leakage Attention Head Calibration

This script chooses attention heads whose attention-to-privileged-context pattern is similar to the leakage-token mask.

## Input Data

Each JSONL row should contain:

```json
{"prompt": "...", "response": "...", "leakage_spans": ["exact leaked text"]}
```

Default files:

```text
generated-responses-with-leakage-spans.jsonl
test-generated-responses-with-leakage.jsonl
```

Rows with empty `leakage_spans` are skipped.

## 1. Calibrate Heads

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase calibrate \
  --model Qwen/Qwen3-4B \
  --top-k 8 \
  --overwrite
```

This writes:

```text
method/calibrate_leakage_detector/calibrate_result.jsonl
```

Example format:

```json
{"Qwen3-4B": [{"layer": 25, "head": 9}, {"layer": 30, "head": 21}]}
```

## 2. Evaluate Heads

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase evaluate \
  --model Qwen/Qwen3-4B \
  --plot-lowest-n 5 \
  --overwrite
```

This reads `calibrate_result.jsonl` and writes:

```text
method/calibrate_leakage_detector/calibrate_eval_result.jsonl
method/calibrate_leakage_detector/lowest_score_plots/
```

## 3. Calibrate And Evaluate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase total \
  --model Qwen/Qwen3-4B \
  --top-k 8 \
  --plot-lowest-n 5 \
  --overwrite
```

## Useful Arguments

```text
--top-k              Number of best attention heads to keep.
--max-samples        Use only the first N leakage samples.
--max-seq-len        Skip samples longer than this many tokens.
--disable-thinking   Render Qwen chat template with thinking disabled.
--dtype              Model dtype: float16, bfloat16, float32, or auto.
--device-map         Device placement, usually auto.
--overwrite          Allow replacing output files.
```

The default attention backend is `eager`, which is safest for `output_attentions=True`.
