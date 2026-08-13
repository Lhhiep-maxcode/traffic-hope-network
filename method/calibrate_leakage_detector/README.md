# Leakage Attention Detector

This script calibrates attention heads, calibrates a token threshold/window, and evaluates the final detector.

Input rows should contain:

```json
{"prompt": "...", "response": "...", "leakage_spans": ["exact leaked text"]}
```

## Calibrate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase calibrate \
  --model Qwen/Qwen3-4B \
  --top-k-values 1,2,4,8,16,32 \
  --batch-size 2 \
  --window-sizes 1,3,5,7 \
  --num-thresholds 80 \
  --overwrite
```

This writes:

```text
method/calibrate_leakage_detector/calibrate_result.jsonl
method/calibrate_leakage_detector/calibrate_head_scores.pt
method/calibrate_leakage_detector/threshold_calibration_metrics.jsonl
method/calibrate_leakage_detector/detector_config.json
```

Head scores use:

```text
cosine(positive_median_normalized_attention, ideal_mask) * positive_contrast
```

The final detector aggregates selected heads by normalizing each head first, then using the calibrated head scores as weights.

When `--top-k-values` is provided, the script calibrates each candidate `k`, then chooses the detector with recall/full-recall equal to `1.0`, highest precision, smallest `k`, smallest window, and highest threshold. The metrics JSONL contains every tested `(top_k, window_size, threshold)` row.

## Evaluate

```bash
python method/calibrate_leakage_detector/calibrate_detector.py \
  --phase evaluate \
  --model Qwen/Qwen3-4B \
  --batch-size 2 \
  --plot-lowest-n 5 \
  --overwrite
```

This reads `detector_config.json`, writes `calibrate_eval_result.jsonl`, and optionally plots the lowest-scoring samples.

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

This reuses `calibrate_head_scores.pt` for head ranking, then recalibrates `k`, threshold, and window.

## Repair

```bash
python method/leakage_repair/repair_with_detector.py \
  --model Qwen/Qwen3-4B \
  --input-path method/calibrate_leakage_detector/test-generated-responses-with-leakage.jsonl \
  --detector-config-path method/calibrate_leakage_detector/detector_config.json \
  --output-path method/leakage_repair/repaired_responses.jsonl \
  --overwrite
```

The repair script detects token spans, regenerates those spans from the prompt with the privileged context removed, and accepts each replacement only when the privileged-vs-clean next-token JS divergence is below `--js-threshold`.
