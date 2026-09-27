# Clean Generation

`method/generate_new` is a small, readable implementation of the same
leakage-aware generation algorithm used by the older generator:

1. Generate from the privileged prompt.
2. Score each proposed token by its selected attention mass to the privileged
   context.
3. If the score crosses the calibrated detector threshold, backtrack by the
   detector window radius.
4. Regenerate from the clean prompt until the candidate tokens are safe again.

It intentionally omits multiprocessing and batch scheduling so the algorithm is
easy to inspect and modify.

Example:

```bash
python method/generate_new/generate.py \
  --input data.jsonl \
  --output output.jsonl \
  --model Qwen/Qwen3-4B \
  --model-key Qwen3-4B \
  --detector-config method/calibrate_leakage_detector/output/detector_config.json
```
