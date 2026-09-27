# V4 — Nemotron / Mamba hybrid

Bản phát triển từ v3, giữ multi-GPU, JSONL, streaming/resume, RNG từng sample
và quy tắc backtrack/repair. Detector dùng trực tiếp config đã calibration.

## Detector config

`detector_config.json` là bản sao nguyên vẹn của `detector_config-Copy1.json`
do người dùng cung cấp. Chọn entry bằng `--model-key`:

| Model key | Heads | Window | Threshold |
|---|---:|---:|---:|
| `Nemo3-Nano-4B-BF16-logic` | 3 | 15 | 0.4624975621700287 |
| `Nemo3-Nano-4B-BF16-math` | 34 | 13 | 0.24431931972503662 |
| `Nemo3-Nano-4B-BF16-multihop-reasoning` | 4 | 15 | 0.4533136487007141 |
| `Nemo3-Nano-4B-BF16-science` | 21 | 15 | 0.32601675391197205 |

`heads[].layer` là **thứ tự attention layer** trong tuple dùng khi calibration.
Với Nano 4B, `0, 1, 2, 3` ánh xạ sang decoder block `12, 17, 24, 32` theo
`hybrid_override_pattern` trong [config NVIDIA](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/blob/main/config.json).
V4 ánh xạ khi tạo generator, giữ nguyên JSON, head weights, threshold và window.
Cache và tensor attention nội bộ vẫn dùng block index.

Không cần truyền `--detector-mode`. Cờ `--detector-mode attention` vẫn được
chấp nhận cho launcher hiện tại. Đã bỏ detector JSD/none và các cờ
`--detector-threshold`, `--repair-window-size`; xóa chúng khỏi script cũ.

`--fix-comparison-method attention_score` mặc định như v3. Có thể chọn
`--fix-comparison-method js_divergence --fix-js-divergence-threshold 0.2`
để dùng JSD **trong bước repair**, sau khi attention detector phát hiện lỗi.

## Chạy trên server

Từ thư mục `method/generate/v4-mamba`, trong môi trường đã cài Mamba:

```bash
python check_environment.py --cuda

bash run.sh \
  --model /path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --model-key Nemo3-Nano-4B-BF16-math \
  --input /path/to/train.jsonl \
  --output /path/to/train_nemotron_v4.jsonl \
  --devices auto --batch-size 4 \
  --max-new-tokens 2048 --limit 16
```

Chọn key theo dữ liệu logic/math/multihop/science. Mỗi dòng input có `question`
và `ground_truth`, ví dụ `{"question":"1 + 1 = ?", "ground_truth":"2"}`.
Mặc định đọc `detector_config.json` cạnh `generate.py`, kể cả khi gọi launcher
từ thư mục khác. Dùng `--detector-config /path/to/detector_config.json` để đổi file.

`--trust-remote-code` mặc định bật; `--model` nhận Hub ID hoặc thư mục checkpoint.
Có thể cố định Hub revision bằng `--revision COMMIT`. Không bọc bằng `torchrun`;
launcher đã spawn worker. Chọn interpreter bằng
`PYTHON=/path/to/.venv/bin/python bash run.sh ...`.
Nếu dùng `run-nemo.sh` tự tạo, kiểm tra nó gọi `v4-mamba/generate.py`.

Các flag `--debug-clean-backtrack`, `--debug-fix-infinite-loop`,
`--debug-wait-safe-window` mặc định bật và có `--no-...` tương ứng.
`--include-unfixed` sinh thêm baseline từ cùng seed như v3.

Nếu gặp `Repair did not converge within max_repair_steps`, sample đã dùng hết
ngân sách cho một lần repair. CLI mặc định `--max-repair-steps 32`. Khi bật
`--debug-wait-safe-window`, window 13/15 yêu cầu 6/7 lần kiểm tra an toàn liên
tiếp; một lần không đạt sẽ đặt lại bộ đếm. Các bước clean backtrack cũng tiêu
tốn ngân sách này nhưng không được tính vào chuỗi an toàn.

Lỗi in `sample_index`, model key, phương pháp/ngưỡng so sánh, số bước an toàn
hiện tại/tốt nhất và tám phép kiểm tra cuối. Có thể thử
`--max-repair-steps 128` với output mới và ít sample trước; tăng giới hạn chỉ
cho repair thêm thời gian, không đảm bảo hội tụ nếu score liên tục không đạt.
V4 vẫn dừng khi chạm giới hạn, giữ nguyên quy tắc repair và threshold như v3.

## Môi trường

Môi trường build trước đó trên server dùng `torch==2.11.0+cu130`.
Ghim phiên bản này trước khi cài các gói còn lại:

```bash
python -m pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
# Chỉ cần khi môi trường chưa có dependency CUDA tương ứng:
python -m pip install --no-build-isolation -r requirements-cuda.txt
python check_environment.py --cuda
```

Giữ `transformers==4.57.6` và Torch 2.11.0 đã dùng để build CUDA extensions.
Wheel `mamba-ssm`/`causal-conv1d` phải tương thích với Torch đang được nạp;
cùng package version chưa đủ để tái sử dụng wheel sang môi trường khác.
Ưu tiên dùng lại môi trường đã import thành công trước đó.

Nếu gặp `undefined symbol: ...c10_cuda_check_implementation...` khi đang dùng
Torch 2.14, kiểm tra phiên bản Torch trước khi build lại: [Torch 2.14](https://github.com/pytorch/pytorch/blob/v2.14.0/c10/cuda/CUDAException.h)
đã thêm tham số vào hàm này so với [Torch 2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/c10/cuda/CUDAException.h).
Wheel dùng ABI cũ sẽ không import được với thư viện mới dù cùng CUDA 13.0.
Sau khi đưa Torch về 2.11.0, chạy `python check_environment.py --cuda` để xác
nhận cả hai extension; số phiên bản package không chứng minh binary tương thích.

V4 tự tìm `z3/lib/libz3.so` trong `z3-solver`, preload và bổ sung
`LD_LIBRARY_PATH`. V4 không dùng vLLM. Nếu cần build từ source qua PyPI mirror
trên server không truy cập GitHub:

```bash
CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 \
python -m pip install -v --no-build-isolation \
  --no-binary=mamba-ssm,causal-conv1d -r requirements-cuda.txt
```

## Runtime và cache

- Luồng generation/repair dùng Nemotron-H hybrid có attention, Mamba2 và
  MLP/MoE, gồm cấu trúc dense Nano 4B và MoE Nano 30B. Adapter dùng block
  pretrained theo contract `backbone.layers`, `block_type`, `norm`, `mixer`,
  `norm_f`, với attention không RoPE; chưa hỗ trợ mọi kiến trúc tên Nemotron.
- Mỗi sample/nhánh giữ riêng convolution state, SSM state và KV cache.
  Backtrack hoặc loại candidate đã sửa state sẽ prefill lại đúng prefix;
  nhánh sạch bắt kịp prefix dài hơn bằng decode từng token.
- Chỉ batch prefix cùng độ dài, không padding recurrent state. `--batch-size`
  là số slot mỗi GPU; prefill và repair chạy từng sample. Xem thống kê
  `batched_forwards`, `batched_samples`, `scalar_forwards`.
- Mỗi GPU chứa toàn bộ model, không tensor parallel. SSM state Nemotron-H
  mặc định FP32; `--ssm-state-dtype model` dùng dtype model.
- Backend mặc định `sdpa`, chỉ tính weights query cuối cho detector.
  `--attention-backend eager` materialize attention đầy đủ khi prefill.
- Adapter native Mamba/Mamba2/Falcon-Mamba vẫn phục vụ baseline và kiểm thử
  cache. Mamba thuần không có attention để áp dụng detector này.

## Kiểm tra, resume và benchmark

```bash
python check_model.py \
  --model /path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --model-key Nemo3-Nano-4B-BF16-math \
  --input /path/to/train.jsonl --devices 0 --dtype bfloat16
```

So sánh logits/attention cached decode với full prefill, rollback và batch
hai state. Mặc định `atol=rtol=0.05` cho BF16; dùng
`--dtype float32 --atol 0.00003 --rtol 0.0003` để đối chiếu chặt hơn nếu mixer
hỗ trợ. `--cache-mode recompute` là đối chứng full prefill chậm.

Output giữ `sample_index`, input fields, `text`, `token_ids`, `fixed`, `unfixed`,
`repair_events`, `generation_stats`; ghi theo thứ tự hoàn tất.
`--resume` kiểm tra hash input/config, model key, generation parameters và quy
ước chỉ số layer. Có thể đổi GPU/slot hoặc tăng `--limit` khi resume.
Dùng output mới khi chuyển từ bản v4 cũ sang config và quy ước layer này.

```bash
python benchmark.py \
  --model /path/to/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --model-key Nemo3-Nano-4B-BF16-math \
  --input /path/to/train.jsonl --devices 0 \
  --batch-sizes 1,2,4,8,16 --limit 32 --max-new-tokens 256 \
  --report benchmark-v4.json

python -m unittest discover -s . -p 'test_*.py' -v
```

Kiểm thử offline dùng Mamba/Mamba2/Falcon-Mamba nhỏ và hybrid với mixer Mamba2
thật: cache, attention, config mapping, rollback, hai cách repair, batching,
RNG/EOS, resume và hai worker `spawn` nạp checkpoint hybrid có remote code.
Không cần tải checkpoint NVIDIA hay cài CUDA kernels cho bộ CPU này.

Adapter đã được đối chiếu trên CPU với code NVIDIA dense/MoE thu nhỏ; chưa
chạy checkpoint đầy đủ hoặc CUDA kernels trên B200. Dùng `check_model.py` trên
server để xác minh GPU.
