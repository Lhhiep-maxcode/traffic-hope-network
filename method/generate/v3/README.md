# V3 — generate nhiều samples trên node B200

V3 là phiên bản độc lập; không sửa `v2_simple_love`. Thiết kế dành cho Qwen3
full attention: một process/model trên mỗi GPU và nhiều sample đang chạy trong
mỗi process. Chạy trực tiếp script, **không bọc thêm bằng `torchrun`** vì launcher
đã tạo các worker.

## Những thay đổi chính

- `--devices auto`: sử dụng mọi GPU CUDA đang visible, một bản model/GPU.
  Hàng đợi chia việc động giúp GPU nhận thêm sample khi xong việc cũ.
- `--batch-size 8`: số sample đang hoạt động **trên mỗi GPU**. Các bước sinh bình
  thường được forward chung, kể cả khi độ dài prompt/cache khác nhau. Sample kết
  thúc được thay bằng sample mới. **Prefill và repair hiện chạy riêng từng sample**;
  các nhánh này chưa được batch. Dữ liệu repair nhiều có thể giảm lợi ích batching.
- Cache sau batch được bỏ padding và tách thành vùng nhớ riêng cho từng sample,
  giữ rollback/KV reuse. Đổi batch size cần benchmark vì việc ghép/tách cache cũng
  tốn băng thông; batch lớn hơn không mặc nhiên nhanh hơn.
- `--attention-backend selective`: dùng SDPA cho prefill, clean branch và các
  layer không cần trả attention; dùng eager tại layer detector cần trọng số.
  Config Qwen3-4B trong repo hiện chỉ dùng head ở layer 24. Backend này không bỏ
  bước kiểm tra leakage. Dùng `--attention-backend eager` để đối chiếu.
- RNG riêng từng sample, seed = `(seed + sample_index) % 2**63`; clean và fixed
  của cùng sample dùng chung RNG. Baseline dùng lại seed đó. Chạy mẫu khác hoặc
  chuyển GPU không tiêu thụ luồng RNG của sample này. Khác v2 dùng cùng seed 42
  cho mọi sample. Kernel/batch shape vẫn có thể làm thay đổi sai số số học.
- `--decode-tokens` bật decode mỗi bước; mặc định tắt. Text cuối luôn được decode.
- Mặc định không sinh baseline (`--include-unfixed` để bật), vẫn bật cả ba tùy
  chọn repair như v2. Không tự ý giảm token limit, safe window hay ngưỡng detector.
- Đọc JSONL tuần tự, giới hạn hàng đợi, một writer ghi/flush mỗi kết quả. Không giữ
  toàn bộ dataset hoặc toàn bộ kết quả trong RAM. Resume chỉ giữ tập chỉ số đã xong.
- `--resume` kiểm tra hash input, config và tham số generation, bỏ mẫu đã ghi;
  chỉ cắt dòng cuối chưa hoàn chỉnh nếu lần chạy trước bị ngắt giữa một lần ghi.
  Lỗi worker dừng run và giữ các mẫu đã hoàn thành để resume.

## Môi trường

Python 3.12, Linux trên node B200. Phần tích hợp attention/cache được cố định với
`transformers==4.57.6`; không tự nâng Transformers cho version này. Bộ kiểm thử
tensor được chạy với PyTorch 2.8.0, Transformers 4.57.6 trên CPU.

Ví dụ tạo môi trường riêng trên node (chạy từ thư mục repo):

```bash
python3.12 -m venv .venv-v3
source .venv-v3/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r method/generate/v3/requirements.txt
```

Wheel CUDA 12.8 có hỗ trợ Blackwell. Nếu node đã có PyTorch phù hợp thì có thể
giữ bản đó; không cần thay môi trường hệ thống. Xem
[hướng dẫn bản PyTorch 2.8](https://pytorch.org/get-started/previous-versions/)
và [hỗ trợ Blackwell](https://pytorch.org/blog/pytorch-2-7/).

## Benchmark trước khi chạy toàn bộ

Đặt đường dẫn dữ liệu tương ứng trên node. Benchmark chỉ nạp model một lần,
warm up trước, đồng bộ CUDA trước/sau đo, báo throughput, VRAM peak và tỷ lệ
sample có token giống batch 1. Chọn tập có cả prompt dài và sample repair nhiều.

```bash
python method/generate/v3/benchmark.py \
  --input /path/to/train_DeepMath-103K.jsonl \
  --devices 0 \
  --batch-sizes 1,4,8,16,32,64 \
  --limit 64 \
  --max-new-tokens 256 \
  --repeats 2 \
  --report benchmark-b200.json
```

`best_measured_batch_size` là batch size có samples/giây trung bình cao nhất
trong các lượt đo thành công; đây không phải bảo đảm nhanh nhất trên toàn dataset.
Kiểm tra thêm với `--max-new-tokens 2048` trước khi chạy dài. Nếu gặp OOM,
benchmark ghi lại và thử cấu hình tiếp theo. Lỗi logic/repair không được che đi.

Muốn đối chiếu backend, chạy lại cùng lệnh với `--attention-backend eager` và
một file `--report` khác. Không dùng tốc độ của model nhỏ chạy CPU để suy ra
tốc độ Qwen3-4B trên B200.

## Generate trên toàn node

Ví dụ dùng batch 8/GPU; thay 8 bằng giá trị đã benchmark:

```bash
python method/generate/v3/generate.py \
  --input /path/to/train_DeepMath-103K.jsonl \
  --output /path/to/train_DeepMath-103K_v3.jsonl \
  --devices auto \
  --batch-size 8
```

Chọn GPU cụ thể: `--devices 0,1,2,3`. Chỉ số GPU tuân theo `CUDA_VISIBLE_DEVICES`.
`--devices auto` dùng CPU nếu không thấy CUDA; để kiểm tra CPU dùng thêm
`--dtype float32 --batch-size 2 --limit 4 --max-new-tokens 16`.

Mặc định dùng config tại `method/calibrate_leakage_detector/output/detector_config.json`.
Có thể thay bằng `--detector-config /path/to/detector_config.json`.
Model khác trong họ Qwen3 cần truyền cả `--model` và `--model-key` tương ứng.
Truyền `--revision` là commit của model nếu cần cố định weights/tokenizer.

Chạy tiếp: dùng lại lệnh với `--resume`. Có thể đổi số GPU/batch size hoặc tăng
`--limit` khi resume; không đổi input, detector hay các tham số generation đã lưu.
Không chạy đồng thời hai launcher vào cùng output: file lock sẽ từ chối launcher
thứ hai. Không dùng `--resume` để ghép kết quả v2 vào v3.

Output được ghi **theo thứ tự hoàn thành**, có `sample_index` (chỉ số dòng input,
bắt đầu từ 0) để khôi phục thứ tự. Vẫn lưu metadata input, question, ground_truth,
prompt, text, token_ids, fixed, unfixed và repair_events. File `.meta.json` ghi
cấu hình/hash, `.stats.json` ghi thống kê lượt chạy hoàn thành.

Prompt được sửa để chuỗi gửi model đúng bằng `prompt_wo_answer + privileged_context`;
đáp án thập phân và dấu chấm cuối câu được giữ nguyên. Vì v2 hiện ghép prompt thiếu
khoảng cách và dùng seed khác, không kỳ vọng output v3 trùng từng token với script v2.

## Kiểm chứng và giới hạn

```bash
python -m unittest discover -s method/generate/v3 -p 'test_*.py' -v
```

Các test dùng Qwen3 thật với trọng số khởi tạo ngẫu nhiên, kích thước nhỏ, không
tải weights từ mạng: so sánh logits/attention/KV cache, rollback sau batch,
token và repair giữa batch 1/2/4, RNG riêng, attention chọn lọc, streaming/resume
và hai worker `spawn` chạy đồng thời trên CPU.

Chưa benchmark hay xác minh số học BF16 trực tiếp trên B200. Selective SDPA và
batching có thể đổi kết quả làm tròn; với sampling hoặc score gần ngưỡng detector,
chuỗi token và quyết định repair có thể đổi. Dùng eager + batch 1 làm đối chứng
trên phần cứng đích trước khi đánh giá chất lượng/hiệu năng. Không dùng v3 cho
sliding-window, dynamic/longrope hoặc model ngoài Qwen3.

Thiết kế cache dựa trên [hợp đồng cache của Transformers](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation);
mỗi GPU giữ một bản model theo [mô hình distributed inference](https://huggingface.co/docs/accelerate/usage_guides/distributed_inference).
