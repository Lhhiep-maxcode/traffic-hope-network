# V5 DeepSeek — generation, leakage detection và repair

Phiên bản độc lập được chỉnh từ `../v3`, dành cho **DeepSeek-R1-Distill-Qwen**
(kiến trúc `Qwen2ForCausalLM`, full attention). Chạy trực tiếp script, **không dùng
`torchrun`**: launcher đã tạo một worker và một bản model trên mỗi GPU.

## Model và detector

Model mặc định: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`.
`detector_config.json` trong thư mục này là bản sao nguyên vẹn của file người dùng
cung cấp `detector_config-Copy1.json`. Đường dẫn mặc định được tính theo vị trí
script, không phụ thuộc thư mục chạy lệnh. **`--model-key` là bắt buộc**, vì
detector được calibrate riêng cho từng model và miền dữ liệu:

| Miền | `--model-key` | Số head | Window |
| --- | --- | ---: | ---: |
| Math | `DeepSeek-R1-Distill-Qwen-14B-math` | 14 | 15 |
| Science | `DeepSeek-R1-Distill-Qwen-14B-science` | 8 | 11 |
| Logic | `DeepSeek-R1-Distill-Qwen-14B-logic` | 3 | 11 |
| Multihop | `DeepSeek-R1-Distill-Qwen-14B-multihop-reasoning` | 5 | 19 |

File còn chứa detector cho Qwen3 và Nemo; điều đó **không** làm v5 hỗ trợ các
kiến trúc này. Dùng v3 cho Qwen3. Với DeepSeek-R1-Distill-Qwen 1.5B/7B/32B,
phần runtime hỗ trợ kiến trúc Qwen2 full attention, nhưng file đính kèm **chưa
có detector cho các kích thước đó**. Cần calibrate riêng rồi truyền `--model`,
`--model-key`, `--detector-config` tương ứng. Không dùng detector 14B cho model khác.

Tên checkpoint DeepSeek được đối chiếu với tiền tố detector key. Với checkpoint
local có tên thư mục tùy ý, người chạy phải chọn detector đúng checkpoint;
code vẫn kiểm tra layer/head có nằm trong kích thước model hay không.

## Môi trường

Giữ **`transformers==4.57.6`**, vì code tích hợp trực tiếp attention và
`DynamicCache`. Ví dụ Linux/CUDA trên B200, chạy từ gốc repo:

```bash
python3.12 -m venv .venv-v5-deepseek
source .venv-v5-deepseek/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r method/generate/v5-deepseek/requirements.txt
```

Nếu đã có PyTorch phù hợp GPU thì dùng môi trường đó. Mỗi GPU phải đủ VRAM cho
toàn bộ weights và cache của các sample đang chạy. Đây là chia sample giữa GPU,
không phải tensor parallel để chia một model qua nhiều GPU.

## Chạy thử rồi generate

Input là JSONL, mỗi dòng có `question` (chuỗi không rỗng) và `ground_truth`.
Các trường metadata khác được giữ trong output:

```json
{"question":"What is 1 + 1?", "ground_truth":"2", "id":"sample-1"}
```

Chạy thử bốn mẫu; giới hạn 256 token trong ví dụ này chỉ dành cho smoke test:

```bash
python method/generate/v5-deepseek/generate.py \
  --input /path/to/input.jsonl \
  --output /path/to/deepseek14b-math-smoke.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --model-key DeepSeek-R1-Distill-Qwen-14B-math \
  --devices 0 --batch-size 1 --limit 4 --max-new-tokens 256
```

Generate trên mọi GPU CUDA đang visible, với batch size đã benchmark:

```bash
python method/generate/v5-deepseek/generate.py \
  --input /path/to/input.jsonl \
  --output /path/to/deepseek14b-math.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --model-key DeepSeek-R1-Distill-Qwen-14B-math \
  --devices auto --batch-size 8
```

Đổi `--model-key` theo miền trong bảng. `--devices 0,1,2,3` chọn GPU cụ thể;
các chỉ số tuân theo `CUDA_VISIBLE_DEVICES`. `--devices cpu --dtype float32`
dành cho kiểm thử CPU. `auto` dùng CPU nếu không thấy CUDA.

Các tham số generation/repair giữ mặc định của v3: `max-new-tokens=2048`,
`temperature=0.1`, `top-p=0.9`, seed 42, sampling bật, `max-repair-steps=32`,
và cả ba debug repair flag đều bật. Không tự thay threshold/window của detector.
Mỗi sample dùng seed `(seed + sample_index) % 2**63`. Dùng `--include-unfixed`
để sinh thêm baseline; mặc định tắt. `--revision COMMIT` cố định weights/tokenizer.

Prompt dùng **chat template gốc của tokenizer DeepSeek**, với
`add_generation_prompt=True`. Không nối thêm BOS hay `<think>` thủ công;
tokenize prompt đã render với `add_special_tokens=False`. Dừng khi sinh EOS của
tokenizer. V5 bỏ `--enable-thinking`/`--no-enable-thinking`: DeepSeek R1 Distill
không có công tắc thinking của Qwen3. Clean branch và baseline dùng cùng quy tắc.

## Attention, batching và benchmark

`--attention-backend selective` là mặc định: SDPA cho prefill/clean branch và
layer không cần attention weights; eager Qwen2 cho **tất cả layer chứa detector
head** khi cần tính score. Giữ placeholder cho layer khác để không lệch chỉ số.
Mask nhân quả hỗ trợ cả prefill và suffix nhiều token nối sau KV cache.

`--attention-backend eager` trả attention theo đường chuẩn để đối chiếu.
Cả hai backend kiểm tra model/detector trước khi nạp weights. Không hỗ trợ
sliding-window/hybrid cache, dynamic/longrope, Qwen3, DeepSeek-R1 đầy đủ hoặc
DeepSeek-R1-Distill-Llama.

Các bước decode thông thường được batch với prompt/cache khác độ dài; cache
được bỏ padding và tách riêng cho rollback. Prefill và repair vẫn chạy từng
sample như v3. Nhiều head/layer detector hoặc nhiều repair có thể giảm lợi ích
SDPA/batching. Benchmark trên GPU đích trước khi chọn batch size:

```bash
python method/generate/v5-deepseek/benchmark.py \
  --input /path/to/input.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --model-key DeepSeek-R1-Distill-Qwen-14B-math \
  --devices 0 --batch-sizes 1,2,4,8,16 \
  --limit 32 --max-new-tokens 256 --repeats 2 \
  --report benchmark-deepseek14b-math.json
```

Benchmark nạp weights một lần, warm up, đo throughput/VRAM và tỷ lệ sample có
token giống batch 1. Báo cáo có detector key và hash config. Thử lại với độ dài
generation thực tế và mẫu có repair; `best_measured_batch_size` chỉ tốt nhất
trong phép đo đó. Dùng eager + batch 1 làm đối chứng; BF16, kernel và batch shape
có thể thay đổi logits, token sampling hoặc quyết định repair gần threshold.

## Output và resume

Output ghi theo thứ tự hoàn thành, có `sample_index` (chỉ số dòng input từ 0),
metadata input, prompt, text/token IDs, fixed/unfixed và repair events.
File `.meta.json` lưu hash input/config và generation parameters, với phiên bản
`5-deepseek`; `.stats.json` lưu thống kê. Launcher flush mỗi kết quả và khóa file
để ngăn hai run ghi cùng output.

Chạy tiếp bằng đúng lệnh cũ thêm `--resume`. Có thể đổi số GPU/batch size hoặc
tăng `--limit`; không đổi model, detector hoặc generation parameters. Chỉ dòng
cuối bị ghi dở mới được tự cắt. Không resume từ output v3/v4. Dùng output mới
khi đổi miền detector.

## Kiểm thử và phạm vi xác minh

```bash
python -m unittest discover -s method/generate/v5-deepseek -p 'test_*.py' -v
```

Test chạy offline với Qwen2 nhỏ, weights khởi tạo ngẫu nhiên: so sánh
logits/attention/KV cache giữa scalar và batch; rollback và thay suffix so với
prefill mới; repair attention/JSD và baseline; EOS, RNG riêng; selective nhiều
layer và causal mask; streaming/resume và hai worker `spawn`. Fixture chứa
config/chat template gốc của DeepSeek 14B để kiểm tra dimensions của bốn detector,
BOS, privileged span và EOS với vocabulary nhỏ.

Kiểm thử CPU dùng Python 3.12, PyTorch 2.8.0, Transformers 4.57.6. Chưa xác minh
inference bằng weights DeepSeek 14B đầy đủ hoặc benchmark BF16 trên B200;
các test nhỏ không đánh giá chất lượng reasoning hay hiệu quả detector.
