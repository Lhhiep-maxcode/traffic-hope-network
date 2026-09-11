# Sinh dữ liệu SFT bằng self-coding

`generate_sft.py` gọi trực tiếp `CustomGenerator` trong `self_coding.py`, bao gồm
attention detector và leakage repair. Cần Python >= 3.10, PyTorch, Transformers,
Accelerate và tqdm tương thích với model. Ví dụ cài dependencies trong môi trường
Python đã có PyTorch phù hợp với CUDA: `pip install transformers accelerate tqdm`.

## Input và prompt

Input là JSONL (đọc streaming) hoặc file `.json` chứa array (nạp array vào RAM):

```json
{"question": "Evaluate the limit ...", "ground_truth": "0"}
```

Mỗi sample được chuyển thành:

```json
{
  "prompt_wo_answer": "Evaluate the limit ...\n\nExplain your solution step by step.",
  "prompt_w_answer": "Evaluate the limit ...\n\nExplain your solution step by step. Given the ground truth answer is 0",
  "privileged_context": "Given the ground truth answer is 0"
}
```

Giá trị đáp án được thay trực tiếp, không thêm dấu ngoặc vuông. Đáp án số `0` cũng
được chấp nhận. Đổi tên cột bằng `--question-field problem --answer-field answer`.

Mặc định, sample có question/answer thiếu, rỗng hoặc sai kiểu sẽ được bỏ qua để
dataset lớn tiếp tục chạy. Script in index và lý do, đồng thời giữ dòng
`status: "skipped"`, `input_sample` và `error` trong file chi tiết. Dòng này
không được đưa vào file SFT và không được gửi tới model. Đáp án `0` vẫn hợp lệ;
script không tự điền hay suy đoán ground truth bị thiếu.

Muốn dừng ngay khi gặp sample lỗi, dùng `--strict-input` hoặc
`SFTConfig(..., strict_input=True)`. Chính sách này cũng áp dụng cho
`prepare_only` và `export_only`. JSON sai cú pháp vẫn làm dừng lượt chạy.
`--limit` tính cả sample bị bỏ qua. Khi generation/export với `--num-traces N`,
một input lỗi tương ứng N dòng `skipped`; khi prepare-only thì một dòng mỗi input.

Kiểm tra prompt trước, không cần model hoặc torch:

```bash
python method/generate/generate_sft.py \
  --input data/math.jsonl --output data/prompts.jsonl --prepare-only
```

## Chạy generation

Từ thư mục gốc repository, chạy một GPU:

```bash
python method/generate/generate_sft.py \
  --input data/math.jsonl \
  --output data/math.traces.jsonl \
  --model Qwen/Qwen3-4B --model-key Qwen3-4B \
  --devices cuda:0 --dtype bfloat16 \
  --max-new-tokens 4096 --require-eos
```

Nhiều GPU, mỗi GPU chứa được toàn bộ model và KV cache:

```bash
python method/generate/generate_sft.py \
  --input data/math.jsonl \
  --output data/math.traces.jsonl \
  --model Qwen/Qwen3-4B --model-key Qwen3-4B \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --dtype bfloat16 \
  --max-new-tokens 4096 --num-traces 3 --require-eos
```

Các device index tuân theo `CUDA_VISIBLE_DEVICES`. Mặc định `--devices auto`
dùng **một worker** với `device_map="auto"` để phân bố model lên các GPU khả dụng.
Có thể dùng `--devices cpu --dtype float32` hoặc `--devices mps` để thử nhỏ.

Một device chạy trực tiếp trong process gọi API. Nhiều device dùng multiprocessing
`spawn`, nạp model một lần mỗi worker, xử lý liên tiếp nhiều sample và giữ allocator
GPU. Các worker chạy độc lập; không dùng chung RNG/model
qua thread. Chỉ có một job đang chạy trên mỗi worker, tránh nạp toàn dataset vào
hàng đợi. Tốc độ thực tế phụ thuộc model, GPU, số token và số lần repair. Một GPU
vẫn giải mã từng sample vì pipeline hiện tại chưa hỗ trợ batch decoding.

Thanh `tqdm` hiển thị số trace đã xử lý/tổng trace cần sinh, tốc độ, ETA và số
`success`/`error`. Thanh cập nhật sau mỗi lần commit cache, dùng chung cho một
device hoặc nhiều worker. Tổng chỉ tính các job còn cần chạy sau khi bỏ cache,
sample trùng và input lỗi; khi resume thanh bắt đầu từ 0 với tổng còn lại.
Không tạo thanh token riêng cho từng worker để tránh chồng log. `tqdm.auto`
tự chọn giao diện phù hợp cho terminal hoặc notebook.

Detector mặc định: `method/calibrate_leakage_detector/output/detector_config.json`.
Chọn detector đúng model bằng `--model-key` hoặc đổi file bằng `--detector-config`.
Attention luôn là `eager` vì detector cần attention weights. Không sinh thêm
baseline `unfixed` để tiết kiệm một lượt generation.

Các tham số repair/sampling hiện có được đưa ra CLI, xem `--help`. Ví dụ:
`--fix-comparison-method js_divergence --fix-js-divergence-threshold 0.1`,
`--debug-clean-backtrack`, `--debug-wait-safe-window`, `--no-enable-thinking`.
Anti-loop và KV reuse bật mặc định theo pipeline; hai debug mode còn lại tắt.

## Import từ Python / Jupyter / Kaggle / Colab

Mở notebook ở thư mục gốc repo và import API trực tiếp:

```python
from method.generate.generate_sft import SFTConfig, generate_dataset

config = SFTConfig(
    input="/path/to/math.jsonl",
    output="/path/to/writable-output/math.traces.jsonl",
    model="/path/to/local/model",
    model_key="Qwen3-4B",  # Key detector tương ứng model của bạn.
    devices="cuda:0",
    dtype="bfloat16",
    local_files_only=True,
    max_new_tokens=4096,
    num_traces=1,
    require_eos=True,
)

summary = generate_dataset(config)
print(summary.counts)
print(summary.sft_output)
```

Nếu notebook nằm ngoài repo, thêm đường dẫn repo một lần trước khi import:

```python
import sys
sys.path.insert(0, "/path/to/repo")
```

API nhận đường dẫn dạng `str` hoặc `Path`. Dùng `devices=("cuda:0", "cuda:1")`
để chạy song song nhiều GPU. Một device (kể cả `"auto"`) chạy trực tiếp trong
kernel, không spawn process. Worker cho nhiều GPU được import từ module thật,
không định nghĩa trong cell. Không cần subprocess, `%%writefile`, hoặc tự đặt
`__file__`. Import module sẽ cung cấp `__file__` đúng và API không đọc `sys.argv`
của notebook.

Gọi lại `generate_dataset(config)` để resume. Khi ngắt cell, API xuất những kết
quả đã commit rồi truyền tiếp `KeyboardInterrupt` cho notebook. Kết quả bình
thường trả về `GenerationSummary` gồm `counts`, `output`, `sft_output`, `cache`
và `exit_code`. Sample lỗi nằm trong file chi tiết; lỗi cấu hình trả về exception
Python. Có thể dùng `prepare_only=True` hoặc `export_only=True` trong config.

Các module được import bình thường: `generate_sft.py` điều phối, `modeling.py`
load model, `self_coding.py` chứa pipeline. Không còn phụ thuộc `generate_oop.py`
đã chuyển vào `trash`. Pipeline hiện tại chỉ dùng module `self_coding`.

CLI vẫn chạy được bằng `python -m method.generate.generate_sft ...` hoặc lệnh
file trực tiếp như các ví dụ trên. Trong file Python dùng API nhiều GPU, đặt
lời gọi `generate_dataset(config)` bên trong `if __name__ == "__main__":`.

## Cache, resume và output

Với `--output data/math.traces.jsonl`, mặc định script tạo:

| File | Nội dung |
| --- | --- |
| `data/math.traces.sqlite` | Cache giao dịch SQLite, commit ngay sau từng job |
| `data/math.traces.jsonl` | Prompt, input gốc, trace, token IDs, repair events, seed, trạng thái và messages |
| `data/math.traces.sft.jsonl` | Chỉ `messages`, dùng cho SFT |

Đổi đường dẫn bằng `--cache` và `--sft-output`. Hai file JSONL được tạo trước khi
quét input. Mỗi job hoàn thành được commit vào SQLite rồi **append ngay** vào
file chi tiết; trace đạt yêu cầu cũng append vào file SFT. Cả hai file được
`flush` và `fsync` sau mỗi job, nên có thể mở hoặc `tail -f` ngay trong lúc chạy.
Chỉ process điều phối ghi file, tránh nhiều worker ghi chồng lên nhau.

Trong lúc generation, thứ tự dòng là thứ tự job hoàn thành; `source_index` và
`trace_index` cho biết vị trí input. Chưa có dòng `pending` trong file đang chạy.
Khi hoàn tất, Ctrl+C/SIGTERM hoặc worker lỗi, script xuất lại snapshot theo thứ
tự input và bổ sung `pending` nếu còn. File SFT luôn chỉ có trace đạt yêu cầu.
Không chạy hai tiến trình điều phối cùng cache; file `.lock` được tự giải phóng
khóa khi process chết.

**Resume: chạy lại cùng lệnh.** Sample thành công được bỏ qua. Sample lỗi được
thử lại, tối đa `--max-retries 1` lần bổ sung mỗi lượt chạy. Với SIGKILL/mất điện,
những kết quả đã commit được giữ; sample đang chạy hoặc chưa commit sẽ chạy lại
từ đầu. Không lưu KV cache giữa chừng của một sample. Giữ file SQLite và các
sidecar `-wal`/`-shm` nếu có; không xóa chúng trong lúc chạy hoặc phục hồi.

Trước khi sinh tiếp, JSONL được dựng lại từ cache cho input/cấu hình hiện tại:
giữ các kết quả thành công và dòng `skipped`, rồi append kết quả mới. Nhờ đó
không ghi trùng khi resume và khôi phục được trường hợp process chết giữa commit
SQLite và append JSONL, hoặc dòng cuối JSONL bị ghi dở. Snapshot `--export-only`
vẫn chứa đầy đủ trạng thái của mọi dòng. Không chỉnh tay file output khi đang chạy.

Cache key gồm nội dung question/answer, trace index, seed, model/revision,
detector, cấu hình generation và hash code. Đổi thứ tự dataset, output, số GPU
hoặc retry budget vẫn tái sử dụng kết quả phù hợp. Sample trùng chỉ sinh một
lần cho mỗi trace index; output vẫn giữ số dòng, thứ tự và metadata input.
`--num-traces N` tạo N seed ổn định cho mỗi sample. Với greedy decoding,
các trace có thể giống nhau. Không đảm bảo bit-identical giữa các GPU/backend.

Pin `--revision` tới commit model để chạy tái lập. Nếu thay nội dung model hoặc
tokenizer local tại cùng đường dẫn, hoặc cập nhật dependency, đổi `--cache-tag`
để vô hiệu hóa cache cũ. Script không hash toàn bộ model weights.

Dùng cùng cấu hình kèm `--export-only` để xuất lại cache mà không nạp model.
`--limit N` giới hạn số sample input, trước khi nhân với `--num-traces`.
`--require-eos` loại trace hết token budget khỏi **file SFT**, vẫn giữ chúng trong
cache và file chi tiết. Bỏ flag rồi export lại để lấy cả các trace đó.

File SFT có dạng:

```json
{"messages": [{"role": "user", "content": "Evaluate the limit ...\n\nExplain your solution step by step."}, {"role": "assistant", "content": "<trace đã qua leakage repair>"}]}
```

Role `user` dùng prompt không chứa ground truth. Trace assistant giữ các thẻ
thinking nếu tokenizer sử dụng; EOS cuối được bỏ. Nếu chat template kết thúc
bằng `<think>`, script thêm lại thẻ mở này vào assistant content. Trường `text`
trong file chi tiết là text nguyên bản pipeline trả về (skip special tokens),
còn `assistant_text` là nội dung đã chuẩn bị cho SFT. Cần dùng chat template
tương thích khi tokenize dữ liệu train.

Trạng thái `success` nghĩa là pipeline hoàn thành, chưa đánh giá đáp án đúng/sai
hoặc xác nhận hết leakage. `finish_reason` là `eos` hoặc `length`. File chi tiết
cũng giữ dòng `error`/`pending`, nhưng file SFT chỉ chứa dòng thành công.
File chi tiết cũng giữ dòng `skipped` cho input lỗi; sửa input rồi chạy lại sẽ
kiểm tra lại các dòng này. Cache của sample hợp lệ được tái sử dụng với cùng
code/cấu hình.
Exit code: 0 khi job hợp lệ đều thành công, 1 nếu lỗi/còn pending hoặc toàn bộ
input bị bỏ qua, 130 nếu Ctrl+C.

## Kiểm tra

```bash
python -m unittest discover -s method/generate/test -p 'test_generate_sft.py' -v
```

Tests không tải model: kiểm tra API notebook, worker process thực, crash/resume,
cache fingerprint, dedup, thứ tự output, retry, atomic export, model loader và
adapter thinking. Có regression test cho đáp án rỗng tại index 99592, kiểm tra
bỏ qua/resume và chạy lại sau khi bổ sung ground truth.
