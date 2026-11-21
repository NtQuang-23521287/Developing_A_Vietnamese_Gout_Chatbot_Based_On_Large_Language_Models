# Gout LLMOps

Hệ thống LLMOps cho khóa luận **Chatbot tư vấn bệnh Gout tiếng Việt**. Repository này được thiết kế để dùng xuyên suốt từ xây dựng Knowledge Base, benchmark mô hình nền, SFT, DPO, đánh giá an toàn đến đóng gói API/UI.

> Trạng thái ban đầu: **LLMOps Foundation v1**. Chưa công bố model tốt nhất. Ba model baseline được cấu hình nhưng phải chạy trên cùng một protocol trước khi chọn.

## 1. Nguyên tắc thiết kế

- **Một codebase, nhiều phase**: baseline, SFT và DPO dùng chung `ChatEngine`, RAG và Evaluation Harness.
- **Config-driven**: đổi model/RAG/evaluation bằng YAML, không copy notebook.
- **Reproducible**: mỗi run lưu model revision, Git commit, KB/testset checksum, prompt/config và seed.
- **Không leakage**: reference/ground truth chỉ đi vào evaluator sau generation; không được dùng để retrieval hoặc prompt model.
- **Multi-turn đúng nghĩa**: history của một case chỉ chứa câu hỏi và câu trả lời thật của model; reset giữa các case.
- **Data lineage**: Git quản lý code/config; DVC quản lý data/index lớn; MLflow quản lý experiment/metrics/artifacts.
- **Medical safety**: đánh giá chất lượng và rủi ro tách riêng; dữ liệu chưa được chuyên môn duyệt không được gọi là gold clinical reference.

## 2. Mapping với đề cương khóa luận

| Đề cương | Trong repository |
|---|---|
| Giai đoạn 1 – Hạ tầng dữ liệu & khung đánh giá | `src/gout_llmops/data`, `rag`, `evaluation`, DVC, MLflow |
| Giai đoạn 2 – Tuyển chọn mô hình nền | `scripts/run_baseline.py`, `configs/models/*` |
| Giai đoạn 3 – SFT | `src/gout_llmops/training/sft.py`, `configs/training/sft_v1.yaml` |
| Giai đoạn 4 – DPO | `scripts/dpo_pipeline.py`, `src/gout_llmops/training/dpo.py`, `configs/training/dpo_v1.yaml` (Qwen3-8B), `configs/training/dpo_gemma3_4b_v1.yaml` (Gemma3-4B) |
| Giai đoạn 5 – Safe Inference & triển khai | `src/gout_llmops/safety`, `serving` |

## 3. Cấu trúc

```text
configs/              cấu hình model, RAG, evaluation, training, safety
data/raw/             nguồn bất biến/ít biến đổi; không ghi output vào đây
data/processed/       chunks, normalized testsets
artifacts/            outputs của build/run/report
src/gout_llmops/      business logic chính
scripts/              entrypoints CLI mỏng
notebooks/            orchestration/phân tích, không chứa logic lõi
prompts/              prompt có version
rubrics/              rubric đánh giá 9 tiêu chí
tests/                unit/integrity tests
docs/                 architecture, protocol, versioning
```

## 4. Dữ liệu được đóng gói trong bản khởi tạo

- `data/raw/knowledge_base/gout_guideline_1.pdf`: nguồn guideline lõi hiện có.
- `data/raw/knowledge_base/Group1_RAG_Corpus.csv`: corpus web bổ sung; **không mặc định coi là nguồn lâm sàng chính thống**.
- `data/raw/testset/Group3_Testset_cases_v02.jsonl`: candidate testset gồm single/multi-turn, trạng thái reference hiện là `pending_kb_alignment`.
- `data/raw/testset/Group3_Testset_candidate_v02.csv`: dạng phẳng theo lượt để kiểm tra.

Trước benchmark chính thức cần chạy alignment với KB và duyệt reference/evidence.

## 5. Cài đặt nhanh

Python 3.10–3.12 được khuyến nghị.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -U pip
pip install -e ".[eval,dev]"
cp .env.example .env       # Windows có thể copy thủ công
```

GPU Colab có thể dùng:

```bash
pip install -e ".[eval]"
```

## 6. Kiểm tra foundation trước khi chạy GPU

```bash
pytest -q
python scripts/validate_testset.py
python scripts/create_data_manifests.py
```

## 7. Xây Knowledge Base v1

```bash
python scripts/build_kb.py --config configs/rag/rag_v1.yaml
```

Output mặc định:

```text
artifacts/kb/gout_kb_v1/
├── index.faiss
├── chunks.jsonl
└── manifest.json
```

## 8. Align candidate testset với KB

Sau khi build KB, xuất gói review để xác nhận mỗi query có được KB hỗ trợ hay cần expected refusal:

```bash
python scripts/export_kb_alignment_review.py
```

Chi tiết ở `docs/reference_review.md`. Không chạy Judge/RAGAS như benchmark y khoa chính thức cho tới khi reference/evidence được duyệt.

## 9. Smoke benchmark

Chỉ chạy 2 case đầu với Qwen3 để kiểm tra toàn pipeline:

```bash
python scripts/run_baseline.py \
  --model configs/models/qwen3_8b.yaml \
  --eval configs/evaluation/baseline_v1.yaml \
  --limit 2 \
  --no-judge \
  --no-ragas
```

Sau smoke test thành công mới chạy full 3 model:

```bash
python scripts/run_baseline.py --model configs/models/seallms_v3_7b.yaml --eval configs/evaluation/baseline_v1.yaml
python scripts/run_baseline.py --model configs/models/qwen3_8b.yaml       --eval configs/evaluation/baseline_v1.yaml
python scripts/run_baseline.py --model configs/models/gemma3_4b.yaml      --eval configs/evaluation/baseline_v1.yaml
```

Mỗi run sinh một thư mục bất biến trong `artifacts/runs/<run_id>/` với prediction, retrieval trace, config snapshot, manifest và summary.

## 10. MLflow

Mặc định dùng local store:

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Truy cập `http://127.0.0.1:5000`.

## 11. DVC

Repository có `dvc.yaml` và `params.yaml`. Sau khi copy project vào Git repository:

```bash
dvc init
dvc add data/raw/knowledge_base
dvc add data/raw/testset
git add .
git commit -m "chore: initialize llmops foundation"
```

Không commit `.env`, model weights, FAISS index hoặc run artifacts lớn vào Git.

## 12. Quy tắc benchmark 3 model

Để phép so sánh có giá trị, cả 3 model phải dùng cùng:

- testset version/checksum;
- KB/index version;
- retrieval policy và top-k;
- system prompt;
- max input/output tokens;
- seed/decoding policy;
- rubric/judge version.

Khác biệt chính ở baseline selection chỉ nên là **model config**.

## 13. Chưa nên làm ngay

Không chọn Best Model từ smoke run; không train SFT/DPO trước khi baseline protocol được khóa; không gọi candidate testset là gold set khi chưa review; không dùng reference để truy xuất; không chỉnh config giữa ba model mà không version thành experiment mới.
