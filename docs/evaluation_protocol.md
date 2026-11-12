# Evaluation Protocol v1

## Mục tiêu

Chọn mô hình nền trong cùng điều kiện RAG trước SFT. Protocol phải được khóa trước full run.

## Đơn vị đánh giá

- Single-turn: 1 case = 1 prediction.
- Multi-turn: 1 case = 3 prediction có thứ tự; T2/T3 dùng output thật của model trước đó.
- Nếu một turn multi lỗi, các turn phụ thuộc sau đó là `skipped_dependency_failure`, không biến thành điểm 0 giả.

## Fair comparison

Giữ cố định testset checksum, KB manifest checksum, system prompt checksum, retrieval config, token limits và evaluator version. Không tuning riêng một model rồi so với model khác bằng config cũ.

## Candidate testset hiện tại

Group3 v02 đã sạch cấu trúc nhưng `reference_status=pending_kb_alignment`. Có thể dùng để smoke-test generation/retrieval; chưa nên báo cáo các điểm faithfulness/completeness như ground-truth clinical benchmark cho đến khi reference/evidence được duyệt.

## 9 tiêu chí

6 quality: Faithfulness, Completeness, Relevance, Context Recall, Patient Utility, Refusal Appropriateness.

3 risk: Hallucination Level, Omission Risk, Safety Risk.

Risk có hướng lower-is-better. Fatal safety violation được ghi riêng, không che bằng average score.

## Ghi chú protocol: reranker (phát hiện 2026-11-12)

`configs/rag/rag_v1.yaml` bật reranker `BAAI/bge-reranker-v2-m3`, nhưng retriever cũ **nuốt lỗi** khi
FlagReranker không nạp được rồi chạy tiếp không có reranker. Kiểm tra `rerank_score` trong context đã lưu cho thấy:

| Run | Reranker |
|---|---|
| Baseline 2026-10-11 (Gemma3, SeaLLMs; testset v02) | có |
| Baseline 2026-10-25 (Qwen3, Gemma3, SeaLLMs; testset v03) | **không** |
| SFT v1 (2026-10-31) | **không** |
| Dữ liệu SFT v2 (`build_sft_v2.py retrieve`, dừng nếu reranker lỗi) và DPO (dùng lại context SFT v2) | có |

Hệ quả và cách xử lý:
- Chọn mô hình nền (3 baseline 2026-10-25) và so sánh Base vs SFT v1 vẫn hợp lệ **trong nội bộ** vì cùng không có
  reranker. Khi báo cáo cần ghi rõ: RAG thực tế ở Giai đoạn 2 là dense top-3, không rerank.
- Từ SFT v2 trở đi dùng reranker như config. Mốc so sánh cho SFT v3/DPO là Qwen gốc **sinh lại có reranker**
  (notebook 04, bước 0, `qwen3_8b_base_rerank`). So run đó với run Giai đoạn 2 cho ablation có/không reranker.
- `FaissRetriever` giờ báo lỗi khi config bật reranker mà không nạp được. `eval_adapter.py gate` từ chối so hai run
  khác protocol truy xuất; `compare_runs_v2.py` in cảnh báo (`src/gout_llmops/evaluation/protocol_check.py`).
