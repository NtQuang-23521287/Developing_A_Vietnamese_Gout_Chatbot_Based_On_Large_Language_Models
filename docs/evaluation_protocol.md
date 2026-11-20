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
- Từ 20/11/2026, "protocol truy xuất" gồm cả embedding (model, revision, tách từ) đọc từ `rag_config.json` của run.
  Reranker nạp đúng revision ghi trong config (trước đó luôn nạp bản mới nhất, khi kiểm tra vẫn trùng revision đã ghim).

## Benchmark truy xuất và ablation retriever (20/11/2026)

`scripts/benchmark_retrieval.py` đo truy xuất trên KB thật so với `evidence_chunk_ids` của testset v03: 104 câu một
lượt của dev (bỏ 9 hội thoại nhiều lượt, 1 câu không có nhãn), mỗi embedding lấy 12 đoạn, có/không reranker giữ 3.
Chạy trên CPU; muốn có reranker thì cần `FlagEmbedding`, còn bkai cần `pip install -e '.[rag-vi]'`.

```bash
PYTHONPATH=src python3 scripts/benchmark_retrieval.py --split dev \
    --check-run artifacts/runs/20261025T074006Z_baseline_qwen3_8b_7963   # đối chiếu top-3 với một run thật
```

Kết quả (`artifacts/reports/rag/retrieval_benchmark_dev.md`): Hit@3 của cấu hình đang dùng (MiniLM + reranker) là
64,4%; bkai không reranker đạt 83,7% (+19,2 điểm %, McNemar p = 0,001). Nhãn bằng chứng chỉ trỏ vào 30 chunk, gần như
toàn bộ thuộc hai tài liệu trong nước, nên các số là cận dưới. Theo luật đã định trước, các thí nghiệm vẫn dùng
`rag_v1`: đổi retriever giữa chừng làm các mốc không so được với nhau, và dữ liệu SFT v2.1 lấy context bằng `rag_v1`.

Ablation đầu-cuối (notebook 04, bước 0R): `eval_adapter.py generate --rag-config configs/rag/rag_v2_bkai.yaml`
sinh với bkai (index dựng sẵn ở `artifacts/kb/gout_kb_v1_bkai`), rồi `compare_runs_v2.py` so với run `rag_v1` cùng
model. Kết quả chỉ dùng để báo cáo, không đưa vào cổng.

## Chốt chặn tập test (từ 19/11/2026)

- `eval_adapter.py generate/score/gate` và `score_run_v2.py` từ chối split `test` hoặc `all` (gồm test) nếu không có
  `--final-test`: mọi lựa chọn model, checkpoint, ngưỡng đều dựa trên dev; test chạy một lần cho cấu hình đã chốt.
- `scripts/evaluate_post_sft.py` (viết cho SFT v1: kiểm tra cứng lr/10 epoch, chạy cả 381 case) cần `--legacy-sft-v1`.
  Từ SFT v3 dùng `eval_adapter.py` trên dev. Chi tiết dữ liệu SFT v2/v2.1: `docs/sft_v2_doctor_review.md`.
