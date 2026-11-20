# Benchmark truy xuất — dev (104 câu hỏi một lượt, KB 86 chunk)

Lấy 12 đoạn bằng embedding, giữ 3. Bỏ qua: {'multi_turn': 9, 'no_evidence': 1}.

| Pipeline | Hit@3 [CI95] | Recall@3 | MRR@12 | Hit@12 |
| --- | --- | --- | --- | --- |
| minilm | 57.7% [48.1%; 66.7%] | 0.239 | 0.466 | 92.3% |
| minilm+rerank | 64.4% [54.9%; 73.0%] | 0.290 | 0.484 | 92.3% |
| bkai | 83.7% [75.4%; 89.5%] | 0.367 | 0.563 | 92.3% |
| bkai+rerank | 76.0% [66.9%; 83.2%] | 0.335 | 0.509 | 92.3% |
| bge-m3 | 49.0% [39.6%; 58.5%] | 0.247 | 0.457 | 96.2% |
| bge-m3+rerank | 69.2% [59.8%; 77.3%] | 0.308 | 0.481 | 96.2% |

| So với | ΔHit@3 [CI95] | McNemar p | ΔMRR@12 [CI95] |
| --- | --- | --- | --- |
| minilm − minilm+rerank | -6.7 điểm % [-17.3; +3.8] | 0.296 | -0.018 [-0.069; +0.035] |
| bkai − minilm+rerank | +19.2 điểm % [+8.7; +29.8] | 0.001 | +0.079 [+0.010; +0.145] |
| bkai+rerank − minilm+rerank | +11.5 điểm % [+1.9; +21.2] | 0.043 | +0.025 [-0.024; +0.075] |
| bge-m3 − minilm+rerank | -15.4 điểm % [-26.9; -2.9] | 0.023 | -0.027 [-0.081; +0.030] |
| bge-m3+rerank − minilm+rerank | +4.8 điểm % [-2.9; +12.5] | 0.359 | -0.003 [-0.053; +0.047] |

| Embedding | Tham số (triệu) | Chiều | Đọc tối đa (token) | Mã hoá 86 chunk (s) | Mỗi câu hỏi (ms) |
| --- | --- | --- | --- | --- | --- |
| minilm (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | 117.7 | 384 | 128 | 7.3 | 72.7 |
| bkai (bkai-foundation-models/vietnamese-bi-encoder) | 135.0 | 768 | 256 | 34.6 | 107.0 |
| bge-m3 (BAAI/bge-m3) | 567.8 | 1024 | 8192 | 237.1 | 520.9 |

Reranker BAAI/bge-reranker-v2-m3: khoảng 67423 ms cho 12 cặp mỗi câu hỏi trên CPU.

Đối chiếu run thật (không reranker): 104/104 câu có cùng tập top-3.
