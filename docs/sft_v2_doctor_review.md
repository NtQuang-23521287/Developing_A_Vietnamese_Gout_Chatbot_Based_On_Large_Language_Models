# Dữ liệu SFT v2: góp ý bác sĩ và các phiên bản dữ liệu

Dữ liệu SFT chính thức là **SFT v2.1** (`data/sft_v2_1`, 150 mẫu). Bác sĩ đã xem 151 mẫu và ghi chú trực tiếp vào 19 dòng.
Nhóm sửa các dòng đó theo ghi chú, giữ nguyên các dòng không có ghi chú, và không gửi bác sĩ duyệt lại (quyết định
19/11/2026).

## Bác sĩ xem dữ liệu (gửi 07/11/2026)

- 151 mẫu: toàn bộ 71 câu dấu hiệu nguy hiểm + 20 câu ngẫu nhiên mỗi loại còn lại (`--per-category 20
  --full-categories red_flag`). File bác sĩ trả: `artifacts/sft_v2/review/sft_v2_review_vong1_bac_si.xlsx`.
- Cột ĐÁNH GIÁ, Câu trả lời đã sửa, Loại lỗi, Ghi chú đều trống. Bác sĩ ghi chú thẳng vào ô câu trả lời, câu hỏi, hội
  thoại trước — chủ yếu chữ đỏ, có chỗ chữ đen (vd. đường link nguồn ở `dlg_017`). Vì vậy `build_sft_v2.py
  review_import` không đọc được file này; dữ liệu được dựng bằng `scripts/fix_sft_v2_data.py`.

| Bác sĩ chỉ ra | Số dòng | Ví dụ ghi chú |
|---|---|---|
| Có phản ứng thuốc mà khuyên "không tự ý ngưng" — phải ngưng thuốc và đi khám ngay | 11 | «ĐÃ TIÊU CHẢY NẶNG NÊN NGƯNG THUỐC NGAY VÀ ĐI KHÁM…» |
| Đau thắt ngực, khó thở là cấp cứu y khoa | 1 | `kbq_doc003_chunk0025_2` |
| Colchicine không phải thuốc hạ acid uric | 1 | `dlg_053` (kèm nguồn Dược thư Quốc gia 2022) |
| Chườm lạnh, tuyệt đối không chườm nóng | 1 | `dlg_017` |
| Không tự đổi sang corticoid khi colchicine không đỡ | 1 | `kbq_doc001_chunk0019_0` |
| Câu hỏi hoặc nhân vật không hợp lý (lao kê, "cháu 65 tuổi", sai tên thuốc) | 3 | `seed_qa_0016`, `dlg_057`, `seed_qa_0006` |
| Liều theo tư vấn bác sĩ | 1 | `kbq_doc001_chunk0005_0` |

Lỗi phổ biến nhất — quy tắc "không tự ý ngưng thuốc" bị áp cả khi đang có phản ứng thuốc — còn ở 2 dòng bác sĩ không
đánh dấu. Quy tắc của bác sĩ được áp cho các dòng đó và được viết thành luật kiểm tra dữ liệu
(`src/gout_llmops/data/doctor_rules.py`) và luật của lớp an toàn (`safety/guard.py`, `safety/policy.py`).

## Các phiên bản dữ liệu

| Phiên bản | Ngày | Nội dung | Dùng cho |
|---|---|---|---|
| `data/sft_v2` | 16/11 | 151 mẫu; 17 dòng có ghi chú được viết lại theo ghi chú (`doctor_feedback_audit.json`) | SFT v3 (17/11) |
| `data/sft_v2_1` | 19/11 | `data/sft_v2` + `doctor_round1_fixes.json` + kiểm tra luật bác sĩ; 150 mẫu (136 train, 14 val) | SFT v3.1, DPO |

`data/sft_v2` (16/11) làm đúng phần lớn: các bản viết lại bám ý bác sĩ, định dạng giống lúc suy luận, context là chunk
KB đầy đủ (453/453 khớp nguyên văn; ghi chú "tối đa 700 ký tự" trong manifest của bản này không đúng). Còn thiếu:
- sót 2 ghi chú ở cột Câu hỏi → ghi chú lọt vào câu hỏi (`seed_qa_0006`, `seed_qa_0016`, đều ở val), và câu trả lời
  `seed_qa_0016` vẫn đánh đồng lao kê với bệnh gút;
- 2 mẫu train cùng tình huống colchicine gây tiêu chảy vẫn khuyên "không tự ý ngưng thuốc"
  (`kbq_doc000_chunk0015_1`, `kbq_doc000_chunk0018_2`);
- nhãn nguồn RAG bị đổi thành `doctor_review_excerpt_i` (lúc suy luận là tên tài liệu).

`scripts/fix_sft_v2_data.py` dựng `data/sft_v2_1`: kiểm tra từng mẫu dựng lại giống hệt bản 16/11 ngoài nhãn nguồn, áp
8 chỉnh sửa (2 viết lại, 3 thêm ngoại lệ bác sĩ đã ghi ở dòng cùng tình huống, 1 sửa câu hỏi, 1 bỏ đường link khỏi lời
chatbot, 1 bỏ mẫu lao kê), rồi kiểm tra mọi mẫu bằng luật bác sĩ: còn vi phạm thì dừng. Mỗi chỉnh sửa ghi căn cứ trong
`doctor_round1_fixes.json`; `artifacts/sft_v2_1/rows.jsonl` là cùng dữ liệu ở định dạng pipeline, làm câu hỏi cho DPO.

Phân bố: 47% dấu hiệu nguy hiểm (dev: 7%), 13% tình huống lâm sàng (dev: 81%). Nếu SFT kém base ở nhóm tình huống lâm
sàng, thêm mẫu AI đã lọc chưa gửi bác sĩ: `fix_sft_v2_data.py --extra-unreviewed artifacts/sft_v2/filtered.jsonl`
(chỉ vào train, mẫu vi phạm luật bác sĩ bị bỏ, manifest ghi rõ).

## Khi viết khóa luận

Mô tả đúng mức độ tham gia của bác sĩ: bác sĩ **xem và góp ý** trên 151 mẫu (ghi chú 19 dòng); nhóm sửa theo góp ý và
không gửi bác sĩ xem lại bản sửa. Không ghi là bác sĩ đã chấm Đạt/Loại hay phê duyệt từng mẫu. Số liệu: bảng ở trên,
`data/sft_v2_1/manifest.json`, `data/sft_v2_1/fix_log.json`.

## Đánh giá SFT trên dữ liệu này

- Chỉ dùng dev (114 case) cho mọi quyết định; mốc so sánh là Qwen gốc có reranker
  (`20261112T060528Z_baseline_qwen3_8b_base_rerank_3504`). Tập test chạy một lần ở cuối.
- `eval_adapter.py score/gate` và `score_run_v2.py` từ chối split test/all nếu không có `--final-test`.
- `scripts/evaluate_post_sft.py` chỉ dành cho tái lập SFT v1 (kiểm tra cứng lr/10 epoch của SFT v1, chạy cả 381 case):
  cần `--legacy-sft-v1`.
- Run SFT v3 đã sinh 381 case trên máy thuê: chỉ chấm dev (notebook 04, bước 3, `SFT_V3_RUN_EXISTING`).
