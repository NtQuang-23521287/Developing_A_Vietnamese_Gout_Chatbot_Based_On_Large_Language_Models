# Giai đoạn 5 — Safe Inference v1

Lớp an toàn đứng **sau** model: nhận câu hỏi, câu trả lời model đã sinh và context RAG, rồi quyết định giữ nguyên,
chèn cảnh báo hay thay bằng mẫu từ chối. Model, prompt và retrieval **không đổi**, nên chênh lệch giữa run gốc và
run an toàn chỉ do lớp an toàn gây ra.

```text
câu hỏi + history ──> Guard (luật) ──┐
                                      ├──> Policy ──> câu trả lời cuối
ChatEngine ──> câu trả lời gốc ──────┤
          └──> MC Dropout trên LoRA ──┘ (độ bất định u, so với ngưỡng τ hiệu chỉnh trên dev)
```

| Thành phần | File | Ghi chú |
|---|---|---|
| Guard theo luật | `src/gout_llmops/safety/guard.py` | red_flag (emergency/prompt), self_medication, out_of_scope |
| Chính sách + mẫu câu | `src/gout_llmops/safety/policy.py` | mẫu câu **cần bác sĩ duyệt** |
| Độ bất định (thuần) | `src/gout_llmops/safety/uncertainty.py` | tín hiệu, dựng lại prompt |
| MC Dropout (GPU) | `src/gout_llmops/safety/mc_dropout.py` | chỉ bật `lora_dropout` |
| Hiệu chỉnh | `src/gout_llmops/safety/calibration.py` | AUROC, risk–coverage, chọn τ |
| Online | `src/gout_llmops/chat/safe_engine.py` | `SafeChatEngine` bọc `ChatEngine` |
| CLI | `scripts/safe_inference.py` | guard-eval, uncertainty, calibrate, apply, report |
| Cấu hình | `configs/safety/safe_inference_v1.yaml` | |
| Colab | `notebooks/05_safe_inference_colab.ipynb` | |

## 1. Guard theo luật

Luật regex chạy trên văn bản đã bỏ dấu (bắt cả câu gõ không dấu), riêng các cặp từ chỉ phân biệt được bằng dấu
(dừng/dùng) chạy trên văn bản có dấu. Ưu tiên **recall**: báo nhầm chỉ thêm một đoạn lưu ý, còn bỏ sót red flag
là lỗi an toàn.

- **red_flag / emergency**: sốt, đau ngực/khó thở/ngất, xuất huyết tiêu hóa, dấu hiệu suy thận/tiểu ra máu,
  dấu hiệu ngộ độc colchicine, dị ứng thuốc nặng, dấu hiệu thần kinh, đau bụng nhiều.
- **red_flag / prompt**: trẻ em sưng đau khớp, tophi vỡ/chảy mủ, rối loạn tiêu hóa khi dùng colchicine,
  hỏi có tự xử trí tại nhà được không khi đang có triệu chứng.
- **self_medication**: tự ý dùng/tăng/giảm/ngừng thuốc, gấp đôi liều, dùng thuốc của người khác, mua thuốc không
  đơn, xin liều cụ thể, corticoid/NSAID kéo dài, thuốc nam/nhịn đói, thuốc + bệnh nền chống chỉ định, thai kỳ,
  dùng lại thuốc từng dị ứng. Chỉ xét câu hỏi hiện tại.
- Red flag ở lượt trước **vẫn còn hiệu lực** ở lượt sau (`history_flags`).
- **out_of_scope**: cả hội thoại không có từ khóa sức khỏe nào. Cố ý rất hẹp; câu hỏi y khoa khác vẫn để model trả lời.

Luật được xây trên câu hỏi SFT v2 (641 câu, nhãn LLM sinh) và split dev; **không dùng split test**.
Kết quả `guard-eval` (guard_rules_v1, `artifacts/reports/safety/guard_eval.md`):

| Nguồn | Recall red flag | Báo nhầm red flag | Recall tự ý dùng thuốc | Báo nhầm tự ý dùng thuốc |
|---|---|---|---|---|
| dev (114) | 1.000 (8/8) | 0.021 (2/94) | 1.000 (12/12) | 0.000 (0/94) |
| SFT v2 (641) | 0.713 (n=80) | 0.012 (n=429) | 0.886 (n=132) | 0.051 (n=429) |

Hai lần báo red flag trên dev đều là trẻ 7 và 9 tuổi sưng ngón chân cái. Nhãn của hai câu này là
clinical_reasoning, nhưng khuyên trẻ đi khám sớm vẫn phù hợp. Recall thấp hơn trên SFT v2 phần lớn do
nhãn nhiễu (vd câu về giảm cân nhanh bị gắn red_flag). Vì luật được chỉnh trên dev, số liệu dev
là lạc quan. Số liệu khách quan lấy từ `guard-eval --final-test`, chạy một lần ở cuối.

## 2. Chính sách

Thứ tự áp dụng: out_of_scope → độ bất định → red flag → tự ý dùng thuốc.

1. `out_of_scope`: thay câu trả lời bằng mẫu từ chối (online: không gọi model).
2. `u > τ` (chỉ khi calibration **bật**): `abstain` thì thay bằng mẫu từ chối an toàn + tên tài liệu liên quan;
   `hedge` thì giữ câu trả lời, thêm cảnh báo độ chắc chắn thấp.
3. red flag: chèn banner (emergency: "Cần đi khám ngay… gọi 115"; prompt: "Nên đi khám sớm…") kèm lý do cụ thể
   theo luật đã khớp, **trừ khi** câu trả lời đã khuyên đi khám ngay/cấp cứu. Khi đã từ chối thì luôn chèn banner.
4. self_medication: chèn lưu ý không tự ý dùng/đổi thuốc, trừ khi câu trả lời đã nói rõ điều đó.

## 3. Độ bất định — MC Dropout trên LoRA

Thay vì sinh lại N lần, câu trả lời greedy y đã sinh được **chấm lại** N lần dưới N mặt nạ dropout khác nhau.
Chỉ các module `lora_dropout` của PEFT ở chế độ train, phần còn lại của model ở eval. Cách này rẻ hơn sinh lại
khoảng (độ dài câu trả lời) lần, và đo độ bất định của đúng câu người dùng nhận được.

Với mỗi lượt i và token t: log p_i(y_t), entropy H(p_i(·|y_<t)). Tín hiệu (lớn = bất định hơn):

| Tín hiệu | Định nghĩa | Cần dropout |
|---|---|---|
| `det_mean_nll` | −mean_t log p(y_t), dropout tắt | không |
| `det_mean_entropy` | mean_t H(p(·)), dropout tắt | không |
| `det_max_nll` | max_t −log p(y_t), dropout tắt | không |
| `mc_mean_nll` | mean_i của NLL trung bình/token | có |
| `mc_nll_std` | std_i của NLL trung bình/token (epistemic) | có |
| `mc_token_lp_var` | mean_t var_i log p_i(y_t) | có |
| `mc_mutual_info` | mean_t [H(p̄) − mean_i H(p_i)] (BALD) | có |
| `ret_low_relevance` | 1 − điểm truy xuất cao nhất | không |

Yêu cầu adapter có `lora_dropout > 0`: SFT v3 và DPO v1 dùng 0.05. SFT v1 (`lora_dropout = 0`) chỉ dùng để
thử pipeline bằng `--p-override`, không đưa số liệu đó vào khóa luận. Prompt được dựng lại từ predictions
(history = câu trả lời thật của model); script so số token với `input_tokens` của run để phát hiện lệch.

## 4. Hiệu chỉnh ngưỡng τ (chỉ trên dev)

- Nhãn **không đạt**: reward < 0.5 hoặc safety_risk ≥ 2 hoặc hallucination ≥ 2 (Judge v2 + Reward v2 có sẵn).
- Với mỗi tín hiệu: AUROC (Mann–Whitney) + CI95 bootstrap, đường **risk–coverage** có trọng số risk_level, AURC.
- Chọn tín hiệu: `auto` = AUROC cao nhất trên dev (có selection bias → báo cáo AUROC trên test ở cuối).
- Chọn τ: `budget` = từ chối tối đa 15% câu bất định nhất; hoặc `target_risk`.
- **Chỉ bật từ chối khi cận dưới CI95 của AUROC > 0.5**: tín hiệu không hơn ngẫu nhiên thì từ chối chỉ làm mất
  câu trả lời đúng.

Kết quả sơ bộ trên Qwen gốc Giai đoạn 2 (run không có reranker), dev, tín hiệu duy nhất không cần GPU: `ret_low_relevance` có AUROC 0.527,
CI95 [0.41, 0.64]. Cơ chế an toàn **tự tắt** nhánh từ chối: độ liên quan truy xuất không dự đoán được câu
trả lời kém.

## 5. Đánh giá

`apply` tạo run mới `<run>__<tag>` (giữ `answer_raw`, ghi `safety_decisions.jsonl`, `safe_manifest.json`).
Run này được chấm bằng `eval_adapter.py score`, như mọi run khác. Cache judge theo sha256 của input nên chỉ
những case bị đổi mới gọi API. `report` tách kết quả theo hành động: unchanged / history_changed /
red_flag_banner / self_medication_note / abstain. Nhóm "unchanged" phải có điểm giống hệt run gốc.

Lưu ý khi diễn giải:
- Lớp an toàn áp **offline** lên câu trả lời đã sinh. Ở hội thoại nhiều lượt, các lượt sau được sinh với history
  chưa qua lớp an toàn. Chỉ lượt cuối có độ bất định.
- `abstain` làm giảm completeness/utility nên reward trung bình có thể giảm. Đây là đánh đổi cần báo cáo cùng
  tỉ lệ safety/hallucination ≥ 2 và risk–coverage, không chỉ nhìn reward.
- τ và tín hiệu chọn trên dev; số liệu cuối lấy trên test (267 case), chạy một lần với `--final-test`.

## 6. Chưa làm ở v1

- `diagnostic_certainty`: phát hiện câu trả lời khẳng định chẩn đoán.
- Vòng phản hồi: case bị từ chối hoặc gắn cờ được xuất cho bác sĩ sửa (dùng lại `data/doctor_review.py`), rồi
  bản sửa thành cặp chosen/rejected cho DPO vòng sau.
- Gửi bác sĩ duyệt mẫu câu trong `policy.py` và danh sách luật red flag.
