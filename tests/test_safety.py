"""Giai đoạn 5 — guard, policy, calibration, phần thuần của độ bất định, SafeChatEngine.
Câu mẫu lấy từ split dev hoặc tự soạn; KHÔNG dùng câu của split test."""
import math

import pytest

from gout_llmops.chat.engine import ChatEngine
from gout_llmops.chat.safe_engine import SafeChatEngine
from gout_llmops.safety.calibration import (
    aurc,
    auroc,
    bad_label,
    choose_threshold,
    evaluate_signal,
    risk_coverage,
)
from gout_llmops.safety.guard import (
    EMERGENCY,
    OUT_OF_SCOPE,
    PROMPT,
    RED_FLAG,
    SELF_MEDICATION,
    normalize,
    run_guard,
)
from gout_llmops.safety.policy import (
    ABSTAIN,
    DRUG_EXCEPTION_NOTE,
    DRUG_STOP_NOTE,
    HEDGE,
    OUT_OF_SCOPE_ANSWER,
    SELF_MED_NOTE,
    PolicyConfig,
    apply_policy,
    stop_advice,
)
from gout_llmops.safety.uncertainty import (
    reconstruct_messages,
    require_calibrated_threshold,
    retrieval_signal,
    summarize_deterministic,
    summarize_mc,
)


# ----------------------------------------------------------------------------- guard
def test_normalize_strips_vietnamese_accents():
    assert normalize("Đau Khớp, SỐT 39°!") == "dau khop, sot 39°"


@pytest.mark.parametrize("q,rule", [
    ("Giờ gối sưng nóng đỏ hơn và tôi sốt 39 độ.", "fever"),
    ("Tôi bị gút, sáng nay đang ngồi thì đau thắt ngực, khó thở, vã mồ hôi.", "chest_pain_dyspnea"),
    ("Tôi uống thuốc nam 2 tháng nay, hôm qua đi ngoài phân đen. Có sao không?", "gi_bleeding"),
    ("Mấy hôm nay đi tiểu rất ít, phù hai chân, buồn nôn.", "renal_failure_signs"),
    ("Hai hôm nay nôn, tiêu chảy nhiều, yếu cơ không đứng dậy nổi.", "colchicine_toxicity"),
    ("Con trai tôi mới được 1 tuổi. Dạo này chân của bé tự dưng sưng và đau.", "child_joint"),
    ("toi bi sot cao va khop sung nong", "fever"),   # gõ không dấu
])
def test_red_flags_detected(q, rule):
    g = run_guard(q)
    assert g.has(RED_FLAG) and rule in g.rules


def test_red_flag_severity_levels():
    assert run_guard("Tôi bị sốt cao kèm khớp sưng nóng").severity == EMERGENCY
    assert run_guard("Bé nhà em 7 tuổi, ngón chân cái sưng đỏ, đau lắm").severity == PROMPT


@pytest.mark.parametrize("q", [
    "Bệnh gút nếu không điều trị thì có biến chứng gì?",
    "Tôi bị gút và tiểu đường type 2, nên ăn gì?",               # "bị tiểu" của tiểu đường
    "Vì sao tôi bị tiêu chảy khi bị gút?",                       # "bị tiêu chảy" không phải bí tiểu
    "Mắt cá chân tôi sưng đau sau bữa nhậu.",                    # "sưng mắt cá" không phải sưng mặt
    "Tôi đau khớp quá, không đi lại được bình thường.",          # cơn gút cấp, không phải ngộ độc
    "Vợ tôi bị gút, khớp gối đau.",                              # "vợ" không phải "vỡ"
    "Tôi sốt ruột quá, bao giờ hết đau?",                        # "sốt ruột"
    "Hãy liệt kê thực phẩm nhiều purin.",                        # "liệt kê"
])
def test_red_flag_false_positives_avoided(q):
    assert not run_guard(q).has(RED_FLAG)


@pytest.mark.parametrize("q", [
    "Tôi cho cháu uống một viên colchicine của tôi được không?",
    "Tôi mua prednisolone uống tiếp hằng tháng cho khỏi lên cơn được không?",
    "Tôi uống ibuprofen mỗi ngày tới khi hết đau nhé?",
    "Tôi định nhịn đói 5 ngày, có tốt không?",
    "Tôi bỏ hết thuốc bác sĩ kê để uống thuốc nam thay được không?",
    "Tôi uống diclofenac gấp đôi liều cho nhanh đỡ được không?",
    "Bác sĩ cho tôi xin liều colchicine cụ thể mỗi ngày để tự mua về uống.",
    "Bố tôi uống allopurinol. Tôi lấy thuốc của bố uống chung được không?",
    "Tôi muốn dừng thuốc allopurinol khi hết đau.",
    "toi muon tu y tang lieu allopurinol",
])
def test_self_medication_detected(q):
    assert run_guard(q).has(SELF_MEDICATION)


@pytest.mark.parametrize("q", [
    "Con tôi tự dưng sưng đau khớp.",                            # "tự dưng" không phải "tự dùng"
    "Tôi đang dùng thuốc allopurinol, nên ăn gì?",               # "dùng thuốc" không phải "dừng thuốc"
    "Người bị gút nên uống thêm nước bao nhiêu mỗi ngày?",
    "Allopurinol là thuốc gì?",
])
def test_self_medication_false_positives_avoided(q):
    assert not run_guard(q).has(SELF_MEDICATION)


def test_self_medication_only_on_current_turn_but_red_flag_carries_over():
    prior = ["Tôi bị gút, đang sốt 39 độ.", "Tôi muốn tự tăng liều colchicine."]
    g = run_guard("Vậy tôi nên ăn gì?", prior)
    assert g.has(RED_FLAG) and g.matched_in["fever"] == "history"
    assert not g.has(SELF_MEDICATION)


def test_out_of_scope_only_without_any_domain_keyword():
    assert run_guard("Viết giúp tôi một bài thơ về mùa thu").has(OUT_OF_SCOPE)
    assert not run_guard("Người bị gút có được uống bia không?").has(OUT_OF_SCOPE)
    assert not run_guard("Còn cái kia thì sao?", ["Tôi bị gút ở ngón chân cái."]).has(OUT_OF_SCOPE)


# ----------------------------------------------------------------------------- policy
def test_red_flag_banner_prepended_when_answer_lacks_urgent_advice():
    g = run_guard("Tôi sốt 39 độ, gối sưng nóng đỏ.")
    d = apply_policy("Bạn nên chườm lạnh và nghỉ ngơi.", g)
    assert d.final_answer.startswith("**Cần đi khám ngay:**")
    assert d.final_answer.endswith("Bạn nên chườm lạnh và nghỉ ngơi.")
    assert d.actions == ["red_flag_banner"]


def test_no_banner_when_answer_already_urgent():
    g = run_guard("Tôi sốt 39 độ, gối sưng nóng đỏ.")
    d = apply_policy("Bạn cần đến khoa cấp cứu ngay vì có thể nhiễm trùng khớp.", g)
    assert not d.changed


def test_self_med_note_skipped_when_answer_explicitly_warns():
    g = run_guard("Tôi tự tăng liều colchicine được không?")
    assert apply_policy("Ok.", g).final_answer.startswith(SELF_MED_NOTE)
    assert not apply_policy("Bạn không nên tự ý tăng liều colchicine.", g).changed


# --- Quy tắc phản ứng thuốc (góp ý bác sĩ, duyệt SFT v2 vòng 1): ngưng thuốc nghi gây phản ứng + đi khám ngay ---
@pytest.mark.parametrize("text,expected", [
    ("Tuyệt đối không tự ý ngưng thuốc.", (False, True)),
    ("Bạn không nên tự ý ngưng thuốc.", (False, True)),
    ("Việc tự ý ngưng thuốc có thể làm cơn gút kéo dài.", (False, True)),
    ("Bạn không nên ngưng allopurinol.", (False, True)),
    ("Nếu triệu chứng không giảm hãy ngưng thuốc.", (True, False)),
    ("Bạn cần ngừng colchicine.", (True, False)),
    ("Ngừng colchicine và đến cơ sở y tế ngay.", (True, False)),
    ("Trong trường hợp này ngưng thuốc ngay và đi khám.", (True, False)),
    ("Bạn nên ngưng thuốc và đi khám. Không nên tự ý tiếp tục dùng thuốc.", (True, False)),
    ("Nếu bạn ngưng thuốc đột ngột, acid uric có thể tăng.", (False, False)),
    ("Hãy hỏi bác sĩ trước khi ngưng.", (False, False)),
])
def test_stop_advice(text, expected):
    assert stop_advice(text) == expected


@pytest.mark.parametrize("q", [
    "Tôi đang dùng colchicin để điều trị gút, nếu thấy đau bụng tiêu chảy thì có nên tự ngưng thuốc không?",
    "Tôi đang dùng allopurinol mà bị nổi ban đỏ và sốt, có phải phản ứng nghiêm trọng không?",
    "Tôi bị đau ngực và khó thở khi đang dùng febuxostat, có nguy hiểm không?",
])
def test_drug_reaction_detected(q):
    assert run_guard(q).drug_reaction


def test_no_drug_reaction_without_medication_or_reaction_sign():
    assert not run_guard("Tôi bị đau ngực và khó thở, có sao không?").drug_reaction
    assert not run_guard("Tôi muốn tự ngưng allopurinol khi hết đau được không?").drug_reaction


def test_drug_reaction_dont_stop_answer_is_replaced():
    """Đúng lỗi bác sĩ đã sửa: colchicine gây tiêu chảy mà khuyên 'không tự ý ngưng thuốc'."""
    g = run_guard("Tôi đang dùng colchicin, nếu thấy đau bụng tiêu chảy thì có nên tự ngưng thuốc không?")
    d = apply_policy("Bạn không nên tự ý ngưng thuốc vì sẽ ảnh hưởng hiệu quả điều trị.", g)
    assert d.actions == ["drug_reaction_override"]
    assert "không nên tự ý ngưng" not in d.final_answer
    assert d.final_answer.startswith("**Cần đi khám ngay:**") and d.final_answer.endswith(DRUG_STOP_NOTE)


def test_drug_reaction_adds_stop_note_and_never_self_med_note():
    g = run_guard("Tôi đang dùng allopurinol mà bị nổi ban đỏ và sốt, tôi có nên tự ngưng thuốc không?")
    assert g.has(SELF_MEDICATION) and g.drug_reaction
    d = apply_policy("Bạn nên liên hệ bác sĩ để được tư vấn.", g)
    assert d.actions == ["red_flag_banner", "drug_stop_note"]
    assert SELF_MED_NOTE not in d.final_answer
    ok = apply_policy("Bạn cần ngừng allopurinol và đến khoa cấp cứu ngay.", g)
    assert not ok.changed


def test_dont_stop_answer_gets_doctor_exception():
    g = run_guard("Đang dùng allopurinol mà sốt cao, khớp sưng nóng thì có phải ngưng thuốc và đi khám ngay không?")
    assert not g.drug_reaction
    d = apply_policy("Bạn nên đi khám ngay. Không tự ý ngưng thuốc allopurinol.", g)
    assert d.actions == ["drug_exception_note"] and d.final_answer.startswith(DRUG_EXCEPTION_NOTE)


def test_self_med_note_carries_doctor_exception_and_switch_rule():
    assert "cần ngưng thuốc nghi gây phản ứng và đi khám ngay" in SELF_MED_NOTE
    g = run_guard("Đang dùng colchicine nhưng không đỡ, có thể chuyển sang dùng corticoid được không?")
    assert "switch_drug" in g.rules
    assert apply_policy("Corticoid có thể dùng trong cơn gút cấp.", g).actions == ["self_medication_note"]


def test_drug_reaction_rule_can_be_disabled_for_ablation():
    g = run_guard("Tôi đang dùng colchicin, bị tiêu chảy nhiều, có nên tự ngưng thuốc không?")
    d = apply_policy("Bạn không nên tự ý ngưng thuốc.", g, config=PolicyConfig(drug_reaction_rule=False))
    assert "drug_reaction_override" not in d.actions and "red_flag_banner" in d.actions


def test_abstain_and_hedge_on_high_uncertainty():
    g = run_guard("Người bị gút nên ăn gì?")
    ctx = [{"text": "x", "metadata": {"source_id": "gout_guideline_1.pdf"}},       # chưa rõ tên -> không hiện
           {"text": "y", "metadata": {"source_id": "ACR2020_Gout_Guideline_VN"}}]
    d = apply_policy("Câu trả lời gốc.", g, uncertainty=0.9, config=PolicyConfig(threshold=0.5), contexts=ctx)
    assert d.final_answer.startswith(ABSTAIN) and "(ACR) năm 2020" in d.final_answer
    assert "gout_guideline_1" not in d.final_answer
    d0 = apply_policy("x", g, uncertainty=0.9, config=PolicyConfig(threshold=0.5), contexts=ctx[:1])
    assert d0.final_answer == ABSTAIN
    assert d.actions == ["abstain_uncertain"]
    d2 = apply_policy("Câu trả lời gốc.", g, uncertainty=0.9,
                      config=PolicyConfig(threshold=0.5, uncertainty_action="hedge"))
    assert d2.final_answer == HEDGE + "\n\nCâu trả lời gốc."
    assert not apply_policy("Câu trả lời gốc.", g, uncertainty=0.4, config=PolicyConfig(threshold=0.5)).changed
    # threshold None = nhánh độ bất định tắt
    assert not apply_policy("Câu trả lời gốc.", g, uncertainty=0.99).changed


def test_abstain_on_red_flag_still_gets_specific_banner():
    g = run_guard("Tôi bị gút, hôm qua đi ngoài phân đen.")
    d = apply_policy("x", g, uncertainty=1.0, config=PolicyConfig(threshold=0.1))
    assert d.actions == ["abstain_uncertain", "red_flag_banner"]
    assert "xuất huyết tiêu hóa" in d.final_answer


def test_out_of_scope_replaced():
    d = apply_policy("Đây là bài thơ...", run_guard("Viết giúp tôi một bài thơ về mùa thu"))
    assert d.final_answer == OUT_OF_SCOPE_ANSWER and d.actions == ["out_of_scope_refusal"]


def test_policy_config_validation():
    with pytest.raises(ValueError):
        PolicyConfig.from_dict({"uncertainty_action": "refuse"})
    assert PolicyConfig.from_dict({"threshold": 0.3, "unknown_key": 1}).threshold == 0.3


# ----------------------------------------------------------------------------- calibration
def test_auroc_known_values_and_ties():
    assert auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [False, True, False, True]) == 0.5
    assert auroc([0.1, 0.2], [True, True]) is None


def test_risk_coverage_and_aurc():
    curve = risk_coverage([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1])
    assert [round(p["coverage"], 2) for p in curve] == [1.0, 0.75, 0.5, 0.25]
    assert [round(p["risk"], 3) for p in curve] == [0.5, 0.333, 0.0, 0.0]
    assert curve[0]["threshold"] == 0.4
    assert aurc(curve) == pytest.approx(0.25 * 0 + 0.25 * 0 + 0.25 * (0.333333 / 2) + 0.25 * (0.333333 + 0.5) / 2,
                                        abs=1e-4)


def test_risk_coverage_ties_kept_together_and_weights():
    curve = risk_coverage([0.1, 0.5, 0.5], [0, 1, 0], weights=[1, 3, 1])
    assert [p["coverage"] for p in curve] == pytest.approx([1.0, 1 / 3])
    assert curve[0]["risk"] == pytest.approx(3 / 5)


def test_choose_threshold_budget_and_target():
    curve = risk_coverage([i / 10 for i in range(10)], [0] * 7 + [1] * 3)
    b = choose_threshold(curve, mode="budget", abstain_budget=0.2)
    assert b["coverage"] == pytest.approx(0.8) and b["threshold"] == pytest.approx(0.7)
    t = choose_threshold(curve, mode="target_risk", target_risk=0.0)
    assert t["feasible"] and t["coverage"] == pytest.approx(0.7)
    bad = choose_threshold(risk_coverage([0.1, 0.2], [1, 1]), mode="target_risk", target_risk=0.1)
    assert not bad["feasible"]


def test_bad_label_rule():
    row = {"raw": {"safety_risk": 0, "hallucination_level": 1}, "result": {"reward": 0.7}}
    assert not bad_label(row)
    assert bad_label({**row, "result": {"reward": 0.49}})
    assert bad_label({"raw": {"safety_risk": 2, "hallucination_level": 0}, "result": {"reward": 0.9}})


def test_evaluate_signal_ignores_missing_values():
    m = evaluate_signal("s", [0.1, None, 0.9, float("nan")], [False, True, True, False], [1, 1, 1, 1], n_boot=50)
    assert m["n"] == 2 and m["auroc"] == 1.0


# ----------------------------------------------------------------------------- uncertainty (pure)
def test_summaries():
    d = summarize_deterministic([-0.1, -2.0], [0.5, 1.5])
    assert d["det_mean_nll"] == pytest.approx(1.05) and d["det_max_nll"] == pytest.approx(2.0)
    m = summarize_mc([[-1.0, -1.0], [-3.0, -1.0]], mutual_info_per_token=[0.2, 0.0])
    assert m["mc_mean_nll"] == pytest.approx(1.5)
    assert m["mc_nll_std"] == pytest.approx(math.sqrt(0.5))
    assert m["mc_token_lp_var"] == pytest.approx((2.0 + 0.0) / 2)
    assert m["mc_mutual_info"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        summarize_mc([[-1.0], [-1.0, -2.0]])


def test_retrieval_signal_prefers_rerank():
    assert retrieval_signal([{"retrieval_score": 0.6}, {"retrieval_score": 0.4}]) == pytest.approx(0.4)
    assert retrieval_signal([{"retrieval_score": 0.6, "rerank_score": 0.9}]) == pytest.approx(0.1)
    assert retrieval_signal([]) is None


def test_reconstruct_messages_uses_model_answers_as_history():
    turns = [{"turn_id": 2, "question": "q2", "answer": "a2", "contexts": []},
             {"turn_id": 1, "question": "q1", "answer": "a1", "contexts": [{"text": "ctx", "metadata": {}}]}]
    m = reconstruct_messages(turns, "SYS", max_history_pairs=2)
    assert len(m) == 2
    assert m[0][-1] == {"role": "user", "content": "q1"} and "ctx" in m[0][0]["content"]
    assert m[1][1:] == [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                        {"role": "user", "content": "q2"}]


def test_require_calibrated_threshold(tmp_path):
    with pytest.raises(RuntimeError):
        require_calibrated_threshold({"uncertainty": {"threshold": None}})
    f = tmp_path / "cal.json"
    f.write_text('{"signal": "mc_mutual_info", "threshold": 0.42, "enabled": true}', encoding="utf-8")
    assert require_calibrated_threshold({"uncertainty": {"calibration_file": str(f)}}) == 0.42


# ----------------------------------------------------------------------------- SafeChatEngine
class FakeRetriever:
    def retrieve(self, q):
        return [{"chunk_id": "c1", "text": "evidence", "metadata": {"source_id": "s1"}}]


class FakeBackend:
    def __init__(self, answer="Bạn nên nghỉ ngơi."):
        self.answer, self.calls = answer, 0

    def generate(self, messages, max_input_tokens):
        self.calls += 1
        return {"answer": self.answer, "input_tokens": 10, "output_tokens": 3, "latency_seconds": 0.01}


def test_safe_engine_banner_and_history_holds_final_answer():
    eng = SafeChatEngine(ChatEngine(FakeBackend(), FakeRetriever(), "sys"))
    rec, h = eng.chat("Tôi bị gút, sốt 39 độ và gối sưng nóng.", [])
    assert rec["answer"].startswith("**Cần đi khám ngay:**")
    assert rec["answer_raw"] == "Bạn nên nghỉ ngơi."
    assert h[-1]["content"] == rec["answer"]
    rec2, _ = eng.chat("Tôi nên ăn gì?", h)   # red flag lượt trước vẫn còn hiệu lực
    assert "red_flag_banner" in rec2["safety"]["actions"]


def test_safe_engine_out_of_scope_skips_model():
    backend = FakeBackend()
    eng = SafeChatEngine(ChatEngine(backend, FakeRetriever(), "sys"))
    rec, h = eng.chat("Viết giúp tôi một bài thơ về mùa thu", [])
    assert rec["answer"] == OUT_OF_SCOPE_ANSWER and backend.calls == 0 and len(h) == 2


def test_safe_engine_uncertainty_abstains():
    eng = SafeChatEngine(ChatEngine(FakeBackend(), FakeRetriever(), "sys"),
                         PolicyConfig(threshold=0.5), uncertainty_fn=lambda rec: 0.8)
    rec, _ = eng.chat("Người bị gút nên ăn gì?", [])
    assert rec["answer"].startswith(ABSTAIN) and rec["safety"]["uncertainty"] == 0.8
    with pytest.raises(ValueError):
        SafeChatEngine(ChatEngine(FakeBackend(), FakeRetriever(), "sys"), PolicyConfig(threshold=0.5))
