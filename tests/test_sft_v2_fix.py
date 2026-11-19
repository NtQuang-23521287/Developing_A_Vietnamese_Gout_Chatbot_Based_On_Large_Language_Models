"""SFT v2.1: luật rút từ góp ý bác sĩ vòng 1 + chuyển dữ liệu 16/11 về định dạng pipeline và áp chỉnh sửa."""
import json
from pathlib import Path

import pytest

from gout_llmops.chat.prompt_builder import format_contexts
from gout_llmops.data.doctor_rules import lint_conversation
from gout_llmops.data.sft_v2_fix import (
    apply_fixes,
    chunk_index,
    row_to_sample,
    same_except_source_labels,
    sample_to_row,
    split_system,
)

ROOT = Path(__file__).resolve().parents[1]
SYS = "Bạn là trợ lý.\n\nQuy tắc: an toàn."
CHUNKS = [{"chunk_id": "c1", "text": "Colchicin dùng cắt cơn gút.", "metadata": {"source_id": "guideline_A"}},
          {"chunk_id": "c2", "text": "Allopurinol là thuốc hạ acid uric.", "metadata": {"source_id": "ACR"}}]


def _sample(label="doctor_review_excerpt", answer="Trả lời.", question="Hỏi?"):
    ctx = "\n\n".join(f"[Nguồn {i}: {label}_{i}]\n{c['text']}" for i, c in enumerate(CHUNKS, 1))
    return {"id": "x1", "source": "doctor_review_excel_151", "category": "red_flag",
            "messages": [{"role": "system", "content": SYS + "\n\nNGỮ CẢNH RAG:\n" + ctx},
                         {"role": "user", "content": "Lượt 1"}, {"role": "assistant", "content": "Đáp 1"},
                         {"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


# ----------------------------------------------------------------------------- luật bác sĩ
def test_lint_flags_dont_stop_during_drug_reaction():
    v = lint_conversation(["Đang dùng colchicin bị tiêu chảy nặng, có cần ngưng thuốc không?"],
                          ["Bạn không nên tự ý ngưng thuốc, hãy hỏi bác sĩ."])
    assert [x["rule"] for x in v] == ["dont_stop_in_drug_reaction"]
    ok = lint_conversation(["Đang dùng colchicin bị tiêu chảy nặng, có cần ngưng thuốc không?"],
                           ["Bạn nên ngưng colchicin và đi khám ngay."])
    assert ok == []


def test_lint_dont_stop_is_fine_without_reaction():
    assert lint_conversation(["Tôi muốn tự ngưng allopurinol khi hết đau?"], ["Không nên tự ý ngưng thuốc."]) == []


@pytest.mark.parametrize("answer,rule", [
    ("Một số thuốc hạ acid uric như colchicine cần giảm liều.", "colchicine_called_ult"),
    ("Bạn nên chườm nóng lên khớp để đỡ đau.", "hot_compress"),
    ("Hãy giữ ấm vùng khớp bị đau.", "hot_compress"),
    ("Xem thêm (https://example.com/gout).", "reviewer_note_leak"),
])
def test_lint_other_doctor_rules(answer, rule):
    assert rule in [x["rule"] for x in lint_conversation(["Câu hỏi về gút?"], [answer])]


@pytest.mark.parametrize("answer", [
    "Colchicine không phải thuốc hạ acid uric; đây là thuốc chống viêm.",
    "Nên chườm lạnh, tránh chườm nóng lên khớp đang sưng viêm.",
    "Không nên chườm nóng.",
])
def test_lint_respects_negation(answer):
    assert lint_conversation(["Câu hỏi về gút?"], [answer]) == []


def test_lint_flags_note_in_question():
    v = lint_conversation(["Uống Colchicin và Allopurinol (SAI TÊN THUỐC ALLOPURINOL) được không?"], ["Được."])
    assert v and v[0]["rule"] == "reviewer_note_leak" and v[0]["where"] == "turn[0]"


# ----------------------------------------------------------------------------- chuyển định dạng
def test_split_system_checks_prompt_and_extracts_excerpts():
    assert split_system(_sample()["messages"][0]["content"], SYS) == [c["text"] for c in CHUNKS]
    with pytest.raises(ValueError):
        split_system(_sample()["messages"][0]["content"], "Prompt khác")


def test_roundtrip_restores_real_source_labels_only():
    s = _sample()
    row = sample_to_row(s, SYS, chunk_index(CHUNKS))
    assert row["turns"] == ["Lượt 1", "Hỏi?"] and row["answers"] == ["Đáp 1", "Trả lời."]
    assert row["contexts"][0] == [] and [c["chunk_id"] for c in row["contexts"][-1]] == ["c1", "c2"]
    rebuilt = row_to_sample(row, SYS)
    assert same_except_source_labels(s, rebuilt, SYS)
    assert format_contexts(CHUNKS) in rebuilt["messages"][0]["content"]       # nhãn nguồn thật
    assert "doctor_review_excerpt" not in rebuilt["messages"][0]["content"]
    assert not same_except_source_labels(_sample(answer="Khác."), rebuilt, SYS)


def test_unknown_excerpt_is_an_error():
    s = _sample()
    s["messages"][0]["content"] = SYS + "\n\nNGỮ CẢNH RAG:\n[Nguồn 1: x]\nĐoạn không có trong KB."
    with pytest.raises(ValueError):
        sample_to_row(s, SYS, chunk_index(CHUNKS))


# ----------------------------------------------------------------------------- chỉnh sửa
def _rows():
    idx = chunk_index(CHUNKS)
    a = sample_to_row(_sample(answer="Không nên tự ý ngưng thuốc."), SYS, idx)
    b = {**sample_to_row(_sample(question="Hỏi (SAI TÊN THUỐC)?"), SYS, idx), "id": "x2"}
    return [a, b]


def test_apply_fixes_replace_edit_drop_without_mutating_input():
    rows = _rows()
    out, log = apply_fixes(rows, [
        {"id": "x1", "action": "edit_answer", "old": "thuốc.", "new": "thuốc. Riêng khi dị ứng thì phải ngưng."},
        {"id": "x2", "action": "edit_question", "old": " (SAI TÊN THUỐC)", "new": ""},
    ])
    assert out[0]["answers"][-1].endswith("thì phải ngưng.") and out[1]["turns"][-1] == "Hỏi?"
    assert rows[0]["answers"][-1] == "Không nên tự ý ngưng thuốc."          # dữ liệu gốc không bị sửa
    out, _ = apply_fixes(rows, [{"id": "x2", "action": "drop"},
                                {"id": "x1", "action": "replace_answer", "new": "Mới."}])
    assert [r["id"] for r in out] == ["x1"] and out[0]["answers"] == ["Đáp 1", "Mới."]
    assert [e["action"] for e in log] == ["edit_answer", "edit_question"]


@pytest.mark.parametrize("fix", [
    {"id": "x1", "action": "edit_answer", "old": "không có trong câu", "new": "x"},
    {"id": "nope", "action": "drop"},
    {"id": "x1", "action": "rewrite_everything"},
])
def test_apply_fixes_rejects_bad_fixes(fix):
    with pytest.raises(ValueError):
        apply_fixes(_rows(), [fix])


def test_committed_sft_v2_1_data_passes_doctor_rules():
    """Dữ liệu SFT v2.1 trong repo: không còn mẫu nào vi phạm luật bác sĩ."""
    for f in ("train", "val"):
        for line in (ROOT / f"data/sft_v2_1/{f}.jsonl").read_text(encoding="utf-8").splitlines():
            s = json.loads(line)
            turns = [m["content"] for m in s["messages"] if m["role"] == "user"]
            answers = [m["content"] for m in s["messages"] if m["role"] == "assistant"]
            assert lint_conversation(turns, answers) == [], s["id"]
