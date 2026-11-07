import pytest

from gout_llmops.data.doctor_review import (COLUMNS, SHEET_REVIEW, apply_review, order_rows, parse_verdict,
                                            read_review_xlsx, select_for_review, write_review_xlsx)


def _row(rid, cat="basic_knowledge", turns=("Gút là gì?",), answers=("Câu trả lời AI.",)):
    return {"id": rid, "category": cat, "source": "kb", "turns": list(turns), "answers": list(answers),
            "contexts": [[{"chunk_id": "c1", "text": "Tài liệu về gút."}] for _ in turns]}


def _rec(rid, verdict=None, corrected=None, answer=None, row=2, **kw):
    rec = {"id": rid, "verdict": verdict, "corrected": corrected, "_row": row, **kw}
    if answer is not None:
        rec["answer"] = answer
    return rec


def test_parse_verdict_accepts_variants():
    assert parse_verdict("Đạt") == "approve"
    assert parse_verdict(" dat ") == "approve"
    assert parse_verdict("CẦN SỬA") == "edit"
    assert parse_verdict("can sua") == "edit"
    assert parse_verdict("Loại") == "reject"
    assert parse_verdict(None) is None
    assert parse_verdict("  ") is None
    with pytest.raises(ValueError):
        parse_verdict("ok")


def test_apply_review_approve_edit_reject_and_drop_unreviewed():
    rows = [_row("a"), _row("b"), _row("c"), _row("d")]
    recs = [_rec("a", "Đạt", row=2), _rec("b", "Cần sửa", "Bản bác sĩ sửa.", row=3),
            _rec("c", "Loại", row=4, error_type="Có thể gây hại"), _rec("d", None, row=5)]
    res = apply_review(rows, recs)
    assert res["errors"] == []
    kept = {r["id"]: r for r in res["kept"]}
    assert set(kept) == {"a", "b"}
    assert kept["a"]["answers"] == ["Câu trả lời AI."]
    assert kept["b"]["answers"] == ["Bản bác sĩ sửa."]
    assert kept["b"]["answer_ai_original"] == "Câu trả lời AI."
    assert [r["id"] for r in res["rejected"]] == ["c"]
    st = res["stats"]
    assert (st["n_approve"], st["n_edit"], st["n_reject"], st["n_unreviewed"], st["n_kept"]) == (1, 1, 1, 1, 2)
    assert st["error_types"] == {"Có thể gây hại": 1}


def test_unreviewed_keep_policy():
    res = apply_review([_row("a"), _row("b")], [_rec("a", "Đạt")], unreviewed="keep")
    kept = {r["id"]: r for r in res["kept"]}
    assert set(kept) == {"a", "b"}
    assert kept["b"]["review"]["verdict"] == "unreviewed"


def test_only_last_turn_is_replaced_in_dialog():
    row = _row("d1", turns=("Q1", "Q2"), answers=("A1", "A2"))
    res = apply_review([row], [_rec("d1", "Cần sửa", "A2 đã sửa")])
    assert res["kept"][0]["answers"] == ["A1", "A2 đã sửa"]


def test_in_place_edit_is_used_as_correction():
    res = apply_review([_row("a")], [_rec("a", "Cần sửa", answer="Bác sĩ sửa thẳng vào cột AI.")])
    assert res["errors"] == []
    assert res["kept"][0]["answers"] == ["Bác sĩ sửa thẳng vào cột AI."]
    assert res["kept"][0]["review"]["edited_in_place"] is True


def test_whitespace_only_change_is_not_an_edit():
    res = apply_review([_row("a")], [_rec("a", "Đạt", answer="  Câu trả lời\n AI.  ")])
    assert res["errors"] == []
    assert res["kept"][0]["answers"] == ["Câu trả lời AI."]


@pytest.mark.parametrize("rec, fragment", [
    (_rec("a", "Đạt", "có sửa"), "chọn 'Đạt'"),
    (_rec("a", "Cần sửa", None), "chưa có câu trả lời đã sửa"),
    (_rec("a", None, "có sửa"), "chưa chọn ĐÁNH GIÁ"),
    (_rec("a", "ok"), "không hợp lệ"),
    (_rec("zzz", "Đạt"), "không có trong dữ liệu gốc"),
])
def test_contradictions_are_errors_not_guesses(rec, fragment):
    res = apply_review([_row("a")], [rec])
    assert any(fragment in e for e in res["errors"]), res["errors"]
    assert res["kept"] == [] and res["rejected"] == []


def test_duplicate_id_is_error():
    res = apply_review([_row("a")], [_rec("a", "Đạt", row=2), _rec("a", "Loại", row=3)])
    assert any("bị lặp" in e for e in res["errors"])


def test_order_rows_puts_high_risk_first():
    rows = [_row("1", "basic_knowledge"), _row("2", "red_flag"), _row("3", "safety_trap")]
    assert [r["category"] for r in order_rows(rows)] == ["red_flag", "safety_trap", "basic_knowledge"]


def test_select_for_review_stratified_and_full_categories():
    rows = ([_row(f"b{i}", "basic_knowledge") for i in range(30)]
            + [_row(f"r{i}", "red_flag") for i in range(12)]
            + [_row(f"s{i}", "safety_trap") for i in range(3)])
    sel = select_for_review(rows, per_category=5, full_categories=("red_flag",), seed=1)
    cats = [r["category"] for r in sel]
    assert cats.count("basic_knowledge") == 5
    assert cats.count("red_flag") == 12          # lấy hết
    assert cats.count("safety_trap") == 3        # ít hơn N thì lấy hết
    assert [r["id"] for r in sel] == [r["id"] for r in select_for_review(rows, per_category=5,
                                                                         full_categories=("red_flag",), seed=1)]
    assert select_for_review(rows) == rows


def test_not_sent_rows_are_unreviewed_without_missing_warning():
    rows = [_row("a"), _row("b"), _row("c")]
    res = apply_review(rows, [_rec("a", "Đạt")], unreviewed="keep", exported_ids=["a"])
    assert res["warnings"] == []
    assert res["stats"]["n_sent_to_doctor"] == 1
    assert res["stats"]["n_unreviewed"] == 2
    assert res["stats"]["pass_rate"] == 1.0
    assert {r["id"] for r in res["kept"]} == {"a", "b", "c"}
    # gửi a, b nhưng b bị xóa khỏi file -> cảnh báo
    res = apply_review(rows, [_rec("a", "Đạt")], exported_ids=["a", "b"])
    assert any("không có trong file duyệt" in w for w in res["warnings"])


def test_xlsx_roundtrip(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    rows = [_row("a", "basic_knowledge"), _row("b", "red_flag", turns=("Q1", "Q2"), answers=("A1", "=A2"))]
    path = tmp_path / "review.xlsx"
    assert write_review_xlsx(rows, path, source_file="filtered.jsonl", source_sha256="abc") == 2

    wb = openpyxl.load_workbook(path)
    ws = wb[SHEET_REVIEW]
    col = {label: i + 1 for i, (_, label, _) in enumerate(COLUMNS)}
    assert ws.cell(2, col["Mã"]).value == "b"          # red_flag lên đầu
    assert ws.cell(2, col["Câu trả lời của AI"]).value == "=A2"   # chuỗi, không thành công thức
    ws.cell(2, col["ĐÁNH GIÁ"]).value = "Cần sửa"
    ws.cell(2, col["Câu trả lời đã sửa"]).value = "A2 bác sĩ sửa"
    ws.cell(3, col["ĐÁNH GIÁ"]).value = "Đạt"
    wb.save(path)

    records, meta = read_review_xlsx(path)
    assert meta["source_sha256"] == "abc"
    assert sorted(meta["exported_ids"]) == ["a", "b"]
    res = apply_review(rows, records, exported_ids=meta["exported_ids"])
    assert res["errors"] == []
    kept = {r["id"]: r for r in res["kept"]}
    assert kept["b"]["answers"] == ["A1", "A2 bác sĩ sửa"]
    assert kept["a"]["answers"] == ["Câu trả lời AI."]
