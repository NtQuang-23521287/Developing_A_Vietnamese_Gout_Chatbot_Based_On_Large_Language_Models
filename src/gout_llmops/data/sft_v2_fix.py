"""SFT v2.1 — chuyển dữ liệu SFT v2 (bản tích hợp ghi chú bác sĩ vòng 1, data/sft_v2) về định dạng pipeline và áp các
chỉnh sửa đã ghi trong data/sft_v2/doctor_round1_fixes.json.

Định dạng pipeline (giống artifacts/sft_v2/filtered.jsonl): {id, source, category, turns, answers, contexts}, với
contexts[i] = các chunk KB (đủ chunk_id, text, metadata) của lượt i. Định dạng này dùng được cho:
  - build bản train/val bằng build_messages — giống hệt lúc suy luận, NHÃN NGUỒN THẬT (bản 16/11 ghi
    "doctor_review_excerpt_i" thay cho tên nguồn);
  - làm câu hỏi cho DPO (dpo_pipeline.py generate --prompts ...).
Bản 16/11 chỉ còn context của lượt cuối trong system prompt, nên contexts của các lượt trước để trống (không lượt nào
trong dữ liệu train hay DPO dùng tới chúng).
Module thuần (không GPU, không gọi API) để unit test.
"""
from __future__ import annotations

import copy
import re
from typing import Any

from gout_llmops.chat.prompt_builder import build_messages

MAX_HISTORY_PAIRS = 2   # giống build_sft_v2.py và protocol suy luận
RAG_MARK = "\n\nNGỮ CẢNH RAG:\n"
NO_CONTEXT = "(Không có context RAG phù hợp.)"
_LABEL = re.compile(r"\[Nguồn \d+: [^\]\n]*\]\n")
ACTIONS = ("replace_answer", "edit_answer", "edit_question", "drop")


def norm_ws(text: str) -> str:
    return " ".join(str(text).split())


def chunk_index(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Văn bản chunk (đã gom khoảng trắng) -> chunk. Trùng văn bản thì giữ chunk đầu tiên."""
    idx: dict[str, dict[str, Any]] = {}
    for c in chunks:
        idx.setdefault(norm_ws(c["text"]), c)
    return idx


def split_system(system_text: str, system_prompt: str) -> list[str]:
    """Tách các đoạn tài liệu khỏi system message; kiểm tra phần đầu đúng là system prompt của protocol."""
    head, sep, tail = system_text.partition(RAG_MARK)
    if not sep:
        raise ValueError("system message không có phần NGỮ CẢNH RAG")
    if head != system_prompt.rstrip():
        raise ValueError("system message không bắt đầu bằng system prompt của protocol")
    if tail.strip() == NO_CONTEXT:
        return []
    parts = _LABEL.split(tail)
    if parts[0].strip():
        raise ValueError(f"định dạng NGỮ CẢNH RAG lạ: {parts[0][:80]!r}")
    return [p.strip() for p in parts[1:]]


def sample_to_row(sample: dict[str, Any], system_prompt: str, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    msgs = sample["messages"]
    conv = msgs[1:]
    if msgs[0]["role"] != "system" or [m["role"] for m in conv] != ["user", "assistant"] * (len(conv) // 2) \
            or not conv:
        raise ValueError(f"{sample['id']}: messages phải là system rồi các cặp user/assistant")
    ctx = []
    for text in split_system(msgs[0]["content"], system_prompt):
        chunk = index.get(norm_ws(text))
        if chunk is None:
            raise ValueError(f"{sample['id']}: đoạn tài liệu không khớp chunk KB nào: {text[:80]!r}")
        ctx.append({k: chunk[k] for k in ("chunk_id", "text", "metadata") if k in chunk})
    turns = [m["content"] for m in conv[0::2]]
    answers = [m["content"] for m in conv[1::2]]
    return {"id": sample["id"], "source": sample.get("source"), "category": sample["category"],
            "turns": turns, "answers": answers, "contexts": [[] for _ in turns[:-1]] + [ctx]}


def row_to_sample(row: dict[str, Any], system_prompt: str) -> dict[str, Any]:
    """Giống _messages_for + to_sample của build_sft_v2.py: system + NGỮ CẢNH RAG lượt cuối + lịch sử + câu hỏi."""
    history: list[dict[str, str]] = []
    for u, a in zip(row["turns"][:-1], row["answers"][:-1], strict=True):
        history += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs = build_messages(system_prompt, row["turns"][-1], history, row["contexts"][-1], MAX_HISTORY_PAIRS)
    return {"id": row["id"], "source": row.get("source"), "category": row["category"],
            "messages": msgs + [{"role": "assistant", "content": row["answers"][-1]}]}


def same_except_source_labels(a: dict[str, Any], b: dict[str, Any], system_prompt: str) -> bool:
    """Hai mẫu giống nhau, trừ nhãn [Nguồn i: ...] và khoảng trắng trong đoạn tài liệu."""
    if a["messages"][1:] != b["messages"][1:]:
        return False
    ea = split_system(a["messages"][0]["content"], system_prompt)
    eb = split_system(b["messages"][0]["content"], system_prompt)
    return [norm_ws(t) for t in ea] == [norm_ws(t) for t in eb]


def apply_fixes(rows: list[dict[str, Any]], fixes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict]]:
    """Áp chỉnh sửa theo mã. edit_* thay ĐÚNG MỘT lần chuỗi `old` (không có hoặc trùng nhiều lần -> lỗi)."""
    out = [copy.deepcopy(r) for r in rows]
    by_id = {r["id"]: r for r in out}
    missing = [f["id"] for f in fixes if f["id"] not in by_id]
    if missing:
        raise ValueError(f"chỉnh sửa cho mã không có trong dữ liệu: {missing}")
    dropped, log = set(), []
    for f in fixes:
        r, act = by_id[f["id"]], f["action"]
        if act not in ACTIONS:
            raise ValueError(f"{f['id']}: action không hợp lệ {act!r}")
        if act == "drop":
            dropped.add(f["id"])
        elif act == "replace_answer":
            r["answers"][-1] = f["new"]
        else:
            key = "answers" if act == "edit_answer" else "turns"
            i = int(f.get("turn", -1))
            n = r[key][i].count(f["old"])
            if n != 1:
                raise ValueError(f"{f['id']}: chuỗi cần sửa xuất hiện {n} lần (phải đúng 1): {f['old'][:60]!r}")
            r[key][i] = r[key][i].replace(f["old"], f["new"])
        log.append({"id": f["id"], "action": act, "basis": f.get("basis"), "reason": f.get("reason")})
    return [r for r in out if r["id"] not in dropped], log
