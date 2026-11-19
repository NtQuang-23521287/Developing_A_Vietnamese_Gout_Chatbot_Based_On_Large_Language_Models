"""Luật kiểm tra dữ liệu rút ra từ góp ý của bác sĩ ở đợt duyệt SFT v2 vòng 1 (11/2026).

Bác sĩ không điền cột ĐÁNH GIÁ mà ghi chú trực tiếp vào 19/151 dòng. Các lỗi bác sĩ chỉ ra được chuyển thành luật để
áp cho MỌI mẫu (cả mẫu bác sĩ chưa đánh dấu và mẫu chưa gửi bác sĩ), vì cùng một lỗi có thể lặp ở dòng khác:

  dont_stop_in_drug_reaction  đang có phản ứng thuốc (dị ứng/quá mẫn, ngộ độc, tác dụng phụ nặng) mà câu trả lời
                              khuyên "không tự ý ngưng thuốc" và không hề khuyên ngưng.
                              Bác sĩ: PHẢI ngưng thuốc và đi khám ngay.
  colchicine_called_ult       gọi colchicine là thuốc hạ acid uric. Bác sĩ: colchicine KHÔNG phải thuốc hạ acid uric.
  hot_compress                khuyên chườm nóng / giữ ấm khớp đang viêm. Bác sĩ: chườm lạnh, tuyệt đối không chườm nóng.
  reviewer_note_leak          ghi chú của người duyệt lọt vào hội thoại ("???", "==>", "SAI TÊN", URL...).

Luật đầu tiên dùng chính guard + chính sách của lớp an toàn (safety/guard.py, safety/policy.py), nên dữ liệu train và
lớp an toàn lúc suy luận áp cùng một quy tắc.
"""
from __future__ import annotations

import re
from typing import Any

from gout_llmops.safety.guard import lower_accented, run_guard
from gout_llmops.safety.policy import stop_advice

RULESET_VERSION = "doctor_round1_rules_v1"

_COLCHICINE_ULT = re.compile(
    r"(hạ|giảm) (acid|axit) uric (như|gồm|là)[^.;\n]{0,40}colchicin"
    r"|colchicin\w*[^.;\n]{0,25}(là|như) (một )?(loại )?thuốc (hạ|giảm) (acid|axit) uric")
_HOT_COMPRESS = re.compile(r"(chườm|đắp) (nóng|ấm)|giữ ấm[^.;\n]{0,20}khớp")
_NEGATED = ("không", "tránh", "đừng", "chớ")
_NOTE_LEAK = re.compile(r"\?\?\?|==>|\bSAI TÊN\b|[Cc]âu hỏi không rõ ràng|https?://")


def _negated(text: str, start: int) -> bool:
    clause = re.split(r"[.!?;:\n,()]", text[:start])[-1]
    return any(w in clause.split() for w in _NEGATED) or "không phải" in clause


def lint_conversation(turns: list[str], answers: list[str]) -> list[dict[str, Any]]:
    """Vi phạm luật bác sĩ trong một hội thoại (turns[i] = lời người hỏi, answers[i] = câu trả lời lượt i)."""
    out: list[dict[str, Any]] = []
    guard = run_guard(turns[-1], turns[:-1])
    says_stop, says_dont_stop = stop_advice(answers[-1])
    if guard.drug_reaction and says_dont_stop and not says_stop:
        out.append({"rule": "dont_stop_in_drug_reaction", "where": f"answer[{len(answers) - 1}]",
                    "excerpt": answers[-1][:160]})
    for i, a in enumerate(answers):
        low = lower_accented(a)
        for m in _COLCHICINE_ULT.finditer(low):
            if "không phải" not in low[max(0, m.start() - 20): m.end()]:
                out.append({"rule": "colchicine_called_ult", "where": f"answer[{i}]", "excerpt": m.group(0)})
        for m in _HOT_COMPRESS.finditer(low):
            if not _negated(low, m.start()):
                out.append({"rule": "hot_compress", "where": f"answer[{i}]", "excerpt": m.group(0)})
    for role, texts in (("turn", turns), ("answer", answers)):
        for i, t in enumerate(texts):
            m = _NOTE_LEAK.search(t)
            if m:
                out.append({"rule": "reviewer_note_leak", "where": f"{role}[{i}]",
                            "excerpt": t[max(0, m.start() - 40): m.end() + 40]})
    return out
