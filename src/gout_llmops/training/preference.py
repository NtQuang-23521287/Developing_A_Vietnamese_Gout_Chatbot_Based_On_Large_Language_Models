"""Giai đoạn 4 — dựng prompt DPO và Preference Mining Rule.

Module thuần (không GPU, không gọi API) để unit test và trích dẫn trong khóa luận.

PREFERENCE MINING RULE (phiên bản preference_rule_v1)
Với mỗi câu hỏi x có K ứng viên y_1..y_K do policy sinh ra, mỗi ứng viên có R(x,c,y) theo công thức (1)–(3):

  1. Ứng viên đủ điều kiện làm CHOSEN khi đồng thời:
       - đã chấm thành công, không bị cắt cụt (sinh tới max_new_tokens mà chưa kết thúc);
       - không vi phạm cổng an toàn (F(y) = 0);
       - R ≥ min_chosen_reward (mặc định 0.5 = mức "đạt" của đề cương);
       - cả ba điểm rủi ro gốc ≤ max_chosen_risk (mặc định 1: hallucination, omission, safety đều ≤ 1).
     CHOSEN = ứng viên đủ điều kiện có R cao nhất; nếu bằng R thì chọn câu NGẮN hơn (hạn chế DPO học thói dài dòng).
  2. REJECTED = ứng viên còn lại có R thấp nhất; nếu bằng R thì chọn câu có tổng rủi ro cao hơn.
  3. Giữ cặp khi R(chosen) − R(rejected) ≥ min_margin (mặc định 0.15); chênh nhỏ hơn coi là nhiễu của judge.
  4. Dự phòng (teacher fallback): nếu KHÔNG có ứng viên nào đủ điều kiện làm chosen và câu hỏi có
     risk_level ∈ {high, critical}, dùng câu trả lời mẫu (đã qua bộ lọc của dữ liệu SFT v2) làm chosen và
     ứng viên tệ nhất làm rejected. Đây đúng là vai trò "câu trả lời được sửa lại" của vòng lặp Giai đoạn 5,
     chỉ khác là người sửa ở đây là dữ liệu mẫu đã lọc tự động.
"""
from __future__ import annotations

import random
from typing import Any

from gout_llmops.chat.prompt_builder import build_messages

RULE_VERSION = "preference_rule_v1"

# category của dữ liệu SFT v2 -> (question_type, risk_level, expected_behavior) theo quy ước testset v03.
# Judge v2 và cổng an toàn (công thức 3) cần ba nhãn này. Ánh xạ bám phân bố của testset v03:
# red_flag đều critical, safety_trap phần lớn high/critical, clinical_reasoning phần lớn moderate.
CATEGORY_LABELS: dict[str, tuple[str, str, str]] = {
    "basic_knowledge": ("basic_knowledge", "low", "answer"),
    "clinical_reasoning": ("clinical_reasoning", "moderate", "answer_with_referral"),
    "red_flag": ("red_flag", "critical", "urgent_referral"),
    "safety_trap": ("safety_trap", "high", "refuse_and_refer"),
    "out_of_scope": ("basic_knowledge", "low", "refuse_and_refer"),
}

DEFAULT_RULE: dict[str, Any] = {
    "min_chosen_reward": 0.5,
    "max_chosen_risk": 1,
    "min_margin": 0.15,
    "teacher_fallback": True,
    "teacher_fallback_risk_levels": ["high", "critical"],
}

RISK_KEYS = ("hallucination_level", "omission_risk", "safety_risk")
SAFETY_FIRST = ("red_flag", "safety_trap")


def labels_for(category: str) -> tuple[str, str, str]:
    return CATEGORY_LABELS.get(category, CATEGORY_LABELS["clinical_reasoning"])


def prompt_messages(system_prompt: str, turns: list[str], answers: list[str],
                    final_contexts: list[dict[str, Any]], max_history_pairs: int = 2) -> list[dict[str, str]]:
    """Đúng định dạng lúc suy luận: system + NGỮ CẢNH RAG của lượt cuối + lịch sử (câu trả lời mẫu) + câu hỏi cuối."""
    i = len(turns) - 1
    history: list[dict[str, str]] = []
    for u, a in zip(turns[:i], answers[:i]):
        history += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return build_messages(system_prompt, turns[i], history, final_contexts, max_history_pairs)


def select_prompts(rows: list[dict[str, Any]], max_prompts: int | None, seed: int = 2026) -> list[dict[str, Any]]:
    """Ưu tiên giữ hết red_flag, safety_trap và hội thoại nhiều lượt; phần còn lại lấy ngẫu nhiên (cố định seed)."""
    rows = sorted(rows, key=lambda r: r["id"])
    if max_prompts is None or len(rows) <= max_prompts:
        return rows
    rng = random.Random(seed)
    priority = [r for r in rows if r.get("category") in SAFETY_FIRST or len(r["turns"]) > 1]
    priority_ids = {r["id"] for r in priority}
    rest = [r for r in rows if r["id"] not in priority_ids]
    rng.shuffle(priority)
    rng.shuffle(rest)
    picked = (priority + rest)[:max_prompts]
    return sorted(picked, key=lambda r: r["id"])


def _risk_sum(c: dict[str, Any]) -> int:
    raw = c.get("raw") or {}
    return sum(int(raw.get(k, 0)) for k in RISK_KEYS)


def _eligible_chosen(c: dict[str, Any], rule: dict[str, Any]) -> bool:
    raw = c.get("raw") or {}
    return (c.get("status") == "ok"
            and not c.get("truncated", False)
            and not c.get("fatal", False)
            and float(c["reward"]) >= rule["min_chosen_reward"]
            and all(int(raw.get(k, 3)) <= rule["max_chosen_risk"] for k in RISK_KEYS))


def mine_pair(row: dict[str, Any], rule: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str]:
    """Trả (cặp, lý do). Cặp = None khi bị loại; lý do dùng để thống kê trong manifest."""
    rule = {**DEFAULT_RULE, **(rule or {})}
    scored = [c for c in row.get("candidates", []) if c.get("status") == "ok"]
    if not scored:
        return None, "no_scored_candidate"

    eligible = [c for c in scored if _eligible_chosen(c, rule)]
    if eligible:
        chosen = max(eligible, key=lambda c: (float(c["reward"]), -len(c["text"])))
        others = [c for c in scored if c is not chosen]
        if not others:
            return None, "single_candidate"
        rejected = min(others, key=lambda c: (float(c["reward"]), -_risk_sum(c)))
        margin = float(chosen["reward"]) - float(rejected["reward"])
        if margin < rule["min_margin"]:
            return None, "small_margin"
        return {
            "source": "on_policy",
            "chosen": chosen["text"], "rejected": rejected["text"],
            "chosen_idx": chosen["idx"], "rejected_idx": rejected["idx"],
            "chosen_reward": float(chosen["reward"]), "rejected_reward": float(rejected["reward"]),
            "margin": margin,
        }, "ok"

    teacher = (row.get("teacher_answers") or [None])[-1]
    if (rule["teacher_fallback"] and teacher
            and row.get("risk_level") in rule["teacher_fallback_risk_levels"]):
        rejected = min(scored, key=lambda c: (float(c["reward"]), -_risk_sum(c)))
        return {
            "source": "teacher_fallback",
            "chosen": teacher, "rejected": rejected["text"],
            "chosen_idx": None, "rejected_idx": rejected["idx"],
            "chosen_reward": None, "rejected_reward": float(rejected["reward"]),
            "margin": None,
        }, "ok_teacher_fallback"
    return None, "no_eligible_chosen"
