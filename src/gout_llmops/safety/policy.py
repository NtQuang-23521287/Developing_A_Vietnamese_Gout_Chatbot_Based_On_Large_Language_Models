"""Giai đoạn 5 — Chính sách Safe Inference: biến (guard, độ bất định) thành câu trả lời cuối.

Lớp an toàn đứng SAU model, không đổi model hay prompt -> đo riêng được tác dụng của nó:
  1. out_of_scope                 -> thay câu trả lời bằng mẫu từ chối (ngoài phạm vi).
  2. độ bất định > ngưỡng τ        -> "abstain": thay bằng mẫu từ chối an toàn + tài liệu liên quan;
                                     hoặc "hedge": giữ câu trả lời, thêm cảnh báo độ chắc chắn thấp.
  3. red_flag                     -> chèn banner khám/cấp cứu lên đầu, trừ khi câu trả lời đã khuyên rõ.
  4. self_medication              -> chèn lưu ý không tự ý dùng/đổi thuốc, trừ khi câu trả lời đã nói rõ.

Mọi mẫu câu ở đây là quy ước của nhóm, CẦN bác sĩ duyệt trước khi triển khai thật.
Module thuần (không GPU) để unit test và dùng chung cho chế độ offline (scripts/safe_inference.py apply)
và online (chat/safe_engine.py) — hai chế độ cho ra cùng một câu trả lời.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gout_llmops.safety.guard import (
    EMERGENCY,
    OUT_OF_SCOPE,
    RED_FLAG,
    RULE_BY_ID,
    SELF_MEDICATION,
    GuardResult,
    normalize,
)

POLICY_VERSION = "safe_policy_v1"

RED_FLAG_REASONS: dict[str, str] = {
    "fever": "sốt (nhất là khi kèm khớp sưng nóng đỏ) có thể là dấu hiệu nhiễm trùng khớp hoặc nhiễm trùng toàn thân",
    "chest_pain_dyspnea": "đau ngực, khó thở hoặc ngất có thể là bệnh lý tim mạch, hô hấp cấp",
    "gi_bleeding": "phân đen, nôn hoặc đi ngoài ra máu có thể là xuất huyết tiêu hóa",
    "renal_failure_signs": ("tiểu rất ít, tiểu ra máu hoặc đau quặn thắt lưng có thể là biến chứng ở thận, "
                            "đường tiết niệu"),
    "colchicine_toxicity": ("nôn, tiêu chảy nhiều hoặc yếu cơ có thể là tác dụng phụ nặng của thuốc "
                            "(ví dụ ngộ độc colchicine)"),
    "severe_drug_reaction": "nổi ban, phồng rộp, bong da, loét miệng hoặc sưng môi, mặt có thể là dị ứng thuốc nặng",
    "neuro_emergency": "lú lẫn, co giật, yếu liệt hoặc méo miệng là dấu hiệu thần kinh cấp",
    "severe_abdominal_pain": "đau bụng nhiều có thể là tác dụng phụ nặng của thuốc hoặc bệnh tiêu hóa cần được khám",
    "child_joint": ("trẻ em bị sưng đau khớp hiếm khi do gút, cần được bác sĩ khám để tìm nguyên nhân khác "
                    "như nhiễm trùng hay chấn thương"),
    "tophus_infection": "hạt tophi hoặc khớp bị vỡ, loét, chảy dịch hay có mủ có nguy cơ nhiễm trùng",
    "colchicine_gi_effect": ("tiêu chảy, đau bụng hoặc nôn khi dùng colchicine cần báo bác sĩ để giảm liều "
                             "hoặc ngừng thuốc kịp thời"),
    "home_management_question": "các triệu chứng bạn mô tả không nên chỉ tự theo dõi tại nhà",
}

URGENT_EMERGENCY = ("**Cần đi khám ngay:** {reasons}. Bạn nên đến cơ sở y tế hoặc khoa cấp cứu gần nhất ngay "
                    "(gọi 115 nếu triệu chứng nặng), không tự điều trị tại nhà và không chờ xem có tự đỡ không.")
URGENT_PROMPT = ("**Nên đi khám sớm:** {reasons}. Bạn nên được bác sĩ khám trực tiếp sớm (trong ngày nếu có thể), "
                 "không tự điều trị tại nhà.")
SELF_MED_NOTE = ("**Lưu ý an toàn:** không tự ý bắt đầu, tăng, giảm hay ngừng thuốc, và không dùng thuốc được kê "
                 "cho người khác. Chọn thuốc và liều phải do bác sĩ điều trị quyết định, dựa trên bệnh kèm theo "
                 "(thận, dạ dày, tim mạch, thai kỳ...) và các thuốc bạn đang dùng.")
ABSTAIN = ("Mình chưa đủ chắc chắn để trả lời chính xác câu hỏi này dựa trên tài liệu hiện có, nên không muốn đưa "
           "ra thông tin có thể sai. Bạn nên hỏi trực tiếp bác sĩ chuyên khoa Cơ xương khớp hoặc bác sĩ đang điều "
           "trị cho bạn. Nếu có sốt, khớp sưng nóng đỏ dữ dội, đau ngực, khó thở hoặc tiểu rất ít, hãy đến cơ sở y tế "
           "ngay.")
ABSTAIN_SOURCES = "Tài liệu tham khảo liên quan: {sources}."
# Tên hiển thị cho nguồn KB đã xác định được. Nguồn chưa xác minh nhà phát hành (gout_guideline_1/2.pdf) KHÔNG đặt tên
# (tránh trích dẫn sai) -> không hiển thị. Metadata "title" của chunk (nếu có) được ưu tiên.
SOURCE_TITLES = {
    "ACR2020_Gout_Guideline_VN": "Khuyến cáo điều trị bệnh gút của Hội Thấp khớp học Hoa Kỳ (ACR) năm 2020",
}
HEDGE = ("**Lưu ý:** mức độ chắc chắn của câu trả lời dưới đây thấp; hãy xác nhận lại với bác sĩ trước khi áp dụng.")
OUT_OF_SCOPE_ANSWER = ("Mình là trợ lý cung cấp thông tin về bệnh gút nên chưa thể hỗ trợ câu hỏi này. Nếu bạn có "
                       "thắc mắc về bệnh gút, acid uric, chế độ ăn uống hay thuốc điều trị gút, mình sẵn sàng giúp.")

# Câu trả lời đã khuyên khám/cấp cứu rõ ràng -> không chèn banner nữa (tránh lặp).
_URGENT_CUE = (r"cap cuu|\b115\b|di kham ngay|kham ngay|den (ngay )?(benh vien|co so y te|vien)( \w+)? ngay|"
               r"ngay lap tuc|khan cap|nhap vien")
# Câu trả lời đã nói rõ KHÔNG tự ý dùng/đổi thuốc -> không chèn lưu ý.
_SELF_MED_CUE = (r"khong (nen |duoc )?tu (y )?(tang|giam|ngung|dung|mua|bo|doi|uong|dieu chinh|su dung|ke)|"
                 r"khong (nen |duoc )?dung thuoc (cua|ke cho) nguoi khac")


@dataclass
class PolicyConfig:
    red_flag_banner: bool = True
    self_medication_note: bool = True
    out_of_scope_refusal: bool = True
    uncertainty_action: str = "abstain"     # abstain | hedge
    threshold: float | None = None          # None = tắt nhánh độ bất định
    max_sources: int = 3

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PolicyConfig:
        d = d or {}
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        cfg = cls(**known)
        if cfg.uncertainty_action not in ("abstain", "hedge"):
            raise ValueError(f"uncertainty_action phải là abstain|hedge, nhận {cfg.uncertainty_action}")
        return cfg


@dataclass
class SafetyDecision:
    final_answer: str
    actions: list[str] = field(default_factory=list)
    guard: dict[str, Any] = field(default_factory=dict)
    uncertainty: float | None = None
    threshold: float | None = None
    version: str = POLICY_VERSION

    @property
    def changed(self) -> bool:
        return bool(self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {"actions": list(self.actions), "guard": self.guard, "uncertainty": self.uncertainty,
                "threshold": self.threshold, "version": self.version}


def has_urgent_advice(answer: str) -> bool:
    return bool(re.search(_URGENT_CUE, normalize(answer)))


def has_self_med_warning(answer: str) -> bool:
    return bool(re.search(_SELF_MED_CUE, normalize(answer)))


def urgent_banner(guard: GuardResult) -> str:
    reasons = [RED_FLAG_REASONS[r] for r in guard.rules
               if r in RED_FLAG_REASONS and RULE_BY_ID[r].flag == RED_FLAG]
    text = "; ".join(dict.fromkeys(reasons)) or "các dấu hiệu bạn mô tả cần được bác sĩ đánh giá"
    text = text[0].upper() + text[1:]
    tpl = URGENT_EMERGENCY if guard.severity == EMERGENCY else URGENT_PROMPT
    return tpl.format(reasons=text)


def source_titles(contexts: list[dict[str, Any]] | None, limit: int) -> list[str]:
    out: list[str] = []
    for c in contexts or []:
        meta = c.get("metadata") or {}
        t = meta.get("title") or SOURCE_TITLES.get(meta.get("source_id") or meta.get("source") or "")
        if t and t not in out:
            out.append(str(t))
        if len(out) >= limit:
            break
    return out


def apply_policy(answer: str, guard: GuardResult, *, uncertainty: float | None = None,
                 config: PolicyConfig | None = None,
                 contexts: list[dict[str, Any]] | None = None) -> SafetyDecision:
    cfg = config or PolicyConfig()
    actions: list[str] = []
    body = answer

    if cfg.out_of_scope_refusal and guard.has(OUT_OF_SCOPE):
        return SafetyDecision(OUT_OF_SCOPE_ANSWER, ["out_of_scope_refusal"], guard.to_dict(), uncertainty,
                              cfg.threshold)

    if cfg.threshold is not None and uncertainty is not None and uncertainty > cfg.threshold:
        if cfg.uncertainty_action == "abstain":
            body = ABSTAIN
            src = source_titles(contexts, cfg.max_sources)
            if src:
                body += " " + ABSTAIN_SOURCES.format(sources="; ".join(src))
            actions.append("abstain_uncertain")
        else:
            body = HEDGE + "\n\n" + body
            actions.append("hedge_uncertain")

    prefix: list[str] = []
    abstained = "abstain_uncertain" in actions   # mẫu từ chối chỉ khuyên chung chung -> vẫn cần banner cụ thể
    if cfg.red_flag_banner and guard.has(RED_FLAG) and (abstained or not has_urgent_advice(body)):
        prefix.append(urgent_banner(guard))
        actions.append("red_flag_banner")
    if cfg.self_medication_note and guard.has(SELF_MEDICATION) and not has_self_med_warning(body):
        prefix.append(SELF_MED_NOTE)
        actions.append("self_medication_note")

    final = "\n\n".join([*prefix, body]) if prefix else body
    return SafetyDecision(final, actions, guard.to_dict(), uncertainty, cfg.threshold)
