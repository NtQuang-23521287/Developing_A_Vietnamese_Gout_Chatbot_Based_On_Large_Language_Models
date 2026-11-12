"""Giai đoạn 5 — Guard đầu vào theo luật (không gọi model, chạy được trên CPU).

Nhận diện ba nhóm tình huống trong lời người dùng:
  - red_flag        : dấu hiệu cần khám/cấp cứu (sốt, đau ngực, khó thở, tiểu rất ít, phân đen, ngộ độc
                      colchicine, dị ứng thuốc nặng, trẻ nhỏ sưng khớp, tophi vỡ/chảy mủ...).
                      Chia hai mức: "emergency" (đi cấp cứu ngay) và "prompt" (đi khám sớm, trong ngày).
  - self_medication : tự ý dùng/tăng/giảm/ngừng thuốc, dùng thuốc của người khác, xin liều để tự mua,
                      corticoid/NSAID kéo dài, bỏ thuốc theo thuốc nam, nhịn đói... (nhóm safety_trap).
  - out_of_scope    : cả hội thoại không có từ khóa nào thuộc lĩnh vực (gút, khớp, thuốc, ăn uống, thận...).

Đa số luật chạy trên văn bản đã BỎ DẤU (đ -> d) để bắt cả câu gõ không dấu; vài luật cần phân biệt dấu
(dừng/dùng) chạy trên văn bản CÓ DẤU (`accented=True`). Guard ưu tiên recall: báo nhầm chỉ làm câu trả lời có
thêm một đoạn lưu ý, còn bỏ sót red flag là lỗi an toàn.

Luật được xây trên câu hỏi SFT v2 và split dev; KHÔNG dùng split test (giữ cho nghiệm thu cuối).
Đổi luật thì tăng GUARD_VERSION.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

GUARD_VERSION = "guard_rules_v1"

RED_FLAG = "red_flag"
SELF_MEDICATION = "self_medication"
OUT_OF_SCOPE = "out_of_scope"
FLAGS = (RED_FLAG, SELF_MEDICATION, OUT_OF_SCOPE)

EMERGENCY = "emergency"
PROMPT = "prompt"


def normalize(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt (đ -> d), dấu câu thành khoảng trắng (giữ . , ° / %), gom khoảng trắng."""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d")
    t = re.sub(r"[^\w\s.,°/%]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def lower_accented(text: str) -> str:
    t = unicodedata.normalize("NFC", text.lower())
    t = re.sub(r"[^\w\s.,°/%]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


NAMED_DRUGS = (r"(colchicin\w*|allopurinol|febuxostat|probenecid|benzbromaron\w*|lesinurad|pegloticase|"
               r"corticoid\w*|cortico\w*|prednisolon\w*|prednison\w*|dexamethason\w*|methylprednisolon\w*|"
               r"medrol|ibuprofen|diclofenac|meloxicam|naproxen|celecoxib|etoricoxib|indomethacin|piroxicam|"
               r"nsaids?|\bult\b|thuoc (giam dau|khang viem|chong viem|ha (acid|axit) uric|tang thai))")
DRUGS = r"(" + NAMED_DRUGS + r"|thuoc)"
_STEROID_NSAID = (r"corticoid|cortico|prednisolon|prednison|dexamethason|medrol|ibuprofen|diclofenac|"
                  r"meloxicam|naproxen|celecoxib|etoricoxib|nsaid|giam dau|khang viem")
_FEVER = r"\bsot\b(?! ruot)|\bret run\b|\blanh run\b|\bon lanh\b|\b(38[.,]5|39|40|41) ?(do\b|°)"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    flag: str
    patterns: tuple[str, ...]          # tất cả phải khớp (AND); trong mỗi regex là OR
    exclude: tuple[str, ...] = ()      # khớp bất kỳ pattern nào thì luật không áp dụng
    severity: str | None = None        # chỉ cho red_flag: emergency | prompt
    accented: bool = False             # True: chạy trên văn bản có dấu

    def match(self, plain: str, accented: str) -> bool:
        text = accented if self.accented else plain
        if any(re.search(p, text) for p in self.exclude):
            return False
        return all(re.search(p, text) for p in self.patterns)


RULES: tuple[Rule, ...] = (
    # ---------------- red flag: emergency ----------------
    Rule("fever", RED_FLAG, (_FEVER,), exclude=(r"\bbo sot\b", r"\bsai sot\b"), severity=EMERGENCY),
    Rule("chest_pain_dyspnea", RED_FLAG,
         (r"dau (that |tuc )?nguc|tuc nguc|\bkho tho\b|\bngat( xiu)?\b",), severity=EMERGENCY),
    Rule("gi_bleeding", RED_FLAG,
         (r"phan den|di (ngoai|cau) ra mau|non ra mau|oi ra mau|phan (co )?mau|xuat huyet tieu hoa",),
         severity=EMERGENCY),
    Rule("renal_failure_signs", RED_FLAG,
         (r"tieu (rat |qua |hoi |)it\b|\bit tieu\b|vo nieu|khong (di )?tieu duoc|\bbi tieu\b(?! (duong|chay))|"
          r"tieu ra mau|tieu mau|dau quan (than|hong|lung|that lung|vung hong)",), severity=EMERGENCY),
    Rule("colchicine_toxicity", RED_FLAG,
         (r"tieu chay (nhieu|nang|lien tuc|du doi|khong (ngung|dut|cam)|ra nuoc)|non (nhieu|lien tuc|oi)|"
          r"\byeu co\b|khong (dung|ngoi) (day|len) (noi|duoc)|yeu tay chan",), severity=EMERGENCY),
    Rule("severe_drug_reaction", RED_FLAG,
         (r"(noi|phat|moc) (ban|man|mun nuoc|bong nuoc)|phong rop|bong (troc )?da|troc da|"
          r"loet (mieng|niem mac)|(phan ung|hoi chung) qua man|di ung nang|"
          r"phan ung (di ung )?nghiem trong|soc phan ve|sung (moi|hong|mi mat|mat\b(?! ca))",),
         exclude=(r"(truoc day|nam ngoai|tung bi|lan truoc|cach day|hoi truoc)[^.;]{0,40}"
                  r"(noi ban|noi man|di ung|phan ve|qua man)",),
         severity=EMERGENCY),
    Rule("neuro_emergency", RED_FLAG,
         (r"lu lan|hon me|co giat|\bli bi\b|\bliet (nua|chan|tay|mat)|\bbi liet\b|meo mieng|noi ngong|"
          r"mat y thuc",), severity=EMERGENCY),
    Rule("severe_abdominal_pain", RED_FLAG,
         (r"dau bung (du doi|qua|nhieu|tren ron|vung thuong vi)",), severity=EMERGENCY),
    # ---------------- red flag: prompt (khám sớm) ----------------
    Rule("child_joint", RED_FLAG,
         (r"\b([0-9]|1[0-5]) tuoi\b|\b\d{1,2} thang tuoi\b|\b(be|chau|con) (moi |duoc )?\d{1,2} thang\b|"
          r"\b(con|chau|em) be\b|\bbe (trai|gai)\b|tre (em|nho|so sinh)",
          r"\bsung\b|\bnong\b|\bdau\b|quay khoc|khong chiu di|tap tenh"), severity=PROMPT),
    Rule("tophus_infection", RED_FLAG,
         (r"tophi|tophus|to phi|\bhat\b|\bcuc\b|\bkhop\b",
          r"chay (mu|dich)|\b(co|ra|mung) mu\b|nhiem trung|vo loet|\bvo ra\b|"
          r"(tophi|hat|cuc)( \w+)? (bi |da )?(vo|loet)\b"), severity=PROMPT),
    Rule("colchicine_gi_effect", RED_FLAG,
         (r"colchicin", r"tieu chay|dau bung|\bnon\b|buon non"), severity=PROMPT),
    Rule("home_management_question", RED_FLAG,
         (r"tu xu (tri|ly) (o|tai) nha|theo doi (them )?(o|tai) nha|nam nghi (chut|them)|"
          r"o nha (co )?(duoc|on) khong",
          _FEVER + r"|sung nong|nong do|dau (du doi|nhieu)|noi ban|phong rop|tieu chay"), severity=PROMPT),

    # ---------------- self-medication / safety trap ----------------
    Rule("self_adjust", SELF_MEDICATION,
         (r"\btu y\b|\btu (tang|ngung|cat|mua|uong|tiem|ke|chuyen|doi|chinh) |"
          r"\btu (giam|bo|dung|dieu chinh) (lieu|thuoc|uong)|\btu (dieu tri|chua tri|chua benh)\b",)),
    Rule("dose_change", SELF_MEDICATION,
         (r"gap (doi|2|hai|ba|3) lieu|tang gap|\b(tang|giam|nang|ha) lieu|uong bu|qua lieu|"
          r"lieu cao (hon|de|cho)|(uong|dung) (tam|chung)\b|(uong|dung) (them|tiep|lai) (" + NAMED_DRUGS
          + r"|vien|lieu)",
          r"\b(muon|dinh|co nen|nen tu|duoc khong|co duoc|duoc chu|nhe\b|cho nhanh|de nhanh|nhanh (khoi|het|do))")),
    Rule("stop_drug", SELF_MEDICATION,
         (r"\b(ngung|cat|thoi) (het |han |ngay )?(uong |dung )?" + DRUGS + r"|\bbo (het |han |ngay )?" + DRUGS,),
         exclude=(r"bac si (cho|bao|yeu cau|chi dinh|dan) (toi |minh |em |chau |bac )?(ngung|dung|bo|cat)",)),
    Rule("stop_drug_accented", SELF_MEDICATION,
         (r"\bdừng (hết |hẳn |ngay )?(uống |dùng )?(thuốc|colchicin|allopurinol|febuxostat)",),
         exclude=(r"bác sĩ (cho|bảo|yêu cầu|chỉ định|dặn) (tôi |mình |em |cháu |bác )?dừng",), accented=True),
    Rule("others_medication", SELF_MEDICATION,
         (r"thuoc cua (bo|me|ong|ba|vo|chong|con|chau|ban|hang xom|nguoi (khac|nha|quen)|toi|anh|chi|em)\b|"
          r"lay thuoc cua|(uong|dung) chung|(cho|dua) (chau|con|be|vo|chong|bo|me|ong|ba|anh|em) (uong|dung)\b|"
          r"mua (ve )?cho (chau|con|be|vo|chong|bo|me|ong|ba|anh|em|no|ong ay|anh ay|ba ay)\b",)),
    Rule("buy_without_prescription", SELF_MEDICATION,
         (r"\bmua (ve |them |san |o ngoai |)" + DRUGS + r"|\btu mua\b|tiem (thuoc )?vao khop|\btiem khop\b|"
          r"\bnha thuoc\b|\bhieu thuoc\b",)),
    Rule("dose_request", SELF_MEDICATION,
         (r"lieu (cu the|bao nhieu|chinh xac)|\bxin (\w+ )?lieu\b|uong (may|bao nhieu) (vien|mg|lan)|"
          r"bao nhieu (mg|vien)",)),
    Rule("long_term_steroid_nsaid", SELF_MEDICATION,
         (_STEROID_NSAID,
          r"dai ngay|hang (thang|ngay)|moi ngay|lau dai|thuong xuyen|keo dai|(toi|den) khi het|"
          r"tiep (tuc|hang)|khoi (len )?con")),
    Rule("alternative_remedy", SELF_MEDICATION,
         (r"thuoc nam|bai thuoc|thuoc bac|thuoc dan gian|thuoc gia truyen|la cay|"
          r"nhin (an|doi)|chi uong nuoc|thuc pham chuc nang|thay (cho |the )?thuoc (bac si|tay)|"
          r"mua (tren|qua) mang",)),
    Rule("contraindicated_combo", SELF_MEDICATION,
         (NAMED_DRUGS,
          r"loet da day|da day ta trang|suy than|than man|benh than|soi than|mang thai|co thai|xuat huyet",
          r"duoc khong|co nen|co the (dung|uong)|dung duoc|uong duoc|an toan khong")),
    Rule("pregnancy_drug", SELF_MEDICATION, (r"mang thai|co thai|co bau|cho con bu", DRUGS)),
    Rule("allergy_rechallenge", SELF_MEDICATION,
         (r"di ung|noi ban|noi man|(phan ung|hoi chung) qua man",
          r"(uong|dung) (lai|tiep)|lieu thap|thu (lai|dung)")),
)

# Từ khóa lĩnh vực: chỉ cần MỘT từ trong cả hội thoại là coi như trong phạm vi (rất rộng, cố ý).
_DOMAIN = (r"\bgut\b|\bgout\b|thong phong|acid uric|axit uric|\burat|purin|\bkhop\b|tophi|to phi|"
           r"\bsung\b|\bdau\b|\bthuoc\b|\blieu\b|\ban\b|\buong\b|kieng|\bbia\b|\bruou\b|\bthan\b|\bsoi\b|"
           r"xet nghiem|\bsot\b|bac si|benh|trieu chung|ngon chan|co chan|dau goi|\bthit\b|"
           r"hai san|can nang|tap (the duc|luyen)|\buric\b|huyet ap|tim mach|tieu duong|mo mau|suc khoe|"
           r"thuc pham|an uong|qua|trai cay|rau|chi so|\bchan\b|\btay\b|\bgan\b|vaccin|tiem|kham|"
           r"dieu tri|tac dung phu|nhiem doc|\bbo\b|\bca\b|\bnuoc\b|furic|urictab|" + DRUGS)


@dataclass
class GuardResult:
    flags: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    matched_in: dict[str, str] = field(default_factory=dict)   # rule_id -> "current" | "history"
    severity: str | None = None                                 # mức cao nhất của red flag
    version: str = GUARD_VERSION

    def has(self, flag: str) -> bool:
        return flag in self.flags

    def to_dict(self) -> dict:
        return {"flags": list(self.flags), "rules": list(self.rules), "matched_in": dict(self.matched_in),
                "severity": self.severity, "version": self.version}


RULE_BY_ID = {r.rule_id: r for r in RULES}


def run_guard(question: str, prior_user_turns: list[str] | tuple[str, ...] = (),
              history_flags: tuple[str, ...] = (RED_FLAG,)) -> GuardResult:
    """Áp luật lên câu hỏi hiện tại; với nhóm trong `history_flags` xét thêm các lượt người dùng trước
    (dấu hiệu nguy hiểm kể ở lượt trước vẫn còn hiệu lực).

    self_medication chỉ xét câu hỏi hiện tại: ý định tự dùng thuốc phải nằm ở chính câu đang hỏi.
    """
    prior_raw = [t for t in prior_user_turns if t and t.strip()]
    cur_p, cur_a = normalize(question), lower_accented(question)
    all_p = " . ".join([normalize(t) for t in prior_raw] + [cur_p])
    all_a = " . ".join([lower_accented(t) for t in prior_raw] + [cur_a])
    res = GuardResult()
    for rule in RULES:
        where = None
        if rule.match(cur_p, cur_a):
            where = "current"
        elif rule.flag in history_flags and prior_raw and rule.match(all_p, all_a):
            where = "history"
        if not where:
            continue
        res.rules.append(rule.rule_id)
        res.matched_in[rule.rule_id] = where
        if rule.flag not in res.flags:
            res.flags.append(rule.flag)
        if rule.severity == EMERGENCY or (rule.severity == PROMPT and res.severity is None):
            res.severity = rule.severity
    if not re.search(_DOMAIN, all_p):
        res.flags.append(OUT_OF_SCOPE)
        res.rules.append("no_domain_keyword")
        res.matched_in["no_domain_keyword"] = "current"
    return res
