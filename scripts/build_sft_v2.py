"""Pipeline tạo dữ liệu SFT v2 (RAG-aware), sửa các lỗi đã đo được ở SFT v1.

Các bước (chạy lần lượt; mỗi bước ghi file trung gian nên chạy lại được, gọi API có cache):

  questions : gom câu hỏi (seed QA + sinh ngược) + sinh thêm câu hỏi từ từng chunk KB và hội thoại 3 lượt;
              loại câu trùng/gần trùng với testset v03 (chống leakage)
  retrieve  : truy xuất context cho TỪNG lượt bằng đúng FaissRetriever + rag_v1 như lúc suy luận
              (cần faiss, sentence-transformers, FlagEmbedding -> nên chạy trên Colab)
  answer    : gpt-4.1-mini viết câu trả lời giọng tư vấn, chỉ dựa trên context
  filter    : lọc bằng luật (Vinmec, tên người, địa chỉ, liều, độ dài, tiếng Anh...) + LLM kiểm tra căn cứ/an toàn
  review_export : xuất file Excel cho BÁC SĨ duyệt (artifacts/sft_v2/review/sft_v2_review.xlsx)
  review_import : đọc kết quả duyệt (Đạt / Cần sửa / Loại) -> reviewed.jsonl + review_manifest.json
              (quy tắc: src/gout_llmops/data/doctor_review.py)
  build     : dựng messages bằng build_messages() của hệ thống (system prompt + NGỮ CẢNH RAG + lịch sử),
              chia train/val, ghi data/sft_v2/ + manifest. Mặc định CHỈ build từ dữ liệu bác sĩ đã duyệt.

Ví dụ:
  PYTHONPATH=src python3 scripts/build_sft_v2.py questions
  PYTHONPATH=src python3 scripts/build_sft_v2.py retrieve          # Colab GPU
  PYTHONPATH=src python3 scripts/build_sft_v2.py answer
  PYTHONPATH=src python3 scripts/build_sft_v2.py filter
  PYTHONPATH=src python3 scripts/build_sft_v2.py review_export
  #   ... bác sĩ duyệt file Excel ...
  PYTHONPATH=src python3 scripts/build_sft_v2.py review_import --dry-run      # xem tiến độ
  PYTHONPATH=src python3 scripts/build_sft_v2.py review_import --reviewer "BS. ..."
  PYTHONPATH=src python3 scripts/build_sft_v2.py build
Thử toàn bộ không tốn tiền, không cần GPU: thêm --mock (KHÔNG dùng dữ liệu mock để train).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from gout_llmops.chat.prompt_builder import build_messages
from gout_llmops.core.io import read_jsonl, write_json, write_jsonl

WORK = Path("artifacts/sft_v2")
OUT = Path("data/sft_v2")
CACHE = WORK / "llm_cache.jsonl"
SYSTEM_PROMPT_PATH = "prompts/system_vi_v1.txt"
MAX_HISTORY_PAIRS = 2
CATEGORIES = ("basic_knowledge", "clinical_reasoning", "red_flag", "safety_trap", "out_of_scope")
PRICE = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4o-mini": (0.15, 0.60)}

# ---------------------------------------------------------------------------
# LLM có cache
# ---------------------------------------------------------------------------
class LLM:
    def __init__(self, model: str, mock: bool):
        self.model, self.mock = model, mock
        self.cache: dict[str, Any] = {}
        self.usage = Counter()
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r["value"]
        if not mock:
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise SystemExit("Thiếu OPENAI_API_KEY")
            self.client = OpenAI()

    def json(self, system: str, user: str, tag: str, mock_value: dict[str, Any]) -> dict[str, Any]:
        if self.mock:
            return mock_value
        key = hashlib.sha256("\x1f".join([tag, self.model, system, user]).encode("utf-8")).hexdigest()
        if key in self.cache:
            self.usage["cached"] += 1
            return self.cache[key]
        err = None
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0.3 if tag != "filter" else 0.0,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                self.usage["calls"] += 1
                if r.usage:
                    self.usage["in"] += r.usage.prompt_tokens or 0
                    self.usage["out"] += r.usage.completion_tokens or 0
                value = json.loads(r.choices[0].message.content)
                self.cache[key] = value
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                with CACHE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")
                return value
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM lỗi: {err}")

    def cost(self) -> float:
        pin, pout = PRICE.get(self.model, (0, 0))
        return round(self.usage["in"] / 1e6 * pin + self.usage["out"] / 1e6 * pout, 4)


# ---------------------------------------------------------------------------
# Chống leakage với testset v03
# ---------------------------------------------------------------------------
def _norm(t: str) -> str:
    t = re.sub(r"(bác sĩ|bác sỹ|dr\.?)\s+meddies", "bác sĩ", t, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


def _grams(t: str, n: int = 4) -> set[str]:
    t = _norm(t)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


class LeakGuard:
    def __init__(self, testset: str, threshold: float = 0.40):
        self.threshold = threshold
        self.test = []
        for c in read_jsonl(testset):
            for t in c["turns"]:
                self.test.append((c["case_id"], _grams(t["user"])))

    def check(self, text: str) -> tuple[float, str | None]:
        g = _grams(text)
        best, who = 0.0, None
        for cid, tg in self.test:
            inter = len(g & tg)
            if not inter:
                continue
            j = inter / len(g | tg)
            if j > best:
                best, who = j, cid
        return best, who


# ---------------------------------------------------------------------------
# Bước 1: questions
# ---------------------------------------------------------------------------
GEN_Q_SYSTEM = """Bạn tạo câu hỏi tiếng Việt của NGƯỜI BỆNH (hoặc người nhà) gửi cho chatbot tư vấn bệnh gút.
Chỉ dựa trên ĐOẠN TÀI LIỆU được cung cấp: câu hỏi phải trả lời được bằng tài liệu đó.
Viết tự nhiên như người thật nhắn tin (có thể kể tuổi, triệu chứng, thuốc đang dùng), 1–3 câu, không trích nguyên văn tài liệu.
Không nêu tên người, tên bệnh viện, địa chỉ.
Trả về JSON: {"questions": [{"q": str, "category": str}]} với category thuộc:
basic_knowledge (hỏi kiến thức chung), clinical_reasoning (kể tình huống của bản thân và hỏi cách xử trí),
red_flag (có dấu hiệu cần đi khám NGAY: sốt cao kèm khớp sưng nóng, ban/phồng rộp da khi dùng thuốc, tiêu chảy nặng
khi dùng colchicine, tophi chảy mủ, đau quặn thận kèm sốt...), safety_trap (người hỏi muốn làm điều nguy hiểm: tự tăng/
ngưng thuốc, xin liều để tự mua, dùng thuốc chống chỉ định, bỏ thuốc theo lời mách...)."""

GEN_DIALOG_SYSTEM = """Bạn viết lời NGƯỜI HỎI trong một hội thoại 3 lượt với chatbot tư vấn bệnh gút, theo KỊCH BẢN BẮT BUỘC.
Lượt 1: giới thiệu đúng người hỏi, tuổi, tình trạng nền và thuốc đang dùng theo kịch bản (không thêm bệnh khác).
Lượt 2: kể triệu chứng/tình huống theo kịch bản.
Lượt 3: hỏi đúng mục đích trong kịch bản, KHÔNG nhắc lại tình trạng nền (để kiểm tra chatbot có nhớ lượt 1 không).
Viết đúng phong cách được giao, như người thật nhắn tin; mỗi lượt 1–2 câu; không mở đầu lượt 1 bằng mẫu
"Tôi X tuổi, bị ..., đang dùng ..., không có tiền sử dị ứng" — hãy diễn đạt tự nhiên và khác đi.
Không nêu tên người, tên bệnh viện. Trả về JSON: {"turns": [str, str, str]}."""


# Kịch bản hội thoại: chỉ định trước người hỏi / bệnh nền / mục đích lượt 3 để đa dạng (tránh GPT lặp một khuôn).
PERSONAS = [("Tôi", (38, 70)), ("Em", (24, 42)), ("Bố tôi", (55, 80)), ("Mẹ tôi (đã mãn kinh)", (55, 80)),
            ("Chồng tôi", (35, 65)), ("Ông nội em", (70, 88)), ("Cháu hỏi giúp bác", (55, 75)), ("Anh trai tôi", (30, 55))]
CONDITIONS = [  # (mô tả, thuốc đang dùng, từ khóa tìm chunk)
    ("suy thận mạn", "thuốc huyết áp", ["suy thận", "thận"]),
    ("viêm loét dạ dày tá tràng", "thuốc dạ dày", ["dạ dày", "NSAID"]),
    ("sỏi thận đã tán sỏi", "không dùng thuốc thường xuyên", ["sỏi", "thận"]),
    ("tiểu đường type 2", "thuốc hạ đường huyết", ["corticoid", "tiểu đường"]),
    ("tăng huyết áp", "thuốc lợi tiểu hydrochlorothiazide", ["lợi tiểu", "thiazid"]),
    ("từng nhồi máu cơ tim", "aspirin liều thấp và thuốc tim mạch", ["tim mạch", "febuxostat", "aspirin"]),
    ("từng dị ứng nổi ban khi uống allopurinol", "chưa dùng thuốc hạ acid uric", ["dị ứng", "allopurinol", "febuxostat"]),
    ("đang dùng allopurinol 8 tháng, acid uric đã về mục tiêu", "allopurinol", ["allopurinol", "liên tục", "ULT"]),
    ("có nhiều hạt tophi ở bàn tay", "chưa dùng thuốc hạ acid uric", ["tophi"]),
    ("béo phì, hay uống bia", "không dùng thuốc", ["bia", "rượu", "giảm cân"]),
    ("trên 75 tuổi, hay quên", "nhiều loại thuốc huyết áp", ["70 tuổi", "colchicin", "người cao tuổi"]),
    ("mới bắt đầu allopurinol 2 tuần", "allopurinol", ["allopurinol", "dị ứng", "đợt cấp"]),
]
INTENTS = [  # (mục đích lượt 3, category, từ khóa)
    ("muốn tự uống thuốc giảm đau kháng viêm (ibuprofen/diclofenac/meloxicam) mua ở tiệm", "safety_trap", ["NSAID", "kháng viêm"]),
    ("xin liều colchicine cụ thể để tự mua uống", "safety_trap", ["colchicin"]),
    ("muốn tự ngưng thuốc hạ acid uric vì thấy đỡ", "safety_trap", ["allopurinol", "liên tục"]),
    ("muốn tự tăng gấp đôi liều thuốc đang dùng cho nhanh", "safety_trap", ["liều", "allopurinol"]),
    ("hỏi có dùng thuốc tăng thải acid uric (probenecid) được không", "safety_trap", ["probenecid", "tăng thải"]),
    ("muốn mua corticoid uống dài ngày để khỏi lên cơn", "safety_trap", ["corticoid"]),
    ("muốn bỏ thuốc bác sĩ kê để uống thuốc nam theo lời mách", "safety_trap", ["điều trị lâu dài", "allopurinol"]),
    ("hỏi có cần dùng thuốc hạ acid uric lâu dài không", "clinical_reasoning", ["chỉ định", "hạ acid uric"]),
    ("hỏi nên ăn uống, sinh hoạt thế nào cho phù hợp với bệnh nền đã kể", "clinical_reasoning", ["chế độ ăn", "purin"]),
    ("kể thêm dấu hiệu mới nguy hiểm (sốt cao rét run, hoặc nổi ban phồng rộp, hoặc tiêu chảy nhiều lần) và hỏi có tự xử trí ở nhà được không",
     "red_flag", ["nhiễm khuẩn", "dị ứng", "tiêu chảy"]),
]
TURN2 = ["đau sưng nóng đỏ khớp ngón chân cái từ đêm qua", "cổ chân sưng đau 2 ngày nay, đi lại khó",
         "đầu gối sưng đau sau bữa nhậu", "mu bàn chân sưng đỏ, đau nhức về đêm",
         "khớp ngón tay sưng đau, tái đi tái lại", "chưa đau nhưng xét nghiệm acid uric cao"]
STYLES = ["nhắn tin ngắn, ít dấu câu", "lịch sự, kể chi tiết", "lo lắng, hỏi dồn", "giọng người lớn tuổi, xưng hô thân mật"]
# Cặp (tình trạng nền, mục đích) đã có trong testset v03 -> KHÔNG tạo để tránh rò rỉ kịch bản
BLACKLIST = {
    ("viêm loét dạ dày tá tràng", 0), ("suy thận mạn", 0), ("sỏi thận đã tán sỏi", 4), ("suy thận mạn", 4),
    ("tiểu đường type 2", 5), ("đang dùng allopurinol 8 tháng, acid uric đã về mục tiêu", 2),
    ("từng dị ứng nổi ban khi uống allopurinol", 2), ("trên 75 tuổi, hay quên", 9),
    ("mới bắt đầu allopurinol 2 tuần", 9), ("tăng huyết áp", 7),
}


def dialog_scenarios(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    combos = [(ci, ii) for ci in range(len(CONDITIONS)) for ii in range(len(INTENTS))
              if (CONDITIONS[ci][0], ii) not in BLACKLIST]
    rng.shuffle(combos)
    out = []
    for d in range(n):
        ci, ii = combos[d % len(combos)]
        cond, meds, kw1 = CONDITIONS[ci]
        intent, cat, kw2 = INTENTS[ii]
        persona, (lo, hi) = PERSONAS[rng.randrange(len(PERSONAS))]
        if "75 tuổi" in cond:
            lo, hi = max(lo, 76), max(hi, 85)
        out.append({"persona": persona, "age": rng.randint(lo, hi),
                    "cond": cond, "meds": meds, "turn2": rng.choice(TURN2), "intent": intent,
                    "category": cat, "style": rng.choice(STYLES), "keywords": kw1 + kw2})
    return out


def pick_chunks(chunks: list[dict[str, Any]], keywords: list[str], k: int = 3) -> list[dict[str, Any]]:
    scored = sorted(chunks, key=lambda c: -sum(c["text"].lower().count(w.lower()) for w in keywords))
    return scored[:k]


def stage_questions(args, llm: LLM) -> None:
    guard = LeakGuard(args.testset)
    rows: list[dict[str, Any]] = []

    for fname, src in (("Group2_SFT_Seed_QA.csv", "seed_qa"), ("Group2b_SFT_SinhNguoc.csv", "sinh_nguoc")):
        df = pd.read_csv(Path("data/data_filtered_final") / fname)
        for i, q in enumerate(df["question"].astype(str)):
            q = re.sub(r"\s+", " ", q).strip()
            if 8 <= len(q) <= 600:
                rows.append({"id": f"{src}_{i:04d}", "source": src, "category": "auto", "turns": [q]})

    chunks = read_jsonl(args.kb_chunks)
    rng = random.Random(args.seed)
    for c in chunks:
        res = llm.json(GEN_Q_SYSTEM, f"ĐOẠN TÀI LIỆU [{c['chunk_id']}]:\n{c['text']}\n\nTạo {args.q_per_chunk} câu hỏi, "
                       "ưu tiên đa dạng category; nếu tài liệu nói về thuốc hoặc biến chứng thì có ít nhất 1 câu "
                       "red_flag hoặc safety_trap.", "gen_q",
                       {"questions": [{"q": f"[mock] Hỏi về {c['chunk_id']} số {k}?", "category":
                                       rng.choice(CATEGORIES[:4])} for k in range(args.q_per_chunk)]})
        for k, item in enumerate(res.get("questions", [])[: args.q_per_chunk]):
            cat = item.get("category") if item.get("category") in CATEGORIES else "clinical_reasoning"
            rows.append({"id": f"kbq_{c['chunk_id']}_{k}", "source": "kb_generated", "category": cat,
                         "seed_chunk": c["chunk_id"], "turns": [str(item.get("q", "")).strip()]})

    for d, sc in enumerate(dialog_scenarios(args.n_dialogs, args.seed)):
        ctx = pick_chunks(chunks, sc["keywords"], k=3)
        brief = (f"NGƯỜI HỎI: {sc['persona']}. TUỔI: {sc['age']}. TÌNH TRẠNG NỀN: {sc['cond']}. "
                 f"THUỐC ĐANG DÙNG: {sc['meds']}.\nLƯỢT 2: {sc['turn2']}.\nLƯỢT 3 (mục đích): {sc['intent']}.\n"
                 f"Phong cách: {sc['style']}.")
        res = llm.json(GEN_DIALOG_SYSTEM, "\n\n".join(f"ĐOẠN TÀI LIỆU [{c['chunk_id']}]:\n{c['text']}" for c in ctx)
                       + "\n\nKỊCH BẢN BẮT BUỘC:\n" + brief, "gen_dialog_v2",
                       {"turns": [f"[mock] {sc['persona']} {sc['age']} {sc['cond']}", f"[mock] {sc['turn2']}",
                                  f"[mock] {sc['intent']} #{d}"], "category": sc["category"]})
        turns = [str(t).strip() for t in res.get("turns", [])][:3]
        if len(turns) == 3 and all(turns):
            rows.append({"id": f"dlg_{d:03d}", "source": "dialog_generated", "category": sc["category"],
                         "scenario": {k: sc[k] for k in ("persona", "cond", "intent")},
                         "seed_chunks": [c["chunk_id"] for c in ctx], "turns": turns})

    kept, dropped, seen = [], [], set()
    for r in rows:
        if not all(r["turns"]):
            continue
        key = _norm(" ".join(r["turns"]))
        if key in seen:
            dropped.append({**r, "drop": "duplicate"})
            continue
        seen.add(key)
        score, who = max((guard.check(t) for t in r["turns"]), key=lambda x: x[0])
        if score >= guard.threshold:
            dropped.append({**r, "drop": f"leak~{who} ({score:.2f})"})
            continue
        r["leak_max_jaccard"] = round(score, 3)
        kept.append(r)
    write_jsonl(WORK / "questions.jsonl", kept)
    write_jsonl(WORK / "questions_dropped.jsonl", dropped)
    print(f"Câu hỏi giữ: {len(kept)} | loại: {len(dropped)} (trùng/leakage)")
    print("Theo nguồn:", dict(Counter(r["source"] for r in kept)))
    print("Theo category:", dict(Counter(r["category"] for r in kept)))
    print(f"Chi phí: ${llm.cost()} | {dict(llm.usage)}")


# ---------------------------------------------------------------------------
# Bước 2: retrieve (đúng retriever lúc suy luận)
# ---------------------------------------------------------------------------
def stage_retrieve(args) -> None:
    qs = read_jsonl(WORK / "questions.jsonl")
    if args.mock:
        chunks = read_jsonl(args.kb_chunks)

        def retrieve(q: str) -> list[dict[str, Any]]:
            g = _grams(q, 3)
            ranked = sorted(chunks, key=lambda c: -len(g & _grams(c["text"][:600], 3)))
            return [dict(c) for c in ranked[:3]]
    else:
        import yaml
        from gout_llmops.rag.retriever import FaissRetriever
        cfg = yaml.safe_load(Path(args.rag_config).read_text(encoding="utf-8"))
        r = FaissRetriever(cfg["output_dir"], cfg)
        if cfg.get("reranker", {}).get("enabled") and r.reranker is None:
            raise SystemExit("Reranker không nạp được — context sẽ lệch so với lúc suy luận. Cài FlagEmbedding / dùng GPU.")
        retrieve = r.retrieve
    out = []
    for i, q in enumerate(qs, 1):
        ctx = []
        for t in q["turns"]:
            hits = retrieve(t)
            ctx.append([{k: h.get(k) for k in ("chunk_id", "text", "metadata", "retrieval_score", "rerank_score")}
                        for h in hits])
        out.append({**q, "contexts": ctx})
        if i % 50 == 0:
            print(f"  {i}/{len(qs)}")
    write_jsonl(WORK / "retrieved.jsonl", out)
    print(f"Đã truy xuất {len(out)} mẫu ({sum(len(q['turns']) for q in out)} lượt){' [MOCK]' if args.mock else ''}")


# ---------------------------------------------------------------------------
# Bước 3: answer
# ---------------------------------------------------------------------------
TEACHER_SYSTEM = """Bạn viết CÂU TRẢ LỜI MẪU để huấn luyện chatbot tư vấn bệnh gút tiếng Việt. Câu trả lời này sẽ được
mô hình học theo, nên phải đúng, an toàn và đúng phong cách.

NGUỒN: chỉ dùng thông tin trong NGỮ CẢNH RAG và lịch sử hội thoại. Không thêm kiến thức ngoài ngữ cảnh.
Nếu ngữ cảnh không đủ để trả lời, nói rõ tài liệu hiện có chưa đủ thông tin và khuyên hỏi bác sĩ; không bịa.

PHONG CÁCH: xưng "bạn" (hoặc theo cách người hỏi tự xưng như bác/anh/chị), tiếng Việt tự nhiên, thấu cảm,
80–220 từ. Đoạn văn ngắn hoặc gạch đầu dòng ngắn. Không viết theo kiểu bài báo, không đặt tiêu đề.
Không trộn tiếng Anh (viết "bác sĩ chuyên khoa Cơ xương khớp", không viết "Rheumatology").

BẮT BUỘC:
- Trả lời đúng điều người hỏi hỏi ở lượt cuối, và dùng thông tin quan trọng họ đã nêu ở các lượt trước
  (tuổi, bệnh thận, dạ dày, mang thai, thuốc đang dùng, tiền sử dị ứng).
- Không chẩn đoán xác định: dùng "có thể là dấu hiệu của…", kèm lý do cần khám/xét nghiệm.
- Không đưa liều thuốc (mg, số viên) để người bệnh tự dùng; không khuyên tự mua, tự tăng, tự ngưng, tự đổi thuốc.
- Khi cần khám: chuyên khoa Cơ xương khớp (không nêu chuyên khoa khác nếu ngữ cảnh không nói).
- Nêu cảnh báo an toàn phù hợp (khi nào cần đi khám, khi nào cần đi NGAY).
- Nếu là tình huống nguy hiểm (sốt cao kèm khớp sưng nóng, ban/phồng rộp da khi dùng thuốc, tiêu chảy/nôn nhiều khi dùng
  colchicine, tophi chảy mủ, đau quặn kèm sốt, nôn ra máu, đau ngực): câu đầu tiên khuyên đến cơ sở y tế/cấp cứu NGAY.
- Nếu người hỏi muốn làm điều nguy hiểm: từ chối rõ ràng phần nguy hiểm, giải thích ngắn lý do theo ngữ cảnh, chuyển hướng an toàn.
- Câu hỏi ngoài phạm vi bệnh gút: nói ngắn gọn là ngoài phạm vi và khuyên hỏi bác sĩ phù hợp.
- TUYỆT ĐỐI không nêu tên người, tên bệnh viện/phòng khám, địa chỉ, số điện thoại, website, "bài viết này được viết bởi".

Trả về JSON: {"answer": str, "category": str, "used_chunks": [chunk_id]} với category thuộc basic_knowledge,
clinical_reasoning, red_flag, safety_trap, out_of_scope (phân loại câu hỏi ở lượt cuối)."""


def _messages_for(system_prompt: str, turns: list[str], answers: list[str], contexts: list[list[dict]], i: int):
    history = []
    for u, a in zip(turns[:i], answers[:i]):
        history += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    return build_messages(system_prompt, turns[i], history, contexts[i], MAX_HISTORY_PAIRS)


def stage_answer(args, llm: LLM) -> None:
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    rows = read_jsonl(WORK / "retrieved.jsonl")
    if args.limit:
        rows = rows[: args.limit]
    out = []
    for n, r in enumerate(rows, 1):
        answers: list[str] = []
        cat = r["category"]
        for i in range(len(r["turns"])):
            msgs = _messages_for(system_prompt, r["turns"], answers, r["contexts"], i)
            rendered = "\n\n".join(f"<{m['role'].upper()}>\n{m['content']}" for m in msgs)
            res = llm.json(TEACHER_SYSTEM, "Đây là đúng những gì chatbot nhìn thấy. Hãy viết câu trả lời cho lượt cuối.\n\n"
                           + rendered, "answer",
                           {"answer": f"[mock] Có thể là dấu hiệu của gút; bạn nên khám chuyên khoa Cơ xương khớp. "
                                      f"Đi khám ngay nếu sốt cao. ({r['id']} lượt {i + 1}) " + "nội dung giả lập " * 20,
                            "category": cat if cat in CATEGORIES else "clinical_reasoning", "used_chunks": []})
            answers.append(str(res.get("answer", "")).strip())
            if i == len(r["turns"]) - 1 and res.get("category") in CATEGORIES:
                cat = res["category"] if r["category"] == "auto" else r["category"]
        out.append({**r, "answers": answers, "category": cat if cat in CATEGORIES else "clinical_reasoning"})
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}  ${llm.cost()}")
    write_jsonl(WORK / "answered.jsonl", out)
    print(f"Đã viết câu trả lời cho {len(out)} mẫu | chi phí ${llm.cost()} | {dict(llm.usage)}")


# ---------------------------------------------------------------------------
# Bước 4: filter
# ---------------------------------------------------------------------------
BANNED = {
    "co_so_y_te_cu_the": r"vinmec|bệnh viện (đa khoa|quốc tế|bạch mai|chợ rẫy)|phòng khám \w+ \w+|times city",
    "ghi_tac_gia": r"bài viết (này )?(được )?viết bởi|tham vấn (y khoa|chuyên môn)|biên tập",
    "dia_chi_lien_he": r"địa chỉ|số \d+,? (đường|phố)|hotline|tổng đài|\b1900\s?\d|https?://|www\.|\.com\b|\.vn\b",
    "tieng_anh": r"rheumatolog|\bgout flare\b|\buric acid\b|\bdoctor\b|\bpatient\b",
    "chuyen_khoa_sai": r"răng hàm mặt|hộ lý|tai mũi họng",
    "lieu_cu_the": r"\b\d+([.,]\d+)?\s?(mg|g|viên|ml)\b(?!/dl)",
    "chan_doan_xac_dinh": r"\b(chắc chắn|khẳng định) (là |bạn |bác |anh |chị )?(bị )?gút\b",
    "giong_benh_nhan": r"nhà mình cũng|cho (em|mình|tôi) hỏi|không biết có phải .* không ạ\?$",
    "dang_bai_bao": r"(^|\n)\s*#+\s|(^|\n)\s*[1-9]\.\s+[A-ZĐ][^\n]{0,60}\n",
}
# Tên người Việt sau danh xưng (ví dụ "bác sĩ Nguyễn Văn A")
PERSON = re.compile(r"(bác sĩ|bs\.?|thạc sĩ|tiến sĩ|ths\.?|ts\.?)\s+[A-ZĐ][a-zà-ỹ]+\s+[A-ZĐ][a-zà-ỹ]+", re.U)

CHECK_SYSTEM = """Bạn kiểm tra một câu trả lời mẫu dùng để huấn luyện chatbot tư vấn bệnh gút.
Đối chiếu với NGỮ CẢNH RAG. Trả về JSON:
{"unsupported_claims": [str],   // khẳng định y khoa KHÔNG có trong ngữ cảnh (diễn đạt lại ngữ cảnh thì vẫn là có căn cứ)
 "unsafe": bool,                // có khuyên điều nguy hiểm: liều để tự dùng, tự ngưng/tăng thuốc, ở nhà khi có dấu hiệu cấp cứu
 "misses_urgent": bool,         // tình huống nguy hiểm mà không khuyên đi khám/cấp cứu NGAY
 "ignores_history": bool,       // bỏ qua thông tin quan trọng người hỏi đã nêu ở lượt trước
 "answers_question": bool,      // có trả lời đúng điều được hỏi
 "role_ok": bool}               // nói với vai chatbot tư vấn (không nhập vai người bệnh, không viết như bài báo)"""


def rule_flags(ans: str) -> list[str]:
    flags = [k for k, p in BANNED.items() if re.search(p, ans, re.I | re.M)]
    if PERSON.search(ans):
        flags.append("ten_nguoi")
    n = len(ans.split())
    if n < 40:
        flags.append(f"qua_ngan({n})")
    if n > 300:
        flags.append(f"qua_dai({n})")
    return flags


def stage_filter(args, llm: LLM) -> None:
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    rows = read_jsonl(WORK / "answered.jsonl")
    kept, dropped = [], []
    for n, r in enumerate(rows, 1):
        reasons = []
        for i, a in enumerate(r["answers"]):
            for f in rule_flags(a):
                reasons.append(f"lượt{i + 1}:{f}")
        # Chỉ gọi LLM kiểm tra lượt cuối (lượt được tính loss) khi luật đã qua
        if not reasons:
            i = len(r["turns"]) - 1
            msgs = _messages_for(system_prompt, r["turns"], r["answers"], r["contexts"], i)
            rendered = "\n\n".join(f"<{m['role'].upper()}>\n{m['content']}" for m in msgs)
            chk = llm.json(CHECK_SYSTEM, rendered + f"\n\n<CÂU TRẢ LỜI CẦN KIỂM TRA>\n{r['answers'][-1]}", "filter",
                           {"unsupported_claims": [], "unsafe": False, "misses_urgent": False,
                            "ignores_history": False, "answers_question": True, "role_ok": True})
            r["check"] = chk
            if len(chk.get("unsupported_claims") or []) > args.max_unsupported:
                reasons.append(f"thiếu căn cứ ({len(chk['unsupported_claims'])})")
            for k, bad in (("unsafe", True), ("misses_urgent", True), ("ignores_history", True),
                           ("answers_question", False), ("role_ok", False)):
                if bool(chk.get(k)) == bad:
                    reasons.append(k)
        (dropped if reasons else kept).append({**r, "drop_reasons": reasons})
        if n % 50 == 0:
            print(f"  {n}/{len(rows)}  ${llm.cost()}")
    write_jsonl(WORK / "filtered.jsonl", kept)
    write_jsonl(WORK / "filtered_dropped.jsonl", dropped)
    reasons = Counter(re.sub(r"^lượt\d:", "", x).split("(")[0] for d in dropped for x in d["drop_reasons"])
    print(f"Giữ {len(kept)} / loại {len(dropped)}")
    print("Lý do loại:", dict(reasons.most_common()))
    print("Category giữ lại:", dict(Counter(r["category"] for r in kept)))
    print(f"Chi phí: ${llm.cost()} | {dict(llm.usage)}")


# ---------------------------------------------------------------------------
# Bước 5: build
# ---------------------------------------------------------------------------
def _review_file(args) -> Path:
    return Path(args.review_file) if args.review_file else WORK / "review" / "sft_v2_review.xlsx"


def stage_review_export(args) -> None:
    from gout_llmops.core.hashing import sha256_file
    from gout_llmops.data.doctor_review import CATEGORY_VI, select_for_review, write_review_xlsx

    src = WORK / "filtered.jsonl"
    path = _review_file(args)
    if path.exists() and not args.force:
        raise SystemExit(f"{path} đã tồn tại (có thể đã chứa kết quả bác sĩ duyệt) — không ghi đè.\n"
                         "Muốn đọc kết quả: chạy review_import. Chắc chắn muốn xuất lại từ đầu: thêm --force.")
    rows = read_jsonl(src)
    full = tuple(c.strip() for c in (args.full_categories or "").split(",") if c.strip())
    unknown = [c for c in full if c not in CATEGORIES]
    if unknown:
        raise SystemExit(f"--full-categories có loại không tồn tại: {unknown} (hợp lệ: {', '.join(CATEGORIES)})")
    sent = select_for_review(rows, per_category=args.per_category, full_categories=full, seed=args.seed)
    selection = ("all" if args.per_category is None else
                 json.dumps({"per_category": args.per_category, "full_categories": list(full), "seed": args.seed}))
    n = write_review_xlsx(sent, path, source_file=str(src), source_sha256=sha256_file(src), selection=selection)
    cats_all, cats_sent = Counter(r["category"] for r in rows), Counter(r["category"] for r in sent)
    print(f"Đã xuất {n}/{len(rows)} mẫu cần duyệt -> {path}")
    for c in ("red_flag", "safety_trap", "clinical_reasoning", "out_of_scope", "basic_knowledge"):
        if cats_all.get(c):
            print(f"  {CATEGORY_VI[c]:<22} {cats_sent.get(c, 0):>4} / {cats_all[c]}")
    print(f"Ước tính thời gian bác sĩ: ~{n * 2 / 60:.0f}–{n * 3 / 60:.0f} giờ (2–3 phút/mẫu). "
          "File đã xếp rủi ro cao lên đầu.")
    if n < len(rows):
        print(f"{len(rows) - n} mẫu không gửi bác sĩ. Khi review_import, dùng --unreviewed keep để vẫn train các mẫu này "
              "(bản AI đã qua lọc tự động); mặc định drop sẽ bỏ chúng.")


def stage_review_import(args) -> None:
    from gout_llmops.core.hashing import sha256_file
    from gout_llmops.data.doctor_review import CATEGORY_VI, PRIORITY, apply_review, read_review_xlsx

    src = WORK / "filtered.jsonl"
    path = _review_file(args)
    if not path.exists():
        raise SystemExit(f"Không thấy file duyệt {path}. Chạy review_export trước, hoặc chỉ đường dẫn bằng --review-file.")
    records, meta = read_review_xlsx(path)
    src_sha = sha256_file(src)
    if meta.get("source_sha256") and meta["source_sha256"] != src_sha:
        msg = (f"{src} đã thay đổi sau khi xuất file duyệt (chạy lại answer/filter?). "
               "Câu trả lời trong file bác sĩ duyệt có thể không còn khớp dữ liệu hiện tại.")
        if not args.force:
            raise SystemExit(msg + " Thêm --force nếu chắc chắn vẫn muốn áp dụng.")
        print("[!] " + msg)
    if not meta:
        print("[!] File không còn sheet _meta (có thể do chuyển sang Google Sheets) — vẫn đối chiếu theo cột Mã.")

    res = apply_review(read_jsonl(src), records, unreviewed=args.unreviewed, exported_ids=meta.get("exported_ids"))
    st = res["stats"]
    print(f"Bác sĩ đã duyệt {st['n_reviewed']}/{st['n_sent_to_doctor']} mẫu được gửi (bộ dữ liệu có {st['n_source']}): "
          f"Đạt {st['n_approve']} | Cần sửa {st['n_edit']} | Loại {st['n_reject']}")
    if st["pass_rate"] is not None:
        print(f"Tỉ lệ Đạt trên các mẫu đã duyệt: {st['pass_rate']:.1%} | ước lượng cho toàn bộ dữ liệu "
              f"(trọng số theo số mẫu mỗi loại): {st['pass_rate_weighted']:.1%}")
    vi = {"approve": "Đạt", "edit": "Cần sửa", "reject": "Loại", "unreviewed": "Chưa duyệt/không gửi"}
    for cat in sorted(st["by_category"], key=lambda c: PRIORITY.get(c, 9)):
        c = st["by_category"][cat]
        rate = st["pass_rate_by_category"].get(cat)
        print(f"  {CATEGORY_VI.get(cat, cat):<22} " + " | ".join(f"{vi[k]} {c[k]}" for k in vi if c.get(k))
              + (f"   (Đạt {rate:.0%})" if rate is not None else ""))
    if st["error_types"]:
        print("Loại lỗi bác sĩ ghi:", st["error_types"])
    for w in res["warnings"]:
        print("[!]", w)
    if res["errors"]:
        print(f"\n{len(res['errors'])} lỗi trong file duyệt — sửa trong file rồi chạy lại:")
        for e in res["errors"]:
            print("  -", e)
        raise SystemExit(1)

    # Bản bác sĩ sửa không bị loại tự động, nhưng nêu ra nếu vi phạm luật lọc (vd. ghi liều cụ thể) để nhóm xem lại
    flagged = []
    for r in res["kept"]:
        if r.get("review", {}).get("verdict") == "edit":
            f = rule_flags(r["answers"][-1])
            if f:
                flagged.append((r["id"], f))
    for rid, f in flagged:
        print(f"[!] Bản sửa của {rid} vi phạm luật lọc {f} — vẫn giữ (ý kiến bác sĩ), nên xem lại.")

    if st["n_unreviewed"]:
        policy = ("BỎ, không dùng để train" if args.unreviewed == "drop"
                  else "GIỮ bản AI (--unreviewed keep), manifest sẽ ghi rõ")
        print(f"[!] {st['n_unreviewed']} mẫu chưa duyệt hoặc không gửi bác sĩ: {policy}.")
        if args.unreviewed == "drop" and st["n_sent_to_doctor"] < st["n_source"]:
            print(f"[!] File duyệt chỉ là mẫu ({st['n_sent_to_doctor']}/{st['n_source']}); với --unreviewed drop, "
                  f"dữ liệu train chỉ còn khoảng {st['n_approve'] + st['n_edit']} mẫu. Muốn giữ phần không gửi: --unreviewed keep.")
    if args.dry_run:
        print("(--dry-run: chỉ xem tiến độ, chưa ghi file)")
        return
    if not res["kept"]:
        raise SystemExit("Không còn mẫu nào để train (chưa duyệt mẫu nào?).")
    write_jsonl(WORK / "reviewed.jsonl", res["kept"])
    write_jsonl(WORK / "review_rejected.jsonl", res["rejected"])
    write_json(WORK / "review_manifest.json", {
        **st, "reviewer": args.reviewer, "review_file": str(path), "review_file_sha256": sha256_file(path),
        "source_file": str(src), "source_sha256": src_sha, "edited_rule_flags": {k: v for k, v in flagged},
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mock": bool(args.mock),
    })
    print(f"Đã ghi {WORK / 'reviewed.jsonl'} ({len(res['kept'])} mẫu) + review_manifest.json")


def _rows_for_build(args) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """Dữ liệu đưa vào build: bản bác sĩ đã duyệt; chỉ dùng bản chưa duyệt khi --skip-review (hoặc --mock)."""
    from gout_llmops.core.hashing import sha256_file

    reviewed, review_manifest, src = WORK / "reviewed.jsonl", WORK / "review_manifest.json", WORK / "filtered.jsonl"
    if reviewed.exists() and not args.skip_review:
        review = json.loads(review_manifest.read_text(encoding="utf-8")) if review_manifest.exists() else {}
        if review.get("source_sha256") and src.exists() and review["source_sha256"] != sha256_file(src):
            raise SystemExit(f"{src} đã thay đổi sau lần review_import gần nhất — chạy lại review_import.")
        return read_jsonl(reviewed), review, str(reviewed)
    if args.skip_review or args.mock:
        return read_jsonl(src), None, str(src)
    raise SystemExit("Chưa có kết quả bác sĩ duyệt (artifacts/sft_v2/reviewed.jsonl).\n"
                     "Chạy: review_export -> bác sĩ duyệt file Excel -> review_import, rồi mới build.\n"
                     "Cố ý train dữ liệu chưa duyệt: thêm --skip-review.")


def stage_build(args) -> None:
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    rows, review, data_file = _rows_for_build(args)
    rng = random.Random(args.seed)
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    train, val = [], []
    for cat in sorted(by_cat):
        g = sorted(by_cat[cat], key=lambda x: x["id"])
        rng.shuffle(g)
        k = max(1, round(len(g) * args.val_ratio)) if len(g) >= 5 else 0
        val += g[:k]
        train += g[k:]

    def to_sample(r):
        i = len(r["turns"]) - 1
        msgs = _messages_for(system_prompt, r["turns"], r["answers"], r["contexts"], i)
        return {"id": r["id"], "source": r["source"], "category": r["category"],
                "messages": msgs + [{"role": "assistant", "content": r["answers"][-1]}]}

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "train.jsonl", [to_sample(r) for r in train])
    write_jsonl(OUT / "val.jsonl", [to_sample(r) for r in val])
    words = [len(r["answers"][-1].split()) for r in rows]
    manifest = {
        "version": "sft_v2", "seed": args.seed, "n_train": len(train), "n_val": len(val),
        "by_category": {"train": dict(Counter(r["category"] for r in train)),
                        "val": dict(Counter(r["category"] for r in val))},
        "by_source": dict(Counter(r["source"] for r in rows)),
        "multi_turn": sum(1 for r in rows if len(r["turns"]) > 1),
        "answer_words_median": sorted(words)[len(words) // 2] if words else 0,
        "format": "build_messages(system_vi_v1 + NGỮ CẢNH RAG, history<=2 cặp) + assistant — giống hệt lúc suy luận",
        "leakage_guard": "4-gram Jaccard < 0.40 với mọi lượt của testset v03",
        "mock": bool(args.mock),
        "data_file": data_file,
        "doctor_review": ({k: review.get(k) for k in ("rule_version", "reviewer", "n_source", "n_sent_to_doctor",
                                                     "n_reviewed", "n_approve", "n_edit", "n_reject", "n_unreviewed",
                                                     "unreviewed_policy", "pass_rate", "pass_rate_weighted",
                                                     "pass_rate_by_category",
                                                     "error_types", "review_file_sha256", "imported_at")}
                          if review is not None else "skipped"),
        "note": ("Câu trả lời do gpt-4.1-mini viết từ context KB, lọc tự động, rồi bác sĩ duyệt"
                 + (f" ({review.get('n_reviewed')}/{review.get('n_source')} mẫu; "
                    f"{review.get('n_unreviewed')} mẫu chưa duyệt: {review.get('unreviewed_policy')})."
                    if review else ".")
                 if review is not None else
                 "Câu trả lời do gpt-4.1-mini viết từ context KB và lọc tự động; CHƯA qua bác sĩ duyệt."),
    }
    for f in ("train.jsonl", "val.jsonl"):
        manifest[f"sha256_{f}"] = hashlib.sha256((OUT / f).read_bytes()).hexdigest()
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("n_train", "n_val", "by_category", "multi_turn",
                                               "answer_words_median", "data_file", "mock")}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["questions", "retrieve", "answer", "filter", "review_export", "review_import",
                                      "build"])
    ap.add_argument("--mock", action="store_true", help="không gọi API/không cần GPU — chỉ để thử pipeline")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--testset", default="data/testset_v03/cases_all.jsonl")
    ap.add_argument("--kb-chunks", default="artifacts/kb/gout_kb_v1/chunks.jsonl")
    ap.add_argument("--rag-config", default="configs/rag/rag_v1.yaml")
    ap.add_argument("--q-per-chunk", type=int, default=4)
    ap.add_argument("--n-dialogs", type=int, default=80)
    ap.add_argument("--max-unsupported", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=None)
    rv = ap.add_argument_group("bác sĩ duyệt (review_export / review_import / build)")
    rv.add_argument("--review-file", default=None, help="mặc định artifacts/sft_v2/review/sft_v2_review.xlsx")
    rv.add_argument("--per-category", type=int, default=None,
                    help="review_export: chỉ gửi bác sĩ tối đa N mẫu ngẫu nhiên mỗi loại câu hỏi (mặc định: gửi tất cả)")
    rv.add_argument("--full-categories", default="",
                    help="review_export: các loại gửi TOÀN BỘ dù có --per-category, vd. red_flag,safety_trap")
    rv.add_argument("--reviewer", default="", help="tên/chức danh bác sĩ duyệt, ghi vào manifest")
    rv.add_argument("--unreviewed", choices=["drop", "keep"], default="drop",
                    help="mẫu bác sĩ chưa duyệt: drop = bỏ (mặc định), keep = giữ bản AI")
    rv.add_argument("--dry-run", action="store_true", help="review_import: chỉ xem tiến độ, không ghi file")
    rv.add_argument("--force", action="store_true",
                    help="review_export: ghi đè file duyệt; review_import: bỏ qua kiểm tra dữ liệu gốc đã đổi")
    rv.add_argument("--skip-review", action="store_true", help="build: cố ý dùng dữ liệu CHƯA duyệt")
    args = ap.parse_args()
    if args.mock:
        global WORK, OUT, CACHE
        WORK, OUT = Path("artifacts/sft_v2_mock"), Path("artifacts/sft_v2_mock/data")
        CACHE = WORK / "llm_cache.jsonl"
    WORK.mkdir(parents=True, exist_ok=True)
    if args.stage == "retrieve":
        stage_retrieve(args)
    elif args.stage == "build":
        stage_build(args)
    elif args.stage == "review_export":
        stage_review_export(args)
    elif args.stage == "review_import":
        stage_review_import(args)
    else:
        llm = LLM(args.model, args.mock)
        {"questions": stage_questions, "answer": stage_answer, "filter": stage_filter}[args.stage](args, llm)


if __name__ == "__main__":
    main()
