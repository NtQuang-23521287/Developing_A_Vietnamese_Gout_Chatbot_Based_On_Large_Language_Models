"""Giai đoạn 5 — Độ bất định: phần thuần (không torch) để unit test.

MC Dropout trên LoRA (phần GPU ở safety/mc_dropout.py) chấm lại CÂU TRẢ LỜI ĐÃ SINH y dưới N mặt nạ dropout
khác nhau, thay vì sinh lại N lần: rẻ hơn ~N×(độ dài câu trả lời) lần và đo đúng độ bất định của chính câu
người dùng nhận được. Với mỗi lượt chấm có:
  token_logprobs[t] = log p(y_t | x, y_<t)    và    token_entropy[t] = H(p(· | x, y_<t))

Các tín hiệu (giá trị lớn = bất định hơn):
  det_mean_nll       NLL trung bình/token, dropout TẮT (mốc so sánh, không cần MC)
  det_mean_entropy   entropy trung bình/token, dropout TẮT
  det_max_nll        NLL của token tệ nhất, dropout TẮT
  mc_mean_nll        NLL trung bình/token, lấy trung bình trên N lượt dropout
  mc_nll_std         độ lệch chuẩn (giữa N lượt) của NLL trung bình/token — bất định tri thức (epistemic)
  mc_token_lp_var    phương sai log p(y_t) giữa N lượt, trung bình theo token
  mc_mutual_info     MI trung bình/token = H(p̄) − mean_i H(p_i)  (BALD; tính trên GPU, truyền vào đây)
  ret_low_relevance  1 − điểm truy xuất cao nhất của context (không cần GPU)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from gout_llmops.chat.prompt_builder import build_messages

SIGNALS_DET = ("det_mean_nll", "det_mean_entropy", "det_max_nll")
SIGNALS_MC = ("mc_mean_nll", "mc_nll_std", "mc_token_lp_var", "mc_mutual_info")
SIGNALS_FREE = ("ret_low_relevance",)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def summarize_deterministic(token_logprobs: list[float], token_entropy: list[float]) -> dict[str, float]:
    if not token_logprobs:
        raise ValueError("câu trả lời rỗng: không có token để chấm")
    nll = [-x for x in token_logprobs]
    return {"det_mean_nll": _mean(nll), "det_mean_entropy": _mean(token_entropy), "det_max_nll": max(nll)}


def summarize_mc(passes_logprobs: list[list[float]], mutual_info_per_token: list[float] | None = None
                 ) -> dict[str, float]:
    """passes_logprobs[i][t]: log p(y_t) ở lượt dropout i. Mọi lượt phải cùng độ dài."""
    if not passes_logprobs:
        return {}
    T = len(passes_logprobs[0])
    if T == 0 or any(len(p) != T for p in passes_logprobs):
        raise ValueError("các lượt MC phải cùng số token và > 0")
    seq_nll = [-_mean(p) for p in passes_logprobs]
    per_tok_var = [_std([p[t] for p in passes_logprobs]) ** 2 for t in range(T)]
    out = {"mc_mean_nll": _mean(seq_nll), "mc_nll_std": _std(seq_nll), "mc_token_lp_var": _mean(per_tok_var)}
    if mutual_info_per_token is not None:
        out["mc_mutual_info"] = _mean(mutual_info_per_token)
    return out


def retrieval_signal(contexts: list[dict[str, Any]] | None) -> float | None:
    """1 − điểm liên quan cao nhất (ưu tiên rerank_score nếu có, nếu không dùng retrieval_score cosine)."""
    if not contexts:
        return None
    key = "rerank_score" if any("rerank_score" in c for c in contexts) else "retrieval_score"
    vals = [float(c[key]) for c in contexts if c.get(key) is not None]
    return 1.0 - max(vals) if vals else None


def reconstruct_messages(case_turns: list[dict[str, Any]], system_prompt: str,
                         max_history_pairs: int) -> list[list[dict[str, str]]]:
    """Dựng lại messages đúng như runner đã đưa vào model cho từng lượt của một case.

    case_turns: các dòng predictions.jsonl của cùng một case (status ok), mỗi dòng có turn_id, question,
    answer, contexts. History của lượt k gồm (question, answer) THẬT của model ở các lượt trước — giống runner.
    """
    turns = sorted(case_turns, key=lambda p: int(p["turn_id"]))
    out, history = [], []
    for t in turns:
        out.append(build_messages(system_prompt=system_prompt, question=t["question"], history=history,
                                  contexts=t.get("contexts") or [], max_history_pairs=max_history_pairs))
        history = [*history, {"role": "user", "content": t["question"]},
                   {"role": "assistant", "content": t["answer"]}]
    return out


def load_calibration(path: str | Path | None) -> dict[str, Any] | None:
    """Đọc file do `safe_inference.py calibrate` ghi. None nếu chưa hiệu chỉnh."""
    if not path:
        return None
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in ("signal", "threshold", "enabled"):
        if k not in d:
            raise ValueError(f"{path}: thiếu khóa {k}")
    return d


def require_calibrated_threshold(config: dict) -> float:
    """Dùng khi phát hành: bắt buộc đã có ngưỡng hiệu chỉnh trên dev."""
    unc = config.get("uncertainty", {})
    cal = load_calibration(unc.get("calibration_file"))
    threshold = cal["threshold"] if cal and cal.get("enabled") else unc.get("threshold")
    if threshold is None:
        raise RuntimeError("Uncertainty threshold must be calibrated on development data before release.")
    return float(threshold)
