"""Giai đoạn 4 — DPO: sinh ứng viên -> chấm Heuristic Reward v2 -> Preference Mining -> train DPO.

  generate : policy (Qwen gốc hoặc Qwen + adapter SFT) sinh K câu trả lời cho mỗi câu hỏi, có RAG, sampling.
             Ghi từng câu ngay khi xong -> Colab ngắt thì chạy lại lệnh, nó làm tiếp phần còn thiếu.
  score    : Judge v2 chấm từng ứng viên, Reward v2 tính R(x,c,y) theo công thức (1)–(3) của đề cương.
  mine     : chọn cặp Chosen/Rejected theo Preference Mining Rule (src/gout_llmops/training/preference.py).
  train    : DPO bằng TRL + QLoRA; reference = chính policy trước khi train.

Câu hỏi lấy từ artifacts/sft_v2/reviewed.jsonl (output bước `review_import` của build_sft_v2.py): đã chống leakage
với testset v03, context RAG truy xuất bằng đúng retriever lúc suy luận, câu trả lời mẫu đã được BÁC SĨ duyệt/sửa làm
REFERENCE cho judge (mẫu bác sĩ loại không được dùng).
KHÔNG dùng câu hỏi dev/test cho DPO.

Ví dụ (Colab, từ thư mục gốc repo):
  python scripts/dpo_pipeline.py generate --policy-adapter artifacts/runs/sft_qwen3_8b_v3 --k 4 --max-prompts 300
  python scripts/dpo_pipeline.py score
  python scripts/dpo_pipeline.py mine
  python scripts/dpo_pipeline.py train --config configs/training/dpo_v1.yaml --smoke-test
  python scripts/dpo_pipeline.py train --config configs/training/dpo_v1.yaml
Thử toàn bộ luồng không cần GPU/API: thêm --mock vào generate/score/mine (ghi vào artifacts/dpo_v1_mock).
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml

from gout_llmops.core.config import load_yaml
from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import read_jsonl, write_json, write_jsonl
from gout_llmops.evaluation.judge_v2 import (JUDGE_VERSION, MockJudge, OpenAIJudgeV2, build_judge_input,
                                             validate_judge_output)
from gout_llmops.evaluation.reward_v2 import compute_reward_v2, validate_config
from gout_llmops.training.dpo import check_adapter_dir, resolve_policy
from gout_llmops.training.preference import (DEFAULT_RULE, RULE_VERSION, labels_for, mine_pair, prompt_messages,
                                             select_prompts)

SYSTEM_PROMPT_PATH = "prompts/system_vi_v1.txt"
MODEL_CONFIG = "configs/models/qwen3_8b.yaml"
EVAL_PROTOCOL = "configs/evaluation/baseline_v03_381.yaml"
PRICE_PER_M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60)}


class Paths:
    def __init__(self, mock: bool):
        self.work = Path("artifacts/dpo_v1_mock" if mock else "artifacts/dpo_v1")
        self.pref_dir = self.work / "data" if mock else Path("data/dpo")
        self.cands = self.work / "candidates.jsonl"
        self.meta = self.work / "candidates_meta.json"
        self.scored = self.work / "scored.jsonl"
        self.pref = self.pref_dir / "preference_v1.jsonl"
        self.manifest = self.pref_dir / "manifest.json"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def _mock_candidates(row: dict[str, Any], k: int) -> list[dict[str, Any]]:
    teacher = row["answers"][-1]
    pool = [teacher, teacher[: max(40, len(teacher) // 3)],
            "Bạn cứ tự tăng gấp đôi liều thuốc đang uống, 500 mg mỗi ngày là ổn, không cần đi khám.",
            teacher + " Bạn nên đi khám bác sĩ chuyên khoa Cơ xương khớp."]
    return [{"idx": i, "kind": "mock", "text": pool[i % len(pool)], "output_tokens": len(pool[i % len(pool)].split()),
             "truncated": False} for i in range(k)]


def stage_generate(args, P: Paths) -> None:
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    protocol = load_yaml(EVAL_PROTOCOL)
    max_input = int(protocol["input"]["max_input_tokens"])
    max_new = int(args.max_new_tokens or protocol["generation"]["max_new_tokens"])
    policy = resolve_policy(args.policy_adapter)
    if policy and not args.mock:
        check_adapter_dir(policy)   # báo lỗi sớm (thiếu adapter / chỉ là con trỏ LFS) trước khi nạp model 8B

    if not Path(args.prompts).exists():
        raise SystemExit(f"Không thấy {args.prompts}. Chạy build_sft_v2.py review_import trước "
                         "(hoặc --prompts artifacts/sft_v2/filtered.jsonl nếu cố ý dùng dữ liệu chưa duyệt).")
    rows = [r for r in read_jsonl(args.prompts) if r.get("answers") and r.get("contexts")]
    selected = select_prompts(rows, args.max_prompts, args.seed)
    meta = {"policy_adapter": policy, "model_config": MODEL_CONFIG, "prompts_file": args.prompts,
            "prompts_sha256": sha256_file(args.prompts), "n_prompts_selected": len(selected),
            "k": args.k, "include_greedy": args.include_greedy, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "max_new_tokens": max_new, "seed": args.seed,
            "mock": bool(args.mock)}
    if P.meta.exists():
        old = json.loads(P.meta.read_text(encoding="utf-8"))
        keys = ("policy_adapter", "prompts_sha256", "k", "include_greedy", "temperature", "top_p", "top_k",
                "max_new_tokens", "seed")
        diff = {k: (old.get(k), meta[k]) for k in keys if old.get(k) != meta[k]}
        if diff and not args.restart:
            raise SystemExit(f"{P.cands} đã có ứng viên với cấu hình khác {diff}. Thêm --restart để sinh lại từ đầu.")
    if args.restart:
        for p in (P.cands, P.meta, P.scored):
            p.unlink(missing_ok=True)
    write_json(P.meta, meta)
    done = {r["id"] for r in read_jsonl(P.cands)} if P.cands.exists() else set()
    todo = [r for r in selected if r["id"] not in done]
    print(f"Câu hỏi: {len(rows)} hợp lệ | chọn {len(selected)} | đã xong {len(done)} | còn {len(todo)} | "
          f"policy = {policy or 'Qwen gốc'}")
    print("Theo category:", dict(Counter(r["category"] for r in selected)))
    if not todo:
        return

    def record(r, cands, n_in):
        qt, rl, eb = labels_for(r["category"])
        return {"id": r["id"], "source": r.get("source"), "category": r["category"], "question_type": qt,
                "risk_level": rl, "expected_behavior": eb, "turns": r["turns"], "teacher_answers": r["answers"],
                "final_contexts": r["contexts"][-1], "input_tokens": n_in, "candidates": cands}

    if args.mock:
        for r in todo:
            _append_jsonl(P.cands, record(r, _mock_candidates(r, args.k), None))
        print(f"[MOCK] Đã ghi {len(todo)} câu -> {P.cands}")
        return

    import torch

    from gout_llmops.models.registry import build_backend

    model_cfg = load_yaml(MODEL_CONFIG)
    model_cfg["name"] = "dpo_candidate_policy"
    if policy:
        model_cfg["adapter_path"] = policy
    backend = build_backend(model_cfg)
    tok, model = backend.tokenizer, backend.model
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    chat_kwargs = dict(model_cfg.get("chat_template_kwargs") or {})
    n_sampled = args.k - (1 if args.include_greedy else 0)

    def decode(ids_row) -> dict[str, Any]:
        ids = ids_row.tolist()
        n_out = sum(1 for t in ids if t != pad_id)
        return {"text": tok.decode(ids_row, skip_special_tokens=True).strip(), "output_tokens": n_out,
                "truncated": not any(t in eos_ids for t in ids)}

    t0, n_done = time.time(), 0
    for b in range(0, len(todo), args.batch_prompts):
        batch = todo[b: b + args.batch_prompts]
        texts = [tok.apply_chat_template(prompt_messages(system_prompt, r["turns"], r["answers"], r["contexts"][-1]),
                                         tokenize=False, add_generation_prompt=True, **chat_kwargs) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        n_in = enc["attention_mask"].sum(dim=1).tolist()
        if max(n_in) > max_input:
            raise ValueError(f"Prompt {max(n_in)} token > max_input_tokens={max_input} (không cắt âm thầm)")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        L = enc["input_ids"].shape[1]
        cands: list[list[dict[str, Any]]] = [[] for _ in batch]
        with torch.inference_mode():
            if args.include_greedy:
                out = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=pad_id)
                for j in range(len(batch)):
                    cands[j].append({"idx": 0, "kind": "greedy", **decode(out[j, L:])})
            if n_sampled > 0:
                torch.manual_seed(args.seed + b)
                out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                     top_k=args.top_k, max_new_tokens=max_new, num_return_sequences=n_sampled,
                                     pad_token_id=pad_id)
                for j in range(len(batch)):
                    for s in range(n_sampled):
                        cands[j].append({"idx": len(cands[j]), "kind": "sample",
                                         **decode(out[j * n_sampled + s, L:])})
        for j, r in enumerate(batch):
            _append_jsonl(P.cands, record(r, cands[j], int(n_in[j])))
        n_done += len(batch)
        el = time.time() - t0
        print(f"  {len(done) + n_done}/{len(selected)} | {el / n_done:.1f}s/câu | còn ~{el / n_done * (len(todo) - n_done) / 60:.0f} phút",
              flush=True)
    print(f"Xong -> {P.cands}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------
def stage_score(args, P: Paths) -> None:
    rows = read_jsonl(P.cands)
    cfg = yaml.safe_load(Path(args.reward_config).read_text(encoding="utf-8"))
    validate_config(cfg)
    if args.mock:
        judge = MockJudge()
    else:
        judge = OpenAIJudgeV2(model=args.judge_model, cache_path="artifacts/judge_cache/judge_v2_dpo.jsonl")
        lock, put = threading.Lock(), judge.cache.put

        def locked_put(k, v):
            with lock:
                put(k, v)
        judge.cache.put = locked_put

    jobs = []
    for r in rows:
        history = list(zip(r["turns"][:-1], r["teacher_answers"][:-1]))
        reference = {"reference_answer": r["teacher_answers"][-1]}
        for c in r["candidates"]:
            ui = build_judge_input(question_type=r["question_type"], risk_level=r["risk_level"],
                                   expected_behavior=r["expected_behavior"], history=history,
                                   question=r["turns"][-1], answer=c["text"], contexts=r["final_contexts"],
                                   reference=reference)
            jobs.append((r, c, ui))
    print(f"Chấm {len(jobs)} ứng viên của {len(rows)} câu (judge {'MOCK' if args.mock else args.judge_model}, "
          f"{args.workers} luồng; câu đã chấm trước đó lấy từ cache, không tốn tiền)")

    def run(job):
        r, c, ui = job
        if not c["text"].strip():
            return {**c, "status": "empty_answer"}
        try:
            multi = len(r["turns"]) > 1
            raw = validate_judge_output(judge.score(ui), question_type=r["question_type"],
                                        context_applicable=bool(r["final_contexts"]) or multi, answer=c["text"],
                                        n_points=0, n_warnings=0, n_forbidden=0, forbidden_texts=[])
            res = compute_reward_v2(raw, question_type=r["question_type"], risk_level=r["risk_level"],
                                    has_context=bool(r["final_contexts"]), multi_turn=multi, config=cfg)
            return {**c, "status": "ok", "raw": raw, "reward": res["reward"], "fatal": res["fatal"],
                    "quality": res["quality_component"], "penalty": res["risk_penalty"]}
        except Exception as e:  # noqa: BLE001
            return {**c, "status": "judge_error", "error": str(e)[:300]}

    with ThreadPoolExecutor(max_workers=1 if args.mock else args.workers) as pool:
        results = []
        for i, res in enumerate(pool.map(run, jobs), 1):
            results.append(res)
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    it = iter(results)
    out = [{**r, "candidates": [next(it) for _ in r["candidates"]]} for r in rows]
    write_jsonl(P.scored, out)

    flat = [c for r in out for c in r["candidates"]]
    ok = [c for c in flat if c["status"] == "ok"]
    u = judge.usage
    pin, pout = PRICE_PER_M.get(args.judge_model, (0, 0))
    print(f"\nĐã chấm {len(ok)}/{len(flat)} | lỗi {len(flat) - len(ok)} -> {P.scored}")
    if ok:
        print(f"Reward TB {mean(c['reward'] for c in ok):.3f} | đạt (≥0.5) {sum(c['reward'] >= 0.5 for c in ok) / len(ok):.1%}"
              f" | fatal {sum(c['fatal'] for c in ok)} | cắt cụt {sum(c.get('truncated', False) for c in ok)}")
    print(f"Judge: {u} | chi phí ước tính ${u['input_tokens'] / 1e6 * pin + u['output_tokens'] / 1e6 * pout:.3f}")
    if len(ok) < len(flat):
        print("Có lỗi judge -> chạy lại `score` (phần đã chấm lấy từ cache).")


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------
def stage_mine(args, P: Paths) -> None:
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    rows = read_jsonl(P.scored)
    gen_meta = json.loads(P.meta.read_text(encoding="utf-8"))
    rule = {**DEFAULT_RULE, "min_chosen_reward": args.min_chosen_reward, "min_margin": args.min_margin,
            "max_chosen_risk": args.max_chosen_risk, "teacher_fallback": args.teacher_fallback}
    pairs, reasons = [], Counter()
    for r in rows:
        pair, reason = mine_pair(r, rule)
        reasons[reason] += 1
        if pair is None:
            continue
        pairs.append({"id": r["id"], "category": r["category"], "risk_level": r["risk_level"],
                      "prompt": prompt_messages(system_prompt, r["turns"], r["teacher_answers"], r["final_contexts"]),
                      **pair})
    write_jsonl(P.pref, pairs)
    words = lambda key: [len(p[key].split()) for p in pairs]  # noqa: E731
    manifest = {
        "version": "preference_v1", "rule_version": RULE_VERSION, "rule": rule,
        "n_prompts": len(rows), "n_pairs": len(pairs), "selection_reasons": dict(reasons),
        "by_source": dict(Counter(p["source"] for p in pairs)),
        "by_category": dict(Counter(p["category"] for p in pairs)),
        "chosen_words_median": median(words("chosen")) if pairs else None,
        "rejected_words_median": median(words("rejected")) if pairs else None,
        "margin_mean_on_policy": mean([p["margin"] for p in pairs if p["margin"] is not None] or [0]),
        "policy_adapter": gen_meta.get("policy_adapter"), "generation": gen_meta,
        "judge_version": JUDGE_VERSION, "reward_config": args.reward_config,
        "sha256_preference": sha256_file(P.pref), "sha256_scored": sha256_file(P.scored),
        "mock": bool(args.mock),
        "note": "Chosen/Rejected tạo tự động bằng Heuristic Reward v2 (judge LLM). Chưa qua chuyên gia y khoa.",
    }
    write_json(P.manifest, manifest)
    print(json.dumps({k: manifest[k] for k in ("n_prompts", "n_pairs", "selection_reasons", "by_source", "by_category",
                                               "chosen_words_median", "rejected_words_median")},
                     ensure_ascii=False, indent=1))
    if pairs and manifest["chosen_words_median"] > 1.3 * manifest["rejected_words_median"]:
        print("[WARN] Chosen dài hơn rejected rõ rệt -> DPO dễ học thói dài dòng; cân nhắc thêm ràng buộc độ dài.")
    if len(pairs) < 150:
        print(f"[WARN] Chỉ {len(pairs)} cặp. Nên tăng --max-prompts / --k, hoặc hạ --min-margin xuống 0.10.")
    print(f"-> {P.pref}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--policy-adapter", default="none", help="thư mục adapter SFT, hoặc none = Qwen gốc")
    g.add_argument("--prompts", default="artifacts/sft_v2/reviewed.jsonl")
    g.add_argument("--k", type=int, default=4, help="số ứng viên mỗi câu (đề cương: 3–5)")
    g.add_argument("--max-prompts", type=int, default=300)
    g.add_argument("--batch-prompts", type=int, default=2, help="số câu hỏi sinh song song (×k chuỗi); L4/A100 dùng 4")
    g.add_argument("--include-greedy", action="store_true", help="1 trong k ứng viên là câu greedy (giống lúc đánh giá)")
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=50)
    g.add_argument("--max-new-tokens", type=int, default=None, help="mặc định = protocol đánh giá (512)")
    g.add_argument("--seed", type=int, default=2026)
    g.add_argument("--restart", action="store_true", help="xóa ứng viên cũ và sinh lại")
    g.add_argument("--mock", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--judge-model", default="gpt-4.1-mini")
    s.add_argument("--reward-config", default="configs/evaluation/heuristic_reward_v2.yaml")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--mock", action="store_true")
    m = sub.add_parser("mine")
    m.add_argument("--min-chosen-reward", type=float, default=DEFAULT_RULE["min_chosen_reward"])
    m.add_argument("--min-margin", type=float, default=DEFAULT_RULE["min_margin"])
    m.add_argument("--max-chosen-risk", type=int, default=DEFAULT_RULE["max_chosen_risk"])
    m.add_argument("--teacher-fallback", action=argparse.BooleanOptionalAction, default=DEFAULT_RULE["teacher_fallback"])
    m.add_argument("--reward-config", default="configs/evaluation/heuristic_reward_v2.yaml")
    m.add_argument("--mock", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/training/dpo_v1.yaml")
    t.add_argument("--policy-adapter", default=None, help="ghi đè policy_adapter trong config (none = Qwen gốc)")
    t.add_argument("--smoke-test", action="store_true", help="2 bước, không lưu adapter")
    t.add_argument("--allow-policy-mismatch", action="store_true")
    args = ap.parse_args()

    if args.stage == "train":
        from gout_llmops.training.dpo import train_dpo
        cfg = load_yaml(args.config)
        train_dpo(cfg, smoke_test=args.smoke_test,
                  policy_adapter="from_config" if args.policy_adapter is None else args.policy_adapter,
                  allow_policy_mismatch=args.allow_policy_mismatch)
        return
    P = Paths(args.mock)
    P.work.mkdir(parents=True, exist_ok=True)
    {"generate": stage_generate, "score": stage_score, "mine": stage_mine}[args.stage](args, P)


if __name__ == "__main__":
    main()
