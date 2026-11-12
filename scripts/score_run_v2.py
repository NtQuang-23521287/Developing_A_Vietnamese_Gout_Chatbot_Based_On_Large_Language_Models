"""Chấm lại predictions của một run bằng Judge v2 + Reward v2 (đề cương), trên testset v03.

Ví dụ:
  # thử pipeline, không tốn tiền
  PYTHONPATH=src python3 scripts/score_run_v2.py --run artifacts/runs/<run_id> --mock --limit 10
  # chấm thật 5 case trước để xem
  PYTHONPATH=src python3 scripts/score_run_v2.py --run artifacts/runs/<run_id> --limit 5
  # chấm hết
  PYTHONPATH=src python3 scripts/score_run_v2.py --run artifacts/runs/<run_id>

Kết quả: <run>/eval_v2/scores.jsonl và <run>/eval_v2/summary.json
Chỉ case có trong cả run và testset mới được chấm (run baseline cũ: 299 câu Group3).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import read_jsonl, write_json, write_jsonl
from gout_llmops.data.testset import load_cases
from gout_llmops.evaluation.judge_v2 import (
    JUDGE_VERSION,
    MockJudge,
    OpenAIJudgeV2,
    build_judge_input,
    validate_judge_output,
)
from gout_llmops.evaluation.reward_v2 import aggregate, compute_reward_v2, validate_config

PRICE_PER_M = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60)}  # USD / 1M token (input, output)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="thư mục run có predictions.jsonl")
    ap.add_argument("--testset", default="data/testset_v03/cases_all.jsonl")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all")
    ap.add_argument("--reward-config", default="configs/evaluation/heuristic_reward_v2.yaml")
    ap.add_argument("--judge-model", default="gpt-4.1-mini")
    ap.add_argument("--mock", action="store_true", help="judge giả lập, không gọi API")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--case-ids", nargs="+", default=None, help="chỉ chấm các case này")
    ap.add_argument("--out-name", default="eval_v2")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load(Path(args.reward_config).read_text(encoding="utf-8"))
    validate_config(cfg)

    cases = load_cases(args.testset, split=None if args.split == "all" else args.split)
    preds = defaultdict(list)
    for p in read_jsonl(run / "predictions.jsonl"):
        preds[p["case_id"]].append(p)

    judge = MockJudge() if args.mock else OpenAIJudgeV2(model=args.judge_model)
    todo = [c for c in cases if c.case_id in preds]
    missing = [c.case_id for c in cases if c.case_id not in preds]
    if args.case_ids:
        todo = [c for c in todo if c.case_id in set(args.case_ids)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Testset {len(cases)} case | có trong run: {len(cases) - len(missing)} | sẽ chấm: {len(todo)}")

    rows = []
    for i, case in enumerate(todo, 1):
        turns = sorted(preds[case.case_id], key=lambda p: int(p["turn_id"]))
        base = {"case_id": case.case_id, "scenario": case.scenario, "question_type": case.question_type,
                "risk_level": case.risk_level, "risk_weight": case.risk_weight, "split": case.split}
        last = turns[-1]
        if last.get("status") != "ok" or len(turns) != len(case.turns):
            rows.append({**base, "status": "prediction_missing_or_error"})
            continue
        history = [(t["question"], t["answer"]) for t in turns[:-1]]
        contexts = last.get("contexts") or []
        ctx_ok = bool(contexts) or case.scenario == "multi"
        user_input = build_judge_input(
            question_type=case.question_type, risk_level=case.risk_level,
            expected_behavior=case.expected_behavior, history=history,
            question=last["question"], answer=last["answer"], contexts=contexts, reference=case.reference)
        try:
            ref = case.reference
            raw = validate_judge_output(judge.score(user_input), question_type=case.question_type,
                                        context_applicable=ctx_ok, answer=last["answer"],
                                        n_points=len(ref.get("required_points", [])),
                                        n_warnings=len(ref.get("mandatory_warnings", [])),
                                        n_forbidden=len(ref.get("forbidden_content", [])),
                                        forbidden_texts=ref.get("forbidden_content", []))
            res = compute_reward_v2(raw, question_type=case.question_type, risk_level=case.risk_level,
                                    has_context=bool(contexts), multi_turn=case.scenario == "multi", config=cfg)
            rows.append({**base, "status": "ok", "raw": raw, "result": res, "answer": last["answer"],
                         "answer_tokens_approx": len(last["answer"].split())})
        except Exception as e:  # noqa: BLE001
            rows.append({**base, "status": "judge_error", "error": str(e)[:300]})
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}  (gọi API {judge.usage['calls']}, cache {judge.usage['cached']})")

    ok = [r for r in rows if r["status"] == "ok"]
    summary = aggregate(ok, cfg)
    pin, pout = PRICE_PER_M.get(args.judge_model, (0, 0))
    u = judge.usage
    summary.update({
        "run": run.name, "judge": "MOCK" if args.mock else args.judge_model, "judge_version": JUDGE_VERSION,
        "reward_version": cfg["version"], "testset": args.testset, "split": args.split,
        "testset_sha256": sha256_file(args.testset), "n_cases_in_testset": len(cases),
        "n_not_in_run": len(missing), "n_errors": len(rows) - len(ok),
        "judge_usage": u,
        "judge_cost_usd_est": round(u["input_tokens"] / 1e6 * pin + u["output_tokens"] / 1e6 * pout, 4),
        "warning": "MOCK JUDGE — không dùng cho kết quả" if args.mock else None,
    })
    out = run / args.out_name
    write_jsonl(out / "scores.jsonl", rows)
    write_json(out / "summary.json", summary)

    print(f"\nĐã chấm {len(ok)} case, lỗi {len(rows) - len(ok)} -> {out}")
    print(f"Reward (TB có trọng số risk): {summary.get('reward_weighted_mean'):.4f}  "
          f"CI95 {summary.get('reward_weighted_ci95')}")
    print("Rate có trọng số:", {k: round(v, 3) for k, v in summary.get("rates_weighted", {}).items()})
    print(f"Fatal: {summary.get('fatal_count')} | Chi phí judge ước tính: ${summary['judge_cost_usd_est']}")


if __name__ == "__main__":
    main()
