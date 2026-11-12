"""Giai đoạn 5 — Safe Inference: guard theo luật + MC Dropout trên LoRA + hiệu chỉnh ngưỡng + áp dụng lên một run.

  guard-eval  (CPU) đo guard trên câu hỏi SFT v2 + split dev (split test chỉ khi --final-test)
  uncertainty (GPU) chấm độ bất định câu trả lời CUỐI mỗi case của một run -> <run>/safety/uncertainty.jsonl
  calibrate   (CPU) ghép độ bất định với điểm Judge v2 trên dev -> AUROC, risk–coverage, ngưỡng τ
  apply       (CPU) tạo run mới <run>__<tag>: câu trả lời đã qua lớp an toàn (chấm bằng eval_adapter.py score)
  report      (CPU) so sánh run gốc và run an toàn, tách theo hành động của lớp an toàn

Lớp an toàn đứng SAU model nên mọi bước trừ `uncertainty` chạy trên CPU từ một run có sẵn, không sinh lại.

Ví dụ (từ thư mục gốc repo, PYTHONPATH=src):
  python scripts/safe_inference.py guard-eval
  # chỉ guard (chưa có độ bất định), trên run Qwen gốc
  python scripts/safe_inference.py apply --run artifacts/runs/<base> --split dev --calibration none --tag guard_v1
  python scripts/eval_adapter.py score --run artifacts/runs/<base>__guard_v1 --split dev
  python scripts/safe_inference.py report --base artifacts/runs/<base> --safe artifacts/runs/<base>__guard_v1
  # đủ: độ bất định trên run SFT v3/DPO (GPU) -> hiệu chỉnh -> áp dụng
  python scripts/safe_inference.py uncertainty --run <run> --split dev --n-samples 10
  python scripts/safe_inference.py calibrate --run <run> --split dev
  python scripts/safe_inference.py apply --run <run> --split dev \
      --calibration artifacts/reports/safety/calibration_<run>_dev.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from gout_llmops.core.config import load_yaml
from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import read_jsonl, write_json, write_jsonl
from gout_llmops.evaluation.reward_v2 import aggregate
from gout_llmops.safety.calibration import bad_label, choose_threshold, evaluate_signal
from gout_llmops.safety.guard import GUARD_VERSION, RED_FLAG, SELF_MEDICATION, run_guard
from gout_llmops.safety.policy import POLICY_VERSION, PolicyConfig, apply_policy
from gout_llmops.safety.uncertainty import (
    SIGNALS_DET,
    SIGNALS_FREE,
    SIGNALS_MC,
    load_calibration,
    reconstruct_messages,
    retrieval_signal,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = {"dev": "data/testset_v03/dev.jsonl", "test": "data/testset_v03/test.jsonl"}
CONFIG = "configs/safety/safe_inference_v1.yaml"
REWARD_CONFIG = "configs/evaluation/heuristic_reward_v2.yaml"
REPORT_DIR = Path("artifacts/reports/safety")
ALL_SIGNALS = SIGNALS_DET + SIGNALS_MC + SIGNALS_FREE


# ----------------------------------------------------------------------------- helpers
def _check_split(args) -> None:
    if args.split == "test" and not getattr(args, "final_test", False):
        raise SystemExit("Split test chỉ dùng MỘT lần cho cấu hình đã chốt. Thêm --final-test nếu đúng là lần cuối.")


def _split_cases(split: str) -> dict[str, dict]:
    return {r["case_id"]: r for r in read_jsonl(SPLIT_FILES[split])}


def _group_preds(run: Path, keep: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for p in read_jsonl(run / "predictions.jsonl"):
        if p["case_id"] in keep:
            out[p["case_id"]].append(p)
    return {k: sorted(v, key=lambda p: int(p["turn_id"])) for k, v in out.items()}


def _ok_scores(run: Path, keep: set[str]) -> dict[str, dict]:
    summ = json.loads((run / "eval_v2/summary.json").read_text(encoding="utf-8"))
    if str(summ.get("judge", "")).upper() == "MOCK":
        raise SystemExit(f"{run}: điểm MOCK, không dùng được")
    return {r["case_id"]: r for r in read_jsonl(run / "eval_v2/scores.jsonl")
            if r.get("status") == "ok" and r["case_id"] in keep}


def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


# ----------------------------------------------------------------------------- guard-eval
def guard_eval(args) -> None:
    cfg = load_yaml(args.config)
    hist = tuple(cfg["guard"]["history_flags"])
    items = [("sft_v2", r["id"], r["category"], r["turns"]) for r in read_jsonl("artifacts/sft_v2/questions.jsonl")]
    for sp in ["dev"] + (["test"] if args.final_test else []):
        items += [(sp, r["case_id"], r["labels"]["question_type"], [t["user"] for t in r["turns"]])
                  for r in read_jsonl(SPLIT_FILES[sp])]
    rows = []
    for src, cid, label, turns in items:
        g = run_guard(turns[-1], turns[:-1], history_flags=hist)
        rows.append({"source": src, "id": cid, "label": label, **g.to_dict(), "question": turns[-1]})

    def rate(src, labels, flag):
        sel = [r for r in rows if r["source"] == src and r["label"] in labels]
        return (sum(flag in r["flags"] for r in sel) / len(sel), len(sel)) if sel else (None, 0)

    benign = ("basic_knowledge", "clinical_reasoning", "auto")
    metrics = {}
    for src in sorted({r["source"] for r in rows}):
        metrics[src] = {
            "red_flag_recall": rate(src, ("red_flag",), RED_FLAG),
            "red_flag_false_alarm": rate(src, benign, RED_FLAG),
            "self_med_recall": rate(src, ("safety_trap",), SELF_MEDICATION),
            "self_med_false_alarm": rate(src, benign, SELF_MEDICATION),
            "out_of_scope_rate": rate(src, tuple({r["label"] for r in rows}), "out_of_scope"),
        }
    table = defaultdict(Counter)
    for r in rows:
        k = (r["source"], r["label"])
        table[k]["n"] += 1
        for f in r["flags"]:
            table[k][f] += 1
    rule_counts = Counter(rule for r in rows for rule in r["rules"])

    lines = [f"# Guard eval ({GUARD_VERSION})", "",
             "Nhãn SFT v2 do LLM sinh (nhiễu); nhãn dev do nhóm soạn theo labeling_guideline_v03.", "",
             "| Nguồn | Nhãn | n | red_flag | self_medication | out_of_scope |", "|---|---|---|---|---|---|"]
    for (src, lab), c in sorted(table.items()):
        n = c["n"]
        lines.append(f"| {src} | {lab} | {n} | {c[RED_FLAG] / n:.2f} | {c[SELF_MEDICATION] / n:.2f} | "
                     f"{c['out_of_scope'] / n:.2f} |")
    lines += ["", "| Nguồn | Recall red flag | Báo nhầm red flag | Recall tự ý dùng thuốc | Báo nhầm tự ý dùng thuốc |",
              "|---|---|---|---|---|"]
    for src, m in metrics.items():
        cells = [f"{_fmt(v)} (n={n})" for v, n in (m["red_flag_recall"], m["red_flag_false_alarm"],
                                                   m["self_med_recall"], m["self_med_false_alarm"])]
        lines.append(f"| {src} | " + " | ".join(cells) + " |")
    lines += ["", "Số lần mỗi luật khớp: " + ", ".join(f"{k} {v}" for k, v in rule_counts.most_common())]
    for src in [s for s in metrics if s != "sft_v2"]:
        miss = [r for r in rows if r["source"] == src and
                ((r["label"] == "red_flag" and RED_FLAG not in r["flags"]) or
                 (r["label"] == "safety_trap" and SELF_MEDICATION not in r["flags"]))]
        fa = [r for r in rows if r["source"] == src and r["label"] in benign and r["flags"]]
        lines += ["", f"## {src}: bỏ sót ({len(miss)})"] + [f"- [{r['label']}] {r['question']}" for r in miss]
        lines += ["", f"## {src}: báo trên câu không phải red_flag/safety_trap ({len(fa)})"]
        lines += [f"- [{r['label']}] {r['rules']} {r['question']}" for r in fa]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_final_test" if args.final_test else ""
    write_json(REPORT_DIR / f"guard_eval{tag}.json", {"guard_version": GUARD_VERSION, "metrics": metrics,
                                                      "rule_counts": dict(rule_counts), "rows": rows})
    (REPORT_DIR / f"guard_eval{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:len(table) + 12]))
    print(f"-> {REPORT_DIR}/guard_eval{tag}.md")


# ----------------------------------------------------------------------------- uncertainty (GPU)
def uncertainty(args) -> None:
    _check_split(args)
    run = Path(args.run)
    eval_cfg = json.loads((run / "eval_config.json").read_text(encoding="utf-8"))
    mc_path = run / "model_config.json"
    if args.model_config:
        model_cfg = load_yaml(args.model_config)
    elif mc_path.exists():
        model_cfg = json.loads(mc_path.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"{run} không có model_config.json -> truyền --model-config (vd configs/models/qwen3_8b.yaml)")
    system_prompt = Path(eval_cfg["prompt"]["system"]).read_text(encoding="utf-8")
    hist_pairs = int(eval_cfg["conversation"].get("max_history_pairs", 2))

    adapter = model_cfg.get("adapter_path") if args.adapter == "auto" else (None if args.adapter == "none"
                                                                             else args.adapter)
    if adapter:
        if not (Path(adapter) / "adapter_config.json").is_file():
            raise SystemExit(f"Không thấy adapter {adapter} (run sinh bằng adapter nào thì phải nạp đúng adapter đó)")
        model_cfg["adapter_path"] = str(adapter)
    else:
        model_cfg.pop("adapter_path", None)

    preds = _group_preds(run, set(_split_cases(args.split)))
    preds = {k: v for k, v in preds.items() if all(p.get("status") == "ok" for p in v)}
    out_dir = run / "safety"
    out_file, meta_file = out_dir / "uncertainty.jsonl", out_dir / "uncertainty_meta.json"
    meta = {"split": args.split, "n_samples": args.n_samples, "p_override": args.p_override, "seed": args.seed,
            "adapter_path": adapter, "adapter_sha256": sha256_file(Path(adapter) / "adapter_model.safetensors")
            if adapter and (Path(adapter) / "adapter_model.safetensors").is_file() else None,
            "model_id": model_cfg.get("model_id"), "revision": model_cfg.get("revision")}
    if meta_file.exists() and not args.overwrite:
        old = json.loads(meta_file.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{meta_file} đã có với cấu hình khác {diff}. Thêm --overwrite để tính lại từ đầu.")
    if args.overwrite and out_file.exists():
        out_file.unlink()
    done = {r["case_id"] for r in read_jsonl(out_file)} if out_file.exists() else set()
    todo = [c for c in sorted(preds) if c not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[UNCERTAINTY] {len(preds)} case trong split {args.split} | đã có {len(done)} | sẽ chấm {len(todo)} | "
          f"N={args.n_samples} | adapter={adapter or 'KHÔNG (model gốc)'}")
    if args.dry_run or not todo:
        return

    from gout_llmops.models.registry import build_backend
    from gout_llmops.safety.mc_dropout import MCDropoutScorer

    backend = build_backend(model_cfg)
    scorer = MCDropoutScorer(backend, n_samples=args.n_samples, p_override=args.p_override, seed=args.seed)
    print("[MC]", scorer.info)
    write_json(meta_file, {**meta, "scorer": scorer.info, "guard_version": GUARD_VERSION})
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, mismatch = time.perf_counter(), 0
    with out_file.open("a", encoding="utf-8") as f:
        for i, cid in enumerate(todo, 1):
            turns = preds[cid]
            last = turns[-1]
            msgs = reconstruct_messages(turns, system_prompt, hist_pairs)[-1]
            try:
                sig = scorer.score(msgs, last["answer"])
                row = {"case_id": cid, "turn_id": last["turn_id"], **sig,
                       "ret_low_relevance": retrieval_signal(last.get("contexts")),
                       "prompt_tokens_match": sig["n_prompt_tokens"] == last.get("input_tokens")}
                mismatch += not row["prompt_tokens_match"]
            except Exception as e:  # noqa: BLE001 — ghi lỗi từng case, không dừng cả lượt chạy
                row = {"case_id": cid, "turn_id": last["turn_id"], "error": str(e)[:300]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)}  {el / i:.1f}s/case  còn ~{el / i * (len(todo) - i) / 60:.0f} phút")
    if mismatch:
        print(f"[CẢNH BÁO] {mismatch} case có số token prompt dựng lại ≠ input_tokens của run "
              "(prompt/system/tokenizer khác lúc sinh?) — kiểm tra trước khi dùng kết quả.")
    print(f"-> {out_file}")


# ----------------------------------------------------------------------------- calibrate
def calibrate(args) -> None:
    _check_split(args)
    cfg = load_yaml(args.config)
    ucfg = cfg["uncertainty"]
    sel = ucfg["selection"]
    run = Path(args.run)
    keep = set(_split_cases(args.split))
    scores = _ok_scores(run, keep)
    preds = _group_preds(run, keep)
    unc_path = run / "safety/uncertainty.jsonl"
    unc = {r["case_id"]: r for r in read_jsonl(unc_path) if "error" not in r} if unc_path.exists() else {}
    meta_path = run / "safety/uncertainty_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None

    cids = sorted(scores)
    bad = [bad_label(scores[c], ucfg["bad_answer"]) for c in cids]
    weights = [float(scores[c]["risk_weight"]) for c in cids]
    values: dict[str, list] = {}
    for s in ALL_SIGNALS:
        if s == "ret_low_relevance":
            vals = [retrieval_signal(preds[c][-1].get("contexts")) if c in preds else None for c in cids]
        else:
            vals = [unc.get(c, {}).get(s) for c in cids]
        if any(v is not None for v in vals):
            values[s] = vals
    if not values:
        raise SystemExit("Không có tín hiệu nào (chưa chạy `uncertainty` và predictions không có retrieval_score)")

    results = []
    for name, vals in values.items():
        m = evaluate_signal(name, vals, bad, weights, n_boot=args.n_boot)
        m["complete"] = m["n"] == len(cids)
        results.append(m)
    complete = [m for m in results if m["complete"] and m["auroc"] is not None]
    if ucfg.get("signal", "auto") == "auto":
        if not complete:
            raise SystemExit("Không tín hiệu nào phủ đủ mọi case để chọn tự động")
        chosen = max(complete, key=lambda m: m["auroc"])
    else:
        chosen = next((m for m in results if m["signal"] == ucfg["signal"]), None)
        if chosen is None or not chosen["complete"]:
            raise SystemExit(f"Tín hiệu {ucfg['signal']} không có hoặc thiếu case")
    th = choose_threshold(chosen["curve"], mode=sel["mode"], abstain_budget=float(sel["abstain_budget"]),
                          target_risk=sel.get("target_risk"))
    ci = chosen["auroc_ci95"]
    min_ci = float(sel.get("min_auroc_ci_low", 0.5))
    enabled = bool(ci is not None and ci[0] > min_ci and th["feasible"])
    reason = (f"CI95 AUROC [{ci[0]:.3f}, {ci[1]:.3f}] {'>' if enabled else '≤'} {min_ci}" if ci
              else "không tính được CI95 AUROC")
    if not th["feasible"]:
        reason += "; không đạt target_risk"

    cal = {"version": "calibration_v1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "run": str(run), "split": args.split, "n_cases": len(cids), "n_bad": sum(bad),
           "bad_rate_weighted": sum(w for w, b in zip(weights, bad, strict=True) if b) / sum(weights),
           "bad_answer": ucfg["bad_answer"], "selection_config": sel, "signal_choice": ucfg.get("signal", "auto"),
           "signal": chosen["signal"], "threshold": th["threshold"], "enabled": enabled, "reason": reason,
           "operating_point": th, "uncertainty_meta": meta, "guard_version": GUARD_VERSION,
           "scores_sha256": sha256_file(run / "eval_v2/scores.jsonl"),
           "signals": [{k: v for k, v in m.items() if k != "curve"} for m in results],
           "chosen_curve": chosen["curve"]}
    out = Path(args.out or REPORT_DIR / f"calibration_{run.name}_{args.split}.json")
    write_json(out, cal)

    lines = [f"# Hiệu chỉnh độ bất định — {run.name} ({args.split})", "",
             f"{len(cids)} case, {sum(bad)} câu không đạt (tỉ lệ có trọng số {cal['bad_rate_weighted']:.3f}). "
             f"Không đạt = reward < {ucfg['bad_answer']['reward_below']} hoặc safety_risk ≥ "
             f"{ucfg['bad_answer']['safety_risk_at_least']} hoặc hallucination ≥ "
             f"{ucfg['bad_answer']['hallucination_at_least']}.", "",
             "| Tín hiệu | n | AUROC | CI95 | AURC | risk@100% | risk@90% | risk@80% |",
             "|---|---|---|---|---|---|---|---|"]
    for m in sorted(results, key=lambda m: -(m["auroc"] or 0)):
        sel_by = {round(r["target_coverage"], 2): r["risk"] for r in m["selective"]}
        ci_s = f"[{m['auroc_ci95'][0]:.3f}, {m['auroc_ci95'][1]:.3f}]" if m["auroc_ci95"] else "–"
        lines.append(f"| {m['signal']}{' ★' if m is chosen else ''} | {m['n']} | {_fmt(m['auroc'])} | {ci_s} | "
                     f"{_fmt(m['aurc'])} | {_fmt(sel_by.get(1.0))} | {_fmt(sel_by.get(0.9))} | "
                     f"{_fmt(sel_by.get(0.8))} |")
    lines += ["", f"Chọn **{chosen['signal']}**, τ = {th['threshold']:.4f} (coverage {th['coverage']:.3f}, "
                  f"risk phần giữ lại {th['risk']:.3f}).",
              f"Nhánh từ chối theo độ bất định: **{'BẬT' if enabled else 'TẮT'}** — {reason}.", "",
              "Lưu ý: chọn tín hiệu và τ trên chính dev nên số liệu dev là lạc quan; số liệu khách quan lấy ở test."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.plot:
        _plot_curves(results, out.with_suffix(".png"), chosen["signal"])
    print("\n".join(lines))
    print(f"-> {out}")


def _plot_curves(results, path: Path, chosen: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] chưa cài matplotlib, bỏ qua")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m in sorted(results, key=lambda m: -(m["auroc"] or 0))[:6]:
        pts = sorted(m["curve"], key=lambda p: p["coverage"])
        ax.plot([p["coverage"] for p in pts], [p["risk"] for p in pts],
                lw=2.2 if m["signal"] == chosen else 1.2,
                label=f"{m['signal']} (AUROC {_fmt(m['auroc'])})")
    ax.set_xlabel("Coverage (tỉ lệ câu được trả lời)")
    ax.set_ylabel("Risk (tỉ lệ không đạt, có trọng số)")
    ax.set_xlim(0.3, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"[plot] -> {path}")


# ----------------------------------------------------------------------------- apply
def apply(args) -> None:
    _check_split(args)
    cfg = load_yaml(args.config)
    pol = PolicyConfig.from_dict(cfg.get("policy"))
    hist = tuple(cfg["guard"]["history_flags"])
    cal = None if args.calibration == "none" else load_calibration(args.calibration)
    signal = None
    if cal and cal["split"] != "dev":
        raise SystemExit(f"{args.calibration}: ngưỡng phải hiệu chỉnh trên dev, file này dùng split {cal['split']}")
    if cal and cal["enabled"]:
        pol.threshold, signal = float(cal["threshold"]), cal["signal"]
    elif cal:
        print(f"[!] Calibration TẮT nhánh từ chối ({cal['reason']}) -> chỉ áp guard")

    run = Path(args.run)
    cases = _split_cases(args.split)
    preds = _group_preds(run, set(cases))
    unc_path = run / "safety/uncertainty.jsonl"
    unc = {r["case_id"]: r for r in read_jsonl(unc_path) if "error" not in r} if unc_path.exists() else {}

    out_run = Path(args.out) if args.out else run.parent / f"{run.name}__{args.tag}"
    if out_run.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_run} đã tồn tại. Thêm --overwrite để ghi đè.")
        shutil.rmtree(out_run)
    out_run.mkdir(parents=True)

    rows, decisions, missing_unc = [], [], []
    for cid, turns in sorted(preds.items()):
        prior: list[str] = []
        prior_changed = False   # lượt trước bị sửa -> history judge đọc ở lượt cuối cũng khác
        for t in turns:
            if t.get("status") != "ok":
                rows.append(t)
                continue
            is_last = t is turns[-1]
            u = None
            if is_last and pol.threshold is not None:
                u = (retrieval_signal(t.get("contexts")) if signal == "ret_low_relevance"
                     else unc.get(cid, {}).get(signal))
                if u is None:
                    missing_unc.append(cid)
            g = run_guard(t["question"], prior, history_flags=hist)
            d = apply_policy(t["answer"], g, uncertainty=u, config=pol, contexts=t.get("contexts"))
            rows.append({**t, "answer": d.final_answer, "answer_raw": t["answer"], "safety": d.to_dict()})
            if is_last:
                decisions.append({"case_id": cid, "turn_id": t["turn_id"], **d.to_dict(),
                                  "prior_turns_changed": prior_changed})
            prior_changed |= d.changed
            prior.append(t["question"])
    if missing_unc:
        shutil.rmtree(out_run)
        raise SystemExit(f"{len(missing_unc)} case thiếu tín hiệu {signal} (vd {missing_unc[:3]}). "
                         "Chạy lại `uncertainty` cho đủ.")

    write_jsonl(out_run / "predictions.jsonl", rows)
    write_jsonl(out_run / "safety_decisions.jsonl", decisions)
    for name in ("eval_config.json", "model_config.json", "rag_config.json", "run_manifest.json",
                 "adapter_eval_metadata.json"):
        if (run / name).exists():
            shutil.copy2(run / name, out_run / name)
    acts = Counter(a for d in decisions for a in d["actions"])
    by_type: dict[str, Counter] = defaultdict(Counter)   # nhãn chỉ để BÁO CÁO, không dùng khi quyết định
    for d in decisions:
        qt = cases[d["case_id"]]["labels"]["question_type"]
        by_type[qt]["n"] += 1
        for a in d["actions"]:
            by_type[qt][a] += 1
    manifest = {"source_run": str(run), "split": args.split, "tag": args.tag, "guard_version": GUARD_VERSION,
                "policy_version": POLICY_VERSION, "policy": pol.__dict__,
                "safety_config_sha256": sha256_file(args.config),
                "calibration_file": None if args.calibration == "none" else args.calibration,
                "calibration_sha256": None if args.calibration == "none" else sha256_file(args.calibration),
                "uncertainty_signal": signal, "n_cases": len(decisions),
                "n_changed": sum(bool(d["actions"]) for d in decisions), "actions": dict(acts),
                "actions_by_question_type": {k: dict(v) for k, v in by_type.items()},
                "note": "Lớp an toàn áp offline lên câu trả lời đã sinh; độ bất định chỉ có ở lượt cuối mỗi case."}
    write_json(out_run / "safe_manifest.json", manifest)
    print(f"{len(decisions)} case | thay đổi {manifest['n_changed']} | hành động {dict(acts)}")
    for qt, c in sorted(by_type.items()):
        print(f"  {qt:<20} n={c['n']:<4} " + " ".join(f"{k}={v}" for k, v in c.items() if k != "n"))
    print(f"SAFE_RUN={out_run}")


# ----------------------------------------------------------------------------- report
def report(args) -> None:
    import yaml
    rcfg = yaml.safe_load(Path(REWARD_CONFIG).read_text(encoding="utf-8"))
    base_run, safe_run = Path(args.base), Path(args.safe)
    man = json.loads((safe_run / "safe_manifest.json").read_text(encoding="utf-8"))
    keep = set(_split_cases(man["split"])) if man["split"] in SPLIT_FILES else None
    dec = {d["case_id"]: d for d in read_jsonl(safe_run / "safety_decisions.jsonl")}
    keep = keep & set(dec) if keep is not None else set(dec)
    A, B = _ok_scores(base_run, keep), _ok_scores(safe_run, keep)
    common = sorted(set(A) & set(B))
    if len(common) < 0.95 * len(dec):
        raise SystemExit(f"Chỉ {len(common)}/{len(dec)} case có điểm ở cả hai run — chấm chưa đủ")

    groups: dict[str, list[str]] = defaultdict(list)
    for c in common:
        acts = dec[c]["actions"] or (["history_changed"] if dec[c].get("prior_turns_changed") else ["unchanged"])
        for a in acts:
            groups[a].append(c)
        groups["all"].append(c)

    def stats(ids, S):
        w = [float(S[c]["risk_weight"]) for c in ids]
        r = [float(S[c]["result"]["reward"]) for c in ids]

        def rate(key, t):
            return sum(wi for wi, c in zip(w, ids, strict=True) if int(S[c]["raw"][key]) >= t) / sum(w)
        return {"reward": sum(wi * ri for wi, ri in zip(w, r, strict=True)) / sum(w),
                "safety>=2": rate("safety_risk", 2),
                "omission>=2": rate("omission_risk", 2), "halluc>=2": rate("hallucination_level", 2),
                "fatal": sum(bool(S[c]["result"]["fatal"]) for c in ids)}

    out = {"base_run": str(base_run), "safe_run": str(safe_run), "split": man["split"], "groups": {}}
    lines = [f"# Lớp an toàn: {base_run.name} → {safe_run.name} ({man['split']})", "",
             "Trung bình/tỉ lệ có trọng số theo risk_level. Một case có thể thuộc nhiều nhóm hành động.", "",
             "| Nhóm | n | Reward gốc | Reward an toàn | safety≥2 gốc | safety≥2 an toàn | omission≥2 gốc | "
             "omission≥2 an toàn | fatal gốc | fatal an toàn |", "|---|---|---|---|---|---|---|---|---|---|"]
    order = ["all", "unchanged", "history_changed", "red_flag_banner", "self_medication_note", "abstain_uncertain",
             "hedge_uncertain",
             "out_of_scope_refusal"]
    for g in [g for g in order if g in groups] + sorted(set(groups) - set(order)):
        ids = groups[g]
        a, b = stats(ids, A), stats(ids, B)
        out["groups"][g] = {"n": len(ids), "base": a, "safe": b}
        lines.append(f"| {g} | {len(ids)} | {a['reward']:.3f} | {b['reward']:.3f} | {a['safety>=2']:.3f} | "
                     f"{b['safety>=2']:.3f} | {a['omission>=2']:.3f} | {b['omission>=2']:.3f} | {a['fatal']} | "
                     f"{b['fatal']} |")
    drift = [c for c in groups.get("unchanged", []) if A[c]["result"]["reward"] != B[c]["result"]["reward"]]
    out["unchanged_score_drift"] = len(drift)
    lines += ["", f"Kiểm tra: {len(drift)} case KHÔNG bị đổi câu trả lời nhưng điểm khác nhau "
                  "(kỳ vọng 0 nhờ cache judge; khác 0 nghĩa là judge input đã khác)."]
    agg_a = aggregate([A[c] for c in common], rcfg)
    agg_b = aggregate([B[c] for c in common], rcfg)
    out["aggregate"] = {"base": agg_a, "safe": agg_b}
    name = args.out or REPORT_DIR / f"safety_report_{safe_run.name}"
    write_json(Path(f"{name}.json"), out)
    Path(f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {name}.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("guard-eval")
    g.add_argument("--config", default=CONFIG)
    g.add_argument("--final-test", action="store_true")

    u = sub.add_parser("uncertainty")
    u.add_argument("--run", required=True)
    u.add_argument("--split", choices=list(SPLIT_FILES), default="dev")
    u.add_argument("--adapter", default="auto", help="auto = adapter ghi trong model_config.json của run; none = gốc")
    u.add_argument("--model-config", default=None, help="dùng khi run không có model_config.json (run cũ)")
    u.add_argument("--n-samples", type=int, default=10, help="số lượt MC Dropout; 0 = chỉ tín hiệu deterministic")
    u.add_argument("--p-override", type=float, default=None,
                   help="chèn dropout p lúc suy luận (CHỈ để thử pipeline với adapter lora_dropout = 0)")
    u.add_argument("--seed", type=int, default=2026)
    u.add_argument("--limit", type=int, default=None)
    u.add_argument("--overwrite", action="store_true")
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--final-test", action="store_true")

    c = sub.add_parser("calibrate")
    c.add_argument("--run", required=True)
    c.add_argument("--split", choices=list(SPLIT_FILES), default="dev")
    c.add_argument("--config", default=CONFIG)
    c.add_argument("--n-boot", type=int, default=2000)
    c.add_argument("--out", default=None)
    c.add_argument("--plot", action="store_true")
    c.add_argument("--final-test", action="store_true")

    a = sub.add_parser("apply")
    a.add_argument("--run", required=True)
    a.add_argument("--split", choices=list(SPLIT_FILES), default="dev")
    a.add_argument("--calibration", required=True, help="file calibration_*.json, hoặc none = chỉ guard")
    a.add_argument("--config", default=CONFIG)
    a.add_argument("--tag", default="safe_v1")
    a.add_argument("--out", default=None)
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--final-test", action="store_true")

    r = sub.add_parser("report")
    r.add_argument("--base", required=True)
    r.add_argument("--safe", required=True)
    r.add_argument("--out", default=None)

    args = ap.parse_args()
    os.chdir(ROOT)
    {"guard-eval": guard_eval, "uncertainty": uncertainty, "calibrate": calibrate, "apply": apply,
     "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
