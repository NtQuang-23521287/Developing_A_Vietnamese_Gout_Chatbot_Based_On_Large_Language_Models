#!/usr/bin/env python3
"""Compare Phase-2 Base with SFT on exactly the same 381 v03 cases.

Example:
  python scripts/compare_post_sft_381.py \
    --base-qwen artifacts/runs/<qwen_base_run> \
    --sft-qwen artifacts/runs/<qwen_sft_eval_run> \
    --base-gemma artifacts/runs/<gemma_base_run> \
    --sft-gemma artifacts/runs/<gemma_sft_eval_run>

Writes artifacts/reports/sft_eval_381/base_vs_sft.{md,json}; does NOT
overwrite the preexisting Phase-2 compare_v2 outputs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_scores(run: Path, expected_name: str | None):
    options = ([run / expected_name] if expected_name else []) + [run / "eval_v2", run]
    folder = next((p for p in options if (p / "summary.json").is_file() and (p / "scores.jsonl").is_file()), None)
    if folder is None:
        raise FileNotFoundError(f"Judge v2 output missing: {run}")
    summary = load_json(folder / "summary.json")
    rows = load_jsonl(folder / "scores.jsonl")
    if len(rows) != 381 or len({r.get("case_id") for r in rows}) != 381:
        raise RuntimeError(f"Not 381 unique scored cases: {folder}")
    if any(r.get("status") != "ok" for r in rows):
        raise RuntimeError(f"Judge errors in {folder}")
    if str(summary.get("judge", "")).upper() == "MOCK":
        raise RuntimeError(f"Mock Judge scores cannot be compared: {folder}")
    if summary.get("n_errors", 0) != 0:
        raise RuntimeError(f"Reported Judge errors in {folder}")
    return {r["case_id"]: r for r in rows}, summary


def weighted_average(rows, field, split=None):
    valid = [x for x in rows.values() if split is None or x.get("split") == split]
    if not valid:
        raise RuntimeError(f"No cases for split {split}")
    ww = [float(x.get("risk_weight", 1)) for x in valid]
    return sum(w * float(x["result"][field]) for x, w in zip(valid, ww, strict=True)) / sum(ww)


def weighted_submetric(rows, group, metric, split=None):
    valid = [x for x in rows.values() if split is None or x.get("split") == split]
    pairs = [(float(x["result"][group][metric]), float(x["risk_weight"]))
             for x in valid if x["result"].get(group, {}).get(metric) is not None]
    return sum(x*w for x, w in pairs)/sum(w for _, w in pairs) if pairs else None


def paired_diff(base, sft, split, seed=2026, trials=3000):
    ids = sorted(k for k in base if split is None or base[k].get("split") == split)
    if not ids:
        raise RuntimeError(f"No cases for {split}")
    weight = [float(base[k]["risk_weight"]) for k in ids]
    diff = [float(sft[k]["result"]["reward"]) - float(base[k]["result"]["reward"]) for k in ids]
    def mean_from(indices):
        return sum(diff[i]*weight[i] for i in indices)/sum(weight[i] for i in indices)
    rng = random.Random(seed)
    n = len(ids)
    boots = sorted(mean_from([rng.randrange(n) for _ in range(n)]) for _ in range(trials))
    return {"n": n, "sft_minus_base": mean_from(range(n)),
            "ci95": [boots[int(.025*trials)], boots[int(.975*trials)-1]]}


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("base-qwen", "sft-qwen", "base-gemma", "sft-gemma"):
        p.add_argument("--"+k,required=True)
    a=p.parse_args()
    labels={"Qwen Base": a.base_qwen,"Qwen SFT":a.sft_qwen,
            "Gemma Base":a.base_gemma,"Gemma SFT":a.sft_gemma}
    info={name:load_scores((ROOT/path).resolve(), "eval_v2_sft_381" if "SFT" in name else None)
          for name,path in labels.items()}
    ref=set(next(iter(info.values()))[0])
    protocols={(v[1].get("judge"),v[1].get("judge_version"),v[1].get("reward_version"),v[1].get("testset_sha256"))
               for v in info.values()}
    if len(protocols)!=1:
        raise RuntimeError(f"Different Judge/Reward/testset protocols across runs: {protocols}")
    for name,(rows,_) in info.items():
        if set(rows)!=ref:
            raise RuntimeError(f"Case ID mismatch: {name}")
        if {k:(v.get("split"),v.get("risk_weight")) for k,v in rows.items()} != \
           {k:(v.get("split"),v.get("risk_weight")) for k,v in next(iter(info.values()))[0].items()}:
            raise RuntimeError(f"Split/risk weights mismatch: {name}")
    results={}
    table=["| Model | All (381) | Dev (114) | Test (267) | Fatal all |", "|---|---:|---:|---:|---:|"]
    for name,(rows,_summary) in info.items():
        val={s:weighted_average(rows,"reward",s) for s in (None,"dev","test")}
        fatal=sum(bool(r["result"].get("fatal",False)) for r in rows.values())
        risks = {k:weighted_submetric(rows, "P", k) for k in
                 ("hallucination_level", "omission_risk", "safety_risk")}
        results[name]={"reward_all":val[None],"reward_dev":val["dev"],"reward_test":val["test"],
                       "fatal_count":fatal,"risk_weighted_mean":risks,"run":labels[name]}
        table.append(f"| {name} | {val[None]:.4f} | {val['dev']:.4f} | {val['test']:.4f} | {fatal} |")
    comparisons={}
    for family in ("Qwen","Gemma"):
        b,s=info[family+" Base"][0],info[family+" SFT"][0]
        comparisons[family]={split:paired_diff(b,s,split) for split in (None,"dev","test")}
    output=ROOT/"artifacts/reports/sft_eval_381"
    output.mkdir(parents=True,exist_ok=True)
    data={"protocol":next(iter(protocols)),"models":results,"pairs":comparisons,
          "caveat":"Potential SFT/test contamination; Judge is not a clinician."}
    (output/"base_vs_sft.json").write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# Phase-2 v03: Base vs SFT (Judge v2)","",*table,"",
           "## Risk metrics (normalized 0–1, lower is better)",
           "", "| Model | Hallucination | Omission | Safety risk |", "|---|---:|---:|---:|"]
    for name, result in results.items():
        risks = result["risk_weighted_mean"]
        lines.append("| " + name + " | " + " | ".join(
            "—" if risks[k] is None else f"{risks[k]:.4f}"
            for k in ("hallucination_level", "omission_risk", "safety_risk")) + " |")
    lines += ["", "Positive paired difference means SFT improves reward. Select checkpoints on dev, not test.", ""]
    for family,splits in comparisons.items():
        for split,v in splits.items():
            lo,hi=v["ci95"]
            lines.append(f"- {family}, {split or 'all'}: SFT−Base = {v['sft_minus_base']:+.4f}; "
                         f"CI95 [{lo:+.4f}, {hi:+.4f}] (n={v['n']})")
    lines += ["", "Warning: Testset may contain a training-data overlap. "
                  "The judge is automated, not a medical validation."]
    (output/"base_vs_sft.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved to:",output)


if __name__=="__main__":
    main()
