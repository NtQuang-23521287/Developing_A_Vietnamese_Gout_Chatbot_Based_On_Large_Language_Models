#!/usr/bin/env python3
"""Post-SFT evaluation on the Phase-2 v03 benchmark (381 cases / 441 turns).

Run from the repository root with PYTHONPATH=src, or directly:
    python scripts/evaluate_post_sft.py generate --model qwen3 --dry-run
    python scripts/evaluate_post_sft.py generate --model qwen3 --limit 3
    python scripts/evaluate_post_sft.py generate --model gemma3
    python scripts/evaluate_post_sft.py verify --run artifacts/runs/<run_id>
    python scripts/evaluate_post_sft.py score --run artifacts/runs/<run_id> --mock --limit 3
    python scripts/evaluate_post_sft.py score --run artifacts/runs/<run_id>

The existing Phase-2 runner performs generation and RAG. The small registry
patch supplied with this script is REQUIRED to attach PEFT LoRA weights.
No API keys are stored by this script. Real Judge v2 calls cost money.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EVAL_SOURCE = ROOT / "configs/evaluation/baseline_v03_381.yaml"
TESTSET = ROOT / "data/testset_v03/cases_all.jsonl"
CONFIG_OUTPUT = ROOT / "artifacts/reports/sft_eval_381/configs"

MODELS = {
    "qwen3": {
        "model_config": "configs/models/qwen3_8b.yaml",
        "sft_run": "artifacts/runs/sft_qwen3_8b_v2",
        "family": "qwen3",
        "lr": 5e-5,
        "name": "qwen3_8b_sft_lr5e5",
    },
    "gemma3": {
        "model_config": "configs/models/gemma3_4b.yaml",
        "sft_run": "artifacts/runs/sft_gemma3_4b_v2",
        "family": "gemma3",
        "lr": 2e-5,
        "name": "gemma3_4b_sft_lr2e5",
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def check_testset() -> list[dict[str, Any]]:
    if not TESTSET.is_file():
        raise FileNotFoundError(TESTSET)
    cases = read_jsonl(TESTSET)
    keys = [(c["case_id"], int(t["turn_id"])) for c in cases for t in c["turns"]]
    if len(cases) != 381 or len(keys) != 441 or len(keys) != len(set(keys)):
        raise RuntimeError(f"Testset unexpected: {len(cases)} cases / {len(keys)} turns")
    return cases


def metadata_and_adapter(model: str, sft_run: Path, requested_step: int | None = None) -> tuple[Path, dict, dict]:
    spec = MODELS[model]
    if not sft_run.is_dir():
        raise FileNotFoundError(f"Missing SFT run {sft_run}. Restore it from Google Drive first.")
    meta_file = sft_run / "sft_run_metadata.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"Missing {meta_file}. Refusing to guess the training LR.")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if meta.get("model_family") != spec["family"]:
        raise RuntimeError(f"Wrong model family in metadata: {meta.get('model_family')}")
    train_config = meta.get("config", {}).get("trainer", {})
    lr = float(train_config.get("learning_rate", -1))
    epochs = float(train_config.get("num_train_epochs", -1))
    if abs(lr - spec["lr"]) > 1e-10 or epochs != 10:
        raise RuntimeError(
            f"WRONG SFT RUN: {sft_run} metadata says lr={lr}, epochs={epochs}; "
            f"expected lr={spec['lr']}, epochs=10. Do not evaluate an old adapter."
        )
    states = []
    for p in sft_run.rglob("trainer_state.json"):
        try:
            st = json.loads(p.read_text(encoding="utf-8"))
            states.append(st)
        except (ValueError, OSError):
            continue
    if not states:
        raise RuntimeError(f"Missing trainer_state.json in {sft_run}")
    state = max(states, key=lambda s: (int(s.get("global_step", 0)), len(s.get("log_history", []))))
    if int(state.get("global_step", 0)) != 250:
        raise RuntimeError(f"Run is not completed to 250 steps: {sft_run}")
    best_step = state.get("best_global_step")
    if best_step is None and state.get("best_model_checkpoint"):
        best_step = int(Path(state["best_model_checkpoint"]).name.rsplit("-", 1)[-1])
    if requested_step is not None:
        best_step = requested_step
    if not best_step:
        raise RuntimeError("No best checkpoint recorded")
    checkpoint = sft_run / f"checkpoint-{int(best_step)}"
    # Prefer the exact best checkpoint. Training script load_best_model_at_end=True,
    # then trainer.save_model() to the run root: fallback is the selected best adapter.
    if (checkpoint / "adapter_model.safetensors").is_file():
        adapter = checkpoint
        note = "selected checkpoint directory"
    elif requested_step is None and (sft_run / "adapter_model.safetensors").is_file():
        adapter = sft_run
        note = "root adapter (SFT Trainer saved best model at end)"
    else:
        raise FileNotFoundError(f"Missing best adapter for step {best_step} in {sft_run}")
    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    model_cfg = yaml.safe_load((ROOT / spec["model_config"]).read_text(encoding="utf-8"))
    base_in_adapter = cfg.get("base_model_name_or_path")
    if base_in_adapter and base_in_adapter != model_cfg["model_id"]:
        raise RuntimeError(f"Adapter base mismatch: {base_in_adapter} vs {model_cfg['model_id']}")
    print(f"[CHECK] {model}: LR={lr}, 10 epochs, best_step={best_step}, best_metric={state.get('best_metric')}")
    print(f"[CHECK] {note}: {adapter}")
    return adapter.resolve(), meta, state


def generate(args: argparse.Namespace) -> Path | None:
    spec = MODELS[args.model]
    check_testset()
    sft_run = (ROOT / args.sft_run).resolve() if args.sft_run else (ROOT / spec["sft_run"]).resolve()
    adapter, meta, state = metadata_and_adapter(args.model, sft_run, args.checkpoint_step)
    if not EVAL_SOURCE.is_file():
        raise FileNotFoundError(f"Missing Phase-2 protocol: {EVAL_SOURCE}")
    loader = ROOT / "src/gout_llmops/models/registry.py"
    src = loader.read_text(encoding="utf-8")
    if "PeftModel.from_pretrained" not in src or "adapter_path" not in src:
        raise RuntimeError(
            "LoRA support not detected in src/gout_llmops/models/registry.py. "
            "Apply patches/registry_lora.patch first; otherwise the runner may generate BASE outputs."
        )
    model_cfg = yaml.safe_load((ROOT / spec["model_config"]).read_text(encoding="utf-8"))
    if meta.get("model_id") != model_cfg["model_id"]:
        raise RuntimeError(f"Training base mismatch: {meta.get('model_id')} vs {model_cfg['model_id']}")
    if meta.get("revision") and model_cfg.get("revision") and meta["revision"] != model_cfg["revision"]:
        raise RuntimeError(f"Training/base revision mismatch: {meta['revision']} vs {model_cfg['revision']}")
    model_cfg["name"] = spec["name"]
    model_cfg["adapter_path"] = str(adapter)
    if args.gemma_compute_dtype is not None and args.model == "gemma3":
        model_cfg.setdefault("quantization", {})["compute_dtype"] = args.gemma_compute_dtype
        print("[WARNING] Gemma compute dtype changed from baseline; report this in comparisons.")
    eval_cfg = yaml.safe_load(EVAL_SOURCE.read_text(encoding="utf-8"))
    if eval_cfg["testset"]["path"] != "data/testset_v03/cases_all.jsonl":
        raise RuntimeError("Evaluation config is NOT the Phase-2 381-case benchmark")
    eval_cfg["phase"] = "post_sft_evaluation"
    eval_cfg["name"] = "post_sft_v03_381"
    eval_cfg.setdefault("judge", {})["enabled"] = False  # v1 judge MUST stay off
    eval_cfg.setdefault("ragas", {})["enabled"] = False
    CONFIG_OUTPUT.mkdir(parents=True, exist_ok=True)
    model_yaml = CONFIG_OUTPUT / f"{spec['name']}_model.yaml"
    eval_yaml = CONFIG_OUTPUT / "post_sft_v03_381.yaml"
    model_yaml.write_text(yaml.safe_dump(model_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    eval_yaml.write_text(yaml.safe_dump(eval_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("[PROTOCOL] Phase-2 RAG:", eval_cfg["rag"])
    print("[PROTOCOL] prompt:", eval_cfg["prompt"])
    print("[PROTOCOL] generation:", eval_cfg["generation"])
    if args.dry_run:
        print("[DRY-RUN] Configs prepared; GPU inference NOT started")
        return None
    # Import after checks: this triggers transformers only when inference starts.
    from gout_llmops.evaluation.runner import run_baseline
    os.chdir(ROOT)
    result = Path(run_baseline(str(model_yaml), str(eval_yaml),
                               limit=args.limit, enable_judge=False, enable_ragas=False)).resolve()
    provenance = {
        "evaluation": "post_sft_v03_381", "model": args.model,
        "sft_run": str(sft_run), "adapter_path": str(adapter),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "sft_learning_rate": spec["lr"], "sft_epochs": 10,
        "best_step": state.get("best_global_step"),
        "best_eval_loss": state.get("best_metric"),
        "testset_sha256": sha256(TESTSET),
        "full_benchmark": args.limit is None,
        "compute_dtype": model_cfg.get("quantization", {}).get("compute_dtype"),
        "warning": "Research only; not for medical use. Check SFT/test contamination.",
    }
    (result / "post_sft_metadata.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False),
                                                   encoding="utf-8")
    verify_predictions(result, args.limit)
    print("\n[OK] Generated:", result)
    print("[NEXT] python scripts/evaluate_post_sft.py score --run", str(result))
    return result


def verify_predictions(run: Path, limit: int | None = None) -> None:
    cases = check_testset()
    cases = cases[:limit] if limit is not None else cases
    expected = {(c["case_id"], int(t["turn_id"])) for c in cases for t in c["turns"]}
    pred_file = run / "predictions.jsonl"
    if not pred_file.is_file():
        raise FileNotFoundError(pred_file)
    rows = read_jsonl(pred_file)
    actual = [(r.get("case_id"), int(r.get("turn_id", -1))) for r in rows]
    duplicates = len(actual) - len(set(actual))
    missing = expected - set(actual)
    extra = set(actual) - expected
    errors = [r for r in rows if r.get("status") != "ok"]
    # A full 381-case run is valid input for a 3-case Judge smoke test.
    # In that situation, verify the FULL run rather than rejecting extra cases.
    if limit is not None and extra and not missing and not duplicates and not errors:
        verify_predictions(run, None)
        return
    print(f"[VERIFY] {run}: {len(rows)} predictions / {len(cases)} cases; "
          f"duplicates={duplicates}, missing={len(missing)}, extra={len(extra)}, errors={len(errors)}")
    if duplicates or missing or extra or errors:
        raise RuntimeError("Incomplete predictions. Fix generation BEFORE Judge. "
                           f"Examples missing={sorted(missing)[:3]}, errors={errors[:1]}")
    if limit is None and (len(cases), len(rows)) != (381, 441):
        raise RuntimeError("Full evaluation must contain 381 cases / 441 turns")
    model_cfg = run / "model_config.json"
    if not model_cfg.is_file() or "adapter_path" not in json.loads(model_cfg.read_text(encoding="utf-8")):
        raise RuntimeError("Run has no adapter_path in model_config.json — may be a BASE run")


def score(args: argparse.Namespace) -> None:
    run = (ROOT / args.run).resolve()
    verify_predictions(run, args.limit)
    if not args.mock and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing. Use Colab Secrets or environment; don't commit keys.")
    command = [sys.executable, str(ROOT / "scripts/score_run_v2.py"),
               "--run", str(run), "--testset", "data/testset_v03/cases_all.jsonl",
               "--split", "all", "--out-name", "eval_v2_sft_381"]
    if args.mock:
        command.append("--mock")
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    print("[JUDGE]", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    scored = run / "eval_v2_sft_381"
    summary = json.loads((scored / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(scored / "scores.jsonl")
    failed = [r for r in rows if r.get("status") != "ok"]
    expected_n = args.limit if args.limit is not None else 381
    if len(rows) != expected_n or failed:
        raise RuntimeError(f"Judge incomplete: rows={len(rows)}/{expected_n}, errors={len(failed)}")
    print("[JUDGE OK] Judge:", summary.get("judge"), "Cases:", len(rows),
          "Reward:", summary.get("reward_weighted_mean"),
          "Fatal:", summary.get("fatal_count"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Load base+LoRA and run Phase-2 RAG generation")
    g.add_argument("--model", choices=tuple(MODELS), required=True)
    g.add_argument("--sft-run", help="Override SFT training run directory")
    g.add_argument("--checkpoint-step", type=int, help="Use an explicit existing checkpoint")
    g.add_argument("--limit", type=int, help="First N cases for smoke testing; omit for 381")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--gemma-compute-dtype", choices=["float16", "float32"], default=None,
                   help="Override Gemma quant compute dtype (changes baseline protocol)")
    s = sub.add_parser("score", help="Judge v2 / Reward v2 on generated predictions")
    s.add_argument("--run", required=True)
    s.add_argument("--limit", type=int, help="Only for smoke test generated with --limit N")
    s.add_argument("--mock", action="store_true", help="No API call, output NOT suitable for report")
    v = sub.add_parser("verify", help="Validate predictions (and optional Judge results)")
    v.add_argument("--run", required=True)
    v.add_argument("--limit", type=int, help="For a smoke run")
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.command == "generate":
        if args.limit is not None and not (1 <= args.limit <= 381):
            ap.error("--limit must be between 1 and 381")
        generate(args)
    elif args.command == "score":
        if args.limit is not None and not (1 <= args.limit <= 381):
            ap.error("--limit must be between 1 and 381")
        score(args)
    else:
        verify_predictions((ROOT / args.run).resolve(), args.limit)


if __name__ == "__main__":
    main()
