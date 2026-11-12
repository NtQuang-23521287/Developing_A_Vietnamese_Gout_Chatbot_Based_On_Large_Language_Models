"""Lightweight regression tests for the post-SFT evaluator (no GPU/API)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def import_script():
    p = Path(__file__).resolve().parents[1] / "scripts/evaluate_post_sft.py"
    spec = importlib.util.spec_from_file_location("sft_eval_test_module", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_expected_v03_cases():
    m = import_script()
    cases = m.check_testset()
    assert len(cases) == 381
    assert sum(len(c["turns"]) for c in cases) == 441


def test_rejects_wrong_lr(tmp_path):
    m = import_script()
    d = tmp_path / "sft_gemma3_4b_v2"
    d.mkdir()
    (d / "sft_run_metadata.json").write_text(json.dumps({
        "model_family": "gemma3",
        "config": {"trainer": {"learning_rate": 1e-4, "num_train_epochs": 10}}
    }))
    try:
        m.metadata_and_adapter("gemma3", d)
    except RuntimeError as exc:
        assert "WRONG SFT RUN" in str(exc)
    else:
        raise AssertionError("Must reject LR mismatch")


def test_duplicate_predictions_are_rejected(tmp_path):
    m = import_script()
    first = m.check_testset()[0]
    run = tmp_path / "run"
    run.mkdir()
    row = {"case_id": first["case_id"], "turn_id": first["turns"][0]["turn_id"], "status": "ok"}
    (run / "predictions.jsonl").write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n")
    (run / "model_config.json").write_text(json.dumps({"adapter_path": "unused"}))
    try:
        m.verify_predictions(run, limit=1)
    except RuntimeError as exc:
        assert "Incomplete predictions" in str(exc)
    else:
        raise AssertionError("Must reject duplicates")
