import dataclasses
import importlib.util
import json
from pathlib import Path

import pytest

from gout_llmops.core.config import load_yaml
from gout_llmops.training.dpo import dpo_fingerprint, reference_setup, render_prompt

ROOT = Path(__file__).resolve().parents[1]
QWEN, GEMMA = "configs/models/qwen3_8b.yaml", "configs/models/gemma3_4b.yaml"


def _pipeline():
    spec = importlib.util.spec_from_file_location("dpo_pipeline", ROOT / "scripts/dpo_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_paths_keep_qwen_layout_and_separate_gemma():
    P = _pipeline().Paths
    assert P(False).cands == Path("artifacts/dpo_v1/candidates.jsonl")       # đường dẫn cũ của Qwen giữ nguyên
    assert P(False).pref == Path("data/dpo/preference_v1.jsonl")
    assert P(False, GEMMA).cands == Path("artifacts/dpo_v1/gemma3_4b/candidates.jsonl")
    assert P(False, GEMMA).manifest == Path("data/dpo/gemma3_4b/manifest.json")
    assert P(True, GEMMA).pref == Path("artifacts/dpo_v1_mock/gemma3_4b/data/preference_v1.jsonl")


@pytest.mark.parametrize("dpo_config, model_config", [("configs/training/dpo_v1.yaml", QWEN),
                                                      ("configs/training/dpo_gemma3_4b_v1.yaml", GEMMA)])
def test_dpo_config_reads_what_mine_writes(dpo_config, model_config):
    cfg = load_yaml(dpo_config)
    assert cfg["base_model_config"] == model_config
    assert Path(cfg["preference_data"]) == _pipeline().Paths(False, model_config).pref
    assert Path(cfg["output_dir"]).name == cfg["name"]   # --checkpoint-mirror phải trùng tên output_dir
    assert int(cfg["trainer"]["save_steps"]) > 0


def test_generation_meta_diff_detects_model_change():
    diff = _pipeline().generation_meta_diff
    old = {"policy_adapter": "a", "k": 4}                    # meta cũ chưa ghi model_config = Qwen3-8B
    assert diff(old, {"policy_adapter": "a", "k": 4, "model_config": "./" + QWEN}) == {}
    assert set(diff(old, {"policy_adapter": "a", "k": 4, "model_config": GEMMA})) == {"model_config"}


def test_reference_setup_for_both_trl_apis():
    @dataclasses.dataclass
    class OldConfig:   # TRL 0.x
        model_adapter_name: str = ""
        ref_adapter_name: str = ""

    @dataclasses.dataclass
    class NewConfig:   # TRL 1.x
        beta: float = 0.1

    assert reference_setup(OldConfig, "") == (True, {"model_adapter_name": "default", "ref_adapter_name": "ref"})
    uses_ref = 'with use_adapter(model, adapter_name="ref" if "ref" in model.peft_config else None):'
    copies_ref = 'model.add_adapter("ref", default_config)\n' + uses_ref
    assert reference_setup(NewConfig, copies_ref) == (False, {})   # TRL 1.x tự chép "default" thành "ref"
    assert reference_setup(NewConfig, uses_ref) == (True, {})
    with pytest.raises(RuntimeError):
        reference_setup(NewConfig, "with model.disable_adapter():")


def test_render_prompt_matches_eval_paths():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]

    class Proc:
        def apply_chat_template(self, m, **kw):
            assert m[1]["content"] == [{"type": "text", "text": "Q"}] and kw["add_generation_prompt"]
            return "gemma"

    class Tok:
        def apply_chat_template(self, m, **kw):
            assert m == msgs and kw["enable_thinking"] is False and not kw["tokenize"]
            return "qwen"

    assert render_prompt(msgs, Tok(), Proc()) == "gemma"
    assert render_prompt(msgs, Tok(), None, {"enable_thinking": False}) == "qwen"


def test_dpo_fingerprint_roundtrips(tmp_path):
    pref = tmp_path / "pref.jsonl"
    pref.write_text('{"id": 1}\n', encoding="utf-8")
    cfg = load_yaml(ROOT / "configs/training/dpo_gemma3_4b_v1.yaml")
    fp = dpo_fingerprint(cfg, load_yaml(ROOT / GEMMA), None, pref)
    assert json.loads(json.dumps(fp)) == fp and fp["trainer"]["save_steps"] == 5
    pref.write_text('{"id": 2}\n', encoding="utf-8")
    assert dpo_fingerprint(cfg, load_yaml(ROOT / GEMMA), None, pref) != fp   # đổi dữ liệu -> không train tiếp
