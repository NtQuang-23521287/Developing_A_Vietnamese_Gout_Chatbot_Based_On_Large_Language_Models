import json
import shutil

import pytest

from gout_llmops.training.resume import (
    RESUME_FINGERPRINT,
    check_mirror_dir,
    last_checkpoint,
    mirror_checkpoints,
    prepare_resume,
    pull_checkpoints,
)
from gout_llmops.training.sft import run_fingerprint

AUDIT = {"model_id": "google/gemma-3-4b-it", "revision": "r", "max_seq_length": 3072,
         "train_sha256": "a", "val_sha256": "b"}
CONFIG = {"seed": 2026, "qlora": {"bits": 4, "lora_r": 16},
          "trainer": {"learning_rate": 0.00002, "num_train_epochs": 10, "bf16": False}}
FP = run_fingerprint(CONFIG, AUDIT)


def _ckpt(root, step, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "adapter_model.safetensors").write_bytes(b"w" * step)
    if complete:
        (d / "trainer_state.json").write_text("{}", encoding="utf-8")
    return d


def test_last_checkpoint_numeric_and_complete_only(tmp_path):
    assert last_checkpoint(tmp_path / "missing") is None
    _ckpt(tmp_path, 9)
    _ckpt(tmp_path, 17)
    _ckpt(tmp_path, 26, complete=False)          # đang lưu dở khi runtime bị ngắt
    (tmp_path / ".tmp-checkpoint-34").mkdir()
    assert last_checkpoint(tmp_path).name == "checkpoint-17"


def test_prepare_resume(tmp_path):
    assert prepare_resume(tmp_path, FP, resume=True) is None   # chưa có checkpoint: train từ đầu
    assert json.loads((tmp_path / RESUME_FINGERPRINT).read_text(encoding="utf-8")) == FP
    _ckpt(tmp_path, 17)
    with pytest.raises(RuntimeError, match="--resume"):
        prepare_resume(tmp_path, FP, resume=False)             # không train đè lên checkpoint cũ
    assert prepare_resume(tmp_path, FP, resume=True).name == "checkpoint-17"
    changed = run_fingerprint(CONFIG, {**AUDIT, "train_sha256": "khac"})
    with pytest.raises(RuntimeError, match="train_sha256"):
        prepare_resume(tmp_path, changed, resume=True)         # dữ liệu đã đổi
    (tmp_path / RESUME_FINGERPRINT).unlink()
    with pytest.raises(RuntimeError):
        prepare_resume(tmp_path, FP, resume=True)              # không rõ checkpoint train với gì


def test_mirror_then_resume_in_new_session(tmp_path):
    src, drive = tmp_path / "local" / "sft_x", tmp_path / "drive" / "sft_x"
    prepare_resume(src, FP, resume=True)
    _ckpt(src, 17)
    mirror_checkpoints(src, drive)
    _ckpt(src, 34)
    shutil.rmtree(src / "checkpoint-17")                       # Trainer xoá checkpoint cũ (save_total_limit)
    (drive / ".tmp-checkpoint-51").mkdir()                     # bản chép dở của lần trước
    mirror_checkpoints(src, drive)
    assert sorted(d.name for d in drive.iterdir() if d.is_dir()) == ["checkpoint-34"]
    assert (drive / RESUME_FINGERPRINT).exists()

    fresh = tmp_path / "new_session" / "sft_x"                 # runtime mới, chưa có gì
    pull_checkpoints(drive, fresh)
    assert prepare_resume(fresh, FP, resume=True).name == "checkpoint-34"
    _ckpt(fresh, 51)
    pull_checkpoints(drive, fresh)                             # đã có checkpoint: không ghi đè
    assert last_checkpoint(fresh).name == "checkpoint-51"


def test_mirror_dir_must_be_the_run_folder(tmp_path):
    check_mirror_dir(tmp_path / "sft_x", tmp_path / "drive" / "sft_x")
    with pytest.raises(ValueError):
        check_mirror_dir(tmp_path / "sft_x", tmp_path / "drive")
