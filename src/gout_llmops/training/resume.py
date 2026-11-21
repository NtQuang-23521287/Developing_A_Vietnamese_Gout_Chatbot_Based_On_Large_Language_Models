"""Train tiếp sau khi Colab ngắt, dùng chung cho SFT và DPO.

Checkpoint được chép sang thư mục bền (Drive) mỗi lần Trainer lưu; phiên sau lấy về rồi Trainer chạy tiếp từ đó.
Chỉ train tiếp khi dữ liệu và config giống hệt lần trước (fingerprint do từng pipeline tự định nghĩa).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

RESUME_FINGERPRINT = "resume_fingerprint.json"
_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def _checkpoints(directory: Path) -> list[Path]:
    """checkpoint-N đã lưu xong (có trainer_state.json), N tăng dần."""
    if not directory.is_dir():
        return []
    found = [(int(m.group(1)), d) for d in directory.iterdir()
             if d.is_dir() and (m := _CHECKPOINT.match(d.name)) and (d / "trainer_state.json").exists()]
    return [d for _, d in sorted(found)]


def last_checkpoint(directory: Path) -> Path | None:
    found = _checkpoints(directory)
    return found[-1] if found else None


def prepare_resume(output_dir: Path, fingerprint: dict[str, Any], resume: bool) -> Path | None:
    """Checkpoint để train tiếp (None = train từ đầu); ghi fingerprint của lần train này."""
    last, path = last_checkpoint(output_dir), output_dir / RESUME_FINGERPRINT
    if last is not None:
        if not resume:
            raise RuntimeError(f"{output_dir} đã có {last.name} của lần train trước: thêm --resume để train "
                               "tiếp, hoặc xoá thư mục này để train lại từ đầu")
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        diff = sorted(k for k in fingerprint if old.get(k) != fingerprint[k])
        if diff:
            raise RuntimeError(f"Không train tiếp từ {last.name}: khác lần train trước ở {', '.join(diff)} "
                               f"(hoặc thiếu {RESUME_FINGERPRINT}). Xoá {output_dir} để train lại từ đầu")
        print(f"Train tiếp từ {last.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    return last


def check_mirror_dir(output_dir: Path, mirror: Path) -> None:
    # Bản sao phải là thư mục riêng của run này: mirror_checkpoints xoá checkpoint cũ trong đó
    if mirror.name != output_dir.name:
        raise ValueError(f"--checkpoint-mirror phải là thư mục tên {output_dir.name}, nhận {mirror}")


def mirror_checkpoints(src: Path, dst: Path) -> None:
    """Chép file gốc và checkpoint mới của src sang dst. Checkpoint được chép vào tên tạm rồi mới đổi
    tên, nên dst không bao giờ có checkpoint dở dang; checkpoint Trainer đã xoá ở src cũng bị xoá ở dst."""
    import shutil
    dst.mkdir(parents=True, exist_ok=True)
    for stale in [*dst.glob(".tmp-checkpoint-*"), *src.glob(".tmp-checkpoint-*")]:
        shutil.rmtree(stale, ignore_errors=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    keep = {d.name for d in _checkpoints(src)}
    for name in sorted(keep - {d.name for d in _checkpoints(dst)}):
        tmp = dst / f".tmp-{name}"
        shutil.copytree(src / name, tmp)
        tmp.rename(dst / name)
    for d in _checkpoints(dst):
        if d.name not in keep:
            shutil.rmtree(d)


def pull_checkpoints(mirror: Path, output_dir: Path) -> None:
    """Phiên mới: output_dir chưa có checkpoint thì lấy từ bản sao."""
    import shutil
    if last_checkpoint(output_dir) is not None or last_checkpoint(mirror) is None:
        return
    for d in _checkpoints(mirror):
        shutil.copytree(d, output_dir / d.name, dirs_exist_ok=True)
    if (mirror / RESUME_FINGERPRINT).exists():
        shutil.copy2(mirror / RESUME_FINGERPRINT, output_dir / RESUME_FINGERPRINT)
    print(f"Lấy {len(_checkpoints(output_dir))} checkpoint từ {mirror}")


def mirror_callback(output_dir: Path, mirror: Path):
    """TrainerCallback chép output_dir sang bản sao mỗi lần Trainer lưu checkpoint. Lỗi chép (Drive trục trặc) chỉ
    in cảnh báo: train vẫn chạy tiếp, chỉ mất khả năng train tiếp ở phiên sau."""
    from transformers import TrainerCallback

    class MirrorCheckpoints(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            mirror_now(output_dir, mirror, f"checkpoint bước {state.global_step}")
            return control

    return MirrorCheckpoints()


def mirror_now(output_dir: Path, mirror: Path, what: str = "adapter") -> None:
    try:
        mirror_checkpoints(output_dir, mirror)
        print(f"Đã chép {what} sang {mirror}")
    except OSError as exc:
        print(f"[!] Không chép được sang {mirror}: {exc}")
