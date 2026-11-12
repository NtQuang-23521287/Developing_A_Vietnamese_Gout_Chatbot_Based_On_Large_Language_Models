from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


@dataclass(frozen=True)
class RunContext:
    run_id: str
    output_dir: Path

    @classmethod
    def create(cls, root: str | Path, model_name: str, prefix: str = "baseline") -> RunContext:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_model = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        run_id = f"{stamp}_{prefix}_{safe_model}_{os.getpid()}"
        out = Path(root) / run_id
        out.mkdir(parents=True, exist_ok=False)
        return cls(run_id=run_id, output_dir=out)
