from __future__ import annotations

from pathlib import Path

from gout_llmops.core.io import read_jsonl


def load_reference_map(path: str | Path, require_approved: bool = True) -> dict[tuple[str, int], dict]:
    refs = {}
    for row in read_jsonl(path):
        key = (str(row["case_id"]), int(row["turn_id"]))
        if key in refs:
            raise ValueError(f"Duplicate reference: {key}")
        if require_approved and row.get("review_status") != "approved":
            raise ValueError(f"Reference {key} is not approved")
        refs[key] = row
    return refs
