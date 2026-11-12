from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gout_llmops.core.io import read_jsonl


@dataclass(frozen=True)
class Turn:
    turn_id: int
    user: str


@dataclass(frozen=True)
class Case:
    case_id: str
    scenario: str
    subset: str
    target_disease: str
    source_row_id: int
    reference_status: str
    turns: tuple[Turn, ...]
    # --- v03 (tùy chọn; v02 không có thì để mặc định) ---
    question_type: str = "unknown"
    risk_level: str = "unknown"
    risk_weight: float = 1.0
    kb_support: str = "unknown"
    expected_behavior: str = "unknown"
    split: str | None = None
    # CHỈ evaluator được dùng. Không đưa vào prompt hay retrieval (tránh leakage).
    reference: dict[str, Any] = field(default_factory=dict)

    @property
    def is_safety_trap(self) -> bool:
        return self.question_type == "safety_trap"


def load_cases(path: str | Path, split: str | None = None) -> list[Case]:
    raw = read_jsonl(path)
    if split is not None:
        raw = [r for r in raw if r.get("split") == split]
    cases: list[Case] = []
    seen = set()
    for r in raw:
        case_id = str(r["case_id"])
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        turns = tuple(Turn(int(t["turn_id"]), str(t["user"]).strip()) for t in r["turns"])
        scenario = str(r["scenario"])
        expected = 1 if scenario == "single" else 3 if scenario == "multi" else None
        if expected is None:
            raise ValueError(f"Unknown scenario {scenario} in {case_id}")
        if len(turns) != expected:
            raise ValueError(f"{case_id}: {scenario} must have {expected} turns")
        if any(not t.user for t in turns):
            raise ValueError(f"{case_id}: empty user turn")
        cases.append(
            Case(
                case_id=case_id,
                scenario=scenario,
                subset=str(r.get("subset", "primary")),
                target_disease=str(r.get("target_disease", "")),
                source_row_id=int(r.get("source_row_id", -1)),
                reference_status=str(r.get("reference_status", "unknown")),
                turns=turns,
                question_type=str(r.get("labels", {}).get("question_type", "unknown")),
                risk_level=str(r.get("labels", {}).get("risk_level", "unknown")),
                risk_weight=float(r.get("labels", {}).get("risk_weight", 1.0)),
                kb_support=str(r.get("labels", {}).get("kb_support", "unknown")),
                expected_behavior=str(r.get("labels", {}).get("expected_behavior", "unknown")),
                split=r.get("split"),
                reference=dict(r.get("reference", {})),
            )
        )
    return cases


def summarize_cases(cases: list[Case]) -> dict[str, Any]:
    by_scenario: dict[str, int] = {}
    by_subset: dict[str, int] = {}
    by_qtype: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    turns = 0
    for c in cases:
        by_scenario[c.scenario] = by_scenario.get(c.scenario, 0) + 1
        by_subset[c.subset] = by_subset.get(c.subset, 0) + 1
        by_qtype[c.question_type] = by_qtype.get(c.question_type, 0) + 1
        by_risk[c.risk_level] = by_risk.get(c.risk_level, 0) + 1
        turns += len(c.turns)
    return {
        "cases": len(cases),
        "turns": turns,
        "by_scenario": by_scenario,
        "by_subset": by_subset,
        "by_question_type": by_qtype,
        "by_risk_level": by_risk,
    }
