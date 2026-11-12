import argparse
from pathlib import Path

from gout_llmops.core.config import load_yaml
from gout_llmops.core.io import write_jsonl
from gout_llmops.data.testset import load_cases
from gout_llmops.rag.retriever import FaissRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag", default="configs/rag/rag_v1.yaml")
    ap.add_argument("--testset", default="data/raw/testset/Group3_Testset_cases_v02.jsonl")
    ap.add_argument("--output", default="artifacts/review/testset_kb_alignment_v1.jsonl")
    args = ap.parse_args()
    cfg = load_yaml(args.rag)
    retriever = FaissRetriever(cfg["output_dir"], cfg)
    rows = []
    for case in load_cases(args.testset):
        for turn in case.turns:
            contexts = retriever.retrieve(turn.user)
            rows.append({
                "case_id": case.case_id,
                "turn_id": turn.turn_id,
                "scenario": case.scenario,
                "subset": case.subset,
                "question": turn.user,
                "retrieved_contexts": contexts,
                "alignment_decision": "pending",
                "reference_answer": None,
                "required_points": [],
                "evidence_chunk_ids": [],
                "reviewer": None,
                "review_status": "pending",
            })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
