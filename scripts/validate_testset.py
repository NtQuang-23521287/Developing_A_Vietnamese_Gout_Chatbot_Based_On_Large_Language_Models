from pathlib import Path

from gout_llmops.core.io import write_json
from gout_llmops.data.testset import load_cases, summarize_cases


def main():
    path = "data/data_filtered_final/Group3_Testset_cases_v02.jsonl"
    cases = load_cases(path)
    source_ids = [c.source_row_id for c in cases]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("A source_row_id appears in more than one case; ST/MT leakage detected")
    summary = summarize_cases(cases)
    summary["source_row_unique"] = len(set(source_ids))
    summary["status"] = "structurally_valid_candidate"
    summary["warning"] = "References are pending_kb_alignment; not a gold clinical testset yet."
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    write_json("artifacts/reports/testset_validation.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
