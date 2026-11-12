import argparse

from gout_llmops.evaluation.runner import run_baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="configs/evaluation/baseline_v1.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judge", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--ragas", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()
    out = run_baseline(
        args.model,
        args.eval,
        limit=args.limit,
        enable_judge=args.judge,
        enable_ragas=args.ragas,
    )
    print(f"Run completed: {out}")


if __name__ == "__main__":
    main()
