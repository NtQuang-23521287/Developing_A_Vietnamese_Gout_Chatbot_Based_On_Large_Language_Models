from __future__ import annotations

import random
import traceback
from pathlib import Path
from statistics import mean
from typing import Any

from gout_llmops.chat.engine import ChatEngine
from gout_llmops.core.config import load_yaml
from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import write_json, write_jsonl
from gout_llmops.core.run_context import RunContext, git_commit
from gout_llmops.data.testset import load_cases, summarize_cases
from gout_llmops.evaluation import metrics as metric_lib
from gout_llmops.evaluation.heuristic_reward import compute_reward
from gout_llmops.evaluation.judge import OpenAIJudge
from gout_llmops.models.registry import build_backend
from gout_llmops.rag.retriever import FaissRetriever
from gout_llmops.tracking.mlflow_tracker import mlflow_run

METRIC_KEYS = (
    "faithfulness",
    "completeness",
    "relevance",
    "context_recall",
    "patient_utility",
    "refusal_appropriateness",
    "hallucination_level",
    "omission_risk",
    "safety_risk",
)

REQUIRED_METRIC_FUNCTIONS = (
    "faithfulness",
    "completeness",
    "cosine_relevance",
    "context_recall",
    "patient_utility",
    "refusal_appropriateness",
    "hallucination_level",
    "omission_risk",
    "safety_risk",
)

# judge.py phải trả các dữ liệu trung gian này để Python tính đúng công thức.
REQUIRED_JUDGE_COMPONENTS = (
    "total_claims",
    "entailed_claims",
    "unsupported_claims",
    "total_key_points",
    "covered_key_points",
    "context_entities",
    "answer_entities",
    "readability",
    "empathy",
    "actionability",
    "is_dangerous",
    "safe_refusal_score",
    "normal_response_score",
    "total_mandatory_warnings",
    "omitted_warnings",
    "safety_risk_score",
)


def _check_metric_module() -> None:
    missing = [
        name
        for name in REQUIRED_METRIC_FUNCTIONS
        if not hasattr(metric_lib, name)
    ]
    if missing:
        raise RuntimeError(
            "metrics.py chưa có các hàm chấm theo 9 công thức: "
            + ", ".join(missing)
        )


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _normalize_entity_set(values: Any) -> set[str]:
    if not values:
        return set()

    out: set[str] = set()
    for value in values:
        text = str(value).strip().casefold()
        if text:
            out.add(text)
    return out


def _validate_judge_components(raw: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_JUDGE_COMPONENTS if k not in raw]
    if missing:
        raise RuntimeError(
            "judge.py đang trả schema cũ hoặc thiếu dữ liệu trung gian. "
            "Thiếu: "
            + ", ".join(missing)
            + ". Runner này cần judge trả counts/entities/subscores, "
              "không trả trực tiếp 9 điểm 0..1."
        )

    integer_nonnegative = (
        "total_claims",
        "entailed_claims",
        "unsupported_claims",
        "total_key_points",
        "covered_key_points",
        "total_mandatory_warnings",
        "omitted_warnings",
    )

    for key in integer_nonnegative:
        value = int(raw[key])
        if value < 0:
            raise ValueError(f"Judge field {key} must be >= 0, got {value}")

    if int(raw["entailed_claims"]) > int(raw["total_claims"]):
        raise ValueError("entailed_claims > total_claims")

    if int(raw["unsupported_claims"]) > int(raw["total_claims"]):
        raise ValueError("unsupported_claims > total_claims")

    if int(raw["covered_key_points"]) > int(raw["total_key_points"]):
        raise ValueError("covered_key_points > total_key_points")

    if int(raw["omitted_warnings"]) > int(raw["total_mandatory_warnings"]):
        raise ValueError("omitted_warnings > total_mandatory_warnings")

    for key in ("readability", "empathy", "actionability"):
        value = float(raw[key])
        if not 1 <= value <= 5:
            raise ValueError(f"{key} must be in [1,5], got {value}")

    for key in (
        "safe_refusal_score",
        "normal_response_score",
        "safety_risk_score",
    ):
        value = float(raw[key])
        if not 0 <= value <= 5:
            raise ValueError(f"{key} must be in [0,5], got {value}")


def _compute_nine_metrics(
    *,
    question: str,
    answer: str,
    judge_components: dict[str, Any],
    retriever: FaissRetriever,
) -> dict[str, float]:
    """
    Tính 9 tiêu chí đúng theo protocol:

    M1 Faithfulness
    M2 Completeness
    M3 Relevance
    M4 Context Recall
    M5 Patient Utility
    M6 Refusal Appropriateness
    P1 Hallucination Level
    P2 Omission Risk
    P3 Safety Risk
    """
    _validate_judge_components(judge_components)

    # M3 được tính deterministic bằng chính embedding model của RAG.
    embeddings = retriever.encoder.encode(
        [question, answer],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    question_embedding = embeddings[0]
    answer_embedding = embeddings[1]

    scores = {
        "faithfulness": metric_lib.faithfulness(
            total_claims=int(judge_components["total_claims"]),
            entailed_claims=int(judge_components["entailed_claims"]),
        ),
        "completeness": metric_lib.completeness(
            total_key_points=int(judge_components["total_key_points"]),
            covered_key_points=int(judge_components["covered_key_points"]),
        ),
        "relevance": metric_lib.cosine_relevance(
            question_embedding,
            answer_embedding,
        ),
        "context_recall": metric_lib.context_recall(
            _normalize_entity_set(judge_components["context_entities"]),
            _normalize_entity_set(judge_components["answer_entities"]),
        ),
        "patient_utility": metric_lib.patient_utility(
            readability_score=float(judge_components["readability"]),
            empathy_score=float(judge_components["empathy"]),
            actionability_score=float(judge_components["actionability"]),
        ),
        "refusal_appropriateness": metric_lib.refusal_appropriateness(
            is_dangerous=bool(judge_components["is_dangerous"]),
            safe_refusal_score=float(
                judge_components["safe_refusal_score"]
            ),
            normal_response_score=float(
                judge_components["normal_response_score"]
            ),
        ),
        "hallucination_level": metric_lib.hallucination_level(
            total_claims=int(judge_components["total_claims"]),
            unsupported_claims=int(
                judge_components["unsupported_claims"]
            ),
        ),
        "omission_risk": metric_lib.omission_risk(
            total_mandatory_warnings=int(
                judge_components["total_mandatory_warnings"]
            ),
            omitted_warnings=int(judge_components["omitted_warnings"]),
        ),
        "safety_risk": metric_lib.safety_risk(
            float(judge_components["safety_risk_score"])
        ),
    }

    for key, value in scores.items():
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"Computed metric {key} outside [0,1]: {value}"
            )
        scores[key] = value

    return scores


def _aggregate_score_records(
    score_records: list[dict[str, Any]],
) -> dict[str, Any]:
    ok = [r for r in score_records if r.get("status") == "ok"]

    metric_means: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        values = [
            float(r["metrics"][key])
            for r in ok
            if r.get("metrics", {}).get(key) is not None
        ]
        metric_means[key] = _mean_or_none(values)

    rewards = [
        float(r["reward"])
        for r in ok
        if r.get("reward") is not None
    ]
    quality_components = [
        float(r["quality_component"])
        for r in ok
        if r.get("quality_component") is not None
    ]
    risk_penalties = [
        float(r["risk_penalty"])
        for r in ok
        if r.get("risk_penalty") is not None
    ]

    fatal_count = sum(
        bool(r.get("fatal_safety_violation", False))
        for r in ok
    )

    return {
        "score_records": len(score_records),
        "scored_ok": len(ok),
        "scoring_errors": len(score_records) - len(ok),
        "scoring_success_rate": (
            len(ok) / len(score_records)
            if score_records
            else 0.0
        ),
        "metric_means": metric_means,
        "reward_mean": _mean_or_none(rewards),
        "quality_component_mean": _mean_or_none(
            quality_components
        ),
        "risk_penalty_mean": _mean_or_none(risk_penalties),
        "fatal_safety_violations": fatal_count,
        "fatal_safety_violation_rate": (
            fatal_count / len(ok)
            if ok
            else None
        ),
    }


def run_baseline(
    model_config_path: str,
    eval_config_path: str,
    limit: int | None = None,
    enable_judge: bool | None = None,
    enable_ragas: bool | None = None,
) -> Path:
    model_cfg = load_yaml(model_config_path)
    eval_cfg = load_yaml(eval_config_path)
    rag_cfg = load_yaml(eval_cfg["rag"]["config"])

    reward_config_path = (
        eval_cfg.get("reward", {}).get(
            "config",
            "configs/evaluation/heuristic_reward_v1.yaml",
        )
    )
    reward_cfg = load_yaml(reward_config_path)

    seed = int(eval_cfg.get("seed", 2026))
    _set_all_seeds(seed)

    # Khóa max_new_tokens từ Evaluation Protocol để 3 model dùng cùng điều kiện.
    model_cfg = dict(model_cfg)
    model_cfg["generation"] = dict(model_cfg.get("generation", {}))
    if eval_cfg.get("generation", {}).get("max_new_tokens") is not None:
        model_cfg["generation"]["max_new_tokens"] = int(
            eval_cfg["generation"]["max_new_tokens"]
        )

    cases = load_cases(eval_cfg["testset"]["path"])
    if limit is not None:
        cases = cases[:limit]

    judge_enabled = bool(
        enable_judge
        if enable_judge is not None
        else eval_cfg.get("judge", {}).get("enabled", False)
    )
    ragas_enabled = bool(
        enable_ragas
        if enable_ragas is not None
        else eval_cfg.get("ragas", {}).get("enabled", False)
    )

    if judge_enabled:
        _check_metric_module()

    # RAGAS chưa được dùng để thay thế công thức 9 tiêu chí trong runner này.
    # Giữ flag để tương thích CLI và log protocol.
    if ragas_enabled:
        print(
            "[INFO] RAGAS requested. The 9-metric scoring below still "
            "uses the thesis formulas; RAGAS can be logged separately."
        )

    run = RunContext.create(
        eval_cfg["output"]["root"],
        model_cfg["name"],
    )

    write_json(
        run.output_dir / "model_config.json",
        model_cfg,
    )
    write_json(
        run.output_dir / "eval_config.json",
        eval_cfg,
    )
    write_json(
        run.output_dir / "rag_config.json",
        rag_cfg,
    )
    write_json(
        run.output_dir / "reward_config.json",
        reward_cfg,
    )

    kb_manifest_path = (
        Path(eval_cfg["rag"]["index_dir"]) / "manifest.json"
    )

    manifest = {
        "run_id": run.run_id,
        "phase": eval_cfg["phase"],
        "model_name": model_cfg["name"],
        "model_id": model_cfg["model_id"],
        "model_revision": model_cfg.get("revision"),
        "git_commit": git_commit(),
        "model_config_sha256": sha256_file(model_config_path),
        "eval_config_sha256": sha256_file(eval_config_path),
        "rag_config_sha256": sha256_file(
            eval_cfg["rag"]["config"]
        ),
        "reward_config_path": reward_config_path,
        "reward_config_sha256": sha256_file(
            reward_config_path
        ),
        "testset_path": eval_cfg["testset"]["path"],
        "testset_sha256": sha256_file(
            eval_cfg["testset"]["path"]
        ),
        "kb_manifest_sha256": sha256_file(
            kb_manifest_path
        ),
        "prompt_sha256": sha256_file(
            eval_cfg["prompt"]["system"]
        ),
        "judge_enabled": judge_enabled,
        "ragas_enabled": ragas_enabled,
        "seed": seed,
        "case_summary": summarize_cases(cases),
    }

    if judge_enabled:
        manifest["judge_model"] = eval_cfg["judge"]["model"]
        manifest["judge_prompt_sha256"] = sha256_file(
            eval_cfg["judge"]["prompt"]
        )
        manifest["judge_rubric_sha256"] = sha256_file(
            eval_cfg["judge"]["rubric"]
        )

    write_json(
        run.output_dir / "run_manifest.json",
        manifest,
    )

    backend = build_backend(model_cfg)
    retriever = FaissRetriever(
        eval_cfg["rag"]["index_dir"],
        rag_cfg,
    )

    system_prompt = Path(
        eval_cfg["prompt"]["system"]
    ).read_text(encoding="utf-8")

    engine = ChatEngine(
        backend,
        retriever,
        system_prompt,
        max_history_pairs=int(
            eval_cfg["conversation"].get(
                "max_history_pairs",
                2,
            )
        ),
        max_input_tokens=int(
            eval_cfg["input"].get(
                "max_input_tokens",
                4096,
            )
        ),
    )

    judge = (
        OpenAIJudge(eval_cfg["judge"])
        if judge_enabled
        else None
    )

    records: list[dict[str, Any]] = []
    retrieval_trace: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    stop_case_ids: set[str] = set()

    params = {
        "model_id": model_cfg["model_id"],
        "model_revision": model_cfg.get("revision"),
        "testset_sha256": manifest["testset_sha256"],
        "kb_manifest_sha256": manifest[
            "kb_manifest_sha256"
        ],
        "prompt_sha256": manifest["prompt_sha256"],
        "reward_config_sha256": manifest[
            "reward_config_sha256"
        ],
        "judge_enabled": judge_enabled,
        "judge_model": (
            eval_cfg["judge"].get("model")
            if judge_enabled
            else None
        ),
        "seed": seed,
    }

    with mlflow_run(
        eval_cfg.get("tracking", {}),
        run.run_id,
        params,
    ) as mlflow:
        for case_index, case in enumerate(cases, start=1):
            history: list[dict[str, str]] = []

            for turn in case.turns:
                if case.case_id in stop_case_ids:
                    records.append(
                        {
                            "case_id": case.case_id,
                            "scenario": case.scenario,
                            "subset": case.subset,
                            "turn_id": turn.turn_id,
                            "question": turn.user,
                            "status": (
                                "skipped_dependency_failure"
                            ),
                        }
                    )
                    continue

                try:
                    result, history = engine.chat(
                        turn.user,
                        history,
                    )

                    record = {
                        "case_id": case.case_id,
                        "scenario": case.scenario,
                        "subset": case.subset,
                        "target_disease": (
                            case.target_disease
                        ),
                        "source_row_id": (
                            case.source_row_id
                        ),
                        "turn_id": turn.turn_id,
                        "reference_status": (
                            case.reference_status
                        ),
                        "status": "ok",
                        **{
                            k: v
                            for k, v in result.items()
                            if k != "messages"
                        },
                    }
                    records.append(record)

                    retrieval_trace.append(
                        {
                            "case_id": case.case_id,
                            "turn_id": turn.turn_id,
                            "question": turn.user,
                            "contexts": result[
                                "contexts"
                            ],
                        }
                    )

                    # -----------------------------
                    # 9-metric evaluation
                    # -----------------------------
                    if judge is not None:
                        try:
                            judge_sample = {
                                "question": turn.user,
                                "answer": result["answer"],
                                "contexts": result[
                                    "contexts"
                                ],
                            }

                            # judge.py mới phải trả component counts /
                            # entities / subscores theo schema phía trên.
                            judge_components = judge.score(
                                judge_sample,
                                reference=None,
                            )

                            _validate_judge_components(
                                judge_components
                            )

                            nine_metrics = (
                                _compute_nine_metrics(
                                    question=turn.user,
                                    answer=result["answer"],
                                    judge_components=(
                                        judge_components
                                    ),
                                    retriever=retriever,
                                )
                            )

                            # Fatal Safety Gate:
                            # mặc định theta=0.8 nếu config chưa khai báo.
                            fatal_cfg = reward_cfg.get(
                                "fatal_safety",
                                {},
                            )
                            theta_critical = float(
                                fatal_cfg.get(
                                    "theta_critical",
                                    reward_cfg.get(
                                        "theta_critical",
                                        0.80,
                                    ),
                                )
                            )

                            fatal_violation = (
                                nine_metrics[
                                    "safety_risk"
                                ]
                                >= theta_critical
                            )

                            reward_input = {
                                **nine_metrics,
                                "fatal_safety_violation": (
                                    fatal_violation
                                ),
                            }

                            # Tương thích cả config cũ
                            # fatal_penalty: 2.0
                            # và config mới:
                            # fatal_safety:
                            #   penalty: 2.0
                            reward_runtime_cfg = dict(
                                reward_cfg
                            )
                            if (
                                "fatal_penalty"
                                not in reward_runtime_cfg
                            ):
                                reward_runtime_cfg[
                                    "fatal_penalty"
                                ] = float(
                                    fatal_cfg.get(
                                        "penalty",
                                        2.0,
                                    )
                                )

                            reward_result = compute_reward(
                                reward_input,
                                reward_runtime_cfg,
                            )

                            score_record = {
                                "case_id": case.case_id,
                                "scenario": case.scenario,
                                "subset": case.subset,
                                "target_disease": (
                                    case.target_disease
                                ),
                                "turn_id": turn.turn_id,
                                "status": "ok",
                                "question": turn.user,
                                "answer": result["answer"],
                                "context_chunk_ids": [
                                    c.get("chunk_id")
                                    for c in result[
                                        "contexts"
                                    ]
                                ],
                                "metrics": nine_metrics,
                                "judge_components": (
                                    judge_components
                                ),
                                "theta_critical": (
                                    theta_critical
                                ),
                                "fatal_safety_violation": (
                                    fatal_violation
                                ),
                                **reward_result,
                            }
                            score_records.append(
                                score_record
                            )

                        except Exception as scoring_error:
                            score_records.append(
                                {
                                    "case_id": (
                                        case.case_id
                                    ),
                                    "scenario": (
                                        case.scenario
                                    ),
                                    "subset": case.subset,
                                    "target_disease": (
                                        case.target_disease
                                    ),
                                    "turn_id": (
                                        turn.turn_id
                                    ),
                                    "status": (
                                        "scoring_error"
                                    ),
                                    "question": (
                                        turn.user
                                    ),
                                    "answer": result[
                                        "answer"
                                    ],
                                    "error_type": type(
                                        scoring_error
                                    ).__name__,
                                    "error": str(
                                        scoring_error
                                    ),
                                    "traceback": (
                                        traceback.format_exc(
                                            limit=3
                                        )
                                    ),
                                }
                            )

                except Exception as generation_error:
                    records.append(
                        {
                            "case_id": case.case_id,
                            "scenario": case.scenario,
                            "subset": case.subset,
                            "target_disease": (
                                case.target_disease
                            ),
                            "turn_id": turn.turn_id,
                            "question": turn.user,
                            "status": "error",
                            "error_type": type(
                                generation_error
                            ).__name__,
                            "error": str(
                                generation_error
                            ),
                            "traceback": (
                                traceback.format_exc(
                                    limit=3
                                )
                            ),
                        }
                    )

                    if case.scenario == "multi":
                        stop_case_ids.add(
                            case.case_id
                        )

            # Checkpoint sau mỗi case để crash giữa chừng
            # vẫn giữ được dữ liệu đã chạy.
            write_jsonl(
                run.output_dir / "predictions.jsonl",
                records,
            )
            write_jsonl(
                run.output_dir
                / "retrieval_trace.jsonl",
                retrieval_trace,
            )
            if judge_enabled:
                write_jsonl(
                    run.output_dir
                    / "scores_9_metrics.jsonl",
                    score_records,
                )

            print(
                f"[{case_index}/{len(cases)}] "
                f"{case.case_id} done"
            )

        # -----------------------------
        # Aggregate runtime
        # -----------------------------
        summary = metric_lib.aggregate_predictions(
            records
        )

        summary["judge_enabled"] = judge_enabled
        summary["ragas_requested"] = ragas_enabled
        summary["scoring_protocol"] = (
            "medical_9_metrics_formula_v1"
        )

        # -----------------------------
        # Aggregate 9 metrics
        # -----------------------------
        if judge_enabled:
            score_summary = _aggregate_score_records(
                score_records
            )
            summary.update(score_summary)

            for metric_name, metric_value in (
                score_summary["metric_means"].items()
            ):
                summary[
                    f"{metric_name}_mean"
                ] = metric_value

        write_json(
            run.output_dir / "summary.json",
            summary,
        )

        # -----------------------------
        # MLflow
        # -----------------------------
        if mlflow:
            for key, value in summary.items():
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and value is not None
                ):
                    mlflow.log_metric(
                        key,
                        float(value),
                    )

            if judge_enabled:
                for metric_name, metric_value in (
                    summary.get(
                        "metric_means",
                        {}
                    ).items()
                ):
                    if metric_value is not None:
                        mlflow.log_metric(
                            f"mean_{metric_name}",
                            float(metric_value),
                        )

            mlflow.log_artifacts(
                str(run.output_dir),
                artifact_path="run_artifacts",
            )

    return run.output_dir
