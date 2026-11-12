from __future__ import annotations

QUALITY_KEYS = (
    "faithfulness",
    "completeness",
    "relevance",
    "context_recall",
    "patient_utility",
    "refusal_appropriateness",
)

RISK_KEYS = (
    "hallucination_level",
    "omission_risk",
    "safety_risk",
)


def compute_reward(
    metrics: dict[str, float],
    config: dict,
) -> dict:
    q_weights = config["quality_weights"]
    risk_weights = config["risk_penalties"]

    quality_component = sum(
        float(q_weights[key]) * float(metrics[key])
        for key in QUALITY_KEYS
    )

    risk_component = sum(
        float(risk_weights[key]) * float(metrics[key])
        for key in RISK_KEYS
    )

    p3 = float(metrics["safety_risk"])

    fatal_cfg = config["fatal_safety"]
    fatal = (
        bool(fatal_cfg.get("enabled", True))
        and p3 >= float(fatal_cfg["theta_critical"])
    )

    fatal_penalty = (
        float(fatal_cfg["penalty"])
        if fatal
        else 0.0
    )

    reward = (
        quality_component
        - risk_component
        - fatal_penalty
    )

    return {
        "quality_component": quality_component,
        "risk_penalty": risk_component,
        "fatal_safety_violation": fatal,
        "fatal_penalty": fatal_penalty,
        "reward": reward,
    }