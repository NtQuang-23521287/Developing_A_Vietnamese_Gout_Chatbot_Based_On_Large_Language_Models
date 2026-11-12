# Thesis Phase Mapping

## Phase 1 — Data & Evaluation Infrastructure

Definition of done:
- source registry/manifests;
- KB v1 build reproducibly;
- candidate testset structurally validated;
- common ChatEngine;
- Evaluation Runner writes reproducible artifacts;
- MLflow tracking works;
- smoke run passes.

## Phase 2 — Baseline Model Selection

Run SeaLLMs-v3-7B-Chat, Qwen3-8B, Gemma-3-4B-it on the same frozen protocol. Select Best Model only after quantitative results + error analysis.

## Phase 3 — SFT

Create/version SFT dataset, QLoRA train Best Model, re-run the same Evaluation Harness. Do not change testset to favor SFT.

## Phase 4 — DPO

Generate K candidates, score with heuristic reward, mine Chosen/Rejected, DPO train, re-run harness. Version preference data and adapter.

## Phase 5 — Safety & Release

Calibrate uncertainty threshold on development data, enable safe refusal, run final acceptance on a separately managed final test set, then package API/UI.

Safe Inference v1 (guard theo luật + MC Dropout trên LoRA + hiệu chỉnh ngưỡng trên dev): `docs/safe_inference.md`,
`scripts/safe_inference.py`, `notebooks/05_safe_inference_colab.ipynb`.
