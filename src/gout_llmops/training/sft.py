"""Two-model text-only QLoRA SFT, using existing Phase-1 checkpoints.

Use entrypoint scripts/train_sft.py from repository root. Data are JSONL
``{"messages": [{"role":"user","content":"..."}, ...]}``.
We create labels explicitly: -100 for prompt, real IDs for the assistant reply.
Gemma3 is multimodal, but this training pipeline is **text-only**.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODELS = {
    "qwen3": {"id": "Qwen/Qwen3-8B", "file": "configs/models/qwen3_8b.yaml", "end": "<|im_end|>"},
    "gemma3": {"id": "google/gemma-3-4b-it", "file": "configs/models/gemma3_4b.yaml", "end": "<end_of_turn>"},
}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def identify_model(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    from gout_llmops.core.config import load_yaml
    model_cfg_path = resolve_path(config["base_model_config"])
    model_cfg = load_yaml(model_cfg_path)
    model_id = model_cfg.get("model_id")
    variants = [key for key, spec in MODELS.items() if spec["id"] == model_id]
    if len(variants) != 1:
        raise ValueError(f"Unsupported Phase-1 model_id={model_id!r}; expected Qwen3-8B or Gemma3-4B")
    family = variants[0]
    if not model_cfg.get("revision"):
        raise ValueError(f"Must pin revision in {model_cfg_path}")
    expected = config.get("model_family")
    if expected and expected != family:
        raise ValueError(f"model_family={expected!r} doesn't match base model {model_id!r}")
    if family == "qwen3" and model_cfg.get("chat_template_kwargs", {}).get("enable_thinking") is not False:
        raise ValueError("Qwen3 baseline requires chat_template_kwargs.enable_thinking: false")
    if family == "gemma3" and model_cfg.get("chat_template_kwargs"):
        raise ValueError("Gemma3 baseline chat_template_kwargs should be {}")
    return family, model_cfg


def read_conversations(filepath: str | Path) -> list[list[dict[str, str]]]:
    path = resolve_path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)
    conversations = []
    for ln, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{ln}: malformed JSON: {exc}") from exc
        messages = record.get("messages") if isinstance(record, dict) else None
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"{path}:{ln}: expected messages list with at least two items")
        clean = []
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or msg.get("role") not in {"user", "system", "assistant"}:
                raise ValueError(f"{path}:{ln}: invalid role at messages[{i}]")
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                raise ValueError(f"{path}:{ln}: empty/non-text content at messages[{i}]")
            clean.append({"role": msg["role"], "content": msg["content"].strip()})
        if clean[-2]["role"] != "user" or clean[-1]["role"] != "assistant":
            raise ValueError(f"{path}:{ln}: last two messages must be user, assistant")
        conversations.append(clean)
    if not conversations:
        raise ValueError(f"Dataset is empty: {path}")
    return conversations


def normalize_question(messages: list[dict[str, str]]) -> str:
    return re.sub(r"\s+", " ", messages[-2]["content"].strip().casefold())


def check_overlap(train: list[list[dict[str, str]]], val: list[list[dict[str, str]]]) -> None:
    for name, collection in (("train", train), ("val", val)):
        counts = Counter(normalize_question(m) for m in collection)
        if any(v > 1 for v in counts.values()):
            print(f"WARNING: {name} contains {sum(v > 1 for v in counts.values())} internally repeated questions")
    common = {normalize_question(m) for m in train} & {normalize_question(m) for m in val}
    if common:
        raise ValueError(f"Train/val exact question leakage, e.g. {sorted(common)[0]}")


def encode_conversation(messages: list[dict[str, str]], tokenizer: Any,
                        family: str) -> dict[str, list[int]]:
    """Prompt is the model-native chat template; supervise only answer + end marker.

    Splitting prompt and answer *before* tokenization is intentional so that
    label boundaries are explicit. Both components are encoded without automatic
    extra BOS/EOS: chat template is responsible for their inclusion.
    """
    if family not in MODELS or messages[-1]["role"] != "assistant":
        raise ValueError("Unsupported model or conversation without final assistant")
    template_args = {"tokenize": False, "add_generation_prompt": True}
    if family == "qwen3":
        template_args["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template(messages[:-1], **template_args)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Tokenizer could not render chat prompt")
    # Explicitly ensure model-specific role header; prevents Qwen/Gemma mix-up.
    marker = "<|im_start|>assistant" if family == "qwen3" else "<start_of_turn>model"
    if marker not in prompt:
        raise ValueError(f"Wrong chat template for {family}: missing {marker}")
    eos_marker = MODELS[family]["end"]
    end_id = tokenizer.convert_tokens_to_ids(eos_marker)
    if not isinstance(end_id, int) or end_id < 0 or end_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Tokenizer does not recognize {eos_marker}")
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    answer = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    if not prefix or not answer:
        raise ValueError("Empty prefix or assistant answer after tokenization")
    ids = prefix + answer + [end_id]
    labels = [-100] * len(prefix) + answer + [end_id]
    assert len(ids) == len(labels) and -100 in labels and labels[-1] == end_id
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def preprocess(conversations: list[list[dict[str, str]]], tokenizer: Any,
               family: str, max_length: int, split: str, smoke: bool = False
               ) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    rows, too_long, lengths = [], [], []
    for row_idx, messages in enumerate(conversations, 1):
        sample = encode_conversation(messages, tokenizer, family)
        size = len(sample["input_ids"])
        lengths.append(size)
        if size > max_length:
            too_long.append({"row": row_idx, "tokens": size, "question": normalize_question(messages)[:110]})
        else:
            rows.append(sample)
    stats = {
        "split": split, "original_count": len(conversations), "fit_count": len(rows),
        "dropped_overlong": len(too_long), "mean_tokens": round(sum(lengths)/len(lengths), 1),
        "min_tokens": min(lengths), "max_tokens": max(lengths),
        "overlong_rows": too_long,
    }
    if smoke:
        rows = rows[:4] if split == "train" else rows[:2]
    if not rows:
        raise ValueError(f"{split}: no rows fit {max_length} tokens")
    return rows, stats


@dataclass
class AssistantOnlyCollator:
    pad_token_id: int

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        longest = max(len(x["input_ids"]) for x in samples)

        def pad(key, value):
            return torch.tensor([x[key] + [value] * (longest - len(x[key])) for x in samples], dtype=torch.long)

        return {
            "input_ids": pad("input_ids", self.pad_token_id),
            "attention_mask": pad("attention_mask", 0),
            "labels": pad("labels", -100),
        }


def _dataset_sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()


def _get_tok(model_id: str, revision: str, trust_remote_code: bool):
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code,
            use_fast=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot access {model_id} @{revision}. For Gemma 3: accept the license "
            "on Hugging Face and run `hf auth login` before retrying."
        ) from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError(f"Model {model_id}: missing padding token")
    return tokenizer


def _load_model(family: str, model_cfg: dict[str, Any], max_length: int,
                qlora: dict[str, Any], seed: int):
    """Keep original Phase-1 HF checkpoint; never silently switch Qwen to Gemma.

    NOTE: FastModel is essential for Gemma3 (conditional/multimodal), not
    FastLanguageModel. All Gemma image layers are frozen for text-only SFT.
    """
    import torch
    from unsloth import FastLanguageModel, FastModel

    model_id = model_cfg["model_id"]
    common = dict(
        model_name=model_id, revision=model_cfg["revision"],
        max_seq_length=max_length, load_in_4bit=True,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        use_exact_model_name=True,
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if family == "qwen3":
        model, returned_tokenizer = FastLanguageModel.from_pretrained(
            **common, dtype=dtype,
        )
    elif family == "gemma3":
        model, returned_tokenizer = FastModel.from_pretrained(
            **common, dtype=dtype, full_finetuning=False,
        )
    else:
        raise ValueError(family)
    tokenizer = getattr(returned_tokenizer, "tokenizer", returned_tokenizer)
    common_lora = dict(
        r=int(qlora.get("lora_r", 16)),
        lora_alpha=int(qlora.get("lora_alpha", 32)),
        lora_dropout=float(qlora.get("lora_dropout", 0.0)),
        bias="none", random_state=seed,
        use_gradient_checkpointing="unsloth",
    )
    if family == "qwen3":
        model = FastLanguageModel.get_peft_model(
            model,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            **common_lora,
        )
    else:
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            **common_lora,
        )
    return model, tokenizer


def train_sft(config: dict[str, Any], *, preflight_only: bool = False,
              smoke_test: bool = False, allow_draft: bool = False) -> None:
    family, model_cfg = identify_model(config)
    train_file = config.get("train_data", "data/sft/train.jsonl")
    val_file = config.get("val_data", "data/sft/val.jsonl")
    output_dir = resolve_path(config.get("output_dir", f"artifacts/runs/sft_{family}"))
    max_length = int(config.get("max_seq_length", 2048))
    if max_length < 128:
        raise ValueError("max_seq_length must be >= 128")
    policy = config.get("long_example_policy", "error")
    if policy not in {"error", "drop"}:
        raise ValueError("long_example_policy must be error or drop")
    medical_reviewed = config.get("medical_answers_approved") is True
    if not (preflight_only or smoke_test or medical_reviewed or allow_draft):
        raise RuntimeError(
            "Medical QA dataset is DRAFT. For research-only experiments use --allow-draft; "
            "for reviewed production data set medical_answers_approved: true with evidence."
        )
    if not preflight_only:
        # Unsloth MUST be imported before transformers/trainer on GPU.
        import unsloth  # noqa: F401
    print(f"=== Phase 3 SFT, family={family}, model={model_cfg['model_id']} ===")
    train_convs, val_convs = read_conversations(train_file), read_conversations(val_file)
    check_overlap(train_convs, val_convs)
    tokenizer = _get_tok(model_cfg["model_id"], model_cfg["revision"], bool(model_cfg.get("trust_remote_code", False)))
    train_rows, train_report = preprocess(train_convs, tokenizer, family, max_length, "train", smoke=smoke_test)
    val_rows, val_report = preprocess(val_convs, tokenizer, family, max_length, "val", smoke=smoke_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "model_family": family, "model_id": model_cfg["model_id"],
        "revision": model_cfg["revision"], "max_seq_length": max_length,
        "medical_answers_approved": medical_reviewed,
        "research_only_draft": not medical_reviewed,
        "train_sha256": _dataset_sha256(train_file),
        "val_sha256": _dataset_sha256(val_file),
        "train": train_report, "val": val_report,
        "length_policy": policy,
    }
    audit_path = output_dir / ("audit_smoke.json" if smoke_test else "audit_preflight.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dataset loaded: train={len(train_convs)}, val={len(val_convs)}. Full token-length audit: {audit_path}")
    for entry in [train_report, val_report]:
        print(f"  {entry['split']}: fits={entry['fit_count']}/{entry['original_count']}; "
              f"overlong={entry['dropped_overlong']}; max={entry['max_tokens']} tokens")
    if preflight_only:
        print("PRECHECK COMPLETE (CPU/tokenizer only; no training)")
        return
    if policy == "error" and not smoke_test and (train_report["dropped_overlong"] or val_report["dropped_overlong"]):
        raise RuntimeError(f"Overlong records detected. See {audit_path}; use drop only if documented.")
    min_ratio = float(config.get("min_fit_ratio", 0.50))
    for entry in [train_report, val_report]:
        if not smoke_test and entry["fit_count"] / entry["original_count"] < min_ratio:
            raise RuntimeError(f"Too few rows fit token budget in {entry['split']}; "
                               "consider editing answers, not silent truncate")

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Unsloth QLoRA training")
    if int(config.get("qlora", {}).get("bits", 4)) != 4:
        raise ValueError("This pipeline uses QLoRA in 4 bits only")
    model, model_tokenizer = _load_model(family, model_cfg, max_length, config.get("qlora", {}),
                                         int(config.get("seed", 2026)))
    # Ensure model's actual tokenizer IDs coincide with preflight tokenizer; otherwise masks corrupt.
    for snippet in ("Thử nghiệm gout", "Bệnh gout có nguy hiểm không?", "assistant"):
        model_ids = model_tokenizer.encode(snippet, add_special_tokens=False)
        if model_ids != tokenizer.encode(snippet, add_special_tokens=False):
            raise RuntimeError("Tokenizer returned by Unsloth differs from pinned Phase-1 checkpoint")
    cfg = config.get("trainer", {})
    can_bf16 = torch.cuda.is_bf16_supported()
    use_bf16 = bool(cfg.get("bf16", True)) and can_bf16
    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        num_train_epochs=float(cfg.get("num_train_epochs", 2)),
        max_steps=2 if smoke_test else -1,
        warmup_steps=int(cfg.get("warmup_steps", 5)),
        lr_scheduler_type="linear", weight_decay=0.01,
        optim="adamw_8bit", fp16=not use_bf16, bf16=use_bf16,
        logging_steps=int(cfg.get("logging_steps", 1)),
        eval_strategy="steps" if smoke_test else "epoch",
        eval_steps=1 if smoke_test else None,
        save_strategy="no" if smoke_test else "epoch",
        load_best_model_at_end=not smoke_test,
        metric_for_best_model="eval_loss" if not smoke_test else None,
        greater_is_better=False if not smoke_test else None,
        save_total_limit=2, report_to="none", seed=int(config.get("seed", 2026)),
        data_seed=int(config.get("seed", 2026)),
        max_length=max_length, packing=False, gradient_checkpointing=False,
        remove_unused_columns=False, dataloader_num_workers=0,
        assistant_only_loss=False, # explicit -100 labels are already in pretokenized rows
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    class StopOnNonFiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            import math
            for metric in ("loss", "eval_loss"):
                if logs and metric in logs:
                    value = float(logs[metric])
                    if not math.isfinite(value):
                        raise RuntimeError(
                            f"{family}: {metric}={value} at global_step={state.global_step}. "
                            "Detected NaN/Inf, abort training to avoid corrupted adapters."
                        )
            return control

    collator = AssistantOnlyCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = SFTTrainer(
        model=model, args=args, processing_class=model_tokenizer,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows), data_collator=collator,
        callbacks=[StopOnNonFiniteLoss()],
    )
    testbatch = collator([train_rows[0]])
    assert (testbatch["labels"] == -100).any() and (testbatch["labels"] != -100).any()
    print(f"GPU: {torch.cuda.get_device_name(0)}, bf16={use_bf16}, "
          f"steps={'2 (SMOKE)' if smoke_test else 'full'}, "
          f"effective train samples={len(train_rows)}, val={len(val_rows)}")
    result = trainer.train()
    if not bool(torch.isfinite(torch.tensor(float(result.training_loss)))):
        raise RuntimeError("Non-finite training_loss! Stop; Gemma3/T4 precision combination might be unstable")
    if smoke_test:
        print("SMOKE PASS. No adapter saved. Not for clinical use.")
        return
    trainer.save_model(str(output_dir))
    model_tokenizer.save_pretrained(str(output_dir))
    runmeta = {
        "model_id": model_cfg["model_id"], "revision": model_cfg["revision"],
        "model_family": family,
        "assistant_only_loss": "manually_masked_labels_minus_100",
        "qwen3_enable_thinking": False if family == "qwen3" else None,
        "gemma3_vision_finetuning": False if family == "gemma3" else None,
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "train_sha256": audit["train_sha256"], "val_sha256": audit["val_sha256"],
        "medical_answers_approved": medical_reviewed,
        "research_only_draft": not medical_reviewed,
        "config": config,
    }
    (output_dir / "sft_run_metadata.json").write_text(json.dumps(runmeta, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    print(f"DONE SFT {family}. Adapter at {output_dir}. "
          + ("WARNING: RESEARCH-ONLY draft answers." if not medical_reviewed else ""))
