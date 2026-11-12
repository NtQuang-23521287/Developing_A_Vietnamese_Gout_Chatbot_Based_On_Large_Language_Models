"""Giai đoạn 4 — huấn luyện DPO bằng TRL + QLoRA (4-bit), không cần Reward Model riêng.

Hai chế độ policy:
  - policy_adapter = thư mục adapter SFT: nạp adapter đó HAI lần — "default" (được train) và "reference"
    (đóng băng). Reference của DPO vì vậy là chính mô hình SFT, đúng định nghĩa π_ref = π_SFT.
    (Nếu chỉ nạp một lần, TRL sẽ tắt adapter để tính reference -> reference thành Qwen gốc: SAI.)
  - policy_adapter = null: tạo LoRA mới trên Qwen gốc; reference = Qwen gốc (tắt adapter).

lora_dropout của adapter được giữ > 0 để Giai đoạn 5 có thể dùng MC Dropout trên LoRA.
Mọi chi phí (thời gian, VRAM đỉnh, GPU, số tham số train) ghi vào dpo_run_metadata.json.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from gout_llmops.core.config import load_yaml
from gout_llmops.core.hashing import sha256_file
from gout_llmops.core.io import read_jsonl, write_json

QWEN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_policy(value: Any) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "base"}:
        return None
    return str(Path(str(value).strip()))   # chuẩn hóa: bỏ "/" cuối, "./" đầu


def check_adapter_dir(path: str | Path) -> Path:
    d = Path(path).expanduser().resolve()
    weights = d / "adapter_model.safetensors"
    if not (d / "adapter_config.json").is_file() or not weights.is_file():
        raise FileNotFoundError(f"Không thấy adapter LoRA hợp lệ tại {d}")
    if weights.stat().st_size < 10_000:
        raise RuntimeError(f"{weights} chỉ là con trỏ Git LFS (chưa tải file thật). "
                           "Chạy `git lfs pull` hoặc copy từ Drive.")
    return d


def _split_train_eval(rows: list[dict[str, Any]], eval_ratio: float, seed: int):
    def bucket(r):
        h = hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF
    ev = [r for r in rows if bucket(r) < eval_ratio]
    tr = [r for r in rows if bucket(r) >= eval_ratio]
    if not tr:
        raise ValueError("Không còn cặp nào cho train")
    return tr, ev


def _filter_kwargs(cls, kwargs: dict[str, Any], required: set[str]) -> dict[str, Any]:
    """Giữ các tham số mà phiên bản TRL đang cài hỗ trợ (TRL đổi API khá thường xuyên)."""
    fields = {f.name for f in dataclasses.fields(cls)}
    if "eval_strategy" in kwargs and "eval_strategy" not in fields and "evaluation_strategy" in fields:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = sorted(k for k in kwargs if k not in fields)
    missing_required = required & set(dropped)
    if missing_required:
        raise RuntimeError(f"Phiên bản TRL này thiếu tham số bắt buộc {sorted(missing_required)}; cần trl>=0.12")
    if dropped:
        print(f"[WARN] TRL không hỗ trợ, bỏ qua: {dropped}")
    return {k: v for k, v in kwargs.items() if k in fields}


def train_dpo(config: dict[str, Any], *, smoke_test: bool = False,
              policy_adapter: Any = "from_config", allow_policy_mismatch: bool = False) -> Path:
    if config.get("status") == "scaffold":
        raise RuntimeError("dpo config đang ở trạng thái scaffold")
    policy = resolve_policy(config.get("policy_adapter") if policy_adapter == "from_config" else policy_adapter)
    model_cfg = load_yaml(config["base_model_config"])
    if model_cfg.get("backend_type", "causal_lm") != "causal_lm":
        raise ValueError("DPO pipeline hiện chỉ hỗ trợ model causal LM dạng text (Qwen3/SeaLLMs)")
    pref_path = Path(config["preference_data"])
    manifest_path = pref_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if manifest.get("mock"):
        raise RuntimeError("Dữ liệu preference là MOCK — không dùng để train")
    gen_policy = resolve_policy(manifest.get("policy_adapter"))
    if manifest and gen_policy != policy and not allow_policy_mismatch:
        raise RuntimeError(
            f"Ứng viên được sinh bởi policy={gen_policy!r} nhưng đang train policy={policy!r}. DPO cần dữ liệu "
            "on-policy: sinh lại ứng viên, hoặc thêm --allow-policy-mismatch nếu cố ý.")
    policy_dir = check_adapter_dir(policy) if policy else None
    out_dir = Path(config["output_dir"] + ("_smoke" if smoke_test else ""))
    seed = int(config.get("seed", 2026))
    random.seed(seed)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Cần GPU CUDA để train DPO")
    torch.manual_seed(seed)

    model_id, revision = model_cfg["model_id"], model_cfg.get("revision")
    trust = bool(model_cfg.get("trust_remote_code", False))
    chat_kwargs = dict(model_cfg.get("chat_template_kwargs") or {})
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # TRL nối eos_token vào cuối chosen/rejected -> eos phải là token kết thúc lượt assistant của chat template
    probe = tok.apply_chat_template([{"role": "user", "content": "a"}, {"role": "assistant", "content": "XYZ"}],
                                    tokenize=False, **chat_kwargs)
    if f"XYZ{tok.eos_token}" not in probe:
        raise RuntimeError(f"eos_token={tok.eos_token!r} không phải token kết thúc lượt assistant trong chat template")

    # ---------- dữ liệu ----------
    max_length = int(config.get("max_length", 2048))
    max_prompt_length = int(config.get("max_prompt_length", 1536))
    rows = read_jsonl(pref_path)
    data, overlong = [], 0
    for r in rows:
        prompt = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, **chat_kwargs)
        n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
        n_c = max(len(tok(r[k], add_special_tokens=False)["input_ids"]) for k in ("chosen", "rejected")) + 1
        if n_p > max_prompt_length or n_p + n_c > max_length:
            overlong += 1
            continue
        data.append({"id": r["id"], "prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    if not data:
        raise ValueError("Không có cặp nào vừa max_length")
    if overlong / len(rows) > 0.10:
        raise RuntimeError(f"{overlong}/{len(rows)} cặp vượt max_length={max_length}; "
                           "tăng max_length thay vì bỏ âm thầm")
    train_rows, eval_rows = _split_train_eval(data, float(config.get("eval_ratio", 0.1)), seed)
    if smoke_test:
        train_rows, eval_rows = train_rows[:4], eval_rows[:2]
    strip = lambda rs: [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rs]  # noqa: E731
    print(f"Cặp preference: {len(rows)} | vượt độ dài: {overlong} | train {len(train_rows)} | eval {len(eval_rows)}")

    # ---------- model ----------
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, trust_remote_code=trust, quantization_config=bnb,
        torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    adapter_kwargs: dict[str, Any] = {}
    if policy_dir:
        model = PeftModel.from_pretrained(model, str(policy_dir), adapter_name="default", is_trainable=True)
        model.load_adapter(str(policy_dir), adapter_name="reference", is_trainable=False)
        model.set_adapter("default")
        adapter_kwargs = {"model_adapter_name": "default", "ref_adapter_name": "reference"}
        reference_desc = f"SFT adapter (đóng băng): {policy_dir}"
    else:
        lcfg = config.get("new_lora", {})
        model = get_peft_model(model, LoraConfig(
            r=int(lcfg.get("r", 16)), lora_alpha=int(lcfg.get("alpha", 32)),
            lora_dropout=float(lcfg.get("dropout", 0.05)), target_modules=QWEN_TARGET_MODULES,
            bias="none", task_type="CAUSAL_LM"))
        reference_desc = f"{model_id} gốc (tắt adapter)"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise RuntimeError("Không có tham số nào được train")
    lora_dropout = float(model.peft_config["default"].lora_dropout)
    print(f"Trainable params: {trainable:,} ({100 * trainable / total:.3f}%) | lora_dropout={lora_dropout} | "
          f"reference = {reference_desc}")
    if lora_dropout == 0:
        print("[WARN] lora_dropout = 0 -> MC Dropout trên LoRA ở Giai đoạn 5 sẽ không có phương sai")

    # ---------- trainer ----------
    t = config.get("trainer", {})
    kwargs: dict[str, Any] = dict(
        output_dir=str(out_dir), seed=seed, data_seed=seed,
        beta=float(t.get("beta", 0.1)), loss_type=t.get("loss_type", "sigmoid"),
        learning_rate=float(t.get("learning_rate", 1e-5)),
        num_train_epochs=float(t.get("num_train_epochs", 2)),
        max_steps=2 if smoke_test else -1,
        per_device_train_batch_size=int(t.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(t.get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(t.get("warmup_ratio", 0.1)), lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        logging_steps=1 if smoke_test else int(t.get("logging_steps", 5)),
        eval_strategy="no" if smoke_test or not eval_rows else "epoch",
        save_strategy="no" if smoke_test else "epoch", save_total_limit=2,
        bf16=bf16, fp16=not bf16, optim=t.get("optim", "paged_adamw_8bit"),
        max_length=max_length, max_prompt_length=max_prompt_length,
        precompute_ref_log_probs=bool(t.get("precompute_ref_log_probs", True)),
        use_logits_to_keep=bool(t.get("use_logits_to_keep", True)),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, report_to="none",
        **adapter_kwargs,
    )
    required = {"beta", "max_length"} | set(adapter_kwargs)
    args = DPOConfig(**_filter_kwargs(DPOConfig, kwargs, required))

    class StopOnNonFinite(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            for k in ("loss", "eval_loss"):
                if logs and k in logs and not math.isfinite(float(logs[k])):
                    raise RuntimeError(f"{k}={logs[k]} tại step {state.global_step}: dừng để tránh hỏng adapter")
            return control

    init_params = inspect.signature(DPOTrainer.__init__).parameters
    tok_arg = "processing_class" if "processing_class" in init_params else "tokenizer"
    trainer = DPOTrainer(
        model=model, ref_model=None, args=args,
        train_dataset=Dataset.from_list(strip(train_rows)),
        eval_dataset=Dataset.from_list(strip(eval_rows)) if eval_rows and not smoke_test else None,
        callbacks=[StopOnNonFinite()], **{tok_arg: tok})

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    runtime = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"Xong: {runtime / 60:.1f} phút | VRAM đỉnh {peak_mb:,.0f} MB | loss {result.training_loss:.4f}")
    if smoke_test:
        print("SMOKE PASS — không lưu adapter.")
        return out_dir

    final_eval = trainer.evaluate() if eval_rows else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(out_dir), selected_adapters=["default"])
    tok.save_pretrained(str(out_dir))
    write_json(out_dir / "log_history.json", trainer.state.log_history)
    meta = {
        "phase": "dpo", "model_id": model_id, "revision": revision,
        "policy_adapter": str(policy_dir) if policy_dir else None,
        "policy_adapter_sha256": sha256_file(policy_dir / "adapter_model.safetensors") if policy_dir else None,
        "reference": reference_desc,
        "preference_data": str(pref_path), "preference_sha256": sha256_file(pref_path),
        "preference_manifest": manifest,
        "n_pairs": len(rows), "n_overlong_dropped": overlong,
        "n_train": len(train_rows), "n_eval": len(eval_rows),
        "trainable_params": trainable, "trainable_ratio": trainable / total, "lora_dropout": lora_dropout,
        "gpu": torch.cuda.get_device_name(0), "precision": "bf16" if bf16 else "fp16",
        "train_runtime_seconds": runtime, "gpu_hours": runtime / 3600, "peak_vram_mb": peak_mb,
        "train_metrics": result.metrics, "final_eval": final_eval,
        "config": config, "dpo_config": {k: v for k, v in args.to_dict().items()
                                         if isinstance(v, (int, float, str, bool, type(None)))},
        "medical_answers_approved": False,
        "warning": "Research only. Chosen/Rejected do Heuristic Reward (judge LLM) tạo, chưa qua chuyên gia.",
    }
    write_json(out_dir / "dpo_run_metadata.json", meta)
    print(f"Adapter DPO: {out_dir}")
    return out_dir
