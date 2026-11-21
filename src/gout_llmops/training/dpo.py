"""Giai đoạn 4 — huấn luyện DPO bằng TRL + QLoRA (4-bit), không cần Reward Model riêng.

Mô hình: Qwen3-8B (causal LM) hoặc Gemma3-4B. Gemma3-4B là model đa phương thức, chỉ dùng phần văn bản, và được nạp
giống hệt lúc đánh giá (models/hf_backend.py) để adapter SFT khớp tên module.

Hai chế độ policy:
  - policy_adapter = thư mục adapter SFT: model có HAI bản adapter đó — "default" (được train) và "ref" (đóng băng).
    Reference của DPO vì vậy là chính mô hình SFT, đúng định nghĩa π_ref = π_SFT.
    (Chỉ có một bản thì TRL tắt adapter để tính reference -> reference thành mô hình gốc: SAI.)
    TRL 0.x: tự nạp bản "ref" và báo qua ref_adapter_name. TRL 1.x: trainer tự chép "default" thành "ref" lúc khởi
    tạo, nên không nạp thêm; train tiếp thì nạp lại "default" từ checkpoint SAU khi tạo trainer (nạp trước thì "ref"
    thành bản đã train dở).
  - policy_adapter = null: tạo LoRA mới trên mô hình gốc; reference = mô hình gốc (tắt adapter).

Train tiếp sau khi Colab ngắt: checkpoint được chép sang thư mục bền (mirror_dir), resume=True chạy tiếp từ
checkpoint mới nhất (training/resume.py). Trainer không tự nạp lại adapter "default" khi model có hai adapter,
nên nạp tay trước khi chạy tiếp.

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
from gout_llmops.training.resume import (
    check_mirror_dir,
    mirror_callback,
    mirror_now,
    prepare_resume,
    pull_checkpoints,
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
QWEN_TARGET_MODULES = LORA_TARGETS
# Gemma 3 4B đa phương thức: LoRA mới chỉ gắn vào các lớp ngôn ngữ, không đụng vision tower
GEMMA3_TEXT_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
SUPPORTED_BACKENDS = ("causal_lm", "gemma3_conditional")
REF_ADAPTER = "ref"


def render_prompt(messages: list[dict[str, str]], tokenizer: Any, processor: Any = None,
                  chat_kwargs: dict[str, Any] | None = None) -> str:
    """Prompt dạng chuỗi giống hệt lúc đánh giá (models/hf_backend.py): Gemma 3 qua processor với nội dung dạng list."""
    if processor is not None:
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **(chat_kwargs or {}))


def dpo_fingerprint(config: dict[str, Any], model_cfg: dict[str, Any], policy_dir: Path | None,
                    pref_path: Path) -> dict[str, Any]:
    """Những gì phải giống nhau thì mới được train tiếp từ checkpoint cũ."""
    fp = {"model_id": model_cfg["model_id"], "revision": model_cfg.get("revision"),
          "policy_adapter_sha256": sha256_file(policy_dir / "adapter_model.safetensors") if policy_dir else None,
          "preference_sha256": sha256_file(pref_path),
          **{k: config.get(k) for k in ("seed", "eval_ratio", "max_length", "max_prompt_length", "new_lora",
                                        "trainer")}}
    return json.loads(json.dumps(fp))


def lora_b_stats(model: Any, adapter: str) -> tuple[int, int]:
    """(số ma trận lora_B khác 0, tổng số) của một adapter. Adapter đã train mà toàn 0 = trọng số không được nạp."""
    bs = [p for n, p in model.named_parameters() if f"lora_B.{adapter}." in n]
    return sum(int(bool(p.detach().abs().sum() > 0)) for p in bs), len(bs)


def restore_trained_adapter(model: Any, checkpoint: Path, adapter: str = "default") -> None:
    """Trainer chỉ nạp lại các adapter nằm trong thư mục con của checkpoint (ở đây "ref"), không nạp adapter "default"
    ở gốc khi model có hai adapter -> nếu không nạp tay, DPO chạy tiếp từ trọng số SFT ban đầu với optimizer của
    checkpoint."""
    result = model.load_adapter(str(checkpoint), adapter_name=adapter, is_trainable=True)
    missing = [k for k in (getattr(result, "missing_keys", None) or []) if "lora_" in k and f".{adapter}." in k]
    if missing:
        raise RuntimeError(f"{checkpoint} thiếu {len(missing)} trọng số LoRA của adapter {adapter!r}")
    model.set_adapter(adapter)


def reference_setup(dpo_config_cls: Any, dpo_trainer_source: str) -> tuple[bool, dict[str, str]]:
    """(có phải tự nạp bản adapter "ref" không, tham số thêm cho DPOConfig) theo phiên bản TRL đang cài.

    TRL 0.x: có tham số ref_adapter_name -> tự nạp "ref". TRL 1.x: bỏ tham số đó, tự chép "default" thành "ref" lúc
    khởi tạo trainer và dùng "ref" làm reference."""
    if "ref_adapter_name" in {f.name for f in dataclasses.fields(dpo_config_cls)}:
        return True, {"model_adapter_name": "default", "ref_adapter_name": REF_ADAPTER}
    if f'add_adapter("{REF_ADAPTER}"' in dpo_trainer_source:
        return False, {}
    if f'"{REF_ADAPTER}" in' in dpo_trainer_source:
        return True, {}
    raise RuntimeError("Phiên bản TRL này không có ref_adapter_name và không dùng adapter 'ref' làm reference: "
                       "reference sẽ thành mô hình gốc thay vì mô hình SFT")


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
              policy_adapter: Any = "from_config", allow_policy_mismatch: bool = False,
              resume: bool = False, mirror_dir: str | None = None) -> Path:
    if config.get("status") == "scaffold":
        raise RuntimeError("dpo config đang ở trạng thái scaffold")
    policy = resolve_policy(config.get("policy_adapter") if policy_adapter == "from_config" else policy_adapter)
    model_cfg = load_yaml(config["base_model_config"])
    backend_type = model_cfg.get("backend_type", "causal_lm")
    if backend_type not in SUPPORTED_BACKENDS:
        raise ValueError(f"DPO chưa hỗ trợ backend_type={backend_type!r} (chỉ {SUPPORTED_BACKENDS})")
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
    gen_model = (manifest.get("generation") or {}).get("model_config")
    if gen_model and Path(gen_model) != Path(config["base_model_config"]) and not allow_policy_mismatch:
        raise RuntimeError(f"Ứng viên được sinh bằng {gen_model} nhưng config train dùng "
                           f"{config['base_model_config']}. DPO cần dữ liệu on-policy: "
                           "sinh ứng viên bằng đúng mô hình này.")
    policy_dir = check_adapter_dir(policy) if policy else None
    out_dir = Path(config["output_dir"] + ("_smoke" if smoke_test else ""))
    mirror = Path(mirror_dir) if mirror_dir and not smoke_test else None
    if mirror is not None:
        check_mirror_dir(out_dir, mirror)
    seed = int(config.get("seed", 2026))
    random.seed(seed)
    resume_from = None
    if not smoke_test:
        if mirror is not None:
            pull_checkpoints(mirror, out_dir)
        resume_from = prepare_resume(out_dir, dpo_fingerprint(config, model_cfg, policy_dir, pref_path), resume)

    import torch
    import trl.trainer.dpo_trainer as trl_dpo
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
    processor = None
    if backend_type == "gemma3_conditional":
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
        tok = processor.tokenizer
    else:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Gemma: đưa dữ liệu dạng hội thoại để TRL tự áp chat template, thêm <bos> đúng một lần và kết lượt bằng
    # <end_of_turn> (TRL 1.x tokenize chuỗi thô có thêm <bos> -> chuỗi đã có <bos> sẽ thành hai lần).
    conversational = backend_type == "gemma3_conditional"
    if not conversational:
        # TRL nối eos_token vào cuối chosen/rejected -> eos phải là token kết thúc lượt assistant của chat template
        probe = tok.apply_chat_template([{"role": "user", "content": "a"}, {"role": "assistant", "content": "XYZ"}],
                                        tokenize=False, **chat_kwargs)
        if f"XYZ{tok.eos_token}" not in probe:
            raise RuntimeError(f"eos_token={tok.eos_token!r} không phải token kết thúc lượt assistant "
                               "trong chat template")

    # ---------- dữ liệu ----------
    max_length = int(config.get("max_length", 2048))
    max_prompt_length = int(config.get("max_prompt_length", 1536))
    rows = read_jsonl(pref_path)
    data, overlong = [], 0
    for r in rows:
        prompt = render_prompt(r["prompt"], tok, processor, chat_kwargs)
        n_p = len(tok(prompt, add_special_tokens=False)["input_ids"])
        n_c = max(len(tok(r[k], add_special_tokens=False)["input_ids"]) for k in ("chosen", "rejected")) + 2
        if n_p > max_prompt_length or n_p + n_c > max_length:
            overlong += 1
            continue
        if conversational:
            data.append({"id": r["id"], "prompt": r["prompt"],
                         "chosen": [{"role": "assistant", "content": r["chosen"]}],
                         "rejected": [{"role": "assistant", "content": r["rejected"]}]})
        else:
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
    load_kw = dict(revision=revision, trust_remote_code=trust, quantization_config=bnb, device_map={"": 0},
                   attn_implementation="sdpa")
    if backend_type == "gemma3_conditional":
        from transformers import Gemma3ForConditionalGeneration
        # Cùng lớp model như lúc đánh giá nên adapter SFT khớp tên module. Phần không lượng tử hoá để float32:
        # Gemma 3 dễ tràn số ở float16, mà T4 không có bf16.
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float32, **load_kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **load_kw)
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    adapter_kwargs: dict[str, Any] = {}
    if policy_dir:
        model = PeftModel.from_pretrained(model, str(policy_dir), adapter_name="default", is_trainable=True)
        load_ref, adapter_kwargs = reference_setup(DPOConfig, inspect.getsource(trl_dpo))
        if load_ref:
            model.load_adapter(str(policy_dir), adapter_name=REF_ADAPTER, is_trainable=False)
            model.set_adapter("default")
        nonzero, total_b = lora_b_stats(model, "default")
        if total_b == 0 or nonzero == 0:
            raise RuntimeError(f"Adapter SFT không được nạp vào model ({nonzero}/{total_b} ma trận lora_B khác 0): "
                               "tên module trong adapter không khớp cấu trúc model")
        reference_desc = f"SFT adapter (đóng băng, tên {REF_ADAPTER!r}): {policy_dir}"
    else:
        lcfg = config.get("new_lora", {})
        targets = GEMMA3_TEXT_TARGETS if backend_type == "gemma3_conditional" else LORA_TARGETS
        model = get_peft_model(model, LoraConfig(
            r=int(lcfg.get("r", 16)), lora_alpha=int(lcfg.get("alpha", 32)),
            lora_dropout=float(lcfg.get("dropout", 0.05)), target_modules=targets,
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
    save_steps = None if smoke_test else t.get("save_steps")   # lưu dày hơn mỗi epoch: Colab ngắt thì mất ít
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
        save_strategy="no" if smoke_test else ("steps" if save_steps else "epoch"), save_total_limit=2,
        **({"save_steps": int(save_steps)} if save_steps else {}),
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
        callbacks=[StopOnNonFinite()] + ([mirror_callback(out_dir, mirror)] if mirror is not None else []),
        **{tok_arg: tok})
    if policy_dir:   # reference phải là bản SFT: có adapter "ref" và trọng số khác 0
        nonzero, total_b = lora_b_stats(model, REF_ADAPTER)
        if REF_ADAPTER not in model.peft_config or total_b == 0 or nonzero == 0:
            raise RuntimeError(f"Không có adapter {REF_ADAPTER!r} làm reference ({nonzero}/{total_b} lora_B khác 0): "
                               "reference sẽ thành mô hình gốc thay vì mô hình SFT")
    if resume_from is not None:   # sau khi tạo trainer: TRL 1.x đã chép "default" (còn là SFT) thành "ref"
        restore_trained_adapter(model, resume_from)
        print(f"Train tiếp từ {resume_from.name}: đã nạp adapter đang train từ checkpoint")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
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
        "phase": "dpo", "model_id": model_id, "revision": revision, "backend_type": backend_type,
        "resumed_from_checkpoint": resume_from.name if resume_from else None,
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
    if mirror is not None:
        mirror_now(out_dir, mirror)
    print(f"Adapter DPO: {out_dir}")
    return out_dir
