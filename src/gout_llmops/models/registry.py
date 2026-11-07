from __future__ import annotations

from pathlib import Path

from gout_llmops.models.hf_backend import HuggingFaceBackend


def build_backend(config: dict):
    provider = config.get("provider", "huggingface")
    if provider == "huggingface":
        backend = HuggingFaceBackend(config)
        adapter_path = config.get("adapter_path")
        if adapter_path:
            from peft import PeftModel

            adapter_dir = Path(adapter_path).expanduser().resolve()
            if not (adapter_dir / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Missing LoRA config: {adapter_dir}")
            if not (adapter_dir / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(f"Missing LoRA weights: {adapter_dir}")
            backend.model = PeftModel.from_pretrained(
                backend.model, str(adapter_dir), is_trainable=False
            )
            backend.model.eval()
            if not backend.model.peft_config:
                raise RuntimeError("PEFT reported no loaded adapter")
            print(f"[SFT] Loaded PEFT LoRA from: {adapter_dir}")
        return backend
    raise ValueError(f"Unsupported provider: {provider}")
