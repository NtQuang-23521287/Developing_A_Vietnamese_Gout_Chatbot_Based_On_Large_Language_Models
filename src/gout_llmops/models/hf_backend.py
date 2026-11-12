from __future__ import annotations

import time
from typing import Any

from gout_llmops.models.base import ModelBackend


class HuggingFaceBackend(ModelBackend):
    def __init__(self, config: dict):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
                BitsAndBytesConfig,
                Gemma3ForConditionalGeneration,
            )

        except ImportError as e:
            raise RuntimeError(
                "Install model dependencies: "
                "pip install -e '.[model]'"
            ) from e

        self.torch = torch
        self.config = config

        self.backend_type = config.get(
            "backend_type",
            "causal_lm",
        )

        # -----------------------------
        # Quantization
        # -----------------------------
        qcfg = config.get(
            "quantization",
            {},
        )

        quant = None

        if qcfg.get("mode") == "nf4":
            dtype_name = qcfg.get(
                "compute_dtype",
                "bfloat16",
            )

            dtype = getattr(
                torch,
                dtype_name,
            )

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=bool(
                    qcfg.get(
                        "double_quant",
                        True,
                    )
                ),
            )

        common = {
            "revision": config.get("revision"),
            "trust_remote_code": bool(
                config.get(
                    "trust_remote_code",
                    False,
                )
            ),
        }

        model_id = config["model_id"]

        # =============================
        # Gemma 3 4B multimodal checkpoint
        # =============================
        if self.backend_type == "gemma3_conditional":

            self.processor = (
                AutoProcessor.from_pretrained(
                    model_id,
                    **common,
                )
            )

            self.tokenizer = (
                self.processor.tokenizer
            )

            self.model = (
                Gemma3ForConditionalGeneration
                .from_pretrained(
                    model_id,
                    **common,
                    quantization_config=quant,
                    device_map="auto",
                )
            )

        # =============================
        # Qwen / SeaLLM / normal causal LM
        # =============================
        elif self.backend_type == "causal_lm":

            self.processor = None

            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    model_id,
                    **common,
                )
            )

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_id,
                    **common,
                    quantization_config=quant,
                    device_map="auto",
                )
            )

        else:
            raise ValueError(
                "Unsupported backend_type: "
                f"{self.backend_type}"
            )

        self.model.eval()

    # --------------------------------------------------
    # Normal causal models:
    # SeaLLM / Qwen
    # --------------------------------------------------
    def _prepare_causal_inputs(
        self,
        messages: list[dict[str, str]],
    ):
        kwargs = dict(
            self.config.get(
                "chat_template_kwargs",
                {},
            )
            or {}
        )

        try:
            prompt = (
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            )

        except TypeError:
            prompt = (
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return inputs

    # --------------------------------------------------
    # Gemma 3 4B:
    # convert string-content messages into multimodal
    # chat-template format, but only provide text.
    # --------------------------------------------------
    def _prepare_gemma3_inputs(
        self,
        messages: list[dict[str, str]],
    ):
        gemma_messages = []

        for message in messages:
            gemma_messages.append(
                {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"],
                        }
                    ],
                }
            )

        inputs = (
            self.processor.apply_chat_template(
                gemma_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        )

        return inputs

    def generate(
        self,
        messages: list[dict[str, str]],
        max_input_tokens: int,
    ) -> dict[str, Any]:

        # -----------------------------
        # Build inputs
        # -----------------------------
        if self.backend_type == "gemma3_conditional":
            inputs = self._prepare_gemma3_inputs(
                messages
            )

        else:
            inputs = self._prepare_causal_inputs(
                messages
            )

        n_input = int(
            inputs["input_ids"].shape[-1]
        )

        if n_input > max_input_tokens:
            raise ValueError(
                f"Input has {n_input} tokens "
                f"> max_input_tokens="
                f"{max_input_tokens}; "
                "no silent truncation"
            )

        device = next(
            self.model.parameters()
        ).device

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        # -----------------------------
        # Generation config
        # -----------------------------
        gen = dict(
            self.config.get(
                "generation",
                {},
            )
        )

        gen = {
            key: value
            for key, value in gen.items()
            if value is not None
        }

        peak_vram_mb = None

        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                **gen,
            )

        elapsed = (
            time.perf_counter()
            - start
        )

        if self.torch.cuda.is_available():
            peak_vram_mb = (
                self.torch.cuda
                .max_memory_allocated()
                / (1024 ** 2)
            )

        # chỉ decode token model vừa sinh,
        # không decode lại prompt
        new_tokens = out[
            0,
            n_input:
        ]

        n_output = int(
            new_tokens.shape[-1]
        )

        if self.backend_type == "gemma3_conditional":
            answer = (
                self.processor.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                .strip()
            )

        else:
            answer = (
                self.tokenizer.decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                .strip()
            )

        return {
            "answer": answer,
            "input_tokens": n_input,
            "output_tokens": n_output,
            "latency_seconds": elapsed,
            "tokens_per_second": (
                n_output / elapsed
                if elapsed > 0
                else None
            ),
            "peak_vram_mb": peak_vram_mb,
        }