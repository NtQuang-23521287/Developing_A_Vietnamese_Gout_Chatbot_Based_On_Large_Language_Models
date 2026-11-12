"""Giai đoạn 5 — MC Dropout trên LoRA (cần torch + GPU). Phần tính toán thuần nằm ở safety/uncertainty.py.

Cách làm: model ở chế độ eval, CHỈ các module `lora_dropout` của PEFT được bật train() -> mỗi lượt forward
dùng một mặt nạ dropout khác nhau trên nhánh LoRA, phần còn lại của model giữ nguyên. Câu trả lời đã sinh
(greedy) được chấm lại N lần (teacher forcing, một forward/lượt, không sinh lại).

Yêu cầu: adapter train với lora_dropout > 0 (SFT v3 và DPO v1 dùng 0.05). Adapter cũ có lora_dropout = 0
(SFT v1) chỉ dùng được để THỬ pipeline bằng `p_override` (chèn dropout lúc suy luận — không phải
MC Dropout đúng nghĩa vì model không được train với dropout đó; không dùng số liệu này trong khóa luận).
"""
from __future__ import annotations

import time
from typing import Any

from gout_llmops.safety.uncertainty import summarize_deterministic, summarize_mc


def _torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("MC Dropout cần torch: pip install -e '.[model]'") from e
    return torch


def lora_dropout_slots(model) -> list[tuple[str, Any, str]]:
    """[(tên module, ModuleDict lora_dropout, adapter_name)] của mọi lớp LoRA."""
    torch = _torch()
    return [(name, module, key) for name, module in model.named_modules()
            if name.endswith("lora_dropout") and isinstance(module, torch.nn.ModuleDict)
            for key in list(module.keys())]


class MCDropoutScorer:
    def __init__(self, backend, n_samples: int = 10, p_override: float | None = None, seed: int = 2026):
        if getattr(backend, "backend_type", "causal_lm") != "causal_lm":
            raise ValueError("MCDropoutScorer chỉ hỗ trợ backend causal_lm (Qwen)")
        self.torch = _torch()
        self.backend, self.model, self.tok = backend, backend.model, backend.tokenizer
        self.n_samples, self.seed = int(n_samples), int(seed)
        self.slots = lora_dropout_slots(self.model)
        if p_override is not None:
            for _, parent, key in self.slots:
                parent[key] = self.torch.nn.Dropout(float(p_override))
        ps = [float(parent[key].p) if isinstance(parent[key], self.torch.nn.Dropout) else 0.0
              for _, parent, key in self.slots]
        self.info = {"n_lora_dropout_modules": len(self.slots), "n_active": sum(p > 0 for p in ps),
                     "p_values": sorted(set(ps)), "p_override": p_override, "n_samples": self.n_samples,
                     "seed": self.seed}
        if self.n_samples > 0 and self.info["n_active"] == 0:
            raise RuntimeError(
                "Không có lora_dropout nào có p > 0 (model gốc hoặc adapter train với lora_dropout = 0). "
                "Dùng --n-samples 0 để chỉ tính tín hiệu deterministic, hoặc --p-override để thử pipeline.")
        self._set_mc(False)

    def _set_mc(self, on: bool) -> None:
        self.model.eval()
        if on:
            for _, parent, key in self.slots:
                parent[key].train()

    def _forward_logits(self, ids, n_keep: int):
        try:
            return self.model(input_ids=ids, use_cache=False, logits_to_keep=n_keep).logits
        except TypeError:   # transformers cũ không có logits_to_keep
            return self.model(input_ids=ids, use_cache=False).logits

    def encode(self, messages: list[dict[str, str]], answer: str):
        prompt_ids = self.backend._prepare_causal_inputs(messages)["input_ids"]
        answer_ids = self.tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return prompt_ids, answer_ids

    def score(self, messages: list[dict[str, str]], answer: str) -> dict[str, Any]:
        torch = self.torch
        t0 = time.perf_counter()
        prompt_ids, answer_ids = self.encode(messages, answer)
        T = int(answer_ids.shape[1])
        if T == 0:
            raise ValueError("câu trả lời rỗng")
        device = next(self.model.parameters()).device
        ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)
        target = answer_ids[0].to(device)

        def one_pass():
            logits = self._forward_logits(ids, T + 1)[0, -(T + 1):-1, :].float()
            logp = torch.log_softmax(logits, dim=-1)
            tok_lp = logp.gather(-1, target[:, None]).squeeze(-1)
            p = logp.exp()
            ent = -(p * logp).sum(-1)
            return tok_lp, ent, p

        # Toàn bộ tính toán trong MỘT khối inference_mode (cộng dồn tại chỗ lên inference tensor ở ngoài khối
        # sẽ lỗi).
        with torch.inference_mode():
            self._set_mc(False)
            lp, ent, _ = one_pass()
            out: dict[str, Any] = {**summarize_deterministic(lp.tolist(), ent.tolist()),
                                   "n_prompt_tokens": int(prompt_ids.shape[1]), "n_answer_tokens": T}
            if self.n_samples > 0:
                passes, ent_sum, p_sum = [], torch.zeros(T, device=device), None
                self._set_mc(True)
                try:
                    for i in range(self.n_samples):
                        torch.manual_seed(self.seed + i)
                        lp_i, ent_i, p_i = one_pass()
                        passes.append(lp_i.tolist())
                        ent_sum += ent_i
                        p_sum = p_i if p_sum is None else p_sum.add_(p_i)
                        del p_i
                finally:
                    self._set_mc(False)
                p_bar = p_sum / self.n_samples
                h_bar = -(p_bar * torch.log(p_bar.clamp_min(1e-12))).sum(-1)
                mi = (h_bar - ent_sum / self.n_samples).clamp_min(0.0)
                out.update(summarize_mc(passes, mi.tolist()))
                del p_sum, p_bar
        out["seconds"] = round(time.perf_counter() - t0, 3)
        return out


def make_uncertainty_fn(scorer: MCDropoutScorer, signal: str):
    """Cho SafeChatEngine: record (có messages, answer) -> giá trị tín hiệu đã hiệu chỉnh."""
    def fn(record: dict[str, Any]) -> float:
        return float(scorer.score(record["messages"], record["answer"])[signal])
    return fn
