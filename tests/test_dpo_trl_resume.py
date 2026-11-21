"""DPOTrainer thật (TRL) trên CPU với model Llama tí hon, dữ liệu dạng hội thoại như nhánh Gemma của train_dpo:

1. Reference là adapter SFT "ref" chứ không phải mô hình gốc: ở bước đầu policy trùng reference nên loss = ln 2.
2. Train tiếp từ checkpoint (nạp tay adapter "default" rồi resume) cho đúng trọng số như chạy một mạch.

Cần torch, transformers, peft, trl, datasets; thiếu thì bỏ qua.
"""
import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
trl = pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")
tokenizers = pytest.importorskip("tokenizers")

import inspect  # noqa: E402

import trl.trainer.dpo_trainer as trl_dpo  # noqa: E402

from gout_llmops.training.dpo import (  # noqa: E402
    REF_ADAPTER,
    _filter_kwargs,
    lora_b_stats,
    reference_setup,
    restore_trained_adapter,
)
from gout_llmops.training.resume import last_checkpoint  # noqa: E402

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>"]
WORDS = ["user", "model", "a", "b", "c", "d", "yes", "no", "maybe"]
TEMPLATE = ("{{ bos_token }}{% for m in messages %}{{ '<start_of_turn>' + ('model' if m['role'] == 'assistant' "
            "else m['role']) + ' ' + m['content'] + ' <end_of_turn> ' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<start_of_turn>model ' }}{% endif %}")


def _tokenizer():
    vocab = {w: i for i, w in enumerate(SPECIAL + WORDS)}
    tk = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                               pad_token="<pad>", unk_token="<unk>",
                                               additional_special_tokens=["<start_of_turn>", "<end_of_turn>"])
    tok.chat_template = TEMPLATE
    return tok


def _base(vocab_size):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    return transformers.LlamaForCausalLM(cfg)


def _policy(sft_dir, vocab_size):
    """Như train_dpo: adapter SFT là "default"; bản "ref" do mình nạp (TRL 0.x) hoặc do TRL tự chép (TRL 1.x)."""
    model = peft.PeftModel.from_pretrained(_base(vocab_size), str(sft_dir), adapter_name="default", is_trainable=True)
    load_ref, kwargs = reference_setup(trl.DPOConfig, inspect.getsource(trl_dpo))
    if load_ref:
        model.load_adapter(str(sft_dir), adapter_name=REF_ADAPTER, is_trainable=False)
        model.set_adapter("default")
    return model, kwargs


def _dataset():
    rows = [{"prompt": [{"role": "user", "content": f"{x} {y}"}],
             "chosen": [{"role": "assistant", "content": "yes"}],
             "rejected": [{"role": "assistant", "content": "no maybe"}]}
            for x in ("a", "b") for y in ("c", "d", "a", "b")]
    return datasets.Dataset.from_list(rows)


def _trainer(policy, tok, out_dir):
    model, ref_kwargs = policy
    kwargs = dict(output_dir=str(out_dir), seed=7, data_seed=7, beta=0.1, loss_type="sigmoid", learning_rate=5e-2,
                  max_steps=4, per_device_train_batch_size=1, gradient_accumulation_steps=1, logging_steps=1,
                  save_strategy="steps", save_steps=2, save_total_limit=2, max_length=64, max_prompt_length=32,
                  precompute_ref_log_probs=True, use_logits_to_keep=True, optim="adamw_torch", use_cpu=True,
                  bf16=False, fp16=False, remove_unused_columns=False, report_to="none", **ref_kwargs)
    args = trl.DPOConfig(**_filter_kwargs(trl.DPOConfig, kwargs, {"beta", "max_length"} | set(ref_kwargs)))
    trainer = trl.DPOTrainer(model=model, ref_model=None, args=args, train_dataset=_dataset(), processing_class=tok)
    assert REF_ADAPTER in model.peft_config and lora_b_stats(model, REF_ADAPTER)[0] > 0   # reference = bản SFT
    return trainer


def _default_weights(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n and ".default." in n}


def test_reference_is_sft_adapter_and_resume_matches_uninterrupted_run(tmp_path):
    tok = _tokenizer()
    vocab = len(tok)
    sft = peft.get_peft_model(_base(vocab), peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for n, p in sft.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.5)   # adapter SFT khác hẳn mô hình gốc
    sft.save_pretrained(tmp_path / "sft")

    full = _trainer(_policy(tmp_path / "sft", vocab), tok, tmp_path / "full")
    full.train()
    first = next(h for h in full.state.log_history if "loss" in h)
    assert math.isclose(first["loss"], math.log(2), abs_tol=1e-3), first   # reference = SFT, không phải mô hình gốc
    final_full = _default_weights(full.model)

    ckpt = tmp_path / "full" / "checkpoint-2"
    assert last_checkpoint(tmp_path / "full").name == "checkpoint-4" and ckpt.is_dir()
    resumed_policy = _policy(tmp_path / "sft", vocab)           # phiên mới: nạp lại từ adapter SFT
    resumed = _trainer(resumed_policy, tok, tmp_path / "resumed")
    restore_trained_adapter(resumed_policy[0], ckpt)            # sau khi tạo trainer, như train_dpo
    resumed.train(resume_from_checkpoint=str(ckpt))
    assert resumed.state.global_step == 4
    final_resumed = _default_weights(resumed.model)
    assert final_resumed.keys() == final_full.keys()
    for n, w in final_full.items():
        assert torch.allclose(final_resumed[n], w, atol=1e-5), n
