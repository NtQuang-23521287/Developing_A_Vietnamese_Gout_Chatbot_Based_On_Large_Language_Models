"""Train tiếp DPO khi model có hai adapter (default đang train + ref đóng băng), chạy trên CPU với model Llama tí hon.

Trainer chỉ nạp lại adapter nằm trong thư mục con của checkpoint ("ref"), nên adapter "default" ở gốc phải được nạp
tay (restore_trained_adapter). Cần torch, transformers, peft; thiếu thì bỏ qua.
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from gout_llmops.training.dpo import REF_ADAPTER, lora_b_stats, restore_trained_adapter  # noqa: E402


def _base():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    return transformers.LlamaForCausalLM(cfg)


def _policy(sft_dir):
    """Giống train_dpo: adapter SFT nạp hai lần, 'default' được train, 'ref' đóng băng."""
    model = peft.PeftModel.from_pretrained(_base(), str(sft_dir), adapter_name="default", is_trainable=True)
    model.load_adapter(str(sft_dir), adapter_name=REF_ADAPTER, is_trainable=False)
    model.set_adapter("default")
    return model


def _lora(model, adapter):
    return {n.replace(f".{adapter}.", "."): p.detach().clone()
            for n, p in model.named_parameters() if "lora_" in n and f".{adapter}." in n}


def test_restore_trained_adapter_after_two_adapter_checkpoint(tmp_path):
    sft = peft.get_peft_model(_base(), peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for n, p in sft.named_parameters():
            if "lora_B" in n:
                p.normal_()           # adapter SFT đã train: lora_B khác 0
    sft.save_pretrained(tmp_path / "sft")

    model = _policy(tmp_path / "sft")
    assert lora_b_stats(model, "default")[0] > 0 and lora_b_stats(model, REF_ADAPTER)[0] > 0
    sft_weights = _lora(model, "default")
    with torch.no_grad():   # vài bước DPO làm adapter "default" lệch khỏi SFT
        for n, p in model.named_parameters():
            if "lora_" in n and ".default." in n:
                p.add_(0.5)
    trained = _lora(model, "default")
    ckpt = tmp_path / "checkpoint-5"
    model.save_pretrained(ckpt)   # như Trainer lưu checkpoint: "default" ở gốc, "ref" ở thư mục con
    assert (ckpt / "adapter_model.safetensors").exists() and (ckpt / REF_ADAPTER / "adapter_model.safetensors").exists()

    fresh = _policy(tmp_path / "sft")   # phiên Colab mới: nạp lại từ adapter SFT
    assert all(torch.equal(fresh_w, sft_weights[n]) for n, fresh_w in _lora(fresh, "default").items())
    restore_trained_adapter(fresh, ckpt)
    restored, ref = _lora(fresh, "default"), _lora(fresh, REF_ADAPTER)
    assert restored.keys() == trained.keys()
    assert all(torch.equal(restored[n], trained[n]) for n in trained)       # adapter đang train được nạp lại
    assert all(torch.equal(ref[n], sft_weights[n]) for n in ref)             # reference vẫn là SFT
    trainable = {n for n, p in fresh.named_parameters() if p.requires_grad}
    assert trainable and all(".default." in n for n in trainable)
