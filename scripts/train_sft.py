"""Entry point for both Qwen3-8B and Gemma3-4B SFT in this repository."""
from __future__ import annotations

import argparse

from gout_llmops.core.config import load_yaml
from gout_llmops.training.sft import train_sft

CONFIGS = {
    "qwen3": "configs/training/sft_qwen3_8b_v3.yaml",
    "gemma3": "configs/training/sft_gemma3_4b_v3.yaml",
}


def main():
    p = argparse.ArgumentParser(description="Phase 3 bilingual-model text-only QLoRA SFT")
    p.add_argument("--model", choices=["qwen3", "gemma3"], default="qwen3")
    p.add_argument("--config", default=None, help="Override YAML config path")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preflight-only", action="store_true", help="Audit tokens; no CUDA training")
    g.add_argument("--smoke-test", action="store_true", help="2 GPU steps, no adapter saved")
    p.add_argument("--full-test", action="store_true",
                   help="Explicitly authorize RESEARCH-ONLY full train on unreviewed QA")
    p.add_argument("--resume", action="store_true",
                   help="Train tiếp từ checkpoint mới nhất trong output_dir (dữ liệu và config phải giống lần trước)")
    p.add_argument("--checkpoint-mirror", default=None,
                   help="Thư mục bền cùng tên với output_dir (vd. trên Drive): nhận checkpoint sau mỗi epoch; "
                        "output_dir chưa có checkpoint thì lấy từ đây")
    args = p.parse_args()
    cfg = args.config or CONFIGS[args.model]
    data = load_yaml(cfg)
    print(f"Using config: {cfg}")
    train_sft(data, preflight_only=args.preflight_only,
              smoke_test=args.smoke_test, allow_draft=args.full_test,
              resume=args.resume, mirror_dir=args.checkpoint_mirror)


if __name__ == "__main__":
    main()
