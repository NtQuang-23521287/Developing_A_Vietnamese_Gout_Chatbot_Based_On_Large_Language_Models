import argparse

from gout_llmops.core.config import load_yaml
from gout_llmops.training.dpo import train_dpo

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training/dpo_v1.yaml")
    args = ap.parse_args()
    train_dpo(load_yaml(args.config))
