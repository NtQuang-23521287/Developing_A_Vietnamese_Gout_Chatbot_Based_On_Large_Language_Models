#!/usr/bin/env python3
"""Analyze SFT/QLoRA training for Qwen3-8B and Gemma3-4B.

Default run directories relative to the repository root:
  artifacts/runs/sft_qwen3_8b_v2          (LR 1e-4, 10 epochs)
  artifacts/runs/sft_gemma3_4b_v2         (LR 2e-5, 10 epochs)

Usage from repository root:
  python scripts/plot_sft_loss.py

Alternative sources (trainer_state.json, run dir, ZIP, .txt/.md/.log):
  python scripts/plot_sft_loss.py --qwen path/to/qwen.json --gemma path/to/gemma.md

Install if necessary: pip install numpy pandas matplotlib
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save figures on servers/Colab without a desktop display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QWEN = REPO_ROOT / "artifacts/runs/sft_qwen3_8b_v2"
DEFAULT_GEMMA = REPO_ROOT / "artifacts/runs/sft_gemma3_4b_v2"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/reports/sft_training"


def _state_rank(state: dict) -> tuple[int, int, int]:
    """Prefer the latest AND most complete trainer_state.json."""
    logs = state.get("log_history", [])
    return (int(state.get("global_step") or 0),
            len(logs),
            sum("eval_loss" in row for row in logs if isinstance(row, dict)))


def _parse_trainer_state(state: dict, description: str):
    if not isinstance(state.get("log_history"), list):
        raise ValueError(f"Không có log_history hợp lệ: {description}")
    return state["log_history"], description, state


def _read_text_log(raw: str, description: str):
    """Extract the small Python-like dictionaries printed by HF Trainer."""
    records = []
    for match in re.finditer(r"\{[^{}]*\}", raw):
        try:
            item = ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError, TypeError):
            continue
        if (isinstance(item, dict) and "epoch" in item
                and ("loss" in item or "eval_loss" in item)):
            records.append(item)
    if not records:
        raise ValueError(f"Không tìm thấy loss/eval_loss trong log: {description}")
    return records, description, None


def _read_from_zip(source: Path):
    with zipfile.ZipFile(source) as z:
        states = []
        for name in z.namelist():
            if name.endswith("trainer_state.json") and not name.endswith("/"):
                try:
                    obj = json.loads(z.read(name))
                    if isinstance(obj, dict) and isinstance(obj.get("log_history"), list):
                        states.append((obj, name))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        if states:
            state, name = max(states, key=lambda item: _state_rank(item[0]))
            return _parse_trainer_state(state, f"{source}!{name}")

        # A ZIP may contain only a text log rather than trainer_state.json.
        log_names = [
            n for n in z.namelist()
            if re.search(r"\.(?:md|txt|log)$", n, flags=re.I)
            and any(word in Path(n).name.lower()
                    for word in ("train", "sft", "output"))
        ]
        for name in sorted(log_names, key=lambda n: ("train" not in n.lower(), n)):
            if z.getinfo(name).file_size > 30 * 1024 * 1024:
                continue
            try:
                return _read_text_log(z.read(name).decode("utf-8", "replace"),
                                      f"{source}!{name}")
            except ValueError:
                continue
    raise FileNotFoundError(f"Không tìm thấy trainer_state.json hoặc training log trong ZIP: {source}")


def load_sft_history(source: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load history and metadata from a run folder, JSON, training log or ZIP."""
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Đường dẫn chưa tồn tại: {source}")

    if source.is_dir():
        states = []
        for path in source.rglob("trainer_state.json"):
            try:
                obj = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and isinstance(obj.get("log_history"), list):
                    states.append((obj, path))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                continue
        if states:
            state, path = max(states, key=lambda item: _state_rank(item[0]))
            records, selected, state = _parse_trainer_state(state, str(path))
        else:
            logs = [p for p in source.rglob("*")
                    if p.is_file() and p.suffix.lower() in (".txt", ".md", ".log")
                    and re.search(r"train|sft|output", p.name, flags=re.I)]
            if not logs:
                raise FileNotFoundError(
                    f"Không thấy trainer_state.json hoặc training log trong {source}. "
                    "Hãy copy trainer_state.json từ checkpoint-250 vào run, "
                    "hoặc chỉ định file log qua --gemma/--qwen."
                )
            records = selected = state = None
            for path in sorted(logs):
                try:
                    records, selected, state = _read_text_log(
                        path.read_text(encoding="utf-8", errors="replace"), str(path)
                    )
                    break
                except ValueError:
                    continue
            if records is None:
                raise ValueError(f"Đã tìm file log nhưng không đọc được số liệu: {source}")
    elif source.suffix.lower() == ".zip":
        records, selected, state = _read_from_zip(source)
    elif source.suffix.lower() == ".json":
        records, selected, state = _parse_trainer_state(
            json.loads(source.read_text(encoding="utf-8")), str(source)
        )
    elif source.suffix.lower() in (".md", ".txt", ".log"):
        records, selected, state = _read_text_log(
            source.read_text(encoding="utf-8", errors="replace"), str(source)
        )
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {source.suffix}")

    df = pd.DataFrame(records)
    for col in ("epoch", "step", "loss", "eval_loss", "grad_norm", "learning_rate"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["epoch"].notna() &
            (df["loss"].notna() | df["eval_loss"].notna())].copy()
    if df.empty:
        raise ValueError(f"Không có số liệu training/validation hợp lệ: {source}")

    meta = {"source": str(source), "selected": selected, "state": state}
    n_train = int(df["loss"].notna().sum())
    n_eval = int(df["eval_loss"].notna().sum())
    print(f"[LOAD] {selected}\n       Train logs: {n_train} | Eval logs: {n_eval}")
    return df, meta


def prepare_sft_metrics(history: pd.DataFrame):
    """Keep step-level data and aggregate means per completed epoch."""
    train = history.loc[history["loss"].notna()].copy()
    valid = history.loc[history["eval_loss"].notna()].copy()
    if train.empty or valid.empty:
        raise ValueError("Thiếu Training Loss hoặc Eval Loss. Cần log đầy đủ.")

    # Printed text logs often have no explicit 'step'; fill sequentially ONLY for train.
    if train["step"].isna().any():
        seq = pd.Series(np.arange(1, len(train) + 1), index=train.index)
        train["step"] = train["step"].fillna(seq)
    train = train.sort_values("step", kind="stable")

    # epoch 0.04 -> 1; epoch 1.0 -> 1; epoch 1.04 -> 2.
    train["epoch_group"] = np.ceil(train["epoch"] - 1e-7).astype(int)
    per_epoch = (train.groupby("epoch_group", as_index=False)
                 .agg(train_loss_mean=("loss", "mean"),
                      train_loss_min=("loss", "min"),
                      train_loss_max=("loss", "max"),
                      grad_norm_mean=("grad_norm", "mean"),
                      grad_norm_max=("grad_norm", "max"),
                      logged_steps=("loss", "count"))
                 .rename(columns={"epoch_group": "epoch"}))

    valid = valid.sort_values(["epoch", "step"], na_position="last", kind="stable")
    valid["epoch_group"] = np.ceil(valid["epoch"] - 1e-7).astype(int)
    val_per_epoch = (valid.groupby("epoch_group", as_index=False).last()
                     [["epoch_group", "eval_loss", "step"]]
                     .rename(columns={"epoch_group": "epoch", "step": "eval_step"}))
    metrics = (per_epoch.merge(val_per_epoch, on="epoch", how="outer")
               .sort_values("epoch").reset_index(drop=True))
    return train, valid, metrics


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def _save_chart(fig, output: Path, basename: str, dpi: int = 240):
    fig.savefig(output / f"{basename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / f"{basename}.svg", bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax, title: str, ylabel: str):
    ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    ax.set_xticks(ax.get_xticks()[np.isclose(ax.get_xticks(),
                                             np.rint(ax.get_xticks()))])


def plot_sft_comparison(models: dict[str, str | Path],
                        output_dir: str | Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Plot 5 standalone charts and one four-panel summary; save CSV/Markdown."""
    if len(models) < 1:
        raise ValueError("Cần tối thiểu một model")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = {}
    summaries = []

    for label, source in models.items():
        history, metadata = load_sft_history(source)
        train, valid, metrics = prepare_sft_metrics(history)
        if valid.empty:
            raise ValueError(f"Không có Eval Loss cho {label}")

        best = valid.loc[valid["eval_loss"].idxmin()]
        last = valid.iloc[-1]
        best_epoch = float(best["epoch"])
        best_step = None if pd.isna(best["step"]) else int(best["step"])
        root = Path(source) if Path(source).is_dir() else None
        current_ckpt = ((root / f"checkpoint-{best_step}") if root is not None
                        and best_step is not None else None)

        info = {
            "model": label,
            "source": str(source),
            "num_train_logs": int(len(train)),
            "num_eval_logs": int(len(valid)),
            "best_epoch": round(best_epoch, 5),
            "best_step": best_step,
            "best_eval_loss": float(best["eval_loss"]),
            "final_eval_loss": float(last["eval_loss"]),
            "final_vs_best_pct": float((last["eval_loss"] /
                                        best["eval_loss"] - 1) * 100),
            "best_checkpoint_available": (current_ckpt.is_dir()
                                          if current_ckpt is not None else pd.NA),
        }
        summaries.append(info)
        dataset[label] = {"train": train, "valid": valid, "metrics": metrics}
        metrics.to_csv(output / f"{_safe_name(label)}_epoch_metrics.csv", index=False)
        train.to_csv(output / f"{_safe_name(label)}_train_steps.csv", index=False)

        old_best = (metadata.get("state") or {}).get("best_model_checkpoint")
        print(f"[BEST] {label}: Epoch {best_epoch:.2f} | "
              f"Eval Loss {float(best['eval_loss']):.6f} | Step {best_step or 'N/A'}")
        if current_ckpt is not None:
            if current_ckpt.is_dir():
                print(f"       Checkpoint đang tồn tại: {current_ckpt}")
            else:
                warnings.warn(f"Không tìm thấy best checkpoint trong {root}: {current_ckpt.name}. "
                              "Có thể checkpoint đã bị xóa bởi save_total_limit.", stacklevel=2)
        if old_best and root is not None and Path(old_best) != current_ckpt:
            print("       Ghi chú: best_model_checkpoint trong TrainerState có thể "
                  "chứa đường dẫn cũ trước khi bạn đổi tên run.")

    def draw_metric(metric, ylabel, title, filename,
                    kind="epoch", annotate_best=False, relative=False):
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for label, item in dataset.items():
            if kind == "validation":
                df = item["valid"]
                x = df["epoch"]
                y = df["eval_loss"]
                if relative:
                    if float(y.iloc[0]) == 0:
                        warnings.warn(f"Eval loss đầu bằng 0: {label}", stacklevel=2)
                        continue
                    y = 100 * (y / float(y.iloc[0]) - 1)
            elif kind == "steps":
                df = item["train"]
                x = df["step"]
                y = df["loss"].rolling(window=10, min_periods=1).mean()
            else:
                df = item["metrics"].dropna(subset=[metric])
                if df.empty:
                    warnings.warn(f"Không có {metric} cho {label}", stacklevel=2)
                    continue
                x, y = df["epoch"], df[metric]
            line, = ax.plot(x, y, label=label, marker=(None if kind == "steps" else "o"),
                            linewidth=1.9, markersize=4)
            if annotate_best and kind == "validation":
                best = item["valid"].loc[item["valid"]["eval_loss"].idxmin()]
                ax.scatter(best["epoch"], best["eval_loss"],
                           color=line.get_color(), marker="*", s=140, zorder=5)
                ax.annotate(f"E{best['epoch']:.0f}: {best['eval_loss']:.3f}",
                            (best["epoch"], best["eval_loss"]), xytext=(6, -14),
                            textcoords="offset points", fontsize=9)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Training step" if kind == "steps" else "Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.27)
        ax.legend()
        if kind != "steps":
            max_epoch = max(float(d["metrics"]["epoch"].max()) for d in dataset.values())
            ax.set_xticks(np.arange(1, int(np.ceil(max_epoch)) + 1))
        if relative:
            ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
        fig.tight_layout()
        _save_chart(fig, output, filename)

    draw_metric("train_loss_mean", "Mean train loss", "Training Loss per Epoch",
                "01_train_loss_epoch")
    draw_metric("eval_loss", "Eval loss", "Validation Loss per Epoch",
                "02_validation_loss_epoch", kind="validation", annotate_best=True)
    draw_metric("eval_loss", "Change from Epoch 1 (%)", "Relative Validation Loss Change",
                "03_relative_validation_loss", kind="validation", relative=True)
    if any(item["metrics"]["grad_norm_mean"].notna().any() for item in dataset.values()):
        draw_metric("grad_norm_mean", "Mean grad norm", "Gradient Norm per Epoch",
                    "04_gradient_norm_epoch")
    draw_metric("loss", "Training loss (moving average: 10 steps)",
                "Training Loss by Step", "05_train_loss_steps", kind="steps")

    # Optional compact four-panel figure for thesis/report overview.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = [
        (axes[0, 0], "train_loss_mean", "Training Loss", "train"),
        (axes[0, 1], "eval_loss", "Validation Loss", "valid"),
        (axes[1, 0], "grad_norm_mean", "Gradient Norm", "grad"),
        (axes[1, 1], "relative", "Eval Loss Change vs Epoch 1 (%)", "rel"),
    ]
    for ax, key, title, panel_type in panels:
        for label, item in dataset.items():
            if panel_type in ("train", "grad"):
                m = item["metrics"].dropna(subset=[key])
                if m.empty:
                    continue
                x, y = m["epoch"], m[key]
            else:
                v = item["valid"]
                x, y = v["epoch"], v["eval_loss"]
                if panel_type == "rel":
                    if float(y.iloc[0]) == 0:
                        continue
                    y = 100 * (y / float(y.iloc[0]) - 1)
            ax.plot(x, y, marker="o", markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        if panel_type == "rel":
            ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)
    fig.suptitle("SFT Training Comparison — Qwen3-8B vs Gemma3-4B", fontsize=14)
    fig.tight_layout()
    _save_chart(fig, output, "00_sft_training_overview")

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "sft_summary.csv", index=False)

    lines = ["# SFT training comparison", "",
             "The per-model Training Loss is the arithmetic mean of recorded step losses.",
             "Validation Loss is measured on the validation split.",
             "Different model architectures/tokenizers mean absolute loss values "
             "are **not** a direct measure of which chatbot is medically better.",
             "Choose candidate checkpoints on validation, and evaluate actual "
             "responses (Judge v2 / safety) before selecting a final model.", ""]
    for row in summaries:
        lines += [f"## {row['model']}",
                  f"- Best epoch: {row['best_epoch']}",
                  f"- Best validation loss: {row['best_eval_loss']:.6f}",
                  f"- Final validation loss: {row['final_eval_loss']:.6f}",
                  f"- Final versus best: {row['final_vs_best_pct']:+.2f}%", ""]
    (output / "README_results.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== SFT SUMMARY ===")
    print(summary[["model", "best_epoch", "best_step", "best_eval_loss",
                   "final_eval_loss", "final_vs_best_pct"]].to_string(index=False))
    print(f"\nĐã lưu 6 biểu đồ (PNG + SVG), CSV và ghi chú tại: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Plot Qwen3/Gemma3 SFT training loss")
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN,
                        help="Qwen run folder, trainer_state.json, .zip or training log")
    parser.add_argument("--gemma", type=Path, default=DEFAULT_GEMMA,
                        help="Gemma run folder, trainer_state.json, .zip or training log")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Folder for plots, CSV and analysis summary")
    args = parser.parse_args()
    plot_sft_comparison({
        "Qwen3-8B (LR=1e-4, 10ep)": args.qwen,
        "Gemma3-4B (LR=2e-5, 10ep)": args.gemma,
    }, output_dir=args.output)


if __name__ == "__main__":
    main()
