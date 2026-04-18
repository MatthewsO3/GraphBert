"""
Plot training metrics from training_history.json.
Creates a comprehensive visualization of all training metrics in a single figure.
Usage: python plot_training_metrics.py [--history_path PATH] [--output_path PATH] [--summary_only]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def find_history_file(start_dir: Path = Path(".")) -> Path:
    """Walk up from start_dir looking for training_history.json."""
    current = start_dir.resolve()
    while True:
        candidate = current / "training_history.json"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find training_history.json. "
                "Pass --history_path explicitly."
            )
        current = parent


def load_history(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(h: dict) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("Training Summary".center(70))
    print(sep)

    def _row(label, values):
        init, final, mn, mx = values[0], values[-1], min(values), max(values)
        pct = (init - final) / init * 100
        direction = "↓" if final < init else "↑"
        print(f"\n  {label}:")
        print(f"    Initial : {init:.4f}")
        print(f"    Final   : {final:.4f}  ({direction}{abs(pct):.1f}%)")
        print(f"    Min/Max : {mn:.4f} / {mx:.4f}")

    _row("Train Total Loss",  h["train_total_loss"])
    _row("Val Total Loss",    h["val_total_loss"])
    _row("Train MLM Loss",    h["train_mlm_loss"])
    _row("Train Edge Loss",   h["train_edge_loss"])
    _row("Val MLM Loss",      h["val_mlm_loss"])
    _row("Val Edge Loss",     h["val_edge_loss"])

    print(f"\n  Best Val Loss : {h['best_val_loss']:.4f}  (epoch {h['best_epoch']})")

    tb = h["train_batch_losses"]
    vb = h["val_batch_losses"]
    print(f"\n  Train batches : {len(tb):,}  "
          f"mean={np.mean(tb):.4f}  std={np.std(tb):.4f}  "
          f"min={np.min(tb):.4f}  max={np.max(tb):.4f}")
    print(f"  Val batches   : {len(vb):,}  "
          f"mean={np.mean(vb):.4f}  std={np.std(vb):.4f}  "
          f"min={np.min(vb):.4f}  max={np.max(vb):.4f}")

    print(f"\n  Learning-rate schedule:")
    for ep, lr in zip(h["epoch"], h["learning_rate"]):
        print(f"    Epoch {ep} : {lr:.2e}")

    # Per-language losses (if present)
    langs = ["python", "javascript", "java"]
    available = [l for l in langs if f"{l}_total_loss" in h]
    if available:
        print(f"\n  Per-language final total loss:")
        for lang in available:
            vals = h[f"{lang}_total_loss"]
            print(f"    {lang.capitalize():12s}: {vals[-1]:.4f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "train":      "#06b6d4",   # cyan
    "val":        "#ef4444",   # red
    "train_mlm":  "#06b6d4",
    "train_edge": "#8b5cf6",   # violet
    "val_mlm":    "#ef4444",
    "val_edge":   "#ec4899",   # pink
    "lr":         "#fbbf24",   # amber
    "python":     "#22c55e",   # green
    "javascript": "#f97316",   # orange
    "java":       "#a855f7",   # purple
}

MOVING_AVG_WIN = 20   # window for batch-loss smoothing


def _moving_avg(data, window):
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def plot_metrics(h: dict, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-darkgrid")

    epochs = h["epoch"]
    train_batches = h["train_batch_losses"]
    val_batches   = h["val_batch_losses"]

    langs = ["python", "javascript", "java"]
    available_langs = [l for l in langs if f"{l}_total_loss" in h]
    has_langs = bool(available_langs)

    n_rows = 4 if has_langs else 3
    fig = plt.figure(figsize=(18, 5 * n_rows))
    gs  = fig.add_gridspec(n_rows, 2, hspace=0.35, wspace=0.28)

    # ── Row 0: Total loss (full width) ──────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(epochs, h["train_total_loss"], marker="o", lw=2.5, ms=9,
             color=COLORS["train"], label="Train")
    ax0.plot(epochs, h["val_total_loss"],   marker="s", lw=2.5, ms=9,
             color=COLORS["val"],   label="Val")
    best_ep = h["best_epoch"]
    ax0.axvline(best_ep, color="#fbbf24", lw=1.5, ls="--",
                label=f"Best val (epoch {best_ep})")
    ax0.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax0.set_title("Total Loss – Train & Validation", fontsize=13, fontweight="bold")
    ax0.legend(fontsize=10)
    ax0.set_xticks(epochs)

    # ── Row 1 left: MLM loss ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(epochs, h["train_mlm_loss"], marker="o", lw=2.5, ms=8,
             color=COLORS["train_mlm"], label="Train MLM")
    ax1.plot(epochs, h["val_mlm_loss"],   marker="s", lw=2.5, ms=8,
             color=COLORS["val_mlm"],   label="Val MLM")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.set_title("MLM Loss", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xticks(epochs)

    # ── Row 1 right: Edge loss ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(epochs, h["train_edge_loss"], marker="o", lw=2.5, ms=8,
             color=COLORS["train_edge"], label="Train Edge")
    ax2.plot(epochs, h["val_edge_loss"],   marker="s", lw=2.5, ms=8,
             color=COLORS["val_edge"],   label="Val Edge")
    ax2.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax2.set_title("Edge (DFG) Loss", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_xticks(epochs)

    # ── Row 2 left: Train batch losses ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    x_tb = np.arange(len(train_batches))
    ax3.plot(x_tb, train_batches, lw=0.8, color=COLORS["train"], alpha=0.4)
    ax3.fill_between(x_tb, train_batches, alpha=0.1, color=COLORS["train"])
    if len(train_batches) > MOVING_AVG_WIN:
        ma = _moving_avg(train_batches, MOVING_AVG_WIN)
        ax3.plot(range(MOVING_AVG_WIN - 1, len(train_batches)), ma,
                 lw=2, color="#3b82f6", label=f"MA({MOVING_AVG_WIN})")
        ax3.legend(fontsize=9)
    ax3.set_xlabel("Batch", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Loss",  fontsize=11, fontweight="bold")
    ax3.set_title("Training Batch Losses", fontsize=12, fontweight="bold")

    # ── Row 2 right: Val batch losses ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    x_vb = np.arange(len(val_batches))
    ax4.plot(x_vb, val_batches, lw=0.8, color=COLORS["val"], alpha=0.4)
    ax4.fill_between(x_vb, val_batches, alpha=0.1, color=COLORS["val"])
    if len(val_batches) > MOVING_AVG_WIN:
        ma_v = _moving_avg(val_batches, MOVING_AVG_WIN)
        ax4.plot(range(MOVING_AVG_WIN - 1, len(val_batches)), ma_v,
                 lw=2, color="#dc2626", label=f"MA({MOVING_AVG_WIN})")
        ax4.legend(fontsize=9)
    ax4.set_xlabel("Batch", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Loss",  fontsize=11, fontweight="bold")
    ax4.set_title("Validation Batch Losses", fontsize=12, fontweight="bold")

    # ── Row 3 (optional): Per-language total loss ────────────────────────────
    if has_langs:
        ax5 = fig.add_subplot(gs[3, 0])
        for lang in available_langs:
            ax5.plot(epochs, h[f"{lang}_total_loss"],
                     marker="o", lw=2.2, ms=7,
                     color=COLORS[lang], label=lang.capitalize())
        ax5.set_ylabel("Loss", fontsize=11, fontweight="bold")
        ax5.set_title("Per-Language Total Loss", fontsize=12, fontweight="bold")
        ax5.legend(fontsize=10)
        ax5.set_xticks(epochs)

        # Learning-rate schedule in the remaining cell
        ax6 = fig.add_subplot(gs[3, 1])
        lr_scaled = [v * 1e6 for v in h["learning_rate"]]
        ax6.plot(epochs, lr_scaled, marker="o", lw=2.5, ms=9,
                 color=COLORS["lr"])
        ax6.fill_between(epochs, lr_scaled, alpha=0.15, color=COLORS["lr"])
        ax6.set_ylabel("LR (×10⁻⁶)", fontsize=11, fontweight="bold")
        ax6.set_title("Learning-Rate Schedule", fontsize=12, fontweight="bold")
        ax6.set_xticks(epochs)
    else:
        # No language data – put LR in row 2 area as full-width strip
        ax_lr = fig.add_subplot(gs[2, :])   # overwrite row 2 centre
        lr_scaled = [v * 1e6 for v in h["learning_rate"]]
        ax_lr.plot(epochs, lr_scaled, marker="o", lw=2.5, ms=9,
                   color=COLORS["lr"])
        ax_lr.fill_between(epochs, lr_scaled, alpha=0.15, color=COLORS["lr"])
        ax_lr.set_ylabel("LR (×10⁻⁶)", fontsize=11, fontweight="bold")
        ax_lr.set_title("Learning-Rate Schedule", fontsize=12, fontweight="bold")
        ax_lr.set_xticks(epochs)

    fig.suptitle("GraphCodeBERT – Comprehensive Training Metrics",
                 fontsize=16, fontweight="bold", y=1.002)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved plot  →  {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GraphCodeBERT training metrics from training_history.json"
    )
    parser.add_argument(
        "--history_path", type=str, default=None,
        help="Path to training_history.json. Auto-detected if omitted."
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Where to save the PNG. Defaults to <history_dir>/training_metrics.png"
    )
    parser.add_argument(
        "--summary_only", action="store_true",
        help="Print summary statistics only; skip plot generation."
    )
    args = parser.parse_args()

    # Resolve history file
    if args.history_path:
        history_path = Path(args.history_path)
        if not history_path.exists():
            print(f"❌  File not found: {history_path}")
            raise SystemExit(1)
    else:
        try:
            history_path = find_history_file(Path("."))
        except FileNotFoundError as e:
            print(f"❌  {e}")
            raise SystemExit(1)

    # Resolve output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = history_path.parent / "training_metrics.png"

    print(f"History : {history_path}")
    print(f"Output  : {output_path}")

    history = load_history(history_path)
    print_summary(history)

    if not args.summary_only:
        try:
            plot_metrics(history, output_path)
        except Exception as exc:
            print(f"❌  Plot generation failed: {exc}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()

#python all_pl_loss_plot.py --history_path /home/mczap/GraphBert/GraphBERT/models/20k_20k_mixed_retokenized/training_history.json --output_path plots/20k_20k_retokenized/metrics.png
