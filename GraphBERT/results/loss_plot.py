"""
Plot training metrics from training_history.json for GraphCodeBERT.
Creates a comprehensive visualization of all training metrics in a single figure.
Works from anywhere by finding the project root and searching for training history files.
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def find_project_root(start_path: Path = None) -> Path:
    """Find project root by looking for config.json."""
    if start_path is None:
        start_path = Path(__file__).parent.absolute()

    current = start_path
    while True:
        config_path = current / 'config.json'
        if config_path.exists():
            return current

        parent = current.parent
        if parent == current:  # Reached filesystem root
            raise FileNotFoundError(
                "Could not find project root. "
                "Make sure config.json exists in the project root directory."
            )
        current = parent


def find_training_history(results_dir: Optional[str] = None) -> Path:
    """Find training_history.json in results directory."""
    project_root = find_project_root()

    if results_dir is None:
        # Try to find it in the default results directory
        try:
            config = json.loads((project_root / 'config.json').read_text())
            results_dir = config.get('train', {}).get('output_dir', 'results')
        except Exception:
            results_dir = 'results'

    history_path = project_root / results_dir / 'training_history.json'

    if not history_path.exists():
        raise FileNotFoundError(
            f"training_history.json not found at {history_path}\n"
            f"Make sure training has been completed and results are saved."
        )

    return history_path


def load_training_history(filepath: Path) -> dict:
    """Load training history from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_all_metrics(history: dict, output_path: Path):
    """Plot all metrics in one comprehensive figure."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'train': '#06b6d4',  # Cyan
        'val': '#ef4444',  # Red
        'train_mlm': '#06b6d4',  # Cyan
        'train_edge': '#8b5cf6',  # Violet
        'val_mlm': '#ef4444',  # Red
        'val_edge': '#ec4899',  # Pink
        'lr': '#fbbf24'  # Amber
    }

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    epochs = history['epoch']
    train_batches = history['train_batch_losses']
    val_batches = history['val_batch_losses']

    # 1. Total Loss
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(epochs, history['train_total_loss'], marker='o', linewidth=2.5,
             label='Train Loss', color=colors['train'], markersize=10)
    ax1.plot(epochs, history['val_total_loss'], marker='s', linewidth=2.5,
             label='Val Loss', color=colors['val'], markersize=10)
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Total Loss (Train & Validation)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Train vs Val Bar Chart
    ax2 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(epochs))
    width = 0.35
    ax2.bar(x - width / 2, history['train_total_loss'], width, label='Train',
            color=colors['train'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.bar(x + width / 2, history['val_total_loss'], width, label='Val',
            color=colors['val'], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Train vs Validation Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(epochs)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[1, 1])
    lr_scaled = [lr_val * 1e6 for lr_val in history['learning_rate']]
    ax3.plot(epochs, lr_scaled, marker='o', linewidth=2.5, markersize=10,
             color=colors['lr'], label='Learning Rate')
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('LR (×10⁻⁶)', fontsize=11, fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Train Batch Losses
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(train_batches, linewidth=1, color=colors['train'], alpha=0.6)
    ax4.fill_between(range(len(train_batches)), train_batches, alpha=0.2, color=colors['train'])
    window = 5
    if len(train_batches) > window:
        moving_avg = np.convolve(train_batches, np.ones(window) / window, mode='valid')
        ax4.plot(range(window - 1, len(train_batches)), moving_avg, linewidth=2,
                 color='#3b82f6', label=f'Moving Avg ({window})')
    ax4.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Training Batch Losses', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    if len(train_batches) > window:
        ax4.legend(fontsize=9)

    # 5. Val Batch Losses
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(val_batches, linewidth=1, color=colors['val'], alpha=0.6)
    ax5.fill_between(range(len(val_batches)), val_batches, alpha=0.2, color=colors['val'])
    if len(val_batches) > window:
        moving_avg_val = np.convolve(val_batches, np.ones(window) / window, mode='valid')
        ax5.plot(range(window - 1, len(val_batches)), moving_avg_val, linewidth=2,
                 color='#dc2626', label=f'Moving Avg ({window})')
    ax5.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax5.set_title('Validation Batch Losses', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    if len(val_batches) > window:
        ax5.legend(fontsize=9)

    fig.suptitle('GraphCodeBERT Training: Comprehensive Metrics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_path}")


def print_summary(history: dict):
    """Print training summary statistics."""
    print("\n" + "=" * 70)
    print("GraphCodeBERT Training Summary".center(70))
    print("=" * 70)

    train_loss = history['train_total_loss']
    val_loss = history['val_total_loss']
    train_mlm = history['train_mlm_loss']
    train_edge = history['train_edge_loss']
    val_mlm = history['val_mlm_loss']
    val_edge = history['val_edge_loss']
    best_val = history['best_val_loss']
    best_epoch = history['best_epoch']
    train_batches = history['train_batch_losses']
    val_batches = history['val_batch_losses']

    print(f"\nTotal Training Loss:")
    print(f"  Initial:  {train_loss[0]:.4f}")
    print(f"  Final:    {train_loss[-1]:.4f}")
    if train_loss[-1] < train_loss[0]:
        improvement = ((train_loss[0] - train_loss[-1]) / train_loss[0] * 100)
        print(f"  Improvement: {improvement:.1f}%")
    else:
        worsening = ((train_loss[-1] - train_loss[0]) / train_loss[0] * 100)
        print(f"  Worsening: {worsening:.1f}%")
    print(f"  Min:      {min(train_loss):.4f}")
    print(f"  Max:      {max(train_loss):.4f}")

    print(f"\nTotal Validation Loss:")
    print(f"  Initial:  {val_loss[0]:.4f}")
    print(f"  Final:    {val_loss[-1]:.4f}")
    print(f"  Best:     {best_val:.4f} (Epoch {best_epoch})")
    if val_loss[-1] < val_loss[0]:
        improvement = ((val_loss[0] - val_loss[-1]) / val_loss[0] * 100)
        print(f"  Improvement: {improvement:.1f}%")
    else:
        worsening = ((val_loss[-1] - val_loss[0]) / val_loss[0] * 100)
        print(f"  Worsening: {worsening:.1f}%")
    print(f"  Min:      {min(val_loss):.4f}")
    print(f"  Max:      {max(val_loss):.4f}")

    print(f"\nTraining MLM Loss:")
    print(f"  Initial:  {train_mlm[0]:.4f}")
    print(f"  Final:    {train_mlm[-1]:.4f}")

    print(f"\nTraining Edge Loss:")
    print(f"  Initial:  {train_edge[0]:.4f}")
    print(f"  Final:    {train_edge[-1]:.4f}")

    print(f"\nValidation MLM Loss:")
    print(f"  Initial:  {val_mlm[0]:.4f}")
    print(f"  Final:    {val_mlm[-1]:.4f}")

    print(f"\nValidation Edge Loss:")
    print(f"  Initial:  {val_edge[0]:.4f}")
    print(f"  Final:    {val_edge[-1]:.4f}")

    print(f"\nBatch Losses Statistics:")
    print(f"  Train Batches:      {len(train_batches)}")
    print(f"    Mean: {np.mean(train_batches):.4f}")
    print(f"    Std:  {np.std(train_batches):.4f}")
    print(f"    Min:  {np.min(train_batches):.4f}")
    print(f"    Max:  {np.max(train_batches):.4f}")

    print(f"  Val Batches:        {len(val_batches)}")
    print(f"    Mean: {np.mean(val_batches):.4f}")
    print(f"    Std:  {np.std(val_batches):.4f}")
    print(f"    Min:  {np.min(val_batches):.4f}")
    print(f"    Max:  {np.max(val_batches):.4f}")

    print(f"\nLearning Rate Schedule:")
    print(f"  Epochs: {len(history['learning_rate'])}")
    for i, (epoch, lr) in enumerate(zip(history['epoch'], history['learning_rate'])):
        print(f"    Epoch {epoch}: {lr:.2e}")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to generate plots."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot GraphCodeBERT training metrics from training_history.json'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Results directory (relative to project root). Uses config.json if not provided.'
    )
    parser.add_argument(
        '--summary_only',
        action='store_true',
        help='Print summary statistics only, do not generate plots'
    )

    args = parser.parse_args()

    try:
        # Find training history file
        history_path = find_training_history(args.results_dir)
        project_root = find_project_root()

        print("\n" + "=" * 70)
        print("GraphCodeBERT Metrics Plotting".center(70))
        print("=" * 70)
        print(f"Project root: {project_root}")
        print(f"History file: {history_path}")
        print("=" * 70)

        # Load history
        history = load_training_history(history_path)

        # Print summary
        print_summary(history)

        # Generate plot
        if not args.summary_only:
            output_dir = history_path.parent
            output_file = output_dir.parent / 'results' / 'graphcodebert_all_metrics.png'
            plot_all_metrics(history, output_file)
            print(f"✓ All metrics visualization complete!\n")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}\n")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        exit(1)


if __name__ == "__main__":
    main()