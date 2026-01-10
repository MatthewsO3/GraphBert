"""
Plot evaluation metrics from evaluation_results.json for GraphCodeBERT.
Creates visualizations of accuracy metrics and perplexity.
Works from anywhere by finding the project root and results directory.
"""

import json
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def find_project_root(start_path: Path = None) -> Path:
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


def find_evaluation_results(results_dir: Optional[str] = None) -> Path:
    project_root = find_project_root()

    if results_dir is None:
        try:
            config = json.loads((project_root / 'config.json').read_text())
            results_dir = config.get('train', {}).get('output_dir', 'results')
        except Exception:
            results_dir = 'results'

    results_path = project_root / results_dir / 'evaluation_results.json'

    if not results_path.exists():
        raise FileNotFoundError(
            f"evaluation_results.json not found at {results_path}\n"
            f"Make sure evaluation has been completed: python evaluate.py"
        )

    return results_path


def load_evaluation_results(filepath: Path) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_metrics(results: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))

    accuracies = {
        'Top-1': results['top1_accuracy'],
        'Top-5': results['top5_accuracy']
    }

    colors = ['#06b6d4', '#8b5cf6']
    bars = ax.bar(accuracies.keys(), accuracies.values(), color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2, width=0.6)

    for bar, (label, accuracy) in zip(bars, accuracies.items()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{accuracy:.2%}\n({int(accuracy * results["total_masked_tokens"])}/{results["total_masked_tokens"]})',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax.set_title('GraphCodeBERT: Top-K Accuracy on C++ Code', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to: {output_path}")


def plot_perplexity(results: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 7))

    perplexity = results['perplexity']

    bar = ax.bar(['Perplexity'], [perplexity], color='#ef4444', alpha=0.8,
                  edgecolor='black', linewidth=2, width=0.4)

    for b in bar:
        height = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2., height,
                f'{perplexity:.4f}',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax.set_ylabel('Perplexity', fontsize=13, fontweight='bold')
    ax.set_title('GraphCodeBERT: Perplexity on C++ Code', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylim(0, perplexity * 1.15)
    ax.grid(True, alpha=0.3, axis='y')



    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved perplexity plot to: {output_path}")


def plot_combined_metrics(results: dict, output_path: Path):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Color palette
    colors = {
        'top1': '#06b6d4',  # Cyan
        'top5': '#8b5cf6',  # Violet
        'perplexity': '#ef4444'  # Red
    }

    # 1. Top-K Accuracy Bar Chart
    ax1 = fig.add_subplot(gs[0, 0])
    accuracies = {
        'Top-1': results['top1_accuracy'],
        'Top-5': results['top5_accuracy']
    }
    bars1 = ax1.bar(accuracies.keys(), accuracies.values(),
                    color=[colors['top1'], colors['top5']], alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    for bar, (_, acc) in zip(bars1, accuracies.items()):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{acc:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Top-K Prediction Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Perplexity Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    bar2 = ax2.bar(['Perplexity'], [results['perplexity']], color=colors['perplexity'],
                   alpha=0.8, edgecolor='black', linewidth=1.5, width=0.4)
    for b in bar2:
        height = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2., height,
                f'{results["perplexity"]:.2f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax2.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
    ax2.set_title('Model Perplexity', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, results['perplexity'] * 1.15)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Correct vs Incorrect Predictions (Top-1)
    ax3 = fig.add_subplot(gs[1, 0])
    top1_correct = results['top1_correct']
    top1_incorrect = results['total_masked_tokens'] - top1_correct
    sizes = [top1_correct, top1_incorrect]
    labels = [f'Correct\n{top1_correct}', f'Incorrect\n{top1_incorrect}']
    colors_pie = [colors['top1'], '#cccccc']
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Top-1 Accuracy Breakdown', fontsize=12, fontweight='bold')

    # 4. Correct vs Incorrect Predictions (Top-5)
    ax4 = fig.add_subplot(gs[1, 1])
    top5_correct = results['top5_correct']
    top5_incorrect = results['total_masked_tokens'] - top5_correct
    sizes = [top5_correct, top5_incorrect]
    labels = [f'Correct\n{top5_correct}', f'Incorrect\n{top5_incorrect}']
    colors_pie = [colors['top5'], '#cccccc']
    wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax4.set_title('Top-5 Accuracy Breakdown', fontsize=12, fontweight='bold')

    fig.suptitle('GraphCodeBERT Evaluation: Comprehensive Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {output_path}")


def print_evaluation_summary(results: dict):
    print("\n" + "=" * 70)
    print("GraphCodeBERT Evaluation Summary".center(70))
    print("=" * 70)

    print(f"\nDataset Statistics:")
    print(f"  Snippets evaluated:     {results['snippets_evaluated']}")
    print(f"  Total masked tokens:    {results['total_masked_tokens']}")

    print(f"\nAccuracy Metrics:")
    print(f"  Top-1 Accuracy:         {results['top1_accuracy']:.2%} ({results['top1_correct']}/{results['total_masked_tokens']})")
    print(f"  Top-5 Accuracy:         {results['top5_accuracy']:.2%} ({results['top5_correct']}/{results['total_masked_tokens']})")

    print(f"\nPerplexity:")
    print(f"  Model Perplexity:       {results['perplexity']:.4f}")

    print(f"\nConfiguration:")
    config = results.get('config', {})
    for k, v in config.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Plot GraphCodeBERT evaluation metrics from evaluation_results.json'
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
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all plots (default behavior)'
    )
    parser.add_argument(
        '--accuracy_only',
        action='store_true',
        help='Generate only accuracy plot'
    )
    parser.add_argument(
        '--perplexity_only',
        action='store_true',
        help='Generate only perplexity plot'
    )

    args = parser.parse_args()

    try:
        results_path = find_evaluation_results(args.results_dir)
        project_root = find_project_root()

        print("\n" + "=" * 70)
        print("GraphCodeBERT Evaluation Metrics Plotting".center(70))
        print("=" * 70)
        print(f"Project root: {project_root}")
        print(f"Results file: {results_path}")
        print("=" * 70)

        results = load_evaluation_results(results_path)

        print_evaluation_summary(results)

        if not args.summary_only:
            output_dir = results_path.parent
            generate_all = args.all or (not args.accuracy_only and not args.perplexity_only)

            if args.accuracy_only or generate_all:
                accuracy_path = output_dir.parent / 'results' / 'accuracy.png'
                plot_accuracy_metrics(results, accuracy_path)

            if args.perplexity_only or generate_all:
                perplexity_path = output_dir.parent / 'results' / 'perplexity.png'
                plot_perplexity(results, perplexity_path)

            if generate_all:
                combined_path = output_dir.parent / 'results' / 'combined.png'
                plot_combined_metrics(results, combined_path)

            print(f"\n✓ Evaluation visualization complete!\n")

    except FileNotFoundError as e:
        print(f"\n Error: {e}\n")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"\n Error parsing JSON: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n Unexpected error: {e}\n")
        exit(1)


if __name__ == "__main__":
    main()