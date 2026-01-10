import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def find_project_root(start_path: Path = None) -> Path:
    if start_path is None:
        start_path = Path(__file__).parent.absolute()

    current = start_path
    while True:
        config_path = current / 'config.json'
        if config_path.exists():
            return current

        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find project root. "
                "Make sure config.json exists in the project root directory."
            )
        current = parent


def load_config() -> dict:
    project_root = find_project_root()
    config_path = project_root / 'config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def run_command(cmd: List[str], stage_name: str, cwd: Optional[Path] = None) -> bool:
    print(f"\n{'=' * 70}")
    print(f"Running: {stage_name}".center(70))
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=cwd, check=True)
        print(f"\n {stage_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n {stage_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n {stage_name} failed: Script not found")
        return False
    except Exception as e:
        print(f"\n {stage_name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Run complete GraphCodeBERT pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Run all stages
  python run.py --stage training          # Start from training
  python run.py --stage evaluation        # Start from evaluation
  python run.py --stage plotting          # Run only plotting
  python run.py --skip_evaluation         # Skip evaluation
  python run.py --skip_plotting           # Skip plotting
        """
    )

    parser.add_argument(
        '--stage',
        type=str,
        choices=['data', 'training', 'evaluation', 'plotting'],
        default='data',
        help='Stage to start from (default: data)'
    )
    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Skip evaluation stage'
    )
    parser.add_argument(
        '--skip_plotting',
        action='store_true',
        help='Skip plotting stages'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be run without executing'
    )

    args = parser.parse_args()

    try:
        project_root = find_project_root()
        config = load_config()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GraphCodeBERT Pipeline Runner".center(70))
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Starting stage: {args.stage}")
    print(f"Skip evaluation: {args.skip_evaluation}")
    print(f"Skip plotting: {args.skip_plotting}")
    print("=" * 70)

    # Define pipeline stages
    stages = {
        'data': {
            'cmd': ['python', 'data.py'],
            'name': 'Data Extraction & DFG Processing',
            'enabled': True
        },
        'training': {
            'cmd': ['python', 'train.py'],
            'name': 'Model Training',
            'enabled': True
        },
        'evaluation': {
            'cmd': ['python', 'evaluate.py'],
            'name': 'Model Evaluation',
            'enabled': not args.skip_evaluation
        },
        'loss_plot': {
            'cmd': ['python', 'results/loss_plot.py'],
            'name': 'Training Metrics Visualization',
            'enabled': not args.skip_plotting
        },
        'metrics_plot': {
            'cmd': ['python', 'results/metrics_plot.py'],
            'name': 'Evaluation Metrics Visualization',
            'enabled': not args.skip_plotting and not args.skip_evaluation
        }
    }

    stage_order = ['data', 'training', 'evaluation', 'loss_plot', 'metrics_plot']
    start_idx = stage_order.index(args.stage) if args.stage in stage_order else 0

    stages_to_run = []
    for stage in stage_order[start_idx:]:
        if stages[stage]['enabled']:
            stages_to_run.append(stage)

    if not stages_to_run:
        print("No stages to run. Check your options.")
        sys.exit(1)

    print(f"\nStages to run:")
    for i, stage in enumerate(stages_to_run, 1):
        print(f"  {i}. {stages[stage]['name']}")

    if args.dry_run:
        print("\n📋 DRY RUN - Not executing")
        for stage in stages_to_run:
            print(f"  Would run: {' '.join(stages[stage]['cmd'])}")
        sys.exit(0)

    results = {}
    for stage in stages_to_run:
        stage_info = stages[stage]
        success = run_command(stage_info['cmd'], stage_info['name'], cwd=project_root)
        results[stage] = success

        if not success:
            print(f"\nPipeline stopped at {stage_info['name']}")
            print(f"Fix the error and run: python run.py --stage {stage}")
            break

    print(f"\n{'=' * 70}")
    print("Pipeline Summary".center(70))
    print("=" * 70)

    all_success = all(results.values())


    if all_success:
        print("\nPipeline completed successfully!")
        print(f"\nResults saved to: {project_root / config.get('train', {}).get('output_dir', 'results')}")
        print("\nGenerated files:")
        print("  - training_history.json")
        print("  - training_summary.json")
        print("  - training_metrics.csv")
        print("  - best_model/ (model checkpoint)")
        print("  - checkpoints/ (epoch checkpoints)")
        if not args.skip_evaluation:
            print("  - evaluation_results.json")
        if not args.skip_plotting:
            print("  - graphcodebert_all_metrics.png (training)")
            if not args.skip_evaluation:
                print("  - accuracy.png")
                print("  - perplexity.png")
                print("  - combined.png")
        print()
        sys.exit(0)
    else:
        print(f"\nPipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()