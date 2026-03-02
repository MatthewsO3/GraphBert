"python python_data.py --max_samples 500 --output_dir data/python"
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser
from tqdm import tqdm
import tree_sitter_python as tspy
import tree_sitter_java as tsjava

"""PYTHON_LANGUAGE = Language(tspy.language())
ts_parser = Parser(PYTHON_LANGUAGE)
print("Tree-sitter initialized")"""

JAVA_LANGUAGE = Language(tsjava.language())
ts_parser = Parser(JAVA_LANGUAGE)
print("Tree-sitter initialized for Java")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
print("Tokenizer loaded")

# Set random seed for reproducible splits
random.seed(42)


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


def load_config() -> Dict:
    project_root = find_project_root()
    config_path = project_root / 'config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def extract_dataflow_graph(code_bytes: bytes, tree) -> List[Tuple]:
    root_node = tree.root_node
    var_definitions = defaultdict(list)
    var_uses = defaultdict(list)
    tokens = []
    node_to_token_pos = {}

    def extract_tokens_recursive(node):
        if node.type in ['identifier']:
            if id(node) not in node_to_token_pos:
                node_to_token_pos[id(node)] = len(tokens)
                tokens.append(node)
        for child in node.children:
            extract_tokens_recursive(child)

    extract_tokens_recursive(root_node)

    def is_definition(node):
        parent = node.parent
        if not parent:
            return False
        if parent.type in ['local_variable_declaration', 'enhanced_for_statement',
                    'formal_parameter', 'method_declaration', 'catch_formal_parameter']:
            return True
        if parent.type == 'assignment_expression' and node == parent.child_by_field_name('left'):
            return True
        return False

    def traverse_for_vars(node):
        if node.type in ['identifier']:
            var_name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
            token_pos = node_to_token_pos.get(id(node), -1)
            if token_pos != -1:
                (var_definitions if is_definition(node) else var_uses)[var_name].append(token_pos)
        for child in node.children:
            traverse_for_vars(child)

    traverse_for_vars(root_node)

    dfg_edges = []
    for var_name, uses in var_uses.items():
        defs = sorted(var_definitions.get(var_name, []))
        for use_pos in uses:
            preceding_defs = [d for d in defs if d < use_pos]
            if preceding_defs:
                def_pos = preceding_defs[-1]
                dfg_edges.append((var_name, use_pos, "comesFrom", [var_name], [def_pos]))

    return dfg_edges


def should_keep_code(code: str) -> bool:
    if len(code) < 100 or len(code) > 10000:
        return False

    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False

    if 'class ' not in code and 'void ' not in code and 'import ' not in code:
        return False

    return True


def preprocess_code(code: str, idx: int) -> Optional[Dict]:
    try:
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)
        tokens = tokenizer.tokenize(code, add_prefix_space=True)

        if len(tokens) < 10 or len(tokens) > 450:
            return None

        dfg = extract_dataflow_graph(code_bytes, tree)
        if not dfg or len(dfg) < 2:
            return None

        return {
            'idx': f'python::{idx}',
            'code': code,
            'code_tokens': tokens,
            'dataflow_graph': dfg,
            'docstring': '',
            'docstring_tokens': []
        }
    except Exception:
        return None


def stream_and_process_dataset(
    output_dir: str,
    max_samples: Optional[int] = None,
    skip_num_snippets: Optional[int] = None,
    train_ratio: float = 0.9
) -> Tuple[int, int, int]:
    """
    Process Python dataset, split into train/val, and save to JSONL files.

    Args:
        output_dir: Directory to save train.jsonl and val.jsonl
        max_samples: Maximum number of samples to process
        skip_num_snippets: Number of snippets to skip from the start
        train_ratio: Fraction for training set (default 0.9 for 90% train, 10% val)

    Returns:
        Tuple of (total_processed, train_count, val_count)
    """
    project_root = find_project_root()
    output_path = project_root / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    val_file = output_path / "val.jsonl"

    print("Loading dataset in streaming mode...")
    dataset = load_dataset("codeparrot/github-code-clean", "Java-all", split="train", streaming=True)

    # Skip the first X snippets if specified
    if skip_num_snippets and skip_num_snippets > 0:
        print(f"Skipping first {skip_num_snippets} snippets...")
        dataset = dataset.skip(skip_num_snippets)

    # First pass: collect all valid processed samples in memory
    print("\nFirst pass: Processing and filtering samples...")
    processed_samples = []
    skipped_count = 0
    filtered_count = 0
    total_seen = 0

    with tqdm(desc="Processing Python files") as pbar:
        for example in dataset:
            if max_samples and len(processed_samples) >= max_samples:
                break

            total_seen += 1
            code = example.get('code')

            if not code or not should_keep_code(code):
                filtered_count += 1
                pbar.update(1)
                continue

            processed = preprocess_code(code, len(processed_samples))
            if processed:
                processed_samples.append(processed)
                pbar.update(1)
            else:
                skipped_count += 1
                pbar.update(1)

    total_processed = len(processed_samples)

    if total_processed == 0:
        print("Error: No valid samples were processed!")
        return 0, 0, 0

    # Second pass: split and write to files
    print(f"\nSecond pass: Splitting into train ({train_ratio:.0%}) and val ({1-train_ratio:.0%})...")

    # Shuffle for random split
    random.shuffle(processed_samples)
    split_idx = int(total_processed * train_ratio)
    train_samples = processed_samples[:split_idx]
    val_samples = processed_samples[split_idx:]

    # Write train set
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(train_samples, desc="Writing training set"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Write validation set
    with open(val_file, 'w', encoding='utf-8') as f:
        for sample in tqdm(val_samples, desc="Writing validation set"):
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Processing and Splitting Complete!")
    print(f"{'=' * 70}")
    print(f"Total snippets seen: {total_seen}")
    if skip_num_snippets:
        print(f"Snippets skipped from start: {skip_num_snippets}")
    print(f"Samples filtered (quality checks): {filtered_count}")
    print(f"Samples skipped (DFG extraction): {skipped_count}")
    print(f"Total valid samples processed: {total_processed}")
    print(f"-" * 70)
    print(f"Training samples: {len(train_samples)} ({len(train_samples)/total_processed*100:.1f}%)")
    print(f"Validation samples: {len(val_samples)} ({len(val_samples)/total_processed*100:.1f}%)")
    print(f"-" * 70)
    print(f"Output directory: {output_path}")
    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")
    print(f"{'=' * 70}\n")

    return total_processed, len(train_samples), len(val_samples)


def main():
    parser = argparse.ArgumentParser(
        description='Extract DFG from Python code and split into train/val sets'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for train.jsonl and val.jsonl (relative to project root, uses config if not provided)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (uses config if not provided)'
    )
    parser.add_argument(
        '--skip_num_snippets',
        type=int,
        default=None,
        help='Number of snippets to skip from the start (uses config if not provided)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=None,
        help='Train/val split ratio, e.g., 0.9 for 90%% train, 10%% val (default 0.9)'
    )

    args = parser.parse_args()

    try:
        config = load_config()
        project_root = find_project_root()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Get parameters from args or config
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config.get('data', {}).get('output_dir', 'data')

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.get('data', {}).get('max_samples', None)

    skip_num_snippets = args.skip_num_snippets
    if skip_num_snippets is None:
        skip_num_snippets = config.get('data', {}).get('skip_num_snippets', None)

    train_ratio = args.train_ratio
    if train_ratio is None:
        train_ratio = config.get('data', {}).get('train_ratio', 0.9)

    print("\n" + "=" * 70)
    print("Python Code Preprocessing Configuration")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples: {max_samples if max_samples else 'All (streaming mode)'}")
    print(f"Skip snippets from start: {skip_num_snippets if skip_num_snippets else 0}")
    print(f"Train/Val split ratio: {train_ratio:.0%} / {1-train_ratio:.0%}")
    print("=" * 70 + "\n")

    stream_and_process_dataset(
        output_dir,
        max_samples=max_samples,
        skip_num_snippets=skip_num_snippets,
        train_ratio=train_ratio
    )


if __name__ == "__main__":
    main()