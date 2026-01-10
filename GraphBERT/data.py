

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import RobertaTokenizer
from tree_sitter import Language, Parser
from tqdm import tqdm

import tree_sitter_cpp as tscpp

CPP_LANGUAGE = Language(tscpp.language())
ts_parser = Parser(CPP_LANGUAGE)
print("✓ Tree-sitter initialized")

# Load tokenizer with explicit settings for consistency
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
print("✓ Tokenizer loaded")


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


def load_config() -> Dict:
    """Load config.json from project root."""
    project_root = find_project_root()
    config_path = project_root / 'config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def extract_dataflow_graph(code_bytes: bytes, tree) -> List[Tuple]:
    """
    Extract data flow graph from C++ code following GraphCodeBERT format.
    Maps tree-sitter nodes to their sequential token index (0, 1, 2...)
    Creates edges from variable definitions to uses.
    """
    root_node = tree.root_node
    var_definitions = defaultdict(list)
    var_uses = defaultdict(list)
    tokens = []
    node_to_token_pos = {}

    def extract_tokens_recursive(node):
        """Recursively extract token positions for identifiers."""
        if node.type in ['identifier', 'field_identifier']:
            if id(node) not in node_to_token_pos:
                node_to_token_pos[id(node)] = len(tokens)
                tokens.append(node)

        for child in node.children:
            extract_tokens_recursive(child)

    extract_tokens_recursive(root_node)

    def is_definition(node):
        """Check if a node represents a variable definition."""
        parent = node.parent
        if not parent:
            return False
        if parent.type in ['declaration', 'init_declarator', 'parameter_declaration']:
            return True
        if parent.type == 'assignment_expression' and node == parent.child_by_field_name('left'):
            return True
        return False

    def traverse_for_vars(node):
        """Traverse AST and collect variable definitions and uses."""
        if node.type in ['identifier', 'field_identifier']:
            var_name = code_bytes[node.start_byte:node.end_byte].decode('utf8', errors='ignore')
            token_pos = node_to_token_pos.get(id(node), -1)
            if token_pos != -1:
                (var_definitions if is_definition(node) else var_uses)[var_name].append(token_pos)

        for child in node.children:
            traverse_for_vars(child)

    traverse_for_vars(root_node)

    # Build DFG edges: connect uses to their preceding definitions
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
    """
    Basic filtering to determine if code snippet should be processed.
    Based on length and presence of C++ constructs.
    """
    if len(code) < 100 or len(code) > 10000:
        return False

    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False

    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code:
        return False

    return True


def preprocess_code(code: str, idx: int) -> Optional[Dict]:
    """
    Preprocess C++ code and extract DFG in GraphCodeBERT format.
    Filters out code snippets that are too short/long or have insufficient DFG.
    Returns a dictionary with code, tokens, DFG, and metadata.
    """
    try:
        code_bytes = code.encode('utf8')
        tree = ts_parser.parse(code_bytes)
        tokens = tokenizer.tokenize(code, add_prefix_space=True)

        # Filter by token count
        if len(tokens) < 10 or len(tokens) > 450:
            return None

        # Extract and filter by DFG quality
        dfg = extract_dataflow_graph(code_bytes, tree)
        if not dfg or len(dfg) < 2:
            return None

        return {
            'idx': f'cpp::{idx}',
            'code': code,
            'code_tokens': tokens,
            'dataflow_graph': dfg,
            'docstring': '',
            'docstring_tokens': []
        }
    except Exception:
        return None


def stream_and_process_dataset(output_file: str, max_samples: Optional[int] = None):
    """
    Stream dataset, extract DFG, and save in JSONL format.

    Args:
        output_file: Output JSONL file path
        max_samples: Maximum number of samples to process (None for all in streaming mode)
    """
    project_root = find_project_root()
    output_path = project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset in streaming mode...")
    dataset = load_dataset("codeparrot/github-code-clean", "C++-all", split="train", streaming=True)

    processed_count = 0
    with open(output_path, 'w', encoding='utf-8') as f, tqdm(desc="Processing C++ files") as pbar:
        for example in dataset:
            if max_samples and processed_count >= max_samples:
                break

            code = example.get('code')
            if not code or not should_keep_code(code):
                continue

            processed = preprocess_code(code, processed_count)
            if processed:
                f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                processed_count += 1
                pbar.update(1)

    print(f"\n{'=' * 70}")
    print(f"Processing complete!")
    print(f"Total samples processed and saved: {processed_count}")
    print(f"Data saved to: {output_path}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract DFG from C++ code')
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSONL file for processed data (relative to project root, uses config if not provided)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None for all in streaming mode, uses config if not provided)'
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config()
        project_root = find_project_root()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Use CLI args if provided, otherwise fall back to config
    output_file = args.output_file
    if output_file is None:
        output_file = config.get('data', {}).get('output_file', 'data/cpp_functions.jsonl')

    max_samples = args.max_samples
    if max_samples is None:
        max_samples = config.get('data', {}).get('max_samples', None)

    print("\n" + "=" * 70)
    print("DFG Extraction Configuration")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Output file: {output_file}")
    print(f"Max samples: {max_samples if max_samples else 'All (streaming mode)'}")
    print("=" * 70 + "\n")

    stream_and_process_dataset(output_file, max_samples)


if __name__ == "__main__":
    main()