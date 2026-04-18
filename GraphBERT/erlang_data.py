import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import RobertaTokenizer
from tqdm import tqdm


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


def should_keep_code(code: str) -> bool:
    """
    Filter for reasonable Erlang code.
    Check for:
    - Code length (reasonable function/module size)
    - Number of lines
    - Presence of Erlang-like structure
    """
    if len(code) < 100 or len(code) > 10000:
        return False

    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False

    # Check for Erlang-specific patterns
    erlang_patterns = [
        'fun(',      # anonymous function
        '->',        # function clause arrow
        ':-',        # directive or rule
        'lists:',    # standard library calls
        'erlang:',   # erlang module calls
        'case ',     # case expression
        'receive ',  # message receive
    ]

    # At least one Erlang pattern should be present
    if not any(pattern in code for pattern in erlang_patterns):
        if '->' not in code and 'module(' not in code:
            return False

    return True


def preprocess_code(code: str, idx: int, tokenizer, version: str = 'unknown') -> Optional[Dict]:
    """
    Preprocess Erlang code for MLM: tokenize and filter.
    No DFG extraction needed.
    
    Parameters:
    -----------
    code : str
        The source code
    idx : int
        Sample index
    tokenizer : RobertaTokenizer
        Tokenizer instance
    version : str
        'old' or 'new' to indicate if this is old_contents or new_contents
    """
    try:
        tokens = tokenizer.tokenize(code, add_prefix_space=True)

        # Filter by token count (reasonable function size)
        if len(tokens) < 10 or len(tokens) > 450:
            return None

        return {
            'idx': f'erlang::{idx}',
            'version': version,  # Track whether old or new version
            'code': code,
            'code_tokens': tokens,
            # No DFG needed for MLM evaluation
        }
    except Exception:
        return None


def stream_and_process_dataset(
    dataset_name: str,
    output_file: str,
    tokenizer,
    config_name: str = None,
    max_samples: Optional[int] = None
):
    """
    Stream and process Erlang dataset from HuggingFace.
    Optimized for MLM (no DFG extraction).
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., "codeparrot/github-code-clean")
    output_file : str
        Output JSONL file path
    tokenizer : RobertaTokenizer
        Tokenizer instance
    config_name : str
        Config/language subset (e.g., "Erlang-all")
    max_samples : int
        Max samples to process (None for all)
    """
    project_root = find_project_root()
    output_path = project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {dataset_name} (config: {config_name}) in streaming mode...")
    
    try:
        dataset = load_dataset(
            dataset_name,
            config_name,
            split="train",
            streaming=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure dataset '{dataset_name}' with config '{config_name}' exists on HuggingFace.")
        raise

    processed_count = 0
    skipped_count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f, tqdm(desc="Processing Erlang files") as pbar:
        for example in dataset:
            if max_samples and processed_count >= max_samples:
                break

            # Process both old_contents and new_contents separately
            for field_name, version in [('new_contents', 'new'), ('old_contents', 'old')]:
                code = example.get(field_name)
                
                if not code:
                    skipped_count += 1
                    pbar.update(1)
                    continue

                if not should_keep_code(code):
                    skipped_count += 1
                    pbar.update(1)
                    continue

                processed = preprocess_code(code, processed_count, tokenizer, version=version)
                if processed:
                    f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                    processed_count += 1
                    pbar.update(1)
                else:
                    skipped_count += 1
                    pbar.update(1)

    print(f"\n{'=' * 70}")
    print(f"Processing complete!")
    print(f"Total samples processed and saved: {processed_count}")
    print(f"Total samples skipped (filtered): {skipped_count}")
    print(f"Data saved to: {output_path}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract Erlang code for MLM evaluation (no DFG needed)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='HuggingFace dataset name (e.g., "codeparrot/github-code-clean")'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Dataset config/language (e.g., "Erlang-all")'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSONL file for processed data'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (None for all)'
    )

    args = parser.parse_args()

    try:
        config = load_config()
        project_root = find_project_root()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Load tokenizer once
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

    # Resolve parameters with fallbacks to config
    dataset_name = args.dataset or config.get('data', {}).get('erlang_dataset', 'codeparrot/github-code-clean')
    config_name = args.config or config.get('data', {}).get('erlang_config', 'Erlang-all')
    output_file = args.output_file or config.get('data', {}).get('erlang_output_file', 'data/erlang_functions.jsonl')
    max_samples = args.max_samples or config.get('data', {}).get('erlang_max_samples', None)

    print("\n" + "=" * 70)
    print("Erlang Code Extraction (MLM Evaluation)")
    print("=" * 70)
    print(f"Project root: {project_root}")
    print(f"Dataset: {dataset_name}")
    print(f"Config: {config_name}")
    print(f"Output file: {output_file}")
    print(f"Max samples: {max_samples if max_samples else 'All (streaming mode)'}")
    print("=" * 70 + "\n")

    stream_and_process_dataset(
        dataset_name,
        output_file,
        tokenizer,
        config_name,
        max_samples
    )


if __name__ == "__main__":
    main()