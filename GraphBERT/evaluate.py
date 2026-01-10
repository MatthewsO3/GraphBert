import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import RobertaForMaskedLM, RobertaTokenizer

try:
    from tree_sitter import Language, Parser

    import tree_sitter_cpp as tscpp

    TS_AVAILABLE = True
    CPP_LANGUAGE = Language(tscpp.language())
    ts_parser = Parser(CPP_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False
    print("Warning: tree_sitter/tree_sitter_cpp not found. DFG extraction will fail.")

random.seed(42)
torch.manual_seed(42)


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


def load_config() -> Dict:
    project_root = find_project_root()
    config_path = project_root / 'config.json'

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"config.json not found at {config_path}")


def should_keep_code(code: str) -> bool:
    if not code:
        return False
    if len(code) < 100 or len(code) > 10000:
        return False
    lines = code.count('\n')
    if lines < 3 or lines > 500:
        return False
    if 'void ' not in code and 'int ' not in code and 'class ' not in code and 'std::' not in code:
        return False
    return True


def fetch_test_snippets_from_db(skip_n: int, take_n: int, tokenizer: RobertaTokenizer, max_tokens: int = 100) -> List[
    str]:
    print(f"Fetching {take_n} test snippets from 'codeparrot/github-code-clean'...")
    print(f"Skipping first {skip_n} samples (training data)")
    print(f"Max tokens per snippet: {max_tokens}")

    try:
        dataset = load_dataset("codeparrot/github-code-clean", "C++-all", split="train", streaming=True)

        snippets = []
        filtered_dataset = (
            ex for ex in dataset.skip(skip_n)
            if should_keep_code(ex.get('code')) and
               len(tokenizer.tokenize(ex.get('code'), add_prefix_space=True)) < max_tokens
        )

        for example in filtered_dataset:
            if len(snippets) >= take_n:
                break
            snippets.append(example['code'])

        if not snippets:
            print("\nWarning: Could not fetch any valid snippets matching criteria.")
            print(f"Try reducing 'training_data_size' in config.json or checking internet connection.\n")
        else:
            print(f"Successfully fetched {len(snippets)} snippets for evaluation.\n")

        return snippets
    except Exception as e:
        print(f"Error fetching data from Hub: {e}")
        return []


class MLMEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: Optional[str] = None):
        if not TS_AVAILABLE:
            raise RuntimeError("Tree-sitter is required for evaluation.")

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")
        print(f"Loading model from {model_path}...")

        self.tokenizer = tokenizer
        self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        print("Model loaded successfully!\n")

    def extract_dfg_for_snippet(self, code_bytes: bytes) -> List[Tuple]:
        tree = ts_parser.parse(code_bytes)
        root = tree.root_node
        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node):
            if node.type in ['identifier', 'field_identifier']:
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            return p and (p.type in ['declaration', 'init_declarator', 'parameter_declaration'] or
                          (p.type == 'assignment_expression' and node == p.child_by_field_name('left')))

        def find_vars(node):
            if node.type in ['identifier', 'field_identifier']:
                name = code_bytes[node.start_byte:node.end_byte].decode('utf8', 'ignore')
                pos = node_map.get(id(node), -1)
                if pos != -1:
                    (defs if is_def(node) else uses)[name].append(pos)
            for child in node.children:
                find_vars(child)

        find_vars(root)

        edges = []
        for name, use_positions in uses.items():
            def_positions = sorted(defs.get(name, []))
            for use_pos in use_positions:
                preds = [d for d in def_positions if d < use_pos]
                if preds:
                    edges.append((name, use_pos, "comesFrom", [name], [preds[-1]]))

        return edges

    def preprocess_for_graphcodebert(self, code: str, masked_code_tokens: List[str]):
        dfg = self.extract_dfg_for_snippet(code.encode('utf8'))
        adj, nodes, node_map = defaultdict(list), [], {}

        for var, use_pos, _, _, dep_list in dfg:
            if use_pos not in node_map:
                node_map[use_pos] = len(nodes)
                nodes.append((var, use_pos))
            for def_pos in dep_list:
                if def_pos not in node_map:
                    node_map[def_pos] = len(nodes)
                    nodes.append((var, def_pos))
                adj[node_map[use_pos]].append(node_map[def_pos])

        tokens = [self.tokenizer.cls_token] + masked_code_tokens + [self.tokenizer.sep_token]
        dfg_start = len(tokens)
        tokens.extend([self.tokenizer.unk_token] * len(nodes))
        tokens.append(self.tokenizer.sep_token)

        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        pos_ids = list(range(len(masked_code_tokens) + 2)) + [0] * len(nodes) + [len(masked_code_tokens) + 2]

        mask = np.zeros((len(ids), len(ids)), dtype=bool)
        code_len = len(masked_code_tokens) + 2
        mask[:code_len, :code_len] = True

        for i in range(len(ids)):
            mask[i, i] = True

        for i, (_, code_pos) in enumerate(nodes):
            if code_pos < len(masked_code_tokens):
                dfg_abs, code_abs = dfg_start + i, code_pos + 1
                mask[dfg_abs, code_abs] = mask[code_abs, dfg_abs] = True

        for i, adjs in adj.items():
            for j in adjs:
                u, v = dfg_start + i, dfg_start + j
                mask[u, v] = mask[v, u] = True

        return {
            'input_ids': torch.tensor([ids]),
            'attention_mask': torch.tensor([mask.tolist()]),
            'position_ids': torch.tensor([pos_ids])
        }

    def evaluate_snippet(self, code: str, mask_ratio: float, top_k: int) -> Optional[Dict]:
        code_tokens = self.tokenizer.tokenize(code, add_prefix_space=True)
        if not code_tokens:
            return None

        num_mask = max(1, int(len(code_tokens) * mask_ratio))
        mask_pos = sorted(random.sample(range(len(code_tokens)), num_mask))
        orig_tokens = [code_tokens[i] for i in mask_pos]
        masked_tokens = code_tokens.copy()

        for pos in mask_pos:
            masked_tokens[pos] = self.tokenizer.mask_token

        inputs = self.preprocess_for_graphcodebert(code, masked_tokens)

        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits

        top1_correct, top5_correct, log_probs = 0, 0, []

        for i, pos in enumerate(mask_pos):
            probs = torch.softmax(logits[0, pos + 1], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            orig_token = orig_tokens[i]
            top_preds = self.tokenizer.convert_ids_to_tokens(top_indices)

            correct_token_prob = 1e-9
            found_top5 = False

            for rank, (pred, prob) in enumerate(zip(top_preds, top_probs), 1):
                if pred == orig_token:
                    correct_token_prob = prob.item()
                    if not found_top5:
                        top5_correct += 1
                        found_top5 = True
                    if rank == 1:
                        top1_correct += 1

            log_probs.append(np.log(correct_token_prob))

        return {
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'num_masked': num_mask,
            'log_probs': log_probs
        }


def save_evaluation_results(results: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_serializable = {
        'snippets_evaluated': results['snippets_evaluated'],
        'total_masked_tokens': results['total_masked_tokens'],
        'top1_accuracy': float(results['top1_accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'perplexity': float(results['perplexity']),
        'top1_correct': int(results['top1_correct']),
        'top5_correct': int(results['top5_correct']),
        'config': results.get('config', {})
    }

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Evaluation results saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate GraphCodeBERT MLM model on C++')
    parser.add_argument('--mask_ratio', type=float, default=None, help='Ratio of tokens to mask')
    parser.add_argument('--top_k', type=int, default=None, help='Top-k predictions to consider')
    parser.add_argument('--use_database_snippets', action='store_true', help='Fetch snippets from database')
    parser.add_argument('--model_checkpoint', type=str, default='best_model', help='Model checkpoint name')

    try:
        config = load_config()
        project_root = find_project_root()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    eval_config = config.get('evaluate', {})
    parser.set_defaults(**eval_config)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GraphCodeBERT Model Evaluation".center(70))
    print("=" * 70)
    print(f"Project root: {project_root}")
    print("\nEvaluation Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")

    print("Loading tokenizer from 'microsoft/graphcodebert-base'...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

    output_dir = project_root / config.get('train', {}).get('output_dir', 'results')
    model_path = output_dir / args.model_checkpoint

    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        exit(1)

    print(f"Model path: {model_path}\n")

    snippets_to_evaluate = []
    if args.use_database_snippets:
        training_data_size = eval_config.get('training_data_size')
        num_test_snippets = eval_config.get('num_test_snippets')
        max_tokens = eval_config.get('max_tokens_per_snippet', 100)

        if training_data_size is None or num_test_snippets is None:
            print("Error: 'training_data_size' and 'num_test_snippets' required for database snippets.")
            exit(1)

        snippets_to_evaluate = fetch_test_snippets_from_db(
            skip_n=training_data_size,
            take_n=num_test_snippets,
            tokenizer=tokenizer,
            max_tokens=max_tokens
        )
    else:
        print("Error: No evaluation snippets configured. Set 'use_database_snippets' to true.")
        exit(1)

    if not snippets_to_evaluate:
        print("No snippets to evaluate. Exiting.")
        exit(1)

    evaluator = MLMEvaluator(str(model_path), tokenizer=tokenizer)

    aggregated_results = {
        'total_top1_correct': 0,
        'total_top5_correct': 0,
        'total_masked': 0,
        'all_log_probs': []
    }

    print(f"Evaluating {len(snippets_to_evaluate)} snippets...\n")

    for i, snippet in enumerate(snippets_to_evaluate, 1):
        results = evaluator.evaluate_snippet(snippet, args.mask_ratio, args.top_k)

        if results:
            aggregated_results['total_top1_correct'] += results['top1_correct']
            aggregated_results['total_top5_correct'] += results['top5_correct']
            aggregated_results['total_masked'] += results['num_masked']
            aggregated_results['all_log_probs'].extend(results['log_probs'])


    if aggregated_results['total_masked'] > 0:
        total_masked = aggregated_results['total_masked']
        top1_acc = aggregated_results['total_top1_correct'] / total_masked
        top5_acc = aggregated_results['total_top5_correct'] / total_masked
        perplexity = np.exp(-np.mean(aggregated_results['all_log_probs']))

        results_dict = {
            'snippets_evaluated': len(snippets_to_evaluate),
            'total_masked_tokens': total_masked,
            'top1_correct': aggregated_results['total_top1_correct'],
            'top5_correct': aggregated_results['total_top5_correct'],
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'perplexity': perplexity,
            'config': {
                'mask_ratio': args.mask_ratio,
                'top_k': args.top_k,
                'model_checkpoint': args.model_checkpoint,
                'use_database_snippets': args.use_database_snippets
            }
        }

        print("\n" + "=" * 70)
        print("Evaluation Results".center(70))
        print("=" * 70)
        print(f"Snippets evaluated:     {len(snippets_to_evaluate)}")
        print(f"Total masked tokens:    {total_masked}")
        print("-" * 70)
        print(f"Top-1 Accuracy:         {top1_acc:.2%} ({aggregated_results['total_top1_correct']}/{total_masked})")
        print(f"Top-5 Accuracy:         {top5_acc:.2%} ({aggregated_results['total_top5_correct']}/{total_masked})")
        print(f"Perplexity:             {perplexity:.4f}")
        print("=" * 70 + "\n")

        results_path = output_dir / 'evaluation_results.json'
        save_evaluation_results(results_dict, results_path)
    else:
        print("No valid results to report.")


if __name__ == "__main__":
    main()