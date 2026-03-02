#python javascript_eval.py --data_file /home/mczap/GraphBert/GraphBERT/data/javascript/train.jsonl --model_checkpoint /home/mczap/GraphBert/GraphBERT/models/20k_20k_mixed_retokenized/best_model --mask_ratio 0.15 --top_k 10 --max_examples 500  --max_seq_length 512 --output_file results/evaluation_results_javascript.json

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer
from tqdm import tqdm

try:
    from tree_sitter import Language, Parser
    import tree_sitter_javascript as tsjs
    TS_AVAILABLE = True
    JS_LANGUAGE = Language(tsjs.language())
    ts_parser = Parser(JS_LANGUAGE)
except ImportError:
    TS_AVAILABLE = False

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


def load_test_snippets_from_jsonl(jsonl_file: str, max_examples: Optional[int] = None) -> List[str]:
    project_root = find_project_root()

    if os.path.isabs(jsonl_file):
        file_path = Path(jsonl_file)
    else:
        file_path = project_root / jsonl_file

    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    snippets = []
    print(f"Loading snippets from {file_path}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            try:
                data = json.loads(line.strip())
                code = data.get('code') or data.get('source_code')
                if code:
                    snippets.append(code)
            except json.JSONDecodeError:
                print(f"Warning: Line {i+1} is not valid JSON, skipping")
                continue

    print(f"Successfully loaded {len(snippets)} snippets from {file_path}\n")
    return snippets


class MLMEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer, device: Optional[str] = None, max_seq_length: int = 512):
        if not TS_AVAILABLE:
            raise RuntimeError("Tree-sitter is required for evaluation.")

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_seq_length = max_seq_length
        self.max_code_tokens = max_seq_length - 2

        print(f"Using device: {self.device}")
        print(f"Max sequence length: {max_seq_length}")
        print(f"Loading model from {model_path}...")

        self.tokenizer = tokenizer
        self.is_pretrained = model_path in ["microsoft/graphcodebert-base"] or "/" in model_path

        try:
            self.model = RobertaForMaskedLM.from_pretrained(model_path).to(self.device).eval()
            source = "HuggingFace Hub" if self.is_pretrained else "local checkpoint"
            print(f"Model loaded successfully from {source}: {model_path}!\n")
        except Exception as e:
            print(f"Failed to load from HuggingFace: {e}")
            print("Trying custom format...")

            import os
            model_bin_path = os.path.join(model_path, 'model.bin')

            if not os.path.exists(model_bin_path):
                raise FileNotFoundError(f"No model file found in {model_path}")

            checkpoint = torch.load(model_bin_path, map_location=self.device, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                from transformers import RobertaConfig
                config_dict = checkpoint.get('config', {})
                if isinstance(config_dict, dict):
                    config = RobertaConfig(**config_dict)
                else:
                    config = config_dict

                self.model = RobertaForMaskedLM(config).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("Model loaded successfully (custom format)!\n")
            else:
                raise ValueError(f"Unknown checkpoint format in {model_bin_path}")

    def extract_dfg_for_snippet(self, code_bytes: bytes) -> List[Tuple]:
        tree = ts_parser.parse(code_bytes)
        root = tree.root_node
        defs, uses = defaultdict(list), defaultdict(list)
        tokens, node_map = [], {}

        def find_tokens(node):
            if node.type in ['identifier']:
                if id(node) not in node_map:
                    node_map[id(node)] = len(tokens)
                    tokens.append(node)
            for child in node.children:
                find_tokens(child)

        find_tokens(root)

        def is_def(node):
            p = node.parent
            return p and (p.type in ['variable_declarator', 'formal_parameters',
                                      'function_declaration', 'arrow_function',
                                      'catch_clause', 'for_in_statement'] or
                          (p.type == 'assignment_expression' and node == p.child_by_field_name('left')))

        def find_vars(node):
            if node.type in ['identifier']:
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

        if len(code_tokens) > self.max_code_tokens:
            code_tokens = code_tokens[:self.max_code_tokens]

        if len(code_tokens) < 2:
            return None

        num_mask = max(1, int(len(code_tokens) * mask_ratio))

        if num_mask > len(code_tokens):
            num_mask = len(code_tokens)

        if num_mask < 1:
            return None

        try:
            mask_pos = sorted(random.sample(range(len(code_tokens)), num_mask))
        except ValueError:
            return None

        orig_tokens = [code_tokens[i] for i in mask_pos]
        masked_tokens = code_tokens.copy()

        for pos in mask_pos:
            masked_tokens[pos] = self.tokenizer.mask_token

        try:
            inputs = self.preprocess_for_graphcodebert(code, masked_tokens)
        except Exception:
            return None

        with torch.no_grad():
            try:
                logits = self.model(**{k: v.to(self.device) for k, v in inputs.items()}).logits
            except RuntimeError:
                return None

        top1_correct, top5_correct, log_probs = 0, 0, []

        for i, pos in enumerate(mask_pos):
            probs = torch.softmax(logits[0, pos + 1], dim=-1)
            top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))
            orig_token = orig_tokens[i]
            top_preds = self.tokenizer.convert_ids_to_tokens(top_indices)

            correct_token_prob = 1e-9
            found_top5 = False

            for rank, (pred, prob) in enumerate(zip(top_preds, top_probs), 1):
                if pred == orig_token:
                    correct_token_prob = prob.item()
                    if not found_top5 and rank <= 5:
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
        'snippets_skipped': results.get('snippets_skipped', 0),
        'config': results.get('config', {})
    }

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"Evaluation results saved to: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate GraphCodeBERT MLM model on JavaScript code from JSONL file'
    )
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--max_examples', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--model_checkpoint', type=str, default='best_model')
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--max_seq_length', type=int, default=512)

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
    print("GraphCodeBERT MLM Evaluation (JavaScript, from JSONL)".center(70))
    print("=" * 70)
    print(f"Project root: {project_root}")
    print("\nEvaluation Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("=" * 70 + "\n")

    data_file = args.data_file or eval_config.get('data_file', None)
    if data_file is None:
        print("Error: --data_file must be specified (either as argument or in config)")
        exit(1)

    max_examples = args.max_examples or eval_config.get('max_examples', None)
    mask_ratio = args.mask_ratio or eval_config.get('mask_ratio', 0.15)
    top_k = args.top_k or eval_config.get('top_k', 10)
    max_seq_length = args.max_seq_length or eval_config.get('max_seq_length', 512)

    print("Loading tokenizer from 'microsoft/graphcodebert-base'...")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

    if args.model_checkpoint.startswith("microsoft/") or "/" in args.model_checkpoint:
        model_path = args.model_checkpoint
        print(f"Using HuggingFace model: {model_path}")
    else:
        output_dir = project_root / config.get('train', {}).get('output_dir', 'results')
        model_path = output_dir / args.model_checkpoint
        if not model_path.exists():
            print(f"Error: Model checkpoint not found at {model_path}")
            exit(1)

    print(f"Model path: {model_path}\n")

    try:
        snippets_to_evaluate = load_test_snippets_from_jsonl(data_file, max_examples)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    if not snippets_to_evaluate:
        print("No snippets to evaluate. Exiting.")
        exit(1)

    evaluator = MLMEvaluator(str(model_path), tokenizer=tokenizer, max_seq_length=max_seq_length)

    aggregated_results = {
        'total_top1_correct': 0,
        'total_top5_correct': 0,
        'total_masked': 0,
        'all_log_probs': [],
        'snippets_skipped': 0
    }

    print(f"Evaluating {len(snippets_to_evaluate)} snippets...\n")

    for snippet in tqdm(snippets_to_evaluate, desc="Evaluating"):
        results = evaluator.evaluate_snippet(snippet, mask_ratio, top_k)

        if results:
            aggregated_results['total_top1_correct'] += results['top1_correct']
            aggregated_results['total_top5_correct'] += results['top5_correct']
            aggregated_results['total_masked'] += results['num_masked']
            aggregated_results['all_log_probs'].extend(results['log_probs'])
        else:
            aggregated_results['snippets_skipped'] += 1

    if aggregated_results['total_masked'] > 0:
        total_masked = aggregated_results['total_masked']
        top1_acc = aggregated_results['total_top1_correct'] / total_masked
        top5_acc = aggregated_results['total_top5_correct'] / total_masked
        perplexity = np.exp(-np.mean(aggregated_results['all_log_probs']))

        results_dict = {
            'snippets_evaluated': len(snippets_to_evaluate),
            'snippets_skipped': aggregated_results['snippets_skipped'],
            'total_masked_tokens': total_masked,
            'top1_correct': aggregated_results['total_top1_correct'],
            'top5_correct': aggregated_results['total_top5_correct'],
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'perplexity': perplexity,
            'config': {
                'data_file': data_file,
                'max_examples': max_examples,
                'mask_ratio': mask_ratio,
                'top_k': top_k,
                'max_seq_length': max_seq_length,
                'model_checkpoint': args.model_checkpoint
            }
        }

        print("\n" + "=" * 70)
        print("Evaluation Results".center(70))
        print("=" * 70)
        print(f"Data file:              {data_file}")
        print(f"Snippets evaluated:     {len(snippets_to_evaluate)}")
        print(f"Snippets skipped:       {aggregated_results['snippets_skipped']}")
        print(f"Total masked tokens:    {total_masked}")
        print("-" * 70)
        print(f"Top-1 Accuracy:         {top1_acc:.2%} ({aggregated_results['total_top1_correct']}/{total_masked})")
        print(f"Top-5 Accuracy:         {top5_acc:.2%} ({aggregated_results['total_top5_correct']}/{total_masked})")
        print(f"Perplexity:             {perplexity:.4f}")
        print("=" * 70 + "\n")

        if args.output_file:
            if os.path.isabs(args.output_file):
                results_path = Path(args.output_file)
            else:
                results_path = project_root / args.output_file
        else:
            output_dir = project_root / config.get('train', {}).get('output_dir', 'results')
            results_path = output_dir / 'evaluation_results_javascript.json'

        save_evaluation_results(results_dict, results_path)
    else:
        print("No valid results to report.")


if __name__ == "__main__":
    main()