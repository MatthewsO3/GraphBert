import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import RobertaForMaskedLM, RobertaTokenizer

random.seed(42)
torch.manual_seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Project helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_project_root(start_path: Path = None) -> Path:
    if start_path is None:
        start_path = Path(__file__).parent.absolute()
    current = start_path
    while True:
        if (current / 'config.json').exists():
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
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    raise FileNotFoundError(f"config.json not found at {config_path}")


# ─────────────────────────────────────────────────────────────────────────────
# JSONL loader
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[Dict]:
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# GraphCodeBERT input builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graphcodebert_inputs(
    masked_code_tokens: List[str],
    dfg: List,
    tokenizer: RobertaTokenizer,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    # Hard caps — same logic as model.py's convert_sample_to_features
    MAX_DFG  = min(64, max_length // 4)
    MAX_CODE = max_length - MAX_DFG - 3   # [CLS] + code + [SEP] + dfg + [SEP]

    masked_code_tokens = masked_code_tokens[:MAX_CODE]
    valid_code_len = len(masked_code_tokens)

    adj: Dict[int, List[int]] = defaultdict(list)
    nodes: List[Tuple[str, int]] = []
    node_map: Dict[int, int] = {}

    for edge in dfg:
        var, use_pos, _, _, dep_pos_list = edge[0], edge[1], edge[2], edge[3], edge[4]
        if use_pos >= valid_code_len:
            continue
        if use_pos not in node_map:
            node_map[use_pos] = len(nodes)
            nodes.append((var, use_pos))
        use_idx = node_map[use_pos]
        for def_pos in dep_pos_list:
            if def_pos >= valid_code_len:
                continue
            if def_pos not in node_map:
                node_map[def_pos] = len(nodes)
                nodes.append((var, def_pos))
            adj[use_idx].append(node_map[def_pos])

    # Cap DFG nodes
    if len(nodes) > MAX_DFG:
        keep = set(range(MAX_DFG))
        nodes = nodes[:MAX_DFG]
        adj = defaultdict(list, {
            i: [j for j in adjs if j in keep]
            for i, adjs in adj.items() if i in keep
        })

    tokens = [tokenizer.cls_token] + masked_code_tokens + [tokenizer.sep_token]
    dfg_start = len(tokens)
    tokens.extend([tokenizer.unk_token] * len(nodes))
    tokens.append(tokenizer.sep_token)

    # Final guard — should never trigger after the caps above, but just in case
    assert len(tokens) <= max_length, f"Sequence too long: {len(tokens)} > {max_length}"

    ids     = tokenizer.convert_tokens_to_ids(tokens)
    pos_ids = (list(range(valid_code_len + 2))
               + [0] * len(nodes)
               + [valid_code_len + 2])

    seq_len  = len(ids)
    mask     = np.zeros((seq_len, seq_len), dtype=bool)
    code_len = valid_code_len + 2

    mask[:code_len, :code_len] = True
    for i in range(seq_len):
        mask[i, i] = True
    for i, (_, code_pos) in enumerate(nodes):
        dfg_abs  = dfg_start + i
        code_abs = code_pos + 1
        mask[dfg_abs, code_abs] = mask[code_abs, dfg_abs] = True
    for i, adjs in adj.items():
        for j in adjs:
            u, v = dfg_start + i, dfg_start + j
            mask[u, v] = mask[v, u] = True

    return {
        'input_ids':      torch.tensor([ids]),
        'attention_mask': torch.tensor([mask.tolist()]),
        'position_ids':   torch.tensor([pos_ids]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class MLMEvaluator:
    def __init__(self, model_path: str, tokenizer: RobertaTokenizer,
                 device: Optional[str] = None):
        self.device    = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.tokenizer = tokenizer
        print(f"Using device: {self.device}")
        print(f"Loading model from {model_path}...")
        self.model = (RobertaForMaskedLM
                      .from_pretrained(model_path)
                      .to(self.device)
                      .eval())
        print("Model loaded successfully!\n")

    def evaluate_sample(self, sample: Dict, mask_ratio: float, top_k: int,
                    max_length: int = 512) -> Optional[Dict]:
        code_tokens = sample.get('code_tokens', [])
        dfg         = sample.get('dataflow_graph', [])

        code_tokens = [t for t in code_tokens if t not in
                    (self.tokenizer.cls_token, self.tokenizer.sep_token)]

        # Pre-truncate here too so mask_pos indices stay valid after truncation
        MAX_CODE    = max_length - min(64, max_length // 4) - 3
        code_tokens = code_tokens[:MAX_CODE]

        if not code_tokens:
            return None

        num_mask  = max(1, int(len(code_tokens) * mask_ratio))
        mask_pos  = sorted(random.sample(range(len(code_tokens)), num_mask))
        orig_toks = [code_tokens[i] for i in mask_pos]

        masked_tokens = code_tokens.copy()
        for pos in mask_pos:
            masked_tokens[pos] = self.tokenizer.mask_token

        inputs = build_graphcodebert_inputs(masked_tokens, dfg, self.tokenizer, max_length)

        with torch.no_grad():
            logits = self.model(
                **{k: v.to(self.device) for k, v in inputs.items()}
            ).logits

        top1_correct = top5_correct = 0
        log_probs: List[float] = []

        for i, pos in enumerate(mask_pos):
            probs      = torch.softmax(logits[0, pos + 1], dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            orig_token = orig_toks[i]
            top_preds  = self.tokenizer.convert_ids_to_tokens(top_indices)

            correct_prob = 1e-9
            found_top5   = False
            for rank, (pred, prob) in enumerate(zip(top_preds, top_probs), 1):
                if pred == orig_token:
                    correct_prob = prob.item()
                    if not found_top5:
                        top5_correct += 1
                        found_top5    = True
                    if rank == 1:
                        top1_correct += 1

            log_probs.append(np.log(correct_prob))

        return {
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
            'num_masked':   num_mask,
            'log_probs':    log_probs,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-language aggregation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    evaluator: MLMEvaluator,
    samples: List[Dict],
    mask_ratio: float,
    top_k: int,
    label: str,
) -> Dict:
    total_top1 = total_top5 = total_masked = 0
    all_log_probs: List[float] = []
    evaluated = 0

    for i, sample in enumerate(samples, 1):
        result = evaluator.evaluate_sample(sample, mask_ratio, top_k)
        if result is None:
            continue
        total_top1    += result['top1_correct']
        total_top5    += result['top5_correct']
        total_masked  += result['num_masked']
        all_log_probs += result['log_probs']
        evaluated     += 1
        if i % 100 == 0:
            print(f"  [{label}] {i}/{len(samples)} samples processed...")

    if total_masked == 0:
        print(f"  [{label}] No valid results.")
        return {}

    return {
        'snippets_evaluated':  evaluated,
        'total_masked_tokens': total_masked,
        'top1_correct':        total_top1,
        'top5_correct':        total_top5,
        'top1_accuracy':       float(total_top1 / total_masked),
        'top5_accuracy':       float(total_top5 / total_masked),
        'perplexity':          float(np.exp(-np.mean(all_log_probs))),
        '_mean_log_prob':      float(np.mean(all_log_probs)),  # used for combined perplexity
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing & saving  (identical style to evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_results(label: str, metrics: Dict):
    print("\n" + "=" * 70)
    print(f"Evaluation Results — {label}".center(70))
    print("=" * 70)
    print(f"Snippets evaluated:     {metrics['snippets_evaluated']}")
    print(f"Total masked tokens:    {metrics['total_masked_tokens']}")
    print("-" * 70)
    print(f"Top-1 Accuracy:         {metrics['top1_accuracy']:.2%}"
          f" ({metrics['top1_correct']}/{metrics['total_masked_tokens']})")
    print(f"Top-5 Accuracy:         {metrics['top5_accuracy']:.2%}"
          f" ({metrics['top5_correct']}/{metrics['total_masked_tokens']})")
    print(f"Perplexity:             {metrics['perplexity']:.4f}")
    print("=" * 70 + "\n")


def save_results(results: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(results))
    for lang_data in clean.get('languages', {}).values():
        lang_data.pop('_mean_log_prob', None)  # internal key, don't expose in file
    with open(output_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"Evaluation results saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate GraphCodeBERT on cpp_val.jsonl and erlang_val.jsonl'
    )
    parser.add_argument('--cpp_val',          type=str,   default=None)
    parser.add_argument('--erlang_val',       type=str,   default=None)
    parser.add_argument('--mask_ratio',       type=float, default=None)
    parser.add_argument('--top_k',            type=int,   default=None)
    parser.add_argument('--model',            type=str,   default=None,
                        help='HuggingFace model ID (e.g. microsoft/graphcodebert-base) '
                             'OR local path. Overrides --model_checkpoint.')
    parser.add_argument('--model_checkpoint', type=str,   default='best_model',
                        help='Checkpoint folder name inside output_dir '
                             '(used when --model is not set)')
    parser.add_argument('--max_samples',      type=int,   default=None)

    try:
        config       = load_config()
        project_root = find_project_root()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    parser.set_defaults(**config.get('evaluate', {}))
    args = parser.parse_args()

    output_dir      = project_root / config.get('train', {}).get('output_dir', 'models')
    cpp_val_path    = project_root / (args.cpp_val    or 'data/cpp_val.jsonl')
    erlang_val_path = project_root / (args.erlang_val or 'data/erlang_val.jsonl')

    # ── Resolve model path ────────────────────────────────────────────────────
    # --model takes priority; falls back to output_dir / --model_checkpoint
    if args.model:
        model_path     = args.model          # HuggingFace ID or absolute/relative path
        model_path_str = args.model
    else:
        local_path     = output_dir / args.model_checkpoint
        if not local_path.exists():
            print(f"Error: model checkpoint not found at {local_path}")
            print(f"  Tip: pass --model microsoft/graphcodebert-base to use the base model")
            exit(1)
        model_path     = local_path
        model_path_str = str(local_path)

    print("\n" + "=" * 70)
    print("GraphCodeBERT — JSONL Validation Evaluation".center(70))
    print("=" * 70)
    print(f"Project root:      {project_root}")
    print(f"Model:             {model_path_str}")
    print(f"C++ val file:      {cpp_val_path}")
    print(f"Erlang val file:   {erlang_val_path}")
    print(f"Mask ratio:        {args.mask_ratio}")
    print(f"Top-k:             {args.top_k}")
    print(f"Max samples/lang:  {args.max_samples or 'all'}")
    print("=" * 70 + "\n")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    evaluator = MLMEvaluator(model_path_str, tokenizer=tokenizer)

    combined_results: Dict = {
        'model_checkpoint': model_path_str,
        'mask_ratio':       args.mask_ratio,
        'top_k':            args.top_k,
        'languages':        {},
    }

    for lang, path in [('C++', cpp_val_path), ('Erlang', erlang_val_path)]:
        if not path.exists():
            print(f"Warning: {lang} val file not found at {path}, skipping.")
            continue

        print(f"Loading {lang} samples from {path}...")
        samples = load_jsonl(str(path))
        print(f"  Loaded {len(samples)} samples.")

        if args.max_samples and len(samples) > args.max_samples:
            samples = random.sample(samples, args.max_samples)
            print(f"  Capped to {args.max_samples} samples.")

        print(f"  Evaluating {lang}...")
        metrics = evaluate_dataset(evaluator, samples, args.mask_ratio, args.top_k, lang)
        if not metrics:
            continue

        print_results(lang, metrics)
        combined_results['languages'][lang] = metrics

    lang_results = combined_results['languages']
    if lang_results:
        all_top1   = sum(v['top1_correct']        for v in lang_results.values())
        all_top5   = sum(v['top5_correct']        for v in lang_results.values())
        all_masked = sum(v['total_masked_tokens'] for v in lang_results.values())
        all_snips  = sum(v['snippets_evaluated']  for v in lang_results.values())

        weighted_log_prob = sum(
            v['_mean_log_prob'] * v['total_masked_tokens']
            for v in lang_results.values()
        ) / all_masked

        combined_results['combined'] = {
            'snippets_evaluated':  all_snips,
            'total_masked_tokens': all_masked,
            'top1_correct':        all_top1,
            'top5_correct':        all_top5,
            'top1_accuracy':       float(all_top1 / all_masked),
            'top5_accuracy':       float(all_top5 / all_masked),
            'perplexity':          float(np.exp(-weighted_log_prob)),
        }
        print_results('Combined (C++ + Erlang)', combined_results['combined'])

    save_results(combined_results, output_dir / 'evaluation_results_val.json')
    
if __name__ == "__main__":
    main()