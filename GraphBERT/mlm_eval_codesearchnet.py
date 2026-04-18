#!/usr/bin/env python3
"""
Extended GraphCodeBERT MLM evaluator that works with CodeSearchNet corpus.

Usage:
    # Evaluate on CodeSearchNet Go test set
    python mlm-eval-codesearchnet.py --language go --split test --data-dir ./codesearchnet

    # Evaluate on specific number of samples
    python mlm-eval-codesearchnet.py --language go --max-samples 1000

    # Use custom model checkpoint
    python mlm-eval-codesearchnet.py --model-path ./models/checkpoint-1000 --language python
"""

import torch
import random
import math
import json
import argparse
import gzip
from pathlib import Path
from typing import List, Dict, Optional
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class CodeSearchNetMLMEvaluator:
    """MLM evaluator that works with CodeSearchNet corpus."""

    def __init__(self, model_name: str = "microsoft/graphcodebert-base", language: str = "go"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.mask_token_id = self.tokenizer.mask_token_id
        self.language = language

        print(f"Loaded model: {model_name}")
        print(f"Language: {language}")
        print(f"Mask token: {self.tokenizer.mask_token} (ID: {self.mask_token_id})")

        # Language-specific keywords and symbols
        self.language_keywords = self._get_language_keywords(language)
        self.symbols_to_skip = {";", ":", "(", ")", "{", "}", "[", "]", ",", "*", "+", "-",
                                "=", ">", "<", ".", "&", "|", "^", "!", "~", "/", "\\", "%"}

    def _get_language_keywords(self, language: str) -> set:
        """Get language-specific keywords for better masking."""
        keywords = {
            "go": {
                "package", "import", "func", "var", "const", "type", "struct", "interface",
                "int", "int8", "int16", "int32", "int64", "uint", "string", "bool", "float32", "float64",
                "return", "if", "else", "for", "range", "switch", "case", "default", "break", "continue",
                "go", "defer", "select", "chan", "map", "append", "make", "new", "len", "cap", "nil"
            },
            "python": {
                "def", "class", "import", "from", "return", "if", "else", "elif", "for", "while",
                "try", "except", "finally", "with", "as", "pass", "break", "continue", "yield",
                "lambda", "True", "False", "None", "and", "or", "not", "in", "is", "assert"
            },
            "java": {
                "public", "private", "protected", "static", "final", "class", "interface", "abstract",
                "int", "long", "short", "byte", "float", "double", "boolean", "char", "void", "String",
                "return", "if", "else", "for", "while", "do", "switch", "case", "default", "break",
                "continue", "try", "catch", "finally", "throw", "throws", "new", "this", "super", "null"
            },
            "javascript": {
                "function", "const", "let", "var", "return", "if", "else", "for", "while", "do",
                "switch", "case", "default", "break", "continue", "try", "catch", "finally", "throw",
                "async", "await", "class", "extends", "import", "export", "from", "new", "this", "null"
            },
            "php": {
                "function", "class", "public", "private", "protected", "static", "final", "abstract",
                "return", "if", "else", "elseif", "for", "foreach", "while", "do", "switch", "case",
                "default", "break", "continue", "try", "catch", "finally", "throw", "new", "this", "null"
            },
            "ruby": {
                "def", "class", "module", "return", "if", "else", "elsif", "unless", "for", "while",
                "until", "loop", "break", "next", "redo", "retry", "case", "when", "begin", "rescue",
                "ensure", "end", "do", "yield", "true", "false", "nil", "and", "or", "not"
            },
            "cpp": {
                "class", "struct", "enum", "namespace", "using", "template", "typename",
                "int", "double", "float", "char", "void", "bool",
                "public", "private", "protected",
                "return", "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
                "const", "static", "virtual", "inline", "new", "delete"
            },
            "erlang": {
                "module", "export", "import", "function", "record", "if", "case", "of", "end", "receive", "after", "try", "catch", "throw", "new", "this", "null"
            }
        }
        return keywords.get(language, set())

    def load_codesearchnet_data(self, data_dir: Path, split: str = "test",
                                max_samples: Optional[int] = None) -> List[Dict]:
        """Load CodeSearchNet data from jsonl.gz files.

        Args:
            data_dir: Path to CodeSearchNet data directory
            split: Data split to load (train/valid/test)
            max_samples: Maximum number of samples to load (None = all)

        Returns:
            List of code samples with metadata
        """
        # CodeSearchNet structure: <data_dir>/<language>/final/jsonl/<split>/*.jsonl.gz
        language_dir = data_dir / self.language / "final" / "jsonl" / split

        if not language_dir.exists():
            # Try alternative structure: <data_dir>/<language>/<split>.jsonl
            alt_path = data_dir / self.language / f"{split}.jsonl"
            if alt_path.exists():
                return self._load_jsonl(alt_path, max_samples)

            # Try another alternative: <data_dir>/<split>.jsonl
            alt_path2 = data_dir / f"{split}.jsonl"
            if alt_path2.exists():
                return self._load_jsonl(alt_path2, max_samples)

            raise FileNotFoundError(
                f"CodeSearchNet data not found at {language_dir} or alternative paths. "
                f"Expected structure: <data_dir>/{self.language}/final/jsonl/{split}/*.jsonl.gz "
                f"or <data_dir>/{self.language}/{split}.jsonl"
            )

        # Load all .jsonl.gz files in the directory
        samples = []
        for gz_file in sorted(language_dir.glob("*.jsonl.gz")):
            print(f"Loading {gz_file.name}...")
            with gzip.open(gz_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    if max_samples and len(samples) >= max_samples:
                        return samples
                    try:
                        data = json.loads(line.strip())
                        # Extract code (handle different field names)
                        code = data.get('code', data.get('original_string', data.get('func_code_string', '')))
                        if code.strip():
                            samples.append({
                                'code': code,
                                'repo': data.get('repo', ''),
                                'path': data.get('path', ''),
                                'func_name': data.get('func_name', ''),
                                'url': data.get('url', '')
                            })
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(samples)} samples from {language_dir}")
        return samples

    def _load_jsonl(self, file_path: Path, max_samples: Optional[int] = None) -> List[Dict]:
        """Load data from a plain .jsonl file."""
        samples = []
        print(f"Loading {file_path}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and len(samples) >= max_samples:
                    break
                try:
                    data = json.loads(line.strip())
                    code = data.get('code', data.get('original_string', data.get('func_code_string', '')))
                    if code.strip():
                        samples.append({
                            'code': code,
                            'repo': data.get('repo', ''),
                            'path': data.get('path', ''),
                            'func_name': data.get('func_name', ''),
                            'url': data.get('url', '')
                        })
                except json.JSONDecodeError:
                    continue

        print(f"Loaded {len(samples)} samples from {file_path}")
        return samples

    def create_masked_samples(self, code_samples: List[Dict], mask_ratio: float = 0.15,
                            max_length: int = 512) -> List[Dict]:
        """Create masked samples for MLM evaluation.

        Args:
            code_samples: List of code samples with 'code' field
            mask_ratio: Proportion of tokens to mask (default 0.15)
            max_length: Maximum sequence length (default 512)

        Returns:
            List of masked samples with targets
        """
        masked_samples = []

        for sample in tqdm(code_samples, desc="Creating masked samples"):
            code = sample['code']

            # Tokenize
            encoding = self.tokenizer(code, return_tensors="pt", truncation=True,
                                     max_length=max_length, padding=False)
            input_ids = encoding["input_ids"][0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            # Find candidate positions for masking
            # Skip special tokens, symbols, and very short tokens
            candidate_positions = [
                i for i, tok in enumerate(tokens)
                if tok not in {self.tokenizer.cls_token, self.tokenizer.sep_token,
                              self.tokenizer.pad_token, self.tokenizer.unk_token}
                and len(tok.replace("Ġ", "").replace("Â", "")) > 1  # Skip single-char tokens
                and tok.replace("Ġ", "").replace("Â", "") not in self.symbols_to_skip
            ]

            if not candidate_positions:
                continue

            # Mask tokens
            num_to_mask = max(1, int(len(candidate_positions) * mask_ratio))
            positions_to_mask = random.sample(candidate_positions,
                                            min(num_to_mask, len(candidate_positions)))

            masked_ids = input_ids.clone()
            targets = []
            for pos in positions_to_mask:
                targets.append({
                    "position": pos,
                    "original_token": tokens[pos],
                    "original_id": int(input_ids[pos])
                })
                masked_ids[pos] = self.mask_token_id

            masked_samples.append({
                "original_code": code,
                "masked_ids": masked_ids.tolist(),
                "targets": targets,
                "attention_mask": encoding["attention_mask"][0].tolist(),
                "metadata": {k: v for k, v in sample.items() if k != 'code'}
            })

        return masked_samples

    def evaluate(self, masked_samples: List[Dict], top_k: int = 5,
                batch_size: int = 8, verbose: bool = False) -> Dict:
        """Evaluate MLM performance on masked samples.

        Args:
            masked_samples: List of masked samples from create_masked_samples()
            top_k: Top-k accuracy to compute (default 5)
            batch_size: Batch size for evaluation (default 8)
            verbose: Print detailed results per sample

        Returns:
            Dictionary with accuracy, perplexity, and other metrics
        """
        total_predictions = 0
        top1_correct = 0
        topk_correct = 0
        total_loss = 0.0
        loss_fn = CrossEntropyLoss(reduction='sum')

        # Process in batches
        for i in tqdm(range(0, len(masked_samples), batch_size), desc="Evaluating"):
            batch = masked_samples[i:i+batch_size]

            # Pad batch to same length
            max_len = max(len(s["masked_ids"]) for s in batch)

            batch_input_ids = []
            batch_attention_masks = []
            batch_targets = []

            for sample in batch:
                # Pad to max_len
                padded_ids = sample["masked_ids"] + [self.tokenizer.pad_token_id] * (max_len - len(sample["masked_ids"]))
                padded_mask = sample["attention_mask"] + [0] * (max_len - len(sample["attention_mask"]))

                batch_input_ids.append(padded_ids)
                batch_attention_masks.append(padded_mask)
                batch_targets.append(sample["targets"])

            input_ids = torch.tensor(batch_input_ids).to(self.device)
            attention_mask = torch.tensor(batch_attention_masks).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Evaluate each sample in batch
            for batch_idx, targets in enumerate(batch_targets):
                for target in targets:
                    pos = target["position"]
                    target_id = target["original_id"]
                    target_token = target["original_token"]

                    # Get top-k predictions
                    pred_probs = probs[batch_idx, pos]
                    top_probs, top_ids = torch.topk(pred_probs, top_k)
                    top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids.cpu().tolist())

                    # Check correctness
                    total_predictions += 1
                    if top_tokens[0] == target_token:
                        top1_correct += 1
                    if target_token in top_tokens:
                        topk_correct += 1

                    # Compute loss
                    total_loss += loss_fn(
                        logits[batch_idx, pos:pos+1],
                        torch.tensor([target_id]).to(self.device)
                    ).item()

                    if verbose and i < 5:  # Show first 5 samples
                        print(f"\nExpected: {target_token}")
                        for j, (tok, prob) in enumerate(zip(top_tokens, top_probs)):
                            marker = "✅" if tok == target_token else f"{j+1}."
                            print(f"  {marker} {tok} ({prob:.4f})")

        # Compute metrics
        avg_loss = total_loss / total_predictions if total_predictions > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 100))  # prevent overflow

        return {
            "top1_accuracy": top1_correct / total_predictions if total_predictions > 0 else 0.0,
            f"top{top_k}_accuracy": topk_correct / total_predictions if total_predictions > 0 else 0.0,
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_predictions": total_predictions,
            "total_samples": len(masked_samples)
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GraphCodeBERT MLM on CodeSearchNet")
    parser.add_argument("--model-path", type=str, default="microsoft/graphcodebert-base",
                       help="Path to model checkpoint or HuggingFace model name")
    parser.add_argument("--language", type=str, default="go",
                       choices=["go", "python", "java", "javascript", "php", "ruby", "cpp", "erlang"],
                       help="Programming language to evaluate")
    parser.add_argument("--data-dir", type=str, default="./codesearchnet",
                       help="Path to CodeSearchNet data directory")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "valid", "test"],
                       help="Data split to use")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                       help="Proportion of tokens to mask (default: 0.15)")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length (default: 512)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Top-k accuracy to compute (default: 5)")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for evaluation (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed predictions")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("="*70)
    print("GraphCodeBERT MLM Evaluation on CodeSearchNet")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Language: {args.language}")
    print(f"Data split: {args.split}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Max length: {args.max_length}")
    print(f"Top-k: {args.top_k}")
    print(f"Batch size: {args.batch_size}")
    print(f"Random seed: {args.seed}")
    print("="*70 + "\n")

    # Initialize evaluator
    evaluator = CodeSearchNetMLMEvaluator(
        model_name=args.model_path,
        language=args.language
    )

    # Load data
    data_dir = Path(args.data_dir)
    try:
        code_samples = evaluator.load_codesearchnet_data(
            data_dir=data_dir,
            split=args.split,
            max_samples=args.max_samples
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure CodeSearchNet data is available at the specified path.")
        print("You can download it from: https://github.com/github/CodeSearchNet")
        return

    if not code_samples:
        print("No code samples loaded. Please check the data path and format.")
        return

    # Create masked samples
    masked_samples = evaluator.create_masked_samples(
        code_samples,
        mask_ratio=args.mask_ratio,
        max_length=args.max_length
    )

    if not masked_samples:
        print("No valid masked samples created. Check the input data.")
        return

    print(f"\nCreated {len(masked_samples)} masked samples from {len(code_samples)} code samples")
    print(f"Starting evaluation...\n")

    # Evaluate
    metrics = evaluator.evaluate(
        masked_samples,
        top_k=args.top_k,
        batch_size=args.batch_size,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Language: {args.language}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Total masked predictions: {metrics['total_predictions']}")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%} ({metrics['top1_accuracy']*100:.2f}%)")
    print(f"Top-{args.top_k} Accuracy: {metrics[f'top{args.top_k}_accuracy']:.2%} ({metrics[f'top{args.top_k}_accuracy']*100:.2f}%)")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Average Loss: {metrics['avg_loss']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
