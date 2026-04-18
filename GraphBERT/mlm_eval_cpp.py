import torch
import random
import math
from typing import List, Dict
from transformers import RobertaTokenizer, RobertaForMaskedLM
from torch.nn import CrossEntropyLoss

class GraphCodeBERTTester:
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # GraphCodeBERT is RoBERTa-based, so RobertaTokenizer + RobertaForMaskedLM works
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.mask_token_id = self.tokenizer.mask_token_id
        print(f"Mask token: {self.tokenizer.mask_token} (ID: {self.mask_token_id})")

        # C++ keywords for masking
        self.cpp_keywords = {
            "class", "struct", "enum", "namespace", "using", "template", "typename",
            "int", "double", "float", "char", "void", "bool",
            "public", "private", "protected",
            "return", "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
            "const", "static", "virtual", "inline", "new", "delete"
        }
        # Symbols to skip
        self.cpp_symbols = {";", ":", "(", ")", "{", "}", "[", "]", ",", "*", "+", "-", "=", ">", "<"}

    def create_masked_samples(self, code_samples: List[str], mask_ratio: float = 0.15) -> List[Dict]:
        masked_samples = []
        for code in code_samples:
            encoding = self.tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoding["input_ids"][0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            candidate_positions = [
                i for i, tok in enumerate(tokens)
                if tok not in {self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token}
                and (tok.replace("Ġ", "") in self.cpp_keywords or tok.replace("Ġ", "").isalpha())
                and tok.replace("Ġ", "") not in self.cpp_symbols
            ]

            if not candidate_positions:
                continue

            num_to_mask = max(1, int(len(candidate_positions) * mask_ratio))
            positions_to_mask = random.sample(candidate_positions, min(num_to_mask, len(candidate_positions)))

            masked_ids = input_ids.clone()
            targets = []
            for pos in positions_to_mask:
                targets.append({
                    "position": pos,
                    "original_token": tokens[pos],
                    "original_id": int(input_ids[pos])
                })
                masked_ids[pos] = self.mask_token_id

            masked_tokens = self.tokenizer.convert_ids_to_tokens(masked_ids)
            masked_code = self.tokenizer.convert_tokens_to_string(masked_tokens)

            masked_samples.append({
                "original_code": code,
                "masked_code": masked_code,
                "masked_ids": masked_ids.tolist(),
                "targets": targets,
                "attention_mask": encoding["attention_mask"][0].tolist()
            })
        return masked_samples

    def predict(self, sample: Dict, top_k: int = 5):
        sample["targets"].sort(key=lambda x: x["position"])

        input_ids = torch.tensor([sample["masked_ids"]]).to(self.device)
        attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        results = []
        for target in sample["targets"]:
            pos = target["position"]
            pred_probs = probs[0, pos]
            top_probs, top_ids = torch.topk(pred_probs, top_k)
            top_tokens = self.tokenizer.convert_ids_to_tokens(top_ids.cpu().tolist())
            correct = target["original_token"] in top_tokens
            results.append({
                "expected": target["original_token"],
                "predicted": list(zip(top_tokens, top_probs.cpu().tolist())),
                "is_correct": correct,
                "prob": pred_probs[target["original_id"]].item()
            })
        return results

    def evaluate(self, masked_samples: List[Dict], top_k: int = 5):
        total_predictions = 0
        top1_correct = 0
        topk_correct = 0
        total_loss = 0.0
        loss_fn = CrossEntropyLoss(reduction='sum')

        for sample in masked_samples:
            predictions = self.predict(sample, top_k=top_k)
            input_ids = torch.tensor([sample["masked_ids"]]).to(self.device)
            attention_mask = torch.tensor([sample["attention_mask"]]).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

            for i, pred in enumerate(predictions):
                total_predictions += 1
                if pred["is_correct"]:
                    topk_correct += 1
                if pred["predicted"][0][0] == pred["expected"]:
                    top1_correct += 1

                # Per-token loss for perplexity
                pos = sample["targets"][i]["position"]
                target_id = sample["targets"][i]["original_id"]
                total_loss += loss_fn(logits[0, pos:pos+1], torch.tensor([target_id]).to(self.device)).item()

        avg_loss = total_loss / total_predictions if total_predictions > 0 else float('inf')
        perplexity = math.exp(min(avg_loss, 100))  # prevent overflow

        return {
            "top1_accuracy": top1_correct / total_predictions if total_predictions > 0 else 0.0,
            f"top{top_k}_accuracy": topk_correct / total_predictions if total_predictions > 0 else 0.0,
            "perplexity": perplexity,
            "total_predictions": total_predictions
        }


# Example C++ snippets
CPP_CODE = [
    "int factorial(int n) { if (n <= 1) return 1; return n * factorial(n - 1); }",
    "class Rectangle { public: int w, h; Rectangle(int w, int h): w(w), h(h) {} int area() { return w * h; } };",
    "void bubbleSort(int arr[], int n) { for (int i=0; i<n; i++) { for (int j=0; j<n-1; j++) { if (arr[j] > arr[j+1]) swap(arr[j], arr[j+1]); } } }",
    "class Rectangle { private: double width; double height; public: Rectangle(double w, double h) : width(w), height(h) {} double area() const { return width * height; } };",
    "template<typename T> class Vector { private: T* data; int size; public: Vector() : data(nullptr), size(0) {} void push(T item) { size++; } };",
    "class BankAccount { private: double balance; public: BankAccount() : balance(0.0) {} void deposit(double amount) { if (amount > 0) balance += amount; } double getBalance() const { return balance; } };",
    "struct Node { int data; Node* next; Node(int val) : data(val), next(nullptr) {} };"
]


def main():
    tester = GraphCodeBERTTester()
    masked_samples = tester.create_masked_samples(CPP_CODE, mask_ratio=0.2)

    for i, sample in enumerate(masked_samples):
        print(f"\n=== Sample {i+1} ===")
        print(f"Original code:\n{sample['original_code']}")
        print(f"Masked code:\n{sample['masked_code']}")
        predictions = tester.predict(sample, top_k=5)
        for r in predictions:
            print(f"\nExpected: {r['expected']}")
            for token, prob in r['predicted']:
                marker = "✅" if token == r['expected'] else "  "
                print(f"  {marker} {token} ({prob:.4f})")

    # Evaluation metrics
    metrics = tester.evaluate(masked_samples, top_k=5)
    print("\n=== Evaluation Metrics ===")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2%}")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2%}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Total masked predictions: {metrics['total_predictions']}")


if __name__ == "__main__":
    main()
