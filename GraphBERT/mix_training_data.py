import json
import random

def merge_balanced(files: list[str], output_file: str, 
                   max_per_language: int | None = None, seed: int = 42):
    all_samples = []
    
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        if max_per_language and len(samples) > max_per_language:
            random.seed(seed)
            samples = random.sample(samples, max_per_language)
            print(f"Sampled {max_per_language} from {file} (had {len(samples)})")
        else:
            print(f"Loaded {len(samples)} samples from {file}")
        
        all_samples.extend(samples)
    
    random.seed(seed)
    random.shuffle(all_samples)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\nTotal: {len(all_samples)} samples saved to {output_file}")

merge_balanced(
    files=["/home/mczap/GraphBert/GraphBERT/data/train.jsonl", "/home/mczap/erlangbert/erlang_corpus_scraper/output/graphcodebert_data/train.jsonl"],
    output_file="data/mixed_train.jsonl",
    max_per_language=20000  # Set to None to use all
)