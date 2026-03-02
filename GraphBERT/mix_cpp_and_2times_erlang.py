import json
import random


def merge_balanced_cpp_erlang(
    cpp_file: str,
    erlang_file: str,
    output_file: str,
    max_cpp: int = 20000,
    max_erlang: int = 10000,
    seed: int = 42,
):
    random.seed(seed)

    # Load C++ samples
    with open(cpp_file, 'r', encoding='utf-8') as f:
        cpp_samples = [json.loads(line) for line in f if line.strip()]

    if len(cpp_samples) > max_cpp:
        cpp_samples = random.sample(cpp_samples, max_cpp)
        print(f"Sampled {max_cpp} from {cpp_file} (had {len(cpp_samples) + (len(cpp_samples) - max_cpp)} total)")
    else:
        print(f"Loaded {len(cpp_samples)} C++ samples from {cpp_file}")

    # Load Erlang samples
    with open(erlang_file, 'r', encoding='utf-8') as f:
        erlang_samples = [json.loads(line) for line in f if line.strip()]

    if len(erlang_samples) > max_erlang:
        erlang_samples = random.sample(erlang_samples, max_erlang)
        print(f"Sampled {max_erlang} from {erlang_file} (had {len(erlang_samples) + (len(erlang_samples) - max_erlang)} total)")
    else:
        print(f"Loaded {len(erlang_samples)} Erlang samples from {erlang_file}")

    # Duplicate Erlang samples
    erlang_doubled = erlang_samples + erlang_samples
    print(f"Duplicated Erlang samples: {len(erlang_samples)} → {len(erlang_doubled)}")

    # Merge and shuffle
    all_samples = cpp_samples + erlang_doubled
    random.seed(seed)
    random.shuffle(all_samples)

    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nTotal: {len(all_samples)} samples saved to {output_file}")
    print(f"  C++:    {len(cpp_samples):>6} samples")
    print(f"  Erlang: {len(erlang_doubled):>6} samples (2 × {len(erlang_samples)})")


merge_balanced_cpp_erlang(
    cpp_file="/home/mczap/GraphBert/GraphBERT/data/train.jsonl",
    erlang_file="/home/mczap/erlangbert/erlang_corpus_scraper/output/graphcodebert_data/train.jsonl",
    output_file="data/mixed_cpp_2x_erlang_train.jsonl",
    max_cpp=20000,
    max_erlang=10000,
)