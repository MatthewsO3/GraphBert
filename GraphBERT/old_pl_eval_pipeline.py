import subprocess

commands = [
    [
        "python", "python_eval.py",
        "--data_file", "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "microsoft/graphcodebert-base",
        "--mask_ratio", "0.15",
        "--top_k", "10",
        "--max_examples", "2500",
        "--max_seq_length", "512",
        "--output_file", "results/testing_new_perplexity/base/evaluation_results_python_2500.json"
    ],
    [
        "python", "python_eval.py",
        "--data_file", "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/20k_20k_mixed_retokenized/best_model",
        "--mask_ratio", "0.15",
        "--top_k", "10",
        "--max_examples", "2500",
        "--max_seq_length", "512",
        "--output_file", "results/testing_new_perplexity/20k-20k_retokenized/evaluation_results_python_2500.json"
    ],
    [
        "python", "java_eval.py",
        "--data_file", "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/20k_20k_mixed_retokenized/best_model",
        "--mask_ratio", "0.15",
        "--top_k", "10",
        "--max_examples", "2500",
        "--max_seq_length", "512",
        "--output_file", "results/testing_new_perplexity/20k-20k_retokenized/evaluation_results_java_2500.json"
    ],
    [
        "python", "java_eval.py",
        "--data_file", "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "microsoft/graphcodebert-base",
        "--mask_ratio", "0.15",
        "--top_k", "10",
        "--max_examples", "2500",
        "--max_seq_length", "512",
        "--output_file", "results/testing_new_perplexity/base/evaluation_results_java_2500.json"
    ],
    [
        "python", "javascript_eval.py",
        "--data_file", "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/20k_20k_mixed_retokenized/best_model",
        "--mask_ratio", "0.15",
        "--top_k", "10",
        "--max_examples", "2500",
        "--max_seq_length", "512",
        "--output_file", "results/testing_new_perplexity/20k-20k_retokenized/evaluation_results_javascript_2500.json"
    ],
    [
        "python", "javascript_eval.py",
        "--data_file", "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "microsoft/graphcodebert-base",
        "--mask_ratio", "0.15",
        "--top_k", "10",
        "--max_examples", "2500",
        "--max_seq_length", "512",
        "--output_file", "results/testing_new_perplexity/base/evaluation_results_javascript_2500.json"
    ],
]

for i, cmd in enumerate(commands, 1):
    print(f"Running command {i}/{len(commands)}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command {i} failed with error: {e}")
        break

print("All done!")