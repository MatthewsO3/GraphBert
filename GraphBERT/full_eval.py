"""
pipeline_eval_loop.py  —  Runs evaluation jobs sequentially with loops.
Evaluates mixed_eval, python_eval, java_eval, and javascript_eval for all 6 epochs.
"""

import subprocess
import sys
import time

# Configuration
BASE_MODEL_DIR = "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM_v2/checkpoints"
BASE_DATA_DIR = "/home/mczap/GraphBert/GraphBERT/data"
EPOCHS = range(1, 7)  # epochs 1-6

# Language configurations for evaluations
LANGUAGES = {
    "mixed": {
        "script": "mixed_eval.py",
        "args": {
            "--cpp_val": f"{BASE_DATA_DIR}/cpp/test.jsonl",
            "--erlang_val": f"{BASE_DATA_DIR}/erlang/test.jsonl",
            "--mask_ratio": "0.15",
            "--top_k": "10",
        }
    },
    "python": {
        "script": "python_eval.py",
        "args": {
            "--data_file": f"{BASE_DATA_DIR}/python/test.jsonl",
            "--mask_ratio": "0.15",
            "--top_k": "10",
            "--max_examples": "500",
            "--max_seq_length": "512",
        }
    },
    "java": {
        "script": "java_eval.py",
        "args": {
            "--data_file": f"{BASE_DATA_DIR}/java/test.jsonl",
            "--mask_ratio": "0.15",
            "--top_k": "10",
            "--max_examples": "500",
            "--max_seq_length": "512",
        }
    },
    "javascript": {
        "script": "javascript_eval.py",
        "args": {
            "--data_file": f"{BASE_DATA_DIR}/javascript/test.jsonl",
            "--mask_ratio": "0.15",
            "--top_k": "10",
            "--max_examples": "500",
            "--max_seq_length": "512",
        }
    },
}


def build_command(language, epoch):
    """Build a command list for a specific language and epoch."""
    config = LANGUAGES[language]
    epoch_str = f"epoch_{epoch:03d}"  # e.g., "epoch_001"
    checkpoint_path = f"{BASE_MODEL_DIR}/{epoch_str}"
    output_path = f"{checkpoint_path}"
    
    cmd = ["python", config["script"]]
    
    # Add common arguments
    for key, value in config["args"].items():
        cmd.append(key)
        cmd.append(value)
    
    # Add model and output paths (naming differs between mixed and single-language evals)
    if language == "mixed":
        cmd.append("--model")
        cmd.append(checkpoint_path)
        cmd.append("--output_dir")
        cmd.append(output_path)
    else:
        cmd.append("--model_checkpoint")
        cmd.append(checkpoint_path)
    
    
    
    # Add output file for single-language evals
    if language != "mixed":
        output_file = f"{output_path}/evaluation_results_{language}.json"
        cmd.append("--output_file")
        cmd.append(output_file)
    
    return cmd


def generate_jobs():
    """Generate all evaluation jobs."""
    jobs = []
    
    # Order: mixed for all epochs, then python, java, javascript
    for language in ["mixed", "python", "java", "javascript"]:
        for epoch in EPOCHS:
            cmd = build_command(language, epoch)
            jobs.append((language, epoch, cmd))
    
    return jobs


if __name__ == "__main__":
    jobs = generate_jobs()
    
    print(f"\nTotal jobs to run: {len(jobs)}\n")
    
    for idx, (language, epoch, cmd) in enumerate(jobs, 1):
        print(f"\n{'='*70}")
        print(f"Job {idx}/{len(jobs)} — {language.upper()} Evaluation, Epoch {epoch}")
        print(f"{'='*70}")
        print(" ".join(cmd))
        print()

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n❌ Job {idx} failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

        print(f"\n✓ Job {idx}/{len(jobs)} finished successfully.")
        print("Waiting 10 seconds for GPU/Driver cleanup...")
        time.sleep(10)

    print("\n" + "="*70)
    print("✓ All evaluation jobs completed successfully!")
    print("="*70)