"""
pipeline.py  —  Runs training jobs sequentially.
Edit the JOBS list to add/remove/reorder runs.
"""

import subprocess
import sys
import time


JOBS_DONE = [
    # ── Job 1: baseline pre-training ─────────────────────────────────────────
    
        #"python", "retokenize_erlang.py",
    
    #[
        #"python", "odm_train.py",
        #"--cpp_file",                "/home/mczap/GraphBert/GraphBERT/data/mixed_cpp_2x_erlang_train.jsonl",
        #"--erl_file",                "/home/mczap/GraphBert/GraphBERT/data/erlang/retokenized_train.jsonl",
       # "--output_dir",              "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM",
      #  "--batch_size",              "32",
     #   "--epochs",                  "6",
    #"--learning_rate",           "2e-5",
        #"--max_length",              "256",
        #"--warmup_steps",            "2000",
       # "--mlm_probability",         "0.15",
       # "--mlm_probability",         "0.15",
      #  "--validation_split",        "0.05",
     #   "--weight_decay",            "0.01",
    #    "--early_stopping_patience", "3",
   #     "--odm_alpha",               "0.65",
  #      "--odm_warmup_steps",        "2000",
 #       "--log_interval",            "100",
#],
   
        # ── Eval ODM - 1. ────────────────
    [
        "python", "mixed_eval.py",
        "--cpp_val",          "/home/mczap/GraphBert/GraphBERT/data/cpp/test.jsonl",
        "--erlang_val",       "/home/mczap/GraphBert/GraphBERT/data/erlang/test.jsonl",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--model",            "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001",
        "--output_dir",       "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001",
    ],
     # ── Eval ODM - 2. ────────────────
    [
        "python", "mixed_eval.py",
        "--cpp_val",          "/home/mczap/GraphBert/GraphBERT/data/cpp/test.jsonl",
        "--erlang_val",       "/home/mczap/GraphBert/GraphBERT/data/erlang/test.jsonl",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--model",            "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002",
        "--output_dir",       "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002",
    ],
     # ── Eval ODM - 3. ────────────────
    [
        "python", "mixed_eval.py",
        "--cpp_val",          "/home/mczap/GraphBert/GraphBERT/data/cpp/test.jsonl",
        "--erlang_val",       "/home/mczap/GraphBert/GraphBERT/data/erlang/test.jsonl",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--model",            "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003",
        "--output_dir",       "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003",
    ],
     # ── Eval ODM - 4. ────────────────
    [
        "python", "mixed_eval.py",
        "--cpp_val",          "/home/mczap/GraphBert/GraphBERT/data/cpp/test.jsonl",
        "--erlang_val",       "/home/mczap/GraphBert/GraphBERT/data/erlang/test.jsonl",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--model",            "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004",
        "--output_dir",       "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004",
    ],
     # ── Eval ODM - 5. ────────────────
    [
        "python", "mixed_eval.py",
        "--cpp_val",          "/home/mczap/GraphBert/GraphBERT/data/cpp/test.jsonl",
        "--erlang_val",       "/home/mczap/GraphBert/GraphBERT/data/erlang/test.jsonl",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--model",            "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005",
        "--output_dir",       "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005",
    ],
     # ── Eval ODM - 6. ────────────────
    [
        "python", "mixed_eval.py",
        "--cpp_val",          "/home/mczap/GraphBert/GraphBERT/data/cpp/test.jsonl",
        "--erlang_val",       "/home/mczap/GraphBert/GraphBERT/data/erlang/test.jsonl",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--model",            "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006",
        "--output_dir",       "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006",
    ],
    # ── Eval ODM - Python 1. ────────────────
    [
        "python", "python_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001/evaluation_results_python.json",
    ],
     # ── Eval ODM - Python 2. ────────────────
    [
        "python", "python_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002/evaluation_results_python.json",
    ],
     # ── Eval ODM - Python 3. ────────────────
    [
        "python", "python_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003/evaluation_results_python.json",
    ],
     # ── Eval ODM - Python 4. ────────────────
    [
        "python", "python_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004/evaluation_results_python.json",
    ],
     # ── Eval ODM - Python 5. ────────────────
    [
        "python", "python_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005/evaluation_results_python.json",
    ],
     # ── Eval ODM - Python 6. ────────────────
    [
        "python", "python_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/python/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006/evaluation_results_python.json",
    ],

    # ── Eval ODM - Java 1. ────────────────
    [
        "python", "java_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001/evaluation_results_java.json",
    ],
     # ── Eval ODM - Java 2. ────────────────
    [
        "python", "java_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002/evaluation_results_java.json",
    ],
     # ── Eval ODM - Java 3. ────────────────
    [
        "python", "java_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003/evaluation_results_java.json",
    ],
     # ── Eval ODM - Java 4. ────────────────
    [
        "python", "java_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004/evaluation_results_java.json",
    ],
     # ── Eval ODM - Java 5. ────────────────
    [
        "python", "java_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005/evaluation_results_java.json",
    ],
     # ── Eval ODM - Java 6. ────────────────
    [
        "python", "java_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/java/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006/evaluation_results_java.json",
    ],

     # ── Eval ODM - Javascript 1. ────────────────
    [
        "python", "javascript_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_001/evaluation_results_javascript.json",
    ],
     # ── Eval ODM - Javascript 2. ────────────────
    [
        "python", "javascript_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_002/evaluation_results_javascript.json",
    ],
     # ── Eval ODM - Javascript 3. ────────────────
    [
        "python", "javascript_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_003/evaluation_results_javascript.json",
    ],
     # ── Eval ODM - Javascript 4. ────────────────
    [
        "python", "javascript_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_004/evaluation_results_javascript.json",
    ],
     # ── Eval ODM - Javascript 5. ────────────────
    [
        "python", "javascript_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_005/evaluation_results_javascript.json",
    ],
     # ── Eval ODM - Javascript 6. ────────────────
    [
        "python", "javascript_eval.py",
        "--data_file",        "/home/mczap/GraphBert/GraphBERT/data/javascript/test.jsonl",
        "--model_checkpoint", "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006",
        "--mask_ratio",       "0.15",
        "--top_k",            "10",
        "--max_examples",     "500",
        "--max_seq_length",   "512",
        "--output_file",      "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM/checkpoints/epoch_006/evaluation_results_javascript.json",
    ],
    

]
JOBS=[
    # ── Job 2: 50%-50% ────────────────
    [
        "python", "train.py",
        "--data_file",               "/home/mczap/GraphBert/GraphBERT/data/50_50mix/train.jsonl",
        "--output_dir",              "/home/mczap/GraphBert/GraphBERT/models/final_train/50_50",
        "--batch_size",              "32",
        "--epochs",                  "6",
        "--learning_rate",           "2e-5",
        "--max_length",              "256",
        "--warmup_steps",            "2000",
        "--mlm_probability",         "0.15",
        "--validation_split",        "0.05",
        "--weight_decay",            "0.01",
        "--early_stopping_patience", "3",
    ],
    [
        "python", "mix_cpp_and_2times_erlang.py",
    ],
    # ── Job 3: 50%-2*25% ────────────────
    [
        "python", "train.py",
        "--data_file",               "/home/mczap/GraphBert/GraphBERT/data/50_2x25mix/train.jsonl",
        "--output_dir",              "/home/mczap/GraphBert/GraphBERT/models/final_train/50_2x25",
        "--batch_size",              "32",
        "--epochs",                  "6",
        "--learning_rate",           "2e-5",
        "--max_length",              "256",
        "--warmup_steps",            "2000",
        "--mlm_probability",         "0.15",
        "--validation_split",        "0.05",
        "--weight_decay",            "0.01",
        "--early_stopping_patience", "3",
    ],]


if __name__ == "__main__":
    for i, cmd in enumerate(JOBS, 1):
        print(f"\n{'='*60}")
        print(f"Starting job {i}/{len(JOBS)}")
        print(" ".join(cmd))
        print(f"{'='*60}\n")

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nJob {i} failed with exit code {result.returncode}. Stopping.")
            sys.exit(result.returncode)

        print(f"\nJob {i}/{len(JOBS)} finished successfully.")
        print("Waiting 10 seconds for GPU/Driver cleanup...")
        time.sleep(100)

    print("\nAll jobs completed.")