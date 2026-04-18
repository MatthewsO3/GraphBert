import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Configuration
base_dir = "/home/mczap/GraphBert/GraphBERT/models/final_train/ODM_v2/checkpoints"
# Mapping the filenames to the labels you want on the chart
file_to_lang = {
    "evaluation_results_java.json": "Java",
    "evaluation_results_javascript.json": "Javascript",
    "evaluation_results_python.json": "Python",
    "evaluation_results_val.json": "Combined_Val" # This likely holds C++ and Erlang
}

metrics = {
    "top1_accuracy": "Top-1 Accuracy",
    "top5_accuracy": "Top-5 Accuracy",
    "perplexity": "Perplexity"
}

def collect_data(path):
    all_data = []
    epoch_dirs = sorted(glob.glob(os.path.join(path, "epoch_*")))
    
    for ed in epoch_dirs:
        epoch_num = int(os.path.basename(ed).split("_")[1])
        
        # Check for each specific language file in the epoch folder
        for json_name, lang_label in file_to_lang.items():
            jf_path = os.path.join(ed, json_name)
            
            if os.path.exists(jf_path):
                with open(jf_path, 'r') as f:
                    content = json.load(f)
                    
                    # If it's the 'val' file, it has a 'languages' nested dict
                    if "languages" in content:
                        for l_name, l_metrics in content["languages"].items():
                            all_data.append({
                                "Epoch": epoch_num,
                                "Language": l_name, # This will pick up C++ and Erlang
                                **{k: l_metrics.get(k) for k in metrics.keys()}
                            })
                    # If it's a specific lang file, the metrics might be at the top level 
                    # or under a "languages" key with one entry
                    elif "top1_accuracy" in content:
                        all_data.append({
                            "Epoch": epoch_num,
                            "Language": lang_label,
                            **{k: content.get(k) for k in metrics.keys()}
                        })
    return pd.DataFrame(all_data)

# Main Execution
df = collect_data(base_dir)

if df.empty:
    print("No data collected. Check if the file paths are correct.")
else:
    for metric_key, metric_name in metrics.items():
        plt.figure(figsize=(10, 6))
        # Ensure we only plot the 5 languages you asked for
        target_langs = ["C++", "Erlang", "Java", "Python", "Javascript"]
        
        for lang in target_langs:
            lang_df = df[df["Language"] == lang].sort_values("Epoch")
            if not lang_df.empty:
                plt.plot(lang_df["Epoch"], lang_df[metric_key], marker='o', label=lang, linewidth=2)
        
        plt.title(f"{metric_name} Across Epochs", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.savefig(f"plot_{metric_key}.png", dpi=300)
        print(f"Generated plot_{metric_key}.png")