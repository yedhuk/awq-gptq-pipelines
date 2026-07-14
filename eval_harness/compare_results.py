import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Configuration
# ==========================================
# Update these paths if your files are named differently
FILES = {
    "FP16 Base": "results/llama3_2_3b_fp16_results.json",
    "AWQ 4-bit": "results/llama3_2_3b_awq_results.json",
    "GPTQ 4-bit": "results/llama3_2_3b_gptq_results.json"
}

# Map the display name to the exact task and metric key in the JSON
CORE_METRICS = {
    "MMLU (Acc)": ("mmlu", "acc,none"),
    "GSM8K (Strict Math)": ("gsm8k", "exact_match,strict-match"),
    "HumanEval (Code)": ("humaneval", "pass@1,create_test"),
    "MBPP (Code)": ("mbpp", "pass_at_1,none")
}

MMLU_CATEGORIES = {
    "STEM": ("mmlu_stem", "acc,none"),
    "Social Sciences": ("mmlu_social_sciences", "acc,none"),
    "Humanities": ("mmlu_humanities", "acc,none"),
    "Other": ("mmlu_other", "acc,none")
}

PERPLEXITY_METRIC = {
    "Wikitext Perplexity": ("wikitext", "word_perplexity,none")
}

# ==========================================
# 2. Data Extraction Function
# ==========================================
def extract_data(file_dict, metric_map, multiply_by_100=True):
    data = []
    for model_name, file_path in file_dict.items():
        if not os.path.exists(file_path):
            print(f"Warning: Could not find {file_path}")
            continue
            
        with open(file_path, 'r') as f:
            results = json.load(f).get("results", {})
            
        row = {"Model": model_name}
        for display_name, (task, metric_key) in metric_map.items():
            # Safely navigate the JSON tree
            value = results.get(task, {}).get(metric_key, 0)
            if multiply_by_100:
                value = value * 100 # Convert decimals to percentages
            row[display_name] = value
            
        data.append(row)
    return pd.DataFrame(data)

# ==========================================
# 3. Plotting Functions
# ==========================================
def plot_grouped_bar(df, title, ylabel, filename, is_perplexity=False):
    if df.empty:
        return
        
    df.set_index("Model", inplace=True)
    
    # Setup the plot
    ax = df.plot(kind="bar", figsize=(10, 6), width=0.8, colormap="viridis", edgecolor="black")
    
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=0, fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add text labels on top of the bars
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            label = f"{val:.1f}" if is_perplexity else f"{val:.1f}%"
            ax.annotate(label, (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 3), 
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig(f"plots//{filename}", dpi=300)
    print(f"Saved plot: {filename}")
    plt.close()

# ==========================================
# 4. Execution
# ==========================================
if __name__ == "__main__":
    print("Extracting data...")
    
    # Extract and plot Core Benchmarks
    df_core = extract_data(FILES, CORE_METRICS, multiply_by_100=True)
    plot_grouped_bar(df_core, "Core Benchmark Accuracy Comparison", "Accuracy (%)", "plot_core_benchmarks.png")

    # Extract and plot MMLU Sub-categories
    df_mmlu = extract_data(FILES, MMLU_CATEGORIES, multiply_by_100=True)
    plot_grouped_bar(df_mmlu, "MMLU Macro-Category Breakdown", "Accuracy (%)", "plot_mmlu_breakdown.png")

    # Extract and plot Perplexity (Do not multiply by 100, lower is better)
    df_perp = extract_data(FILES, PERPLEXITY_METRIC, multiply_by_100=False)
    plot_grouped_bar(df_perp, "Wikitext Word Perplexity (Lower is Better)", "Perplexity Score", "plot_perplexity.png", is_perplexity=True)

    print("Done! Check your directory for the PNG files.")