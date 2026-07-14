import os
import json
import torch
import argparse
import subprocess
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import handle_non_serializable

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

def run_benchmarks(model_path_or_id: str, output_path: str, is_quantized: bool = False):
    print(f"Running benchmarks for model: {model_path_or_id}")
    print(f"Initializing {model_path_or_id} for evaluation...")
    
    try:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", "/workspace"],
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except Exception:
        pass
    # 1. Setup Model Arguments
    # The HFLM class accepts either an HF ID or a local path
    model_kwargs = {
        "pretrained": model_path_or_id,
        "device": "cuda:0",
        "batch_size": 1
    }

    # If the model is a standard FP16 base model, specify bfloat16/float16
    # If it is AWQ/GPTQ quantized, the weights are 4-bit, but compute is usually done in fp16
    if not is_quantized:
        model_kwargs["dtype"] = torch.bfloat16
    
    # Initialize the model via the harness wrapper
    lm = HFLM(**model_kwargs)

    # 2. Define the exact benchmark tasks
    # tasks = ["wikitext", "mmlu", "gsm8k", "humaneval", "mbpp"]

    task_configs = {
        "wikitext": 0,
        "mmlu": 5,      # Show 5 examples before asking the MMLU question
        "gsm8k": 5,     # Show 5 examples before asking the GSM8K question
        "humaneval": 0, # Code generation is typically evaluated zero-shot
        "mbpp": 3       # Show 3 examples to stop it from being conversational
    }

    # print(f"Starting evaluation on tasks: {tasks}")
    
    # # 3. Execute the evaluation
    # results = lm_eval.simple_evaluate(
    #     model=lm,
    #     tasks=tasks,
    #     num_fewshot=None,
    #     log_samples=True,
    #     confirm_run_unsafe_code=True,
    # )
    print(f"Starting evaluation on tasks: {list(task_configs.keys())}")
    
    # 3. Execute the evaluation sequentially to support dynamic few-shot counts
    master_results = {}
    
    for task_name, fewshot_count in task_configs.items():
        print(f"\n--- Evaluating {task_name.upper()} ({fewshot_count}-shot) ---")
        
        task_result = lm_eval.simple_evaluate(
            model=lm,                  # Reuse the model already loaded in VRAM
            tasks=[task_name],         # Pass as a valid list of strings
            num_fewshot=fewshot_count, # Apply the task-specific few-shot count
            log_samples=True,
            confirm_run_unsafe_code=True,
        )
        
        # Merge the specific task's output into the master dictionary
        if not master_results:
            master_results = task_result
        else:
            master_results["results"].update(task_result.get("results", {}))
            if "versions" in task_result:
                master_results["versions"].update(task_result["versions"])
            if "n-shot" in task_result:
                master_results["n-shot"].update(task_result["n-shot"])

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # 4. Save the results
    print(f"Saving comprehensive results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(master_results, f, default=handle_non_serializable, indent=2)

    # 5. Print Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    task_results = master_results.get("results", {})
    for task_name, metrics in task_results.items():
        print(f"\nTask: {task_name.upper()}")
        for metric_name, value in metrics.items():
            if not metric_name.endswith("stderr") and metric_name != "alias":
                formatted_val = f"{value:.4f}" if isinstance(value, float) else value
                print(f"  - {metric_name}: {formatted_val}")

if __name__ == "__main__":
    # Setup argparse to handle dynamic inputs
    parser = argparse.ArgumentParser(description="Run LLM Benchmarks (MMLU, GSM8K, Code, PPL)")
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Hugging Face Hub ID or absolute path to a local model directory."
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="/workspace/benchmark_results.json", 
        help="Path to save the JSON results."
    )

    parser.add_argument(
        "--quantized", 
        action="store_true", 
        help="Flag to indicate if the model is quantized (AWQ/GPTQ)."
    )
    
    args = parser.parse_args()
    
    # Run the evaluation
    run_benchmarks(
        model_path_or_id=args.model, 
        output_path=args.output,
        is_quantized=args.quantized
    )