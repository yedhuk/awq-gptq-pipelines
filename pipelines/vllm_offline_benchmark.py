import time
import pandas as pd
from vllm import LLM, SamplingParams

SAMPLE_SIZE = 128


# ---------------------------------------------------------
# 1. Initialize vLLM Engine
# ---------------------------------------------------------
# gpu_memory_utilization=0.85 reserves 85% of your 16GB VRAM purely for 
# model weights and the KV cache, leaving enough overhead to prevent OOMs.
print("Initializing vLLM Engine...")
llm = LLM(
    model="meta-llama/Llama-3.2-3B-Instruct", 
    dtype="float16",
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=4096
)

file_path = f"/home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines/notebooks/datasets/shareGPT_custom/static_shareGPT_dataset_{SAMPLE_SIZE}.jsonl"
static_shareGPT_dataset = pd.read_json(file_path, orient="records", lines=True)
# Optional: Sort by expected_len to eliminate padding tax if you decide 
# to use dynamic expected lengths instead of the static 256 above.
df = static_shareGPT_dataset.sort_values(by="expected_len").reset_index(drop=True)


prompts = df["prompt"].tolist()
expected_lens = df["expected_len"].tolist()


tokenizer = llm.get_tokenizer()

valid_prompts = []
valid_lens = []

# We set the absolute ceiling to 4096 to match max_model_len
MAX_TOTAL_TOKENS = 4096 

for prompt_text, target_len in zip(prompts, expected_lens):
    # Count the tokens in the user's prompt
    prompt_token_count = len(tokenizer.encode(prompt_text))
    
    # Check if the Prompt + The expected generation fits inside our engine ceiling
    if (prompt_token_count + target_len) <= MAX_TOTAL_TOKENS:
        valid_prompts.append(prompt_text)
        valid_lens.append(target_len)

print(f"Filtered dataset size: {len(valid_prompts)} valid prompts")

# ---------------------------------------------------------
# 2. Configure Dynamic Sampling Parameters
# ---------------------------------------------------------
# vLLM allows you to pass a unique SamplingParams object for every single prompt.
# We set max_tokens dynamically based on your dataset calculations.
sampling_params_list = [
    SamplingParams(
        temperature=0.0,      # Greedy decoding for deterministic benchmarks
        max_tokens=length, 
        ignore_eos=False      # Allow the model to stop naturally if it finishes early
    )
    for length in valid_lens
]

# ---------------------------------------------------------
# 3. Execute Continuous Batching
# ---------------------------------------------------------
print(f"Starting vLLM inference on {len(valid_prompts)} prompts...")

# The system timer tracks the wall-clock time for the ENTIRE workload
start_time = time.perf_counter()

# Pass the entire list at once. vLLM will automatically chunk, batch, 
# and schedule them on the GPU for maximum throughput.
outputs = llm.generate(valid_prompts, sampling_params=sampling_params_list)

total_time = time.perf_counter() - start_time

# ---------------------------------------------------------
# 4. Extract Metrics
# ---------------------------------------------------------
benchmark_results = []
total_completion_tokens = 0

for output in outputs:
    # Extract token counts directly from the output object
    query_tokens = len(output.prompt_token_ids)
    completion_tokens = len(output.outputs[0].token_ids)
    total_completion_tokens += completion_tokens
    
    # vLLM automatically tracks high-precision telemetry per request
    metrics = output.metrics
    
    if metrics is not None and metrics.first_token_time is not None:
        ttft = metrics.first_token_time - metrics.arrival_time
        generation_time = metrics.finished_time - metrics.first_token_time
    else:
        # Failsafe if the model generated zero tokens
        ttft = 0.0
        generation_time = 0.0

    # Calculate individual throughput and ITL
    if completion_tokens > 1 and generation_time > 0:
        itl = generation_time / (completion_tokens - 1)
        tps = (completion_tokens - 1) / generation_time
    else:
        itl = 0.0
        tps = 0.0
        
    benchmark_results.append({
        "prompt_id": output.request_id,
        "query_tokens": query_tokens,
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft,
        "itl_seconds": itl,
        "tokens_per_sec": tps,
        "total_latency": ttft + generation_time
    })

# Convert to a DataFrame for easy analysis and CSV/Parquet exporting
results_df = pd.DataFrame(benchmark_results)

# ---------------------------------------------------------
# 5. Output Final System Diagnostics
# ---------------------------------------------------------
system_tps = total_completion_tokens / total_time

print("\n--- vLLM Benchmark Diagnostics ---")
print(f"Total Workload Time:     {total_time:.2f} seconds")
print(f"Total Useful Tokens:     {total_completion_tokens}")
print(f"Average TTFT:            {results_df['ttft_seconds'].mean():.4f} seconds")
print(f"Average Inter-Token Lat: {results_df['itl_seconds'].mean():.4f} seconds")
print(f"Average Tokens/Sec:      {results_df['tokens_per_sec'].mean():.2f} tps")
print(f"SYSTEM THROUGHPUT:       {system_tps:.2f} tokens/sec\n")