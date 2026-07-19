import asyncio
import aiohttp
import time
import pandas as pd
import json
from transformers import AutoTokenizer
import re
import mlflow
from mlflow.tracking import MlflowClient
# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
API_URL = "http://localhost:8000/v1/completions"
METRICS_URL = "http://localhost:8000/metrics"
MODEL_FP16 = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME = "/home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines/notebooks/models/llama-3.2-3b-instruct-gptq-custom"
CONCURRENCY_LIMIT = 256  # Number of simultaneous requests
SAMPLE_SIZE = 4096
RESULTS_DIR = "results"

LOG_NAME = f"vllm_online_{re.sub(r'[.-]', '_', MODEL_NAME.split('/')[-1])}_CL_{CONCURRENCY_LIMIT}_SS_{SAMPLE_SIZE}"


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Quantization Benchmarking")
print(mlflow.get_tracking_uri())

def mlflow_start_run(name: str):
    if mlflow.active_run():
        mlflow.end_run()
    return mlflow.start_run(run_name=name)

def log_request_to_mlflow(run_id, result):
    """Logs metrics directly to the main run ID using the client."""
    client = MlflowClient()
    step = result["request_id"]
    
    # Notice we pass run_id as the first argument
    client.log_metric(run_id, "req_ttft_ms", result["ttft_ms"], step=step)
    client.log_metric(run_id, "req_itl_ms", result["itl_ms"], step=step)
    client.log_metric(run_id, "req_tokens_per_sec", result["tokens_per_sec"], step=step)

def log_hardware_to_mlflow(run_id, step, kv_perc, running_reqs):
    client = MlflowClient()
    client.log_metric(run_id, "live_kv_cache_percent", kv_perc, step=step)
    client.log_metric(run_id, "live_running_requests", running_reqs, step=step)

async def monitor_vllm_memory(session, stop_event,run_id, poll_interval=1.0, log_every_n_polls=1):
    """Background task to poll vLLM's internal KV cache memory usage."""
    
    max_kv_cache_usage = 0.0
    max_running_requests = 0
    poll_count = 0
    
    # Pre-compile regex for slight efficiency in the loop
    # kv_regex = re.compile(r'vllm:gpu_cache_usage_perc\{.*?\}\s+([0-9.]+)')
    kv_regex = re.compile(r'vllm:(?:gpu|kv)_cache_usage_perc\{.*?\}\s+([0-9.]+)')
    req_regex = re.compile(r'vllm:num_requests_running\{.*?\}\s+([0-9.]+)')
    
    while not stop_event.is_set():
        try:
            async with session.get(METRICS_URL) as response:
                if response.status == 200:
                    text = await response.text()
                    
                    # 1. Extract Metrics
                    current_kv = 0.0
                    current_reqs = 0
                    
                    kv_match = kv_regex.search(text)
                    if kv_match:
                        current_kv = float(kv_match.group(1))
                        max_kv_cache_usage = max(max_kv_cache_usage, current_kv)
                        
                    req_match = req_regex.search(text)
                    if req_match:
                        current_reqs = int(float(req_match.group(1)))
                        max_running_requests = max(max_running_requests, current_reqs)
                    
                    # 2. MLflow Logging Logic
                    poll_count += 1
                    
                    # Log based on the frequency you chose (e.g., every 5 seconds)
                    if poll_count % log_every_n_polls == 0:
                        # Use poll_count as the "step" so your MLflow X-axis scales linearly over time
                        await asyncio.to_thread(
                            log_hardware_to_mlflow, 
                            run_id,
                            poll_count, 
                            current_kv*100, 
                            current_reqs
                        )
                        
        except Exception as e:
            # Silently ignore brief connection drops
            pass 
            
        await asyncio.sleep(poll_interval)
        
    return max_kv_cache_usage, max_running_requests
# ---------------------------------------------------------
# The Asynchronous Worker
# ---------------------------------------------------------
async def fetch_stream(session, prompt, expected_len, request_id, semaphore,run_id):
    """Sends a single streaming request and tracks precise latencies."""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": expected_len,
        "temperature": 0.0,
        "stream": True,
        # vLLM supports returning token usage in the final stream chunk
        "stream_options": {"include_usage": True} 
    }
    
    # The semaphore ensures we don't exceed our concurrency limit
    async with semaphore:
        start_time = time.perf_counter()
        first_token_time = None
        chunk_count = 0
        completion_tokens = 0
        
        try:
            async with session.post(API_URL, json=payload) as response:
                response.raise_for_status()
                
                # Iterate over the streaming chunks
                async for line in response.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            
                            # Capture TTFT on the very first chunk
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                                
                            chunk_count += 1
                            
                            # Parse JSON to grab final token usage if available
                            data = json.loads(line[6:])
                            if data.get("usage") is not None:
                                completion_tokens = data["usage"].get("completion_tokens", chunk_count)

                end_time = time.perf_counter()
                
        except Exception as e:
            print(f"Request {request_id} failed: {e}")
            return None

        # Fallback if usage stats weren't returned
        if completion_tokens == 0:
            completion_tokens = chunk_count

        # Calculate Latencies
        ttft = (first_token_time - start_time) if first_token_time else 0.0
        generation_time = (end_time - first_token_time) if first_token_time else 0.0
        itl = generation_time / (completion_tokens - 1) if completion_tokens > 1 else 0.0
        tps = (completion_tokens - 1) / generation_time if generation_time > 0 else 0.0

        result = {
                "request_id": request_id,
                "completion_tokens": completion_tokens,
                "ttft_seconds": ttft,
                "itl_seconds": itl,
                "ttft_ms": ttft*1000,
                "itl_ms": itl*1000,
                "tokens_per_sec": tps,
                "total_latency": ttft + generation_time
            }
        
        # Fire and forget the MLflow logging in a separate thread
        # This ensures your asyncio loop instantly moves on to the next API request
        await asyncio.to_thread(log_request_to_mlflow, run_id, result)
        
        return result
# ---------------------------------------------------------
# The Main Execution Loop
# ---------------------------------------------------------
async def main():
    print(f"Starting API Load Test with Concurrency Limit: {CONCURRENCY_LIMIT}")
    # mlflow_start_run(f"Benchmarking {LOG_NAME}")
    # Start the run BEFORE any async tasks are created
    with mlflow.start_run(run_name=f"Benchmarking {LOG_NAME}") as run:
        # Extract the universal run ID for this benchmark
        run_id = run.info.run_id
    
        # Log static parameters immediately
        mlflow.log_param("model_name",MODEL_NAME)
        mlflow.log_param("concurrency_limit", CONCURRENCY_LIMIT)
        mlflow.log_param("gpu_memory_utilization", 0.85)


        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = []


        file_path = f"/home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines/notebooks/datasets/shareGPT_custom/static_shareGPT_dataset_{SAMPLE_SIZE}.jsonl"
        static_shareGPT_dataset = pd.read_json(file_path, orient="records", lines=True)
        # Optional: Sort by expected_len to eliminate padding tax if you decide 
        # to use dynamic expected lengths instead of the static 256 above.
        df = static_shareGPT_dataset.sort_values(by="expected_len").reset_index(drop=True)


        prompts = df["prompt"].tolist()
        expected_lens = df["expected_len"].tolist()


        tokenizer =  AutoTokenizer.from_pretrained(MODEL_FP16)

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
        
        # We use a single TCP connection pool (ClientSession) for maximum efficiency
        async with aiohttp.ClientSession() as session:

            # Create an event to tell the monitor when the load test finishes
            stop_event = asyncio.Event()
            
            # Start the background memory monitor
            monitor_task = asyncio.create_task(
                monitor_vllm_memory(session, stop_event,run_id)
            )

            system_start_time = time.perf_counter()
            
            # Schedule all requests immediately; the semaphore will throttle them internally
            for i, (prompt, target_len) in enumerate(zip(valid_prompts, valid_lens)):
                task = asyncio.create_task(
                    fetch_stream(session, prompt, target_len, i, semaphore,run_id)
                )
                tasks.append(task)
                
            # Wait for all requests to finish
            results = await asyncio.gather(*tasks)
            system_end_time = time.perf_counter()

            # Stop the monitor and retrieve the peak metrics
            stop_event.set()
            peak_kv_perc, peak_running = await monitor_task


        # ---------------------------------------------------------
        # KV CACHE FOOTPRINT
        # ---------------------------------------------------------

        TOTAL_KV_POOL_GIB = 10.14 #10.1 #6.79
        
        exact_kv_gib_used = TOTAL_KV_POOL_GIB * peak_kv_perc

        print("\n--- Hardware Diagnostics ---")
        print(f"Peak KV Cache Usage: {peak_kv_perc*100:.1f}% ({exact_kv_gib_used:.2f} GiB)")
        print(f"Peak Concurrent GPU Requests: {peak_running}")


            
        # Log the memory footprint
        # mlflow.log_metric("peak_kv_cache_percent", peak_kv_perc)
        # mlflow.log_metric("peak_kv_cache_gib", exact_kv_gib_used)
        # mlflow.log_metric("peak_concurrent_requests", peak_running)
        # ---------------------------------------------------------
        # Aggregate and Report
        # ---------------------------------------------------------
        valid_results = [r for r in results if r is not None]
        results_df = pd.DataFrame(valid_results)
        # results_df["ttft_ms"] = results_df["ttft_seconds"]*100
        # results_df["itl_ms"] = results_df["itl_seconds"]*100


        # Saves inside the directory
        results_df.to_csv(f"{RESULTS_DIR}/vllm_online_bm_{re.sub(r'[.-]', '_', MODEL_NAME.split('/')[-1])}_{CONCURRENCY_LIMIT}_SS_{SAMPLE_SIZE}.csv")

        
        total_time = system_end_time - system_start_time
        total_tokens = results_df['completion_tokens'].sum()
        system_tps = total_tokens / total_time

        
        print("\n--- REST API Load Test Diagnostics ---")
        print(f"Total Workload Time:     {total_time:.2f} seconds")
        print(f"Total Useful Tokens:     {total_tokens}")
        print(f"Average TTFT:            {results_df['ttft_seconds'].mean():.4f} seconds")
        print(f"Average Inter-Token Lat: {results_df['itl_seconds'].mean():.4f} seconds")
        print(f"Average User Tokens/Sec: {results_df['tokens_per_sec'].mean():.2f} tps")
        print(f"SYSTEM THROUGHPUT:       {system_tps:.2f} tokens/sec\n")

        # Log the final aggregates to the SAME run (the context manager is still active here)


        mlflow.log_metric("total_workload_time_seconds",total_time )
        mlflow.log_metric("total_useful_token",total_tokens )
        mlflow.log_metric("avg_ttft_ms", results_df['ttft_ms'].mean())
        mlflow.log_metric("avg_itl_ms",results_df['itl_ms'].mean())
        mlflow.log_metric("avg_tps",results_df['tokens_per_sec'].mean())
        mlflow.log_metric("system_tps", system_tps)


        mlflow.log_metric("peak_kv_cache_percent", peak_kv_perc*100)
        mlflow.log_metric("peak_kv_cache_gib", exact_kv_gib_used)
        mlflow.log_metric("peak_concurrent_requests", peak_running)
        
        results_df.to_csv("benchmark_results.csv", index=False)
        mlflow.log_artifact("benchmark_results.csv")

# Run the async event loop
if __name__ == "__main__":
    asyncio.run(main())