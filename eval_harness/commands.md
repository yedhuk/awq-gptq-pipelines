```
docker build -f Dockerfile.nvidia.quant -t llm-quant-benchmark:cu130 .
```

```
docker run -it --rm --gpus all \
  <!-- --user $(id -u):$(id -g) \ -->
  -v /home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines/notebooks/models:/models \
  -v $HF_HOME:/hf_home \
  -v /home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines:/workspace \
  -w /workspace/eval_harness \
  -e HF_HOME=/hf_home \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint /bin/bash \
  llm-benchmark:cu130 \
```
```
docker run -it --rm --gpus all \
  <!-- --user $(id -u):$(id -g) \ -->
  -v /home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines/notebooks/models:/models \
  -v $HF_HOME:/hf_home \
  -v /home/yedhu/workspace/ai-ml/projects/awq-gptq-pipelines:/workspace \
  -w /workspace/eval_harness \
  -e HF_HOME=/hf_home \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e HF_DATASETS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  --entrypoint /bin/bash \
  llm-quant-benchmark:cu130 

```


```
python evaluate_models.py --model "meta-llama/Llama-3.2-3B-Instruct" --output "/workspace/eval_harness/results/fp16_results.json"
python evaluate_models.py --model "/models/llama-3.2-3b-instruct-awq-custom" --output "/workspace/eval_harness/results/llama3_2_3b_awq_results.json" --quantized
python evaluate_models.py --model "/models/llama-3.2-3b-instruct-gptq-custom" --output "/workspace/eval_harness/results/llama3_2_3b_gptq_results.json" --quantized
```