```
docker run -it --rm --gpus all \
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
python evaluate_models.py --model "meta-llama/Llama-3.2-3B-Instruct" --output "/workspace/eval_harness/results/fp16_results.json"
```