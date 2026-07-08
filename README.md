# AWQ / GPTQ Pipelines

A benchmarking suite for studying modern LLM quantization techniques. Rather than just loading a quantized model and calling it done, this project benchmarks the full pipeline — download, load, memory footprint, layer/kernel structure, and inference performance — so quantization methods can be compared on real numbers.

Current focus: **FP16 vs INT4 AWQ vs INT4 GPTQ** on `meta-llama/Llama-3.2-3B-Instruct`. The benchmarking engine is designed to be reusable for future quantization methods (GGUF, FP8, SmoothQuant, ...) by swapping only the model-loading step.

## What it measures

- Storage savings (disk size of checkpoint)
- GPU memory savings (allocated / reserved / peak VRAM)
- Inference latency (time-to-first-token, decode time, total time)
- Throughput (tokens/sec)
- Output quality (side-by-side response comparison)
- Architecture changes (which layers/modules change under quantization)
- Quantized kernel/backend verification (confirming the expected kernel path is actually used)

## Project structure

```
awq-gptq-pipelines/
├── notebooks/
│   ├── llama_3_2_3b_awq_experiments.ipynb      # Quantizes the FP16 model to AWQ and saves a local checkpoint
│   ├── llama_3_2_3b_gptq_experiments.ipynb     # Quantizes the FP16 model to GPTQ (GPTQModel) and saves a local checkpoint
│   ├── llama_3_2_3b_inference_benchmark.ipynb  # Benchmark engine: loads FP16 + AWQ + GPTQ, runs prompts, logs metrics
│   ├── test.py                                 # Standalone script version of env checks + benchmark helpers
│   ├── models/                                 # Local model checkpoints (quantized output lands here)
│   ├── logs/                                   # GPTQModel quantization logs (gitignored)
│   └── deprecated/                             # Superseded early notebooks, kept for reference only
├── pipelines/
│   ├── awq_pipeline.py                         # AWQPipeline class: load/quantize/generate/save (AutoAWQ)
│   └── gptq_pipeline.py                        # GPTQPipeline class: load/quantize/generate/save (GPTQModel)
├── awq-requirements.txt                        # Deps for the AWQ quantization notebook/pipeline
├── gptq-requirements.txt                       # Deps for the GPTQ quantization notebook/pipeline
├── benchmark-requirements.txt                  # Deps for the inference benchmark notebook
└── project.md                                  # Design notes / roadmap for the benchmarking framework
```

## Setup

1. Create/activate a Python environment (this project was developed against Python 3.10).
2. Install PyTorch for your CUDA setup (pick the wheel that matches your GPU/driver from https://pytorch.org).
3. Install the dependencies for whichever notebook(s) you plan to run (they overlap on `torch`/`transformers`/jupyter, so installing more than one into the same environment is fine):

   ```bash
   pip install -r awq-requirements.txt        # AWQ quantization notebook/pipeline
   pip install -r gptq-requirements.txt       # GPTQ quantization notebook/pipeline
   pip install -r benchmark-requirements.txt  # inference benchmark notebook
   ```

4. Authenticate with Hugging Face, since Llama-3.2 is a gated model:

   ```bash
   huggingface-cli login
   # or: export HF_TOKEN=hf_xxx
   ```

5. Sanity-check your environment (Python/PyTorch/CUDA/GPU/VRAM):

   ```bash
   python notebooks/test.py
   ```

## Usage

1. **Quantize the base model**
   - AWQ: run `notebooks/llama_3_2_3b_awq_experiments.ipynb` to produce a local AWQ checkpoint under `notebooks/models/`.
   - GPTQ: run `notebooks/llama_3_2_3b_gptq_experiments.ipynb` to produce a local GPTQ checkpoint under `notebooks/models/`. This downloads a calibration dataset (`allenai/c4`, 1024 samples) and takes noticeably longer than AWQ quantization.
2. **Benchmark** — run `notebooks/llama_3_2_3b_inference_benchmark.ipynb` to load the FP16 baseline and both quantized checkpoints, run them over a fixed prompt set (QA, reasoning, coding, summarization, long-context, math), and collect metrics.
3. **(Optional) Track results in MLflow** — the inference notebook logs metrics to an MLflow tracking server. Start one locally before running it:

   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

   Metrics are logged under the experiment `AWQ Quantization Benchmarking`, namespaced by stage (`baseline.*`, `post_load.*`, `post_inference.*`, `post_cleanup.*`).

## Roadmap

See `project.md` for the full design spec. Planned extensions:

- Additional quantization methods: SmoothQuant, LLM.int8(), FP8, GGUF, HQQ, AQLM, SpQR
- Additional inference engines: vLLM, TensorRT-LLM, llama.cpp, ExLlamaV2
- Multi-batch and long-context scaling benchmarks
- Hardware comparisons across GPUs
