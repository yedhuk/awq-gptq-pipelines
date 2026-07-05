import sys, platform
import torch

def _v(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "not installed"

print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"Python              : {platform.python_version()}")
print(f"PyTorch             : {torch.__version__}")
print(f"CUDA (torch built)  : {torch.version.cuda}")
print(f"cuDNN               : {torch.backends.cudnn.version()}")
print(f"Transformers        : {_v('transformers')}")
print(f"AutoAWQ             : {_v('awq')}")
print(f"Accelerate          : {_v('accelerate')}")

cuda_ok = torch.cuda.is_available()
print("-" * 60)
print(f"CUDA available      : {cuda_ok}")
if cuda_ok:
    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    total_vram = props.total_memory / 1024**3
    # Tensor cores exist on Volta (sm_70) and newer
    tensor_cores = props.major >= 7
    print(f"GPU name            : {props.name}")
    print(f"Compute capability  : sm_{props.major}{props.minor}")
    print(f"Total VRAM          : {total_vram:.2f} GiB")
    print(f"Tensor Core support : {tensor_cores}")
    # Blackwell (RTX 50-series) is sm_120 and needs torch>=2.7.0 built with CUDA 12.8
    if props.major == 12:
        torch_major_minor = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if torch_major_minor < (2, 7):
            print("!! WARNING: Blackwell (sm_120) GPU detected but torch < 2.7.0.")
            print("   Expect 'no kernel image is available'. Install torch>=2.7.0 cu128.")
        else:
            print("Blackwell note      : sm_120 detected; AWQ will try Marlin "
                  "(auto-round) kernels. See Section 11 for the backend fallback chain.")
else:
    print("WARNING: No CUDA GPU detected. Benchmarks need a GPU.")
print("=" * 60)

import os
import gc
import time
import shutil
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# AWQ models are loaded through transformers (Section 11) using a kernel-backend
# fallback chain. AutoAWQ is optional and only used as a last-resort backend; it is
# deprecated and its CUDA kernels have no sm_120 build, so it is not required here.
try:
    from awq import AutoAWQForCausalLM  # noqa: F401  (optional legacy fallback)
    HAS_AWQ = True
except Exception:
    HAS_AWQ = False

pd.set_option("display.max_colwidth", 80)
pd.set_option("display.precision", 3)


MODEL_FP16     = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_AWQ      = "casperhansen/llama-3.2-3b-instruct-awq"

PROMPT         = "Explain what quantization means for large language models in two sentences."
MAX_NEW_TOKENS = 256
TEMPERATURE    = 0.7
TOP_P          = 0.9

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE          = torch.float16

# On Blackwell (RTX 50-series, sm_120) the AutoAWQ CUDA kernels have no build and
# fused modules don't work. The loader in Section 11 tries these backends in order
# and uses the first that loads. "marlin" (via auto-round) mirrors what vLLM does:
# it repacks a standard zero-point AWQ checkpoint into Marlin format and runs it
# with sm_120-capable kernels. Reorder if you want to force a specific backend.
AWQ_BACKEND_PREFERENCE = ["marlin", "exllama", "auto"]

OUTPUT_FOLDER  = Path("outputs")


# ---- output directories -----------------------------------------------------
FIG_DIR = OUTPUT_FOLDER / "figures"
CSV_DIR = OUTPUT_FOLDER / "csv"
LOG_DIR = OUTPUT_FOLDER / "logs"
for d in (FIG_DIR, CSV_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- GPU memory --------------------------------------------------------------
def gpu_allocated_gb(device=DEVICE):
    if device != "cuda":
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3

def gpu_reserved_gb(device=DEVICE):
    if device != "cuda":
        return 0.0
    return torch.cuda.memory_reserved() / 1024**3

def gpu_peak_gb(device=DEVICE):
    if device != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3

def gpu_reset_peak(device=DEVICE):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ---- CPU memory --------------------------------------------------------------
_proc = psutil.Process(os.getpid())

def cpu_resident_gb():
    return _proc.memory_info().rss / 1024**3

def cpu_virtual_gb():
    return _proc.memory_info().vms / 1024**3

# ---- disk size ---------------------------------------------------------------
def human_readable(num_bytes):
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PiB"

def folder_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not p.is_symlink():
            total += p.stat().st_size
    return total

def model_size(model):
    # On-GPU/parameter size of a loaded model in bytes (params + buffers).
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    for b in model.buffers():
        total += b.nelement() * b.element_size()
    return total

# ---- timer -------------------------------------------------------------------
class BenchmarkTimer:
    # Context manager that records wall-clock seconds and syncs CUDA.
    # Usage:  with BenchmarkTimer() as t: ...work... ; then read t.elapsed
    def __init__(self, sync=True):
        self.sync = sync
        self.elapsed = None

    def __enter__(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start
        return False

print("Helper utilities ready.")

from huggingface_hub import snapshot_download

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def ensure_model(repo_id, label):
    # Download repo_id into models/<label> if not already present.
    local_dir = MODELS_DIR / label
    if local_dir.exists() and any(local_dir.rglob("*.safetensors")):
        size = folder_size(local_dir)
        print(f"[{label}] already present — {human_readable(size)} (skipped)")
        return str(local_dir), 0.0, size

    print(f"[{label}] downloading {repo_id} ...")
    with BenchmarkTimer(sync=False) as t:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.pth", "original/*"],
        )
    size = folder_size(local_dir)
    print(f"[{label}] done in {t.elapsed:.1f}s — {human_readable(size)}")
    return str(local_dir), t.elapsed, size

FP16_PATH, FP16_DL_TIME, FP16_DISK = ensure_model(MODEL_FP16, "fp16")
AWQ_PATH,  AWQ_DL_TIME,  AWQ_DISK  = ensure_model(MODEL_AWQ,  "awq")

print(f"\nFP16 on disk : {human_readable(FP16_DISK)}")
print(f"AWQ  on disk : {human_readable(AWQ_DISK)}")

PROMPTS = [
    ("simple_qa_1",   "What is the capital of Japan?", 32),
    ("simple_qa_2",   "Who wrote the play 'Hamlet'?", 32),
    ("reasoning_1",   "If a train travels 60 km in 45 minutes, what is its average speed in km/h? Explain.", 128),
    ("reasoning_2",   "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much is the ball? Reason step by step.", 128),
    ("coding_1",      "Write a Python function that returns the n-th Fibonacci number using memoization.", 200),
    ("coding_2",      "Write a SQL query to find the second highest salary from an Employees table.", 128),
    ("summarization", "Summarize the following in one sentence: Quantization reduces the numerical precision of model weights and activations, lowering memory and bandwidth needs while usually keeping accuracy close to the full-precision baseline.", 64),
    ("long_context",  "Here are five facts: (1) The sky is blue. (2) Water boils at 100C at sea level. (3) Paris is in France. (4) Honey does not spoil. (5) Spiders have eight legs. Which fact is about cooking, and which is about geography?", 96),
    ("math_1",        "Compute 17 * 23 and show the multiplication steps.", 96),
    ("math_2",        "What is the derivative of f(x) = 3x^2 + 2x - 5 with respect to x?", 64),
]

dataset = pd.DataFrame(PROMPTS, columns=["name", "prompt", "expected_len"])
dataset

def load_fp16(path):
    clear_cache(); gpu_reset_peak()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with BenchmarkTimer() as t:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=DTYPE,
            device_map=DEVICE,
            attn_implementation="sdpa",
        )
    model.eval()
    return tok, model, t.elapsed

fp16_tok, fp16_model, fp16_load_time = load_fp16(FP16_PATH)

fp16_alloc = gpu_allocated_gb()
fp16_peak  = gpu_peak_gb()
fp16_params_bytes = model_size(fp16_model)

print(f"Load time          : {fp16_load_time:.2f} s")
print(f"Allocated VRAM     : {fp16_alloc:.3f} GiB")
print(f"Peak VRAM (load)   : {fp16_peak:.3f} GiB")
print(f"Param/buffer size  : {human_readable(fp16_params_bytes)}")
print(f"Architecture       : {fp16_model.config.architectures}")
print(f"Model dtype        : {next(fp16_model.parameters()).dtype}")
print(f"Attn implementation: {getattr(fp16_model.config, '_attn_implementation', 'n/a')}")

def inspect_layers(model, named_targets):
    rows = []
    modules = dict(model.named_modules())
    for label, name in named_targets:
        mod = modules.get(name)
        if mod is None:
            rows.append({"label": label, "module": name, "type": "MISSING",
                         "weight_shape": None, "dtype": None})
            continue
        w = getattr(mod, "weight", None)
        # AWQ packed layers expose `qweight` instead of `weight`
        if w is None:
            w = getattr(mod, "qweight", None)
        rows.append({
            "label": label,
            "module": name,
            "type": type(mod).__name__,
            "weight_shape": tuple(w.shape) if w is not None else None,
            "dtype": str(w.dtype) if w is not None else None,
        })
    return pd.DataFrame(rows)

FP16_TARGETS = [
    ("embedding",  "model.embed_tokens"),
    ("attn.q_proj","model.layers.0.self_attn.q_proj"),
    ("attn.k_proj","model.layers.0.self_attn.k_proj"),
    ("attn.v_proj","model.layers.0.self_attn.v_proj"),
    ("attn.o_proj","model.layers.0.self_attn.o_proj"),
    ("mlp.gate",   "model.layers.0.mlp.gate_proj"),
    ("mlp.up",     "model.layers.0.mlp.up_proj"),
    ("mlp.down",   "model.layers.0.mlp.down_proj"),
    ("lm_head",    "lm_head"),
]

inspect_layers(fp16_model, FP16_TARGETS)