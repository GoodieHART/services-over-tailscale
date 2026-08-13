#!/usr/bin/env python3
"""
GLM-OCR CPU-only inference test / benchmark script.

Target: 2x Cortex-A72 (or Neoverse N1) class ARM, ~12 GB RAM, no GPU.
Verified on: aarch64 Linux, 2 vCPUs, 11 GB RAM, Python 3.12, torch CPU wheel.

------------------------------------------------------------------------
INSTALL (CPU-only, aarch64) -- do NOT install any CUDA/torch CUDA build:
------------------------------------------------------------------------
    python3 -m venv glmocr_venv
    source glmocr_venv/bin/activate
    pip install --upgrade pip
    # CPU-only torch for aarch64 (the default PyPI 'torch' may pull CUDA deps
    # on aarch64; force the CPU wheel explicitly):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    # GLM-OCR needs the dev transformers that registers GlmOcrForConditionalGeneration:
    pip install "git+https://github.com/huggingface/transformers.git"
    pip install pillow torchvision accelerate

NOTE on dtype: the repo defaults to bfloat16 (torch_dtype="auto"). Cortex-A72 /
Neoverse-N1 class ARM CPUs have NO bf16 fast path, so oneDNN falls back to a slow
bf16->fp32 path. Pass --dtype float32 (or set the default below) for the native
fp32 path. Memory with float32 is ~2x the bf16 weights but still fits 12 GB.

------------------------------------------------------------------------
USAGE:
------------------------------------------------------------------------
    GLM_OCR_THREADS=2 python glm_ocr_test.py /path/to/image.png
    # or
    python glm_ocr_test.py --image /path/to/image.png --threads 4 --max-new-tokens 2048

The script loads the model once, runs OCR, and prints load time, per-image
inference wall time, and peak RSS (resident set size). It also returns the
OCR text. Peak RSS is read from resource.getrusage(RUSAGE_SELF).ru_maxrss
(which is monotonic / process-lifetime maximum on Linux, in KB).
"""
import os
import time
import resource
import argparse

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

MODEL_ID = "zai-org/GLM-OCR"


def peak_rss_gb() -> float:
    """Process-lifetime peak resident set size in GB (Linux: ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def load_model(model_id: str = MODEL_ID, dtype: str = "auto"):
    """Load processor + model. Returns (processor, model, load_time_s)."""
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    load_time = time.perf_counter() - t0
    return processor, model, load_time


def run_ocr(processor, model, image_path: str, prompt: str = "Text Recognition:",
            max_new_tokens: int = 2048):
    """Run single-image OCR. Returns (text, infer_time_s)."""
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    t0 = time.perf_counter()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    infer_time = time.perf_counter() - t0

    out = processor.decode(
        generated_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=False,
    )
    return out, infer_time


def main():
    ap = argparse.ArgumentParser(description="GLM-OCR CPU inference test")
    ap.add_argument("image", nargs="?", default="/tmp/sample_doc.png")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("GLM_OCR_THREADS", "2")))
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print(f"[glm_ocr_test] threads={args.threads} model={args.model} image={args.image}",
          flush=True)

    processor, model, load_t = load_model(args.model)
    print(f"[glm_ocr_test] load_time={load_t:.2f}s  peak_rss={peak_rss_gb():.2f}GB", flush=True)

    text, infer_t = run_ocr(processor, model, args.image, max_new_tokens=args.max_new_tokens)
    print(f"[glm_ocr_test] infer_time={infer_t:.2f}s  peak_rss={peak_rss_gb():.2f}GB", flush=True)
    print("----- OCR OUTPUT -----")
    print(text)
    print("----- END OCR OUTPUT -----")


if __name__ == "__main__":
    main()
