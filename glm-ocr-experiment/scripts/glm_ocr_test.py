#!/usr/bin/env python3
"""GLM-OCR CPU-only inference benchmark.

dtype="auto" resolves to bfloat16; this CPU class has no bf16 fast path and
oneDNN falls back to a slow bf16->fp32 path, so pass --dtype float32.
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
    """Process-lifetime peak RSS in GB. Linux ru_maxrss is KB; macOS is bytes."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def load_model(model_id: str = MODEL_ID, dtype: str = "auto"):
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
