# GLM-OCR CPU Benchmark

This is the repo for scripts that lets **anyone** re-run the experiment i conducted using `zai-org/GLM-OCR`
(a ~1.1B multimodal image-text-to-text OCR model) for a benchmark across three runtime types on a GPU-less host:

- **(A) PyTorch / transformers** — CPU wheel, `float32`, 2 threads
- **(B) Ollama** — no-sudo extract install, llama.cpp **F16**
- **(C) SGLang** — **not viable on aarch64 CPU**; the script detects this and skips

## Goal of the experiment

I aimed to benchmark GLM-OCR on my host system to check if the system is capable of being used as an always-on OCR endpoint tunneled through Tailscale to transform my physical notes into digital form.

## Bundle layout

```
glm-ocr-experiment/
├── reproduce.sh            # one-shot 3-section runner (pytorch | ollama | sglang | all)
├── sample_doc.png          # 900×1200 RGB invoice test image used for all runs
├── scripts/
    ├── glm_ocr_test.py
    └── ollama_ocr_test.sh
```

## Prerequisites

- aarch64 Linux or x86_64 (both Ollama and SGLang will run on x86_64; SGLang still skips on aarch64)
- Python 3.12, `python3-venv`, `curl`, `zstd`, `tar`, `pigz`/`base64`
- ~12 GB free disk
- **No sudo**: The script doesnt request for sudo password for any install, instead Ollama is installed by extracting a tarball into a local dir.

## How to run

```bash
chmod +x glm-ocr-experiment/reproduce.sh        # already executable in this bundle
./glm-ocr-experiment/reproduce.sh pytorch       # path A: CPU torch + transformers, fp32, threads=2
./glm-ocr-experiment/reproduce.sh ollama        # path B: extract-install Ollama, pull + run F16
./glm-ocr-experiment/reproduce.sh sglang        # path C: skips on aarch64; real CPU install + benchmark on x86_64
./glm-ocr-experiment/reproduce.sh all           # run all three in sequence
```

Running with **no argument prints usage and exits** — it does *not* auto-run all
three (to avoid hammering your system). The original
`glm-ocr-experiment/scripts/glm_ocr_test.py` / `glm-ocr-experiment/scripts/ollama_ocr_test.sh` can be invoked as-is.

## Results summary

Host: aarch64 CPU, 2 vCPU, ~11.6 GiB RAM, no GPU,
Python 3.12.3. Measurements exclude model load.

| Runtime  | dtype | Infer / image | Peak RSS | Notes |
|----------|-------|---------------|----------|-------|
| PyTorch  | fp32  | **~206–262 s** | **~6.6 GB** | `threads=2` |
| Ollama   | F16   | **~79 s**      | **~3.0 GB** | llama.cpp; ~2.6× faster than PyTorch |
| SGLang   | —     | not viable     | —        | aarch64 CPU; there's no torchvision CPU wheel, AMX-only engine |
