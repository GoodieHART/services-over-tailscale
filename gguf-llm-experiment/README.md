# LLM-GGUF CPU Benchmark

This is directory contains scripts that lets **anyone** re-run the experiment i used to benchmarked 6 text-LLMs (quantized to Q4_K_M) via llama.cpp with the core goal of asserting which would be great as a daily personal chat assistant with strong reasoning. 


## Bundles

| Bundle | What it benchmarks | Result |
|---|---|---|
| [`gguf-llm-experiment/`](#gguf-llm-experiment-gguf-text-llm-cpu-benchmark) | 6 text-LLMs (Q4_K_M) via llama.cpp | depends on your setup|

---

### Bundle layout

```
gguf-llm-experiment/
├── reproduce.sh            # setup | e2b | e4b | ornith | 12b | qwen3-4b | qwen3-8b | all
├── prompt.txt              # test prompt (same for every run)
├── scripts/bench.sh        # benchmark one GGUF (TPS + true peak RSS)
```

### Prerequisites

- aarch64 or x86_64 Linux, **CPU-only** (no GPU needed)
- `git`, `cmake`, `make`, `gcc`, `wget`
- ~10 GB free disk

### How to run

```bash
cd gguf-llm-experiment
./reproduce.sh setup        # clone + build llama.cpp (once)
./reproduce.sh qwen3-4b 
./reproduce.sh qwen3-8b
./reproduce.sh all          # run all six in sequence
```

Running with **no argument prints usage and exits**. Each model is downloaded,
benchmarked, then **deleted immediately**. One model at a time.

### Results summary for my setup

Host: aarch64 CPU (2× Cores), 12.2 GB RAM, no GPU. llama.cpp CPU-only,
Q4_K_M, `-t 2 -n 64 --no-mmap --reasoning off --single-turn` (Ornith `-c 2048`).

| Model | File Size | Peak RSS | Prompt t/s | Gen t/s |
|---|---|---|---|---|
| gemma-4-E2B-it | 3.11 GB | 3.27 GB | 4.8 | 3.5 |
| Qwen3-4B-Instruct-2507 | 2.50 GB | 2.96 GB | 3.3 | 2.0 |
| gemma-4-E4B-it | 4.98 GB | 5.23 GB | 2.3 | 1.4 |
| Qwen3-8B | 5.03 GB | 5.32 GB | 1.3 | 0.8 |
| Ornith-1.0-9B | 5.63 GB | 5.52 GB | 1.5 | 1.0 |
| gemma-4-12B-agentic | 7.38 GB | 8.31 GB | 0.9 | 0.7 |

Each model benchmarked twice; figures are the mean of both runs. run1/run2
values are in `results/gguf_*_benchmark.json`.