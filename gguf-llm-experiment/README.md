# LLM-GGUF CPU Benchmark

This is directory contains scripts that lets **anyone** re-run the experiment i used to benchmarked 6 text-LLMs (quantized to Q4_K_M) via llama.cpp to select a daily personal chat assistant with strong reasoning. 


## Bundles

| Bundle | What it benchmarks | Result |
|---|---|---|
| [`gguf-llm-experiment/`](#gguf-llm-experiment-gguf-text-llm-cpu-benchmark) | 6 text-LLMs (Q4_K_M) via llama.cpp, selecting a daily chat assistant with strong reasoning | depends on your setup|

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

Host: aarch64 CPU, 11.9 GB RAM, no GPU. llama.cpp CPU-only,
Q4_K_M, `-t 2 -n 64 --no-mmap --reasoning off --single-turn` (Ornith `-c 2048`).

| Model | File | Peak RSS | Prompt t/s | Gen t/s |
|---|---|---|---|---|
| gemma-4-E2B-it | 3.11 GB | 3.99 GB | 9.9 | 3.4 |
| Qwen3-4B-Instruct-2507 | 2.38 GB | 3.25 GB | 8.3 | 2.4 |
| gemma-4-E4B-it | 4.98 GB | 7.16 GB | 4.0 | 2.3 |
| Qwen3-8B | 4.79 GB | 5.31 GB | 4.3 | 1.5 |
| Ornith-1.0-9B | 5.63 GB | 6.11 GB | 3.5 | 1.2 |
| gemma-4-12B-agentic | 7.38 GB | 10.83 GB | 0.3 | 0.8 |