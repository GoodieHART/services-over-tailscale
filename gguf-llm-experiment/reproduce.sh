#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${BUNDLE_DIR}/scripts"
BENCH="${SCRIPTS_DIR}/bench.sh"
MODELS_DIR="${BUNDLE_DIR}/models"
LLAMA_DIR="${BUNDLE_DIR}/llama.cpp"
LLAMA_BIN="${LLAMA_DIR}/build/bin/llama-cli"

HF_BASE="https://huggingface.co"

declare -A REPO FILE EXTRA CTX
REPO[e2b]="unsloth/gemma-4-E2B-it-GGUF"
FILE[e2b]="gemma-4-E2B-it-Q4_K_M.gguf"

REPO[e4b]="unsloth/gemma-4-E4B-it-GGUF"
FILE[e4b]="gemma-4-E4B-it-Q4_K_M.gguf"

REPO[ornith]="ornith-ai/Ornith-1.0-9B-GGUF"
FILE[ornith]="ornith-1.0-9b-Q4_K_M.gguf"
CTX[ornith]=2048        # If we don't cap the 262K native context your system will hit an OOM or may hang, well depends on your setup.

REPO[12b]="yuxinlu1/gemma-4-12B-agentic-fable5-composer2.5-v2-3.5x-tau2-GGUF"
FILE[12b]="gemma4-v2-Q4_K_M.gguf"

REPO[qwen3-4b]="unsloth/Qwen3-4B-Instruct-2507-GGUF"
FILE[qwen3-4b]="Qwen3-4B-Instruct-2507-Q4_K_M.gguf"

REPO[qwen3-8b]="unsloth/Qwen3-8B-GGUF"
FILE[qwen3-8b]="Qwen3-8B-Q4_K_M.gguf"

setup() {
  echo "==================== SETUP: llama.cpp ===================="
  if [ -x "${LLAMA_BIN}" ]; then
    echo "llama.cpp already built at ${LLAMA_BIN}"
    return 0
  fi
  if [ ! -d "${LLAMA_DIR}" ]; then
    echo "Cloning llama.cpp (depth 1) ..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${LLAMA_DIR}"
  fi
  if [ ! -f "${LLAMA_DIR}/CMakeLists.txt" ]; then
    echo "ERROR: llama.cpp clone at ${LLAMA_DIR} is incomplete (no CMakeLists.txt — clone failed?)" >&2
    exit 1
  fi
  cmake -B "${LLAMA_DIR}/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF "${LLAMA_DIR}"
  cmake --build "${LLAMA_DIR}/build" --config Release -j 2
  echo "Built: ${LLAMA_BIN}"
  echo "==================== END SETUP ===================="
}

run_model() {
  local key="$1"
  local repo="${REPO[$key]}" file="${FILE[$key]}"
  local extra="${EXTRA[$key]:-}" ctx="${CTX[$key]:-4096}"
  local dest="${MODELS_DIR}/${file}"
  local url="${HF_BASE}/${repo}/resolve/main/${file}"

  echo "==================== MODEL: ${key} (${repo}) ===================="

  # Check for free space before pulling
  local free_kb
  free_kb=$(df -Pk "${MODELS_DIR}" | awk 'NR==2{print $4}')
  if [ "${free_kb}" -lt 10485760 ]; then
    echo "ERROR: <10 GB free on $(df -Pk "${MODELS_DIR}" | awk 'NR==2{print $6}') — refusing to download" >&2
    return 1
  fi

  mkdir -p "${MODELS_DIR}"
  if [ ! -f "${dest}" ]; then
    echo "Downloading ${url} ..."
    wget -c -q "${url}" -O "${dest}"
  fi
  echo "Downloaded: ${dest} ($(du -h "${dest}" | cut -f1))"

  echo "Benchmarking (n=64, reasoning off, no-mmap for true peak RSS) ..."
  CTX="${ctx}" EXTRA="${extra}" LLAMA_BIN="${LLAMA_BIN}" bash "${BENCH}" "${dest}"

  rm -f "${dest}"
  echo "Deleted ${dest} (disk rule: one model at a time)"
  echo "==================== END MODEL: ${key} ===================="
}

usage() {
  cat >&2 <<'EOF'
Usage: ./reproduce.sh {setup|e2b|e4b|ornith|12b|qwen3-4b|qwen3-8b|all}

  setup       Clone + build llama.cpp (needed once before any model).
  e2b         gemma-4-E2B-it      Q4_K_M
  e4b         gemma-4-E4B-it      Q4_K_M
  ornith      Ornith-1.0-9B       Q4_K_M  (qwen35, -c 2048)
  12b         gemma-4-12B-agentic Q4_K_M  (borderline)
  qwen3-4b    Qwen3-4B-Instruct-2507 Q4_K_M
  qwen3-8b    Qwen3-8B            Q4_K_M
  all         Run all six in sequence, deleting each model after its benchmark.

Each model is downloaded, benchmarked, then deleted immediately.
Running with no argument prints this usage and exits without doing anything.
EOF
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    setup)     setup ;;
    e2b|e4b|ornith|12b|qwen3-4b|qwen3-8b) run_model "${cmd}" ;;
    all)       setup; for k in e2b e4b ornith 12b qwen3-4b qwen3-8b; do run_model "$k"; done ;;
    "")        usage; exit 0 ;;
    *)         echo "Unknown target: ${cmd}" >&2; usage; exit 1 ;;
  esac
}

main "$@"