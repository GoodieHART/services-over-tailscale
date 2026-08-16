#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_IMG="${BUNDLE_DIR}/sample_doc.png"
SCRIPTS_DIR="${BUNDLE_DIR}/scripts"

if [ ! -f "${SAMPLE_IMG}" ]; then
  echo "ERROR: sample image not found: ${SAMPLE_IMG}" >&2
  exit 1
fi

run_pytorch() {
  echo "==================== PYTORCH PATH ===================="
  echo "Installing CPU-only torch + transformers and running the fp32 benchmark."
  local venv="${BUNDLE_DIR}/glmocr_venv"
  local bench_py="${BUNDLE_DIR}/glmocr_bench_fp32.py"

  python3 -m venv "${venv}"
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"

  pip install --upgrade pip

  # aarch64 default pip torch pulls a CUDA wheel; CPU wheel required
  pip install torch --index-url https://download.pytorch.org/whl/cpu

  # dtype="auto" resolves to bfloat16 here and is pathologically slow (killed >86 min); force float32
  pip install "transformers>=5.0.0" pillow

  python3 - "${SCRIPTS_DIR}/glm_ocr_test.py" "${bench_py}" <<'PY'
import sys
src = open(sys.argv[1]).read()
src = src.replace(
    'def load_model(model_id: str = MODEL_ID, dtype: str = "auto"):',
    'def load_model(model_id: str = MODEL_ID, dtype: str = "float32"):',
)
open(sys.argv[2], "w").write(src)
PY

  # CONSTRAINT 3: torch.set_num_threads(2) (== physical cores) beats 4.
  GLM_OCR_THREADS=2 python3 "${bench_py}" "${SAMPLE_IMG}" --threads 2
  echo "==================== END PYTORCH ===================="
}

run_ollama() {
  echo "==================== OLLAMA PATH ===================="
  echo "No-sudo extract install of Ollama, pull GLM-OCR (F16), run benchmark."
  local install_dir="${BUNDLE_DIR}/ollama_install"
  local arch ollama_url ollama_tar

  # tarball + extract + ~/.ollama model need ~6 GB; refuse on short disk
  for d in "${BUNDLE_DIR}" "$HOME"; do
    local free_kb
    free_kb=$(df -Pk "$d" | awk 'NR==2{print $4}')
    if [ "${free_kb}" -lt 10485760 ]; then
      echo "ERROR: <10 GB free on $(df -Pk "$d" | awk 'NR==2{print $6}') — refusing ollama path" >&2
      return 1
    fi
  done

  # system installer needs sudo; extract tarball locally instead
  arch="$(uname -m)"
  case "${arch}" in
    aarch64|arm64) ollama_url="https://ollama.com/download/ollama-linux-arm64.tar.zst" ;;
    x86_64|amd64)  ollama_url="https://ollama.com/download/ollama-linux-amd64.tar.zst" ;;
    *) echo "ERROR: unsupported architecture '${arch}' for Ollama" >&2; return 1 ;;
  esac
  ollama_tar="/tmp/ollama-linux-${ollama_url#*ollama-linux-}"

  if [ ! -x "${install_dir}/bin/ollama" ]; then
    echo "Downloading ${ollama_url} ..."
    curl -fsSL -o "${ollama_tar}" "${ollama_url}"
    mkdir -p "${install_dir}"
    zstd -d -c "${ollama_tar}" | tar -xf - -C "${install_dir}"
    rm -f "${ollama_tar}"
  fi

  export PATH="${install_dir}/bin:${PATH}"
  export OLLAMA_HOME="${OLLAMA_HOME:-$HOME/.ollama}"

  # starts the server if needed and pulls glm-ocr if missing
  bash "${SCRIPTS_DIR}/ollama_ocr_test.sh" "${SAMPLE_IMG}" "Text Recognition:"
  echo "==================== END OLLAMA ===================="
}

run_sglang() {
  echo "==================== SGLANG PATH ===================="
  local arch
  arch="$(uname -m)"

  # no CPU-only aarch64 torchvision wheel (imported at load time); CPU engine is Intel/AMX-only
  if [ "${arch}" = "aarch64" ] || [ "${arch}" = "arm64" ]; then
    echo "SKIPPED: SGLang not viable on aarch64/arm64 CPU" \
         "(no CPU torchvision wheel; Intel/AMX-only CPU engine)"
    echo "==================== END SGLANG ===================="
    return 0
  fi

  if [ "${arch}" != "x86_64" ] && [ "${arch}" != "amd64" ]; then
    echo "SKIPPED: SGLang CPU install only implemented for x86_64 in this bundle (arch=${arch})"
    echo "==================== END SGLANG ===================="
    return 0
  fi

  echo "x86_64 detected — installing SGLang (CPU engine) and running the benchmark."
  local venv="${BUNDLE_DIR}/sglang_venv"
  python3 -m venv "${venv}"
  # shellcheck disable=SC1091
  source "${venv}/bin/activate"
  pip install --upgrade pip
  # CPU-only torch (avoid a CUDA pull on x86_64 too)
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  git clone --depth 1 https://github.com/sgl-project/sglang.git
  ( cd sglang/python && cp pyproject_cpu.toml pyproject.toml && pip install . )

  SGLANG_USE_CPU_ENGINE=1 python -m sglang.launch_server \
      --model zai-org/GLM-OCR --device cpu --port 30000 &
  local srv_pid=$!
  for _ in $(seq 1 60); do
    curl -fsS http://localhost:30000/health >/dev/null 2>&1 && break
    sleep 5
  done

  GLM_OCR_THREADS=2 python3 - "${SAMPLE_IMG}" <<'PY'
import sys, time, base64, json, urllib.request
img_path = sys.argv[1]
b64 = base64.b64encode(open(img_path, "rb").read()).decode()
payload = json.dumps({
    "model": "zai-org/GLM-OCR",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Text Recognition:"},
    ]}],
    "max_tokens": 2048,
}).encode()
t0 = time.time()
req = urllib.request.Request(
    "http://localhost:30000/v1/chat/completions",
    data=payload, headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req))
print("[sglang] infer %.2fs" % (time.time() - t0))
print(resp["choices"][0]["message"]["content"])
PY

  kill "${srv_pid}" 2>/dev/null || true
  echo "==================== END SGLANG ===================="
}

usage() {
  cat >&2 <<'EOF'
Usage: ./reproduce.sh {pytorch|ollama|sglang|all}

  pytorch   Install CPU-only torch + transformers and run the fp32 benchmark.
  ollama    No-sudo extract install of Ollama, pull GLM-OCR, run F16 benchmark.
  sglang    Skips on arm64 (not viable); attempts a real CPU install on x86_64.
  all       Run all three in sequence (pass explicitly — NOT the default).

Running with no argument prints this usage and exits without doing anything.
EOF
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    pytorch) run_pytorch ;;
    ollama)  run_ollama ;;
    sglang)  run_sglang ;;
    all)     run_pytorch; run_ollama; run_sglang ;;
    "")      usage; exit 0 ;;
    *)       echo "Unknown target: ${cmd}" >&2; usage; exit 1 ;;
  esac
}

main "$@"
