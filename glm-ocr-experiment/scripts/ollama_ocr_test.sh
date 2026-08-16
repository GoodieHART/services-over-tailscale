#!/usr/bin/env bash
set -u

IMAGE_PATH="${1:-/tmp/sample_doc.png}"
PROMPT="${2:-Text Recognition:}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
export OLLAMA_HOME="${OLLAMA_HOME:-$HOME/.ollama}"

if [ -n "${OLLAMA_BIN_DIR:-}" ]; then
    export PATH="$OLLAMA_BIN_DIR:$PATH"
fi
if ! command -v ollama >/dev/null 2>&1; then
    if [ -x /tmp/ollama_install/bin/ollama ]; then
        export PATH="/tmp/ollama_install/bin:$PATH"
    else
        echo "ERROR: 'ollama' not found in PATH and /tmp/ollama_install/bin absent." >&2
        echo "Install via: curl -fsSL https://ollama.com/install.sh | sh" >&2
        echo "  (or, without sudo: download ollama-linux-<arch>.tar.zst, e.g. via reproduce.sh which picks the arch from uname -m)" >&2
        exit 1
    fi
fi

echo "Using ollama: $(command -v ollama)  (version $(ollama --version 2>/dev/null))"

if curl -fsS --max-time 3 "${OLLAMA_HOST}/" >/dev/null 2>&1; then
    echo "Ollama server already up at ${OLLAMA_HOST}"
else
    echo "Starting ollama serve in background..."
    nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
    for i in $(seq 1 60); do
        if curl -fsS --max-time 3 "${OLLAMA_HOST}/" >/dev/null 2>&1; then
            echo "Ollama server is up."
            break
        fi
        sleep 1
    done
    if ! curl -fsS --max-time 3 "${OLLAMA_HOST}/" >/dev/null 2>&1; then
        echo "ERROR: ollama server did not start. See /tmp/ollama_serve.log" >&2
        exit 1
    fi
fi

if ! ollama list 2>/dev/null | grep -q "glm-ocr"; then
    echo "Pulling glm-ocr (first run: ~2 GB download + cold load, can take 10+ min)..."
    ollama pull glm-ocr
else
    echo "glm-ocr already present."
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: image not found: $IMAGE_PATH" >&2
    exit 1
fi

echo
echo "=== Running OCR on: $IMAGE_PATH ==="
echo "=== Prompt: $PROMPT ==="
echo

python3 - "$IMAGE_PATH" "$PROMPT" "$OLLAMA_HOST" <<'PY'
import sys, json, time, base64, subprocess, threading, urllib.request

IMG, PROMPT, URL = sys.argv[1], sys.argv[2], sys.argv[3]

with open(IMG, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "glm-ocr",
    "messages": [{"role": "user", "content": PROMPT, "images": [b64]}],
    "stream": True,
    "options": {"temperature": 0},
}

def find_runner():
    try:
        out = subprocess.check_output(["pgrep", "-f", "llama-server"]).decode().split()
        return int(out[0]) if out else None
    except Exception:
        return None

rss_samples = []
stop = False
def sampler():
    while not stop:
        pid = find_runner()
        if pid:
            try:
                rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).decode().strip())
                rss_samples.append(rss)
            except Exception:
                pass
        time.sleep(0.2)

t = threading.Thread(target=sampler, daemon=True)
t.start()

req = urllib.request.Request(URL + "/api/chat",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
start = time.time()
first_tok = None
last_tok = None
full = []
try:
    with urllib.request.urlopen(req, timeout=900) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            c = obj.get("message", {}).get("content")
            if c:
                if first_tok is None:
                    first_tok = time.time()
                last_tok = time.time()
                full.append(c)
            if obj.get("done"):
                break
except Exception as e:
    print("ERROR:", e, file=sys.stderr)
stop = True
t.join()
end = time.time()

wall = end - start
ttft = (first_tok - start) if first_tok else None
gen = (last_tok - first_tok) if (first_tok and last_tok) else None
peak_kb = max(rss_samples) if rss_samples else None
peak_gb = (peak_kb / 1024 / 1024) if peak_kb else None

print("=== TIMING ===")
print(f"wall_total_s        = {wall:.3f}")
print(f"time_to_first_token = {ttft:.3f} s" if ttft is not None else "time_to_first_token = n/a")
print(f"generation_s        = {gen:.3f} s" if gen is not None else "generation_s = n/a")
print(f"output_tokens       = {len(full)}")
print(f"peak_rss_gb         = {peak_gb:.3f}" if peak_gb is not None else "peak_rss_gb = n/a")
print("=== OCR OUTPUT ===")
print("".join(full))
PY

echo
echo "=== Done. (Server left running; stop with: pkill -f 'ollama serve') ==="
