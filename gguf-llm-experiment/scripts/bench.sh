#!/usr/bin/env bash
set -u

[ -n "${1:-}" ] || { echo "usage: bench.sh <model.gguf>" >&2; exit 2; }
MODEL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(dirname "${SCRIPT_DIR}")"

if [ -n "${LLAMA_BIN:-}" ]; then
  BIN="$LLAMA_BIN"
elif [ -x "${BUNDLE_DIR}/llama.cpp/build/bin/llama-cli" ]; then
  BIN="${BUNDLE_DIR}/llama.cpp/build/bin/llama-cli"
else
  BIN="$(command -v llama-cli 2>/dev/null || true)"
fi

# llama.cpp bakes RUNPATH at build time; moved build dirs fail to load libs without this
BIN_DIR="$(dirname "${BIN}")"
if compgen -G "${BIN_DIR}"/*.so >/dev/null 2>&1; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${BIN_DIR}:"*) ;;
    *) export LD_LIBRARY_PATH="${BIN_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi

PROMPT="$(cat "${BUNDLE_DIR}/prompt.txt")"
LOG="/tmp/bench_$$.log"

[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "MISSING BIN: $BIN (set LLAMA_BIN or run reproduce.sh setup)"; exit 2; }
[ -f "$MODEL" ] || { echo "MISSING MODEL: $MODEL"; exit 2; }

echo "== bench: $(basename "$MODEL")"
echo "== bin:   $BIN"
echo "== flags: -n ${N:-64} -t 2 --temp 0.7 --no-display-prompt --no-mmap ${REASONING:---reasoning off} --single-turn ${EXTRA:-}"

# --no-mmap forces full residency: VmHWM is the true peak, not the lazy-mmap under-report
"$BIN" -m "$MODEL" -p "$PROMPT" -n ${N:-64} -t 2 --temp 0.7 --no-display-prompt --no-mmap ${REASONING:---reasoning off} --single-turn ${EXTRA:-} > "$LOG" 2>&1 &
PID=$!
MAX=0
while kill -0 "$PID" 2>/dev/null; do
  r=$(awk '/^VmHWM:/{print $2}' /proc/$PID/status 2>/dev/null)
  if [ -n "${r:-}" ]; then
    if [ "$r" -gt "$MAX" ] 2>/dev/null; then MAX=$r; fi
  fi
  sleep 0.2
done
wait "$PID"; RC=$?
echo "EXIT_CODE=$RC"
echo "PEAK_RSS_KB=$MAX"
echo "PEAK_RSS_GB=$(awk "BEGIN{printf \"%.2f\", $MAX/1048576}")"
echo "--- llama.cpp timing ---"
grep -iE "tokens per second|t/s|eval time|prompt eval|load time" "$LOG" || true
echo "--- generated text ---"
tail -n 18 "$LOG"
rm -f "$LOG"
exit "$RC"