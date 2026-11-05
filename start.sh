#!/usr/bin/env bash
# Ripple-65B Preview on one RTX PRO 6000 Blackwell: NVFP4 weights + DFlash2 drafter on SGLang 0.5.19.
#
#   ./start.sh                dflash profile: NVFP4 + DFlash2, 16-token blocks (135.2 tok/s code-v1, 97.9 reasoning-v1)
#   ./start.sh nospec         NVFP4 without the drafter (24.0 / 24.0), the reference
#   ./start.sh tp2            the same on two cards, tensor parallel 2 (GPU=0,1 by default; 212.4 / 152.0)
#   ./start.sh --list         profiles and their measured numbers
#   ./start.sh dflash --print print the exact server command and exit
#
# Environment: GPU (default 0, tp2: 0,1), PORT (8000), HOST (127.0.0.1), CONTEXT (65536), MODELS_DIR (./models),
#              VENV (./.venv), CUDA_HOME (auto: newest /usr/local/cuda-13.*), HF_TOKEN (optional).
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

PROFILE="${1:-dflash}"
GPU="${GPU:-$([ "$PROFILE" = tp2 ] && echo 0,1 || echo 0)}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
CONTEXT="${CONTEXT:-65536}"
MODELS_DIR="$(mkdir -p "${MODELS_DIR:-./models}" && cd "${MODELS_DIR:-./models}" && pwd)"
VENV="${VENV:-$HERE/.venv}"
SERVED="ripple"

# Exactly as measured (2026-11-04/05). Revisions are pinned so results reproduce.
TARGET="ricursiveAI/Ripple-65B-Preview-NVFP4@99eea4477a001a41b86914e41feb4b2ed86105e5"
DRAFT="ricursiveAI/Ripple-65B-Preview-DFlash2@35609b3be3dd08a822b3ba1257c1ea5dd6f87632"
# serve.py imports Ricursive's SGLang plugin from the checkpoint; check it is the code we ran.
VENDOR_SHA256="007ec77d0d64d3007d68a256f726ff69318abf1a45ab5de08828979425e5b986  inference/local/model.py
19e432554ad48c57a1b67b989695ff4e35a44ac769821ad92ba229ecf5266ace  inference/local/ripple_format.py"

profiles() {  # profile | drafter | code-v1 | reasoning-v1 | cards
  cat <<'EOF'
dflash   DFlash2-16   135.2   97.9   1
nospec   -             24.0   24.0   1
tp2      DFlash2-16   212.4  152.0   2
EOF
}

if [ "$PROFILE" = "--list" ] || [ "$PROFILE" = "-h" ] || [ "$PROFILE" = "--help" ]; then
  printf '%-8s %-11s %8s %13s %6s\n' profile drafter code-v1 reasoning-v1 cards
  profiles | awk '{printf "%-8s %-11s %8s %13s %6s\n", $1,$2,$3,$4,$5}'
  echo "tok/s, batch 1, RTX PRO 6000 Blackwell Workstation Edition (600 W) cards. Sources in README.md."
  exit 0
fi
read -r _ DRAFTER CODE REASON CARDS < <(profiles | awk -v p="$PROFILE" '$1==p') || true
[ -n "${DRAFTER:-}" ] || { echo "Unknown profile '$PROFILE'. Try ./start.sh --list" >&2; exit 1; }

dir_of() { local r="${1%@*}"; echo "$MODELS_DIR/${r/\//__}"; }
T_DIR="$(dir_of "$TARGET")"; D_DIR="$(dir_of "$DRAFT")"

if [ -z "${CUDA_HOME:-}" ]; then
  CUDA_HOME="$(ls -d /usr/local/cuda-13.* 2>/dev/null | sort -V | tail -1 || true)"
fi
CMD=(env CUDA_VISIBLE_DEVICES="$GPU" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_HOME="$CUDA_HOME" PATH="$CUDA_HOME/bin:$PATH"
     RIPPLE_NOSPEC="$([ "$PROFILE" = nospec ] && echo 1 || echo 0)"
     "$VENV/bin/python" "$HERE/serve.py" "$T_DIR" "$D_DIR"
     --host "$HOST" --port "$PORT" --context-length "$CONTEXT" --swa-full-tokens-ratio 0.25)
if [ "$PROFILE" = tp2 ]; then
  # Two cards over PCIe: SGLang's custom all-reduce (P2P) is 2 ms/step faster than NCCL alone. Cards that also hold
  # other work have unequal free memory, which SGLang refuses unless the check is a warning; KV is sized from the smaller.
  CMD=("${CMD[0]}" SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false "${CMD[@]:1}" --tp-size 2)
fi

if [ "${2:-}" = "--print" ]; then  # show the exact launch, change nothing
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

# --- checks
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "Need a CUDA 13 toolkit (nvcc) for SGLang's JIT kernels; set CUDA_HOME. We used 13.2." >&2; exit 1; }
NEED=$([ "$CARDS" = 2 ] && echo 50000 || echo 84000)
for g in ${GPU//,/ }; do
  FREE=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
  [ "$FREE" -ge "$NEED" ] || { echo "GPU $g has ${FREE} MiB free; this profile wants about $((NEED / 1000)) GB per card." >&2; exit 1; }
done
if [ "$CARDS" = 2 ] && [ "$(echo "${GPU//,/ }" | wc -w)" != 2 ]; then echo "tp2 needs two cards: GPU=0,1" >&2; exit 1; fi
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then echo "Already running (pid $(cat .server.pid)); ./stop.sh first." >&2; exit 1; fi

# --- Python environment (uv): SGLang 0.5.19 on torch 2.13 + CUDA 13.0 wheels
if ! "$VENV/bin/python" -c "import sglang, torch; assert sglang.__version__ == '0.5.19' and torch.__version__ == '2.13.0+cu130'" 2>/dev/null; then
  command -v uv >/dev/null || { echo "Need uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
  echo ">> creating $VENV from requirements.lock (SGLang 0.5.19, torch 2.13.0+cu130; about 10 GB)"
  [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install --prerelease allow --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 -r "$HERE/requirements.lock"
fi

# --- weights at the pinned revisions
fetch() {  # repo@commit -> dir
  local repo="${1%@*}" rev="${1#*@}" dir; dir="$(dir_of "$1")"
  [ -f "$dir/.pinned-$rev" ] && return
  local hf="$VENV/bin/hf"; [ -x "$hf" ] || hf="$(command -v hf || true)"
  [ -n "$hf" ] || { echo "Need the Hugging Face CLI: pip install -U huggingface_hub" >&2; exit 1; }
  echo ">> downloading $repo at $rev into $dir"
  "$hf" download "$repo" --revision "$rev" --local-dir "$dir"
  touch "$dir/.pinned-$rev"
}
fetch "$TARGET"
[ "$PROFILE" = nospec ] || fetch "$DRAFT"
(cd "$T_DIR" && sha256sum -c --quiet <<<"$VENDOR_SHA256") || { echo "!! vendor plugin files differ from the measured ones" >&2; exit 1; }

# --- start
mkdir -p logs
setsid nohup "${CMD[@]}" > logs/server.log 2>&1 &
echo $! > .server.pid
echo ">> $PROFILE: NVFP4, drafter=$DRAFTER, $CARDS card(s) (measured $CODE tok/s code-v1, $REASON reasoning-v1), context $CONTEXT"
echo ">> waiting for http://$HOST:$PORT (first start JIT-compiles kernels: 2 to 10 minutes; log in logs/server.log)"
for _ in $(seq 1 180); do
  curl -sf -m 5 "http://$HOST:$PORT/health" >/dev/null && break
  kill -0 "$(cat .server.pid)" 2>/dev/null || { tail -40 logs/server.log; echo "!! server exited" >&2; exit 1; }
  sleep 10
done
curl -sf -m 5 "http://$HOST:$PORT/health" >/dev/null || { tail -40 logs/server.log; echo "!! not ready after 30 min" >&2; exit 1; }

REPLY=$(curl -s -m 300 "http://$HOST:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with one word: ready\"}],\"max_tokens\":512,\"temperature\":0}" \
  | python3 -c 'import sys,json; print((json.load(sys.stdin)["choices"][0]["message"]["content"] or "").strip())')
echo ">> smoke test: $REPLY"
cat <<EOF
>> serving $SERVED at http://$HOST:$PORT/v1 (OpenAI-compatible; tools and reasoning_content parsed)
   measure it:  python3 bench.py --url http://$HOST:$PORT
   stop it:     ./stop.sh
EOF
