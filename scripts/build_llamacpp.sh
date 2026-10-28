#!/usr/bin/env bash
# Build llama-server with CUDA for the H200 (Hopper, sm_90).
# Usage: scripts/build_llamacpp.sh [install_dir]   (default: ./llama.cpp next to this repo)
# serve.py runs this automatically when llama-server is missing ([server] auto_build = true).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${1:-$REPO_ROOT/llama.cpp}"

missing=0
need() { command -v "$1" >/dev/null 2>&1 || { echo "[build] missing: $1 -- $2"; missing=1; }; }

# nvcc is often installed but not on PATH.
if ! command -v nvcc >/dev/null 2>&1 && [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

need git   "apt install git  (or: conda install -c conda-forge git)"
need cmake "run via 'uv run serve.py' (uv sync installs cmake) or: pip install cmake"
need g++   "apt install build-essential  (or: conda install -c conda-forge gxx_linux-64)"
need nvcc  "install the CUDA toolkit: apt install nvidia-cuda-toolkit, or conda install -c nvidia cuda-toolkit"
[ "$missing" -eq 0 ] || exit 1

if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$DIR"
fi

generator=()
command -v ninja >/dev/null 2>&1 && generator=(-G Ninja)

cmake -S "$DIR" -B "$DIR/build" "${generator[@]}" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90-real \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$DIR/build" --config Release -j "$(nproc)" --target llama-server llama-bench

echo
echo "Built: $DIR/build/bin/llama-server"
