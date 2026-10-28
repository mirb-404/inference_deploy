#!/usr/bin/env bash
# Build llama-server with CUDA for the H200 (Hopper, sm_90).
# Usage: scripts/build_llamacpp.sh [install_dir]   (default: ~/llama.cpp)
set -euo pipefail

DIR="${1:-$HOME/llama.cpp}"

if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone https://github.com/ggml-org/llama.cpp "$DIR"
fi

cmake -S "$DIR" -B "$DIR/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90-real \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$DIR/build" --config Release -j "$(nproc)" --target llama-server

echo
echo "Built: $DIR/build/bin/llama-server"
echo "Set [server].binary in config.toml to that path, or add $DIR/build/bin to PATH."
