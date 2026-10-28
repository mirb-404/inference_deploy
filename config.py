# Everything you tune lives in this file. Restart `uv run serve.py` after editing.
# Based on https://dev.to/dmitryame/inside-my-llamacpp-setup-tuning-qwen-38-27b-for-512k-context-o3c
# and retuned for a single NVIDIA H200 (141 GB VRAM): near-lossless quality, full native context,
# ~200+ tok/s single-stream on code via MTP. Research: reports/Qwen 27B H200 decode speed.md
#
# ============================== VRAM BUDGET (H200 = 141 GB) ==============================
# Total ~= weights + MTP head + KV cache + DeltaNet state + compute buffer + CUDA overhead
#
#   Default below:  UD-Q6_K_XL 25.3 + MTP 1.4 + KV f16 @262K 16.0 + state ~0.2
#                   + compute ~4-6 + CUDA ~1  ~= 48-50 GB   (~90 GB free)
#
# Weights by quant (exact file sizes) and projected H200 decode tok/s (plain / with MTP on code):
#   "UD-Q4_K_XL" 17.6 GB   ~175-205 / ~250-350   small quality loss
#   "UD-Q5_K_XL" 20.9 GB   ~160-185 / ~250-300   very small loss
#   "UD-Q6_K_XL" 25.3 GB   ~140-160 / ~220-280   near-lossless  <- default
#   "Q8_0"       29.1 GB   ~105-120 / ~180-210   ~lossless, borderline for 200 tok/s
#   "UD-Q8_K_XL" 31.5 GB   ~100-115 / ~170-200
#   "BF16"       54.7 GB   ~56-65   / ~100-120   reference quality, too slow for 200
#
# KV cache (only 16 of 64 layers are full attention, so it is small):
#   f16  = 64 KiB/token -> 128K: 8 GB | 256K: 16 GB | 512K: 32 GB | 1M: 64 GB
#   q8_0 = 34 KiB/token -> 128K: 4.3 GB | 256K: 8.5 GB | 512K: 17 GB | 1M: 34 GB
#   With kv-unified = True the ctx-size pool is shared by all slots (cost does not grow with parallel).
#   With kv-unified = False each slot gets ctx-size / parallel tokens.
#
# Per extra parallel slot: ~0.15-0.2 GB DeltaNet recurrent state (estimate).
# Compute buffer scales with ubatch-size: 4096 ~4-6 GB, 2048 ~2-3 GB, 8192 ~8-12 GB (estimates).
# Projections are extrapolated (no public H200 llama.cpp runs). Check real usage with nvidia-smi.
# ==========================================================================================

MODEL = {
    # Hugging Face GGUF repo + quantization tag (llama-server downloads it with -hf repo:quant).
    "hf_repo": "unsloth/Qwen3.8-27B-GGUF",
    "quant": "UD-Q6_K_XL",  # see VRAM BUDGET table above for options
    # Local .gguf file instead of downloading. When set, hf_repo/quant are ignored.
    "path": "",
    # Model name reported by /v1/models and accepted in the "model" field of requests.
    "alias": "qwen3.8-27b",
}

SERVER = {
    # llama-server binary: name on PATH or full path. If not found, ./llama.cpp/build/bin/llama-server
    # is used; with auto_build = True it is cloned + built with CUDA on first run (~10-20 min,
    # needs git, g++, nvcc; ~2 GB disk). Needs a build >= b10450 (older builds garble Qwen3.8).
    "binary": "llama-server",
    "auto_build": True,
    "host": "127.0.0.1",
    "port": 8080,
    # The tunnel makes the server public, so requests need "Authorization: Bearer <key>".
    # Empty = a random key is generated and printed on every start. Set a fixed value to keep it.
    "api_key": "",
    # False disables auth entirely. Anyone with the tunnel URL can then use your GPU.
    "require_api_key": True,
    # Seconds to wait for model download + load before giving up.
    "startup_timeout": 3600,
}

TUNNEL = {
    # Started first on every run; the public /v1 link prints before the model finishes loading.
    "enabled": True,
    # cloudflared on PATH, or auto-downloaded to ./bin on Linux/Windows when missing.
    "binary": "cloudflared",
    # Empty token = free quick tunnel (random https://*.trycloudflare.com URL, no account).
    #   Cloudflare docs say quick tunnels do not support SSE, but streaming was verified working
    #   (2026-10-28, 1200 chunks arrived live). Max 200 in-flight requests, 125 s timeout for
    #   non-streamed replies, URL changes every run.
    # Set token + hostname for a named tunnel (free Cloudflare account + a domain on Cloudflare):
    #   Zero Trust dashboard -> Networks -> Tunnels -> Create, add a public hostname pointing to
    #   http://127.0.0.1:<port>, then paste the tunnel token and that hostname here.
    "token": "",
    "hostname": "",  # e.g. "llm.example.com"
}

ENV = {
    # Extra environment variables for llama-server.
    # "CUDA_VISIBLE_DEVICES": "0",
    # "LLAMA_CACHE": "/data/llama-cache",  # where -hf downloads go (needs disk = weights + 1.4 GB MTP)
    # "HF_TOKEN": "hf_...",                # only for gated/private repos
}

LLAMA = {
    # Raw llama-server flags. Key = long flag name without "--".
    #   value          -> --key value
    #   True           -> --key           (switch)
    #   False / None   -> flag omitted    (quick way to disable a line)
    #   [a, b]         -> --key a --key b (repeatable flags)
    # Any flag from `llama-server --help` can be added here.

    # --- Speculative decoding: built-in MTP head (biggest speed lever, ~1.6-2.4x on code) ---
    # VRAM: +1.4 GB. Output quality is unchanged in principle (drafts are verified by the main model).
    # Gains vanish by ~4 concurrent streams; set spec-type = False when running many agents.
    "spec-type": "draft-mtp",
    "spec-draft-n-max": 3,  # 2-4 is the sweet spot; 5+ gets slower

    # --- GPU ---
    "n-gpu-layers": 999,  # all layers on the H200
    "flash-attn": "on",

    # --- Context ---
    # Full native window. VRAM: f16 KV 16 GB at 262144 (see table).
    # Known issue: llama.cpp #27756 reports instant EOS beyond ~130K on some builds.
    # If you hit it, drop to 131072 (8 GB) until your build has the fix.
    "ctx-size": 262144,
    # 512K via YaRN (+16 GB KV at f16). Qwen advises YaRN only when needed: it slightly hurts
    # short-context quality. To enable: ctx-size = 524288 and swap in the commented values.
    "override-kv": False,  # ["qwen35.context_length=int:524288"]  (arch key is qwen35)
    "rope-scaling": False,  # "yarn"
    "rope-scale": False,  # 2   (4 for 1M context, +48 GB KV at f16)
    "yarn-orig-ctx": False,  # 262144

    # --- KV cache ---
    # f16 = full precision (VRAM is plentiful). q8_0 halves KV memory, ~2% faster, tiny quality loss.
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "kv-offload": True,

    # --- Slots ---
    # 1 = max single-stream speed (MTP works best). For several agents raise it; each slot adds
    # ~0.2 GB state. kv-unified = True: all slots share one ctx-size pool (any single request can
    # use up to the full 262K). kv-unified = False: each slot is capped at ctx-size / parallel.
    "parallel": 1,
    "kv-unified": True,

    # --- Throughput ---
    "batch-size": 16384,  # logical batch (prefill)
    "ubatch-size": 4096,  # physical micro-batch; compute buffer ~4-6 GB (see table)
    "threads": 16,  # CPU threads; little effect with every layer on GPU
    "threads-batch": 16,
    "load-mode": "auto",  # none / auto / mmap / mlock / mmap+mlock / dio

    # --- Chat / tool calling ---
    "jinja": True,  # use the GGUF's chat template (needed for tool calls)
    "no-mmproj": True,  # skip the vision encoder (saves ~0.9 GB); False to accept images
    # Bound thinking so non-streamed replies finish before Cloudflare's 125 s timeout.
    # "reasoning-budget": 8192,

    # --- Default sampling (Qwen3.8 card, thinking mode; clients can override per request) ---
    # Non-thinking: temp 0.7, top-p 0.8, top-k 20, presence-penalty 1.5
    "temp": 1.0,
    "top-p": 0.95,
    "top-k": 20,
    "min-p": 0.0,
}
