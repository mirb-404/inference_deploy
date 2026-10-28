# inference-deploy

Serves Qwen3.8-27B (GGUF) with llama.cpp on one NVIDIA H200 (141 GB). The config follows
[this llama.cpp setup](https://dev.to/dmitryame/inside-my-llamacpp-setup-tuning-qwen-38-27b-for-512k-context-o3c).
The server is exposed through a free Cloudflare quick tunnel as an OpenAI-compatible `/v1` API.

## Setup (Linux GPU box)

```bash
uv run serve.py
```

If `llama-server` is not on PATH, the first run clones and builds llama.cpp with CUDA into `./llama.cpp` (~10-20 min). It needs `git`, `g++` and the CUDA toolkit (`nvcc`); `uv sync` supplies `cmake` and `ninja`. To use an existing build instead, set `[server] binary`.

## Run

```bash
uv run serve.py               # llama-server + Cloudflare tunnel
uv run serve.py --no-tunnel   # local only
uv run serve.py --dry-run     # print the llama-server command
uv run serve.py --config other.toml
```

On first start the GGUF downloads to the llama.cpp cache (set `LLAMA_CACHE` in `[env]` to move it).
After `/health` is ready, the script prints:

```
Public  base URL : https://<random>.trycloudflare.com/v1
Local   base URL : http://127.0.0.1:8080/v1
Model            : qwen3.8-27b
API key          : <key>
```

Use it from any OpenAI client:

```python
from openai import OpenAI
client = OpenAI(base_url="https://<random>.trycloudflare.com/v1", api_key="<key>")
client.chat.completions.create(model="qwen3.8-27b", messages=[{"role": "user", "content": "Hi"}])
```

## Editing parameters

All settings are in `config.toml`:

| Change | Where |
|---|---|
| Model / quantization | `[model] hf_repo`, `quant` (or `path` for a local .gguf) |
| Context, KV cache, batch, parallel slots, sampling | `[llama]` |
| Port, API key, binary path | `[server]` |
| Tunnel on/off | `[tunnel] enabled` |
| GPU selection, cache dir, HF token | `[env]` |

Each `[llama]` key is a llama-server long flag without the `--` prefix. Any flag from
`llama-server --help` works: `true` is a switch, `false` drops the line, and a list repeats the flag.

When switching models, also check:
- `override-kv`: the prefix must match the GGUF's `general.architecture` (Qwen3.8 is `qwen35`). It is only needed for YaRN context extension.
- `rope-scaling` / `rope-scale` / `yarn-orig-ctx`: these are off by default. Enable them only for context beyond the model's native window.
- `spec-type = "draft-mtp"`: only works if the GGUF has MTP heads. Set it to `false` otherwise.

## Speed (target ~200 tok/s on code)

Full research: `reports/Qwen 27B H200 decode speed.md`.

- Decode is memory-bandwidth bound, so the quant decides speed. Plain decode on the H200 is projected at ~175-205 tok/s for UD-Q4_K_XL and ~105-120 tok/s for Q8_0.
- The built-in MTP head (`spec-type = "draft-mtp"`, `spec-draft-n-max = 3`) gives ~1.6-2.4x on code at `parallel = 1`.
- MTP gains vanish by ~4 concurrent streams. For many agents at once, raise `parallel` and set `spec-type = false`.
- Use llama.cpp build >= b10450. Older CUDA builds produce garbage on Qwen3.8.
- No public H200 llama.cpp benchmark exists for this model. These numbers are extrapolated from H100 and RTX 5090 runs, so measure your own:
  `llama-bench -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL -fa 1 -ngl 999`.

## Tunnel

- **Quick tunnel** (default, no account): the URL changes every run, and the tunnel allows at most 200 in-flight requests. Cloudflare documents that quick tunnels **do not support SSE**, so streamed responses may stall.
- **Named tunnel** (free account + a domain on Cloudflare): supports streaming and keeps a fixed URL. Set `[tunnel] token` and `hostname`, and the same `uv run serve.py` uses it.
- Both tunnel types cut non-streamed responses off after 125 s (Cloudflare 524 error). Cap `max_tokens` or `reasoning-budget` to stay under that.

## Notes

- Keep `require_api_key = true`. Without it, anyone who finds the URL can use your GPU.
- `cloudflared` is auto-downloaded to `./bin` on Linux/Windows if it is not on PATH.
