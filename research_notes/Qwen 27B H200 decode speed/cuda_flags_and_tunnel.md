# llama.cpp CUDA build + llama-server flags for single H200, and Cloudflare quick tunnel limits

Note: all llama.cpp facts come from the `master` branch fetched on 2026-10-28. Flags change often. Re-check `llama-server --help` on the exact build you deploy.

## (A1) CMake build flags for Hopper (sm_90)

### Takeaway
Use `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90` (or `90-real`). CUDA graphs and FlashAttention kernels are ON by default in a standalone llama.cpp build. `GGML_CUDA_F16` and `GGML_CUDA_PEER_MAX_BATCH_SIZE` no longer exist. `GGML_CUDA_FA_ALL_QUANTS` is deprecated and only needed for mixed K/V cache types. NCCL only matters for multi-GPU.

### Cited Findings
- Basic build: `cmake -B build -DGGML_CUDA=ON` then `cmake --build build --config Release`. Set arch with e.g. `-DCMAKE_CUDA_ARCHITECTURES="86;89"`. — [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- Default arch is `native` when `GGML_NATIVE` is on and CUDA >= 11.6. Otherwise a fixed list is built, and Hopper appears only as `90-virtual` (PTX, JIT on first run). The file never emits `90a`. The `a` suffix rewrite only applies to 120-129 (Blackwell). The comment labels 90 as "Hopper H100/200", CUDA >= 11.8. — [ggml/src/ggml-cuda/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt)
- `XX-real` compiles device code for that arch. `XX-virtual` compiles PTX and JITs on first run. — [ggml-cuda/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt)
- Current `GGML_CUDA_*` options and defaults: `FORCE_MMQ` OFF, `FORCE_CUBLAS` OFF, `NO_PEER_COPY` OFF, `NO_VMM` OFF, `FA` ON ("compile ggml FlashAttention CUDA kernels"), `FA_ALL_QUANTS` OFF, `FA_QUANTS` = `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`, `GRAPHS` = `GGML_CUDA_GRAPHS_DEFAULT`, `NCCL` ON, `COMPRESSION_MODE` "size" (none/speed/balance/size, CUDA 12.8+), `CCCL_VERSION` empty. `GGML_CUDA_F16` and `GGML_CUDA_PEER_MAX_BATCH_SIZE` are not defined. — [ggml/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt)
- Root llama.cpp CMakeLists: `if (NOT DEFINED GGML_CUDA_GRAPHS) set(GGML_CUDA_GRAPHS_DEFAULT ON)`. So graphs are ON in a normal llama.cpp build. Graphs add the `GGML_CUDA_USE_GRAPHS` define. — [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/CMakeLists.txt); [ggml-cuda/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt)
- `GGML_CUDA_FORCE_MMQ`: custom quantized matmul kernels instead of FP16 cuBLAS. Lower VRAM. Worse large-batch speed. `GGML_CUDA_FORCE_CUBLAS`: FP16 cuBLAS for quantized models. May overflow numerically and use more memory. "May speed up prompt processing on newer datacenter GPUs." — [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- `GGML_CUDA_FA_QUANTS` picks which K/V type pairs get compiled FA kernels. Uncompiled pairs fall back to f16-f16 with a warning. `GGML_CUDA_FA_ALL_QUANTS` is a deprecated alias for `GGML_CUDA_FA_QUANTS=all` and takes much longer to compile. — [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- NCCL: if found, links `NCCL::NCCL` and defines `GGML_USE_NCCL`. If not found, it prints a warning that multi-GPU performance will be suboptimal. — [ggml-cuda/CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt)
- `LLAMA_CURL` is deprecated (no longer an option). `LLAMA_OPENSSL` defaults ON. `LLAMA_BUILD_SERVER` defaults to standalone. — [CMakeLists.txt](https://github.com/ggml-org/llama.cpp/blob/master/CMakeLists.txt)
- Runtime env vars: `CUDA_VISIBLE_DEVICES`; `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (Linux: spill to RAM instead of crashing); `GGML_CUDA_P2P`; `CUDA_SCALE_LAUNCH_QUEUES=4x` (mainly helps multi-GPU pipeline parallel); `GGML_CUDA_CUBLAS_COMPUTE_TYPE` (auto/f16/bf16/f32); `GGML_CUDA_MMQ_PREC` (NVFP4/MXFP4 only). The doc does not list `GGML_CUDA_DISABLE_GRAPHS`. — [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

### Inferences
- Recommended build for one H200:
  ```bash
  cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90-real \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
  cmake --build build --config Release -j --target llama-server
  ```
  `90-real` skips the PTX JIT on first launch. `native` also works when building on the H200 box itself.
- `90a` gives no benefit here: ggml does not target the arch-specific (wgmma/TMA) features, since its build file never emits `90a`. Leave it out.
- Do not set `FORCE_MMQ` or `FORCE_CUBLAS` for decode. Decode at batch 1 is memory-bandwidth bound and uses the default mmvq/mmq kernels. `FORCE_CUBLAS` only targets prefill and risks overflow. Benchmark it with `llama-bench` before you adopt it.
- Only set `GGML_CUDA_FA_QUANTS` (or `FA_ALL_QUANTS`) if you use mixed K/V types, such as `-ctk q8_0 -ctv f16`. Matched q8_0/q8_0 and f16/f16 are already compiled.
- `GGML_CUDA_NCCL` is irrelevant on one GPU. The warning if NCCL is missing is harmless.

### Gaps
- No primary source found that measures decode tokens/s for `90` vs `90a`, or for FORCE_CUBLAS vs default, on H200.
- The runtime env var for turning off CUDA graphs (historically `GGML_CUDA_DISABLE_GRAPHS`) is not in current build.md. It was not verified in source.

## (A2) llama-server runtime flags

### Takeaway
Current master renamed several flags. mmap/mlock/direct-io are now values of `--load-mode`. `--draft-max` is removed in favour of `--spec-*`. `--fit` defaults ON and can silently change unset args such as ctx. FA, cont-batching, prompt cache, jinja and slots are all ON or auto by default. Set `-c`, `-np`, `-fa on` and `-ngl all` explicitly.

### Cited Findings (all quotes from [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) unless noted)
- `-fa, --flash-attn [on|off|auto]`, default `auto`.
- `-ngl, --gpu-layers N`: "exact number, 'auto', or 'all' (default: auto)".
- `-b, --batch-size` default 2048 (logical). `-ub, --ubatch-size` default 512 (physical).
- `-c, --ctx-size`: "default: 0, 0 = loaded from model".
- `-fit, --fit [on|off]`: "adjust unset arguments to fit in device memory ... default: 'on'". `--fit-ctx`: "minimum ctx size that can be set by --fit option, default: 4096". `--fit-target`: per-device margin.
- `-np, --parallel N`: "number of server slots (default: -1, -1 = auto)".
- `-kvu, --kv-unified / -no-kvu`: "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)".
- `--kv-unified-per-slot`: "context limit per parallel slot (default: unset, behavior unchanged)". The README says that when this flag is set without `-c`, the shared KV pool is sized to n_parallel x N. Its `/slots` example shows `n_ctx` 65536 for each of two slots.
- `--cache-prompt`: "whether to enable prompt caching (default: enabled)".
- `--cache-reuse N`: "min chunk size to attempt reusing from the cache via KV shifting, requires prompt caching to be enabled (default: 0)".
- `-cram, --cache-ram N`: "maximum cache size in MiB (default: 8192 ... -1 - no limit, 0 - disable)".
- `-ctxcp, --ctx-checkpoints`: "max number of context checkpoints to create per slot (default: 32)".
- `--swa-full`: "use full-size SWA cache (default: false)".
- `-ctk / -ctv`: KV cache types, default f16.
- `-lm, --load-mode MODE` (default auto). Values: `none`, `auto`, `mmap`, `mlock`, `mmap+mlock`, `dio` ("use DirectIO if available"). `--no-mmap`, `--mlock` and `--direct-io` are NOT listed as standalone flags in current master.
- `-kvo, --kv-offload / -nkvo, --no-kv-offload`: default enabled.
- `-t, --threads` default -1. `--threads-http` default -1. `--poll` default 50. `--prio` default 0.
- `-cb, --cont-batching / -nocb`: default enabled.
- `--metrics`: Prometheus endpoint, default disabled. `--slots / --no-slots`: default enabled.
- `--context-shift / --no-context-shift`: default disabled.
- `--timeout`: "server read/write timeout in seconds (default: 3600)". `--api-key`: comma-separated list allowed.
- `--port` is listed as "default: 9931" in the fetched README. This differs from the historic 8080 default. Set `--port` explicitly.
- `--spec-type`: "comma-separated list of types of speculative decoding". Values: none, draft-simple, draft-eagle3, draft-mtp, draft-dflash, draft-dspark, ngram-simple, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache. `--spec-default`: "enable default speculative decoding config". Draft model is `--spec-draft-model` (`-md`). `--spec-draft-n-max` default 3. `--draft-max`: "the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max".
- `-sm, --split-mode`: layer (default), row, tensor, none.

### Inferences
- Recommended single-H200 launch for a coding-agent workload (adjust `-c`/`-np` to VRAM):
  ```bash
  ./build/bin/llama-server -m model.gguf \
    --host 127.0.0.1 --port 8080 \
    -ngl all -fa on -c 131072 -np 2 --kv-unified \
    -b 2048 -ub 2048 \
    -ctk q8_0 -ctv q8_0 \
    --cache-reuse 256 -cram 16384 \
    --jinja --reasoning-format deepseek \
    --metrics --no-context-shift --fit off \
    --load-mode mmap+mlock --api-key "$KEY"
  ```
- `-ngl all` plus `-fa on` keeps all compute on GPU. Once the model is fully offloaded, `--threads` mostly affects tokenization and sampling overhead. The default is fine.
- `-ub` raises prefill throughput at the cost of a larger compute buffer. On 141 GB VRAM, `-ub 1024`-`2048` is affordable. Decode speed does not depend on `-b`/`-ub`.
- With `--kv-unified` on, all slots share one KV pool of size `-c`. Without it, the context is split per slot. Use `--kv-unified-per-slot` to cap per-slot context. This follows from the flag descriptions above, not from a direct statement in the docs.
- KV q8_0 halves KV memory and bandwidth versus f16. It is compiled in the default FA kernels. For long-context decode it often helps or is neutral. At short context, f16 may be slightly faster. Benchmark with `llama-bench -fa 1 -ctk ... -ctv ...`.
- `--swa-full` only matters for SWA models (Gemma-style). Leave it off for Qwen dense models unless you need cache reuse across the window.
- `--context-shift` stays off for agents: silently dropping context corrupts tool-call history.
- `--fit off` plus an explicit `-c` prevents silent ctx changes. The alternative is to keep `--fit on` and read the startup log.

### Gaps
- No primary benchmark found for q8_0 vs f16 KV decode speed on Hopper with FA.
- The exact per-slot split formula when `--kv-unified` is off was not quoted from source code (common/arg.cpp / server-context.cpp were not fetched).
- `--port` default 9931 looks unusual. It may be a README typo or a recent change. Verify with `--help`.

## (A3) llama-server API features for coding agents

### Takeaway
Current master serves OpenAI Chat Completions, OpenAI Responses (`/v1/responses`), and Anthropic Messages (`/v1/messages`, with streaming and tool use), plus token-count endpoints. Tool use needs `--jinja`, which is now ON by default.

### Cited Findings (from [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md))
- `--jinja / --no-jinja`: "default: enabled".
- `--reasoning-format`: values `none` ("leaves thoughts unparsed in message.content"), `deepseek`, `deepseek-legacy`; default `auto`.
- `-rea, --reasoning [on|off|auto]`: default auto (detected from template).
- `--reasoning-budget N`: "-1 for unrestricted, 0 for immediate end, N>0 for token budget (default: -1)".
- `--chat-template-kwargs STRING`: JSON object passed to the template parser. Example use: Qwen `{"enable_thinking":false}`.
- Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses` ("converted internally to a Chat Completions request"), `POST /v1/responses/input_tokens`, `POST /v1/chat/completions/input_tokens`, `POST /v1/messages` (Anthropic-compatible, streaming), `POST /v1/messages/count_tokens`, `POST /v1/embeddings`, WebSocket `/v1/realtime`, `/infill`, `/props`, `/health` (503 while loading, 200 when ready), `/slots`, `/metrics` (requires `--metrics`).
- Messages API: "Tool use requires `--jinja` flag." It accepts `tools` and `tool_choice` (auto/any/named).

### Inferences
- Claude-Code-style clients can point at `/v1/messages`. Codex-style clients can use `/v1/responses`. OpenAI-SDK agents can use `/v1/chat/completions`.
- Use `--reasoning-format deepseek` so thinking goes into a separate `reasoning_content` field. This is the long-standing behavior of the deepseek format. The fetched README excerpt does not show the field name.

### Gaps
- The `reasoning_content` field name and Responses API reasoning-item mapping were not confirmed in the fetched README text.

## (B1) Cloudflare quick tunnel (trycloudflare.com) limits

### Takeaway
Quick tunnels are testing-only. Official docs say they do NOT support Server-Sent Events. That is a critical problem for `stream: true` OpenAI or Anthropic APIs, which use SSE. Other limits: 200 in-flight requests (429 beyond), no uptime guarantee, and a new hostname every run. The proxy read timeout is now 125 s (not 100 s). Use a named tunnel for any real or streaming use.

### Cited Findings
- "Each Quick Tunnel supports up to 200 in-flight requests. Additional requests return a `429` response." — [TryCloudflare docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
- "Quick Tunnels do not support Server-Sent Events (SSE)." — [TryCloudflare docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
- "Quick Tunnels have no uptime guarantee." "Quick Tunnels are for testing and development." Use a Cloudflare Tunnel for "stable hostnames and production traffic." "The hostname changes each time you create a Quick Tunnel." No account or domain is needed. Start with `cloudflared tunnel --url http://localhost:8080`. — [TryCloudflare docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
- Secondary report: the trycloudflare edge buffers `text/event-stream` responses, so events do not reach the client. WebSockets still work. — [daleseo.com (Korean)](https://daleseo.com/cloudflare-tunnel/)
- Beeper advises MCP users to switch from SSE to Streamable HTTP transport, because SSE does not work through a quick tunnel. — [Beeper docs](https://developers.beeper.com/desktop-api/advanced/remote-access/cloudflare)
- Streaming *request bodies* (fetch streams) over a quick tunnel with `--protocol=http2` reported broken. — [cloudflared issue #1385](https://github.com/cloudflare/cloudflared/issues/1385)
- Proxy Read Timeout: 125 s, returns 524, configurable only on Enterprise (up to 6000 s). Proxy Write Timeout: 30 s. Proxy Idle Timeout: 900 s (520). Client keep-alive: 400 s. Request and response headers: 128 KB each. — [Connection limits](https://developers.cloudflare.com/fundamentals/reference/connection-limits/); [Error 524](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/). This contradicts the task's assumed "100-second" figure. The old 100 s value is outdated.
- 524 mitigations from Cloudflare: poll status for long jobs, move long requests to a DNS-only subdomain, or raise the timeout (Enterprise). — [Error 524](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/)

### Inferences
- With SSE unsupported or buffered, a streamed completion through a quick tunnel may arrive all at once at the end. If generation (including long reasoning) takes more than 125 s before the edge sees a byte, the client gets a 524. A non-streamed long completion has the same risk.
- Keep-alive pings (SSE `:` comment lines) do not help if the edge buffers the whole SSE body. They do help on a named tunnel, where streaming passes through.
- Cap `max_tokens` and `--reasoning-budget` so a non-streamed reply finishes in well under 125 s, if you stay on a quick tunnel.

### Gaps
- No primary source states whether a few streamed bytes reset the 125 s read timer on trycloudflare specifically. The article that covers this ([levelup "125 second silence"](https://levelup.gitconnected.com/the-125-second-silence-why-your-ai-stream-dies-behind-cloudflare-40f776d9e055)) returned 403.
- No source found for a request body size limit on quick tunnels. The 100 MB Free-plan figure was not confirmed on the connection-limits page.
- No source found for per-IP rate limits on trycloudflare beyond the 200 in-flight cap.
- Community reports of successful LLM token streaming via trycloudflare exist anecdotally, but none were verified here. Test with `curl -N` before relying on it.

## (B2) cloudflared flags and quick-tunnel URL output

### Takeaway
`--protocol` defaults to `auto` (QUIC, with http2 fallback). Force `http2` on hosts that block UDP. The quick-tunnel URL is printed in a boxed banner on **stderr**. Grep it with `https://[a-zA-Z0-9-]+\.trycloudflare\.com`.

### Cited Findings
- `--protocol` auto|http2|quic. "The `auto` value will automatically configure the `quic` protocol." Falls back to http2 if UDP fails. Env var: `TUNNEL_TRANSPORT_PROTOCOL`. — [run parameters](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/cloudflared-parameters/run-parameters/)
- `--no-autoupdate`: turns off auto-updates. It has no effect on Windows, package-manager installs, or interactive terminal runs. — [run parameters](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/cloudflared-parameters/run-parameters/)
- `--metrics IP:PORT` exposes a Prometheus endpoint. `--loglevel` (default info; debug may leak sensitive data). `--logfile` (not rotated). `--output default|json` (version 2025.6.1+). `--edge-ip-version` default 4. `--grace-period` default 30s. `--retries` default 5. `--post-quantum` (quic only). — [run parameters](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/configure-tunnels/cloudflared-parameters/run-parameters/)
- The URL is written to stderr, so background runs need `2>&1` or a log file before you grep for `trycloudflare.com`. The banner says "Your quick Tunnel has been created! Visit it at ...", and the URL is on the following boxed line. Banner capitalization varies by version. — [flaviocopes guide](https://flaviocopes.com/cloudflare-quick-tunnels/); [cloudflared mirror commit TUN-4821](https://git.nixnet.services/curben/cloudflared-mirror/commit/a4a9f45b0a4fc826a0fbd5ed2173d78c1130ccf5)

### Inferences
- Robust launch:
  ```bash
  cloudflared tunnel --no-autoupdate --protocol http2 \
    --url http://127.0.0.1:8080 --metrics 127.0.0.1:20241 \
    --logfile /tmp/cf.log 2>&1 | tee /tmp/cf.out &
  until URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/cf.out | head -1); [ -n "$URL" ]; do sleep 1; done
  ```
- Use `--http-host-header` only if the origin checks the Host header. llama-server does not, so it is not needed. This flag was not verified on the fetched page.

### Gaps
- The `--http-host-header` description was not fetched from the docs.

## (B3) Mitigations

### Takeaway
For streaming agents, a named tunnel (free Cloudflare account plus a domain) is the real fix. On a quick tunnel, use non-streamed requests that finish within 125 s, or use a WebSocket/other transport.

### Cited Findings
- Cloudflare's own fix for quick-tunnel limits: create a Cloudflare Tunnel with an account. — [TryCloudflare docs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)
- Enable HTTP keep-alive on the origin to avoid connection resets. — [Connection limits](https://developers.cloudflare.com/fundamentals/reference/connection-limits/)
- At the origin, send `Content-Type: text/event-stream` and turn off proxy buffering (community advice for Cloudflare tunnel streaming). — [lobe-chat discussion #1016](https://www.github.com/lobehub/lobe-chat/discussions/1016)
- Use polling for long jobs instead of holding requests open past 125 s. — [Error 524](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/)

### Inferences
- Mitigation list:
  1. Use a named tunnel for anything beyond a demo. It gives a stable hostname and working SSE.
  2. On a quick tunnel, run a `curl -N` streaming smoke test first. If tokens arrive in bursts at the end, SSE is being buffered.
  3. Set `--api-key` on llama-server, because the trycloudflare URL is public.
  4. Keep `-np` well below 200 so the server queues work, not the edge (429s).
  5. Bound `max_tokens` and `--reasoning-budget` so total time stays under 125 s.
  6. Re-read the URL from stderr after every restart. Do not hard-code it.
  7. Use `--protocol http2` if UDP/QUIC is blocked on the GPU host.

### Gaps
- No verified evidence on whether SSE heartbeat comments help on quick tunnels.
