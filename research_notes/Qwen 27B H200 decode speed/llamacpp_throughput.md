# llama.cpp decode throughput for ~27-32B dense LLMs on H100/H200, and theoretical ceilings

Research date: 2026-10-28. Note up front: I found **no published llama.cpp llama-bench result for any 27-32B dense model on an H200 or H100**. The H100/H200 figures below come from Ollama (which uses llama.cpp/GGML underneath) or SGLang. llama.cpp-specific figures come from consumer GPUs (RTX 3090/5090). H200 llama.cpp numbers below are **projections** built from those data points.

## Q1. Measured benchmark numbers for 27-32B dense models (H100/H200 and reference GPUs) per quant

### Takeaway
No direct H200 llama.cpp numbers exist publicly. The closest point is H100 SXM with Ollama: Qwen3.8-27B Q4_K_M decodes at **134 tok/s at a ~16k prompt**. On H200, SGLang without speculation gets **69 tok/s** (likely BF16). llama.cpp on an RTX 5090 gets ~74 tok/s at Q4_K_M. On an RTX 3090 it gets 40 / 36 / 32 / 27 / 15.5 tok/s for Q4 / Q5 / Q6 / Q8 / BF16.

### Cited Findings
- **Model facts, Qwen3.8-27B:** 27.8B params, dense. 64 layers: 48 Gated DeltaNet plus 16 full attention (hybrid). Has a built-in MTP head (GGUF layer 64). Native context is 262,144, extendable to 1M with YaRN. Embeddings are untied. Published 19 Aug 2026, updated 24 Sep 2026 — [Context Studios hardware guide](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **H200 (SGLang, not llama.cpp):** 69 tok/s single-stream autoregressive. MTP gives 135-179 tok/s. DFlash 2 (block 8, z-lab) gives 184-236 tok/s single-stream (2.67-3.43x). Aggregate: 1,090-1,368 tok/s at c8 and 1,525-1,952 tok/s at c32 — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **H200 community endpoint (engine unspecified):** spec decoding takes it from 70 to 126 tok/s (V. Mustar) — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **H100 SXM 80GB, Ollama, Qwen3.8:27b Q4_K_M:** prompt sizes are real (filled), median of 2 trials, Ollama version not stated. Article dated 2026-10-02 — [SimeonOnSecurity Vast.ai benchmark](https://simeononsecurity.com/articles/qwen38-27b-vast-ai-gpu-benchmark/)
  - Decode: 134.4 tok/s at 16.4k, 120.0 at 32.8k, 112.0 at 65.5k, 108.2 at 131k. The 256k run did not complete.
  - Prefill: 89,974 / 139,748 / 195,252 / 242,093 tok/s at the same sizes. These look implausibly high, possibly prompt-cache effects, so treat them with caution.
  - A100 SXM4 40GB: 60.6 / 51.1 / 49.3 tok/s decode at 16k / 32k / 65k. RTX 6000 Ada: 89.1 / 75.7 / 70.8 / 61.7.
- **RTX 5090 (1.79 TB/s), llama.cpp b10448, Qwen3.8-27B Q4_K_M:** 73.6 tok/s with MTP off. MTP n=1/2/3/4/5 gives 104.8 / 125.5 / **133.6** / 119.5 / 108.8 tok/s, so the peak is 1.81x at n=3. Flags: `-ngl 999 -fa on -np 1 --spec-type draft-mtp --spec-draft-n-max 2`. Published Aug 16 2026 — [KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings)
- **RTX 5090, llama.cpp, Q4, MTP off (Hardware Corner data):** prefill/decode is 3,750/74.8 at 4k, 1,146/29.0 at 32k, 718/26.2 at 64k, 461/22.8 at 128k — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **RTX 5090, standalone llama.cpp, Qwen3.5-27B GGUF:** 70.37 tok/s decode — [WanGP issue #2193](https://github.com/deepbeepmeep/WanGP/issues/2193)
- **RTX 3090 (936 GB/s), llama.cpp build ce8fd4b1a (8777), Qwen3.6-27B, tg128, June 10 2026** — [ahelpme RTX 3090](https://ahelpme.com/ai/llamacpp-ai/llama-bench-the-qwen3-6-27b-and-nvidia-rtx-3090/)
  - Per-quant: Q4_K_M (15.65 GiB) 40.14 tok/s, Q5_K_M (18.16 GiB) ~36.3, Q6_K (20.97 GiB) ~32.2.
  - Summary averages: Q4 39.9, Q5 35.9, Q6 31.6, Q8 26.8, BF16 15.5 tok/s.
  - Q8 and BF16 needed multiple GPUs (layer split).
- **Tesla V100 SXM2 32GB, CUDA, Qwen3.6-27B Q4_K_M:** ~32 tok/s tg128 — [ahelpme V100](https://ahelpme.com/ai/llamacpp-ai/llama-bench-the-qwen3-6-27b-and-nvidia-tesla-v100-sxm2-32gb/)
- **4x RTX 3090, tensor split, FA on, Qwen3.8 27B Q5_K_M:** ~75-77 tok/s (tg512/tg1024). Build 10809 (5266f24), posted Sep 7 2026 — [llama.cpp CUDA scoreboard discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- **CUDA scoreboard reference model (Llama 2 7B Q4_0, 3.56 GiB), commit 5143fa8** — [Discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
  - H100 80GB (variant unspecified): tg128 267.8 tok/s with FA off and 280.7 with FA on; pp512 9,918 / 11,263.
  - A100 80GB: tg128 190.9 / 200.9. RTX 5090: 290.0 / 300.4.
  - The scoreboard has no H200, B200 or GH200 rows.
- **Older dense 32B (non-hybrid), H100 80GB SXM5, Ollama v0.5.7, Qwen2.5-32B Q4_K_M:** ~38 tok/s, and ~94 tok/s at batch 4. These look oddly low and are possibly misconfigured. The source is an aggregator-style page, which makes it low-trust — [markaicode](https://markaicode.com/benchmarks/cuda-qwen-25-h100-memory-benchmark/)
- **Qwen3-32B on H100 PCIe:** the only figure is an estimate (93.6 tok/s at Q4_K_M), not a measurement — [willitrunai](https://willitrunai.com/de/can-run/qwen-3-32b-on-h100-pcie-80gb)
- **GH200 144G HBM3e (H200-class memory), llama.cpp commit c6f6e4f, Dec 2025:** gpt-oss-20B MXFP4 (11.27 GiB, MoE) tg32 = 322.9 tok/s, and gpt-oss-120B = 209.0 tok/s. No dense-model results — [Discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)

### Inferences
- **Effective bandwidth (weights GB x tok/s):**
  - RTX 3090 Q4_K_M: 16.8 GB x 40.1 = ~674 GB/s, about 72% of 936 GB/s. Q6_K: 22.5 GB x 32.2 = ~725 GB/s, about 77%.
  - RTX 5090 Q4_K_M: ~16.8 GB x 74 = ~1.24 TB/s, about 69% of 1.79 TB/s.
  - H100 SXM via Ollama at 16k: ~17 GB x 134 = ~2.3 TB/s, about 68% of 3.35 TB/s. This includes attention over 16k of context.
  - H200 SGLang BF16, if that is the 69 tok/s run: ~55.6 GB x 69 = ~3.8 TB/s, about 80% of 4.8 TB/s.
- **5090 to H100 scaling:** the decode ratio (134/74 = 1.81) almost equals the bandwidth ratio (3.35/1.79 = 1.87). Decode looks bandwidth-bound, with little fixed overhead, at this model size.
- **Projected H200 llama.cpp single-stream decode at short context, no speculation.** Assumes 65-75% of 4.8 TB/s, i.e. 3.1-3.6 TB/s effective. GGUF sizes are approximate.

  | Quant | Approx. weights | Projected decode (tok/s) |
  |---|---|---|
  | Q4_K_M | ~16.8 GB | ~185-215 |
  | UD-Q4_K_XL | ~17.5-18 GB | ~175-205 |
  | Q5_K_M | ~19.5 GB | ~160-185 |
  | Q6_K | ~22.5 GB | ~140-160 |
  | Q8_0 | ~29.5 GB | ~105-120 |
  | BF16 | ~55.6 GB | ~56-65 (SGLang measured 69) |

- **Cross-check from H100 Ollama:** 134 x (4.8/3.35) = ~192 tok/s at 16k context.
- **Verdict on the 200 tok/s target:**
  - Plain decode: borderline, only at Q4-class quants with short context.
  - Q5 and above: unlikely without speculation.
  - With llama.cpp's MTP (`--spec-type draft-mtp`), the 1.8x gain seen on the 5090 projects to ~300+ tok/s at Q4. MTP makes 200+ comfortable at Q4-Q6, and plausible at Q8.

### Gaps
- No public llama-bench/llama-server run of any 27-32B dense model on H200 or H100 with llama.cpp proper. Ollama and SGLang results are proxies.
- Unclear which quant SGLang's 69 tok/s H200 figure used. BF16 is assumed.
- No measured UD-Q4_K_XL vs Q4_K_M speed difference on datacenter GPUs.
- No pp512 data for 27B on H100/H200 in llama.cpp.

## Q2. Theoretical bandwidth ceiling and achieved llama.cpp efficiency

### Takeaway
The ceiling is 4.8 TB/s divided by the bytes read per token. That gives ~285 tok/s for Q4_K_M (~16.8 GB), ~210 for Q6_K, ~160 for Q8_0, and ~86 for BF16. llama.cpp CUDA on ~27B models measures at about 68-77% of peak bandwidth on consumer and Hopper GPUs. That lands Q4 near, but often just under, 200 tok/s.

### Cited Findings
- RTX 3090 per-quant tg128 results and GGUF sizes (data for the efficiency calculation) — [ahelpme RTX 3090](https://ahelpme.com/ai/llamacpp-ai/llama-bench-the-qwen3-6-27b-and-nvidia-rtx-3090/)
- On small models, llama.cpp does not saturate bandwidth. The Llama 2 7B Q4_0 run on H100 (280.7 tok/s) implies only ~1.07 TB/s effective, because per-token fixed overheads dominate at ~3.6 ms/token — [Discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)
- Third-party kernel analysis says llama.cpp's MMVQ decode GEMV kernels reach 776-894 GB/s on an RTX 4090 (~1.0 TB/s peak), i.e. ~77-89% — [paiml/aprender issue #5221](https://github.com/paiml/aprender/issues/5221)
- llama.cpp's fused CUDA `gated_delta_net_cuda<128>` kernel runs at ~2.5 µs per call on a 4B model (RTX 4090), so it is cheap relative to the GEMVs — [paiml/aprender issue #5222](https://github.com/paiml/aprender/issues/5222)
- On GH200, a commenter notes that Hopper needs `wgmma` instructions for full tensor-core throughput, which llama.cpp lacked as of Dec 2025. This affects prefill and batched work more than batch-1 decode — [Discussion #18005](https://github.com/ggml-org/llama.cpp/discussions/18005)

### Inferences
- At 200 tok/s the budget is only 5 ms per token. The 7B scoreboard result shows fixed per-token overhead can be a few ms: 64 layers plus GDN state ops, kernel launches and sync.
- The H100 Ollama result (134 tok/s) caps that overhead at ≤ ~2.4 ms. On H200 that overhead does not shrink while weight-read time does. A ~165-195 tok/s Q4 range is therefore more realistic than the naive 70% x 285 = 200.
- Use CUDA graphs (enabled by default for batch-1 on CUDA), FA on, `-ngl 999`, and a single GPU. Split-mode layer adds no bandwidth for single-stream decode.

### Gaps
- No nsys/ncu profile of a 27B hybrid model on Hopper in llama.cpp.
- The exact fixed overhead per token is unknown. The figures above are inferred.

## Q3. Decode degradation at long context (32K-512K), flash attention, KV cache type

### Takeaway
Qwen3.5/3.6/3.8-27B is hybrid: only 16 of 64 layers keep a KV cache. Long-context decode therefore degrades far less than in a pure-attention 32B. Measured on H100 (Ollama), decode drops ~19% from 16k to 131k (134 to 108 tok/s). On a 5090 with MTP it drops ~9% from 4K to 256K. Quantized KV (q8_0/q4_0) is slightly faster. One open llama.cpp bug reports failures beyond ~130k context.

### Cited Findings
- H100 SXM Ollama, filled prompts: 134.4 tok/s at 16.4k, 120.0 at 32.8k, 112.0 at 65.5k, 108.2 at 131k. The 256k run did not complete — [SimeonOnSecurity](https://simeononsecurity.com/articles/qwen38-27b-vast-ai-gpu-benchmark/)
- For contrast, dense non-hybrid Llama 3.1 8B on the same H100 drops much more: 163.5 tok/s at 32k, 124.0 at 64k, 82.4 at 128k (about -50% from 32k to 128k) — [SimeonOnSecurity](https://simeononsecurity.com/articles/qwen38-27b-vast-ai-gpu-benchmark/)
- **RTX 5090, llama.cpp b10448, Q4_K_M + MTP, short prompt** — [KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings)
  - Decode: ~126 tok/s from 4K to 64K, 120.6 at 128K, 115.1 at 256K. VRAM grows from 16.5 GB to 27.1 GB.
  - TTFT when the window is filled: 1.4 s at 4K, 78 s at 128K, 757 s at 256K.
- **KV cache type (same source, MTP n=2, 32K):** f16 125.5 tok/s (18,798 MB), q8_0 128.3 tok/s (17,924 MB), q4_0 136.7 tok/s (17,412 MB). The author reports no quality loss — [KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings)
- **Conflicting data, RTX 5090 llama.cpp Q4 with MTP off (Hardware Corner):** 74.8 tok/s at 4k but 29.0 at 32k, 26.2 at 64k, 22.8 at 128k. That is a much steeper drop. The likely cause is a filled-context run under different settings (VRAM pressure or no FA), but this is unverified — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **DGX Spark, llama.cpp Q4, decode:** 12.2 at 4k, 9.7 at 64k, 8.0 at 128k, 5.9 tok/s at 256k (-52% from 4k to 256k) — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **Bug, open issue #27756:** Qwen3.8-27B (64-layer Qwen3.5-hybrid) silently emits EOS as the first token once the prompt exceeds ~130k positions, on both CUDA and CPU. The author's hypothesis of GDN recurrent-state degradation is unconfirmed — [llama.cpp issue #27756](https://github.com/ggml-org/llama.cpp/issues/27756)

### Inferences
- **Projected H200 Q4_K_M plain decode:** ~170-195 tok/s at 32K, ~150-175 at 128K, ~130-160 at 256K. This applies the H100 decay curve (-11% at 32k, -19% at 131k relative to 16k) and extrapolates.
- **512K and beyond:** needs YaRN, and the 256k native limit is exceeded. Expect a further drop, plus the issue #27756 risk.
- Prefill time at 128K-256K dominates end-to-end latency. On a 5090, TTFT is minutes. H200 prefill (Hopper compute ~2x the 5090 in dense BF16 tensor throughput) would still take tens of seconds at 256K.
- **KV quant:** q8_0 KV is a safe default for long context on H200. Memory is not the constraint on 141 GB, so the gain is small speed (~2-9%), not capacity.

### Gaps
- No measured llama.cpp decode at 256K or 512K on Hopper.
- No published KV size per token for Qwen3.8-27B's 16 full-attention layers.
- No direct FA on/off comparison for this hybrid model.

## Q4. Aggregate throughput with parallel slots (-np 2/4/8) in llama-server on CUDA

### Takeaway
No CUDA llama-server/batched-bench aggregate numbers for 27B hybrid models on H100/H200 were found. Evidence suggests llama.cpp batches the hybrid Gated DeltaNet path poorly. At high concurrency, vLLM and SGLang reach 3-4x higher aggregate throughput. On H200, SGLang reaches 1,090-1,952 tok/s at c8-c32 with speculation.

### Cited Findings
- **llama.cpp issue #20006 (Vulkan, Radeon 780M iGPU, build 8182, UD-Q4_K_XL, Mar 2026)** — [llama.cpp issue #20006](https://github.com/ggml-org/llama.cpp/issues/20006)
  - S_TG for Qwen3.5-27B: B=1 3.83, B=8 6.87, B=16 8.95, B=32 10.53 tok/s (2.75x at B=32).
  - Dense Gemma 3 27B on the same hardware: B=1 4.28, B=8 8.72, B=16 16.55, B=32 30.13 tok/s (7x).
  - The reporter attributes the gap to non-matmul GDN ops (MUL, REPEAT, CPY, GET_ROWS on `cache_s_l`).
  - PR #20334 reportedly made it "much better", with no numbers given. The issue auto-closed Apr 27 2026.
- **llmkube bakeoff, 2x RTX 5060 Ti, Qwen3.6-27B, Apr 23 2026** — [LLMKube blog](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
  - Single stream: llama.cpp Q4_K_M ~20 tok/s vs vLLM NVFP4 ~19 tok/s.
  - At c=64 chat: llama.cpp 94 tok/s aggregate (`--parallel 16`) vs vLLM 345 tok/s (3.7x).
  - Coding: 133 vs 377 tok/s. Agentic: 72 vs 262 tok/s.
- **H200 SGLang + DFlash 2:** 1,090-1,368 tok/s at c8 and 1,525-1,952 tok/s at c32. **RTX 3090 vLLM:** ~1,000 tok/s aggregate at 64 parallel — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **How slots work:** extra slots only pay off through continuous batching of decode steps into one forward pass. Raising `--parallel` without raising batch/ubatch spreads the same throughput across more queues — [Big Iron guide](https://www.bigiron.cc/guides/llamacpp-server-multi-user-batching-and-queueing-patterns)
- **Unified KV on CUDA (unverified):** a search summary quoted a llama.cpp maintainer (Feb 2026) saying the default unified KV cache with multiple slots is not yet optimized on CUDA and slows down over time. The suggested workaround was disabling unified KV. I did not fetch the original thread; it may be [Discussion #23147](https://github.com/ggml-org/llama.cpp/discussions/23147). Verify before relying on it.
- **Recommended method:** `llama-batched-bench -fa 1 -npp <ctx> -ntg <n> -npl 1,2,4,8,16,32`. The aggregate S_TG column is the metric. On DGX Spark with gpt-oss-20b, this goes from ~74 tok/s at B=1 to ~208 tok/s at B=8 — [JetsonHacks DGX Spark bench](https://jetsonhacks.com/wp-content/uploads/2025/10/spark-llamacpp-bench.html)
- **Recurrent state across slots:** on HIP/ROCm, the fused GDN op carried recurrent state across requests on a reused server slot, leaking earlier prompts' text into later completions. This is a correctness risk for multi-slot hybrid serving on non-CUDA backends — [llama.cpp issue #29092](https://github.com/ggml-org/llama.cpp/issues/29092)
- **Fused GGML_OP_GATED_DELTA_NET** with CPU and CUDA backends was added by PR #19504, merged in b8233 (per search summary). A separate Qwen3.5-27B CUDA crash regression with the fused GDN path in the prompt cache was reported in [issue #21383](https://github.com/ggml-org/llama.cpp/issues/21383)

### Inferences
- **Projected H200 llama.cpp aggregate at Q4_K_M** (assuming GDN batching is near the issue #20006 post-fix behavior):
  - -np 2: ~1.6-1.9x single-stream, ~300-360 tok/s.
  - -np 4: ~2.3-3x, ~400-550 tok/s.
  - -np 8: ~3-4.5x, ~550-800 tok/s.
  - Per-stream speed falls to ~70-100 tok/s at np 8.
  - These are rough extrapolations, not measurements.
- **If aggregate throughput at np ≥ 8 matters:** vLLM or SGLang on H200 is likely 2-4x better than llama.cpp for this hybrid architecture.
- **For a single interactive stream:** llama.cpp with MTP is competitive.

### Gaps
- No CUDA batched-bench or llama-server multi-slot data for any 27B hybrid or dense model on Hopper.
- Effect of PR #20334 on CUDA batched GDN speed is unquantified.
- Whether MTP spec decoding works together with -np > 1 in llama-server, and at what efficiency, is unknown.

## Q5. Effect of the hybrid architecture (Gated DeltaNet + full attention) on llama.cpp CUDA speed

### Takeaway
On CUDA batch-1 decode, the hybrid design costs little. The fused GDN CUDA kernel is cheap, and 27B hybrid decode tracks the weight-bandwidth expectation (5090 ~69% efficiency, H100 Ollama ~68%). It helps at long context, since only 16 of 64 layers have KV. Its weaknesses are batched/multi-slot throughput, backend-specific bugs, and long-context (>130k) correctness.

### Cited Findings
- Qwen3.8-27B layout is 48 Gated DeltaNet + 16 full-attention layers out of 64 — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- The llama.cpp CUDA GDN recurrence kernel costs ~2.5 µs per call on a 4B model, vs 44.7 µs for a third-party implementation — [aprender issue #5222](https://github.com/paiml/aprender/issues/5222)
- Hybrid decode degrades slowly with context, while dense Llama 3.1 8B degrades fast (data in Q3) — [SimeonOnSecurity](https://simeononsecurity.com/articles/qwen38-27b-vast-ai-gpu-benchmark/)
- Batched GDN scaled 2.75x at B=32 vs 7x for dense Gemma 3 27B (Vulkan) — [issue #20006](https://github.com/ggml-org/llama.cpp/issues/20006)
- **Backend bugs:** Vulkan lacked a GDN shader and fell back to CPU. Strix Halo Qwen3.5-27B Q4_K_M: Vulkan tg128 11.87 tok/s vs ROCm 11.81 tok/s — [issue #20354](https://github.com/ggml-org/llama.cpp/issues/20354). HIP silent corruption: [issue #27556](https://github.com/ggml-org/llama.cpp/issues/27556). CUDA prompt-cache crash: [issue #21383](https://github.com/ggml-org/llama.cpp/issues/21383). Instant EOS beyond ~130k: [issue #27756](https://github.com/ggml-org/llama.cpp/issues/27756)
- llama.cpp supports the built-in MTP head via `--spec-type draft-mtp` (build b10448), giving up to 1.81x on a 5090 — [KGP Talkie](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings)

### Inferences
- **Best llama.cpp path to ≥200 tok/s single-stream on H200:**
  - Q4_K_M or UD-Q4_K_XL (Q5/Q6 also plausible with MTP).
  - Flags: `-fa on -ngl 999 -np 1 --spec-type draft-mtp --spec-draft-n-max 2-3`, KV q8_0.
  - Recent build (≥ b10448).
- **Expected:** ~300+ tok/s at short context on code/structured text. Lower on prose, since MTP acceptance varies (5090 code 148 vs Japanese prose 129 tok/s with tuned spec).
- **Without MTP:** expect ~165-210 tok/s at Q4, so 200 is marginal.

### Gaps
- No H200/H100 llama.cpp MTP measurements.
- MTP acceptance rates per task type were not found for llama.cpp.
- Unknown whether llama.cpp uses CUDA graphs with the GDN ops on the MTP draft path.
