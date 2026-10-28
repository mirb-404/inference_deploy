# vLLM / SGLang vs llama.cpp for Qwen ~27B on a single H200 (agentic coding, ~200 tok/s per stream target)

Research date: 2026-10-28. Data dated where known. Tool budget limited this to about 15 calls, so some H100/H200 numbers could not be found and are listed as gaps.

Model context (affects all questions below):
- Qwen3.8-27B exists. Released around Aug 13-14, 2026 under Apache 2.0, with 27.78B params and multimodal input — [DataNorth](https://datanorth.ai/news/alibaba-releases-qwen3-8-27b); [OpenRouter listing 20260814](https://openrouter.ai/qwen/qwen3.8-27b-20260814); [HF model card](https://huggingface.co/Qwen/Qwen3.8-27B)
- Qwen3.8-27B and Qwen3.6-27B (Apr 2026) are NOT pure dense transformers. Both are hybrids with 64 layers in 16 blocks of (3x Gated DeltaNet linear attention, then FFN, then 1x Gated Attention, then FFN). Gated Attention has 24 Q / 4 KV heads with head dim 256. Hidden size is 5120, FFN is 17408, and both have built-in MTP heads — [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B); [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)
- The hybrid layout means engine support depends on Mamba/linear-attention state handling for prefix caching and spec decode. Qwen3-32B (2025) is a plain dense model and has none of these caveats.

## 1. Measured single-stream decode and aggregate throughput (BF16 / FP8 / INT4) on H100/H200

### Takeaway
The best H200 single-GPU data is z-lab's SGLang run of Qwen3.8-27B (Aug 2026). Plain autoregressive decode gives about 69 tok/s at c=1. Native MTP gives 152-153 tok/s on code, and DFlash2 speculative decoding gives 215-227 tok/s on code. DFlash2 is the only measured config that clears 200 tok/s per stream on one H200. I found no clean FP8 or INT4 c=1 numbers on H100/H200.

### Cited Findings
- **Setup:** z-lab DFlash2 model card, run on 1x NVIDIA H200 with SGLang and FlashAttention 3. Block size 8 (7 draft tokens). Sampling was temp 1.0, top-p 0.95, top-k 20, reasoning effort `xhigh`, with 4096 max new tokens. The card does not state target precision or engine version, and the target checkpoint is BF16 — [z-lab/qwen3.8-27b-dflash2](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- **c=1 output tok/s (same source):**

  | Task | AR | MTP (7 tok) | DSpark | DFlash2 |
  |---|---|---|---|---|
  | GSM8K | 68.9 | 178.5 | 185.3 | 236.1 |
  | MATH-500 | 69.0 | 172.8 | 174.5 | 230.7 |
  | HumanEval | 69.0 | 151.9 (2.20x) | 159.9 | 214.6 (3.11x) |
  | MBPP | 69.0 | 153.1 (2.22x) | 163.3 | 226.9 (3.29x) |
  | MT-Bench | 68.9 | 134.9 | 137.6 | 184.0 |

  — [z-lab/qwen3.8-27b-dflash2](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- **c=8 aggregate:** AR runs about 467-483 tok/s, MTP 835-1,024, and DFlash2 1,090-1,368. On HumanEval: AR 483.4, MTP 934.2, DFlash2 1,291.5 — [z-lab card](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- **c=32 aggregate:** AR runs about 1,330-1,547 tok/s. MTP is slower than AR on most tasks (0.77-1.04x; HumanEval 0.84x). DFlash2 still gains 1.01-1.45x (HumanEval 1,799 tok/s, 1.16x) — [z-lab card](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- **Secondary summary of the same H200 data:** AR 69 t/s, MTP 135-179, DFlash2 184-236. Also cites a community H200 endpoint at 70 to 126 t/s with spec decode, engine and precision not stated (article dated Aug 19, 2026, updated Sep 24) — [Context Studios hardware guide](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **Hardware sizing for Qwen3.6-27B (vLLM recipe):** BF16 fits on 1x H200 or 2x H100. FP8 (`Qwen/Qwen3.6-27B-FP8`) fits on a single 40 GB+ GPU. A `Qwen/Qwen3.6-27B-GPTQ-Int4` checkpoint exists for a 24 GB GPU. NVFP4 (`nvidia/Qwen3.6-27B-NVFP4`, about 22 GB) is Blackwell-only, so it does not apply to H200 — [recipes.vllm.ai Qwen3.6-27B](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- **H100, vLLM 0.19.1 (Reddit sweep, ~May 2026):** Qwen3.6-27B-FP8 reached 557 tok/s at c=16 with 128 in / 128 out. FP8 gave about +27% over BF16 for the dense 27B. No c=1 figure was given — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1sva0b9/snapshots/2026-05-01T23%3A43%3A03.013822Z)
- **H100 public benchmarks:** One guide states no publicly verified, apples-to-apples tok/s figure for Qwen3.6-27B on H100 exists from Qwen or a reproducible third party — [markaicode](https://markaicode.com/benchmarks/hugging-face-qwen-3-h100-latency-benchmark/)
- **AMD MI300X, vLLM, single-stream, BF16, prompts of ~200 / 2k / 12k / 32k tokens, 512 out (not NVIDIA, useful as a reference):**
  - Tuned AITER FA: 67 / 67 / 65 / 62 tok/s.
  - Adding MTP (3 tokens): 139 / 176 / 168 / 157 tok/s.
  - Real coding-agent requests (10-25k context): 68 tok/s without MTP and 161 tok/s with MTP.
  - The recipe notes its synthetic prompts are repetitive, which favors MTP.

  — [recipes.vllm.ai Qwen3.6-27B](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- **Other GPUs, single stream:**
  - B200 FP8 Qwen3.6-27B: peak 105 tok/s, spec-decode status unclear — [Runpod](https://www.runpod.io/articles/guides/best-gpu-for-llm-inference)
  - RTX Pro 6000 FP8, c=1, 1K context: 46.1 tok/s — [Millstone AI](https://www.millstoneai.com/inference-benchmark/qwen3-6-27b-fp8-1x-rtx-pro-6000-blackwell)
  - 2x RTX PRO 6000 NVFP4 + MTP on vLLM: mean 161 tok/s, max 190 tok/s — [loftllc](https://loftllc.dev/en/docs/tech/llm-research/qwen3-6-27b-nvfp4-mtp-vllm-benchmark/)
- **Older reference (2025, Qwen3 on SGLang, H200):** about 75 tok/s per user at batch 1 — [Baseten day-zero Qwen3](https://www.baseten.co/blog/day-zero-benchmarks-for-qwen-3-with-sglang-on-baseten/)

### Inferences
- **BF16 is the cause of the 69 tok/s AR baseline.** H200 bandwidth is about 4.8 TB/s and BF16 weights are about 54 GB, so the roofline is about 89 tok/s. The measured 69 tok/s (~78% of roofline) is consistent with BF16 weights.
- **FP8 should roughly double that.** FP8 weights (~28 GB) would put AR near 110-130 tok/s on H200. That estimate is extrapolated, not measured.
- **FP8 + MTP should reach the target, unmeasured.** FP8 with MTP (~2.2x on code per z-lab) would plausibly reach about 220-280 tok/s on code at c=1. Verify this before relying on it.
- **DFlash2 has the most headroom.** BF16 + DFlash2 measured 215-227 tok/s on code. FP8 + DFlash2 would likely exceed 250 tok/s, also unmeasured.
- **Spec decode helps one user, not a crowd.** At c=32 MTP hurts throughput. For an agent fleet at high concurrency, turn spec decode off or use DFlash2 only. For 1-8 agent streams, keep it on.

### Gaps
- No measured c=1 tok/s for FP8 or GPTQ-Int4/AWQ Qwen3.6/3.8-27B on H100/H200 with vLLM or SGLang.
- No head-to-head vLLM vs SGLang on the same H200 for this model.
- z-lab did not publish target precision or engine version.
- No H200 Qwen3-32B dense numbers were gathered (budget).

## 2. Speculative decoding: MTP, EAGLE-3, DFlash, ngram. Acceptance on code and exact flags

### Takeaway
Native MTP is supported out of the box in both engines. It gives about 1.5-1.6x on code with TP2 throughput benchmarks and about 2.2x at c=1 on H200. DFlash2 (z-lab/incoai drafter) is stronger: acceptance length 4.39-4.79 on code vs 3.91-3.99 for MTP, and about 3.1-3.3x at c=1. No Qwen3.6/3.8-27B EAGLE-3 or ngram data was found.

### Cited Findings
- **Acceptance length on H200 (7 draft tokens), MTP vs DSpark vs DFlash2:**
  - HumanEval: 3.91 / 3.30 / 4.39.
  - MBPP: 3.99 / 3.51 / 4.79.
  - GSM8K: 5.02 / 4.36 / 5.46.

  — [z-lab card](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- **vLLM blog (Aug 23, 2026; Qwen3.6-27B, AMD, TP2, output throughput), HumanEval baseline 1,481 tok/s:**
  - MTP N=1: 1.19x. N=2: 1.42x. N=3: 1.53x. N=4: 1.60x. N=5: 1.60x.
  - At N=5 on HumanEval: mean accepted length 4.36, acceptance rate 67.2%. MBPP: MAL 4.16, AR 63.3%.
  - Per-position acceptance is about 96% at position 1 and falls to about 47-55% at position 5.
  - DFlash (v1 drafter `z-lab/Qwen3.6-27B-DFlash`) peaked at about 1.40x on HumanEval. In this throughput setting, MTP beat DFlash on code.

  — [vLLM blog](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- **Community acceptance on code (2x RTX 3090, vLLM):** MTP per-position acceptance 94% / 82% / 72% — [k0zakinio/qwen36-vllm-setup](https://github.com/k0zakinio/qwen36-vllm-setup) (via search snippet)
- **Earlier DFlash claims for Qwen3.6-27B on B200:** 4.7x Math500 and 5.2x HumanEval at c=1 (BF16). The drafter was still training as of the Apr 26, 2026 snapshot — [NYU Shanghai RITS / Lucebox](https://rits.shanghai.nyu.edu/ai/luce-dflash-brings-2x-speculative-decoding-to-qwen3-6-27b-on-a-single-rtx-3090/)
- **Prompt-quoting answers:** One single-24 GB-card vLLM stack reports 127 tok/s single-user, rising to 381 tok/s "when the answer quotes the prompt". This shows the effect of prompt-lookup or ngram-style acceptance on edit-heavy code — [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen)

**Exact flags:**
- vLLM MTP (Qwen3.6-27B model card): `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B)
- vLLM MTP (recipe and blog): `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (recipe FP8 example). Benchmarks use 3. The blog command is:
  ```
  VLLM_USE_V2_MODEL_RUNNER=1 vllm serve Qwen/Qwen3.6-27B --trust-remote-code --tensor-parallel-size 2 --max-num-batched-tokens 32768 --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  ```
  — [recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.6-27B); [vLLM blog](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- vLLM DFlash (v1): `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":15}'` — [vLLM blog](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus)
- vLLM DFlash2 for Qwen3.8 needs an unmerged PR build:
  ```
  pip install -U "vllm @ git+https://github.com/vllm-project/vllm.git@refs/pull/52816/head"
  vllm serve Qwen/Qwen3.8-27B --speculative-config '{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
  ```
  — [z-lab card](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- SGLang MTP (Qwen3.6-27B card): `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B). SGLang resolves NEXTN to EAGLE internally — [SGLang Qwen3.5 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- SGLang DFlash2 (the H200 benchmark config). Support was merged via SGLang PR #35371 on Aug 19 — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide). Command:
  ```
  python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8
  ```
  — [z-lab card](https://huggingface.co/z-lab/qwen3.8-27b-dflash2)
- llama.cpp equivalents: `--spec-type draft-mtp`, where depth 3 gave 1.81x on an RTX 5090. The default depth of 16 halves throughput on Strix Halo, so use depth 3-4. A DFlash option exists: `--spec-type draft-dflash` — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)

### Inferences
- **Recommended for ~200 tok/s on one H200:** SGLang (recent main or a release after Aug 19, 2026) + Qwen3.8-27B + DFlash2 with 8 draft tokens. This is the only config measured above 200 tok/s at c=1 on H200.
- **Fallback with released vLLM:** native MTP with 3-4 tokens, ideally on the FP8 checkpoint. Expect about 150 tok/s BF16, and likely 200+ on FP8, which is inferred.
- **Coding agents favor high acceptance.** Edits that quote existing code accept unusually well (see the 381 tok/s HyperQwen figure). Real-agent speedups may beat HumanEval-style numbers.

### Gaps
- No EAGLE-3 drafter results for Qwen3.6/3.8-27B were found.
- No vLLM `ngram` / prompt-lookup or suffix-decoding acceptance numbers for this model on code were found.
- No vLLM-on-H200 MTP c=1 number was found.

## 3. Architecture support and long context (256K-1M, YaRN). Recommended launch commands

### Takeaway
Both vLLM (>=0.19.0) and SGLang (>=0.5.10) officially support the Qwen3.6/3.8 hybrid architecture. Native context is 262,144 tokens. YaRN (factor 4.0) extends to about 1M, and factor 2.0 targets about 524K. Static YaRN can hurt short-prompt quality, so enable it only when needed.

### Cited Findings
- **Qwen3.6-27B card commands:**
  - vLLM `>=0.19.0`: `vllm serve Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3`. Add `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, and `--language-model-only` for text-only.
  - SGLang `>=0.5.10`: `python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --port 8000 --tp-size 8 --mem-fraction-static 0.8 --context-length 262144 --reasoning-parser qwen3`, plus `--tool-call-parser qwen3_coder`.
  - The TP8 in these card examples is generic. One H200 is enough for BF16 per the vLLM recipe.

  — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B); [recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- **vLLM recipe, single-GPU FP8:** `vllm serve Qwen/Qwen3.6-27B-FP8 --max-model-len 262144 --reasoning-parser qwen3`. For the text-only + prefix caching variant, add `--language-model-only --enable-prefix-caching`. If Mamba cache or CUDA graph errors occur, lower `--max-cudagraph-capture-size` (default 512) — [recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- **Qwen3.8-27B card, base commands:** `vllm serve "Qwen/Qwen3.8-27B"` and `python3 -m sglang.launch_server --model-path "Qwen/Qwen3.8-27B" --host 0.0.0.0 --port 30000` — [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.8-27B card, 1M context:**
  - vLLM: `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` + `--hf-overrides` setting `text_config.rope_parameters` + `--max-model-len 1000000`.
  - SGLang: `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1` + `--json-model-override-args` + `--context-length 1000000`.
  - YaRN params: `rope_type` yarn, `factor` 4.0, `original_max_position_embeddings` 262144, `rope_theta` 10000000, `partial_rotary_factor` 0.25, `mrope_interleaved` true, `mrope_section` [11,11,10].
  - Use factor 2.0 for about 524K.

  — [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **Qwen3.8 card does not state:**
  - tool or reasoning parser flags
  - minimum engine versions ("use latest")
  - It does reference a `Qwen3.8-27B-FP8` checkpoint in its eval section.
  - Sampling for thinking mode: temp 1.0, top_p 0.95, top_k 20. For instruct mode: temp 0.7, top_p 0.8, presence_penalty 1.5. `reasoning_effort` takes xhigh (default), medium, or low.

  — [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B)
- **SGLang hybrid concurrency pitfall:** `--mamba-full-memory-ratio` (default 0.9) can silently cap concurrency on this hybrid model — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide)
- **SGLang FP8 KV caveat:** `--kv-cache-dtype fp8_e4m3` has no calibrated scales and may hurt reasoning accuracy. Use it only if memory forces it — [SGLang Qwen3.5 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)

### Inferences
- **KV memory is small, so long context fits on one H200.** Only 16 of 64 layers carry full-attention KV (4 KV heads x 256 dim). KV per token is about 16 layers x 2 x 4 x 256 x 2 bytes = 64 KiB in BF16, so 262K tokens is about 16 GiB. On one H200 with BF16 weights (~54 GB), a 256K-512K context per stream fits comfortably. This is computed from card specs, not measured.
- **Use FP8 when serving multiple long-context agents at once.** FP8 weights leave room for many concurrent 256K contexts.

### Gaps
- Qwen3.8-27B tool/reasoning parser names were not confirmed on its own card. The assumption is the same `qwen3_coder` / `qwen3` as Qwen3.6, which is likely given the identical architecture, but unverified.
- No long-context (>256K) decode-speed numbers on H200 were found.

## 4. Prefix caching for coding agents: vLLM APC / SGLang radix vs llama.cpp prompt cache

### Takeaway
Prefix caching matters a lot for agents, which resend long, growing prompts. The hybrid architecture complicates it: linear-attention state can only be restored at block or checkpoint boundaries. vLLM support uses `align` mode, which is marked experimental. SGLang has a Mamba radix cache with a `no_buffer` (default) or `extra_buffer` strategy. Both work but need tuning. llama.cpp is weaker on prefill and TTFT.

### Cited Findings
- **vLLM docs (dev, Oct 13, 2026), prefix caching on hybrid models:**
  - Mamba state is stored only on the Mamba block grid, so hits resume only at block boundaries.
  - `--enable-mamba-shared-prefix-checkpoint` (off by default) checkpoints at the shared-prefix junction. It requires prefix caching, EAGLE/MTP spec decode, `--prefix-match-unit` smaller than the Mamba block size, and no multi-module MTP.
  - Example: `vllm serve <hybrid-model> --prefix-match-unit 64 --enable-mamba-shared-prefix-checkpoint`

  — [vLLM APC docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- **vLLM default:** sets `align` Mamba cache mode for Qwen3.5 when prefix caching is enabled, flagged experimental — [vLLM forum](https://discuss.vllm.ai/t/whats-is-mamba-and-hybrid-model/2462); [recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) ("Mamba prefix caching experimental in align mode")
- **LMCache:** GDN does not support `all` mode, so `align` is required. LMCache needs `--separate-object-groups` — [LMCache Qwen3.5 recipe](https://docs.lmcache.ai/recipes/qwen3_5.html)
- **Block-size pitfall:** For Qwen3.5-4B, vLLM set the attention block size to 528 tokens to match the Mamba page. Prompts shorter than that got 0% hits — [vLLM issue #40696](https://github.com/vllm-project/vllm/issues/40696)
- **Conflicting report:** One older third-party write-up claims Qwen3.5 prefix caching in vLLM gives near-zero hit rates. This conflicts with current vLLM docs and is likely dated — [Zenn](https://zenn.dev/syoyo/articles/188508eb6ab389)
- **SGLang:** `--mamba-radix-cache-strategy no_buffer` (V1, default, no overlap scheduler) or `extra_buffer` (V2: overlap scheduling plus branching-point caching, needs `--page-size 64`, NVIDIA only, FLA backend). Docs say `extra_buffer` is compatible with speculative decoding — [SGLang Qwen3.5 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5); [PyTorch blog: hybrid models meet SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- **vLLM vs llama.cpp (Apr 23, 2026; Qwen3.6-27B on 2x RTX 5060 Ti; vLLM NVFP4 TP2 with prefix caching vs llama.cpp Q4_K_M):**
  - p50 TTFT, agentic pattern with a 4K shared prefix: vLLM 409 ms vs llama.cpp 911 ms.
  - p50 TTFT, coding: 106 ms vs 413 ms.
  - p50 TTFT, 5K long context: 581 ms vs 2,279 ms.
  - A ~43K prompt on llama.cpp took 186 s TTFT.

  — [LLMKube bakeoff](https://llmkube.com/blog/qwen3-6-27b-bakeoff)

### Inferences
- **vLLM agent config:**
  - Enable `--enable-prefix-caching`.
  - With MTP, add `--prefix-match-unit 64 --enable-mamba-shared-prefix-checkpoint` so that partial-block shared prefixes hit.
- **SGLang agent config:** use `--mamba-radix-cache-strategy extra_buffer --page-size 64`.
- **Measure hit rates.** Hybrid caching is new and version-sensitive.
- **llama.cpp on one H200:** Its prompt cache (`--cache-reuse`, slot save/restore) is single-slot oriented and prefills slower. It is weaker for multi-agent shared-prefix workloads.

### Gaps
- No measured prefix-cache hit rate or TTFT on H200 for Qwen3.6/3.8-27B in either engine.
- llama.cpp prompt-cache behavior for this hybrid model was not researched (budget).

## 5. OpenAI-compatible API: tool calling and reasoning parsers

### Takeaway
Both engines expose OpenAI-compatible `/v1/chat/completions` with Qwen-specific parsers. Use `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` (not `hermes`) for Qwen3.6, and likely Qwen3.8. vLLM also needs `--enable-auto-tool-choice`.

### Cited Findings
- **vLLM:** `--reasoning-parser qwen3`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. Disable thinking with `--default-chat-template-kwargs '{"enable_thinking": false}'` — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B); [recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- **vLLM on MI300X recipe:** also uses `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3` — [recipes.vllm.ai](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- **SGLang:** `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Per-request thinking off via `chat_template_kwargs: {"enable_thinking": false}` — [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B); [SGLang Qwen3.5 cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- **Coding-agent use:** OpenCode with vLLM serving Qwen3.6-27B is documented in a Docker Compose recipe (Apr 2026) — [Chris Dzombak](https://www.dzombak.com/blog/2026/04/a-vllm-docker-compose-recipe-for-running-qwen-3-6-27b-on-dual-rtx-3090s-opencode-configuration/)

### Inferences
- **Proposed single-H200 SGLang command for ~200 tok/s/stream.** This combines the sourced flags; the combination is untested:
  ```
  python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8 \
    --mamba-radix-cache-strategy extra_buffer --page-size 64 \
    --context-length 262144 --mem-fraction-static 0.85
  ```
- **vLLM fallback, released build.** Also untested as a combination:
  ```
  vllm serve Qwen/Qwen3.8-27B-FP8 --max-model-len 262144 \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --enable-prefix-caching --prefix-match-unit 64 --enable-mamba-shared-prefix-checkpoint --language-model-only
  ```
- **Engine verdict vs llama.cpp:**
  - Single stream: llama.cpp is at parity on small GPUs (LLMKube: 20 vs 19 tok/s c=1).
  - Concurrency: vLLM leads 2.8-3.7x at c=64 — [LLMKube](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
  - On H200, the measured 200+ tok/s/stream path exists only on SGLang (DFlash2). llama.cpp has MTP/DFlash spec types, but I found no H200 numbers for it.

### Gaps
- Tool-call parser reliability (malformed XML tool calls, streaming tool calls) with `qwen3_coder` on Qwen3.8 is not benchmarked in sources found.
- `hermes` parser compatibility is not verified for Qwen3.6/3.8.
- No llama.cpp H200 decode numbers for Qwen3.6/3.8-27B were found.
