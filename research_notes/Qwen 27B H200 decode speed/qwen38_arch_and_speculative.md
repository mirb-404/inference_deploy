# Qwen3.8-27B architecture and llama.cpp speculative decoding

Research date: 2026-10-28. llama.cpp latest release at research time: `v0.8.0`, published 2026-10-28 (GitHub releases API, https://api.github.com/repos/ggml-org/llama.cpp/releases/latest).

## Q1. Does Qwen3.8-27B exist and what is it?

### Takeaway
Yes. Qwen3.8-27B is a real model, released by the Qwen team in August 2026. It is not a pure dense transformer. It uses the Qwen3.5/3.6-style hybrid layout: 3 Gated DeltaNet (linear attention) layers per 1 Gated Attention layer. It has 262,144 native context, YaRN to 1M, a trained MTP head, and vision input. Some third-party pages call it "dense". That means not MoE (all 27B params active). It does not mean full attention in every layer.

### Cited Findings
- Official card: causal LM with vision encoder, 27B params, BF16 weights. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Layers: 64. Hidden 5120. FFN intermediate 17,408. Vocab 248,320 (padded). — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Layout: 16 x (3 x (Gated DeltaNet -> FFN) + 1 x (Gated Attention -> FFN)). So 48 linear-attention layers and 16 full-attention layers. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Gated DeltaNet: 48 V heads, 16 QK heads, head dim 128. Gated Attention: 24 Q heads, 4 KV heads, head dim 256, RoPE dim 64. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- MTP: "trained with multiple steps". No other detail on the card. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- HF tag is `qwen3_5`. The card does not name a separate `model_type`. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Release: BibTeX says August 2026. The collection shows "Updated Aug 13". A search summary gives August 14, 2026. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Context: 262,144 native, extensible to 1,000,000. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- YaRN config, set in `text_config.rope_parameters`: `rope_type` "yarn", `factor` 4.0, `original_max_position_embeddings` 262144, `rope_theta` 10000000, `partial_rotary_factor` 0.25, `mrope_interleaved` true, `mrope_section` [11,11,10]. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- The card warns that static YaRN can hurt short texts. Enable it only for long contexts. Example: factor 2.0 for 524,288 tokens. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Thinking mode sampling: temp 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B); same on [unsloth docs](https://unsloth.ai/docs/models/qwen3.8)
- Non-thinking/instruct sampling: temp 0.7, top_p 0.80, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0. Turn it on with `enable_thinking: False`. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Coding: there is no separate coding preset. Qwen's SWE-bench Pro and DeepSWE evals used temp 1.0, top_p 0.95, 256K context. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- Thinking is on by default. Depth is set by `reasoning_effort` (`xhigh` default, `medium`, `low`, none), and `preserve_thinking` keeps history reasoning. Unsloth gives this llama.cpp form: `--chat-template-kwargs '{"reasoning_effort":"medium"}'`. — [unsloth README](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/raw/main/README.md); [unsloth docs](https://unsloth.ai/docs/models/qwen3.8)
- Output budget recommendation: up to 262,144 tokens for reasoning and 131,072 for the final response. — [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

### Inferences
- Only 16 of 64 layers keep a growing KV cache: 4 KV heads x 256 dim each. That works out to about 16 x 2 x 4 x 256 x 2 B = 64 KiB per token at f16, or about 16 GiB at 262K context. The 48 DeltaNet layers have a fixed-size recurrent state. This is my arithmetic from the card's dimensions; I did not measure it.
- The recurrent state complicates speculative rollback. That fits the reported llama.cpp limits on partial rollback and -np>1 (see Q4).
- For coding agents, use the thinking preset (temp 1.0 / top_p 0.95 / top_k 20 / min_p 0). Use non-thinking temp 0.7 / top_p 0.8 / presence 1.5 for fast edits.

### Gaps
- No public detail on the MTP head: number of steps, or the depth trained versus the depth llama.cpp uses.
- No official "coding" sampling preset exists.

## Q2. GGUF files in unsloth/Qwen3.8-27B-GGUF

### Takeaway
The repo has Q4_0 through Q8_0 and UD quants up to UD-Q8_K_XL. BF16 is split into 2 shards. There is a separate `MTP/` folder with an MTP-head-only Q4_0 file, plus mmproj files. Exact sizes from the HF API are below (GB = 1e9 bytes).

### Cited Findings
All sizes come from [HF API tree](https://huggingface.co/api/models/unsloth/Qwen3.8-27B-GGUF/tree/main?recursive=true). Repo page: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
- `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` 49.99 GB + `-00002-of-00002.gguf` 4.67 GB = 54.66 GB total
- `Qwen3.8-27B-UD-Q8_K_XL.gguf` 31.46 GB
- `Qwen3.8-27B-Q8_0.gguf` 29.05 GB
- `Qwen3.8-27B-UD-Q8_K_L.gguf` 28.05 GB
- `Qwen3.8-27B-UD-Q6_K_XL.gguf` 25.30 GB; `UD-Q6_K_L` 24.19 GB; `UD-Q6_K_M` 23.09 GB; `UD-Q6_K` 21.98 GB
- `UD-Q5_K_XL` 20.88 GB; `UD-Q5_K_M` 19.77 GB; `UD-Q5_K_S` 18.67 GB
- `UD-Q4_K_XL` 17.56 GB; `Q4_1` 17.54 GB; `UD-Q4_K_M` 16.46 GB; `Q4_0` 16.06 GB; `UD-Q4_K_S` 15.36 GB; `UD-IQ4_XS` 14.25 GB
- `UD-Q3_K_XL` 13.15 GB; `UD-IQ3_S` 12.04 GB; `UD-IQ3_XXS` 10.93 GB
- `UD-Q2_K_XL` 9.83 GB; `UD-IQ2_S` 8.37 GB; `UD-IQ2_XXS` 7.27 GB; `UD-IQ1_M` 6.73 GB; `UD-IQ1_S` 6.19 GB
- `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` 1.37 GB (MTP head only)
- `mmproj-BF16.gguf` 0.93 GB; `mmproj-F16.gguf` 0.93 GB; `imatrix_unsloth.gguf` ~0.01 GB
- Unsloth's example command: `llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`. — [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
- Unsloth docs: keep 1-2 GB extra memory headroom when using MTP. — [unsloth docs](https://unsloth.ai/docs/models/qwen3.8)
- Third-party MTP-only drafts also exist, e.g. `Qwen3.8-27B-MTP-ONLY-Q6_K.gguf` and `Qwen3.8-27B-MTP-ONLY-Q4_K_M.gguf`. — [llama.cpp discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164)

### Inferences
- On a 141 GB H200, even BF16 (54.7 GB) or Q8_0 (29 GB) leaves plenty of room for 262K KV. Quant choice there is about decode bandwidth, not fit. Decode is memory-bound, so smaller weights decode faster.

### Gaps
- `UD-Q8_K_XL` quality versus `Q8_0` was not benchmarked here.

## Q3. GGUF general.architecture string

### Takeaway
The architecture string is `qwen35`. Use `--override-kv qwen35.context_length=int:N`. Native value in the GGUF is 262144. The main GGUFs also carry `qwen35.nextn_predict_layers = 1`, i.e. MTP/NextN tensors are inside the main file.

### Cited Findings
- I read the GGUF header bytes directly (HTTP range request) from `Qwen3.8-27B-UD-Q4_K_M.gguf`. Results: `general.architecture = "qwen35"`, `qwen35.context_length = 262144` (0x00040000), `qwen35.nextn_predict_layers = 1`. The file `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` also has `general.architecture = "qwen35"`. — [file](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf)
- Unsloth GGUFs carry `blk.*.nextn.*` tensors. llama.cpp loads them and ignores them unless MTP is enabled. Verified for UD-Q4_K_M, UD-Q4_K_XL and UD-Q2_K_XL. — [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)
- `--override-kv KEY=TYPE:VALUE,...` exists. Types: int, float, bool, str. — [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `--rope-scaling {none,linear,yarn}` exists. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### Inferences
- The arch is not qwen2, qwen3 or qwen3next. It is `qwen35`, the same path as Qwen3.5/3.6.
- For >262K: use `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 -c 1000000` (flags are standard llama.cpp; I did not confirm them for this model). That is likely cleaner than overriding context_length. For <=262K, no override is needed.

### Gaps
- I did not test whether llama.cpp's qwen35 path applies YaRN correctly to the 16 attention layers with mrope.

## Q4. llama.cpp MTP speculative decoding

### Takeaway
Supported upstream via `--spec-type draft-mtp`, from PR #22673 "llama + spec: MTP Support" by am17an, merged around 2026-05-16. Measured CUDA single-stream decode speedup on Qwen3.8-27B is about 1.3x-2.4x. Code prompts gain the most (+90-150%) and prose the least. Acceptance is about 50-80% aggregate. Best `--spec-draft-n-max` is 2 on slower cards and 3-4 on fast ones. The gain shrinks with concurrency: by about 4 parallel streams it is gone or negative. `--spec-default` is NOT MTP: it enables ngram-mod.

### Cited Findings
- PR #22673 adds MTP speculative decoding. The MTP head runs as a separate model loaded from the same GGUF, with its own context and KV cache. Merged; the page header shows May 16, 2026, and it was opened May 4, 2026. — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). Conflict: one recipe says it landed "July 2026" — [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp). Treat May 2026 as authoritative (the PR page itself).
- Flags in the PR: `--spec-type draft-mtp` (early comments used `mtp`), `--spec-draft-n-max N`, `--no-mmproj` to save memory. Early builds errored with "MTP currently supports only n_parallel=1". Post-merge TODOs: ngram compatibility, and fixing partial rollback for batch >1. Prompt processing is slower due to device-to-host embedding transfers. — [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673)
- Current `--spec-type` values: `none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache`. The list is comma-separated, so types can be combined. Default is none. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `draft-mtp` = "Use Multi Token Prediction (MTP) heads from the main model". — [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- `--spec-draft-n-max N` default 3. `--spec-draft-n-min` default 0. `--spec-draft-p-min` default 0.00. `--spec-draft-p-split` default 0.10. `--spec-draft-sampling {greedy,probabilistic}` default greedy. `--spec-draft-backend-sampling` default enabled. `-ctkd/-ctvd` draft KV types default f16. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `--spec-default` code: pushes `COMMON_SPECULATIVE_TYPE_NGRAM_MOD` with n_match 24, n_min 48, n_max 64. MTP is not enabled by it. — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)
- In `common/arg.cpp`, `-hf` downloads auto-resolve an "mtp" sidecar when `draft-mtp` is requested. If the draft repo ships an MTP sidecar and no spec type is set, the type is inferred as `draft-mtp`. An explicit `-md` disables sidecar resolution. `llama-download --mtp` "also download[s] the MTP head, if available". — [common/arg.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp)
- Recipe: `llama-server -m Qwen3.8-27B-Q4_K_M.gguf -c 131072 -ngl 999 -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1`. Optional `--spec-draft-p-min` 0.60-0.75 on bandwidth-poor cards. — [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)
- Qwen3.8-27B MTP results (baseline -> MTP tok/s, n-max, acceptance). Source: [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp).
  - RTX 5090, UD-Q4_K_XL, 262K: 74.3 -> 179.7, n=4, 0.35-0.91.
  - RTX 5090, Q6_K: 61.9 -> 130.0, n=2.
  - RTX 5090, UD-Q4_K_XL, b10680: 71.9 -> 170.3, n=7, 0.48 aggregate.
  - RTX 4090, UD-Q4_K_XL: 44.7 -> 84.9, n=4, 0.64 aggregate.
  - RTX PRO 6000 Blackwell, Q4_K_M: 63.8 -> 91.5, n=2, 0.72-0.81.
  - RTX A6000: 26.7 -> 52.5, n=2.
  - V100 32GB, Q4_K_M: 33.3 -> 48.0, n=2, 0.75.
  - Apple M4 and M3 Ultra: about 0% gain.
- Per-workload results. Source: [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp).
  - RTX 4090, n=4: code 44.7 -> 114.7 (2.57x), bash 44.4 -> 84.9, prose 44.9 -> 59.5.
  - RTX 5080+3090, n=2: code 55.7 -> 105.2, prose 54.0 -> 79.2.
  - V100, n=2: code 33.0 -> 48.1 (1.46x).
  - JSON/tool-call synthetic prompt, RTX 3090, n=4: 34.0 -> 84.4, acceptance 0.971.
- Controlled study, 1x RTX 3090, UD-Q4_K_XL, 25 prompts x 5 passes, `--parallel 1`, `-ctk q8_0 -ctv q8_0`, baseline 41.55 tok/s. Source: [HackMD thc1006](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090).
  - MTP n=2: 66.39 (+59.8%).
  - MTP n=3: 63.29 (+52.3%).
  - DFlash2 n=4: 63.13 (+51.9%).
  - MTP n=5: 54.89 (+32.1%).
  - DFlash2 n=7: +22.6%.
  - Code-class gains: MTP n=2 +92.0%, n=3 +98.9%, DFlash2 n=4 +116.8%.
- Output is not bit-exact versus serial greedy decoding. 92-100% of requests had diverged by 1,600 tokens. The cause has not been established, and the quality impact is unknown. — [HackMD thc1006](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- Concurrency: on Qwen3.6-27B Q4_K_M on DGX Spark, MTP gave about 2.2x at c=1 (28.3 vs 13.1). At c=4 it was slower (29.9 vs 41.5). Prompt processing was about 570-720 t/s with MTP versus 860-1,080 without. — [NVIDIA forum](https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298). Also "advantage gone by 4 concurrent streams" — [sudoingX/qwen38-mtp](https://github.com/sudoingX/qwen38-mtp)
- MTP on Apple Metal is a net loss at every config tested. — [issue #23752](https://github.com/ggml-org/llama.cpp/issues/23752)
- Pitfall: old CUDA builds (e.g. commit `221f0f6`) produced garbage on Qwen3.8 through the DeltaNet path, with or without MTP. Fixed around b10450 (`ece963f41`). Update all libggml*/libllama .so files together. — [discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164)
- Other reports: RTX PRO 4000 21.2 -> 59.5 t/s (2.81x) — [Context Studios](https://www.contextstudios.ai/blog/qwen-3-8-27b-hardware-guide.md). RTX 5090 coding task about 142 tok/s average with NVFP4-MTP GGUF — [DataCamp](https://www.datacamp.com/tutorial/how-to-run-qwen3-8-27b-locally). These are secondary sources.

### Inferences
- On H200 (4.8 TB/s), baseline single-stream decode is already high. MTP should still help: the verify batch of 3-5 tokens costs about the same as 1 token when memory-bound. Expect about 1.5-2.5x on code at n=3-4, extrapolating from the RTX 5090 / PRO 6000 rows. This is an inference only; I found no H100/H200 numbers.
- Use `-np 1` for maximum single-stream speed. With many concurrent agents, MTP may cut aggregate throughput.
- Combining types is allowed (`--spec-type draft-mtp,ngram-mod`). PR #22673 lists ngram compatibility as a post-merge TODO, so verify that it works on the current build.

### Gaps
- No H100/H200 llama.cpp MTP benchmarks for Qwen3.8-27B were found.
- Unclear whether the main GGUF's in-file nextn tensors or the `MTP/` sidecar file get used when both are present with `-m` (local path). Code shows sidecar auto-use only via `-hf`.

## Q5. Other speculative options: draft model, ngram, lookup

### Takeaway
For Qwen3.8-27B, a separate small draft model is a net loss: a 0.8B draft was 29.8% slower than no speculation on an RTX 3090. Qwen3 0.6B/1.7B use a 151,936-token vocab, versus Qwen3.8's 248,320. N-gram methods (`ngram-mod`, the `--spec-default`) help only when the output repeats context, e.g. rewriting or editing existing code. There, acceptance can reach 0.8-0.9. Otherwise it is near 0. No rigorous code-editing benchmark exists.

### Cited Findings
- An external 0.8B draft-then-verify setup against the dense Qwen3.8 target lost 29.8% (CI -33.1% to -26.4%). — [HackMD thc1006](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- Draft model flags: `-md/--spec-draft-model FNAME`, `-hfd/--spec-draft-hf`, `-ngld`, `-devd`, `-ctkd/-ctvd`, `--spec-draft-n-ubatch`. Type `draft-simple`. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- The old flags `--draft/--draft-max/--draft-min/--spec-ngram-size-n/--spec-ngram-min-hits` were removed. Use `--spec-draft-n-max` or `--spec-ngram-mod-n-max`. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `ngram-mod` (PR #19164, merged 2026-01-30) is a rolling hash table of n-gram -> next token. About 16 MB, constant memory, and one pool shared across all slots. Sample: `--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64`. Dense models can use smaller n-min/n-max. Uses: iterating over code blocks, reasoning models repeating their thinking, summarization. — [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md); [PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164)
- ngram-mod measured (other models): repeating 200 lines of code on gpt-oss-120b gave 0.92 acceptance and about 814 tok/s, but 0.00 and about 182 tok/s on another identical run. Devstral-2-123B with CRLF files gave 0.11 acceptance; with LF it gave 0.83 and about 2.3 -> 29.8 tok/s. Qwen 30B acceptance varied 0.05-0.63. The maintainer says it only helps when output repeats existing text. — [PR #19164](https://github.com/ggml-org/llama.cpp/pull/19164)
- Other ngram types: `ngram-simple` (example: "server-instance used to refactor source code"; defaults size-n 12, size-m 48), `ngram-map-k`, `ngram-map-k4v` (experimental), and `ngram-cache`. Lookup caches: `-lcs/--lookup-cache-static`, `-lcd/--lookup-cache-dynamic`. — [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md); [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- DFlash2 (`draft-dflash`, PR #27342, merged 2026-08-27): Qwen3.8-27B n=4 gave +51.9% overall and +116.8% on code on an RTX 3090. — [HackMD thc1006](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- EAGLE-3 (`draft-eagle3`) is supported. Listed drafts are for Llama 3.x, Qwen3-4B and Gemma 4. None is listed for Qwen3.8. — [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- Qwen3.8 vocab is 248,320. — [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

### Inferences
- Qwen3 0.6B/1.7B drafts are a poor fit: different vocab and lower acceptance than the built-in MTP. A Qwen3.5 0.8B-class draft shares the tokenizer but was measured slower.
- Best stack for code editing: `--spec-type draft-mtp` with n-max 2-4. Try adding `ngram-mod` for diff and rewrite-heavy agent loops, but only after verifying that combining works on the current build.

### Gaps
- I found no measured ngram-mod or ngram-simple numbers on Qwen3.8-27B, and no measured MTP+ngram combo numbers.
- Whether llama.cpp handles draft/target vocab mismatch (e.g. Qwen3 0.6B) is not verified.

## Q6. --load-mode, --kv-offload, --kv-unified with -np

### Takeaway
All three exist in current llama-server. `--load-mode` replaces the separate mmap/mlock/direct-io controls. `--kv-offload` is on by default. `--kv-unified` defaults to enabled only when `-np` is auto (-1). Unified KV lets slots share one pool, which matters with -np >1.

### Cited Findings
- `-lm, --load-mode MODE` (default auto). Modes: auto (mmap unless a device can't use it), none, mmap, mlock, mmap+mlock, dio (DirectIO if available). Env `LLAMA_ARG_LOAD_MODE`. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `-kvo, --kv-offload, -nkvo, --no-kv-offload`: KV cache offloading, default enabled. Env `LLAMA_ARG_KV_OFFLOAD`. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `-kvu, --kv-unified, -no-kvu, --no-kv-unified`: single unified KV buffer shared across all sequences. Default enabled if the number of slots is auto. Env `LLAMA_ARG_KV_UNIFIED`. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `--kv-unified-per-slot N`: context limit per slot. Without `-c`, the shared pool is sized to n_parallel*N. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `-np, --parallel N`: default -1 (auto). — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- `-fit, --fit [on|off]` is on by default and adjusts unset arguments to fit device memory. `--fit-ctx` minimum is 4096. With fit on, an instance takes the largest context that fits unless `-c` is pinned. — [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### Inferences
- With explicit `-np 1`, kv-unified is not auto-enabled, and it is irrelevant for one slot. With `-np >1`, pass `-kvu` explicitly so slots share context. Without it, `-c` is split evenly per slot.
- Pin `-c` explicitly (e.g. `-c 262144`) to avoid `--fit` picking an unexpected size.
- `--load-mode mmap+mlock` or `none` is fine on a host with plenty of RAM. With full GPU offload (`-ngl 999`), load mode mostly affects load time and host RAM.

### Gaps
- I did not verify how `-kvu` interacts with the per-slot recurrent DeltaNet state, or with MTP when -np >1.
