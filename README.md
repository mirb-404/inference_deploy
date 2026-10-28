# inference-deploy

```bash
uv run serve.py
```

Edit `config.py` to change settings.

## Using it with OpenCode

`serve.py` prints one ready-made `opencode.jsonc` block (endpoint URL, key, capabilities,
context limits, a `#fast` no-think variant) and saves the same file next to itself.

1. `uv run serve.py` on the GPU box.
2. Copy the printed block or the file:

   ```bash
   scp datalab:inference_deploy/opencode.jsonc ~/.config/opencode/opencode.jsonc
   ```

   (or paste the block into that file / merge `providers` + `model` into your own config).
3. `/models` → `llamacpp/qwen3.8-27b`. Done — config reloads on save.

Details:

- The API key is generated once and kept in `.api_key`, so it never changes between runs.
- A quick tunnel gets a new URL every restart → paste the block again (all other details
  stay identical). With a named tunnel (`TUNNEL.token` + `hostname`) or when OpenCode runs
  on the GPU box itself, the URL is permanent and this is one-time setup.
- `config.py` stays the single source of truth: `alias`, `ctx-size`, `jinja`,
  `reasoning-format`, `no-mmproj` are propagated into the printed config automatically.
