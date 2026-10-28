# inference-deploy

```bash
uv run serve.py
```

Edit `config.py` to change settings.

## Using it with OpenCode

After `uv run serve.py` starts, it prints a prompt at the top. Copy that prompt and
paste it into a new OpenCode session — it adds the model to your OpenCode config and
verifies it.

When the model is registered, pick it in the TUI:

```
/models → llamacpp/qwen3.8-27b
```

Notes:

- The API key is generated once and kept in `.api_key`, so it never changes between runs.
- A quick tunnel gets a new URL every restart → run the prompt again with the new URL
  (all other details stay the same). With a named tunnel (`TUNNEL.token` + `hostname`)
  or when OpenCode runs on the GPU box itself, the URL is permanent and setup is one-time.
- The prompt is generated from `config.py`, so any change to model, context or
  capabilities is picked up automatically on the next run.
