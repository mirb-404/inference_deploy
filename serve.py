"""Start llama-server from config.toml and expose it through a Cloudflare quick tunnel."""

import argparse
import os
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN_DIR = ROOT / "bin"
TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CLOUDFLARED_RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download"


def to_flags(params: dict) -> list[str]:
    """Turn the [llama] table into llama-server CLI arguments."""
    args = []
    for key, value in params.items():
        flag = key if key.startswith("-") else f"--{key}"
        if value is False or value is None:
            continue
        if value is True:
            args.append(flag)
        elif isinstance(value, list):
            for item in value:
                args += [flag, str(item)]
        else:
            args += [flag, str(value)]
    return args


def build_command(cfg: dict, api_key: str | None) -> list[str]:
    model, server = cfg["model"], cfg["server"]
    cmd = [server["binary"]]
    if model.get("path"):
        cmd += ["-m", model["path"]]
    else:
        cmd += ["-hf", f"{model['hf_repo']}:{model['quant']}"]
    if model.get("alias"):
        cmd += ["--alias", model["alias"]]
    cmd += ["--host", server["host"], "--port", str(server["port"])]
    if api_key:
        cmd += ["--api-key", api_key]
    return cmd + to_flags(cfg.get("llama", {}))


def resolve_llama_binary(server: dict) -> str:
    """Use the configured binary, else the repo-local build, building it first if allowed."""
    found = shutil.which(server["binary"])
    if found:
        return found
    local = ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
    if local.exists():
        return str(local)
    if not server.get("auto_build", True):
        sys.exit(f"llama-server binary not found: {server['binary']!r}. Set [server].binary or auto_build = true")
    if platform.system() != "Linux":
        sys.exit("auto_build supports Linux only. Build llama.cpp yourself and set [server].binary")
    print("[build] llama-server not found; building llama.cpp with CUDA (first run only, ~10-20 min) ...")
    result = subprocess.run(["bash", str(ROOT / "scripts" / "build_llamacpp.sh"), str(ROOT / "llama.cpp")])
    if result.returncode != 0 or not local.exists():
        sys.exit("[build] llama.cpp build failed; see messages above")
    return str(local)


def local_url(server: dict) -> str:
    host = server["host"]
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{server['port']}"


def find_cloudflared(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    system = platform.system().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(
        platform.machine().lower()
    )
    if system not in ("linux", "windows") or arch is None:
        sys.exit("cloudflared not found. Install it: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
    suffix = ".exe" if system == "windows" else ""
    target = BIN_DIR / f"cloudflared{suffix}"
    if not target.exists():
        url = f"{CLOUDFLARED_RELEASE}/cloudflared-{system}-{arch}{suffix}"
        print(f"[tunnel] downloading {url}")
        BIN_DIR.mkdir(exist_ok=True)
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
        target.chmod(0o755)
    return str(target)


def start_tunnel(binary: str, target: str, tunnel_cfg: dict) -> tuple[subprocess.Popen, threading.Event, list[str]]:
    """Launch a quick or named tunnel; the public URL lands in found[0] once ready is set."""
    token, hostname = tunnel_cfg.get("token"), tunnel_cfg.get("hostname")
    if token:
        if not hostname:
            sys.exit("[tunnel] token is set but hostname is empty")
        cmd = [binary, "tunnel", "--no-autoupdate", "run", "--token", token]
        ready_re = re.compile(r"Registered tunnel connection")
    else:
        cmd = [binary, "tunnel", "--no-autoupdate", "--url", target]
        ready_re = TUNNEL_URL_RE
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    ready, found = threading.Event(), []

    def reader():
        # Keep draining output so cloudflared never blocks on a full pipe.
        for line in proc.stdout:
            if not ready.is_set():
                match = ready_re.search(line)
                if match:
                    found.append(f"https://{hostname}" if token else match.group(0))
                    ready.set()
            if " ERR " in line:
                print(f"[tunnel] {line.rstrip()}")
        ready.set()

    threading.Thread(target=reader, daemon=True).start()
    return proc, ready, found


def wait_healthy(proc: subprocess.Popen, base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass  # still downloading/loading (503) or not listening yet
        time.sleep(2)
    sys.exit(f"llama-server not healthy after {timeout}s")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "config.toml", type=Path)
    parser.add_argument("--no-tunnel", action="store_true", help="serve locally only")
    parser.add_argument("--dry-run", action="store_true", help="print the llama-server command and exit")
    args = parser.parse_args()

    cfg = tomllib.loads(args.config.read_text(encoding="utf-8"))
    server = cfg["server"]
    api_key = None
    if server.get("require_api_key", True):
        api_key = server.get("api_key") or secrets.token_urlsafe(32)

    cmd = build_command(cfg, api_key)
    if args.dry_run:
        print(shlex.join(cmd))
        return
    cmd[0] = resolve_llama_binary(server)

    env = os.environ | {k: str(v) for k, v in cfg.get("env", {}).items()}
    base = local_url(server)
    use_tunnel = cfg.get("tunnel", {}).get("enabled", True) and not args.no_tunnel

    llama = tunnel = None
    try:
        # Tunnel first so the public /v1 link is printed right away, before the model loads.
        public = None
        if use_tunnel:
            binary = find_cloudflared(cfg["tunnel"].get("binary", "cloudflared"))
            tunnel, tunnel_ready, tunnel_url = start_tunnel(binary, base, cfg["tunnel"])
            tunnel_ready.wait(timeout=60)
            if not tunnel_url:
                sys.exit("cloudflared did not report a tunnel URL")
            public = tunnel_url[0]

        model_name = cfg["model"].get("alias") or "default"
        shown = public or base
        auth = f' -H "Authorization: Bearer {api_key}"' if api_key else ""
        print("\n" + "=" * 70)
        if public:
            print(f"Public  base URL : {public}/v1")
        print(f"Local   base URL : {base}/v1")
        print(f"Model            : {model_name}")
        print(f"API key          : {api_key or '(none)'}")
        print("\nTest (works once the model is loaded; returns 503 while loading):")
        print(
            f'curl {shown}/v1/chat/completions -H "Content-Type: application/json"{auth} '
            f"-d '{{\"model\": \"{model_name}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello\"}}]}}'"
        )
        if public and not cfg["tunnel"].get("token"):
            print("\nNote: Cloudflare quick tunnels do not support SSE, so streaming may stall."
                  " Set [tunnel] token + hostname for a named tunnel.")
        print("=" * 70 + "\n")

        print(f"[llama] {shlex.join(cmd)}")
        llama = subprocess.Popen(cmd, env=env)
        print("[llama] waiting for model download/load ...")
        wait_healthy(llama, base, server.get("startup_timeout", 3600))
        print(f"\n[ready] {shown}/v1  (model: {model_name})  Ctrl+C to stop.\n")

        while llama.poll() is None:
            if tunnel and tunnel.poll() is not None:
                print("[tunnel] cloudflared exited; server still reachable locally")
                tunnel = None
            time.sleep(1)
        sys.exit(f"llama-server exited with code {llama.returncode}")
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        stop(tunnel)
        stop(llama)


if __name__ == "__main__":
    main()
