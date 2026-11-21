"""A/B --prefill-cpu-share: restart the engine per value, measure one long prompt, print a table.

Run in the foreground of a background shell: it leaves no server behind (the last value is
killed too), so the caller restarts the service afterwards.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:/Work/LLM/Basalt")
ENGINE = ROOT / "runtime/basalt-v0.5.0-windows-x64/bin/basalt.exe"
PY = ROOT / ".venv/Scripts/python.exe"
CFG = Path(r"C:/Work/Projects/basalt-fork") / "launch-history/iq3_s-6x256k.json"
LOG = ROOT / "ab_prefill_engine.log"


def keys() -> str:
    out = subprocess.run([str(PY), str(ROOT / "keys-arg.py"), str(ROOT)],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip()


def kill() -> None:
    for exe in ("basalt.exe", "basalt-vision.exe"):
        subprocess.run(["taskkill", "/IM", exe, "/T", "/F"],
                       capture_output=True)
    time.sleep(3)


def patch(value) -> None:
    data = json.loads(CFG.read_text(encoding="utf-8"))
    if value is None:
        data["engine"].pop("prefill-cpu-share", None)
    else:
        data["engine"]["prefill-cpu-share"] = value
    CFG.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")


def start() -> subprocess.Popen:
    env = dict(os.environ)
    env["PATH"] = r"C:\Work\cuda130\Library\bin;" + env.get("PATH", "")
    log = LOG.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [str(ENGINE), "serve", "--config", str(CFG), "--python", str(PY),
         "--api-key", keys()], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env=env)
    deadline = time.time() + 240
    while time.time() < deadline:
        time.sleep(5)
        text = LOG.read_text(encoding="utf-8", errors="replace")
        if "slots    par=" in text or "EXITCODE" in text:
            time.sleep(8)
            return proc
        if proc.poll() is not None:
            return proc
    return proc


def bench(tokens: int) -> tuple[float, float]:
    out = subprocess.run(
        [str(PY), str(Path(__file__).resolve().parent / "bench_slots.py"), "--key", keys().split(",")[0],
         "--concurrency", "1", "--prompt-chars", "787000",
         "--max-tokens", str(tokens)],
        capture_output=True, text=True, cwd=ROOT)
    text = out.stdout
    pre = re.search(r"预填\s+合计\s+\d+ tok ÷\s+([\d.]+)s =\s+(\d+) tok/s", text)
    dec = re.search(r"生成\s+合计\s+\d+ tok ÷\s+([\d.]+)s =\s+([\d.]+) tok/s", text)
    if not pre:
        print(text[-800:], file=sys.stderr)
        return (0.0, 0.0)
    return (float(pre.group(2)), float(dec.group(2)) if dec else 0.0)


def main() -> None:
    values = [None, 0, 0.3, 0.9]      # None = engine default (0.6)
    rows = []
    for value in values:
        kill()
        patch(value)
        start()
        pre, dec = bench(256)
        label = "默认(0.6)" if value is None else str(value)
        print(f"share={label:<8} 预填 {pre:>6.0f} tok/s   生成 {dec:>6.1f} tok/s",
              flush=True)
        rows.append((label, pre, dec))
    kill()
    patch(None)
    print("\n=== 汇总 ===")
    for label, pre, dec in rows:
        print(f"share={label:<8} 预填 {pre:>6.0f}   生成 {dec:>6.1f}")


if __name__ == "__main__":
    main()
