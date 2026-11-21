"""A/B the speculative window (`spec`): restart per value, one long prompt, print a table.

Values 0 (no speculation), 2, 4, 8 (engine default for an MTP drafter).  Short decode
(2048 tokens) keeps each round around a minute; prefill is the same 24万 tokens every time.
The caller restarts the service afterwards - this script leaves no server behind.
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
LOG = ROOT / "ab_spec_engine.log"
CHARS = 787000
OUT_TOKENS = 2048


def keys() -> str:
    out = subprocess.run([str(PY), str(ROOT / "keys-arg.py"), str(ROOT)],
                         capture_output=True, text=True, cwd=ROOT)
    return out.stdout.strip()


def kill() -> None:
    for exe in ("basalt.exe", "basalt-vision.exe"):
        subprocess.run(["taskkill", "/IM", exe, "/T", "/F"], capture_output=True)
    time.sleep(3)


def patch(value) -> None:
    data = json.loads(CFG.read_text(encoding="utf-8"))
    if value is None:
        data["engine"].pop("spec", None)
    else:
        data["engine"]["spec"] = value
    CFG.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")


def start() -> None:
    env = dict(os.environ)
    env["PATH"] = r"C:\Work\cuda130\Library\bin;" + env.get("PATH", "")
    log = LOG.open("w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        [str(ENGINE), "serve", "--config", str(CFG), "--python", str(PY),
         "--api-key", keys()], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env=env)
    deadline = time.time() + 300
    while time.time() < deadline:
        time.sleep(5)
        text = LOG.read_text(encoding="utf-8", errors="replace")
        if "slots    par=" in text:
            time.sleep(8)
            return
        if proc.poll() is not None:
            return
    return


def bench() -> tuple[float, float]:
    out = subprocess.run(
        [str(PY), str(Path(__file__).resolve().parent / "bench_slots.py"), "--key", keys().split(",")[0],
         "--concurrency", "1", "--prompt-chars", str(CHARS),
         "--max-tokens", str(OUT_TOKENS)],
        capture_output=True, text=True, cwd=ROOT)
    text = out.stdout
    pre = re.search(r"预填\s+合计\s+\d+ tok ÷\s+([\d.]+)s =\s+(\d+) tok/s", text)
    dec = re.search(r"生成\s+合计\s+\d+ tok ÷\s+([\d.]+)s =\s+([\d.]+) tok/s", text)
    if not pre:
        print(text[-600:], file=sys.stderr)
        return (0.0, 0.0)
    return (float(pre.group(2)), float(dec.group(2)) if dec else 0.0)


def main() -> None:
    values = [0, 2, 4, 8]
    rows = []
    for value in values:
        kill()
        patch(value)
        start()
        pre, dec = bench()
        print(f"spec={value:<3} 预填 {pre:>6.0f} tok/s   生成 {dec:>6.1f} tok/s",
              flush=True)
        rows.append((value, pre, dec))
    kill()
    patch(8)
    print("\n=== 汇总 ===")
    for value, pre, dec in rows:
        print(f"spec={value:<3} 预填 {pre:>6.0f}   生成 {dec:>6.1f}")


if __name__ == "__main__":
    main()
