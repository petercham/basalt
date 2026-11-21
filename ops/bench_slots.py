#!/usr/bin/env python3
"""Concurrency bench for the N slots, with prefill and decode measured separately.

Each request carries its own x-session-id, so the engine puts it in its own slot instead of
reusing one. Every prompt starts at a different offset in the corpus, so no request rides on
another's cached prefix: this measures the real thing, not a cache hit.

The response is read as a stream, so the client itself sees when the first token arrives:
  prefill = send -> first token   (the prompt being read, plus any queueing)
  decode  = first token -> last token
The two are reported apart: an "aggregate tok/s" that divides everything by the wall clock mixes
a 240K-token read with a 16K-token write and answers no question anyone asked.

  bench_slots.py --probe --chars 60000          # how many tokens that many characters make
  bench_slots.py --concurrency 4 --chars 787000 --max-tokens 16384
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # ops/
DOCS = ROOT.parent / "docs"                     # the repo's docs tree


def corpus() -> str:
    """The project's own documentation, glued together: real prose, varied, and long enough.

    The pages that quote tool-call markup are left out: a model writing a novel over them once
    wrote something the server read as a malformed tool call and dropped the stream, which costs
    a whole bench slot for a reason that has nothing to do with speed.
    """
    parts = []
    for path in sorted(DOCS.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "DSML" in text or "<tool_calls" in text:
            continue
        parts.append(text)
    text = "\n\n".join(parts)
    if len(text) < 200_000:                      # a short docs tree: repeat it, still varied per request
        text = (text * (200_000 // max(1, len(text)) + 1))
    return text


CORPUS = corpus()

# A task that cannot finish early: one paragraph per year, three thousand years, far past any token cap.
# The prose above it fills the context; this instruction makes the model keep writing to the limit.
INSTRUCTION = ("\n\n请以上面的资料为背景，写一部 100 章的长篇小说：每章以「第 N 章」开头，"
               "正文不少于 500 字，章节编号从 1 到 100 连续，不许跳过、不许合并、不许提前结束。")


def build_prompt(chars: int, seed: int) -> str:
    """`chars` characters starting at a different place for every seed, so no two share a prefix."""
    body = max(1000, chars - len(INSTRUCTION))
    if body >= len(CORPUS):
        text = CORPUS * (body // len(CORPUS) + 2)
        start = (seed * 7919) % (len(text) - body)
        return text[start:start + body] + INSTRUCTION
    start = (seed * 7919) % (len(CORPUS) - body)
    return CORPUS[start:start + body] + INSTRUCTION


def request(base: str, key: str, prompt: str, max_tokens: int, session: str, out: dict, idx: int,
            barrier: threading.Barrier) -> None:
    body = {
        "model": "qwen3.8-flash-next",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,                          # the client needs the first token's time, not just the total
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key,
                 "x-session-id": session},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never through a proxy
    barrier.wait()                                                          # every request starts together
    t0 = t_first = t_last = None
    usage, timings, finish = {}, {}, None
    n_chunks = 0
    t0 = time.time()
    try:
        with opener.open(req, timeout=7200) as r:
            for raw in r:                       # server-sent events, one JSON per "data:" line
                if not raw:
                    continue
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue                    # ": keep-alive" beats and blank separators
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                try:
                    ev = json.loads(payload)
                except ValueError:
                    continue
                now = time.time()
                if ev.get("usage"):
                    usage = ev["usage"]
                    timings = ev.get("timings") or {}
                choice = (ev.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    n_chunks += 1
                    if t_first is None:
                        t_first = now           # first visible token: prefill ends here
                    t_last = now
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        t_end = time.time()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        prefill = (t_first - t0) if t_first else (t_end - t0)
        decode = (t_last - t_first) if (t_first and t_last) else 0.0
        out[idx] = {
            "ok": True,
            "wall_s": round(t_end - t0, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "prefill_s": round(prefill, 2),
            "decode_s": round(decode, 2),
            "pp_tps": round(prompt_tokens / prefill, 1) if prefill > 0 else 0.0,
            "tg_tps": round(completion_tokens / decode, 1) if decode > 0 else 0.0,
            "t_first": t_first or t_end,
            "t_last": t_last or t_end,
            "chunks": n_chunks,
            "tg_tps_server": round(timings.get("predicted_per_second", 0), 1),
            "draft": f"{timings.get('draft_n_accepted', 0)}/{timings.get('draft_n', 0)}",
            "finish": finish,
        }
    except urllib.error.HTTPError as e:
        out[idx] = {"ok": False, "wall_s": round(time.time() - t0, 2),
                    "error": f"HTTP {e.code}: {e.read()[:300].decode('utf-8', 'replace')}"}
    except Exception as e:                                                 # noqa: BLE001
        out[idx] = {"ok": False, "wall_s": round(time.time() - t0, 2), "error": repr(e)}


def run(base: str, key: str, n: int, chars: int, max_tokens: int, prompt_chars: int | None) -> int:
    out: dict[int, dict] = {}
    barrier = threading.Barrier(n)
    threads = []
    t0 = time.time()
    run_seed = int(t0) % 100000                    # a fresh start for every run: no prefix cache left over
    for i in range(n):
        text = build_prompt(prompt_chars if prompt_chars else chars, seed=run_seed * 100 + i + 1)
        t = threading.Thread(target=request,
                             args=(base, key, text, max_tokens, f"bench-{int(t0)}-{i}", out, i, barrier))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0

    rows = [out.get(i, {"ok": False, "error": "no result"}) for i in range(n)]
    ok = [r for r in rows if r.get("ok")]
    print(f"\n=== {n} 路并发 · prompt 目标 {prompt_chars or chars} 字符 · max_tokens {max_tokens} ===")
    print(f"{'路':>3} {'HTTP':>4} {'输入tok':>8} {'输出tok':>8} {'首字s':>7} {'生成s':>7} "
          f"{'预填tok/s':>10} {'生成tok/s':>10} {'墙钟s':>7} {'finish':>7}")
    for i, r in enumerate(rows):
        if r.get("ok"):
            print(f"{i + 1:>3} {'200':>4} {r['prompt_tokens']:>8} {r['completion_tokens']:>8} "
                  f"{r['prefill_s']:>7.1f} {r['decode_s']:>7.1f} {r['pp_tps']:>10.0f} {r['tg_tps']:>10.1f} "
                  f"{r['wall_s']:>7.1f} {str(r['finish']):>7}")
        else:
            print(f"{i + 1:>3} {'ERR':>4}  {r.get('error', '')[:110]}")

    if ok:
        total_in = sum(r["prompt_tokens"] for r in ok)
        total_out = sum(r["completion_tokens"] for r in ok)
        # 预填：所有请求同时开跑，最后一路出首字时全部 prompt 才算读完
        pp_window = max(r["prefill_s"] for r in ok)
        pp_agg = total_in / pp_window if pp_window else 0
        pp_slot = statistics.median(r["pp_tps"] for r in ok)
        # 生成：从最早一路出首字，到最后一路收尾，这段里所有槽都在解码
        first = min(r["t_first"] for r in ok)
        last = max(r["t_last"] for r in ok)
        tg_window = last - first
        tg_agg = total_out / tg_window if tg_window > 0 else 0
        tg_slot = statistics.median(r["tg_tps"] for r in ok)
        print(f"\n成功 {len(ok)}/{n} · 输入合计 {total_in} tok · 输出合计 {total_out} tok · 墙钟 {wall:.1f}s")
        print(f"预填  合计 {total_in:>7} tok ÷ {pp_window:6.1f}s = {pp_agg:7.0f} tok/s"
              f"   单路中位 {pp_slot:7.0f} tok/s")
        print(f"生成  合计 {total_out:>7} tok ÷ {tg_window:6.1f}s = {tg_agg:7.1f} tok/s"
              f"   单路中位 {tg_slot:7.1f} tok/s")
        return 0 if len(ok) == n else 1
    return 1


def probe(base: str, key: str, chars: int) -> int:
    """One short request: how many prompt tokens that many characters cost."""
    prompt = build_prompt(chars, seed=1)
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4, "temperature": 0, "stream": False}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + key, "x-session-id": "bench-probe"},
                                 method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t0 = time.time()
    try:
        with opener.open(req, timeout=1800) as r:
            d = json.load(r)
        u = d.get("usage") or {}
        print(json.dumps({"chars": chars, "prompt_tokens": u.get("prompt_tokens"),
                          "completion_tokens": u.get("completion_tokens"),
                          "wall_s": round(time.time() - t0, 2),
                          "timings": d.get("timings")}, ensure_ascii=False))
        return 0
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.read()[:400].decode('utf-8', 'replace')}", file=sys.stderr)
        return 1
    except Exception as e:                                                 # noqa: BLE001
        print(repr(e), file=sys.stderr)
        return 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8787")
    ap.add_argument("--key", required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--chars", type=int, default=64000, help="prompt length in characters")
    ap.add_argument("--prompt-chars", type=int, default=None, help="override for this run")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.probe:
        return probe(a.base, a.key, a.chars)
    return run(a.base, a.key, a.concurrency, a.chars, a.max_tokens, a.prompt_chars)


if __name__ == "__main__":
    raise SystemExit(main())
