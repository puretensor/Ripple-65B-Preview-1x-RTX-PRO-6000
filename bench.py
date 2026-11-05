#!/usr/bin/env python3
"""Batch-1 decode benchmark for Ripple-65B Preview, measured the way LocalMaxxing verifies a run.

Cases:
  code-v1, reasoning-v1   LocalMaxxing's canonical prompts, fetched from localmaxxing.com/api/agent-context and
                          checked against their sha256
  essay                   "write an essay and a poem about regret"   (long-form prose)
  script                  "write a simple python script 1000 lines"   (long code)

Each case streams at temperature 0, batch 1: 2 warmups discarded, then the median of 5. Output is capped at 2,048
tokens on the canonical prompts (as in our Verified runs) and 16,384 on ours, so the script can finish.
Ripple always reasons (adaptive; the template has no switch), so reasoning tokens count as output.

  decode tok/s     = (completion_tokens - 1) / (t_last_token - t_first_token)
  prefill tok/s    = prompt_tokens / time to first token
  tokens per step  = (completion_tokens - 1) / verify steps        (from SGLang's per-request spec details)
  ms per step      = decode time / verify steps                    (so decode tok/s = tokens per step / s per step)

usage: python3 bench.py [--url http://127.0.0.1:8000] [--cases essay,script,code-v1,reasoning-v1] [--out results.jsonl]
"""
import argparse, hashlib, json, secrets, statistics, time, urllib.request

OWN = {
    "essay": "write an essay and a poem about regret",
    "script": "write a simple python script 1000 lines",
}

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="ripple")
ap.add_argument("--out", default="results.jsonl")
ap.add_argument("--label", default="ripple-nvfp4-dflash2")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=None, help="default: 2048 canonical, 16384 ours")
ap.add_argument("--cases", default="essay,script,code-v1,reasoning-v1")
a = ap.parse_args()


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ripple-recipe-bench/1.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode()


def run(prompt_text, max_tokens):
    # LocalMaxxing's cache-bust prefix: a fresh nonce per request so no prefix cache can help.
    nonce = f"[LocalMaxxing cache-bust nonce: {secrets.token_hex(16)}]\n"
    body = {"model": a.model, "messages": [{"role": "user", "content": nonce + prompt_text}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "return_spec_tokens_details": True}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = t_last = None; usage = None; text = []; spec = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            if (ev.get("sglext") or {}).get("spec_tokens_details"):
                spec = ev["sglext"]["spec_tokens_details"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    text.append(piece)
    ct, pt = usage["completion_tokens"], usage["prompt_tokens"]
    ttft, dec = t_first - t0, t_last - t_first
    out = "".join(text)
    rec = {"label": a.label, "prompt_tokens": pt, "completion_tokens": ct,
           "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "ttft_ms": ttft * 1000, "tok_s_out": (ct - 1) / dec, "tok_s_prefill": pt / ttft,
           "tok_s_total": (pt + ct) / (t_last - t0), "spec": {},
           "output_sha256": hashlib.sha256(out.encode()).hexdigest(),
           "output_sample": out[:4000], "prompt_sample": (nonce + prompt_text)[:2000]}
    if spec and spec.get("spec_verify_ct"):
        steps = float(spec["spec_verify_ct"])
        rec["spec"] = {"drafts": steps, "draft_tokens": float(spec["spec_num_proposed_drafts"]),
                       "accepted_tokens": float(spec["spec_num_correct_drafts"])}
        rec["mean_accept_len"] = 1 + rec["spec"]["accepted_tokens"] / steps
        rec["accept_rate"] = rec["spec"]["accepted_tokens"] / max(rec["spec"]["draft_tokens"], 1)
        rec["tokens_per_step"] = (ct - 1) / steps
        rec["ms_per_step"] = dec * 1000 / steps
    return rec


prompts = dict(OWN)
if any(c not in OWN for c in a.cases.split(",")):
    ctx = json.loads(get("https://www.localmaxxing.com/api/agent-context"))
    for p in ctx["canonicalPrompts"]:
        assert hashlib.sha256(p["text"].encode()).hexdigest() == p["sha256"], f"{p['id']}: sha256 mismatch"
        prompts[p["id"]] = p["text"]

with open(a.out, "a") as f:
    for name in a.cases.split(","):
        cap = a.max_tokens or (16384 if name in OWN else 2048)
        for _ in range(a.warmup):
            run(prompts[name], cap)
        recs = []
        for i in range(a.runs):
            r = run(prompts[name], cap)
            r.update(case=name, i=i, prompt_sha256=hashlib.sha256(prompts[name].encode()).hexdigest())
            recs.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        med = lambda k: statistics.median(x[k] for x in recs if x.get(k) is not None) if any(x.get(k) is not None for x in recs) else None
        line = (f"{name:13s} prompt={recs[0]['prompt_tokens']:4d} out={med('completion_tokens'):6.0f}  "
                f"decode={med('tok_s_out'):6.1f} tok/s (min {min(x['tok_s_out'] for x in recs):.1f}, "
                f"max {max(x['tok_s_out'] for x in recs):.1f})  ttft={med('ttft_ms'):6.1f} ms")
        if med("tokens_per_step"):
            line += f"  {med('tokens_per_step'):.2f} tok/step at {med('ms_per_step'):.1f} ms/step"
        print(line, flush=True)
