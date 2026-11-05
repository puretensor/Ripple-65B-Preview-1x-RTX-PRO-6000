# Ripple-65B Preview on one RTX PRO 6000 Blackwell

**135.2 tok/s on code and 97.9 on reasoning, batch 1, from a dense 65B on one card.** Without a drafter it does 24.0.
Ricursive's NVFP4 weights and their DFlash2 drafter, on SGLang 0.5.19, with the vendor's own flags.

On our own long-output prompts it holds **75.9 tok/s writing an essay and a poem** and **130.8 tok/s writing a
1,000-line Python script**.

Every LocalMaxxing number below is a run on [LocalMaxxing](https://www.localmaxxing.com), linked, so you can check the
output, the prompt and the exact launch command. `./start.sh` starts the same server.

**Recipe page, with the numbers explained:** https://puretensor.systems/models/Ripple-65B-Preview-1x-RTX-PRO-6000/

```bash
git clone https://github.com/puretensor/Ripple-65B-Preview-1x-RTX-PRO-6000
cd Ripple-65B-Preview-1x-RTX-PRO-6000
./start.sh          # builds the Python environment, downloads 62 GB of weights, starts SGLang, smoke-tests it
python3 bench.py    # measures it the way LocalMaxxing does
```

## Performance

One RTX PRO 6000 Blackwell Workstation Edition (96 GB, 600 W), batch 1, temperature 0, median of 5 runs after 2
warmups. Ripple always reasons when it needs to (there is no thinking switch), so reasoning tokens count as output.

| Profile | Drafter | code-v1 tok/s | reasoning-v1 tok/s | Tokens per step | Our prompts: essay / script |
|---|---|---:|---:|---|---:|
| `dflash` (default) | DFlash2, 16-token blocks | [**135.2**](https://www.localmaxxing.com/runs/cmw2sfbvd0bcdoa01shm40fuh) | [**97.9**](https://www.localmaxxing.com/runs/cmw2sfbi20bcaoa01hqj3newo) | 6.9 / 5.1 at 51.6 ms | [75.9](https://www.localmaxxing.com/runs/cmw2xzw1n0bgeoa01atqkp6l2) / [130.8](https://www.localmaxxing.com/runs/cmw2xzwmh0bghoa017hrwhnx9) |
| `nospec` | none | [24.0](https://www.localmaxxing.com/runs/cmw2s16bu0bc7oa01jms6k4uk) | [24.0](https://www.localmaxxing.com/runs/cmw2s15m70bc4oa01fzbv9hen) | 1 at 41.7 ms | |

- **code-v1 and reasoning-v1** are LocalMaxxing's canonical prompts, and those four runs are Verified.
- **Our prompts** are "write an essay and a poem about regret" (2,168 tokens out) and "write a simple python
  script 1000 lines" (16,384, the cap tokens out). LocalMaxxing marks them unverified only because they are not canonical.
- **Long prompts.** With a 2,644-token prompt it decodes at [96.8 tok/s](https://www.localmaxxing.com/runs/cmw2sfc5l0bcgoa01poovjnf5)
  and prefills at 3.4K tok/s. With 15,699 tokens it decodes at [98.5](https://www.localmaxxing.com/runs/cmw2sfcs00bcjoa01ciqggo8d),
  prefills at 3.5K tok/s and shows its first token after 4.5 s.
- **In a coding agent.** Through our agent harness (omp, with about 18K tokens of system prompt and tools in front of
  every request, and its own sampling settings) we saw a steady 50-70 tok/s on prose and 110-120 on long code after
  warmup. The longer context and non-greedy sampling both lower the tokens accepted per step.

## Is it any good? Early signs, not a thorough test

We ran a short quality check on 4 November, on the BF16 weights (not this NVFP4 build), paired item by item with
DeepSeek-V4.1-Flash on the same harness:

| Suite | Ripple-65B Preview | DeepSeek-V4.1-Flash |
|---|---:|---:|
| Private coding set: 20 real defects from our own code, fixed with a regression test | 19/20 | 19/20 |
| Reasoning mix (40 items) | 30/40 | 32/40 |
| of which AIME 2026 | 7/8 | 7/8 |
| of which HMMT Feb 2026 | 5/8 | 4/8 |
| of which GPQA-Diamond, 16K-token budget | 18/24 | 21/24 |

- The reasoning gap is within noise (McNemar p = 0.63). 9 of Ripple's 10 misses ran out of the token budget; on the
  items it finished it was 30/31.
- Tools (auto, required, parallel, a full round trip), JSON-schema output and a 30K-token needle all worked.
- This is a small sample on one harness, not a thorough evaluation. Treat it as "worth your time", not as a ranking.

## Why these numbers

**Without drafts, 24.0 tok/s is what the weights allow.**
- Ripple-65B Preview is dense: 64.8B parameters, 80 layers. Every generated token reads every weight once.
- Ricursive's NVFP4 export quantizes the MLPs to 4 bits (28 GB) and keeps attention in BF16 (27 GB). With the 2 GB
  output head, one forward pass reads about 57 GB.
- At the card's 1.79 TB/s that caps decode near 31 tok/s. Measured: 41.7 ms per token, 24.0 tok/s, 76% of the cap.

**DFlash2 turns one pass into several tokens.**
- [DFlash2](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) is Ricursive's 4.04B, 5-layer BF16 drafter.
  It drafts a whole 16-token block at once from the target's hidden states (the anchor plus 15 proposals), and one
  target pass checks all of them.
- A verify step reads the target plus the drafter, about 65 GB, and takes 51.6 ms.
- tok/s = tokens per step / seconds per step. On code-v1, 6.9 tokens land per step: 6.9 / 0.0516 s = 134 tok/s.
  Reasoning text is less predictable: 5.1 per step, 97.9 tok/s. The essay lands 3.9 per step and the
  1,000-line script 6.9.
- Greedy verification is lossless: at temperature 0 the output is what the model alone would write.

**There is little left in flags.** A torch-profiler capture of 30 verify steps showed BF16 matrix multiplies (attention
projections, output head, drafter) at 50% of GPU time and the FP4 MLP multiplies at 38%, both streaming at about
1.46 TB/s, 82% of the card's peak. The remaining lever is fewer bytes per step without losing acceptance, for example a
calibrated FP8 or NVFP4 attention export. That is a quantization project, not a flag.

**What we tried** (code-v1 / reasoning-v1 tok/s, screening runs of 3 after 1 warmup, so read them against each other):

| Change from the vendor flags | code-v1 | reasoning-v1 | ms per step | Result |
|---|---:|---:|---:|---|
| none (`./start.sh`) | 135.6 | 99.7 | 51.6 | kept |
| 32-token draft blocks | 130.0 | 101.9 | 53 | the checkpoint drafts 16-token blocks |
| Marlin FP4 GEMM | 133.1 | 102.0 | 51.6 | equal, slower prefill |
| cuDNN FP4 GEMM | 124.2 | 93.6 | 55 | slower |
| Triton attention | 112.8 | 91.1 | 55 | slower |
| Online FP8 attention projections | 116.7 | 95.6 | 53 | fewer bytes, but 6.2 tokens per step instead of 7.0 |
| Decode-mode verify attention | 133.0 | 100.0 | 51.2 | equal |
| cuteDSL FP4, torch.compile, FP8 drafter | - | - | - | no SM120 support / CUDA-graph capture fails / the vendor loader requires a BF16 drafter |

## Requirements

- Linux x86-64 and one RTX PRO 6000 Blackwell (96 GB) with about 85 GB free. These numbers are for the 600 W
  Workstation Edition. We have not measured the 325 W Max-Q.
- NVIDIA driver 615.71.09 (tested) and a CUDA 13 toolkit with `nvcc`: SGLang compiles some kernels on first start.
  We used 13.2. CUDA 13.0's `nvcc` fails against glibc 2.43 (for example Ubuntu 26.04).
- [uv](https://docs.astral.sh/uv/). `start.sh` builds `.venv` from `requirements.lock` (about 10 GB).
- Disk for the weights: 54 GB (NVFP4) plus 7.6 GB (DFlash2).

## Usage

```bash
./start.sh --list              # profiles and their measured numbers
./start.sh                     # NVFP4 + DFlash2 (default)
./start.sh nospec              # no drafter, the reference
./start.sh dflash --print      # print the exact server command, start nothing
GPU=1 PORT=8001 ./start.sh     # pick the card and port
CONTEXT=32768 ./start.sh       # the context of the Verified runs (default 65536; decode speed is the same)
MODELS_DIR=/data/models ./start.sh
./stop.sh
```

The server is OpenAI-compatible at `http://127.0.0.1:8000/v1`, model name `ripple`, with tool calls and
`reasoning_content` parsed by Ricursive's parsers. Bind it beyond localhost with `HOST=0.0.0.0`. It serves one request
at a time: the CUDA graphs are captured for batch 1, and further requests queue.

Pinned versions:

| What | Pin |
|---|---|
| Engine | SGLang 0.5.19, torch 2.13.0+cu130, flashinfer 0.6.18 from pip; the full environment is `requirements.lock` |
| NVFP4 weights | [`ricursiveAI/Ripple-65B-Preview-NVFP4`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-NVFP4) at `99eea447` |
| Drafter | [`ricursiveAI/Ripple-65B-Preview-DFlash2`](https://huggingface.co/ricursiveAI/Ripple-65B-Preview-DFlash2) at `35609b3b` |
| Model plugin | `inference/local/model.py` and `ripple_format.py` from the NVFP4 checkpoint, sha256-checked before start |

`serve.py` is Ricursive's DGX Spark launcher (`inference/local/spark.py`) turned into an HTTP server for any SM12x
card. It registers their SGLang model class and parsers, applies their two Spark numerical fixes (PDL off in the DFlash
acceptance softmax; FP8 k/v scales forced to 1.0 because the KV cache is BF16), and uses their Spark flags.

## How we measured

`bench.py` is the measurement, so you can compare your card with ours:
- **Prompts:** LocalMaxxing's canonical code-v1 and reasoning-v1 (fetched and hash-checked), plus our `essay` and
  `script`, each behind LocalMaxxing's cache-bust nonce so no prefix cache helps.
- **Settings:** batch 1, temperature 0, 2 warmups, median of 5. Output is capped at 2,048 tokens on the canonical
  prompts, as in our Verified runs, and at 16,384 on ours.
- **Decode:** (output tokens - 1) / (time from first to last token).
- **Tokens per step:** from SGLang's per-request speculative counts (`return_spec_tokens_details`), with milliseconds
  per step, the two numbers that set decode speed with drafts on.
- The Verified runs had our CPU in a performance window (performance governor, boost on). The essay and script runs had
  it in its normal state.

**How we picked the config:** one change at a time against the vendor's flags, compared on milliseconds per verify step
because acceptance moves run to run, plus the profiler capture above. Nothing beat the vendor's flags, so there was no
gain to confirm with fresh-boot A/B runs.

{{FRESH}}

## Credits

- [Ricursive Intelligence](https://www.ricursive.com/ripple) for Ripple-65B Preview, the NVFP4 export, the DFlash2
  drafter and the SGLang plugin and Spark launcher this server is built from.
- [SGLang](https://github.com/sgl-project/sglang) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
- [LocalMaxxing](https://www.localmaxxing.com) for the canonical prompts and verification.

The weights and Ricursive's plugin files are Apache-2.0 (see the model cards). They download from Hugging Face and are not
redistributed here. The scripts in this repository are MIT.

By [PureTensor](https://puretensor.ai). Results are posted on X as [@Dataphil3](https://x.com/Dataphil3).
