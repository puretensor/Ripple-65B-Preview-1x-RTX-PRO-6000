"""Ripple-65B Preview NVFP4 + DFlash2 on one RTX PRO 6000 Blackwell (SM120): SGLang 0.5.19 OpenAI-compatible server.

Ripple is a custom architecture, so stock SGLang cannot load it. Ricursive ships an SGLang model plugin inside the NVFP4
checkpoint (inference/local/model.py and ripple_format.py) and a DGX Spark launcher (inference/local/spark.py, an offline
engine for SM121). This file is that launcher turned into an HTTP server for any SM12x card:

  * registers the vendor RicursiveForCausalLM class and the ripple tool-call and reasoning parsers, loaded unmodified
    from the checkpoint you downloaded (start.sh checks their sha256 first)
  * applies the vendor's two Spark numerical fixes, at import so every spawned worker gets them:
      - PDL off in the DFlash acceptance softmax (SM12x PDL softmax can produce NaNs)
      - the export carries FP8 k/v scales although the KV cache is BF16: force them to 1.0
  * uses the vendor's Spark flags as defaults; anything after the two paths overrides them (argparse: last wins)

usage: python serve.py TARGET_DIR DRAFT_DIR [extra sglang flags]      (RIPPLE_NOSPEC=1 drops the drafter)
"""
import functools
import os
import shlex
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("SGLANG_FLASHINFER_USE_PAGED", "1")   # vendor Spark kernel choices
os.environ.setdefault("FLASHINFER_USE_CUDA_NORM", "1")

TARGET, DRAFT, *EXTRA = sys.argv[1:]
sys.path.insert(0, str(Path(TARGET) / "inference" / "local"))

import torch  # noqa: E402
from sglang.srt.models.registry import ModelRegistry  # noqa: E402

from model import RicursiveForCausalLM  # noqa: E402  (vendor plugin, from the checkpoint)
import ripple_format  # noqa: E402

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


def _patch():
    from sglang.kernels.ops.speculative.dspark import dspark_accept
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

    if not getattr(dspark_accept._flashinfer_softmax, "_ripple", False):
        softmax = dspark_accept._flashinfer_softmax

        @functools.wraps(softmax)
        def safe_softmax(*a, **k):
            k["enable_pdl"] = False
            return softmax(*a, **k)

        safe_softmax._ripple = True
        dspark_accept._flashinfer_softmax = safe_softmax

    if not getattr(ModelRunner.load_model, "_ripple", False):
        load_model = ModelRunner.load_model

        @functools.wraps(load_model)
        def load(self):
            result = load_model(self)
            for name, module in self.model.named_modules():
                if isinstance(module, RadixAttention):
                    for kind in ("k", "v"):
                        scale = getattr(module, kind + "_scale", None)
                        if scale is not None:
                            scale.data.fill_(1.0)
                            setattr(module, kind + "_scale_float", 1.0)
                weight = getattr(module, "weight", None)
                if self.is_draft_worker and weight is not None and weight.dtype != torch.bfloat16:
                    raise ValueError(f"Draft weight {name} must be BF16")
            return result

        load._ripple = True
        ModelRunner.load_model = load


_patch()

DEFAULT_ARGS = shlex.split(f"""
    --host 127.0.0.1 --port 8000 --served-model-name ripple --trust-remote-code
    --dtype bfloat16 --kv-cache-dtype bfloat16 --tp-size 1
    --attention-backend flashinfer --fp4-gemm-backend flashinfer_cutlass
    --flashinfer-autotune-skip-ops fp8_gemm
    --speculative-algorithm DFLASH --speculative-draft-model-quantization unquant
    --speculative-num-draft-tokens 16 --speculative-draft-window-size 16384
    --speculative-draft-attention-backend triton
    --context-length 32768 --mem-fraction-static 0.85
    --max-running-requests 1 --cuda-graph-bs-decode 1
    --cuda-graph-backend-prefill disabled
    --random-seed 42 --watchdog-timeout 1800
    --tool-call-parser ripple --reasoning-parser ripple --enable-metrics
    --chat-template {Path(TARGET) / "chat_template.jinja"}
""")
SPEC_FLAGS = {"--speculative-algorithm", "--speculative-draft-model-quantization", "--speculative-num-draft-tokens",
              "--speculative-draft-window-size", "--speculative-draft-attention-backend"}


def main():
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    base, spec = DEFAULT_ARGS, ["--speculative-draft-model-path", DRAFT]
    if os.environ.get("RIPPLE_NOSPEC") == "1":
        base = [x for i, x in enumerate(base) if x not in SPEC_FLAGS and (i == 0 or base[i - 1] not in SPEC_FLAGS)]
        spec = []
    args = prepare_server_args(base + ["--model-path", TARGET] + spec + EXTRA)
    print(f"[ripple] sglang args: {' '.join(base + spec + EXTRA)}", flush=True)
    try:
        launch_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
