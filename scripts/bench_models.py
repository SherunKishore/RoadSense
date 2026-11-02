"""Size, compute and speed of the three detectors under one setup (batch 1, 640 x 640, fused models).

  params / GFLOPs : Ultralytics count (thop) and PyTorch's FlopCounterMode, which also counts the matmuls inside
                    attention blocks that thop skips
  GPU latency     : FP16, CUDA, 50 warm-up + 300 timed forward passes (median)
  CPU latency     : FP32, 5 warm-up + 30 timed forward passes (median)

    python scripts/bench_models.py            # writes runs/bench_models.json
"""
import json
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bsw_yolo  # noqa: E402,F401  (BSW-YOLO11 classes)
import gc_yolo  # noqa: E402,F401  (YOLO11 + GIA + CDSA classes)
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

MODELS = {"YOLO11s": "runs/abl2_a0_base", "GGC-YOLO": "runs/gdgf_gia_cdsa_s", "BSW-YOLO11s": "runs/final_bsw_s"}


def timed(net, x, warm, n):
    with torch.inference_mode():
        for _ in range(warm):
            net(x)
        out = []
        for _ in range(n):
            if x.is_cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            net(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(out), 1)


def main() -> None:
    res = {}
    for name, run in MODELS.items():
        net = YOLO(str(ROOT / run / "weights" / "best.pt")).model.float().eval().fuse()
        x = torch.zeros(1, 3, 640, 640)
        with FlopCounterMode(display=False) as fc, torch.inference_mode():
            deepcopy(net)(x)
        r = {"params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2),
             "gflops_thop": round(get_flops(net, 640), 1), "gflops_torch": round(fc.get_total_flops() / 1e9, 1),
             "cpu_fp32_ms": timed(net, x, 5, 30)}
        if torch.cuda.is_available():
            g = deepcopy(net).cuda().half()
            r["gpu_fp16_ms"] = timed(g, x.cuda().half(), 50, 300)
            del g
            torch.cuda.empty_cache()
        res[name] = r
        print(name, r)
    res["_setup"] = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "torch": torch.__version__, "threads": torch.get_num_threads()}
    (ROOT / "runs" / "bench_models.json").write_text(json.dumps(res, indent=1))
    print("Saved runs/bench_models.json")


if __name__ == "__main__":
    main()
