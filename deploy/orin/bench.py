"""Inference-only latency on the Orin, which is the number a laptop cannot give.

Excludes JPEG decode and JSON writing so what is left is the cost the flight
loop actually pays per frame. Warms up first: the first call builds kernels and
is representative of nothing.

Measured on an Orin NX 16GB, JetPack 6 / L4T R36.4.3, 25W power mode:
both models land at 27.4 ms p50 and about 28 ms p95, so roughly 36 fps each.
FP16 made no measurable difference. That fits a 33 ms frame budget, but it is
inference alone -- no capture, no PnP, no control -- and running both models
on the same frame costs about 61 ms, which is a viewing configuration rather
than a flight one.
"""
import sys
import time

sys.path.insert(0, "/home/dcl/gate-inference")
import numpy as np
import torch
from inference import GateDetector

W, H = 640, 360
frame = np.random.default_rng(0).integers(0, 255, (H, W, 3)).astype(np.uint8)
for name, weights in (("teammate", "/home/dcl/gate-inference/best.pt"),
                      ("hybrid",
                       "/home/dcl/gate-compare/models/gate_pose_hybrid_v1.pt")):
    for half in (False, True):
        det = GateDetector(weights, imgsz=640, conf=0.4, device="0", half=half)
        for _ in range(15):
            det.detect(frame)
        torch.cuda.synchronize()
        times = []
        for _ in range(80):
            t = time.perf_counter()
            det.detect(frame)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        times = np.array(times)
        print(f"{name:9} half={str(half):5} p50 {np.median(times):6.2f} ms  "
              f"p95 {np.percentile(times, 95):6.2f} ms  "
              f"max {times.max():6.2f} ms  -> {1000 / np.median(times):5.1f} fps")
        del det
        torch.cuda.empty_cache()
