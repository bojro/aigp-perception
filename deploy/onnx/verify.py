"""Check a deployment before anything depends on it.

Run this on whatever machine is about to fly the model. It answers, in order,
the questions that have actually gone wrong before:

  1. Does the model load, and on which execution provider? A silent fall back
     to CPU is the failure that hides best -- it looks like sluggishness and
     it means the aircraft flies the gaps blind.
  2. How long does a frame take at the real stream size, not at 1920x1080?
  3. On real frames, how often does a gate turn into a pose? A detection that
     never solves is of no use to the state estimator.
  4. How often does the aim point come from the pose rather than from
     averaging the visible corners?

Exit status is non-zero if something would bite in flight, so this is usable
as a gate in a script.

    python deploy/onnx/verify.py --model models/gate_pose_hand497.onnx \
        --frames "~/gate frames" --limit 120 --budget-ms 33
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_detector import GateDetector          # noqa: E402
from gate_pose import solve_gate_pose           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--frames", type=str, default=None,
                    help="folder of stills; omitted means timing only")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--stream", default="640x360",
                    help="size the camera actually delivers")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--budget-ms", type=float, default=33.0)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="do not fail when no GPU provider is present")
    args = ap.parse_args()

    sw, sh = (int(v) for v in args.stream.lower().split("x"))
    problems: list[str] = []

    print(f"model      {args.model}")
    if not args.model.is_file():
        print(f"FAIL       no such file")
        return 2
    t0 = time.perf_counter()
    det = GateDetector(args.model, conf=args.conf,
                       pose_solver=solve_gate_pose)
    print(f"loaded     {time.perf_counter() - t0:.1f} s")
    print(f"provider   {det.provider}")
    if det.on_cpu and not args.allow_cpu:
        problems.append("running on CPU: no GPU provider was available")

    # Timing on synthetic frames of the right size, so this works with no data.
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, (sh, sw, 3), dtype=np.uint8)
    for _ in range(3):
        det.detect(noise)
    times = []
    for _ in range(30):
        t = time.perf_counter()
        det.detect(noise)
        times.append((time.perf_counter() - t) * 1000.0)
    p50, p95 = float(np.median(times)), float(np.percentile(times, 95))
    print(f"latency    p50 {p50:.1f} ms   p95 {p95:.1f} ms   "
          f"at {sw}x{sh}  (budget {args.budget_ms:.0f} ms)")
    if p95 > args.budget_ms:
        problems.append(f"p95 latency {p95:.1f} ms exceeds the "
                        f"{args.budget_ms:.0f} ms frame budget")

    if args.frames:
        folder = Path(args.frames).expanduser()
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:args.limit]
        if not files:
            print(f"FAIL       no images in {folder}")
            return 2
        seen = solved = from_pose = 0
        errs, rngs = [], []
        for fp in files:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            if img.shape[1] != sw:
                img = cv2.resize(img, (sw, sh))
            gates = det.detect(img)
            if not gates:
                continue
            seen += 1
            g = gates[0]
            if g.aim_is_from_pose:
                from_pose += 1
            if g.pose is not None:
                solved += 1
                errs.append(g.pose.reproj_err_px)
                rngs.append(g.pose.range_m)
        n = len(files)
        print(f"frames     {n} read from {folder}")
        print(f"detected   {seen} ({100.0 * seen / n:.0f}% of frames)")
        if seen:
            print(f"solved     {solved} ({100.0 * solved / seen:.0f}% of detections)")
            print(f"aim source {from_pose}/{seen} from pose, "
                  f"{seen - from_pose} from corner average")
        if errs:
            print(f"reproj px  p50 {np.median(errs):.2f}   p90 "
                  f"{np.percentile(errs, 90):.2f}")
            print(f"range m    p50 {np.median(rngs):.2f}   "
                  f"[{np.min(rngs):.1f}, {np.max(rngs):.1f}]")
        if seen and solved / seen < 0.4:
            problems.append(f"only {100.0 * solved / seen:.0f}% of detections "
                            "produced a pose")
        if seen == 0:
            problems.append("no gates detected in any frame")

    print()
    if problems:
        for p in problems:
            print(f"PROBLEM    {p}")
        return 1
    print("OK         nothing here would bite in flight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
