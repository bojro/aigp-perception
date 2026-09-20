"""A/B harness. Metric: how far two independently found rings disagree about
the gate's known 1500/2700 geometry, before any rectification hides it.

Reports the instance count alongside, because a variant can always look
accurate by quietly dropping every hard gate.
"""
import sys
from pathlib import Path, glob, importlib, dataclasses
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A

SRC = "/Users/bojro/Downloads/gate frames"
FRAMES = sorted(glob.glob(f"{SRC}/*.jpg"))[::6]


def run(tag, **overrides):
    importlib.reload(A)
    cfg = dataclasses.replace(A.AutolabelConfig(), **overrides)
    res, sides, kept, both = [], [], 0, 0
    for path in FRAMES:
        img = cv2.imread(path)
        for item in A.gate_candidates(img, cfg):
            if item.verdict == "rejected":
                continue
            kept += 1
            if item.anchor == "both" and np.isfinite(item.residual):
                both += 1
                res.append(item.residual)
                sides.append(item.outer_side_px)
    res, sides = np.array(res), np.array(sides)
    if not len(res):
        print(f"{tag:28s} no paired instances"); return None
    px = res * sides
    print(f"{tag:28s} kept={kept:4d} both={both:4d} | residual "
          f"p50={np.median(res):.4f} p75={np.percentile(res,75):.4f} "
          f"p90={np.percentile(res,90):.4f} | px p50={np.median(px):5.1f} "
          f"p90={np.percentile(px,90):5.1f}")
    return np.median(res)


if __name__ == "__main__":
    print(f"frames={len(FRAMES)}\n")
    run("baseline")
    print()
    for k in (3, 5, 7, 9, 13):
        run(f"close_kernel={k}", close_kernel=k)
    print()
    for k in (0, 3, 5, 7):
        run(f"open_kernel={k}", open_kernel=k)
    print()
    for w in (3, 5, 7, 11, 15):
        run(f"subpix_window={w}", subpix_window=w)
