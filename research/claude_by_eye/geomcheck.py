"""Score both label sets against the gate's physical geometry, not each other.

Given the four outer corners, the homography to the canonical front face is
fixed, and the 1500/2700 opening has exactly one place it can project to. How
far the labelled opening sits from that spot is a measure of the label alone -
no reference to any other labeller. Reported as a fraction of gate side so it
is comparable across distances.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception.autolabel_gate_pose import opening_residual, mean_side_length
from aigp_perception import paths as P

SP = P.WORK
W, H = 1920, 1080

CLAUDE = {
 "0919_214639_000087": [(254,73),(561,97),(551,381),(242,372),(315,147),(472,158),(468,300),(312,295)],
 "0919_214639_000073": [(365,114),(600,117),(600,355),(363,352),(408,171),(553,172),(552,305),(407,304)],
 "0919_220524_000277": [(572,74),(706,96),(688,245),(546,232),(592,120),(668,132),(658,212),(585,202)],
}

def load(stem):
    out = []
    for line in (SP/"full"/"train"/"labels"/f"{stem}.txt").read_text().split("\n"):
        if not line.strip(): continue
        kp = np.array([float(v) for v in line.split()[5:]]).reshape(8, 3)
        out.append(kp[:, :2] * [W, H])
    return out

print(f"{'frame':22s} {'labeller':10s} {'side_px':>8s} {'residual':>9s} {'miss_px':>8s}")
print("-" * 62)
for stem, guess in CLAUDE.items():
    mine = np.array(guess, np.float32) * 2.0
    theirs = min(load(stem), key=lambda p: np.linalg.norm(p.mean(0) - mine.mean(0)))
    for name, pts in (("claude", mine), ("algorithm", theirs)):
        pts = pts.astype(np.float32)
        res, _ = opening_residual(pts[:4], pts[4:])
        side = mean_side_length(pts[:4])
        print(f"{stem:22s} {name:10s} {side:8.0f} {res:9.4f} {res*side:8.1f}")
    print()
