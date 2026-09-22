"""Rescue weak labels by aligning the gate's learned appearance to the image.

Edge evidence runs out on a cropped or distant gate - too few clean stations to
say much. Appearance does not: the whole visible face still carries the
wordmark, the hexagon columns and the sponsor row, and those pin down position,
scale and rotation together. So nudge the four outer corners until the warped
face best matches the learned template.

Validation is deliberately kept separate: the template proposes, and EDGE
SUPPORT - which knows nothing about the template - decides whether the result
actually improved. Scoring the refinement with the objective it optimised would
prove nothing.
"""
from pathlib import Path
import glob, sys, time
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

SP = str(P.WORK)
SIZE, MARGIN = 192, 0.60
TEMPLATE = np.load(f"{SP}/gate_template.npy")
CANON = (A.CANONICAL_OUTER * SIZE * MARGIN + SIZE / 2).astype(np.float32)
WEIGHT = 1.0 / (1.0 + np.load(f"{SP}/gate_template_spread.npy"))
WEIGHT /= WEIGHT.mean()


def score(grey_image, outer):
    H, _ = cv2.findHomography(outer.astype(np.float32), CANON, 0)
    if H is None:
        return -1.0
    w = cv2.warpPerspective(grey_image, H, (SIZE, SIZE), flags=cv2.INTER_LINEAR)
    if w.std() < 1e-3:
        return -1.0
    w = (w - w.mean()) / (w.std() + 1e-6)
    return float((w * TEMPLATE * WEIGHT).mean())


def refine(grey_image, outer, side):
    """Coordinate descent on the four corners, coarse to fine."""
    best = outer.astype(np.float32).copy()
    best_score = score(grey_image, best)
    step = max(side * 0.045, 2.0)
    for _ in range(5):
        improved = True
        while improved:
            improved = False
            for corner in range(4):
                for axis in (0, 1):
                    for direction in (+1, -1):
                        trial = best.copy()
                        trial[corner, axis] += direction * step
                        value = score(grey_image, trial)
                        if value > best_score + 1e-5:
                            best, best_score, improved = trial, value, True
        step *= 0.5
        if step < 0.35:
            break
    return best, best_score


cfg = A.AutolabelConfig()
rows = []
start = time.time()
for path in P.frames()[::12]:
    img = cv2.imread(path)
    if img is None: continue
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    field = A.orange_field(img)
    for it in A.gate_candidates(img, cfg):
        if it.verdict != "review" or it.outer_side_px < 80: continue
        before_support, before_stations = A.edge_support(field, it.outer)
        before_ncc = score(grey, it.outer)
        fixed, after_ncc = refine(grey, it.outer, it.outer_side_px)
        after_support, after_stations = A.edge_support(field, fixed)
        moved = float(np.linalg.norm(fixed - it.outer, axis=1).mean())
        rows.append((before_support, after_support, before_ncc, after_ncc,
                     moved / max(it.outer_side_px, 1), before_stations, after_stations))

r = np.array(rows)
print(f"{len(r)} review instances refined in {time.time()-start:.0f}s")
print(f"  edge support (INDEPENDENT judge): before p50={np.median(r[:,0]):.3f}  "
      f"after p50={np.median(r[:,1]):.3f}")
print(f"  improved on independent judge: {100*(r[:,1] > r[:,0]).mean():.0f}%  "
      f"| worsened: {100*(r[:,1] < r[:,0]).mean():.0f}%")
print(f"  appearance (what was optimised): before p50={np.median(r[:,2]):.3f}  "
      f"after p50={np.median(r[:,3]):.3f}")
print(f"  corners moved, as fraction of gate side: p50={np.median(r[:,4]):.3f} "
      f"p90={np.percentile(r[:,4],90):.3f}")
gain = r[:,1] - r[:,0]
print(f"  support gain: p25={np.percentile(gain,25):+.3f} p50={np.median(gain):+.3f} "
      f"p75={np.percentile(gain,75):+.3f}")
