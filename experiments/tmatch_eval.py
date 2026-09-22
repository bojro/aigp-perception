"""Does appearance matching separate good labels from bad? Test before trusting."""
from pathlib import Path
import glob, sys
from collections import defaultdict
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

SP = str(P.WORK)
SIZE, MARGIN = 192, 0.60
TEMPLATE = np.load(f"{SP}/gate_template.npy")
CANON = (A.CANONICAL_OUTER * SIZE * MARGIN + SIZE / 2).astype(np.float32)
WEIGHT = 1.0 / (1.0 + np.load(f"{SP}/gate_template_spread.npy"))
WEIGHT /= WEIGHT.mean()


def appearance(img, outer):
    H, _ = cv2.findHomography(outer.astype(np.float32), CANON, 0)
    if H is None: return -1.0
    w = cv2.warpPerspective(img, H, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    g = cv2.cvtColor(w, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.std() < 1e-3: return -1.0
    g = (g - g.mean()) / (g.std() + 1e-6)
    # Weighted correlation: trust the parts of the face that are consistent.
    return float((g * TEMPLATE * WEIGHT).mean())


cfg = A.AutolabelConfig()
buckets = defaultdict(list)
for path in P.frames()[::6]:
    img = cv2.imread(path)
    if img is None: continue
    for it in A.gate_candidates(img, cfg):
        if it.reason == "unlabelable": continue
        score = appearance(img, it.outer)
        key = it.verdict if not it.reason else f"rejected/{it.reason}"
        buckets[key].append(score)
        if it.verdict in ("auto", "review"):
            buckets[f"{it.verdict}:{it.anchor}"].append(score)

print(f"{'bucket':24s} {'n':>5s} {'ncc p10':>8s} {'p50':>8s} {'p90':>8s}")
for key in sorted(buckets):
    v = np.array(buckets[key])
    print(f"{key:24s} {len(v):5d} {np.percentile(v,10):8.3f} {np.median(v):8.3f} {np.percentile(v,90):8.3f}")

auto = np.array(buckets.get("auto", []))
rev = np.array(buckets.get("review", []))
if len(auto) and len(rev):
    for thr in (0.20, 0.30, 0.40, 0.50):
        print(f"  threshold {thr:.2f}: keeps {100*(auto>=thr).mean():5.1f}% of auto, "
              f"{100*(rev>=thr).mean():5.1f}% of review")
