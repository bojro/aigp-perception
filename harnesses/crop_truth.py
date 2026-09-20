"""How far can a clipped gate's corners be trusted? Measure it against truth.

Take gates the detector reads confidently with every point on screen, crop the
image so corners fall outside it, re-detect on the crop, and compare what came
back with what the uncropped frame said. The uncropped reading is not perfect
truth, but it is an independent, well-conditioned measurement of the same gate,
which is exactly what is missing for clipped gates.
"""
import glob, sys
from collections import defaultdict
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A

import dataclasses
JOINT = sys.argv[1] == 'on' if len(sys.argv) > 1 else True
cfg = dataclasses.replace(A.AutolabelConfig(), joint_refine=JOINT)
rows = []
frames = sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg"))[::4]

for path in frames:
    img = cv2.imread(path)
    if img is None: continue
    H, W = img.shape[:2]
    refs = [it for it in A.gate_candidates(img, cfg)
            if it.verdict == "auto" and not it.clipped and it.support >= 0.85
            and it.outer_side_px > 200]
    if not refs: continue
    ref = max(refs, key=lambda it: it.outer_side_px)
    truth = ref.points.copy()

    x0, y0 = truth[:4, 0].min(), truth[:4, 1].min()
    x1, y1 = truth[:4, 0].max(), truth[:4, 1].max()
    side = ref.outer_side_px
    for cut in (0.12, 0.25, 0.40, 0.55):
        # Slice off the left and top of the gate by `cut` of its width.
        cx = int(np.clip(x0 + cut * (x1 - x0), 1, W - 50))
        cy = int(np.clip(y0 + cut * (y1 - y0), 1, H - 50))
        crop = img[cy:, cx:]
        if crop.shape[0] < 120 or crop.shape[1] < 120: continue
        got = [it for it in A.gate_candidates(crop, cfg) if it.verdict != "rejected"]
        if not got: 
            rows.append((cut, None, None, None, None)); continue
        # match by the corner that stayed in view
        best, bestd = None, 1e9
        for it in got:
            pts = it.points + [cx, cy]
            d = float(np.linalg.norm(pts[6] - truth[6]))
            if d < bestd: best, bestd = pts, d
        if bestd > 0.5 * side:
            rows.append((cut, None, None, None, None)); continue
        err = np.linalg.norm(best - truth, axis=1)
        outside = np.maximum.reduce([
            np.maximum(cx - truth[:, 0], 0), np.maximum(cy - truth[:, 1], 0)])
        reach = outside / side
        for e, r in zip(err, reach):
            rows.append((cut, float(e), float(r), float(e / side), float(side)))

ok = [r for r in rows if r[1] is not None]
miss = sum(1 for r in rows if r[1] is None)
print(f"joint_refine={JOINT} | trials={len(rows)//8 if rows else 0} point-samples={len(ok)}  failed re-detections={miss}")

bands = defaultdict(list)
for cut, err, reach, rel, side in ok:
    if reach <= 0.001: bands["in-frame (reach 0)"].append(rel)
    elif reach < 0.15:  bands["reach 0.00-0.15"].append(rel)
    elif reach < 0.30:  bands["reach 0.15-0.30"].append(rel)
    elif reach < 0.50:  bands["reach 0.30-0.50"].append(rel)
    elif reach < 0.80:  bands["reach 0.50-0.80"].append(rel)
    else:               bands["reach 0.80+"].append(rel)
print(f"\n{'band':22s} {'n':>5s} {'err/side p50':>13s} {'p90':>8s}  (x100 = % of gate size)")
for key in ("in-frame (reach 0)", "reach 0.00-0.15", "reach 0.15-0.30",
            "reach 0.30-0.50", "reach 0.50-0.80", "reach 0.80+"):
    v = np.array(bands.get(key, []))
    if not len(v): continue
    print(f"{key:22s} {len(v):5d} {100*np.median(v):12.2f}% {100*np.percentile(v,90):7.2f}%")
