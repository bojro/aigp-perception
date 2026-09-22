"""Crop single-anchor labels sitting just BELOW the accept bar, for audit.

If these read as correct at native resolution, the bar is set too high and
lowering it is evidence, not indulgence. If they read as wrong, the bar stays.
"""
from pathlib import Path
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

SP = str(P.WORK)
LO, HI = float(sys.argv[1]), float(sys.argv[2])
out = f"{SP}/crops_border"
os.makedirs(out, exist_ok=True)
for f in glob.glob(f"{out}/*.jpg"): os.remove(f)

cfg = A.AutolabelConfig()
hits = []
for path in P.frames()[::3]:
    img = cv2.imread(path)
    if img is None: continue
    for it in A.gate_candidates(img, cfg):
        if (it.verdict == "review" and it.anchor != "both"
                and it.outer_side_px >= 110
                and LO <= min(it.support, it.projected_support) < HI):
            hits.append((path, it))
print(f"{len(hits)} candidates in support band [{LO}, {HI})")
step = max(len(hits) // 18, 1)
for n, (path, it) in enumerate(hits[::step][:18]):
    img = cv2.imread(path); H, W = img.shape[:2]
    pts = it.points
    cx, cy = pts.mean(axis=0)
    half = max(it.outer_side_px, 80) * 0.85
    pad = 60
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(24,24,28))
    x0, y0 = max(int(cx-half)+pad, 0), max(int(cy-half)+pad, 0)
    x1, y1 = min(int(cx+half)+pad, canvas.shape[1]), min(int(cy+half)+pad, canvas.shape[0])
    if x1-x0 < 40 or y1-y0 < 40: continue
    crop = canvas[y0:y1, x0:x1].copy()
    local = pts + np.array([pad-x0, pad-y0], np.float32)
    scale = max(1.0, min(4.0, 640.0/max(crop.shape[:2])))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    local = local*scale
    cv2.polylines(crop, [local[:4].astype(np.int32)], True, (90,230,90), 2)
    cv2.polylines(crop, [local[4:].astype(np.int32)], True, (255,200,60), 2)
    for k, p in enumerate(local):
        q = p.astype(int)
        if not (0 <= q[0] < crop.shape[1] and 0 <= q[1] < crop.shape[0]): continue
        cv2.drawMarker(crop, tuple(q), (255,0,255), cv2.MARKER_CROSS, 13, 2)
        cv2.putText(crop, str(k), tuple(q+6), 0, 0.5, (255,255,255), 1)
    bar = np.full((34, crop.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, f"{os.path.basename(path)} {it.anchor} measured={it.support:.2f} "
                     f"projected={it.projected_support:.2f} side={it.outer_side_px:.0f}px x{scale:.1f}",
                (8,23), 0, 0.5, (235,235,245), 1)
    cv2.imwrite(f"{out}/{n:02d}_{os.path.basename(path)}", np.vstack([bar, crop]))
print(f"-> {out}")
