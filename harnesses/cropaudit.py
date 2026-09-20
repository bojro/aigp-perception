"""Crop each accepted label at native resolution so errors are actually visible.

Contact sheets shrink a 1920x1080 frame to a few hundred pixels, which hides
exactly the few-pixel errors worth auditing. This cuts a window around each
instance at 1:1 (upscaling small gates) so a reviewer sees real pixels.
"""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A

SP = str(Path(__file__).resolve().parents[1] / "work")
out = f"{SP}/{sys.argv[1]}"
predicate = eval(f"lambda it: {sys.argv[2]}")
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
want = int(sys.argv[4]) if len(sys.argv) > 4 else 18
os.makedirs(out, exist_ok=True)
for f in glob.glob(f"{out}/*.jpg"): os.remove(f)

cfg = A.AutolabelConfig()
hits = []
for path in sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg"))[::stride]:
    img = cv2.imread(path)
    if img is None: continue
    for it in A.gate_candidates(img, cfg):
        if predicate(it):
            hits.append((path, it))
step = max(len(hits) // want, 1)
for n, (path, it) in enumerate(hits[::step][:want]):
    img = cv2.imread(path); H, W = img.shape[:2]
    pts = it.points
    cx, cy = pts.mean(axis=0)
    half = max(mean := it.outer_side_px, 80) * 0.85
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = int(cx + half), int(cy + half)
    pad = 60
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(24, 24, 28))
    x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, canvas.shape[1]), min(y1, canvas.shape[0])
    if x1 - x0 < 40 or y1 - y0 < 40: continue
    crop = canvas[y0:y1, x0:x1].copy()
    shift = np.array([pad - x0, pad - y0], np.float32)
    local = pts + shift
    scale = max(1.0, min(4.0, 640.0 / max(crop.shape[:2])))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    local = local * scale
    cv2.polylines(crop, [local[:4].astype(np.int32)], True, (90, 230, 90), 2)
    cv2.polylines(crop, [local[4:].astype(np.int32)], True, (255, 200, 60), 2)
    for k, p in enumerate(local):
        q = p.astype(int)
        if not (0 <= q[0] < crop.shape[1] and 0 <= q[1] < crop.shape[0]): continue
        cv2.drawMarker(crop, tuple(q), (255, 0, 255), cv2.MARKER_CROSS, 13, 2)
        cv2.putText(crop, str(k), tuple(q + 6), 0, 0.5, (255, 255, 255), 1)
    bar = np.full((34, crop.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, f"{os.path.basename(path)} {it.anchor} sup={it.support:.2f} "
                     f"proj={it.projected_support:.2f} side={it.outer_side_px:.0f}px x{scale:.1f}",
                (8, 23), 0, 0.5, (235, 235, 245), 1)
    cv2.imwrite(f"{out}/{n:02d}_{os.path.basename(path)}", np.vstack([bar, crop]))
print(f"{min(len(hits), want)} crops of {len(hits)} -> {out}")
