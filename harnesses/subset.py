"""Render a named subset of instances, one frame per image, for inspection.
Usage: subset.py <out_dir> <predicate> [stride]
"""
import glob, os, sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A

SP = str(Path(__file__).resolve().parents[1] / "work")
out = f"{SP}/{sys.argv[1]}"
predicate = eval(f"lambda it: {sys.argv[2]}")
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 3
os.makedirs(out, exist_ok=True)
for f in glob.glob(f"{out}/*.jpg"): os.remove(f)

cfg = A.AutolabelConfig()
count = 0
_all = sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg"))[::stride]
_matched = []
for _p in _all:
    _img = cv2.imread(_p)
    if _img is not None and any(predicate(it) for it in A.gate_candidates(_img, cfg)):
        _matched.append(_p)
_step = max(len(_matched) // 24, 1)
for path in _matched[::_step][:24]:
    img = cv2.imread(path); H, W = img.shape[:2]
    hits = [it for it in A.gate_candidates(img, cfg) if predicate(it)]
    if not hits: continue
    for it in hits:
        pts = it.points
        cv2.polylines(img, [pts[:4].astype(np.int32)], True, (90,230,90), 4)
        cv2.polylines(img, [pts[4:].astype(np.int32)], True, (255,200,60), 3)
        for k, p in enumerate(pts):
            q = np.clip(p.astype(int), [14,14], [W-14,H-14])
            inside = 0 <= p[0] < W and 0 <= p[1] < H
            cv2.circle(img, tuple(q), 10, (255,0,255) if inside else (0,255,255), -1)
            cv2.putText(img, str(k), tuple(q+15), 0, 0.95, (255,255,255), 3)
        cv2.putText(img, f"{it.verdict} res={it.residual:.3f} {it.tags}",
                    tuple(np.clip(pts[0].astype(int)+[0,-20],[10,34],[W-620,H-10])),
                    0, 1.0, (255,255,255), 3)
    cv2.imwrite(f"{out}/{os.path.basename(path)}", cv2.resize(img, (760, 428)))
    count += 1
print(f"{count} frames -> {out}")
