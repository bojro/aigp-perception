"""What drives the p90 tail? Suspect: the gate's 260mm depth under oblique views."""
import sys
from pathlib import Path
import glob
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

cfg = A.AutolabelConfig()
rows = []
for path in P.frames()[::6]:
    img = cv2.imread(path)
    for item in A.gate_candidates(img, cfg):
        if item.verdict == "rejected" or item.anchor != "both": continue
        if not np.isfinite(item.residual): continue
        q = item.outer
        top = np.linalg.norm(q[1]-q[0]); bottom = np.linalg.norm(q[2]-q[3])
        left = np.linalg.norm(q[3]-q[0]); right = np.linalg.norm(q[2]-q[1])
        # foreshortening: a square seen square-on has equal opposite sides
        taper = max(abs(1-top/bottom), abs(1-left/right))
        aspect = max(top, bottom) / max(min(left, right), 1e-6)
        squash = abs(math.log(aspect)) if (math:=__import__('math')) else 0
        rows.append((item.residual, taper, squash, item.outer_side_px, int(item.clipped)))

r = np.array(rows)
res, taper, squash, side, clip = r[:,0], r[:,1], r[:,2], r[:,3], r[:,4]
print(f"n={len(r)}")
for name, v in (("taper(perspective)", taper), ("squash(log aspect)", squash),
                ("side_px", side), ("clipped", clip)):
    c = np.corrcoef(res, v)[0,1]
    print(f"  corr(residual, {name:20s}) = {c:+.3f}")
hi = res > np.percentile(res, 75)
print(f"\nworst quartile vs rest:")
for name, v in (("taper", taper), ("squash", squash), ("side_px", side), ("clipped", clip)):
    print(f"  {name:10s} worst={v[hi].mean():7.3f}  rest={v[~hi].mean():7.3f}")
