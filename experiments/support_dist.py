from pathlib import Path
import glob, sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A
cfg = A.AutolabelConfig()
rows = []
for path in P.frames()[::6]:
    img = cv2.imread(path); field = A.orange_field(img)
    for it in A.gate_candidates(img, cfg):
        so, no = A.edge_support(field, it.outer)
        si, ni = A.edge_support(field, it.inner)
        rows.append((it.verdict, it.anchor, it.reason, so, si, no, ni, it.residual, it.outer_side_px))

def show(name, sel):
    v = [r for r in rows if sel(r)]
    if not v: print(f"{name}: none"); return
    so = np.array([r[3] for r in v]); si = np.array([r[4] for r in v])
    both = np.minimum(so, si)
    sides_ok = np.array([(r[5] >= 3 and r[6] >= 3) for r in v])
    print(f"{name:34s} n={len(v):4d} | min(sup) p50={np.median(both):.2f} "
          f"p75={np.percentile(both,75):.2f} p90={np.percentile(both,90):.2f} | "
          f"both rings >=3 sides: {sides_ok.sum():4d} "
          f"| of those, min_sup>=.70: {sum(1 for r,k in zip(v,sides_ok) if k and min(r[3],r[4])>=0.70):4d}"
          f" >=.80: {sum(1 for r,k in zip(v,sides_ok) if k and min(r[3],r[4])>=0.80):4d}")

show("auto (reference)", lambda r: r[0]=="auto")
show("review anchor=outer", lambda r: r[0]=="review" and r[1]=="outer")
show("review anchor=inner", lambda r: r[0]=="review" and r[1]=="inner")
show("review anchor=both", lambda r: r[0]=="review" and r[1]=="both")
show("rejected opening_mismatch", lambda r: r[2]=="opening_mismatch")
