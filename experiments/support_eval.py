"""Does edge support separate good labels from bad ones? If not, it is useless."""
from pathlib import Path
import glob, sys
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
import aigp_perception.autolabel_gate_pose as A

cfg = A.AutolabelConfig()
buckets = {}
for path in P.frames()[::6]:
    img = cv2.imread(path)
    field = A.orange_field(img)
    for it in A.gate_candidates(img, cfg):
        so, no = A.edge_support(field, it.outer)
        si, ni = A.edge_support(field, it.inner)
        key = it.verdict if it.verdict != "review" else f"review/{it.anchor}"
        if it.reason: key = f"rejected/{it.reason}"
        buckets.setdefault(key, []).append((so, si, no, ni))

print(f"{'bucket':26s} {'n':>5s} {'outer_sup':>10s} {'inner_sup':>10s} {'sides_o':>8s} {'sides_i':>8s}")
for key in sorted(buckets):
    v = np.array(buckets[key])
    print(f"{key:26s} {len(v):5d} {v[:,0].mean():10.3f} {v[:,1].mean():10.3f} "
          f"{v[:,2].mean():8.2f} {v[:,3].mean():8.2f}")
