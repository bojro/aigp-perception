"""Check the written labels, not the internals: are they a real gate shape?"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception.autolabel_gate_pose import opening_residual, mean_side_length, order_corners

out = Path(sys.argv[1]); W, H = 1920, 1080
res, sides, nvis, order_bad, convex_bad = [], [], [], 0, 0
import itertools
for path in sorted(itertools.chain((out/"train"/"labels").glob("*.txt"), (out/"valid"/"labels").glob("*.txt"))):
    for line in path.read_text().split("\n"):
        if not line.strip(): continue
        kp = np.array([float(v) for v in line.split()[5:]]).reshape(8, 3)
        pts = (kp[:, :2] * [W, H]).astype(np.float32)
        vis = kp[:, 2]
        nvis.append(int((vis > 0).sum()))
        if (vis[:4] > 0).all():
            o = pts[:4]
            if not np.allclose(order_corners(o), o, atol=1.0): order_bad += 1
            def _cz(u, v): return float(u[0]*v[1] - u[1]*v[0])
            cross = [_cz(o[(i+1) % 4]-o[i], o[(i+2) % 4]-o[(i+1) % 4]) for i in range(4)]
            if not (np.all(np.array(cross) > 0) or np.all(np.array(cross) < 0)): convex_bad += 1
        if (vis > 0).all():
            r, _ = opening_residual(pts[:4], pts[4:])
            res.append(r); sides.append(mean_side_length(pts[:4]))
res, sides = np.array(res), np.array(sides)
print(f"labels with all 8 visible: {len(res)}")
if len(res):
    print(f"  geometric residual  p50={np.median(res):.5f}  p90={np.percentile(res,90):.5f}  max={res.max():.5f}")
    print(f"  in pixels           p50={np.median(res*sides):.2f}px  p90={np.percentile(res*sides,90):.2f}px")
print(f"  corner order violations: {order_bad} | non-convex outer quads: {convex_bad}")
import collections
print(f"  visible-point histogram: {dict(sorted(collections.Counter(nvis).items()))}")
