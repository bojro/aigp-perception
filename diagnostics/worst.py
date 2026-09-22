import sys
from pathlib import Path
import glob
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aigp_perception.autolabel_gate_pose as A
SP = str(Path(__file__).resolve().parents[1] / "work")
cfg = A.AutolabelConfig()
found = []
for path in sorted(glob.glob("/Users/bojro/Downloads/gate frames/*.jpg"))[::6]:
    img = cv2.imread(path)
    for item in A.gate_candidates(img, cfg):
        if item.verdict == "rejected" or item.anchor != "both": continue
        if np.isfinite(item.residual): found.append((item.residual, path, item))
found.sort(key=lambda f: -f[0])
import os
os.makedirs(f"{SP}/worst", exist_ok=True)
for rank, (res, path, item) in enumerate(found[:9]):
    img = cv2.imread(path)
    cv2.polylines(img, [item.outer.astype(np.int32)], True, (0,255,0), 4)
    cv2.polylines(img, [item.inner.astype(np.int32)], True, (0,200,255), 4)
    for k, pt in enumerate(np.vstack([item.outer, item.inner]).astype(int)):
        cv2.circle(img, tuple(pt), 10, (255,0,255), -1)
        cv2.putText(img, str(k), tuple(pt+14), 0, 1.1, (255,255,255), 3)
    cv2.putText(img, f"res={res:.3f} ({res*item.outer_side_px:.0f}px) {os.path.basename(path)}",
                (20,60), 0, 1.3, (255,255,255), 3)
    cv2.imwrite(f"{SP}/worst/{rank}.jpg", cv2.resize(img, (640,360)))
    print(f"{rank} res={res:.4f} px={res*item.outer_side_px:6.1f} side={item.outer_side_px:6.0f} {os.path.basename(path)}")
