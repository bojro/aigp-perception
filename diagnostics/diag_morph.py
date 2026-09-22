"""Does the morphological close eat the aperture of small and mid gates?"""
import sys
from pathlib import Path
import glob
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception import paths as P
from aigp_perception.autolabel_gate_pose import AutolabelConfig, orange_mask

SRC = str(P.CAPTURE)
files = sorted(glob.glob(f"{SRC}/*.jpg"))[::17][:70]

for close_k, iters in ((9, 2), (5, 1), (3, 1), (0, 0)):
    cfg = AutolabelConfig(close_kernel=close_k, close_iterations=iters)
    parents = holes = 0
    hole_sizes = []
    for path in files:
        img = cv2.imread(path)
        mask = orange_mask(img, cfg)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None: continue
        hier = hier[0]
        for i, c in enumerate(cnts):
            if hier[i][3] != -1: continue
            a = cv2.contourArea(c)
            if a < 900: continue
            parents += 1
            kids = [cv2.contourArea(cnts[j]) for j in range(len(cnts)) if hier[j][3] == i]
            kids = [k for k in kids if k > 0.02 * a]
            if kids:
                holes += 1
                hole_sizes.append(max(kids) / a)
    hs = np.array(hole_sizes) if hole_sizes else np.array([0.0])
    print(f"close={close_k}x{close_k} it={iters:d} | parents={parents:4d} "
          f"with_hole={holes:4d} ({100*holes/max(parents,1):4.1f}%) | "
          f"hole/parent area p10={np.percentile(hs,10):.3f} p50={np.percentile(hs,50):.3f}")
