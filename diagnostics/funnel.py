"""Where do gates fall out? Count every stage on the same frames."""
import sys
from pathlib import Path
import glob
from collections import Counter
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aigp_perception.autolabel_gate_pose import (AutolabelConfig, orange_mask, fit_quadrilateral,
                                       order_corners, refine_subpixel, opening_residual,
                                       mean_side_length)

SRC = "/Users/bojro/Downloads/gate frames"
files = sorted(glob.glob(f"{SRC}/*.jpg"))[::6]
cfg = AutolabelConfig()
c = Counter()
hole_fracs = []

for path in files:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = orange_mask(img, cfg)
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        c["frame_no_contours"] += 1; continue
    hier = hier[0]
    got = False
    for i, cont in enumerate(cnts):
        if hier[i][3] != -1: continue
        area = cv2.contourArea(cont)
        if area < cfg.minimum_outer_side_px**2 * 0.25:
            c["parent_too_small"] += 1; continue
        c["parent_ok"] += 1
        kids = [(j, cv2.contourArea(cnts[j])) for j in range(len(cnts)) if hier[j][3] == i]
        kids = [k for k in kids if k[1] > 0]
        if not kids:
            c["no_holes_at_all"] += 1; continue
        hole_fracs.append(max(k[1] for k in kids) / area)
        inrange = [j for j, a in kids
                   if cfg.minimum_hole_area_fraction*area < a < cfg.maximum_hole_area_fraction*area]
        if not inrange:
            c["holes_all_out_of_range"] += 1; continue
        c["has_usable_hole"] += 1
        oq = fit_quadrilateral(cont)
        if oq is None:
            c["outer_quad_fail"] += 1; continue
        c["outer_quad_ok"] += 1
        oq = order_corners(refine_subpixel(gray, oq, cfg.subpix_window))
        best = None
        for j in inrange:
            q = fit_quadrilateral(cnts[j])
            if q is None: continue
            q = order_corners(refine_subpixel(gray, q, cfg.subpix_window))
            s, _ = opening_residual(oq, q)
            if best is None or s < best[0]: best = (s, q)
        if best is None:
            c["inner_quad_fail"] += 1; continue
        side = mean_side_length(oq)
        if side < cfg.minimum_outer_side_px: c["too_small_side"] += 1; continue
        if best[0] > cfg.review_residual: c["residual_reject"] += 1; continue
        c["ACCEPTED"] += 1; got = True
    if not got: c["FRAME_EMPTY"] += 1

print(f"frames={len(files)}")
for k, v in c.most_common(): print(f"  {k:26s} {v}")
hf = np.array(hole_fracs)
print(f"\nmax-hole/parent-area over all parents: p25={np.percentile(hf,25):.3f} "
      f"p50={np.percentile(hf,50):.3f} p75={np.percentile(hf,75):.3f} (gate expects ~0.309)")
print(f"fraction of parents whose best hole is in [0.10,0.60]: "
      f"{100*((hf>0.10)&(hf<0.60)).mean():.0f}%")
